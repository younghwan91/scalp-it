"""키움 WebSocket 실시간 데몬 — 0B(체결)·0D(호가) 를 받아 배치 저장한다.

프로토콜(20번 문서 실측):

1. connect ``wss://api.kiwoom.com:10000/api/dostk/websocket``
2. ``{"trnm":"LOGIN","token":"<기존 REST 토큰>"}`` → ``return_code=0``
3. ``{"trnm":"REG","grp_no":..,"refresh":"1","data":[{"item":[...],"type":["0B","0D"]}]}``
4. 장중 ``{"trnm":"REAL","data":[{"type":"0B","item":"005930","values":{"<FID>":..}}]}`` push
5. 키움이 ``{"trnm":"PING"}`` 을 보내면 그대로 되돌려줘야 연결이 유지된다.

한 커넥션 등록 한도 ≈ 100 건(종목×type). 0B+0D 면 종목당 2 건이라 커넥션당 ~50
종목, **0B+0D+1h(VI) 면 종목당 3 건이라 커넥션당 ~33종목**. 종목이 많으면 커넥션을
쪼갠다. 끊기면 재연결하고 **재REG** 한다.

.. warning::
   키움 WebSocket 은 **App key 당 동시 세션 1개**만 허용한다(장중 실측: 두 번째
   접속 시 ``SYSTEM R10001 "동일한 App key 로 접속 … 기존 세션은 종료"``). 같은
   키로 커넥션을 여러 개 열면 서로 kick·재연결을 반복하며 **thrash** 하고
   ``105110 (LOGIN 요청 건수 초과)`` 가 뜬다. 다커넥션이 필요하면 **App key 를
   커넥션 수만큼** 두거나, 한 커넥션에 담기는 종목 수(≤33, 3 type)로 줄여야 한다.

**FID 매핑은 표준값**(20번 문서)이고 장중 REAL 프레임으로 최종 대조해야 한다. 그
대비로 원문 ``values`` 를 ``raw`` 에 통째 저장한다 — 오매핑이어도 복구 가능하다.

파싱 함수(:func:`parse_real_message` 등)는 I/O 와 분리돼 있어 mock 프레임으로 단위
테스트한다. WebSocket 연결은 :meth:`KiwoomRealtimeCollector.run` 이 담당한다.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

WS_URL = "wss://api.kiwoom.com:10000/api/dostk/websocket"
WS_URL_MOCK = "wss://mockapi.kiwoom.com:10000/api/dostk/websocket"

REG_LIMIT = 100  # 커넥션당 등록(종목×type) 한도 실측치

# --------------------------------------------------------------- FID 매핑(표준값)

# 0B 주식체결
FID_TRADE_TIME = "20"      # 체결시각 HHMMSS
FID_PRICE = "10"           # 현재가(체결가) — 부호는 전일대비 방향, 뗀다
FID_TRADE_QTY = "15"       # 체결량 — 부호 = 매수(+)/매도(-)
FID_ACC_QTY = "13"         # 누적거래량
FID_STRENGTH = "228"       # 체결강도
FID_BEST_ASK = "27"        # 최우선매도호가
FID_BEST_BID = "28"        # 최우선매수호가

# 0D 주식호가잔량
FID_QUOTE_TIME = "21"      # 호가시각 HHMMSS
FID_ASK_PX = [str(f) for f in range(41, 51)]    # 매도호가 1..10
FID_BID_PX = [str(f) for f in range(51, 61)]    # 매수호가 1..10
FID_ASK_QTY = [str(f) for f in range(61, 71)]   # 매도잔량 1..10
FID_BID_QTY = [str(f) for f in range(71, 81)]   # 매수잔량 1..10
FID_TOT_ASK_QTY = "121"    # 총매도잔량
FID_TOT_BID_QTY = "125"    # 총매수잔량

TYPE_TRADE = "0B"
TYPE_QUOTE = "0D"
TYPE_VI = "1h"             # VI(변동성완화장치) 발동/해제

# 1h VI발동/해제 — FID 매핑은 키움 문서(koapy realtype ``1h`` = "VI발동/해제") 기준.
# **장중 REAL 프레임으로 아직 최종 대조하지 못했다**(VI 는 드문 이벤트라 프로브에서
# 안 잡힘). REG(1h) return_code=0 로 타입 수용은 확인. 원문은 ``raw`` 에 통째 보존해
# 오매핑이어도 되살린다. csv 확정 = koapy fid.csv 에 이름이 있는 FID, 추측 = 90xx 계열.
FID_VI_CODE = "9001"          # 종목코드 (csv 확정)
FID_VI_KIND = "9068"         # VI발동구분(발동/해제) — 추측(90xx, csv 미수록)
FID_VI_APPLY = "1225"        # VI적용구분(정적/동적/정적&동적) (csv 확정)
FID_VI_TRIGGER_PRICE = "1221"  # 발동가격 (csv 확정)
FID_VI_TRIGGER_TIME = "1223"   # 매매체결처리시각=발동시각 HHMMSS (csv 확정)
FID_VI_RELEASE_TIME = "1224"   # VI해제시각 HHMMSS (csv 확정)
FID_VI_BASE_STATIC = "1236"    # 정적기준가격 (csv 확정)
FID_VI_BASE_DYNAMIC = "1237"   # 동적기준가격 (csv 확정)
FID_VI_GAP_STATIC = "1238"     # 정적괴리율 (csv 확정)
FID_VI_GAP_DYNAMIC = "1239"    # 동적괴리율 (csv 확정)


# --------------------------------------------------------------- 파싱 헬퍼


def _clean(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _abs_int(value: Any) -> int | None:
    """부호 붙은 값을 절댓값 정수로. 부호는 방향 표시라 뗀다. 빈 값은 None."""
    text = _clean(value).lstrip("+-")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _signed_int(value: Any) -> tuple[int | None, int | None]:
    """체결량처럼 부호가 **의미 있는** 값. (절댓값, 부호+1/-1) 을 준다.

    부호 없이 오면(예: 상승 체결을 +로 안 붙이는 경우) 기본 +1 로 본다.
    """
    text = _clean(value)
    if not text.lstrip("+-"):
        return None, None
    sign = -1 if text.startswith("-") else 1
    return _abs_int(text), sign


def _to_float(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_hhmmss(value: Any, trade_date: date) -> datetime | None:
    """``HHMMSS`` (또는 ``HHMMSSmmm``) 를 그날 날짜와 합쳐 naive datetime 으로.

    키움 시각은 KST 이고 DB ``timestamp`` 도 tz 없는 KST 라 naive 로 둔다.
    """
    text = _clean(value)
    if not text.isdigit() or len(text) < 6:
        return None
    hh, mm, ss = int(text[0:2]), int(text[2:4]), int(text[4:6])
    if hh > 23 or mm > 59 or ss > 59:
        return None
    # KST naive — DB timestamp 도 tz 없는 KST 라 프로젝트 표준을 따른다.
    return datetime(trade_date.year, trade_date.month, trade_date.day, hh, mm, ss)  # noqa: DTZ001


class SeqAssigner:
    """같은 ``(code, ts)`` 초 안에서 수신 순서대로 ``seq`` 를 매긴다.

    틱 PK 가 ``(code, ts, seq)`` 라 같은 초에 여러 체결이 오면 구분자가 필요하다.
    ts 가 바뀌면 그 종목의 카운터를 0 부터 다시 센다.
    """

    def __init__(self) -> None:
        self._last: dict[str, datetime] = {}
        self._seq: dict[str, int] = {}

    def next(self, code: str, ts: datetime) -> int:
        if self._last.get(code) != ts:
            self._last[code] = ts
            self._seq[code] = 0
        else:
            self._seq[code] += 1
        return self._seq[code]

    def count(self, code: str, ts: datetime) -> int:
        """이 종목의 ``ts`` 초에 지금까지 매긴 번호 수(= 그 초에 받은 건수). 없으면 0."""
        if self._last.get(code) != ts:
            return 0
        return self._seq[code] + 1


# --------------------------------------------------------------- 프레임 파싱


def parse_tick_values(
    code: str, values: dict, trade_date: date, seq_assigner: SeqAssigner
) -> dict | None:
    """0B ``values`` 한 건을 ``ticks`` 행으로. 체결시각을 못 읽으면 None."""
    ts = _parse_hhmmss(values.get(FID_TRADE_TIME), trade_date)
    if ts is None:
        return None
    qty, side = _signed_int(values.get(FID_TRADE_QTY))
    return {
        "code": code,
        "ts": ts,
        "seq": seq_assigner.next(code, ts),
        "price": _abs_int(values.get(FID_PRICE)),
        "volume": qty,
        "side": side,
        "strength": _to_float(values.get(FID_STRENGTH)),
        "cum_volume": _abs_int(values.get(FID_ACC_QTY)),
        "best_bid": _abs_int(values.get(FID_BEST_BID)),
        "best_ask": _abs_int(values.get(FID_BEST_ASK)),
        "raw": values,
    }


def parse_quote_values(code: str, values: dict, trade_date: date) -> dict | None:
    """0D ``values`` 한 건을 ``quotes`` 행으로. 호가시각을 못 읽으면 None."""
    ts = _parse_hhmmss(values.get(FID_QUOTE_TIME), trade_date)
    if ts is None:
        return None
    row: dict[str, Any] = {"code": code, "ts": ts, "raw": values}
    for i in range(10):
        row[f"ask{i + 1}"] = _abs_int(values.get(FID_ASK_PX[i]))
        row[f"bid{i + 1}"] = _abs_int(values.get(FID_BID_PX[i]))
        row[f"askqty{i + 1}"] = _abs_int(values.get(FID_ASK_QTY[i]))
        row[f"bidqty{i + 1}"] = _abs_int(values.get(FID_BID_QTY[i]))
    row["total_ask_qty"] = _abs_int(values.get(FID_TOT_ASK_QTY))
    row["total_bid_qty"] = _abs_int(values.get(FID_TOT_BID_QTY))
    row["vi_flag"] = None  # VI 는 1h 타입. 0D 에 실려 오면 장중 확인 후 매핑.
    return row


def _vi_type(value: Any) -> str:
    """FID 9068(VI발동구분) → '발동'/'해제'. 인코딩 미확인이라 방어적으로 매핑한다.

    값이 비면 대부분의 프레임이 발동이므로 '발동' 으로 본다(raw 로 되살릴 수 있다).
    """
    text = _clean(value)
    if not text:
        return "발동"
    if "해제" in text or text == "2":
        return "해제"
    if "발동" in text or text == "1":
        return "발동"
    return text


def _vi_kind(value: Any) -> str | None:
    """FID 1225(VI적용구분) → '정적'/'동적'/'정적동적'. 못 읽으면 원문 그대로."""
    text = _clean(value)
    if not text:
        return None
    mapping = {"1": "정적", "2": "동적", "3": "정적동적"}
    if text in mapping:
        return mapping[text]
    has_static = "정적" in text
    has_dynamic = "동적" in text
    if has_static and has_dynamic:
        return "정적동적"
    if has_static:
        return "정적"
    if has_dynamic:
        return "동적"
    return text


def _first_int(*values: Any) -> int | None:
    for v in values:
        got = _abs_int(v)
        if got is not None:
            return got
    return None


def parse_vi_values(
    code: str, values: dict, trade_date: date, received_at: datetime | None = None
) -> dict | None:
    """1h ``values`` 한 건을 ``vi_events`` 행으로.

    발동시각(FID 1223)을 못 읽으면 ``received_at`` (수신 시각)으로 대체한다. 둘 다
    없으면 None. 원문은 ``raw`` 에 통째 보존한다(FID 미확정 대비).
    """
    ts = _parse_hhmmss(values.get(FID_VI_TRIGGER_TIME), trade_date)
    if ts is None:
        ts = received_at
    if ts is None:
        return None
    release_raw = _clean(values.get(FID_VI_RELEASE_TIME))
    return {
        "code": code,
        "ts": ts,
        "vi_type": _vi_type(values.get(FID_VI_KIND)),
        "vi_kind": _vi_kind(values.get(FID_VI_APPLY)),
        "base_price": _first_int(
            values.get(FID_VI_BASE_STATIC), values.get(FID_VI_BASE_DYNAMIC)
        ),
        "trigger_price": _abs_int(values.get(FID_VI_TRIGGER_PRICE)),
        "gap_rate": _to_float(values.get(FID_VI_GAP_STATIC))
        if _to_float(values.get(FID_VI_GAP_STATIC)) is not None
        else _to_float(values.get(FID_VI_GAP_DYNAMIC)),
        "release_time": release_raw or None,
        "raw": values,
    }


def parse_real_message(
    message: Any,
    trade_date: date,
    seq_assigner: SeqAssigner,
    received_at: datetime | None = None,
    quote_seq: SeqAssigner | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """REAL 프레임(문자열 또는 dict)을 ``(ticks, quotes, vi_events)`` 로 파싱한다.

    REAL 이 아니거나 data 가 없으면 빈 세 리스트. 알 수 없는 type 은 무시한다.
    ``received_at`` 은 VI 발동시각 FID 결손 시 대체할 수신 시각이다.

    ``quote_seq`` 를 주면 호가 행마다 ``qseq``(같은 초 0D 수신 순서)와 ``ticks_before``
    (그 전에 받은 같은 초 틱 수)를 붙인다 — ``quote_events`` 가 초 안 선후를 복원하는
    열쇠다. 프레임 안 item 순서대로 틱 ``seq`` 와 함께 매기므로 선후가 정확하다.
    """
    msg = json.loads(message) if isinstance(message, (str, bytes)) else message
    if not isinstance(msg, dict) or msg.get("trnm") != "REAL":
        return [], [], []
    ticks: list[dict] = []
    quotes: list[dict] = []
    vis: list[dict] = []
    for item in msg.get("data") or []:
        if not isinstance(item, dict):
            continue
        code = _clean(item.get("item"))
        values = item.get("values")
        if not code or not isinstance(values, dict):
            continue
        rtype = _clean(item.get("type"))
        if rtype == TYPE_TRADE:
            row = parse_tick_values(code, values, trade_date, seq_assigner)
            if row is not None:
                ticks.append(row)
        elif rtype == TYPE_QUOTE:
            row = parse_quote_values(code, values, trade_date)
            if row is not None:
                if quote_seq is not None:
                    row["qseq"] = quote_seq.next(code, row["ts"])
                    row["ticks_before"] = seq_assigner.count(code, row["ts"])
                quotes.append(row)
        elif rtype == TYPE_VI:
            row = parse_vi_values(code, values, trade_date, received_at)
            if row is not None:
                vis.append(row)
    return ticks, quotes, vis


# --------------------------------------------------------------- 메시지 빌더


def chunk_codes(codes: Sequence[str], per_conn: int) -> list[list[str]]:
    """종목을 커넥션당 최대 ``per_conn`` 개로 쪼갠다(중복 제거·순서 보존)."""
    seen: set[str] = set()
    uniq: list[str] = []
    for c in codes:
        c = _clean(c)
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    per_conn = max(1, int(per_conn))
    return [uniq[i : i + per_conn] for i in range(0, len(uniq), per_conn)]


def codes_per_connection(types: Sequence[str], *, reg_limit: int = REG_LIMIT) -> int:
    """type 개수로 커넥션당 종목 수를 계산한다(0B+0D → 50)."""
    n = max(1, len(types))
    return max(1, reg_limit // n)


def build_login_message(token: str) -> dict:
    return {"trnm": "LOGIN", "token": token}


def build_reg_message(
    codes: Sequence[str], types: Sequence[str], grp_no: str, *, refresh: str = "1"
) -> dict:
    return {
        "trnm": "REG",
        "grp_no": str(grp_no),
        "refresh": str(refresh),
        "data": [{"item": list(codes), "type": list(types)}],
    }


def build_remove_message(
    codes: Sequence[str], types: Sequence[str], grp_no: str
) -> dict:
    """구독 해제 메시지. ``REG`` 와 모양이 같고 ``trnm`` 만 다르다.

    자리를 비우는 용도다 — 커넥션당 등록 한도(:data:`REG_LIMIT`)가 있어서, 장중에
    시장 전체를 훑어 새 대장을 붙이려면 식은 종목을 내려야 한다. 해제를 못 하면
    자리가 차는 순간부터 그날 나머지 대장을 영영 못 본다.
    """
    return {
        "trnm": "REMOVE",
        "grp_no": str(grp_no),
        "refresh": "0",
        "data": [{"item": list(codes), "type": list(types)}],
    }


def quote_levels(row: Mapping[str, Any]) -> tuple[list[tuple[float, float]],
                                                  list[tuple[float, float]]]:
    """0D 행 → ``(bids, asks)`` — 각 원소가 ``(가격, 잔량)``, 1호가부터.

    만쥬 §4-1 이 **최우선**이라 한 호가 두께 필터(`max_roundtrip_bp`)의 유일한 입력이다.
    2026-08-26 까지 CLI 가 `on_quote(code, ts, bid1)` 로 **가격 하나만** 넘겨서 감지기
    호가창이 프로덕션에서 늘 비어 있었고, 그래서 그 필터가 입력 없음 → 무조건 통과로
    돌았다. cron 플래그는 켜져 있는데 아무것도 안 거르는 상태였다.

    가격이나 잔량이 없거나 0 인 단은 버린다 — 없는 물량을 소진 계산에 넣으면 왕복
    충격이 실제보다 낙관적으로 나온다.
    """
    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []
    for i in range(1, 11):
        for key, qkey, out in (("bid", "bidqty", bids), ("ask", "askqty", asks)):
            px, qty = row.get(f"{key}{i}"), row.get(f"{qkey}{i}")
            if px and qty:
                out.append((float(px), float(qty)))
    return bids, asks


def plan_additions(
    subscribed: Sequence[str], wanted: Sequence[str], types: Sequence[str],
    *, reg_limit: int = REG_LIMIT,
) -> tuple[list[str], list[str]]:
    """장중에 더 구독할 종목을 한도 안에서 고른다 → ``(추가할 것, 거절된 것)``.

    만쥬 §6("돌발 뉴스가 나오면 즉각 관련 종목 매수")을 하려면 장전 유니버스에 없던
    종목을 **장중에** 구독해야 한다. 그런데 등록 한도(커넥션당 종목×type ≈ 100)를
    넘기면 수집기가 커넥션을 쪼개고, 같은 App key 로 둘을 열면 서로 kick 한다 —
    2026-08-24 에 80종목을 넣었다가 **30종목이 10:58 에 끊겨** 하루의 3분의 2를 잃었다.

    그래서 조용히 넘기지 않고 **거절 목록을 돌려준다**. 호출부가 그걸 남겨야 나중에
    "자리가 없어서 못 봤다"를 셀 수 있다.
    """
    have = list(dict.fromkeys(str(c) for c in subscribed))
    room = max(0, codes_per_connection(types, reg_limit=reg_limit) - len(have))
    seen = set(have)
    ok: list[str] = []
    rejected: list[str] = []
    for code in dict.fromkeys(str(c) for c in wanted):
        if code in seen:
            continue
        seen.add(code)
        if len(ok) < room:
            ok.append(code)
        else:
            rejected.append(code)
    return ok, rejected


# --------------------------------------------------------------- 데몬


class KiwoomRealtimeCollector:
    """LOGIN→REG→listen→파싱→배치 insert. 재연결·재REG·graceful flush."""

    def __init__(
        self,
        token: str,
        codes: Sequence[str],
        *,
        conn: Any = None,
        dsn: str | None = None,
        types: Sequence[str] = (TYPE_TRADE, TYPE_QUOTE),
        url: str = WS_URL,
        per_conn: int | None = None,
        flush_interval: float = 1.5,
        store: bool = True,
        trade_date: date | None = None,
        on_frame: Callable[[dict], None] | None = None,
        reg_limit: int = REG_LIMIT,
        max_backoff: float = 30.0,
        log: Callable[[str], None] | None = None,
        spool_dir: str | Path | None = None,
        quote_events: bool | None = None,
    ) -> None:
        self._token = token
        self._types = list(types)
        self._url = url
        self._conn = conn
        self._dsn = dsn  # DB 순단 시 재연결용
        self._store = store
        self._flush_interval = max(0.2, float(flush_interval))
        self._trade_date = trade_date or date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
        self._on_frame = on_frame
        self._max_backoff = max_backoff
        self._log = log or (lambda m: print(m, flush=True))

        self._reg_limit = int(reg_limit)   # 장중 추가(add_codes)가 한도를 알아야 한다
        if per_conn is None:
            per_conn = codes_per_connection(types, reg_limit=reg_limit)
        # 빈 목록이어도 커넥션 하나는 연다 — 듀얼키 B 는 장전 유니버스가 정원보다
        # 작으면 빈 손으로 시작하는데, 커넥션이 0개면 LOGIN 이 없어 장중 추가가
        # 전부 조용히 거절된다(09-07~10 정원 100 이라 믿었는데 실제 50).
        self._groups = chunk_codes(codes, per_conn) or [[]]

        self._seq = SeqAssigner()
        # quote_events — 0D 수신 전부(초당 마지막만 남기는 quotes 와 별개). 부가 수집이라
        # 실패가 틱·호가 경로에 번지지 않게 격리한다(_flush_quote_events). 끄려면
        # SCALP_QUOTE_EVENTS=0 — 코드 되돌림 없이 다음 런처부터.
        if quote_events is None:
            quote_events = os.environ.get("SCALP_QUOTE_EVENTS", "1").strip() != "0"
        self._quote_events_on = bool(quote_events)
        self._qseq = SeqAssigner()
        self._qevent_buf: list[dict] = []
        self._qevent_failures = 0
        self._tick_buf: list[dict] = []
        self._quote_buf: list[dict] = []
        self._vi_buf: list[dict] = []
        self._stop = asyncio.Event()

        # DB 순단이 "프로세스 생존 중"이면 위 버퍼 재시도로 충분하지만, 그동안
        # 프로세스 자체가 죽으면(재시작·OOM·서버 재부팅) 메모리 버퍼는 통째로
        # 사라진다 — 틱은 그 순간 못 받으면 영영 복구 불가능이라 이 손실은
        # 특히 위험하다(2026-09-11, DB 를 이 수집기와 다른 호스트로 옮기기 전
        # 선행 조건). spool_dir 를 주면 flush 실패마다 그 시점 버퍼를 통째로
        # 디스크에 미러링하고, 다음 프로세스가 뜰 때 그걸 읽어 버퍼로 되살린다.
        self._spool_dir = Path(spool_dir) if spool_dir else None
        if self._spool_dir:
            self._load_spool()

        # 통계
        self.frames_seen = 0
        self.ticks_stored = 0
        self.quotes_stored = 0
        self.quote_events_stored = 0
        self.vi_stored = 0
        self.reconnects = 0
        # 장중 구독 추가(만쥬 §6)용 — grp_no → (소켓, 그 커넥션의 종목 목록).
        self._live_ws: dict[str, tuple[Any, list[str]]] = {}
        self.added_codes: list[str] = []      # 장중에 붙인 종목(집계·복기용)
        self.rejected_codes: list[str] = []   # 자리가 없어 못 붙인 종목
        self.removed_codes: list[str] = []    # 자리를 비우려 내린 종목

    # ------------------------------------------------------------- 공개 API

    @property
    def n_connections(self) -> int:
        return len(self._groups)

    def stop(self) -> None:
        self._stop.set()

    async def run(self, duration: float | None = None) -> None:
        """모든 커넥션 태스크 + flush 태스크를 띄운다. ``duration`` 초 뒤 종료.

        ``duration`` 이 None 이면 :meth:`stop` 이 호출될 때까지 돈다.
        """
        tasks = [asyncio.create_task(self._connection_loop(i, g))
                 for i, g in enumerate(self._groups)]
        flusher = asyncio.create_task(self._flush_loop())
        try:
            if duration is not None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=duration)
            else:
                await self._stop.wait()
        finally:
            self._stop.set()
            for t in tasks:
                t.cancel()
            flusher.cancel()
            for t in [*tasks, flusher]:
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            self._flush()  # graceful: 남은 버퍼를 마저 쓴다
            self._log(
                f"종료 · 프레임 {self.frames_seen:,} · 틱 {self.ticks_stored:,} · "
                f"호가 {self.quotes_stored:,} · 호가이벤트 {self.quote_events_stored:,} · "
                f"VI {self.vi_stored:,} · "
                f"재연결 {self.reconnects}"
            )

    # ------------------------------------------------------------- 내부

    async def _connect(self):
        """websockets 로 연결한다. 지연 임포트라 파싱 테스트는 websockets 불필요."""
        import websockets

        return await websockets.connect(self._url, ping_interval=None, max_size=None)

    async def _connection_loop(self, idx: int, codes: list[str]) -> None:
        grp_no = str(idx + 1)
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = await self._connect()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[conn{grp_no}] 연결 실패: {exc} — {backoff:.0f}s 후 재시도")
                await self._sleep_or_stop(backoff)
                backoff = min(self._max_backoff, backoff * 2)
                continue
            try:
                async with ws:
                    await self._login_and_reg(ws, grp_no, codes)
                    backoff = 1.0  # 성공적으로 붙었으면 백오프 리셋
                    # 장중 추가(만쥬 §6)를 보내려면 살아 있는 소켓이 필요하다.
                    self._live_ws[grp_no] = (ws, codes)
                    await self._listen(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — 어떤 끊김도 재연결로 흡수
                if self._stop.is_set():
                    break
                self.reconnects += 1
                self._log(f"[conn{grp_no}] 끊김: {exc} — {backoff:.0f}s 후 재연결·재REG")
                await self._sleep_or_stop(backoff)
                backoff = min(self._max_backoff, backoff * 2)

    @property
    def subscribed_codes(self) -> list[str]:
        """지금 실제로 구독 중인 종목 — 장중에 붙고 떨어진 결과가 반영된 값이다.

        장전 목록(`_groups`)이 아니라 라이브 소켓의 등록 목록을 본다. 수집 슬롯
        채우기가 "자리가 남았나"를 판단하는 근거다.
        """
        if not self._live_ws:
            return []
        return list(self._live_ws[min(self._live_ws)][1])

    @property
    def code_capacity(self) -> int:
        """커넥션 하나에 담기는 종목 수 — 등록 한도 ÷ type 개수(0B+0D → 50)."""
        return codes_per_connection(self._types, reg_limit=self._reg_limit)

    async def add_codes(self, codes: Sequence[str]) -> tuple[list[str], list[str]]:
        """장중에 종목을 구독에 **추가**한다 → ``(추가된 것, 거절된 것)``.

        만쥬 §6 — "예상치 못한 돌발 뉴스가 나오면 즉각 관련 종목 매수". 장전에 고른
        유니버스에 그 종목이 없으면 구독조차 없어서 감시를 못 한다.

        ``refresh="1"`` 로 보낸다 — 키움의 ``refresh`` 는 **기존등록유지여부**라
        1 이 유지(추가), 0 이 해제 후 등록(교체)이다. 종전엔 0 으로 보냈고, 그래서
        장중 추가 한 번이 **그때까지 구독하던 종목을 전부 날렸다**.

        2026-08-28 틱 실측이 그대로 보여준다 — 종목별 마지막 틱이 다음 추가 종목의
        첫 틱과 **초 단위로 일치**한다(09:02:59 에 12종목이 동시에 끊기고 430690
        하나만 남는 식). 그날 09시 39종목이던 것이 10시엔 5종목, 이후 종일 3~8종목
        이었다. 08-27 도 11시 이후 1~2종목으로 무너졌다(장중 추가가 없던 08-26 은
        49종목이 종일 유지됐다 — 대조군).

        한도를 넘는 건 거절하고 목록으로 돌려준다(조용히 넘기면 왜 못 봤는지 모른다).
        """
        if not self._live_ws:
            rejected = list(codes)
            self._log(f"소켓 미연결 — 거절 {len(rejected)}종목: {', '.join(rejected[:5])}")
            self.rejected_codes.extend(rejected)
            return [], rejected
        # 첫 커넥션에만 붙인다 — 여러 커넥션은 같은 키에서 서로 kick 하므로 애초에
        # 단일 커넥션으로 운영하는 게 전제다.
        grp_no = min(self._live_ws)
        ws, have = self._live_ws[grp_no]
        ok, rejected = plan_additions(have, codes, self._types,
                                      reg_limit=self._reg_limit)
        if rejected:
            self._log(f"[conn{grp_no}] 구독 자리 없음 — 거절 {len(rejected)}종목: "
                      f"{', '.join(rejected[:5])}")
            self.rejected_codes.extend(rejected)
        if not ok:
            return [], rejected
        try:
            await ws.send(json.dumps(
                build_reg_message(ok, self._types, grp_no, refresh="1")))
        except Exception as exc:  # noqa: BLE001 — 추가 실패로 수집을 죽이지 않는다
            self._log(f"[conn{grp_no}] 장중 REG 추가 실패: {exc}")
            return [], list(codes)
        have.extend(ok)
        self.added_codes.extend(ok)
        self._log(f"[conn{grp_no}] 장중 구독 추가 {len(ok)}종목: {', '.join(ok)}")
        return ok, rejected

    async def remove_codes(self, codes: Sequence[str]) -> list[str]:
        """장중에 종목 구독을 **해제**한다 → 실제로 내린 것.

        등록 한도가 있어 새 대장을 붙이려면 식은 종목을 내려야 한다. 실패해도
        수집을 죽이지 않는다 — 자리를 못 비웠을 뿐이고, 그러면 다음 추가가 거절될
        뿐이다(거절은 이미 기록으로 남는다).
        """
        if not self._live_ws or not codes:
            return []
        grp_no = min(self._live_ws)
        ws, have = self._live_ws[grp_no]
        drop = [c for c in codes if c in have]
        if not drop:
            return []
        try:
            await ws.send(json.dumps(build_remove_message(drop, self._types, grp_no)))
        except Exception as exc:  # noqa: BLE001 — 해제 실패로 수집을 죽이지 않는다
            self._log(f"[conn{grp_no}] 장중 REMOVE 실패: {exc}")
            return []
        for c in drop:
            have.remove(c)
        self.removed_codes.extend(drop)
        self._log(f"[conn{grp_no}] 장중 구독 해제 {len(drop)}종목: {', '.join(drop)}")
        return drop

    async def _login_and_reg(self, ws, grp_no: str, codes: list[str]) -> None:
        await ws.send(json.dumps(build_login_message(self._token)))
        login = json.loads(await ws.recv())
        if int(login.get("return_code", -1)) != 0:
            raise RuntimeError(f"LOGIN 실패: {login}")
        if not codes:
            self._log(f"[conn{grp_no}] LOGIN ok · 0종목 — 장중 추가 대기")
            return
        await ws.send(json.dumps(build_reg_message(codes, self._types, grp_no)))
        reg = json.loads(await ws.recv())
        if int(reg.get("return_code", -1)) != 0:
            raise RuntimeError(f"REG 실패: {reg}")
        self._log(f"[conn{grp_no}] LOGIN·REG ok · {len(codes)}종목 × {self._types}")

    async def _listen(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                break
            self._handle_raw(ws, raw)

    def _handle_raw(self, ws, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        trnm = msg.get("trnm") if isinstance(msg, dict) else None
        if trnm == "PING":
            # 키움 keep-alive. 받은 그대로 돌려줘야 연결이 유지된다.
            asyncio.create_task(ws.send(raw))
            return
        if trnm != "REAL":
            return
        self.frames_seen += 1
        if self._on_frame is not None:
            self._on_frame(msg)
        # VI 발동시각 FID 결손 대비 수신 시각(naive KST) 을 넘긴다.
        now = datetime.now()  # noqa: DTZ005 — KST naive, 프로젝트 표준
        ticks, quotes, vis = parse_real_message(
            msg, self._trade_date, self._seq, received_at=now,
            quote_seq=self._qseq if self._quote_events_on else None,
        )
        self._tick_buf.extend(ticks)
        self._quote_buf.extend(quotes)
        if self._quote_events_on and quotes:
            for q in quotes:
                q["recv_ts"] = now
            self._qevent_buf.extend(quotes)
        self._vi_buf.extend(vis)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_or_stop(self._flush_interval)
            self._flush()

    def _reconnect_db(self) -> bool:
        """DB 순단 후 커넥션 재생성. 실패해도 예외를 삼키고 False."""
        if not self._dsn:
            return False
        from .store import connect as _db_connect
        try:
            self._conn = _db_connect(self._dsn)
            return True
        except Exception as exc:  # noqa: BLE001 — 재연결 실패는 다음 flush 에서 재시도
            if self._log:
                self._log(f"DB 재연결 실패: {exc}")
            return False

    _SPOOL_FILES: ClassVar[dict[str, str]] = {
        "_tick_buf": "ticks.jsonl",
        "_quote_buf": "quotes.jsonl",
        "_vi_buf": "vi.jsonl",
    }

    def _spool_path(self, buf_attr: str) -> Path | None:
        if not self._spool_dir:
            return None
        return self._spool_dir / self._SPOOL_FILES[buf_attr]

    def _write_spool(self, buf_attr: str, batch: list[dict]) -> None:
        """그 시점 버퍼 전체를 spool 파일에 통째로 다시 쓴다(추가 아니라 미러).

        누적 실패마다 버퍼가 자라므로, append 하면 예전에 쓴 행이 중복으로 쌓인다.
        전체를 다시 쓰면 파일은 항상 "지금 안 들어간 것 전부"만 담는다.
        """
        path = self._spool_path(buf_attr)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            text = "\n".join(json.dumps(row, default=str) for row in batch)
            path.write_text(text + "\n" if text else "")
        except OSError as exc:
            if self._log:
                self._log(f"spool 쓰기 실패({buf_attr}): {exc}")

    def _clear_spool(self, buf_attr: str) -> None:
        path = self._spool_path(buf_attr)
        if path is not None and path.exists():
            path.unlink()

    def _load_spool(self) -> None:
        """이전 프로세스가 flush 못 하고 남긴 spool 을 버퍼로 되살린다.

        아직 지우지 않는다 — 이 프로세스도 flush 전에 죽으면 다음 번이 또
        읽어야 하므로, 실제로 DB 에 들어가고 나서(``_clear_spool``)만 지운다.
        """
        for buf_attr in self._SPOOL_FILES:
            path = self._spool_path(buf_attr)
            if path is None or not path.exists():
                continue
            try:
                lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
                rows = [json.loads(ln) for ln in lines]
            except (OSError, json.JSONDecodeError) as exc:
                if self._log:
                    self._log(f"spool 복구 실패({buf_attr}), 원본 파일 보존: {exc}")
                continue
            if rows:
                getattr(self, buf_attr).extend(rows)
                if self._log:
                    self._log(f"spool 복구({buf_attr}): {len(rows)}건 되살림")

    def _flush(self) -> None:
        if not self._store:
            self._tick_buf.clear()
            self._quote_buf.clear()
            self._qevent_buf.clear()
            self._vi_buf.clear()
            return
        from .store import insert_quotes, insert_ticks, insert_vi_events

        # DB 순단에 견딘다: insert 가 실패하면 배치를 버퍼에 되돌리고(유실 방지)
        # + spool_dir 가 있으면 디스크에도 미러링해 프로세스가 죽어도 살아남게
        # 하고, 커넥션을 재생성해 다음 flush 에서 재시도한다. flush 루프는
        # 죽지 않는다.
        def _try(buf_attr: str, insert_fn: Any, counter: str) -> bool:
            buf = getattr(self, buf_attr)
            if not buf:
                return True
            if self._conn is None:
                # 콜드 스타트(아직 한 번도 연결 못 함)나 재연결이 방금 또 실패한
                # 경우 — insert 를 시도할 커넥션 자체가 없다. 그렇다고 그냥
                # 넘어가면 spool_dir 가 있어도 디스크에 안 쓰여서, 이 상태로
                # 프로세스가 죽으면 스풀도 못 건진 채 버퍼가 통째로 날아간다
                # (실전 통합 테스트 2026-09-11 에서 발견 — 단위 테스트는 전부
                # `self._conn` 에 죽은 커넥션 객체를 넣어 검증해서 이 None
                # 경로를 못 잡았다). insert 실패와 동일하게 스풀에 미러링한다.
                self._write_spool(buf_attr, list(buf))
                return False
            batch = list(buf)
            try:
                setattr(self, counter, getattr(self, counter) + insert_fn(self._conn, batch))
                buf.clear()
                self._clear_spool(buf_attr)
                return True
            except Exception as exc:  # noqa: BLE001 — DB 순단: 배치 보존 후 재연결
                if self._log:
                    self._log(f"insert 실패({buf_attr}), 배치 {len(batch)}건 보존·재연결 시도: {exc}")
                self._write_spool(buf_attr, batch)
                self._reconnect_db()  # 버퍼는 그대로 두어 다음 flush 에서 재시도(순서 유지)
                return False

        if self._conn is None:
            self._reconnect_db()
        ok_ticks = _try("_tick_buf", insert_ticks, "ticks_stored")
        ok_quotes = _try("_quote_buf", insert_quotes, "quotes_stored")
        _try("_vi_buf", insert_vi_events, "vi_stored")
        self._flush_quote_events(db_ok=ok_ticks and ok_quotes)

    #: DB 순단 동안 들고 있을 quote_events 상한 — 스풀하지 않으므로 메모리만 지킨다.
    #: 09-10 기준 키 하나 하루 ~170만 건(초당 ~70건)이라 20만 건 ≈ 45분.
    _QEVENT_BUF_CAP = 200_000
    #: DB 는 멀쩡한데(틱·호가 insert 성공) quote_events 만 연속 실패하면 끈다.
    _QEVENT_MAX_FAILURES = 3

    def _flush_quote_events(self, *, db_ok: bool) -> None:
        """quote_events 배치 insert — **틱·호가 경로와 격리**한다.

        실패해도 스풀·재연결을 하지 않는다(그건 틱 경로 몫이고, 여기서 재연결하면 틱
        커넥션을 흔든다). 트랜잭션만 롤백하고 버퍼는 상한까지 들고 있다가 다음 flush 에
        재시도한다. 틱·호가는 들어갔는데 이것만 연속 실패하면(테이블·권한 문제) 이 프로세스
        에서는 끄고 크게 한 번 남긴다 — 틱 수집이 부가 수집 때문에 흔들리면 본말전도다.
        """
        buf = self._qevent_buf
        if not self._quote_events_on or not buf:
            return
        if self._conn is None or not db_ok:
            if len(buf) > self._QEVENT_BUF_CAP:
                del buf[: len(buf) - self._QEVENT_BUF_CAP]
            return
        from .store import insert_quote_events

        batch = list(buf)
        try:
            self.quote_events_stored += insert_quote_events(self._conn, batch)
            buf.clear()
            self._qevent_failures = 0
        except Exception as exc:  # noqa: BLE001 — 부가 수집: 틱 경로로 번지지 않게
            with contextlib.suppress(Exception):
                self._conn.rollback()
            self._qevent_failures += 1
            if len(buf) > self._QEVENT_BUF_CAP:
                del buf[: len(buf) - self._QEVENT_BUF_CAP]
            if self._qevent_failures >= self._QEVENT_MAX_FAILURES:
                self._quote_events_on = False
                buf.clear()
                if self._log:
                    self._log(f"[경보] quote_events insert {self._qevent_failures}회 연속 실패 — "
                              f"이 프로세스에서 끈다(틱·호가 수집은 계속): {exc}")
            elif self._log:
                self._log(f"quote_events insert 실패({self._qevent_failures}): {exc}")

    async def _sleep_or_stop(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)


class DualKeyCollector:
    """App key 두 개로 :class:`KiwoomRealtimeCollector` 두 개를 나란히 굴린다.

    App key 는 동시 세션 1개라 정원(≈50종목, 0B+0D)이 키 하나에 묶인다(모듈
    docstring 참고). 두 번째 키를 받으면 커넥션을 하나 더 열어 정원을 두 배로
    늘릴 수 있다 — 같은 키로 커넥션을 늘리면 서로 kick 하지만, 키가 다르면
    독립된 세션이라 문제없다.

    두 인스턴스가 받은 프레임은 **같은 `on_frame` 콜백**으로 흘러야 한다(둘 다
    같은 `PairDetector`/`RiskGuard`/`OrderExecutor` 를 보게) — 그건 생성하는
    쪽(:func:`scalp_it.cli_pair_detect.run_live`)의 책임이고, 이 클래스는 두
    인스턴스를 하나처럼 보이게 감싸기만 한다. `KiwoomRealtimeCollector` 를 쓰는
    호출부(`add_codes`·`remove_codes`·`subscribed_codes`·`code_capacity`·`run`·
    `stop`·`frames_seen`·`ticks_stored` 등)와 같은 인터페이스를 낸다 — 드롭인
    교체라 워처 함수(`_watch_leader_scan` 등)는 수정이 필요 없다.
    """

    def __init__(self, primary: KiwoomRealtimeCollector, secondary: KiwoomRealtimeCollector) -> None:
        self._a = primary
        self._b = secondary

    # ------------------------------------------------------------- 통계(합산)

    @property
    def frames_seen(self) -> int:
        return self._a.frames_seen + self._b.frames_seen

    @property
    def ticks_stored(self) -> int:
        return self._a.ticks_stored + self._b.ticks_stored

    @property
    def quotes_stored(self) -> int:
        return self._a.quotes_stored + self._b.quotes_stored

    @property
    def quote_events_stored(self) -> int:
        return self._a.quote_events_stored + self._b.quote_events_stored

    @property
    def vi_stored(self) -> int:
        return self._a.vi_stored + self._b.vi_stored

    @property
    def reconnects(self) -> int:
        return self._a.reconnects + self._b.reconnects

    @property
    def n_connections(self) -> int:
        return self._a.n_connections + self._b.n_connections

    @property
    def rejected_codes(self) -> list[str]:
        return [*self._a.rejected_codes, *self._b.rejected_codes]

    @property
    def added_codes(self) -> list[str]:
        return [*self._a.added_codes, *self._b.added_codes]

    @property
    def removed_codes(self) -> list[str]:
        return [*self._a.removed_codes, *self._b.removed_codes]

    # ------------------------------------------------------------- 구독 관리

    @property
    def subscribed_codes(self) -> list[str]:
        return [*self._a.subscribed_codes, *self._b.subscribed_codes]

    @property
    def code_capacity(self) -> int:
        """두 커넥션 정원의 합."""
        return self._a.code_capacity + self._b.code_capacity

    async def add_codes(self, codes: Sequence[str]) -> tuple[list[str], list[str]]:
        """이미 구독 중인 건 제외하고, A 먼저 채우고 남으면 B 로 넘긴다."""
        have = set(self.subscribed_codes)
        new = [str(c) for c in dict.fromkeys(codes) if str(c) not in have]
        if not new:
            return [], []
        added_a, rejected_a = await self._a.add_codes(new)
        if not rejected_a:
            return added_a, []
        added_b, rejected_b = await self._b.add_codes(rejected_a)
        return [*added_a, *added_b], rejected_b

    async def remove_codes(self, codes: Sequence[str]) -> list[str]:
        """각 종목이 실제로 붙어있는 쪽(A 또는 B)에서 내린다."""
        removed_a = await self._a.remove_codes(codes)
        remaining = [c for c in codes if c not in removed_a]
        removed_b = await self._b.remove_codes(remaining) if remaining else []
        return [*removed_a, *removed_b]

    # ------------------------------------------------------------- 실행

    def stop(self) -> None:
        self._a.stop()
        self._b.stop()

    async def run(self, duration: float | None = None) -> None:
        """두 커넥션을 동시에 돌린다. 하나가 죽어도 나머지는 계속(``return_exceptions``).

        완전히 독립된 세션이라 한쪽의 재연결·REG 실패가 다른 쪽 구독을 건드리지
        않는다 — 키를 나눈 목적 자체가 이 격리다.
        """
        results = await asyncio.gather(
            self._a.run(duration=duration), self._b.run(duration=duration),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                raise r


async def dry_run(
    token: str,
    codes: Sequence[str],
    *,
    types: Sequence[str] = (TYPE_TRADE, TYPE_QUOTE),
    url: str = WS_URL,
    max_frames: int = 20,
    timeout: float = 30.0,
    log: Callable[[str], None] | None = None,
) -> int:
    """구독만 하고 저장하지 않는다. REAL 프레임 몇 개를 찍고 종료.

    장 마감 후엔 REAL push 가 없어 프레임 0 으로 끝날 수 있다 — 그래도 LOGIN·REG
    성공 여부는 확인된다(연결·인증·구독 경로 검증). 파싱된 요약도 같이 찍는다.
    반환값은 본 REAL 프레임 수.
    """
    log = log or (lambda m: print(m, flush=True))
    import websockets

    seq = SeqAssigner()
    seen = 0
    first_group = chunk_codes(codes, codes_per_connection(types))[0]
    log(f"dry-run: {len(first_group)}종목 구독 · {types} · {url}")
    async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
        await ws.send(json.dumps(build_login_message(token)))
        login = json.loads(await ws.recv())
        log(f"LOGIN return_code={login.get('return_code')}")
        if int(login.get("return_code", -1)) != 0:
            return -1
        await ws.send(json.dumps(build_reg_message(first_group, types, "1")))
        reg = json.loads(await ws.recv())
        log(f"REG return_code={reg.get('return_code')} ({len(first_group)}종목)")
        if int(reg.get("return_code", -1)) != 0:
            return -1
        deadline = time.monotonic() + timeout
        while seen < max_frames and time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if msg.get("trnm") == "PING":
                await ws.send(raw)
                continue
            if msg.get("trnm") != "REAL":
                continue
            seen += 1
            ticks, quotes, vis = parse_real_message(
                msg, date.today(), seq, received_at=datetime.now()  # noqa: DTZ011,DTZ005
            )
            log(f"  REAL #{seen}: 틱 {len(ticks)} · 호가 {len(quotes)} · VI {len(vis)}"
                f" · raw={json.dumps(msg, ensure_ascii=False)[:200]}")
    log(f"dry-run 종료 · REAL 프레임 {seen} (0 이면 장중 아님 — 연결·인증·구독만 검증)")
    return seen
