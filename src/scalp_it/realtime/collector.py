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
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

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
) -> tuple[list[dict], list[dict], list[dict]]:
    """REAL 프레임(문자열 또는 dict)을 ``(ticks, quotes, vi_events)`` 로 파싱한다.

    REAL 이 아니거나 data 가 없으면 빈 세 리스트. 알 수 없는 type 은 무시한다.
    ``received_at`` 은 VI 발동시각 FID 결손 시 대체할 수신 시각이다.
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


# --------------------------------------------------------------- 데몬


class KiwoomRealtimeCollector:
    """LOGIN→REG→listen→파싱→배치 insert. 재연결·재REG·graceful flush."""

    def __init__(
        self,
        token: str,
        codes: Sequence[str],
        *,
        conn: Any = None,
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
    ) -> None:
        self._token = token
        self._types = list(types)
        self._url = url
        self._conn = conn
        self._store = store
        self._flush_interval = max(0.2, float(flush_interval))
        self._trade_date = trade_date or date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
        self._on_frame = on_frame
        self._max_backoff = max_backoff
        self._log = log or (lambda m: print(m, flush=True))

        if per_conn is None:
            per_conn = codes_per_connection(types, reg_limit=reg_limit)
        self._groups = chunk_codes(codes, per_conn)

        self._seq = SeqAssigner()
        self._tick_buf: list[dict] = []
        self._quote_buf: list[dict] = []
        self._vi_buf: list[dict] = []
        self._stop = asyncio.Event()

        # 통계
        self.frames_seen = 0
        self.ticks_stored = 0
        self.quotes_stored = 0
        self.vi_stored = 0
        self.reconnects = 0

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
                f"호가 {self.quotes_stored:,} · VI {self.vi_stored:,} · "
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

    async def _login_and_reg(self, ws, grp_no: str, codes: list[str]) -> None:
        await ws.send(json.dumps(build_login_message(self._token)))
        login = json.loads(await ws.recv())
        if int(login.get("return_code", -1)) != 0:
            raise RuntimeError(f"LOGIN 실패: {login}")
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
            msg, self._trade_date, self._seq, received_at=now
        )
        self._tick_buf.extend(ticks)
        self._quote_buf.extend(quotes)
        self._vi_buf.extend(vis)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_or_stop(self._flush_interval)
            self._flush()

    def _flush(self) -> None:
        if not self._store or self._conn is None:
            self._tick_buf.clear()
            self._quote_buf.clear()
            self._vi_buf.clear()
            return
        from .store import insert_quotes, insert_ticks, insert_vi_events

        if self._tick_buf:
            batch, self._tick_buf = self._tick_buf, []
            self.ticks_stored += insert_ticks(self._conn, batch)
        if self._quote_buf:
            batch, self._quote_buf = self._quote_buf, []
            self.quotes_stored += insert_quotes(self._conn, batch)
        if self._vi_buf:
            batch, self._vi_buf = self._vi_buf, []
            self.vi_stored += insert_vi_events(self._conn, batch)

    async def _sleep_or_stop(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)


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
