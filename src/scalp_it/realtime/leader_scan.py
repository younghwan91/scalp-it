"""장중 시장 전체 대장 스캔 — 만쥬가 HTS 순간체결량·등락률 상위 창으로 하는 일.

만쥬는 대장을 **시장 전체에서** 찾는다:

> "오전 09:00~09:30 사이, (비공개 회신 — 공개본에서 뺐다)

우리는 전날 밤에 45종목을 골라 그것만 구독했다(키움 단일 앱키 = 50종목 제약).
게다가 선정 기준이 **전날 거래대금**이라, 오늘 상한가로 가는 종목과 거의 안 겹친다:

    최근 7거래일 상한가 52건 중 우리 유니버스 안에 있던 것 1건 = **커버리지 2%**

방아쇠를 아무리 정확히 맞춰도 대장을 안 보고 있으면 신호가 안 뜬다. 그래서
등락률 상위 조회(``ka10027``)를 주기적으로 돌려 **시장 전체에서** 대장 후보를
찾고, 그 종목과 짝꿍을 장중에 구독에 추가한다.

응답에 마침 만쥬가 보는 것이 다 들어 있다:

    flu_rt      등락률 — "20% 이상으로 수직 상승"
    sel_req     매도 잔량 — **0 이면 상한가 잠김**
    buy_req     매수 잔량 — 만쥬의 "상한가 잔량"
    cntr_str    체결강도 — "매수 호가로 가장 빠르게 반응하는 종목"

DB 도 시계도 모른다 — 응답 행을 받아 판정만 한다(다른 realtime 모듈과 같은 계약).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from krx_quant_core.market.session import in_window as in_adoption_window
from krx_quant_core.market.session import past_cutoff

__all__ = [
    "RANK_API_ID",
    "RANK_PATH",
    "RANK_ROWS_KEY",
    "ScanRow",
    "in_adoption_window",
    "leader_candidates",
    "parse_scan_rows",
    "past_cutoff",
    "rank_body",
    "sector_followers",
]

#: 전일대비등락률상위요청. 코스닥 전체를 등락률 순으로 최대 200행 준다.
RANK_API_ID = "ka10027"
RANK_PATH = "/api/dostk/rkinfo"
#: 응답의 행 배열 키.
RANK_ROWS_KEY = "pred_pre_flu_rt_upper"


def rank_body(market: str = "101") -> dict[str, str]:
    """``ka10027`` 요청 바디. ``market`` 은 000 전체 · 001 코스피 · 101 코스닥.

    상한가·하한가를 **포함**해서 받는다(``updown_incls="1"``) — 이미 잠긴 대장도
    짝꿍매매의 방아쇠라 빼면 안 된다.
    """
    return {
        "mrkt_tp": market, "sort_tp": "1", "trde_qty_cnd": "0000", "stk_cnd": "0",
        "crd_cnd": "0", "updown_incls": "1", "pric_cnd": "0", "trde_prica_cnd": "0",
        "stex_tp": "3",
    }


# ----- 시각 게이트 -----------------------------------------------------------
# ``past_cutoff``(신규 진입 차단 시각)과 ``in_adoption_window``(새 대장 낙점 창)는
# krx-quant-core ``market.session`` 의 ``past_cutoff``·``in_window`` 로 옮겼다. 규약은
# 그대로다 — 빈 값·형식 오류는 막지 않는다(오타로 하루를 날리지 않는다).
#
# 근거(만쥬 통합 매뉴얼 49 §1-5): "오전 핵심 구간 매매 후 **오후 추격 매매는 철저히
# 배제**". 컷오프가 막는 건 **신규 진입뿐**이고 열린 포지션의 청산은 계속 돈다. 창을
# 벗어나도 **이미 받아들인 대장은 종일 그대로 돈다** — 막는 건 '새 대장 낙점'뿐이다.
# cron 은 ``0900-1000`` 창을 건다.


def _num(value: object) -> float:
    """``"+29.98"`` · ``"-1331"`` 같은 부호 붙은 문자열을 숫자로. 못 읽으면 0."""
    try:
        return float(str(value).replace("+", "").replace(",", "").strip() or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _code(value: object) -> str:
    """``"043200_AL"`` → ``"043200"``. 응답 종목코드에 거래소 접미사가 붙어 온다."""
    return str(value or "").strip().split("_")[0]


@dataclass(frozen=True)
class ScanRow:
    """등락률 상위 한 행. 만쥬가 창에서 읽는 것만 남긴다."""

    code: str
    name: str
    price: float
    flu_rt: float          # 등락률(%)
    sell_qty: float        # 매도 잔량 — 0 이면 상한가 잠김
    buy_qty: float         # 매수 잔량 — 만쥬의 "상한가 잔량"
    cntr_str: float        # 체결강도
    volume: float

    @property
    def locked(self) -> bool:
        """상한가 잠김 — 매도 잔량이 0 이고 매수가 쌓여 있다."""
        return self.sell_qty <= 0 and self.buy_qty > 0


def parse_scan_rows(rows: Iterable[Mapping[str, object]]) -> list[ScanRow]:
    """``ka10027`` 응답 행들을 :class:`ScanRow` 로. 코드가 없는 행은 버린다."""
    out: list[ScanRow] = []
    for r in rows:
        code = _code(r.get("stk_cd"))
        if not code:
            continue
        out.append(ScanRow(
            code=code, name=str(r.get("stk_nm") or "").strip(),
            price=abs(_num(r.get("cur_prc"))), flu_rt=_num(r.get("flu_rt")),
            sell_qty=_num(r.get("sel_req")), buy_qty=_num(r.get("buy_req")),
            cntr_str=_num(r.get("cntr_str")), volume=_num(r.get("now_trde_qty")),
        ))
    return out


def leader_candidates(rows: Sequence[ScanRow], *, min_flu_rt: float = 20.0,
                      min_buy_qty: float = 0.0,
                      prefer_unlocked: bool = True) -> list[ScanRow]:
    """대장 후보 — 만쥬의 "20% 이상으로 수직 상승".

    **아직 안 잠긴 종목을 먼저 준다.** 만쥬가 2등주를 사는 시점은 대장이 상한가
    호가 3개를 남긴 순간이다. 이미 잠긴 대장은 그 순간이 지나간 것이라, 구독 자리를
    거기 쓰면 정작 방아쇠가 남아 있는 종목을 못 본다.

    2026-08-28 실측이 정확히 그랬다: 코이즈·모아데이타 같은 **잠긴 종목이 하루 종일
    등락률 상위를 붙잡고** 자리를 다 먹었고, 그날 만쥬 방아쇠 구간(3호가 이내·미잠김)에
    틱이 들어온 종목은 50개 구독 중 **2개**뿐이었다. 문턱만 20%→12% 로 낮추는 걸로는
    안 고쳐진다 — 상위 N 을 등락률로 자르면 상위는 언제나 29%대 잠긴 종목이다.

    ``min_buy_qty`` 를 주면 잠긴 종목에 한해 매수 잔량이 그만큼은 쌓여 있어야 한다
    (만쥬 §5-2 "수십만~수백만 주를 안정적으로 굳히면"). 아직 안 잠긴 종목은 상한가
    호가가 없어 잔량을 잴 수 없으니 이 조건을 안 건다.
    """
    picked = [r for r in rows if r.flu_rt >= min_flu_rt]
    if min_buy_qty > 0:
        picked = [r for r in picked if not r.locked or r.buy_qty >= min_buy_qty]
    if prefer_unlocked:
        # 미잠김 먼저, 그 안에서 등락률 높은 순. 잠긴 건 뒤로 밀어 남는 자리만 쓴다 —
        # 잠긴 대장도 짝꿍 청산 신호(상한가 이탈)로는 쓸모가 있어 버리지는 않는다.
        return sorted(picked, key=lambda r: (r.locked, -r.flu_rt, r.code))
    return sorted(picked, key=lambda r: (-r.flu_rt, r.code))


def sector_followers(rows: Sequence[ScanRow], leader: str,
                     groups: Mapping[str, str], *, rank_min: int = 2,
                     rank_max: int = 3) -> list[str]:
    """대장과 같은 무리에서 등락률 ``rank_min~rank_max`` 위를 짝꿍으로.

    ``groups`` 는 ``{종목: 무리키}`` 다. 호출부가 실전 테마를 알면 테마 코드를,
    모르면 27종 섹터를 넣는다 — 감지기 :meth:`PairDetector.peers` 와 **같은 무리**
    여야 스캐너가 붙인 짝꿍을 감지기가 그대로 고른다.

    순위는 **대장을 포함해** 매기고 대장 자신만 뺀다 — 감지기의 `_dynamic_pairs` 와
    같은 규칙이라야 "2·3등주"라는 말이 두 곳에서 같은 뜻이 된다.

    이미 잠긴 종목은 짝꿍에서 뺀다. 상한가에 붙어 있으면 살 수가 없고, 만쥬가
    사는 것도 **아직 안 간** 2·3등주다.
    """
    key = groups.get(leader)
    if not key:
        return []
    same = [r for r in rows if groups.get(r.code) == key]
    same.sort(key=lambda r: (-r.flu_rt, r.code))
    out: list[str] = []
    for rank, row in enumerate(same, start=1):
        if row.code == leader or rank < rank_min or rank > rank_max or row.locked:
            continue
        out.append(row.code)
    return out
