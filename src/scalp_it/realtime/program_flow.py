"""장중 프로그램 매매 수급 — 만쥬의 "쏟아내는 장세면 진입을 거른다".

만쥬가 신호를 거르는 기준 셋 중 마지막:

> (비공개 회신 — 공개본에서 뺐다)
> "프로그램이 순매도(차익 실현) 중인 돌파 자리는 가짜 돌파이므로 절대 추격하지 않음"

우리는 지금까지 **대장 종목**의 매도 폭탄(`--dump-krw`)만 봤다. 만쥬가 말한 건
시장 전체 프로그램 수급이라 종목 단위로는 표현이 안 된다.

키움 ``ka90005``(프로그램매매추이)가 분 단위로 준다. 장중에는 첫 행이 **직전 분**이라
그것만 읽으면 된다.

    all_netprps          전체 순매수 금액(백만원)
    all_netprps 계열 _qty 순매수 수량(천주) ← 만쥬가 "70~80만 주"라고 말한 단위
    ndiffpro_*           비차익(실질 수급) · dfrt_*  차익

**응답 음수가 ``"--4756"`` 로 온다.** 그냥 float() 하면 예외가 나서 0 이 되고, 그러면
쏟아내는 장세가 조용히 "수급 중립"으로 읽힌다 — 필터가 통째로 무동작이 된다.

DB 도 시계도 모른다 — 응답 행을 받아 판정만 한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "PROGRAM_API_ID",
    "PROGRAM_PATH",
    "PROGRAM_ROWS_KEY",
    "ProgramFlow",
    "latest_flow",
    "program_body",
]

PROGRAM_API_ID = "ka90005"
PROGRAM_PATH = "/api/dostk/mrkcond"
PROGRAM_ROWS_KEY = "prm_trde_trnsn"


def program_body(day: str, market: str = "P101") -> dict[str, str]:
    """``ka90005`` 요청 바디. ``day`` 는 ``YYYYMMDD``, ``market`` 은 P001 코스피 · P101 코스닥."""
    return {"date": day, "amt_qty_tp": "1", "mrkt_tp": market,
            "min_tic_tp": "1", "stex_tp": "3"}


def _num(value: object) -> float:
    """``"+158549"`` · ``"--4756"`` → 숫자.

    음수가 ``--`` 로 온다(부호 하나가 아니라 둘). 이걸 안 다루면 `float()` 이 터져
    0 이 되고, **쏟아내는 장세가 수급 중립으로 읽혀 필터가 무동작이 된다.**
    """
    text = str(value or "").replace(",", "").strip()
    if not text:
        return 0.0
    neg = text.startswith("-")
    text = text.lstrip("+-")
    try:
        out = float(text)
    except ValueError:
        return 0.0
    return -out if neg else out


@dataclass(frozen=True)
class ProgramFlow:
    """직전 분의 프로그램 수급. 수량 단위는 **천주**, 금액은 **백만원**."""

    time: str
    net_qty: float          # 전체 순매수 수량(천주) — 음수면 순매도
    net_amount: float       # 전체 순매수 금액(백만원)
    nondiff_net_qty: float  # 비차익 순매수 수량(천주) — 실질 수급

    def dumping(self, *, min_net_qty: float) -> bool:
        """만쥬의 "쏟아내는 장세" — 순매도 수량이 문턱을 넘었는가.

        ``min_net_qty`` 는 **천주** 단위 양수로 준다(만쥬 70~80만 주 = 700~800).
        차익거래는 지수 헤지라 방향이 시장 심리와 다를 수 있어 **비차익**도 같이
        본다 — 둘 중 하나만 쏟아내도 만쥬는 안 산다.
        """
        if min_net_qty <= 0:
            return False
        return (self.net_qty <= -min_net_qty
                or self.nondiff_net_qty <= -min_net_qty)


def latest_flow(rows: Sequence[Mapping[str, object]]) -> ProgramFlow | None:
    """응답 행들 중 **가장 최근 것**. 장중엔 첫 행이 직전 분이다. 비면 None."""
    if not rows:
        return None
    r = rows[0]
    return ProgramFlow(
        time=str(r.get("cntr_tm") or ""),
        net_qty=_num(r.get("all_netprps_qty"))
        or (_num(r.get("ndiffpro_trde_netprps_qty"))
            + _num(r.get("dfrt_trde_netprps_qty"))),
        net_amount=_num(r.get("all_netprps")),
        nondiff_net_qty=_num(r.get("ndiffpro_trde_netprps_qty")),
    )
