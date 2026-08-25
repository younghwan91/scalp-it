"""KRX 호가단위(틱) 유틸.

정의가 스크립트마다 갈라지면 상한가 판정(`scripts/limitup_timing.py`)과 실시간
감지기가 서로 다른 가격을 보게 된다. 그래서 한 곳에만 둔다 — 값은
`scripts/limitup_timing.py` 에 있던 것을 그대로 옮겼고, verify_* 검증 스크립트의
재현성이 걸려 있으므로 **경계 하나도 바꾸지 않는다**.

저가주는 1틱이 손익 목표폭에 비해 크다. 539원이면 1틱 1원 = 0.186% 라, −0.4% 손절은
2.2틱밖에 안 된다. 신호의 옳고 그름과 무관하게 스프레드 왕복 한 번이면 손절이 터진다.
그 판정에 :func:`ticks_in` 을 쓴다.
"""

from __future__ import annotations

__all__ = ["tick_size", "ticks_in"]


def tick_size(price: float) -> int:
    """KRX 호가단위 (2023-01-25 이후 코스피·코스닥 공통)."""
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def ticks_in(width: float, price: float) -> float:
    """가격 ``price`` 대에서 폭 ``width``(원)가 몇 틱인지. 소수 그대로 돌려준다.

    반올림하지 않는 이유: "2.2틱"과 "3.0틱"의 차이가 곧 노이즈 청산 위험의 차이라
    버림하면 판정이 뭉개진다.
    """
    return abs(width) / tick_size(price)
