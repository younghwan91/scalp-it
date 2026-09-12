"""KRX 호가단위(틱) 유틸.

정의가 스크립트마다 갈라지면 상한가 판정(`scripts/limitup_timing.py`)과 실시간
감지기가 서로 다른 가격을 보게 된다. 그래서 한 곳에만 둔다.

표 자체의 정본은 이제 ``kiwoom_client.tick_size`` 다 — krx-signal-engine과
kiwoom-client도 같은 표를 각자 구현하고 있던 걸 발견해서(2026-09-12) 주문을
실제로 내는 kiwoom-client 쪽으로 모았다. 이 모듈은 그 위로 얇게 얹은
호환 레이어다: 이 레포의 실시간 경로(risk_guard·pair_detector)는 가격을
plain float로 다루고 ``Decimal``과 float를 섞으면 바로 TypeError가 나므로,
반환형은 원래대로 int/float를 유지한다. 0 이하 가격에 대해 kiwoom_client
쪽은 ValueError를 던지는데, 이 레포의 기존 호출부 중 그런 값을 걸러내지
않는 곳이 있을 수 있어(예: 초기화 전 price=0) 그 경우만 원래처럼 조용히
1틱을 반환한다 — verify_* 검증 스크립트의 재현성이 걸려 있으므로 **경계
하나도 동작을 바꾸지 않는다**.

저가주는 1틱이 손익 목표폭에 비해 크다. 539원이면 1틱 1원 = 0.186% 라, −0.4% 손절은
2.2틱밖에 안 된다. 신호의 옳고 그름과 무관하게 스프레드 왕복 한 번이면 손절이 터진다.
그 판정에 :func:`ticks_in` 을 쓴다.
"""

from __future__ import annotations

from kiwoom_client.tick_size import tick_size as _tick_size

__all__ = ["tick_size", "ticks_in"]


def tick_size(price: float) -> int:
    """KRX 호가단위 (2023-01-25 이후 코스피·코스닥 공통)."""
    if price <= 0:
        return 1
    return int(_tick_size(price))


def ticks_in(width: float, price: float) -> float:
    """가격 ``price`` 대에서 폭 ``width``(원)가 몇 틱인지. 소수 그대로 돌려준다.

    반올림하지 않는 이유: "2.2틱"과 "3.0틱"의 차이가 곧 노이즈 청산 위험의 차이라
    버림하면 판정이 뭉개진다.
    """
    return abs(width) / tick_size(price)
