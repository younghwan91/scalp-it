"""불타기 — 만쥬 A-4 Pyramiding.

> - Martingale / Averaging Down (물타기): **ABSOLUTELY FORBIDDEN**.
> - Pyramiding (불타기):
>   * Only permitted when position is in confirmed profit **>= +3.0% ~ +4.0%**.
>   * Sizing limit: **Maximum 2x** initial size.

STATUS 의 "빠뜨린 축 #2 불타기·평단 하향 진입"이 **1·2차 타점 무출처**라 대기 중이었다.
이 지시서가 그 수치를 줬다(+3%, 2배).

물타기는 구조적으로 불가능하다 — 이 함수는 **이익 구간에서만** 배수를 올린다.
"""
import pytest

from scalp_it.realtime.risk_guard import pyramid_multiplier


@pytest.mark.parametrize("ret,expected", [
    (-0.05, 1.0),   # 손실 — 물타기 금지
    (0.0, 1.0),
    (0.029, 1.0),   # 문턱 직전
    (0.03, 2.0),    # +3% 도달 → 2배
    (0.10, 2.0),    # 상한은 2배
])
def test_only_confirmed_profit_pyramids(ret, expected):
    assert pyramid_multiplier(ret, arm=0.03, max_mult=2.0) == expected


def test_off_when_arm_is_zero():
    assert pyramid_multiplier(0.10, arm=0.0, max_mult=2.0) == 1.0


def test_never_below_one():
    """어떤 입력에도 1 미만으로 안 내려간다 — 축소는 다른 규칙의 몫이다."""
    assert pyramid_multiplier(-0.5, arm=0.03, max_mult=2.0) == 1.0
