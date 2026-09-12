"""ticksize.py는 kiwoom_client.tick_size로 위임하는 얇은 호환 레이어다.

실시간 경로(risk_guard·pair_detector)가 plain float 가격에 이 함수의
반환값을 그대로 더하고 곱한다 — Decimal이 새어나오면 그 자리에서
TypeError로 죽는다. 그래서 반환형(int/float)과, price<=0일 때 조용히
1틱을 돌려주던 기존 동작을 여기서 고정해둔다.
"""

from __future__ import annotations

import pytest

from scalp_it.ticksize import tick_size, ticks_in


@pytest.mark.parametrize(
    "price,expected",
    [
        (1_000, 1),
        (2_000, 5),
        (5_000, 10),
        (20_000, 50),
        (50_000, 100),
        (200_000, 500),
        (500_000, 1_000),
        (1_000_000, 1_000),
    ],
)
def test_tick_size_matches_krx_table(price, expected):
    assert tick_size(price) == expected


def test_tick_size_returns_plain_int_not_decimal():
    result = tick_size(70_000)
    assert type(result) is int
    # float 가격과 바로 섞어도 죽지 않아야 한다 (실시간 경로가 이렇게 쓴다).
    assert 70_000.0 + 3 * result == 70_300.0


@pytest.mark.parametrize("price", [0, -1, -100])
def test_tick_size_nonpositive_falls_back_to_one_tick(price):
    # kiwoom_client.tick_size 자체는 이 입력에 ValueError를 던지지만, 이 레포의
    # 기존 호출부 중 걸러내지 않는 곳이 있을 수 있어 예전처럼 조용히 1을 돌려준다.
    assert tick_size(price) == 1


def test_ticks_in_returns_plain_float_unrounded():
    result = ticks_in(2.2, 539)
    assert type(result) is float
    assert result == pytest.approx(2.2)
