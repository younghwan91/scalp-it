"""95f번 청산 규칙 — **미래를 안 보는지**와 우선순위(손절 먼저)가 맞는지.

익절이 손절보다 먼저 판정되면 승률이 공짜로 올라가고, 창 밖을 보면 규칙이 오라클이 된다.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "exit_rules.py"
spec = importlib.util.spec_from_file_location("exit_rules95", _p)
E = importlib.util.module_from_spec(spec)
sys.modules["exit_rules95"] = E
spec.loader.exec_module(E)


def _base(n=60):
    bid = np.full(n, 100.0)
    ask = np.full(n, 101.0)
    tick = np.full(n, 1.0)
    e = np.full(n, -1, np.int64)
    e[3] = 5                                   # t=3 진입가 bid[4]=100, 체결 t=5
    return bid, ask, tick, e


def test_take_profit_fills_when_ask_rises_above_target():
    bid, ask, tick, e = _base()
    ask[8] = 103.0                             # 목표 100+2=102 위로 올라갔다
    r = E.run_rule(bid, ask, tick, e, 2, 0, 20)
    assert abs(r[3] - (102 / 100 - 1 - E.COST)) < 1e-12


def test_stop_loss_wins_when_it_comes_first():
    bid, ask, tick, e = _base()
    bid[7] = 97.0                              # 손절선 100−2=98 아래
    ask[9] = 105.0                             # 익절은 나중에 온다 — 져야 한다
    r = E.run_rule(bid, ask, tick, e, 2, 2, 20)
    assert abs(r[3] - (97 / 100 - 1 - E.COST)) < 1e-12


def test_stop_and_take_in_same_second_prefers_stop():
    """같은 초에 둘 다면 보수적으로 손절 — 승률을 공짜로 올리지 않는다."""
    bid, ask, tick, e = _base()
    bid[7] = 97.0
    ask[7] = 105.0
    r = E.run_rule(bid, ask, tick, e, 2, 2, 20)
    assert r[3] < 0


def test_timeout_falls_back_to_market_bid():
    bid, ask, tick, e = _base()
    bid[15] = 95.0                             # 시한 f+10=15
    r = E.run_rule(bid, ask, tick, e, 2, 0, 10)
    assert abs(r[3] - (95 / 100 - 1 - E.COST)) < 1e-12


def test_does_not_see_past_the_horizon():
    bid, ask, tick, e = _base()
    ask[30] = 200.0                            # 창(f+10=15) 밖의 큰 값
    r = E.run_rule(bid, ask, tick, e, 2, 0, 10)
    assert r[3] < 0.01, "창 밖을 보면 규칙이 오라클이 된다"


def test_unfilled_entry_is_nan():
    bid, ask, tick, e = _base()
    e[:] = -1
    r = E.run_rule(bid, ask, tick, e, 2, 0, 10)
    assert np.isnan(r).all()
