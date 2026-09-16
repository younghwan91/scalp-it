"""95e번 양쪽 지정가 왕복 — **매도 체결 판정**이 진입과 대칭인지, 비용이 한 번만 빠지는지.

매도가 실제보다 쉽게 체결된 걸로 치면 스프레드를 공짜로 먹는 숫자가 나온다. 거기서 결론이 뒤집힌다.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "spread_edge.py"
spec = importlib.util.spec_from_file_location("spread_edge95", _p)
S = importlib.util.module_from_spec(spec)
sys.modules["spread_edge95"] = S
spec.loader.exec_module(S)


def test_sells_at_ask_when_ask_moves_up():
    n = 40
    bid = np.full(n, 100.0)
    ask = np.full(n, 101.0)
    bid[5] = 99.0                                   # t=3 진입가 bid[4]=100 → t=5 뚫림 체결
    e = np.full(n, -1, np.int64)
    e[3] = 5
    ask[8] = 102.0                                  # 매도1호가가 올라갔다 = 우리 101 이 팔렸다
    r = S.round_trip(bid, ask, e, 10)
    assert abs(r[3] - (101 / 100 - 1 - S.COST)) < 1e-12


def test_falls_back_to_market_exit_when_never_sold():
    n = 40
    bid = np.full(n, 100.0)
    ask = np.full(n, 101.0)
    e = np.full(n, -1, np.int64)
    e[3] = 5
    bid[15] = 90.0                                  # 창 끝(5+10=15) 매수1호가로 던진다
    r = S.round_trip(bid, ask, e, 10)
    assert abs(r[3] - (90 / 100 - 1 - S.COST)) < 1e-12


def test_unfilled_entry_is_nan():
    n = 40
    e = np.full(n, -1, np.int64)
    r = S.round_trip(np.full(n, 100.0), np.full(n, 101.0), e, 10)
    assert np.isnan(r).all()


def test_cost_subtracted_once_when_spread_is_zero():
    n = 40
    bid = np.full(n, 100.0)
    ask = np.full(n, 100.0)
    e = np.full(n, -1, np.int64)
    e[3] = 5
    r = S.round_trip(bid, ask, e, 10)
    assert abs(r[3] + S.COST) < 1e-12
