"""95번 천장 측정 — 오라클 계산이 **미래를 정확히 그만큼만** 보는지 확인한다.

여기서 한 칸이 밀리면 상한이 부풀거나 줄어 결론이 뒤집힌다. 그래서 값을 손으로 아는 경우로 고정한다.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "ceiling95", Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "ceiling.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_oracle_picks_best_exit_in_window():
    #        t:  0    1     2     3     4
    bid = np.array([100, 100, 110, 130, 100], float)
    ask = np.array([101, 101, 111, 131, 101], float)
    tk, mk = C.oracle(bid, ask, h=2)              # t=0 → 진입 ask[1]=101, 매도 max(bid[2],bid[3])=130
    assert abs(tk[0] - (130 / 101 - 1 - C.COST)) < 1e-12
    assert abs(mk[0] - (131 / 100 - 1 - C.COST)) < 1e-12   # 메이커 진입 bid[1]=100, 매도 max(ask)=131


def test_window_does_not_reach_beyond_horizon():
    bid = np.array([100, 100, 100, 100, 999], float)      # 큰 값이 창 밖(t+3)에 있다
    ask = bid + 1
    tk, _ = C.oracle(bid, ask, h=1)                       # t=0 창 = [t+2, t+2]
    assert tk[0] < 0, "창 밖 미래를 보면 안 된다"


def test_missing_prices_become_nan():
    bid = np.array([0, 0, 0, 0], float)
    ask = np.array([0, 0, 0, 0], float)
    tk, mk = C.oracle(bid, ask, h=2)
    assert np.isnan(tk).all() and np.isnan(mk).all()


def test_cost_is_subtracted_once():
    bid = np.full(6, 100.0)
    ask = np.full(6, 100.0)                                # 스프레드 0 · 움직임 없음
    tk, _ = C.oracle(bid, ask, h=3)
    assert abs(tk[0] + C.COST) < 1e-12


def test_pessimistic_fill_requires_price_through():
    bid = np.array([100.0] * 20)
    e = C.fill_second(bid)
    assert (e < 0).all(), "가격이 안 뚫렸는데 체결로 보면 역선택을 과소평가한다"
    bid2 = bid.copy()
    bid2[5] = 99.0                                     # t=3 지정가 bid[4]=100 → t=5 에 뚫린다
    e2 = C.fill_second(bid2)
    assert e2[3] == 5


def test_pessimistic_return_measured_from_fill_second():
    bid = np.array([100.0] * 20)
    bid[5] = 99.0
    ask = np.array([101.0] * 20)
    ask[7] = 120.0
    e = C.fill_second(bid)
    mp = C.maker_pessimistic(bid, e, C.window_max(ask, 3))   # 체결 t=5 → 매도창 [6,8] 최댓값 120
    assert abs(mp[3] - (120 / 100 - 1 - C.COST)) < 1e-12


def test_short_oracle_profits_when_price_falls():
    #          t: 0    1    2    3    4
    bid = np.array([100, 100, 90, 80, 80], float)
    ask = np.array([101, 101, 91, 81, 81], float)
    tk, mk = C.oracle_short(bid, ask, h=2)      # t=0 → 진입 bid[1]=100, 환매수 min(ask[2],ask[3])=81
    assert abs(tk[0] - (100 / 81 - 1 - C.COST)) < 1e-12
    assert abs(mk[0] - (101 / 80 - 1 - C.COST)) < 1e-12


def test_short_loses_when_price_rises():
    bid = np.array([100, 100, 110, 120, 120], float)
    tk, _ = C.oracle_short(bid, bid + 1, h=2)
    assert tk[0] < 0


def test_tick_bins_split_by_tick_width():
    tk = np.array([-30.0, -10.0, -1.0, +5.0])
    mk = np.array([-5.0, -2.0, +1.0, +3.0])
    tb = np.array([20.0, 8.0, 3.0, 1.0])            # 저가주 → 고가주
    out = C._by_tick(tk, mk, tb)
    assert out["0~2"]["maker_pess"] is None, "체결 자료가 없으면 비관 열은 비어 있어야 한다"
    assert out["0~2"]["n"] == 1 and out["0~2"]["taker"] == 5.0
    assert out["12~25"]["n"] == 1 and out["12~25"]["taker"] == -30.0
    assert sum(r["n"] for r in out.values()) == 4, "구간이 모든 초를 덮어야 한다"


def test_tick_bins_report_pessimistic_when_given():
    tk = np.array([-30.0, -10.0])
    mk = np.array([-5.0, -2.0])
    tb = np.array([20.0, 3.0])
    mp = np.array([-1.0, np.nan])                   # 첫 행만 체결됐다
    out = C._by_tick(tk, mk, tb, mp)
    assert out["12~25"]["maker_pess"] == -1.0 and out["12~25"]["n_pess"] == 1
    assert out["2~4"]["maker_pess"] is None and out["2~4"]["n_pess"] == 0
