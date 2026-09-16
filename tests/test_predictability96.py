"""96번 예측가능성 탐침 — 라벨이 **미래를 정확히 그만큼만** 보는지 확인한다.

95번과 같은 이유다. 라벨 창이 한 칸 밀리거나 비용이 두 번 빠지면 "예측 가능"이 숫자에서
만들어진다. 학습은 무거워서 여기서 못 돌리지만, 라벨 산술은 손으로 아는 값으로 고정할 수 있다.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "predictability96", Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "predictability.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def _rows(*t):
    return np.asarray(t, np.int64)


# ── 규칙 A: 고정 보유 ────────────────────────────────────────────────────────
def test_fixed_entry_is_next_bid_and_exit_is_exactly_h_later():
    bid = np.full(20, 100.0)
    bid[1] = 50.0                                   # 진입가는 bid[t+1] — bid[t] 가 아니다
    bid[1 + 3] = 110.0                              # t=0, H=3 → 청산 bid[4]
    y = P.label_fixed(bid, _rows(0), hold=3)
    assert abs(y[0] - (110 / 50 - 1 - P.COST)) < 1e-12


def test_fixed_does_not_peek_past_horizon():
    bid = np.full(20, 100.0)
    bid[5] = 999.0                                  # t=0, H=3 이면 창 밖(bid[4] 만 본다)
    y = P.label_fixed(bid, _rows(0), hold=3)
    assert abs(y[0] + P.COST) < 1e-12


def test_fixed_cost_subtracted_once():
    bid = np.full(20, 100.0)
    y = P.label_fixed(bid, _rows(0, 5), hold=3)     # 움직임 0 → 남는 건 비용뿐
    assert np.allclose(y, -P.COST)


def test_fixed_out_of_range_is_nan():
    bid = np.full(10, 100.0)
    y = P.label_fixed(bid, _rows(8), hold=5)        # 8+1+5 = 14 > 마지막 인덱스
    assert np.isnan(y[0])


# ── 규칙 B: 익절/손절 브래킷 ─────────────────────────────────────────────────
def test_bracket_take_profit_first_touch_wins():
    bid = np.full(30, 100.0)
    bid[4] = 100.6                                  # +60bp → +50bp 익절선 돌파
    bid[8] = 90.0                                   # 그 뒤의 폭락은 이미 나간 뒤라 무관하다
    y = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=20)
    assert abs(y[0] - (0.005 - P.COST)) < 1e-12


def test_bracket_stop_loss_when_it_comes_first():
    bid = np.full(30, 100.0)
    bid[3] = 99.4                                   # −60bp 가 먼저
    bid[6] = 101.0                                  # 뒤늦은 익절은 못 먹는다
    y = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=20)
    assert abs(y[0] - (-0.005 - P.COST)) < 1e-12


def test_bracket_starts_at_t_plus_2():
    bid = np.full(30, 100.0)
    bid[1] = 100.0
    bid[1] = 100.0
    y_no = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=20)
    assert abs(y_no[0] + P.COST) < 1e-12            # 아무 데도 안 닿음 → 마지막 bid = 진입가
    bid2 = bid.copy()
    bid2[2] = 101.0                                 # 감시 시작은 t+2 부터
    y = P.label_bracket(bid2, _rows(0), tp=0.005, sl=0.005, horizon=20)
    assert abs(y[0] - (0.005 - P.COST)) < 1e-12


def test_bracket_does_not_peek_past_horizon():
    bid = np.full(30, 100.0)
    bid[12] = 200.0                                 # t=0, H=10 → 창 [2, 11] 밖
    y = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=10)
    assert abs(y[0] + P.COST) < 1e-12               # 창 끝 bid[11]=100 으로 청산


def test_bracket_timeout_uses_last_bid_of_window():
    bid = np.full(30, 100.0)
    bid[11] = 100.2                                 # 창 끝(t+1+10) 값으로 청산, 익절선 미도달
    y = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=10)
    assert abs(y[0] - (100.2 / 100 - 1 - P.COST)) < 1e-12


def test_bracket_cost_subtracted_once():
    bid = np.full(30, 100.0)
    y = P.label_bracket(bid, _rows(0, 1, 2), tp=0.005, sl=0.005, horizon=10)
    assert np.allclose(y, -P.COST)


def test_bracket_bad_price_is_nan():
    bid = np.zeros(30)                              # 호가가 없는 구간
    y = P.label_bracket(bid, _rows(0), tp=0.005, sl=0.005, horizon=10)
    assert np.isnan(y[0])


# ── 피처 선택: 라벨/식별자가 입력으로 새면 안 된다 ──────────────────────────
def test_feature_columns_exclude_labels_and_ids():
    cols = ["day", "code", "row", "sec", "ret1", "y_F1", "hold_F1", "y_F4", "hold_F4"]
    feats = P.feature_columns(cols)
    assert feats == ["sec", "ret1"]
