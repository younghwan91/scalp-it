"""98번 체크포인트 선택 편향 — 선택 규칙 재현과 기준 분포 산술을 고정한다.

여기서 재현이 어긋나면 "선택 프리미엄"이 규칙이 아니라 내 코드에서 나온 게 된다. 그래서
`pick()` 과 같은 필터·같은 동점 처리·같은 순회 순서를 손으로 아는 경우로 박아 둔다.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "selection98", Path(__file__).resolve().parents[1] / "scripts" / "selection_98" / "selection_bias.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _r(seed, steps, n, day, mean):
    return {"seed": seed, "steps": steps, "val_n": n, "val_daymean_bp": day, "val_mean_bp": mean}


# ── 선택 규칙 재현: rl_*/dqn.py 의 pick() 과 같아야 한다 ────────────────────
def test_eligible_needs_thirty_trades():
    rows = [_r(1, 1, 29, 100.0, 0.0), _r(1, 2, 30, -5.0, 0.0)]
    assert [r["steps"] for r in S.eligible(rows)] == [2], "거래 29건짜리가 통과하면 규칙이 다른 것이다"


def test_eligible_drops_nan_daymean():
    rows = [_r(1, 1, 50, float("nan"), 0.0), _r(1, 2, 50, -5.0, 0.0)]
    assert [r["steps"] for r in S.eligible(rows)] == [2]


def test_argmax_keeps_first_on_tie():
    rows = [_r(1, 1, 50, 7.0, 0.0), _r(2, 2, 50, 7.0, 0.0)]      # pick() 은 strict > 라 먼저가 이긴다
    assert S.argmax_by(rows, "val_daymean_bp")["seed"] == 1


def test_argmax_picks_the_maximum():
    rows = [_r(1, 1, 50, -9.0, 0.0), _r(1, 2, 50, 3.0, 0.0), _r(1, 3, 50, -1.0, 0.0)]
    assert S.argmax_by(rows, "val_daymean_bp")["steps"] == 2


def test_two_rules_can_disagree():
    """일평균으로 고르면 A, 풀링 평균으로 고르면 B — 이게 갈리는 게 이 측정의 전부다."""
    rows = [_r(1, 1, 31, 160.0, -17.0), _r(1, 2, 200, -5.0, -4.0)]
    assert S.argmax_by(rows, "val_daymean_bp")["steps"] == 1
    assert S.argmax_by(rows, "val_mean_bp")["steps"] == 2


# ── 무작위 선택 기준 분포 ───────────────────────────────────────────────────
def test_expected_max_of_one_is_the_mean():
    v = np.array([1.0, 2.0, 6.0])
    assert abs(S.expected_max(v, 1) - 3.0) < 1e-12


def test_expected_max_of_all_is_the_max():
    v = np.array([1.0, 2.0, 6.0])
    assert abs(S.expected_max(v, 3) - 6.0) < 1e-12


def test_expected_max_is_monotone_in_m():
    v = np.array([-5.0, -1.0, 0.5, 3.0, 40.0])
    got = [S.expected_max(v, m) for m in range(1, 6)]
    assert all(got[i] < got[i + 1] for i in range(4))


def test_expected_max_matches_brute_force():
    """비복원 2개 뽑기의 최댓값 기댓값 — 손으로 셀 수 있는 경우로 맞춘다."""
    v = np.array([1.0, 2.0, 6.0])                 # 쌍 (1,2)(1,6)(2,6) → 최댓값 2,6,6
    assert abs(S.expected_max(v, 2) - (2 + 6 + 6) / 3) < 1e-12


# ── 일평균과 풀링 평균의 벌어짐 ─────────────────────────────────────────────
def test_gap_is_daymean_minus_pooled_mean():
    assert abs(S.gap_bp(_r(1, 1, 32, 166.6, -17.5)) - 184.1) < 1e-9


def test_gap_is_zero_when_days_are_balanced():
    assert abs(S.gap_bp(_r(1, 1, 100, -4.0, -4.0))) < 1e-12
