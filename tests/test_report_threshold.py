import numpy as np
import pandas as pd

from swing_it.report.threshold import concurrency_threshold


def _pred(n_days: int, per_day: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for i in range(per_day):
            rows.append({
                "code": f"{i:06d}",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                "excess_return": rng.normal(scale=0.05),
                "prediction": rng.normal(),
                "fold": 0,
            })
    return pd.DataFrame(rows)


def test_threshold_yields_the_target_concurrency():
    pred = _pred(200, 50)
    got = concurrency_threshold(pred, hold_days=40, target_concurrent=20)
    assert np.isclose(got["expected_concurrent"], 20, rtol=0.25)


def test_threshold_is_a_prediction_quantile():
    pred = _pred(200, 50)
    got = concurrency_threshold(pred, hold_days=40, target_concurrent=20)
    passing = (pred["prediction"] >= got["threshold"]).sum()
    assert np.isclose(passing / pred["date"].nunique(), got["expected_per_day"], rtol=0.25)


def test_higher_target_lowers_the_threshold():
    pred = _pred(200, 50)
    low = concurrency_threshold(pred, target_concurrent=10)["threshold"]
    high = concurrency_threshold(pred, target_concurrent=40)["threshold"]
    assert high < low


def test_report_has_all_keys():
    got = concurrency_threshold(_pred(50, 20))
    for key in ("threshold", "expected_per_day", "expected_concurrent", "n_train_days"):
        assert key in got


def _folded(n_days: int = 200, per_day: int = 30, seed: int = 0) -> pd.DataFrame:
    """폴드마다 예측 스케일이 다른 프레임 — 실제 walk-forward 가 그렇다."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold, shift in enumerate([0.0, 3.0, -2.0, 1.0]):
        for d in range(n_days // 4):
            for i in range(per_day):
                rows.append({
                    "code": f"{i:06d}",
                    "date": pd.Timestamp("2024-01-01")
                            + pd.Timedelta(days=fold * (n_days // 4) + d),
                    "excess_return": rng.normal(scale=0.05),
                    "prediction": rng.normal() + shift,
                    "fold": fold,
                })
    return pd.DataFrame(rows)


def test_threshold_is_per_fold_when_folds_differ():
    """폴드마다 예측 스케일이 다르면 임계도 폴드마다 달라야 한다.

    회귀 방지: 전역 하나로 잡으면 스케일이 높은 폴드가 통과분을 독식하고
    다른 폴드는 0 건이 된다. 실측에서 그렇게 됐다.
    """
    got = concurrency_threshold(_folded(), hold_days=40, target_concurrent=20)
    assert "per_fold" in got
    assert len(got["per_fold"]) == 4
    values = [f["threshold"] for f in got["per_fold"]]
    assert max(values) - min(values) > 1.0   # 스케일 차이가 임계에 반영됐다


def test_every_fold_contributes_candidates():
    got = concurrency_threshold(_folded(), hold_days=40, target_concurrent=20)
    for fold in got["per_fold"]:
        assert fold["n_pass"] > 0


def test_expected_concurrent_is_near_target_overall():
    got = concurrency_threshold(_folded(), hold_days=40, target_concurrent=20)
    assert np.isclose(got["expected_concurrent"], 20, rtol=0.35)


def test_single_fold_frame_still_works():
    frame = _folded()
    frame = frame[frame["fold"] == 0]
    got = concurrency_threshold(frame, hold_days=40, target_concurrent=20)
    assert len(got["per_fold"]) == 1
    assert got["per_fold"][0]["n_pass"] > 0
