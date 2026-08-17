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
