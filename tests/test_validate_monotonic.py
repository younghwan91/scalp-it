import numpy as np
import pandas as pd

from swing_it.validate.monotonic import decile_table, monotonicity


def _pred(n: int, signal: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prediction = rng.normal(size=n)
    excess = prediction * signal + rng.normal(scale=0.02, size=n)
    return pd.DataFrame({
        "code": [f"{i % 50:06d}" for i in range(n)],
        "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D"),
        "excess_return": excess,
        "prediction": prediction,
        "fold": 0,
    })


def test_decile_table_shape():
    table = decile_table(_pred(2000, 0.05))
    assert list(table.columns) == ["decile", "n", "mean_excess", "median_excess"]
    assert len(table) == 10
    assert table["decile"].tolist() == list(range(10))


def test_monotone_when_prediction_carries_signal():
    got = monotonicity(_pred(2000, 0.05))
    assert got["is_monotone"] is True
    assert got["spearman"] > 0.6
    assert got["top_minus_bottom"] > 0


def test_not_monotone_when_prediction_is_noise():
    got = monotonicity(_pred(2000, 0.0))
    assert got["is_monotone"] is False


def test_inverted_signal_is_not_monotone():
    """예측이 거꾸로면 단조가 아니다 — 상위를 골라 들어갈 수 없다."""
    pred = _pred(2000, 0.05)
    pred["prediction"] = -pred["prediction"]
    assert monotonicity(pred)["is_monotone"] is False


def test_report_has_all_keys():
    got = monotonicity(_pred(500, 0.05))
    for key in ("spearman", "top_minus_bottom", "top_vs_median", "is_monotone"):
        assert key in got
