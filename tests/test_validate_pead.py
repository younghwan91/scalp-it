import numpy as np
import pandas as pd

from swing_it.validate.pead import (
    COMPARABLE_RATIO,
    PEAD_SHARPE,
    PEAD_T,
    benchmark_all_layers,
    overlap_stats,
    pead_benchmark,
)


def _curve(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(values), freq="B"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_registered_constants_match_the_spec():
    """스펙 §7 — kr-quant PEAD 검증값 Sharpe ~0.8, t ~2.2."""
    assert PEAD_SHARPE == 0.80
    assert PEAD_T == 2.20
    assert COMPARABLE_RATIO == 0.75


def test_sharpe_is_annualised_from_the_holding_period():
    values = [0.02] * 400
    values[0] = 0.01  # sd 가 0 이 되지 않게
    stats = overlap_stats(_curve(values), hold_days=40)
    series = pd.Series(values)
    expected = series.mean() / series.std(ddof=1) * np.sqrt(252 / 40)
    assert np.isclose(stats["sharpe_annual"], expected)


def test_effective_sample_divides_by_the_holding_period():
    stats = overlap_stats(_curve(list(np.random.default_rng(0).normal(size=400))), hold_days=40)
    assert stats["n_obs"] == 400
    assert stats["n_effective"] == 10


def test_effective_sample_never_drops_below_one():
    assert overlap_stats(_curve([0.01, 0.02, 0.03]), hold_days=40)["n_effective"] == 1


def test_overlap_correction_shrinks_the_t_stat():
    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=0.01, scale=0.05, size=400))
    stats = overlap_stats(_curve(values), hold_days=40)
    naive_t = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(400))
    assert stats["t_stat"] < naive_t / 5


def test_beats_pead_needs_both_sharpe_and_t():
    rng = np.random.default_rng(1)
    out = pead_benchmark(_curve(list(rng.normal(loc=0.06, scale=0.05, size=800))))
    assert out["sharpe_annual"] > PEAD_SHARPE
    assert out["t_stat"] > PEAD_T
    assert out["verdict"] == "BEATS_PEAD"


def test_loses_to_pead_when_there_is_no_edge():
    rng = np.random.default_rng(2)
    out = pead_benchmark(_curve(list(rng.normal(loc=0.0, scale=0.05, size=800))))
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_negative_mean_never_reads_as_comparable():
    rng = np.random.default_rng(3)
    out = pead_benchmark(_curve(list(rng.normal(loc=-0.03, scale=0.05, size=800))))
    assert out["sharpe_annual"] < 0
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_report_carries_the_pead_reference_values():
    out = pead_benchmark(_curve([0.01, 0.02, 0.03, 0.04] * 50))
    assert out["pead_sharpe"] == PEAD_SHARPE
    assert out["pead_t"] == PEAD_T


def test_benchmark_all_layers_tags_each_row():
    rng = np.random.default_rng(4)
    curves = {
        "supply": _curve(list(rng.normal(size=200))),
        "supply_news": _curve(list(rng.normal(size=200))),
    }
    rows = benchmark_all_layers(curves)
    assert [r["layer"] for r in rows] == ["supply", "supply_news"]
    assert all("verdict" in r for r in rows)
