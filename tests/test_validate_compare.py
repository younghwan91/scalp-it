import numpy as np
import pandas as pd
import pytest

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.validate.compare import compare_layers, llm_verdict, paired_bootstrap

NEWS = ["article_count"]
LLM = ["llm_specificity"]


def _ds(n: int, *, llm_is_signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [*FEATURE_COLUMNS, *NEWS, *LLM]
    df = pd.DataFrame({col: rng.normal(size=n) for col in cols})
    df["code"] = [f"{i % 30:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 30, unit="D")
    noise = rng.normal(scale=0.02, size=n)
    df["excess_return"] = noise + (df["llm_specificity"] * 0.08 if llm_is_signal else 0.0)
    return df[["code", "date", *cols, "excess_return"]]


def _spec() -> dict[str, list[str]]:
    return {
        "supply": list(FEATURE_COLUMNS),
        "supply_news": [*FEATURE_COLUMNS, *NEWS],
        "supply_news_llm": [*FEATURE_COLUMNS, *NEWS, *LLM],
    }


def _portfolio(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_paired_bootstrap_detects_a_real_difference():
    base = np.random.default_rng(0).normal(scale=0.01, size=200)
    out = paired_bootstrap(
        _portfolio(list(base + 0.02)), _portfolio(list(base)), n_boot=1000
    )
    assert np.isclose(out["mean_diff"], 0.02, atol=0.005)
    assert out["ci_low"] > 0
    assert out["p_greater"] > 0.95
    assert out["n_dates"] == 200


def test_paired_bootstrap_finds_nothing_when_series_are_identical():
    values = list(np.random.default_rng(1).normal(scale=0.01, size=200))
    out = paired_bootstrap(_portfolio(values), _portfolio(values), n_boot=1000)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_paired_bootstrap_pairs_on_date_not_position():
    a = _portfolio([0.1, 0.2, 0.3])
    b = _portfolio([0.1, 0.2, 0.3]).iloc[::-1].reset_index(drop=True)
    assert paired_bootstrap(a, b, n_boot=200)["mean_diff"] == 0.0


def test_paired_bootstrap_rejects_disjoint_dates():
    a = _portfolio([0.1, 0.2])
    b = _portfolio([0.1, 0.2])
    b["date"] = b["date"] + pd.Timedelta(days=365)
    with pytest.raises(ValueError, match="공통 날짜"):
        paired_bootstrap(a, b, n_boot=100)


def test_compare_layers_returns_one_row_per_layer():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [r["layer"] for r in result["rows"]] == [
        "supply", "supply_news", "supply_news_llm"
    ]
    assert [r["n_features"] for r in result["rows"]] == [7, 8, 9]


def test_all_layers_share_an_identical_null():
    """널이 다르면 세 층이 다른 표본 위에 있다는 뜻이고, 비교는 그 순간 무효다."""
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert len({r["null_mean"] for r in result["rows"]}) == 1
    assert len({r["null_p95"] for r in result["rows"]}) == 1


def test_every_layer_carries_its_composite_verdict():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    for row in result["rows"]:
        assert row["overall_verdict"] in {"PASS", "FAIL"}
        assert set(row["detail"]["checks"]) == {
            "전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"
        }


def test_every_layer_carries_selectivity_and_downside():
    """평균만 보면 뉴스·LLM 층의 진짜 기여를 놓친다 (baseline-verdict 시사점 2·3)."""
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    for row in result["rows"]:
        assert 0.0 < row["selection_ratio"] <= 1.0
        assert row["mean_pool_per_date"] == 30.0
        assert row["low_selectivity"] is False  # 30건 중 15 = 0.5, 초과가 아니다
        assert 0.0 <= row["tail_capture_rate"] <= 1.0
        assert "p10" in row and "p10_gap" in row


def test_a_signal_layer_avoids_the_tail_better_than_a_noise_layer():
    signal = compare_layers(
        _ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200
    )["rows"][-1]
    noise = compare_layers(
        _ds(1500, llm_is_signal=False), _spec(), n_trials=50, n_boot=200
    )["rows"][-1]
    assert signal["tail_capture_rate"] < noise["tail_capture_rate"]


def test_llm_justified_when_llm_feature_carries_the_signal():
    result = compare_layers(
        _ds(3000, llm_is_signal=True), _spec(), n_trials=200, n_boot=1000,
        cost_one_way=0.0,
    )
    assert result["llm"]["verdict"] == "LLM_JUSTIFIED"
    assert result["llm"]["delta"] > 0


def test_llm_not_justified_when_llm_feature_is_noise():
    result = compare_layers(
        _ds(3000, llm_is_signal=False), _spec(), n_trials=200, n_boot=1000
    )
    assert result["llm"]["verdict"] == "LLM_NOT_JUSTIFIED"


def test_llm_not_justified_when_layer3_only_passes_the_naive_gate():
    """계획 1 이 걸린 함정 — gate 만 PASS 고 종합은 FAIL 이면 정당화되지 않는다."""
    rows = [
        {"layer": "supply_news", "strategy_mean": 0.001, "verdict": "PASS",
         "overall_verdict": "FAIL"},
        {"layer": "supply_news_llm", "strategy_mean": 0.004, "verdict": "PASS",
         "overall_verdict": "FAIL"},
    ]
    deltas = [
        {"layer": "supply_news_llm", "baseline": "supply_news", "mean_diff": 0.003,
         "ci_low": 0.001, "ci_high": 0.005, "p_greater": 0.99, "n_dates": 100}
    ]
    assert llm_verdict(rows, deltas)["verdict"] == "LLM_NOT_JUSTIFIED"


def test_deltas_compare_each_layer_to_the_previous_one():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [(d["layer"], d["baseline"]) for d in result["deltas"]] == [
        ("supply_news", "supply"),
        ("supply_news_llm", "supply_news"),
    ]


def test_pead_rows_accompany_every_layer():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [r["layer"] for r in result["pead"]] == [
        "supply", "supply_news", "supply_news_llm"
    ]


def test_sample_metadata_is_reported():
    ds = _ds(1500, llm_is_signal=True)
    result = compare_layers(ds, _spec(), n_trials=50, n_boot=200)
    assert result["sample"]["n_rows"] == len(ds)
    assert result["sample"]["n_codes"] == ds["code"].nunique()
    assert result["sample"]["top_n"] == 15


def test_compare_layers_rejects_a_spec_whose_layers_are_not_nested():
    bad = {"supply": list(FEATURE_COLUMNS), "supply_news": NEWS}
    with pytest.raises(ValueError, match="누적"):
        compare_layers(_ds(1500, llm_is_signal=True), bad, n_trials=50, n_boot=200)
