import numpy as np
import pandas as pd

from swing_it.validate.layer_verdict import (
    CHECK_NAMES,
    downside_stats,
    layer_verdict,
    selectivity,
)


def _pred(n_dates: int, n_per_date: int, *, signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n_per_date):
            excess = rng.normal(scale=0.05)
            rows.append(
                {
                    "code": f"{i:06d}",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                    "excess_return": excess,
                    "prediction": excess if signal else rng.normal(),
                    "fold": d * 4 // n_dates,
                }
            )
    return pd.DataFrame(rows)


def test_check_names_match_the_baseline_verdict_format():
    assert CHECK_NAMES == (
        "전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"
    )


def test_selectivity_reports_pool_size_and_ratio():
    out = selectivity(_pred(20, 40, signal=False), top_n=15)
    assert out["mean_pool_per_date"] == 40.0
    assert out["median_pool_per_date"] == 40.0
    assert out["min_pool_per_date"] == 40
    assert np.isclose(out["selection_ratio"], 15 / 40)
    assert out["low_selectivity"] is False


def test_selection_ratio_is_capped_at_one_when_the_pool_is_smaller():
    """풀이 10개인데 상위 15를 고르면 전부 담는 것이다. 1.0 을 넘을 수 없다."""
    out = selectivity(_pred(20, 10, signal=False), top_n=15)
    assert out["selection_ratio"] == 1.0
    assert out["low_selectivity"] is True


def test_low_selectivity_trips_above_half():
    """계획 1 폴드 3 은 하루 24.1건 중 상위 15 = 0.62 였다."""
    assert selectivity(_pred(20, 24, signal=False), top_n=15)["low_selectivity"] is True
    assert selectivity(_pred(20, 40, signal=False), top_n=15)["low_selectivity"] is False


def test_random_prediction_captures_the_tail_at_the_expected_rate():
    out = downside_stats(_pred(200, 40, signal=False), top_n=15, tail_quantile=0.2)
    assert np.isclose(out["expected_tail_rate"], 0.2)
    assert abs(out["tail_capture_rate"] - 0.2) < 0.03
    assert abs(out["tail_avoidance"]) < 0.03
    assert abs(out["p10_gap"]) < 0.02


def test_perfect_prediction_avoids_the_tail_entirely():
    out = downside_stats(_pred(60, 40, signal=True), top_n=15, tail_quantile=0.2)
    assert out["tail_capture_rate"] == 0.0
    assert np.isclose(out["tail_avoidance"], 0.2)
    assert out["p10"] > out["pool_p10"]  # 선택분의 하방이 풀보다 얕다
    assert out["p10_gap"] > 0


def test_inverted_prediction_concentrates_in_the_tail():
    pred = _pred(60, 40, signal=True)
    pred["prediction"] = -pred["prediction"]
    out = downside_stats(pred, top_n=15, tail_quantile=0.2)
    assert out["tail_capture_rate"] > 0.4
    assert out["tail_avoidance"] < 0
    assert out["p10_gap"] < 0


def test_noise_layer_fails_overall():
    out = layer_verdict(_pred(60, 40, signal=False), top_n=15, n_trials=100)
    assert out["overall_verdict"] == "FAIL"
    assert out["checks"]["전체_널_대비"] is False


def test_perfect_layer_passes_every_check():
    out = layer_verdict(
        _pred(80, 40, signal=True), top_n=15, n_trials=100, cost_one_way=0.0
    )
    assert out["overall_verdict"] == "PASS"
    assert all(out["checks"].values())


def test_placebo_check_passes_only_when_shuffled_labels_fail():
    """라벨을 섞었는데 PASS 가 나오면 누수다. 그래서 FAIL 이 정상이다."""
    out = layer_verdict(
        _pred(80, 40, signal=True), top_n=15, n_trials=100, cost_one_way=0.0
    )
    assert out["placebo"]["verdict"] == "FAIL"
    assert out["checks"]["플라시보_FAIL"] is True


def test_a_layer_that_only_wins_overall_still_fails():
    """전체 널만 이기고 폴드가 갈리면 FAIL. 계획 1 이 걸린 바로 그 함정이다."""
    rng = np.random.default_rng(0)
    frames = []
    for fold in range(4):
        block = _pred(20, 40, signal=(fold == 3), seed=fold)
        block["date"] = block["date"] + pd.Timedelta(days=20 * fold)
        block["fold"] = fold
        if fold != 3:
            block["prediction"] = rng.normal(size=len(block))
        frames.append(block)
    pred = pd.concat(frames, ignore_index=True)
    out = layer_verdict(pred, top_n=15, n_trials=100, cost_one_way=0.0)
    assert out["checks"]["모든_폴드_널_대비"] is False
    assert out["overall_verdict"] == "FAIL"


def test_report_carries_every_section():
    out = layer_verdict(_pred(60, 40, signal=False), top_n=15, n_trials=50)
    for key in ["gate", "folds", "untouched", "placebo", "downside",
                "selectivity", "checks", "overall_verdict"]:
        assert key in out
    assert len(out["folds"]) == 4
    assert set(out["checks"]) == set(CHECK_NAMES)
