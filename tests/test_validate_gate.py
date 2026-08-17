import numpy as np
import pandas as pd

from scalp_it.validate.gate import gate_report, random_null


def _pred(n_dates: int, n_per_date: int, signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n_per_date):
            excess = rng.normal(scale=0.05)
            prediction = excess if signal else rng.normal()
            rows.append(
                {
                    "code": f"{i:06d}",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                    "excess_return": excess,
                    "prediction": prediction,
                    "fold": 0,
                }
            )
    return pd.DataFrame(rows)


def test_random_null_has_requested_length():
    null = random_null(_pred(20, 30, signal=False), top_n=15, n_trials=50)
    assert null.shape == (50,)


def test_random_null_centres_near_zero():
    # 초과수익이 날짜별로 디민돼 있지 않아도, 무작위 선택의 기대값은 전체 평균 근처다
    pred = _pred(30, 40, signal=False)
    null = random_null(pred, top_n=15, n_trials=200)
    assert abs(null.mean() - pred["excess_return"].mean()) < 0.01


def test_gate_fails_when_prediction_is_noise():
    report = gate_report(_pred(30, 40, signal=False), top_n=15, n_trials=200)
    assert report["verdict"] == "FAIL"


def test_gate_passes_when_prediction_is_perfect_and_beats_cost():
    # 예측 = 실제 초과수익이므로 상위 15개는 확실히 널을 이긴다
    report = gate_report(_pred(30, 40, signal=True), top_n=15, n_trials=200, cost_one_way=0.0)
    assert report["verdict"] == "PASS"
    assert report["percentile"] > 95


def test_gate_report_has_all_keys():
    report = gate_report(_pred(20, 30, signal=False), top_n=15, n_trials=50)
    for key in ["strategy_mean", "null_mean", "null_p95", "percentile", "net_mean", "verdict"]:
        assert key in report
