import numpy as np
import pandas as pd

from scalp_it.validate.diagnostics import fold_null_table, placebo_shuffle, untouched_window


def _pred(n_dates: int, n_per_date: int, signal: bool, n_folds: int = 4,
          seed: int = 0) -> pd.DataFrame:
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
                    "fold": (d * n_folds) // n_dates,
                }
            )
    return pd.DataFrame(rows)


def test_fold_null_table_has_one_row_per_fold():
    pred = _pred(40, 30, signal=True)
    table = fold_null_table(pred, top_n=15, n_trials=50)
    assert list(table.columns) == [
        "fold", "n_dates", "strategy_mean", "null_mean", "percentile", "beats_null",
    ]
    assert len(table) == pred["fold"].nunique()


def test_fold_null_table_marks_signal_folds_as_beating_null():
    table = fold_null_table(_pred(40, 30, signal=True), top_n=15, n_trials=100)
    assert table["beats_null"].all()


def test_fold_null_table_marks_noise_folds_as_not_beating_null():
    table = fold_null_table(_pred(40, 30, signal=False), top_n=15, n_trials=100)
    assert not table["beats_null"].any()


def test_untouched_window_uses_only_the_last_fraction():
    pred = _pred(40, 30, signal=True)
    result = untouched_window(pred, fraction=0.25, top_n=15, n_trials=50)
    assert result["n_dates"] == 10
    assert result["start"] == "2026-01-31"  # 40일 중 마지막 10일의 첫날
    assert result["verdict"] in ("PASS", "FAIL")


def test_untouched_window_passes_on_real_signal():
    result = untouched_window(_pred(40, 30, signal=True), fraction=0.25,
                              top_n=15, n_trials=100, cost_one_way=0.0)
    assert result["verdict"] == "PASS"


def test_placebo_shuffle_destroys_the_signal():
    pred = _pred(40, 30, signal=True)
    shuffled = placebo_shuffle(pred, seed=0)
    # 셔플은 날짜별 라벨 집합을 보존한다
    for date, group in pred.groupby("date"):
        original = np.sort(group["excess_return"].to_numpy())
        after = np.sort(shuffled[shuffled["date"] == date]["excess_return"].to_numpy())
        assert np.allclose(original, after)
    # 예측력은 사라져야 한다
    assert abs(shuffled["prediction"].corr(shuffled["excess_return"])) < 0.3
