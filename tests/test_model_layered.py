import numpy as np
import pandas as pd
import pytest

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.model.baseline import walk_forward_predict
from swing_it.model.layered import (
    fit_final_model,
    fold_dates,
    walk_forward_predict_features,
)

EXTRA = ["article_count", "llm_specificity"]


def _dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in [*FEATURE_COLUMNS, *EXTRA]})
    df["code"] = [f"{i % 40:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    df["excess_return"] = df[FEATURE_COLUMNS[0]] * 0.05 + rng.normal(scale=0.01, size=n)
    return df[["code", "date", *FEATURE_COLUMNS, *EXTRA, "excess_return"]]


def test_matches_plan1_walk_forward_exactly():
    """계획 1 과의 호환 계약. 이게 깨지면 층 비교의 ①층이 베이스라인이 아니다."""
    ds = _dataset(1200)
    expected = walk_forward_predict(ds, n_folds=4, min_train=200, seed=0)
    got = walk_forward_predict_features(
        ds, FEATURE_COLUMNS, n_folds=4, min_train=200, seed=0
    )
    pd.testing.assert_frame_equal(expected, got)


def test_extra_features_are_actually_used():
    ds = _dataset(1200)
    ds["planted"] = ds["excess_return"] * 3.0  # 라벨을 거의 그대로 담은 피처
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, "planted"])
    assert b["prediction"].corr(b["excess_return"]) > a["prediction"].corr(a["excess_return"])


def test_fold_dates_are_contiguous_and_ordered():
    folds = fold_dates(_dataset(1200), n_folds=4, min_train=200)
    assert len(folds) == 4
    for i in range(1, len(folds)):
        assert folds[i][0] > folds[i - 1][-1]


def test_fold_dates_start_after_the_min_train_row():
    ds = _dataset(1200).sort_values(["date", "code"]).reset_index(drop=True)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    assert folds[0][0] > ds.loc[200, "date"]


def test_shared_folds_give_identical_fold_boundaries_across_feature_sets():
    ds = _dataset(1200)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS, folds=folds)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, *EXTRA], folds=folds)
    pd.testing.assert_frame_equal(
        a[["code", "date", "excess_return", "fold"]],
        b[["code", "date", "excess_return", "fold"]],
    )


def test_never_trains_on_the_dates_it_predicts():
    ds = _dataset(1200)
    pred = walk_forward_predict_features(ds, FEATURE_COLUMNS)
    bounds = pred.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    for i in range(1, len(bounds)):
        assert bounds.iloc[i]["min"] > bounds.iloc[i - 1]["max"]


def test_too_small_a_sample_fails_loudly():
    with pytest.raises(ValueError, match="min_train"):
        fold_dates(_dataset(50), n_folds=4, min_train=200)


def test_fit_final_model_returns_model_and_median_baseline():
    ds = _dataset(600)
    model, baseline = fit_final_model(ds, [*FEATURE_COLUMNS, *EXTRA])
    assert list(baseline.index) == [*FEATURE_COLUMNS, *EXTRA]
    assert np.isclose(baseline["article_count"], ds["article_count"].median())
    assert model.predict(ds[[*FEATURE_COLUMNS, *EXTRA]].head(3)).shape == (3,)


def test_model_stays_shallow():
    """스펙 §6 — 층이 늘어도 모델은 얕게 고정한다."""
    model, _ = fit_final_model(_dataset(600), FEATURE_COLUMNS)
    assert model.max_depth == 2
    assert model.n_estimators == 100
