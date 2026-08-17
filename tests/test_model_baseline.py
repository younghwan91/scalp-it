import numpy as np
import pandas as pd

from scalp_it.dataset import FEATURE_COLUMNS
from scalp_it.model.baseline import portfolio_returns, walk_forward_predict


def _dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    )
    df["code"] = [f"{i % 40:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    # 첫 피처가 라벨을 결정하는 학습 가능한 구조를 넣는다
    df["excess_return"] = df[FEATURE_COLUMNS[0]] * 0.05 + rng.normal(scale=0.01, size=n)
    return df[["code", "date", *FEATURE_COLUMNS, "excess_return"]]


def test_walk_forward_returns_expected_columns():
    pred = walk_forward_predict(_dataset(600), n_folds=3, min_train=100)
    assert list(pred.columns) == ["code", "date", "excess_return", "prediction", "fold"]


def test_walk_forward_never_predicts_on_training_dates():
    ds = _dataset(600)
    pred = walk_forward_predict(ds, n_folds=3, min_train=100)
    # 각 폴드의 최소 예측일은 이전 폴드의 최대 예측일보다 커야 한다
    bounds = pred.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    for i in range(1, len(bounds)):
        assert bounds.iloc[i]["min"] > bounds.iloc[i - 1]["max"]


def test_walk_forward_learns_a_real_signal():
    pred = walk_forward_predict(_dataset(1200), n_folds=4, min_train=200)
    corr = pred["prediction"].corr(pred["excess_return"])
    assert corr > 0.2  # 구조를 심어둔 데이터이므로 양의 상관이 나와야 한다


def test_portfolio_returns_picks_top_n_per_date():
    pred = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "excess_return": [0.1, 0.2, 0.3, 0.4],
            "prediction": [0.4, 0.3, 0.2, 0.1],
            "fold": [0] * 4,
        }
    )
    out = portfolio_returns(pred, top_n=2)
    assert len(out) == 1
    assert out.iloc[0]["n_positions"] == 2
    # 예측 상위 2개는 A(0.1) 와 B(0.2) → 평균 0.15
    assert np.isclose(out.iloc[0]["mean_excess"], 0.15)
