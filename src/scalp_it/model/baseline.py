"""walk-forward 베이스라인 — 수급 피처만으로 초과수익을 예측한다 (스펙 §3 ①층)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ..dataset import FEATURE_COLUMNS


def walk_forward_predict(
    ds: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """시간순 폴드로 나눠 각 폴드를 그 이전 데이터만으로 학습·예측한다.

    폴드 경계는 날짜 단위로 끊는다. 같은 날짜가 학습과 예측에 동시에 들어가면
    같은 날 다른 종목의 결과가 새어 들어온다.
    """
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    if len(df) <= min_train:
        raise ValueError(f"표본이 {len(df)} 행뿐이라 min_train={min_train} 을 못 채운다")

    # min_train 행을 채우는 최초 날짜 이후를 n_folds 로 등분한다
    first_test_date = df.loc[min_train, "date"]
    test_dates = dates[dates > first_test_date]
    if len(test_dates) < n_folds:
        raise ValueError(f"검증 가능한 날짜가 {len(test_dates)} 개뿐이라 {n_folds} 폴드를 못 만든다")

    chunks = np.array_split(test_dates, n_folds)
    results = []
    for fold, chunk in enumerate(chunks):
        fold_start = chunk[0]
        train = df[df["date"] < fold_start]
        test = df[df["date"].isin(chunk)]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=seed)
        model.fit(train[FEATURE_COLUMNS], train["excess_return"])
        block = test[["code", "date", "excess_return"]].copy()
        block["prediction"] = model.predict(test[FEATURE_COLUMNS])
        block["fold"] = fold
        results.append(block)

    return pd.concat(results, ignore_index=True)


def portfolio_returns(pred: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    """날짜별 예측 상위 ``top_n`` 종목의 평균 초과수익.

    ``top_n`` 하한이 15 인 이유는 스펙 §2 — 4~8 종목 집중은 이미 죽은 영역이다.
    """
    rows = []
    for date, group in pred.groupby("date"):
        picked = group.nlargest(top_n, "prediction")
        rows.append(
            {
                "date": date,
                "n_positions": len(picked),
                "mean_excess": picked["excess_return"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
