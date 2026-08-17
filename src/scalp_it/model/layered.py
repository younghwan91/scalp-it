"""피처 목록을 인자로 받는 walk-forward — 층 비교의 실행부.

계획 1 의 ``walk_forward_predict`` 는 ``FEATURE_COLUMNS`` 를 고정으로 쓴다.
그 시그니처를 깨지 않기 위해 여기에 확장판을 따로 둔다. 두 함수가 같은 결과를
내는지는 ``tests/test_model_layered.py::test_matches_plan1_walk_forward_exactly``
가 고정한다.

``fold_dates`` 를 분리한 것이 이 모듈의 요점이다. 폴드 경계를 한 번 계산해
세 층에 그대로 넘겨야 "③이 ②를 이겼다"가 폴드 차이가 아님을 보증할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

MAX_DEPTH = 2
N_ESTIMATORS = 100


def fold_dates(
    ds: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 200,
) -> list[np.ndarray]:
    """검증 구간을 시간순 ``n_folds`` 등분한 날짜 배열 목록.

    계획 1 ``walk_forward_predict`` 와 글자 그대로 같은 규칙이다. 폴드 경계는
    날짜 단위로 끊는다 — 같은 날짜가 학습과 예측에 동시에 들어가면 같은 날
    다른 종목의 결과가 새어 들어온다.
    """
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    if len(df) <= min_train:
        raise ValueError(f"표본이 {len(df)} 행뿐이라 min_train={min_train} 을 못 채운다")

    first_test_date = df.loc[min_train, "date"]
    test_dates = dates[dates > first_test_date]
    if len(test_dates) < n_folds:
        raise ValueError(f"검증 가능한 날짜가 {len(test_dates)} 개뿐이라 {n_folds} 폴드를 못 만든다")
    return list(np.array_split(test_dates, n_folds))


def walk_forward_predict_features(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    folds: list[np.ndarray] | None = None,
    n_folds: int = 4,
    min_train: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """지정한 피처만 써서 시간순 폴드로 학습·예측한다.

    Args:
        folds: ``fold_dates`` 결과. 층 비교에서는 **반드시** 미리 계산해 넘긴다.
            생략하면 이 함수가 직접 계산한다(단일 층 실행용).
    """
    columns = list(feature_columns)
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    if folds is None:
        folds = fold_dates(df, n_folds=n_folds, min_train=min_train)

    results = []
    for fold, chunk in enumerate(folds):
        train = df[df["date"] < chunk[0]]
        test = df[df["date"].isin(chunk)]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(
            max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
        )
        model.fit(train[columns], train["excess_return"])
        block = test[["code", "date", "excess_return"]].copy()
        block["prediction"] = model.predict(test[columns])
        block["fold"] = fold
        results.append(block)

    return pd.concat(results, ignore_index=True)


def fit_final_model(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[GradientBoostingRegressor, pd.Series]:
    """전 구간으로 적합한 모델과 피처 중앙값을 함께 돌려준다.

    중앙값은 근거 카드의 기여도 계산에서 '평범한 이벤트' 기준점으로 쓴다
    (Task 6). 모델과 함께 나와야 기준점이 학습 분포와 어긋나지 않는다.
    """
    columns = list(feature_columns)
    model = GradientBoostingRegressor(
        max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
    )
    model.fit(ds[columns], ds["excess_return"])
    return model, ds[columns].median()
