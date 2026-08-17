"""예측값 십분위 단조성 — "확실한 것만"이 성립하는지 (스펙 §6).

층이 랜덤 널을 이겨도 **예측값이 클수록 실제 수익이 좋지 않으면** 상위만 골라
들어가는 전략은 성립하지 않는다. 게이트와 별개로 반드시 본다.

**분위는 폴드 안에서 나눈다.** walk-forward 는 폴드마다 다른 학습 데이터로
적합하므로 예측값의 절대 수준이 폴드마다 다르다. 전역으로 풀링해 분위를 나누면
"예측이 센 행"이 아니라 "예측 스케일이 큰 폴드의 행"이 최상위에 모인다.

실측(2024+ ①층)에서 정확히 그렇게 됐다 — 전역 분위로는 최상위 분위가 −0.78% 로
꺾였는데, 폴드 안에서 보면 4개 폴드 전부 최상위가 분위 7보다 좋았다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FOLD_COLUMN = "fold"


def _bin_within(frame: pd.DataFrame, n_bins: int) -> pd.Series:
    ranks = frame["prediction"].rank(method="first")
    if ranks.nunique() < n_bins:
        return pd.Series(np.full(len(frame), np.nan), index=frame.index)
    return pd.qcut(ranks, n_bins, labels=False)


def decile_table(pred: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """예측값 분위별 실제 초과수익. ``decile`` 0 이 예측 최저다.

    ``fold`` 컬럼이 있으면 **폴드 안에서** 분위를 나눈 뒤 합친다.
    """
    frame = pred.dropna(subset=["prediction", "excess_return"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_excess", "median_excess"])

    if _FOLD_COLUMN in frame.columns and frame[_FOLD_COLUMN].nunique() > 1:
        frame["decile"] = (
            frame.groupby(_FOLD_COLUMN, group_keys=False)
            .apply(lambda g: _bin_within(g, n_bins), include_groups=False)
        )
    else:
        frame["decile"] = _bin_within(frame, n_bins)

    frame = frame.dropna(subset=["decile"])
    if frame.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_excess", "median_excess"])

    table = (
        frame.groupby("decile")
        .agg(n=("excess_return", "size"),
             mean_excess=("excess_return", "mean"),
             median_excess=("excess_return", "median"))
        .reset_index()
    )
    table["decile"] = table["decile"].astype(int)
    return table.sort_values("decile").reset_index(drop=True)


def monotonicity(pred: pd.DataFrame, *, n_bins: int = 10) -> dict:
    """분위 평균이 예측 순서를 따라 오르는가.

    ``is_monotone`` 은 세 가지를 모두 요구한다:

    1. 스피어만 상관 ≥ 0.6 — 전반적으로 우상향인가
    2. 최상위 − 최하위 > 0 — 끝점이 뒤집히지 않았는가
    3. **최상위 분위가 최고인가** (``top_decile_is_best``)

    3번이 필요한 이유: 스피어만이 0.8 을 넘으면서도 꼭대기만 꺾일 수 있다.
    그러면 "예측이 가장 높은 것을 사라"가 성립하지 않는다 — 정확히 우리가
    하려는 일이 안 된다는 뜻이라, 전체 추세보다 이쪽이 결정적이다.

    임계 0.6 은 사전등록 값이다 — 결과를 보고 낮추지 않는다.
    """
    table = decile_table(pred, n_bins=n_bins)
    if len(table) < 3:
        return {"spearman": float("nan"), "top_minus_bottom": float("nan"),
                "top_vs_median": float("nan"), "top_decile_is_best": False,
                "is_monotone": False}

    ranks = table["decile"].to_numpy(dtype=float)
    means = table["mean_excess"].to_numpy(dtype=float)
    spearman = float(pd.Series(ranks).corr(pd.Series(means), method="spearman"))
    top_minus_bottom = float(means[-1] - means[0])
    top_vs_median = float(means[-1] - np.median(means))
    top_is_best = bool(means[-1] >= means.max() - 1e-12)
    return {
        "spearman": spearman,
        "top_minus_bottom": top_minus_bottom,
        "top_vs_median": top_vs_median,
        "top_decile_is_best": top_is_best,
        "is_monotone": bool(spearman >= 0.6 and top_minus_bottom > 0 and top_is_best),
    }
