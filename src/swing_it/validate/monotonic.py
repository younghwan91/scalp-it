"""예측값 십분위 단조성 — "확실한 것만"이 성립하는지 (스펙 §6).

층이 랜덤 널을 이겨도 **예측값이 클수록 실제 수익이 좋지 않으면** 상위만 골라
들어가는 전략은 성립하지 않는다. 게이트와 별개로 반드시 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decile_table(pred: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """예측값 분위별 실제 초과수익. ``decile`` 0 이 예측 최저다."""
    frame = pred.dropna(subset=["prediction", "excess_return"]).copy()
    frame["decile"] = pd.qcut(
        frame["prediction"].rank(method="first"), n_bins, labels=False
    )
    table = (
        frame.groupby("decile")
        .agg(n=("excess_return", "size"),
             mean_excess=("excess_return", "mean"),
             median_excess=("excess_return", "median"))
        .reset_index()
    )
    return table.sort_values("decile").reset_index(drop=True)


def monotonicity(pred: pd.DataFrame, *, n_bins: int = 10) -> dict:
    """분위 평균이 예측 순서를 따라 오르는가.

    ``is_monotone`` 은 스피어만 상관 0.6 이상 **그리고** 최상위−최하위 > 0 이다.
    완전한 단조를 요구하지 않는 이유는 분위 평균에 잡음이 있기 때문이고,
    0.6 은 사전등록 값이다 — 결과를 보고 낮추지 않는다.
    """
    table = decile_table(pred, n_bins=n_bins)
    if len(table) < 3:
        return {"spearman": float("nan"), "top_minus_bottom": float("nan"),
                "top_vs_median": float("nan"), "is_monotone": False}

    ranks = table["decile"].to_numpy(dtype=float)
    means = table["mean_excess"].to_numpy(dtype=float)
    spearman = float(pd.Series(ranks).corr(pd.Series(means), method="spearman"))
    top_minus_bottom = float(means[-1] - means[0])
    top_vs_median = float(means[-1] - np.median(means))
    return {
        "spearman": spearman,
        "top_minus_bottom": top_minus_bottom,
        "top_vs_median": top_vs_median,
        "is_monotone": bool(spearman >= 0.6 and top_minus_bottom > 0),
    }
