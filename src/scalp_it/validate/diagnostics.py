"""전체 평균이 숨기는 것을 드러내는 진단.

전체 표본 평균만 보고 PASS 를 선언하는 것이 이전 두 번의 실패에서 반복된
실수다. 여기서 요구하는 것은 셋이다.

1. **폴드별 널 대조** — 전체 평균이 아니라 각 폴드가 따로 널을 이기는가.
   프랍 스윙 탐색의 판별기가 정확히 이것이었다.
2. **손 안 댄 최종창** — 가장 최근 구간에서도 살아있는가 (스펙 §10-3).
   레짐이 바뀌어 죽은 시그널은 여기서만 드러난다.
3. **플라시보** — 라벨을 날짜 안에서 섞으면 게이트가 반드시 FAIL 해야 한다.
   PASS 가 나오면 파이프라인 어딘가에 누수가 있다는 뜻이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.baseline import portfolio_returns
from .gate import gate_report, random_null


def fold_null_table(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """폴드마다 전략 성과와 그 폴드 자체의 랜덤 널을 대조한다."""
    rows = []
    for fold, group in pred.groupby("fold"):
        strategy = portfolio_returns(group, top_n=top_n)["mean_excess"].mean()
        null = random_null(group, top_n=top_n, n_trials=n_trials, seed=seed)
        percentile = float((null < strategy).mean() * 100.0)
        rows.append(
            {
                "fold": int(fold),
                "n_dates": int(group["date"].nunique()),
                "strategy_mean": float(strategy),
                "null_mean": float(null.mean()),
                "percentile": percentile,
                "beats_null": bool(percentile > 95.0),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def untouched_window(
    pred: pd.DataFrame,
    *,
    fraction: float = 0.25,
    top_n: int = 15,
    n_trials: int = 200,
    cost_one_way: float = 0.0023,
    seed: int = 0,
) -> dict:
    """가장 최근 ``fraction`` 구간만 잘라 따로 판정한다 (스펙 §10-3)."""
    dates = np.sort(pred["date"].unique())
    n_keep = max(1, round(len(dates) * fraction))
    window = dates[-n_keep:]
    tail = pred[pred["date"].isin(window)]
    report = gate_report(
        tail, top_n=top_n, n_trials=n_trials, cost_one_way=cost_one_way, seed=seed
    )
    report["n_dates"] = len(window)
    report["start"] = str(pd.Timestamp(window[0]).date())
    report["end"] = str(pd.Timestamp(window[-1]).date())
    return report


def placebo_shuffle(pred: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """날짜 안에서 라벨을 섞는다. 예측과 라벨의 대응만 끊고 분포는 보존한다."""
    rng = np.random.default_rng(seed)
    out = pred.copy()
    for date in out["date"].unique():
        mask = out["date"] == date
        values = out.loc[mask, "excess_return"].to_numpy()
        out.loc[mask, "excess_return"] = rng.permutation(values)
    return out
