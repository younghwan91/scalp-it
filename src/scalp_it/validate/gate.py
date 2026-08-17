"""음성대조 게이트 — 랜덤 진입 널을 이기지 못하면 아무것도 승격하지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.baseline import portfolio_returns


def random_null(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """날짜마다 ``top_n`` 종목을 무작위로 골랐을 때의 평균 초과수익 분포."""
    rng = np.random.default_rng(seed)
    by_date = [group["excess_return"].to_numpy() for _, group in pred.groupby("date")]
    out = np.empty(n_trials)
    for trial in range(n_trials):
        daily = []
        for values in by_date:
            k = min(top_n, len(values))
            picked = rng.choice(values, size=k, replace=False)
            daily.append(picked.mean())
        out[trial] = float(np.mean(daily))
    return out


def gate_report(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    cost_one_way: float = 0.0023,
    seed: int = 0,
) -> dict:
    """전략 성과를 랜덤 널 및 비용과 대조해 판정한다."""
    strategy = portfolio_returns(pred, top_n=top_n)["mean_excess"].mean()
    null = random_null(pred, top_n=top_n, n_trials=n_trials, seed=seed)
    percentile = float((null < strategy).mean() * 100.0)
    net = float(strategy - cost_one_way * 2)
    passed = percentile > 95.0 and net > 0.0
    return {
        "strategy_mean": float(strategy),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "percentile": percentile,
        "net_mean": net,
        "verdict": "PASS" if passed else "FAIL",
    }
