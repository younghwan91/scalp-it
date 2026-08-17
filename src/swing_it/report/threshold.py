"""절대 임계 — "확실한 것만" (스펙 §6).

날짜당 상위 N 을 억지로 뽑으면 재료가 시원찮은 날에도 후보가 나온다.
임계 미달이면 그날은 0건이어야 한다.

**임계값은 구조에서 정한다, 성과에서가 아니다.** 목표 동시 보유를 정하고
거기서 하루 통과 건수를 역산한 뒤, 학습 구간 예측값의 해당 분위수를 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency_threshold(
    pred: pd.DataFrame,
    *,
    hold_days: int = 40,
    target_concurrent: int = 20,
) -> dict:
    """목표 동시 보유를 만족하는 예측값 임계를 학습 구간에서 구한다.

    Args:
        hold_days: 보유 거래일. 동시 보유 ≈ 하루 통과 건수 × hold_days.
        target_concurrent: 목표 동시 보유 종목 수. 스펙 §6 의 15~25 중앙값이 20.
    """
    frame = pred.dropna(subset=["prediction"])
    n_days = int(frame["date"].nunique())
    if n_days == 0 or frame.empty:
        return {"threshold": float("nan"), "expected_per_day": 0.0,
                "expected_concurrent": 0.0, "n_train_days": 0}

    per_day = target_concurrent / float(hold_days)
    n_pass = max(1, round(per_day * n_days))
    quantile = 1.0 - min(1.0, n_pass / len(frame))
    threshold = float(np.quantile(frame["prediction"].to_numpy(), quantile))

    passing = int((frame["prediction"] >= threshold).sum())
    return {
        "threshold": threshold,
        "expected_per_day": passing / n_days,
        "expected_concurrent": passing / n_days * hold_days,
        "n_train_days": n_days,
    }
