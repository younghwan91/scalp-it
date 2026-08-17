"""절대 임계 — "확실한 것만" (스펙 §6).

날짜당 상위 N 을 억지로 뽑으면 재료가 시원찮은 날에도 후보가 나온다.
임계 미달이면 그날은 0건이어야 한다.

**임계값은 구조에서 정한다, 성과에서가 아니다.** 목표 동시 보유를 정하고
거기서 하루 통과 건수를 역산한 뒤, 해당 분위수를 쓴다.

**임계는 폴드마다 따로 잡는다.** walk-forward 는 폴드마다 다른 학습 데이터로
적합하므로 예측값의 절대 수준이 폴드마다 다르다. 전역 임계 하나를 쓰면 예측
스케일이 높은 폴드가 통과분을 독식하고 나머지 폴드는 0건이 된다.

실측(2024+ ①층)에서 전역 임계 상위 286건의 실제 초과수익이 −1.77% 였다.
같은 데이터에서 날짜당 상위 2 는 +2.91% 였다. 차이는 신호가 아니라 스케일 혼합이었다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FOLD_COLUMN = "fold"


def _one_threshold(frame: pd.DataFrame, per_day: float) -> dict:
    n_days = int(frame["date"].nunique())
    if n_days == 0 or frame.empty:
        return {"threshold": float("nan"), "n_pass": 0, "n_days": 0, "n_rows": 0}
    n_target = max(1, round(per_day * n_days))
    quantile = 1.0 - min(1.0, n_target / len(frame))
    threshold = float(np.quantile(frame["prediction"].to_numpy(), quantile))
    return {
        "threshold": threshold,
        "n_pass": int((frame["prediction"] >= threshold).sum()),
        "n_days": n_days,
        "n_rows": len(frame),
    }


def concurrency_threshold(
    pred: pd.DataFrame,
    *,
    hold_days: int = 40,
    target_concurrent: int = 20,
) -> dict:
    """목표 동시 보유를 만족하는 예측값 임계를 폴드마다 구한다.

    Args:
        hold_days: 보유 거래일. 동시 보유 ≈ 하루 통과 건수 × hold_days.
        target_concurrent: 목표 동시 보유 종목 수. 스펙 §6 의 15~25 중앙값이 20.

    Returns:
        ``per_fold`` 에 폴드별 임계와 통과 건수, 최상위에 전체 집계.
        ``threshold`` 는 폴드가 하나일 때만 의미가 있으며, 여러 폴드면
        폴드별 값의 중앙값을 참고용으로 담는다 — **운영에서는 최신 폴드의
        임계를 쓴다** (그것이 가장 최근 학습 구간에서 나온 값이다).
    """
    frame = pred.dropna(subset=["prediction"]).copy()
    if frame.empty:
        return {"threshold": float("nan"), "expected_per_day": 0.0,
                "expected_concurrent": 0.0, "n_train_days": 0, "per_fold": []}

    per_day = target_concurrent / float(hold_days)
    if _FOLD_COLUMN not in frame.columns:
        frame[_FOLD_COLUMN] = 0

    per_fold = []
    for fold, group in frame.groupby(_FOLD_COLUMN, sort=True):
        entry = _one_threshold(group, per_day)
        entry["fold"] = int(fold)
        per_fold.append(entry)

    total_pass = sum(f["n_pass"] for f in per_fold)
    total_days = sum(f["n_days"] for f in per_fold)
    thresholds = [f["threshold"] for f in per_fold if not np.isnan(f["threshold"])]
    return {
        "threshold": float(np.median(thresholds)) if thresholds else float("nan"),
        "latest_threshold": per_fold[-1]["threshold"] if per_fold else float("nan"),
        "expected_per_day": total_pass / total_days if total_days else 0.0,
        "expected_concurrent": (total_pass / total_days * hold_days) if total_days else 0.0,
        "n_train_days": total_days,
        "per_fold": per_fold,
    }
