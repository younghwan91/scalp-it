"""PEAD 대비 벤치마크 (스펙 §7).

PEAD 는 이 레포의 중심이 아니라 **넘어야 할 선**이다. 이미 검증된 알파가
있는데 뉴스 모델이 그보다 못하다면, 뉴스 모델을 운영에 올릴 이유가 없다.

기준값은 kr-quant PEAD 검증 결과다: 전 구간 Sharpe ~0.8, t ~2.2
(저회전, point-in-time 유동성 하한 적용본).

겹침 보정: 이벤트는 매일 나고 보유는 40거래일이라 날짜별 관측이 독립이 아니다.
보정 없는 t 는 크게 부풀려진다. 여기서는 겹치지 않는 블록 수
(``n_obs // hold_days``) 를 유효 표본으로 쓴다. 보수적인 쪽으로 틀리는 선택이다.

이 모듈은 ``math`` 만 쓴다. ``numpy`` 를 임포트하지 마라 — ruff 가 F401 로 잡는다.
"""

from __future__ import annotations

import math

import pandas as pd

PEAD_SHARPE: float = 0.80
PEAD_T: float = 2.20
COMPARABLE_RATIO: float = 0.75

TRADING_DAYS = 252


def overlap_stats(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """날짜별 포트폴리오 초과수익 시계열의 겹침 보정 통계.

    Args:
        curve: ``portfolio_returns`` 결과. ``mean_excess`` 는 40거래일 보유 수익률.
        hold_days: 보유 거래일 수. 겹침 정도를 결정한다.
    """
    values = curve["mean_excess"].dropna().to_numpy(dtype=float)
    n_obs = len(values)
    if n_obs < 2:
        raise ValueError(f"관측이 {n_obs} 개뿐이라 분산을 못 잰다")

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    n_effective = max(1, n_obs // hold_days)
    sharpe = mean / sd * math.sqrt(TRADING_DAYS / hold_days) if sd > 0 else 0.0
    t_stat = mean / (sd / math.sqrt(n_effective)) if sd > 0 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "n_obs": n_obs,
        "n_effective": int(n_effective),
        "sharpe_annual": float(sharpe),
        "t_stat": float(t_stat),
    }


def pead_benchmark(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """검증된 PEAD 대비 판정. 이기든 지든 명시적으로 남긴다 (스펙 §10-5)."""
    stats = overlap_stats(curve, hold_days=hold_days)
    sharpe = stats["sharpe_annual"]
    if sharpe > PEAD_SHARPE and stats["t_stat"] > PEAD_T:
        verdict = "BEATS_PEAD"
    elif sharpe >= PEAD_SHARPE * COMPARABLE_RATIO:
        verdict = "COMPARABLE_TO_PEAD"
    else:
        verdict = "LOSES_TO_PEAD"
    return {**stats, "pead_sharpe": PEAD_SHARPE, "pead_t": PEAD_T, "verdict": verdict}


def benchmark_all_layers(
    curves: dict[str, pd.DataFrame], *, hold_days: int = 40
) -> list[dict]:
    """층별 PEAD 판정 목록."""
    return [
        {"layer": layer, **pead_benchmark(curve, hold_days=hold_days)}
        for layer, curve in curves.items()
    ]
