"""규칙 기반 종가베팅 백테스트.

규칙은 컬럼별 ``(min, max)`` 구간 딕셔너리다. 탐색·최적화는 여기 없다 —
엔진은 주어진 규칙을 정확히 실행하고 숫자를 돌려줄 뿐이다.

비용이 결정적이다. 하루 보유라 총수익 자체가 작아서 왕복 비용 몇 bp 가
부호를 뒤집는다. 그래서 ``net_mean`` 은 물론 ``cost_grid`` 로 여러 비용
가정을 한 번에 낸다.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

#: 다음날 시가가 이 수익률을 넘으면 사실상 상한가 갭이라 체결이 의심스럽다.
LIMIT_UP_THRESHOLD = 0.29

#: 리포트 기본 비용 시나리오 (왕복).
DEFAULT_COST_GRID = (0.0, 0.0015, 0.0023, 0.0035)

_EXIT_COLUMNS = ("ret_open", "ret_high", "ret_close", "gap_next")


def apply_rules(df: pd.DataFrame, rules: dict[str, tuple[float | None, float | None]]) -> pd.DataFrame:
    """``{컬럼: (min, max)}`` 구간 조건을 모두 만족하는 행만 남긴다.

    경계는 포함이다 (``min <= x <= max``). ``None`` 은 무제한.
    조건 컬럼이 NaN 인 행은 조건을 만족하지 않은 것으로 본다.
    """
    mask = pd.Series(True, index=df.index)
    for col, bounds in rules.items():
        if col not in df.columns:
            raise KeyError(f"규칙 컬럼이 프레임에 없다: {col}")
        lo, hi = bounds
        values = pd.to_numeric(df[col], errors="coerce")
        ok = values.notna()
        if lo is not None:
            ok &= values >= lo
        if hi is not None:
            ok &= values <= hi
        mask &= ok
    return df[mask]


def _payoff(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if wins.empty or losses.empty:
        return float("nan")
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return float("nan")
    return float(wins.mean() / avg_loss)


def _empty_result(cost_round_trip: float, exit_col: str, n_excluded: int) -> dict[str, Any]:
    nan = float("nan")
    return {
        "exit_col": exit_col,
        "cost_round_trip": cost_round_trip,
        "n_trades": 0,
        "n_days": 0,
        "trades_per_day": 0.0,
        "mean": nan,
        "median": nan,
        "win_rate": nan,
        "payoff": nan,
        "p5": nan,
        "p25": nan,
        "p75": nan,
        "p95": nan,
        "net_mean": nan,
        "by_year": {},
        "cost_grid": {},
        "n_excluded_no_next_day": n_excluded,
        "n_limit_up": 0,
        "limit_up_share": nan,
        "mean_ex_limit_up": nan,
    }


def select_close_bet(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    rules: dict[str, tuple[float | None, float | None]],
    top_n: int | None = None,
    exit_col: str = "ret_open",
) -> tuple[pd.DataFrame, int]:
    """규칙·top_n 을 적용해 **선택된 거래 행 자체**를 돌려준다.

    :func:`run_close_bet` 의 선택 단계만 떼어낸 것이다 — 요약 통계가 아니라
    "어떤 (종목, 일자) 가 뽑혔는가" 가 필요한 진단(상한가 도달 시각 등)이 쓴다.
    선택 논리는 한 곳에만 있어야 백테스트와 진단이 같은 종목을 본다.

    Returns:
        ``(selected, n_excluded)``. ``selected`` 는 피처·라벨이 합쳐진 행이고
        ``n_excluded`` 는 다음날 봉이 없어 버린 행 수다.
    """
    if exit_col not in _EXIT_COLUMNS:
        raise ValueError(f"exit_col 은 {_EXIT_COLUMNS} 중 하나여야 한다: {exit_col}")

    merged = features.merge(labels, on=["code", "date"], how="inner")
    selected = apply_rules(merged, rules)

    # 거래정지·상장폐지로 다음날 봉이 없는 행. 몇 건인지 반드시 보고한다.
    exit_values = pd.to_numeric(selected[exit_col], errors="coerce")
    n_excluded = int(exit_values.isna().sum())
    selected = selected[exit_values.notna()]

    if top_n is not None and not selected.empty:
        if "tv_ratio" not in selected.columns:
            raise KeyError("top_n 을 쓰려면 tv_ratio 컬럼이 있어야 한다")
        selected = (
            selected.sort_values(["date", "tv_ratio"], ascending=[True, False])
            .groupby("date", sort=False)
            .head(top_n)
        )
    return selected, n_excluded


def run_close_bet(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    rules: dict[str, tuple[float | None, float | None]],
    top_n: int | None = None,
    cost_round_trip: float = 0.0023,
    exit_col: str = "ret_open",
    cost_grid: tuple[float, ...] = DEFAULT_COST_GRID,
) -> dict[str, Any]:
    """규칙에 맞는 종가베팅 거래를 뽑아 성과를 낸다.

    Args:
        features: :func:`~scalp_it.closebet.features.close_bet_features` 출력.
        labels: :func:`~scalp_it.closebet.labels.close_bet_labels` 출력.
        rules: ``{컬럼: (min, max)}``. 피처·라벨 어느 쪽 컬럼이든 쓸 수 있다.
        top_n: 주어지면 **날짜별로** ``tv_ratio`` 상위 N 종목만 남긴다.
        cost_round_trip: 왕복 비용. ``net_mean = mean - cost``.
        exit_col: 청산 방식 컬럼.
        cost_grid: ``cost_grid`` 결과에 넣을 비용 시나리오들.

    Returns:
        요약 통계 dict. 키는 모듈 docstring 과 스펙 참고.
    """
    selected, n_excluded = select_close_bet(
        features, labels, rules=rules, top_n=top_n, exit_col=exit_col
    )

    if selected.empty:
        return _empty_result(cost_round_trip, exit_col, n_excluded)

    returns = pd.to_numeric(selected[exit_col], errors="coerce")
    dates = pd.to_datetime(selected["date"])
    n_trades = len(returns)
    n_days = int(dates.nunique())
    mean = float(returns.mean())

    limit_up = returns > LIMIT_UP_THRESHOLD
    n_limit_up = int(limit_up.sum())
    ex_limit = returns[~limit_up]

    by_year: dict[str, dict[str, float]] = {}
    for year, grp in returns.groupby(dates.dt.year):
        by_year[str(int(year))] = {"mean": float(grp.mean()), "n": len(grp)}

    return {
        "exit_col": exit_col,
        "cost_round_trip": cost_round_trip,
        "n_trades": n_trades,
        "n_days": n_days,
        "trades_per_day": float(n_trades / n_days) if n_days else 0.0,
        "mean": mean,
        "median": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "payoff": _payoff(returns),
        "p5": float(returns.quantile(0.05)),
        "p25": float(returns.quantile(0.25)),
        "p75": float(returns.quantile(0.75)),
        "p95": float(returns.quantile(0.95)),
        "net_mean": mean - cost_round_trip,
        "by_year": by_year,
        "cost_grid": {f"{c:.4f}": mean - c for c in cost_grid},
        "n_excluded_no_next_day": n_excluded,
        "n_limit_up": n_limit_up,
        "limit_up_share": float(n_limit_up / n_trades),
        "mean_ex_limit_up": float(ex_limit.mean()) if not ex_limit.empty else math.nan,
    }
