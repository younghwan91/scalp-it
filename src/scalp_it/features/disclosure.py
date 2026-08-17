"""②층 공시 피처 — 무료·규칙 기반, LLM 불필요.

거래일 이월은 계획 2 의 ``news.calendar.align_to_trading_days`` 를 재사용한다.
휴장일 접수분이 사라지는 버그를 뉴스에서 이미 겪었고, 같은 문제를 두 번 풀지 않는다.
"""

from __future__ import annotations

import pandas as pd

from ..news.calendar import align_to_trading_days

_TYPE_FLAGS: dict[str, str] = {
    "is_insider": "내부자",
    "is_contract": "공급계약",
    "is_capex": "시설투자",
    "is_payout": "주주환원",
    "is_funding": "자본조달",
    "is_earnings": "실적",
}

DISC_FEATURE_COLUMNS: list[str] = [
    "disc_count",
    "disc_ratio",
    "disc_correction_ratio",
    "days_since_disc",
    *_TYPE_FLAGS,
]

_OUT_COLUMNS = ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def disclosure_features(
    disc: pd.DataFrame,
    calendar: pd.Series,
    *,
    baseline_window: int = 60,
    min_periods: int = 20,
    baseline_floor: float = 0.2,
) -> pd.DataFrame:
    """``(code, date)`` 별 공시 건수·급증 배수·유형 비율·신선도."""
    if disc.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    frame = disc.rename(columns={"rcept_dt": "date"}).copy()
    frame = align_to_trading_days(frame, calendar)
    if frame.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)
    frame["is_correction"] = frame["is_correction"].astype(bool)

    aggs = {
        "disc_count": ("rcept_no", "size"),
        "n_correction": ("is_correction", "sum"),
        "top_material_type": ("material_type", lambda s: s.value_counts().index[0]),
    }
    daily = frame.groupby(["code", "date"], sort=False).agg(**aggs).reset_index()
    for flag, label in _TYPE_FLAGS.items():
        hits = (
            frame.assign(_hit=(frame["material_type"] == label).astype(float))
            .groupby(["code", "date"], sort=False)["_hit"].sum()
            .reset_index(name=flag)
        )
        daily = daily.merge(hits, on=["code", "date"], how="left")

    # 전 거래일 격자로 펼쳐야 '평소'가 과대평가되지 않는다.
    trading = pd.Index(pd.to_datetime(pd.Series(calendar)).unique()).sort_values()
    codes = pd.Index(sorted(daily["code"].unique()), name="code")
    grid = pd.MultiIndex.from_product([codes, trading], names=["code", "date"]).to_frame(
        index=False
    )
    dense = grid.merge(daily[["code", "date", "disc_count"]], on=["code", "date"], how="left")
    dense["disc_count"] = dense["disc_count"].fillna(0.0)
    dense = dense.sort_values(["code", "date"])
    grouped = dense.groupby("code", sort=False)

    dense["baseline"] = grouped["disc_count"].transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=min_periods).mean()
    )
    dense["baseline"] = dense["baseline"].fillna(baseline_floor).clip(lower=baseline_floor)

    dense["_ordinal"] = grouped.cumcount()
    dense["_last"] = dense["_ordinal"].where(dense["disc_count"] > 0)
    dense["_prev"] = grouped["_last"].transform(lambda s: s.shift(1).ffill())
    dense["days_since_disc"] = (dense["_ordinal"] - dense["_prev"]).fillna(baseline_window)

    out = daily.merge(
        dense[["code", "date", "baseline", "days_since_disc"]], on=["code", "date"], how="left"
    )
    count = out["disc_count"].astype(float)
    out["disc_ratio"] = count / out["baseline"]
    out["disc_correction_ratio"] = out["n_correction"] / count
    for flag in _TYPE_FLAGS:
        out[flag] = out[flag].fillna(0.0) / count
    out["days_since_disc"] = out["days_since_disc"].fillna(baseline_window)

    return out[_OUT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
