"""이벤트 · 피처 · 라벨을 하나의 학습 테이블로 조립한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import extract_events
from .features.supply import supply_features
from .labels import excess_return, forward_return

FEATURE_COLUMNS: list[str] = [
    "tv_ratio",
    "log_adv",
    "foreign_net_ratio",
    "inst_net_ratio",
    "smart_money_net_5d_scaled",
    "indiv_net_5d_scaled",
    "pension_net_5d_scaled",
]

_SCALED = {
    "smart_money_net_5d_scaled": "smart_money_net_5d",
    "indiv_net_5d_scaled": "indiv_net_5d",
    "pension_net_5d_scaled": "pension_net_5d",
}


def build_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """이벤트 행에 피처와 라벨을 붙여 학습 가능한 테이블을 만든다."""
    events = extract_events(
        prices, adv_floor=adv_floor, tv_multiple=tv_multiple
    )
    if events.empty:
        return pd.DataFrame(columns=["code", "date", *FEATURE_COLUMNS,
                                     "fwd_return", "excess_return"])

    supply = supply_features(supply_demand)
    labels = excess_return(
        forward_return(
            prices,
            entry_offset=entry_offset,
            hold_start=hold_start,
            hold_end=hold_end,
        )
    )

    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    # 순매수 주식 수를 '평균 거래대금이 감당하는 주식 수'로 나눠 규모 차이를 제거한다.
    # adv 단위는 백만원이므로 1e6 을 곱해 원 단위로 되돌린 뒤 진입가로 나눈다.
    shares_capacity = (df["adv"] * 1e6) / df["entry_price"]
    for scaled, raw in _SCALED.items():
        df[scaled] = df[raw] / shares_capacity

    df["log_adv"] = np.log10(df["adv"])

    columns = ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]
    out = df[columns].replace([np.inf, -np.inf], np.nan)
    return out.dropna().sort_values(["date", "code"]).reset_index(drop=True)
