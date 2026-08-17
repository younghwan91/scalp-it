"""공시 이벤트 + 수급 피처 + 공시 피처 + 라벨을 한 표로 조립한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset import FEATURE_COLUMNS
from ..features.disclosure import DISC_FEATURE_COLUMNS, disclosure_features
from ..features.supply import supply_features
from ..labels import excess_return, forward_return
from .events import build_disclosure_events

DISC_DATASET_COLUMNS: list[str] = [
    "code", "date", *FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS,
    "fwd_return", "excess_return", "top_material_type",
]

_SCALED = {
    "smart_money_net_5d_scaled": "smart_money_net_5d",
    "indiv_net_5d_scaled": "indiv_net_5d",
    "pension_net_5d_scaled": "pension_net_5d",
}


def build_dart_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    disc: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """공시 이벤트 행에 피처와 라벨을 붙인다."""
    events = build_disclosure_events(
        disc, prices, adv_floor=adv_floor, adv_window=adv_window
    )
    if events.empty:
        return pd.DataFrame(columns=DISC_DATASET_COLUMNS)

    supply = supply_features(supply_demand)
    disc_feats = disclosure_features(disc, prices["date"])
    labels = excess_return(
        forward_return(
            prices, entry_offset=entry_offset, hold_start=hold_start, hold_end=hold_end
        )
    )

    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(disc_feats, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    shares_capacity = (df["adv"] * 1e6) / df["entry_price"]
    for scaled, raw in _SCALED.items():
        df[scaled] = df[raw] / shares_capacity
    df["log_adv"] = np.log10(df["adv"])

    out = df[DISC_DATASET_COLUMNS].replace([np.inf, -np.inf], np.nan)
    numeric = [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS, "excess_return"]
    out = out.dropna(subset=numeric)
    return out.sort_values(["date", "code"]).reset_index(drop=True)
