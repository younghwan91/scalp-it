"""수급 피처 — 텍스트와 독립된 두 번째 증거 (스펙 §3).

핵심 가설: 재료가 강하다면 기관·외국인이 따라 들어온다. 기사 텍스트만으로
강도를 판단하면 결국 기자의 어조를 읽는 것에 가깝다.

당일 순매수를 포함하는 이유: 수급은 장마감 후 확정되고 진입은 D+1 시가이므로
시점 안전하다 (스펙 §3).
"""

from __future__ import annotations

import pandas as pd

_INVESTORS = {
    "foreign_net_5d": "foreign_",
    "inst_net_5d": "institution",
    "indiv_net_5d": "individual",
    "pension_net_5d": "penfnd_etc",
}


def supply_features(sd: pd.DataFrame, *, window: int = 5) -> pd.DataFrame:
    """투자자별 누적 순매수와 거래량 대비 비율을 계산한다."""
    out = sd.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)

    for name, column in _INVESTORS.items():
        out[name] = grouped[column].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )

    volume_sum = grouped["acc_trde_qty"].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )
    out["foreign_net_ratio"] = out["foreign_net_5d"] / volume_sum
    out["inst_net_ratio"] = out["inst_net_5d"] / volume_sum
    out["smart_money_net_5d"] = out["foreign_net_5d"] + out["inst_net_5d"]

    columns = ["code", "date", *_INVESTORS, "foreign_net_ratio", "inst_net_ratio",
               "smart_money_net_5d"]
    return out[columns].reset_index(drop=True)
