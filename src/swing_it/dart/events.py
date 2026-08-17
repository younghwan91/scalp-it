"""이벤트 정의 — 공시가 난 (종목, 날짜) 다 (스펙 rev2 §3).

계획 1 의 ``events.extract_events``(거래대금 급증)를 대체한다. 거래대금 급증은
폐기되지 않고 ``tv_ratio`` 로 남아 "시장이 이 재료에 반응했는가"를 재는 피처가 된다.

왜 뒤집었나 (스펙 §0):
  - 검증된 PEAD 가 이미 '재료 → 드리프트' 구조다.
  - 공시 접수일자는 정확한 정보 공개 시점이라 자연 실험이 깨끗하다.
  - 급증 종목의 재료를 찾는 것은 이미 오른 걸 보고 이유를 붙이는 역인과다.
"""

from __future__ import annotations

import pandas as pd

from ..events import add_liquidity_stats
from ..news.calendar import align_to_trading_days
from .classify import EXCLUDED_TYPES

EVENT_COLUMNS: list[str] = ["code", "date", "adv", "tv_ratio"]


def build_disclosure_events(
    disc: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
) -> pd.DataFrame:
    """공시를 이벤트로 바꾼다.

    순서가 결과를 바꾸므로 고정한다: 악재 배제 → 거래일 이월 → (code,date) 중복
    제거 → 유동성 하한 → 가격 없는 행 제거.

    Args:
        adv_floor: 직전 ``adv_window`` 거래일 평균 거래대금 하한. 단위 백만원.
            기본 5000 = 50억. PEAD 검증에서 이 조건 없이는 소형주 꼬리가
            시그널을 음수로 뒤집었다.
    """
    if disc.empty or prices.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    kept = disc[~disc["material_type"].isin(EXCLUDED_TYPES)].copy()
    if kept.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    kept = kept.rename(columns={"rcept_dt": "date"})
    kept = align_to_trading_days(kept, prices["date"])
    if kept.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    pairs = kept[["code", "date"]].drop_duplicates()

    stats = add_liquidity_stats(prices, window=adv_window)
    events = pairs.merge(
        stats[["code", "date", "adv", "tv_ratio"]], on=["code", "date"], how="inner"
    )
    events = events[events["adv"] >= adv_floor]
    return events[EVENT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
