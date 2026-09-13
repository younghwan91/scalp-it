"""유니버스 필터와 이벤트 정의 — 순수 함수, DB 를 모른다.

이벤트는 '거래대금 급증'이다. 스펙 §4 의 '핫함' 정의 중 과거·현재 양쪽에서
동일하게 계산 가능한 축이며, 뉴스 없이도 성립한다(계획 1 베이스라인).
"""

from __future__ import annotations

import pandas as pd


def add_liquidity_stats(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """종목별 직전 ``window`` 일 평균 거래대금(``adv``)과 당일 배수(``tv_ratio``)를 붙인다.

    ``adv`` 는 반드시 ``shift(1)`` 이후 계산한다. 당일 거래대금을 자기 평균에
    포함하면 lookahead 이며, 배수가 구조적으로 축소된다.
    """
    out = prices.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)["trade_value"]
    out["adv"] = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )
    out["tv_ratio"] = out["trade_value"] / out["adv"]
    return out


def extract_events(
    prices: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    window: int = 20,
) -> pd.DataFrame:
    """유동성 하한을 통과하면서 거래대금이 급증한 (종목, 날짜) 를 추린다.

    Args:
        adv_floor: 직전 평균 거래대금 하한. 단위 백만원. 기본 5000 = 50억원
            (swing-it(구 kr-quant) PEAD 검증에서 필수였던 point-in-time 유동성 하한과 같은 값).
        tv_multiple: 평균 대비 당일 거래대금 배수 하한.
        window: 평균 계산 구간(거래일).
    """
    stats = add_liquidity_stats(prices, window=window)
    hit = (stats["adv"] >= adv_floor) & (stats["tv_ratio"] >= tv_multiple)
    return stats.loc[hit, ["code", "date", "adv", "tv_ratio"]].reset_index(drop=True)
