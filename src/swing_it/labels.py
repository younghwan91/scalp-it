"""라벨 생성 — 순수 함수.

라벨은 D+1 시가 진입 → D+``hold_end`` 종가 청산의 초과수익이다.
중간 손절·이익실현은 없다. 프랍 스윙 탐색에서 PEAD-집중을 죽인 직접 원인이
10% 손절 + 10일 캡으로 드리프트를 자른 것이었다 (스펙 §2).
"""

from __future__ import annotations

import pandas as pd


def forward_return(
    prices: pd.DataFrame,
    *,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """(종목, 날짜) 마다 진입가·청산가·수익률을 계산한다.

    Args:
        entry_offset: 이벤트일로부터 진입까지의 거래일 수. 1 = 다음날 시가.
        hold_start: 최소 보유 거래일. 현재는 데이터 충분성 확인에만 쓴다.
        hold_end: 청산 시점(이벤트일로부터의 거래일 수). 종가로 청산.
    """
    out = prices.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)
    out["entry_price"] = grouped["open"].shift(-entry_offset)
    out["exit_price"] = grouped["close"].shift(-hold_end)
    # 보유 구간 자체가 확보되지 않은 행은 라벨을 만들지 않는다
    has_min_hold = grouped["close"].shift(-hold_start).notna()
    out.loc[~has_min_hold, "exit_price"] = pd.NA
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["fwd_return"] = (out["exit_price"] - out["entry_price"]) / out["entry_price"]
    return out[["code", "date", "entry_price", "exit_price", "fwd_return"]].reset_index(drop=True)


def excess_return(fwd: pd.DataFrame) -> pd.DataFrame:
    """같은 날짜 유니버스 동일가중 평균 대비 초과수익을 붙인다.

    업종 지수(``sector_index``)가 2024-01-26 부터만 존재해 전 기간 벤치마크로
    쓸 수 없으므로 횡단면 디민으로 대신한다 (계획 1 데이터 실측).
    """
    out = fwd.copy()
    market = out.groupby("date")["fwd_return"].transform("mean")
    out["excess_return"] = out["fwd_return"] - market
    return out
