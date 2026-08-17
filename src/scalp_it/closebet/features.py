"""종가베팅 피처 — "오늘 흐름이 좋았나" 를 일봉으로 재는 순수 함수.

**모든 값은 D 종가 시점에 알 수 있어야 한다.** 이동평균류는 반드시
``shift(1)`` 한 뒤 나눈다 — 당일 거래대금이 당일 평균에 섞여 들어가면
그 자체로 미래 정보는 아니지만 분모가 분자를 흡수해 비율이 눌린다.
그보다 중요한 건 D+1 이후 값이 절대 들어오지 않는다는 점이고,
회귀 테스트(`tests/test_closebet_features.py`)가 이를 강제한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 반환 프레임의 피처 컬럼. 순서가 계약이다.
CB_FEATURE_COLUMNS = [
    "ret_day",
    "close_pos",
    "body_ratio",
    "range_pct",
    "tv_ratio",
    "log_tv",
    "gap_prev",
    "vol_ratio",
]

_OHLCV_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "trade_value"]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """0 나눗셈을 NaN 으로 만든다 (inf 를 만들지 않는다)."""
    den = den.where(den != 0)
    return num / den


def close_bet_features(ohlcv: pd.DataFrame, *, adv_window: int = 20) -> pd.DataFrame:
    """일봉에서 종가베팅 피처를 만든다.

    Args:
        ohlcv: ``code, date, open, high, low, close, volume, trade_value``.
        adv_window: 평균 거래대금·거래량 창 길이. 창은 **직전** 구간이다.

    Returns:
        ``[code, date, *CB_FEATURE_COLUMNS]``. 입력과 같은 행 수이며
        워밍업이 모자란 앞부분은 ``tv_ratio``·``vol_ratio`` 가 NaN 이다.
    """
    missing = [c for c in _OHLCV_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv 에 컬럼이 없다: {missing}")
    if ohlcv.empty:
        return pd.DataFrame(columns=["code", "date", *CB_FEATURE_COLUMNS])

    out = ohlcv.sort_values(["code", "date"]).reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close", "volume", "trade_value"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = out.groupby("code", sort=False)
    prev_close = grouped["close"].shift(1)

    out["ret_day"] = _safe_div(out["close"] - prev_close, prev_close)

    span = out["high"] - out["low"]
    close_pos = _safe_div(out["close"] - out["low"], span)
    # high == low (상한가 점상 등) 면 종가 위치를 말할 수 없다. 중립 0.5.
    out["close_pos"] = close_pos.where(span != 0, 0.5)

    out["body_ratio"] = _safe_div(out["close"] - out["open"], out["open"])
    out["range_pct"] = _safe_div(span, prev_close)

    # shift(1) 이 먼저다. rolling 을 먼저 하면 당일이 창에 포함된다.
    def _prior_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(adv_window, min_periods=adv_window).mean()

    adv = grouped["trade_value"].transform(_prior_mean)
    advol = grouped["volume"].transform(_prior_mean)
    out["tv_ratio"] = _safe_div(out["trade_value"], adv)
    out["vol_ratio"] = _safe_div(out["volume"], advol)

    tv = out["trade_value"]
    out["log_tv"] = pd.Series(
        np.where(tv > 0, np.log10(tv.where(tv > 0, 1.0)), np.nan), index=out.index
    )

    out["gap_prev"] = _safe_div(out["open"] - prev_close, prev_close)

    return out[["code", "date", *CB_FEATURE_COLUMNS]].reset_index(drop=True)
