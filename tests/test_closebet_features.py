"""종가베팅 피처 테스트 — 합성 OHLCV 로 손계산과 대조한다.

가장 중요한 건 마지막 lookahead 회귀 테스트다. 미래 봉을 통째로 바꿔도
과거 피처가 한 톨도 안 움직여야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_it.closebet.features import CB_FEATURE_COLUMNS, close_bet_features


def make_ohlcv(rows: list[dict], code: str = "000001") -> pd.DataFrame:
    """``rows`` 를 거래일 순서대로 붙여 OHLCV 프레임을 만든다."""
    dates = pd.bdate_range("2024-01-01", periods=len(rows))
    out = pd.DataFrame(rows)
    out.insert(0, "date", dates)
    out.insert(0, "code", code)
    return out


def flat_rows(n: int, *, close: float = 100.0, trade_value: float = 1e9,
              volume: float = 1e5) -> list[dict]:
    """워밍업용 밋밋한 봉들."""
    return [
        {"open": close, "high": close, "low": close, "close": close,
         "volume": volume, "trade_value": trade_value}
        for _ in range(n)
    ]


def test_feature_columns_exact_and_ordered():
    assert CB_FEATURE_COLUMNS == [
        "ret_day", "close_pos", "body_ratio", "range_pct",
        "tv_ratio", "log_tv", "gap_prev", "vol_ratio",
    ]
    df = close_bet_features(make_ohlcv(flat_rows(3)))
    assert list(df.columns) == ["code", "date", *CB_FEATURE_COLUMNS]


def test_ret_day_is_close_over_prev_close():
    rows = flat_rows(1, close=100.0) + [
        {"open": 100.0, "high": 112.0, "low": 99.0, "close": 110.0,
         "volume": 1e5, "trade_value": 1e9}
    ]
    out = close_bet_features(make_ohlcv(rows))
    assert pd.isna(out.loc[0, "ret_day"])  # 전일 종가가 없다
    assert out.loc[1, "ret_day"] == pytest.approx((110.0 - 100.0) / 100.0)


def test_close_pos_is_position_within_day_range():
    rows = [{"open": 100.0, "high": 120.0, "low": 100.0, "close": 115.0,
             "volume": 1e5, "trade_value": 1e9}]
    out = close_bet_features(make_ohlcv(rows))
    # (115 - 100) / (120 - 100) = 0.75
    assert out.loc[0, "close_pos"] == pytest.approx(0.75)


def test_close_pos_is_half_when_high_equals_low():
    """상한가 점상처럼 고가=저가면 종가 위치를 말할 수 없다 → 중립 0.5."""
    rows = [{"open": 130.0, "high": 130.0, "low": 130.0, "close": 130.0,
             "volume": 1e5, "trade_value": 1e9}]
    out = close_bet_features(make_ohlcv(rows))
    assert out.loc[0, "close_pos"] == pytest.approx(0.5)
    assert np.isfinite(out.loc[0, "close_pos"])


def test_body_ratio_and_range_pct_hand_computed():
    rows = flat_rows(1, close=100.0) + [
        {"open": 102.0, "high": 118.0, "low": 101.0, "close": 114.0,
         "volume": 1e5, "trade_value": 1e9}
    ]
    out = close_bet_features(make_ohlcv(rows))
    assert out.loc[1, "body_ratio"] == pytest.approx((114.0 - 102.0) / 102.0)
    assert out.loc[1, "range_pct"] == pytest.approx((118.0 - 101.0) / 100.0)


def test_gap_prev_is_open_over_prev_close():
    rows = flat_rows(1, close=100.0) + [
        {"open": 105.0, "high": 110.0, "low": 104.0, "close": 108.0,
         "volume": 1e5, "trade_value": 1e9}
    ]
    out = close_bet_features(make_ohlcv(rows))
    assert out.loc[1, "gap_prev"] == pytest.approx(0.05)


def test_log_tv_is_log10_of_trade_value():
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
             "volume": 1e5, "trade_value": 1e9}]
    out = close_bet_features(make_ohlcv(rows))
    assert out.loc[0, "log_tv"] == pytest.approx(9.0)


def test_tv_ratio_uses_prior_window_only():
    """당일 거래대금이 평균에 섞이면 값이 달라진다 — shift(1) 를 강제한다."""
    rows = flat_rows(3, trade_value=1e9) + [
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0,
         "volume": 1e5, "trade_value": 5e9}
    ]
    out = close_bet_features(make_ohlcv(rows), adv_window=3)
    # 직전 3일 평균은 1e9 이므로 5.0. 당일을 포함하면 (1+1+5)/3=2.33 → 2.14 가 된다.
    assert out.loc[3, "tv_ratio"] == pytest.approx(5.0)
    assert out.loc[3, "tv_ratio"] != pytest.approx(5e9 / ((1e9 + 1e9 + 5e9) / 3))


def test_vol_ratio_uses_prior_window_only():
    rows = flat_rows(3, volume=1000.0) + [
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0,
         "volume": 4000.0, "trade_value": 1e9}
    ]
    out = close_bet_features(make_ohlcv(rows), adv_window=3)
    assert out.loc[3, "vol_ratio"] == pytest.approx(4.0)


def test_rolling_windows_are_nan_before_warmup():
    """창이 다 안 찼으면 부분 평균으로 때우지 않고 NaN 을 낸다."""
    out = close_bet_features(make_ohlcv(flat_rows(5)), adv_window=3)
    assert out.loc[:2, "tv_ratio"].isna().all()
    assert out.loc[3, "tv_ratio"] == pytest.approx(1.0)


def test_features_do_not_leak_across_codes():
    """다른 종목의 직전 봉이 전일 종가로 새어 들어오면 안 된다."""
    a = make_ohlcv(flat_rows(2, close=100.0), code="000001")
    b = make_ohlcv(flat_rows(2, close=500.0), code="000002")
    out = close_bet_features(pd.concat([a, b], ignore_index=True))
    first_of_b = out[out["code"] == "000002"].iloc[0]
    assert pd.isna(first_of_b["ret_day"])
    assert pd.isna(first_of_b["gap_prev"])


def test_unsorted_input_is_sorted_before_computing():
    rows = flat_rows(1, close=100.0) + [
        {"open": 100.0, "high": 112.0, "low": 99.0, "close": 110.0,
         "volume": 1e5, "trade_value": 1e9}
    ]
    ordered = make_ohlcv(rows)
    shuffled = ordered.iloc[::-1].reset_index(drop=True)
    out = close_bet_features(shuffled)
    assert out.loc[1, "ret_day"] == pytest.approx(0.10)


def test_missing_column_raises():
    df = make_ohlcv(flat_rows(2)).drop(columns=["high"])
    with pytest.raises(ValueError, match="high"):
        close_bet_features(df)


def test_empty_input_returns_empty_frame_with_columns():
    out = close_bet_features(pd.DataFrame(
        columns=["code", "date", "open", "high", "low", "close", "volume", "trade_value"]
    ))
    assert out.empty
    assert list(out.columns) == ["code", "date", *CB_FEATURE_COLUMNS]


def test_no_lookahead_future_bars_cannot_change_past_features():
    """회귀 테스트: 미래 봉을 극단적으로 바꿔도 과거 피처는 불변이다."""
    base_rows = flat_rows(5) + [
        {"open": 100.0, "high": 112.0, "low": 98.0, "close": 111.0,
         "volume": 3e5, "trade_value": 4e9},
        {"open": 111.0, "high": 115.0, "low": 110.0, "close": 113.0,
         "volume": 1e5, "trade_value": 1e9},
        {"open": 113.0, "high": 120.0, "low": 112.0, "close": 119.0,
         "volume": 1e5, "trade_value": 1e9},
    ]
    tampered_rows = [dict(r) for r in base_rows]
    for row in tampered_rows[6:]:  # 인덱스 5 이후의 미래만 손댄다
        row.update(open=9_999.0, high=99_999.0, low=1.0, close=50_000.0,
                   volume=9e9, trade_value=9e15)

    base = close_bet_features(make_ohlcv(base_rows), adv_window=3)
    tampered = close_bet_features(make_ohlcv(tampered_rows), adv_window=3)

    # 0..5 행 — 인덱스 5 가 조작 직전 행이다
    pd.testing.assert_frame_equal(
        base[CB_FEATURE_COLUMNS].iloc[:6], tampered[CB_FEATURE_COLUMNS].iloc[:6]
    )
    # 조작이 실제로 뒤쪽은 바꿨는지 확인 — 테스트가 헛돌지 않게
    assert base.loc[7, "ret_day"] != pytest.approx(tampered.loc[7, "ret_day"])
