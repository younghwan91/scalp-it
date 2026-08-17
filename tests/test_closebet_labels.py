"""종가베팅 라벨 테스트 — D 종가 → D+1 청산이 정확한지."""

from __future__ import annotations

import pandas as pd
import pytest

from scalp_it.closebet.labels import CB_LABEL_COLUMNS, close_bet_labels


def make_ohlcv(rows: list[dict], code: str = "000001") -> pd.DataFrame:
    """``rows`` 를 거래일 순서대로 붙여 OHLCV 프레임을 만든다."""
    dates = pd.bdate_range("2024-01-01", periods=len(rows))
    out = pd.DataFrame(rows)
    out.insert(0, "date", dates)
    out.insert(0, "code", code)
    return out


def test_label_columns_exact():
    assert CB_LABEL_COLUMNS == ["ret_open", "ret_high", "ret_close", "gap_next"]
    out = close_bet_labels(make_ohlcv([
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
         "volume": 1.0, "trade_value": 1.0},
    ]))
    assert list(out.columns) == ["code", "date", *CB_LABEL_COLUMNS]


def test_ret_open_is_close_to_next_open():
    rows = [
        {"open": 90.0, "high": 101.0, "low": 89.0, "close": 100.0,
         "volume": 1.0, "trade_value": 1.0},
        {"open": 103.0, "high": 110.0, "low": 102.0, "close": 105.0,
         "volume": 1.0, "trade_value": 1.0},
    ]
    out = close_bet_labels(make_ohlcv(rows))
    assert out.loc[0, "ret_open"] == pytest.approx(0.03)
    assert out.loc[0, "ret_high"] == pytest.approx(0.10)
    assert out.loc[0, "ret_close"] == pytest.approx(0.05)


def test_gap_next_equals_ret_open():
    rows = [
        {"open": 90.0, "high": 101.0, "low": 89.0, "close": 100.0,
         "volume": 1.0, "trade_value": 1.0},
        {"open": 96.0, "high": 99.0, "low": 90.0, "close": 92.0,
         "volume": 1.0, "trade_value": 1.0},
    ]
    out = close_bet_labels(make_ohlcv(rows))
    assert out.loc[0, "gap_next"] == pytest.approx(out.loc[0, "ret_open"])
    assert out.loc[0, "ret_open"] == pytest.approx(-0.04)


def test_last_row_has_no_next_day_and_is_nan():
    """거래정지·상폐로 다음 봉이 없으면 라벨은 NaN 이다 (행은 남긴다)."""
    rows = [
        {"open": 90.0, "high": 101.0, "low": 89.0, "close": 100.0,
         "volume": 1.0, "trade_value": 1.0},
        {"open": 103.0, "high": 110.0, "low": 102.0, "close": 105.0,
         "volume": 1.0, "trade_value": 1.0},
    ]
    out = close_bet_labels(make_ohlcv(rows))
    assert out.loc[1, CB_LABEL_COLUMNS].isna().all()
    assert len(out) == 2


def test_labels_do_not_leak_across_codes():
    """앞 종목의 마지막 날이 뒤 종목 첫날을 다음날로 삼으면 안 된다."""
    a = make_ohlcv([
        {"open": 90.0, "high": 101.0, "low": 89.0, "close": 100.0,
         "volume": 1.0, "trade_value": 1.0},
    ], code="000001")
    b = make_ohlcv([
        {"open": 500.0, "high": 510.0, "low": 490.0, "close": 505.0,
         "volume": 1.0, "trade_value": 1.0},
    ], code="000002")
    out = close_bet_labels(pd.concat([a, b], ignore_index=True))
    assert out[CB_LABEL_COLUMNS].isna().all().all()


def test_empty_input_returns_empty_frame():
    out = close_bet_labels(pd.DataFrame(columns=["code", "date", "open", "high", "close"]))
    assert out.empty
    assert list(out.columns) == ["code", "date", *CB_LABEL_COLUMNS]
