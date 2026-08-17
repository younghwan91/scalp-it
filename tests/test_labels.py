import numpy as np
import pandas as pd

from scalp_it.labels import excess_return, forward_return


def _series(code: str, opens: list[float], closes: list[float]) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": opens,
            "close": closes,
        }
    )


def test_entry_is_next_day_open_and_exit_is_hold_end_close():
    # 0일차 이벤트 → 1일차 시가 진입(=110), 3일차 종가 청산(=132) → 20%
    opens = [100.0, 110.0, 120.0, 130.0]
    closes = [105.0, 115.0, 125.0, 132.0]
    fwd = forward_return(_series("A", opens, closes), entry_offset=1, hold_start=2, hold_end=3)
    row = fwd.iloc[0]
    assert row["entry_price"] == 110.0
    assert row["exit_price"] == 132.0
    assert row["fwd_return"] == (132.0 - 110.0) / 110.0


def test_rows_without_enough_future_are_nan():
    opens = [100.0, 110.0, 120.0, 130.0]
    closes = [105.0, 115.0, 125.0, 132.0]
    fwd = forward_return(_series("A", opens, closes), entry_offset=1, hold_start=2, hold_end=3)
    # 마지막 행은 미래가 없다
    assert np.isnan(fwd.iloc[-1]["fwd_return"])


def test_excess_return_demeans_within_entry_date():
    fwd = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "entry_price": [100.0, 100.0, 100.0],
            "exit_price": [110.0, 120.0, 130.0],
            "fwd_return": [0.10, 0.20, 0.30],
        }
    )
    out = excess_return(fwd)
    # 평균 0.20 → 초과수익 -0.10 / 0.00 / +0.10
    assert np.allclose(out["excess_return"].tolist(), [-0.10, 0.0, 0.10])


def test_excess_return_ignores_nan_rows_in_mean():
    fwd = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "entry_price": [100.0, 100.0, 100.0],
            "exit_price": [110.0, 130.0, np.nan],
            "fwd_return": [0.10, 0.30, np.nan],
        }
    )
    out = excess_return(fwd)
    assert np.allclose(out["excess_return"].iloc[:2].tolist(), [-0.10, 0.10])
    assert np.isnan(out["excess_return"].iloc[2])
