import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS, build_dataset


def _prices(codes: list[str], n: int, spike_at: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            tv = 6000.0 * (5.0 if i == spike_at else 1.0)
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "open": 100.0 + i,
                    "close": 100.0 + i,
                    "trade_value": tv,
                }
            )
    return pd.DataFrame(rows)


def _supply(codes: list[str], n: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "individual": -100,
                    "foreign_": 60,
                    "institution": 40,
                    "penfnd_etc": 10,
                    "acc_trde_qty": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_has_expected_columns():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, tv_multiple=3.0, adv_floor=5000.0,
                       hold_start=10, hold_end=20)
    assert list(ds.columns) == ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]


def test_dataset_drops_rows_without_labels():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    assert ds["excess_return"].notna().all()
    assert ds[FEATURE_COLUMNS].notna().all().all()


def test_dataset_only_contains_event_rows():
    prices = _prices(["A"], 80, spike_at=25)
    sd = _supply(["A"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    # 급증일은 하나뿐이므로 이벤트도 최대 하나
    assert len(ds) <= 1
    if len(ds) == 1:
        assert ds.iloc[0]["date"] == pd.Timestamp("2026-01-01") + pd.Timedelta(days=25)


def test_log_adv_is_log10_of_adv():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    if not ds.empty:
        assert np.isclose(ds.iloc[0]["log_adv"], np.log10(6000.0), atol=1e-6)
