import numpy as np
import pandas as pd

from scalp_it.dataset import FEATURE_COLUMNS, build_dataset


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


def test_scaling_uses_event_day_close_not_future_entry_price():
    """규모 조정 분모는 이벤트일 종가여야 한다 — 진입가는 미래다.

    후보를 고르는 시점은 D 종가이고 진입은 D+1 시가다. 진입가로 나누면
    선택 시점에 모르는 값이 피처에 들어간다. 실측 영향은 9bp 로 작았지만
    (2024+ strategy +0.0300 → +0.0291) 시점 안전이 공짜다.
    """
    # 종가와 다음날 시가를 크게 벌려, 어느 쪽으로 나눴는지 값으로 구분한다.
    n = 80
    prices = pd.DataFrame({
        "code": ["A"] * n,
        "date": [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(n)],
        "open": [1000.0] * n,     # 시가는 종가의 10배
        "close": [100.0] * n,
        "trade_value": [6000.0 * (5.0 if i == 25 else 1.0) for i in range(n)],
    })
    sd = pd.DataFrame({
        "code": ["A"] * n,
        "date": prices["date"],
        "individual": [-100] * n, "foreign_": [60] * n, "institution": [40] * n,
        "penfnd_etc": [10] * n, "acc_trde_qty": [1000] * n,
    })
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    assert not ds.empty

    adv = 6000.0
    capacity_close = (adv * 1e6) / 100.0     # 종가 기준
    capacity_open = (adv * 1e6) / 1000.0     # 진입가 기준 (10배 작다)
    smart = 100.0                             # foreign 60 + institution 40, 5일 합/5일
    got = ds.iloc[0]["smart_money_net_5d_scaled"]

    expected_close = (smart * 5) / capacity_close
    expected_open = (smart * 5) / capacity_open
    assert np.isclose(got, expected_close)
    assert not np.isclose(got, expected_open)
