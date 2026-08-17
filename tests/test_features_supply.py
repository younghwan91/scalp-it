import numpy as np
import pandas as pd

from scalp_it.features.supply import supply_features


def _sd(code: str, foreign: list[int], inst: list[int], indiv: list[int], vol: list[int]):
    n = len(foreign)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "individual": indiv,
            "foreign_": foreign,
            "institution": inst,
            "penfnd_etc": [0] * n,
            "acc_trde_qty": vol,
        }
    )


def test_net_sums_include_current_day():
    df = _sd("A", foreign=[10, 20, 30], inst=[1, 2, 3], indiv=[-11, -22, -33], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["foreign_net_5d"] == 60  # 10+20+30, 당일 포함
    assert out.iloc[2]["inst_net_5d"] == 6


def test_window_not_filled_is_nan():
    df = _sd("A", foreign=[10, 20], inst=[1, 2], indiv=[-11, -22], vol=[100, 100])
    out = supply_features(df, window=3)
    assert np.isnan(out.iloc[0]["foreign_net_5d"])
    assert np.isnan(out.iloc[1]["foreign_net_5d"])


def test_ratio_divides_by_volume_sum():
    df = _sd("A", foreign=[10, 20, 30], inst=[0, 0, 0], indiv=[0, 0, 0], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["foreign_net_ratio"] == 60 / 300


def test_smart_money_is_foreign_plus_institution():
    df = _sd("A", foreign=[10, 20, 30], inst=[1, 2, 3], indiv=[0, 0, 0], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["smart_money_net_5d"] == 66


def test_features_are_per_code():
    df = pd.concat(
        [
            _sd("A", [10, 10, 10], [0, 0, 0], [0, 0, 0], [100, 100, 100]),
            _sd("B", [99, 99, 99], [0, 0, 0], [0, 0, 0], [100, 100, 100]),
        ],
        ignore_index=True,
    )
    out = supply_features(df, window=3)
    assert out[out["code"] == "A"].iloc[2]["foreign_net_5d"] == 30
    assert out[out["code"] == "B"].iloc[2]["foreign_net_5d"] == 297
