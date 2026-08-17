import numpy as np
import pandas as pd

from scalp_it.features.disclosure import DISC_FEATURE_COLUMNS, disclosure_features

CAL = pd.Series(pd.to_datetime(
    ["2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19"]))


def _disc(rows, corrections=None):
    corrections = corrections or [False] * len(rows)
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(rows))],
        "rcept_dt": pd.to_datetime([r[0] for r in rows]),
        "code": [r[1] for r in rows],
        "corp_name": ["회사"] * len(rows),
        "corp_cls": ["K"] * len(rows),
        "report_nm": [r[2] for r in rows],
        "material_type": [r[2] for r in rows],
        "is_correction": corrections,
    })


def test_columns_contract():
    out = disclosure_features(_disc([("2026-08-14", "A", "공급계약")]), CAL, min_periods=1)
    assert list(out.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def test_count_and_type_ratios():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "내부자")])
    row = disclosure_features(disc, CAL, min_periods=1).iloc[0]
    assert row["disc_count"] == 2
    assert row["is_contract"] == 0.5
    assert row["is_insider"] == 0.5
    assert row["is_capex"] == 0.0


def test_holiday_disclosure_moves_forward():
    disc = _disc([("2026-08-15", "A", "공급계약")])
    out = disclosure_features(disc, CAL, min_periods=1)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_baseline_uses_only_the_past():
    disc = _disc([("2026-08-13", "A", "행정"), ("2026-08-14", "A", "행정"),
                  ("2026-08-17", "A", "공급계약"), ("2026-08-17", "A", "공급계약"),
                  ("2026-08-17", "A", "공급계약")])
    out = disclosure_features(disc, CAL, baseline_window=2, min_periods=2)
    row = out[out["date"] == pd.Timestamp("2026-08-17")].iloc[0]
    assert np.isclose(row["disc_ratio"], 3.0)   # 직전 2일 평균 1.0 → 3배


def test_days_since_counts_trading_days():
    disc = _disc([("2026-08-13", "A", "행정"), ("2026-08-19", "A", "공급계약")])
    out = disclosure_features(disc, CAL, min_periods=1).set_index("date")
    assert out.loc[pd.Timestamp("2026-08-19"), "days_since_disc"] == 4


def test_correction_ratio():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "공급계약")],
                 corrections=[True, False])
    assert disclosure_features(disc, CAL, min_periods=1).iloc[0]["disc_correction_ratio"] == 0.5


def test_top_material_type_is_most_common_and_not_a_model_feature():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "공급계약"),
                  ("2026-08-14", "A", "행정")])
    out = disclosure_features(disc, CAL, min_periods=1)
    assert out.iloc[0]["top_material_type"] == "공급계약"
    assert "top_material_type" not in DISC_FEATURE_COLUMNS


def test_empty_input():
    out = disclosure_features(_disc([]), CAL)
    assert out.empty
    assert list(out.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]
