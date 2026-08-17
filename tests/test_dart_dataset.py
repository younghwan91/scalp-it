import pandas as pd

from swing_it.dart.dataset import DISC_DATASET_COLUMNS, build_dart_dataset
from swing_it.dataset import FEATURE_COLUMNS
from swing_it.features.disclosure import DISC_FEATURE_COLUMNS


def _prices(codes, n):
    rows = []
    for code in codes:
        for i in range(n):
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "open": 100.0 + i, "close": 100.0 + i, "trade_value": 9000.0})
    return pd.DataFrame(rows)


def _supply(codes, n):
    rows = []
    for code in codes:
        for i in range(n):
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "individual": -100, "foreign_": 60, "institution": 40,
                         "penfnd_etc": 10, "acc_trde_qty": 1000})
    return pd.DataFrame(rows)


def _disc(pairs):
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(pairs))],
        "rcept_dt": pd.to_datetime([p[0] for p in pairs]),
        "code": [p[1] for p in pairs],
        "corp_name": ["회사"] * len(pairs),
        "corp_cls": ["K"] * len(pairs),
        "report_nm": [p[2] for p in pairs],
        "material_type": [p[2] for p in pairs],
        "is_correction": [False] * len(pairs),
    })


def test_dataset_columns_contract():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약"), ("2026-01-30", "B", "내부자")]),
        hold_start=10, hold_end=20,
    )
    assert list(ds.columns) == DISC_DATASET_COLUMNS


def test_feature_budget_is_within_spec_limit():
    """스펙 §7: 피처 10~20개."""
    assert 10 <= len(FEATURE_COLUMNS) + len(DISC_FEATURE_COLUMNS) <= 20


def test_no_column_collision_between_plan1_and_disclosure():
    assert set(FEATURE_COLUMNS).isdisjoint(DISC_FEATURE_COLUMNS)


def test_rows_are_events_only():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약")]), hold_start=10, hold_end=20,
    )
    assert set(ds["code"]) <= {"A"}


def test_labels_and_features_have_no_nan():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약"), ("2026-01-31", "B", "시설투자")]),
        hold_start=10, hold_end=20,
    )
    cols = [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS, "excess_return"]
    assert ds[cols].notna().all().all()
