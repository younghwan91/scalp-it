import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.layers import (
    LAYER_LABELS,
    LAYER_NAMES,
    build_layered_dataset,
    layer_features,
)

NEWS_COLS = ["article_count", "hotness"]
LLM_COLS = ["llm_specificity", "llm_persistence"]


def _prices(codes: list[str], n: int, spike_at: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "open": 100.0 + i,
                    "close": 100.0 + i,
                    "trade_value": 6000.0 * (5.0 if i == spike_at else 1.0),
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


def _side(codes: list[str], n: int, columns: list[str], value: float) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            row = {
                "code": code,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
            row.update({col: value for col in columns})
            rows.append(row)
    return pd.DataFrame(rows)


def _built(codes, *, spike_at=25, news=None, llm=None, **kwargs):
    n = kwargs.pop("n", 80)
    return build_layered_dataset(
        _prices(codes, n, spike_at=spike_at),
        _supply(codes, n),
        news=_side(codes, n, NEWS_COLS, 2.0) if news is None else news,
        news_columns=NEWS_COLS,
        llm=_side(codes, n, LLM_COLS, 0.5) if llm is None else llm,
        llm_columns=LLM_COLS,
        hold_start=10,
        hold_end=20,
        **kwargs,
    )


def test_layers_are_cumulative():
    spec = layer_features(NEWS_COLS, LLM_COLS)
    assert list(spec) == LAYER_NAMES
    assert spec["supply"] == FEATURE_COLUMNS
    assert spec["supply_news"] == [*FEATURE_COLUMNS, *NEWS_COLS]
    assert spec["supply_news_llm"] == [*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]


def test_layer_features_defaults_to_supply_only():
    spec = layer_features()
    assert spec["supply"] == spec["supply_news"] == spec["supply_news_llm"]


def test_layer_labels_cover_every_layer():
    assert set(LAYER_LABELS) == set(LAYER_NAMES)


def test_layered_dataset_column_order():
    ds = _built(["A", "B"])
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS,
        "fwd_return", "excess_return",
    ]


def test_every_layer_sees_identical_rows():
    """세 층이 같은 행 위에서 돌지 않으면 층 비교는 아무 의미가 없다."""
    ds = _built(["A", "B"])
    for columns in layer_features(NEWS_COLS, LLM_COLS).values():
        assert ds[columns].notna().all().all()


def test_missing_llm_row_is_dropped_from_all_layers():
    codes = ["A", "B"]
    llm = _side(codes, 80, LLM_COLS, 0.5)
    spike_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=25)
    mask = (llm["code"] == "B") & (llm["date"] == spike_date)
    llm.loc[mask, LLM_COLS] = np.nan
    ds = _built(codes, llm=llm)
    assert "B" not in set(ds["code"])


def test_event_without_a_news_row_is_dropped():
    """뉴스 결측은 채우지 않는다. 표본 정의는 '세 층 피처가 모두 있는 행' 하나뿐이다."""
    codes = ["A", "B"]
    news = _side(["A"], 80, NEWS_COLS, 2.0)  # B 의 뉴스 행이 아예 없다
    ds = _built(codes, news=news)
    assert set(ds["code"]) == {"A"}


def test_require_label_false_keeps_unlabelled_recent_rows():
    ds = _built(["A", "B"], spike_at=78, require_label=False)
    assert len(ds) == 2
    assert ds["excess_return"].isna().all()
    assert ds[[*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]].notna().all().all()


def test_require_label_true_drops_those_same_rows():
    ds = _built(["A", "B"], spike_at=78, require_label=True)
    assert ds.empty


def test_no_side_frames_reduces_to_baseline_columns():
    codes = ["A", "B"]
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25), _supply(codes, 80), hold_start=10, hold_end=20
    )
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"
    ]


def test_side_frame_missing_a_declared_column_fails_loudly():
    import pytest

    codes = ["A"]
    with pytest.raises(ValueError, match="hotness"):
        build_layered_dataset(
            _prices(codes, 80, spike_at=25),
            _supply(codes, 80),
            news=_side(codes, 80, ["article_count"], 2.0),
            news_columns=NEWS_COLS,
            hold_start=10,
            hold_end=20,
        )


def test_rows_are_sorted_by_date_then_code():
    ds = _built(["B", "A"])
    assert ds[["date", "code"]].equals(
        ds[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True)
    )
