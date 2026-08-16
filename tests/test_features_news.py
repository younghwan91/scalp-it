import pandas as pd
import pytest

from swing_it.features.news import NEWS_FEATURE_COLUMNS, news_features


def _daily_news(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14"] * n),
            "code": [f"{i:06d}" for i in range(n)],
            "n_articles": [7, 3, 1],
            "n_outlets": [5, 2, 1],
            "article_ratio": [4.2, 1.8, 0.9],
            "adv": [8420.0, 6100.0, 5200.0],
            "tv_ratio": [4.6, 2.1, 1.2],
            "hotness": [0.81, 0.44, 0.20],
            "rank": [1, 2, 3],
            "in_ops": [True, True, False],
            "primary_news_id": ["n1", "n2", "n3"],
        }
    )


def test_registered_feature_columns():
    assert NEWS_FEATURE_COLUMNS == [
        "article_count", "outlet_count", "article_ratio", "hotness", "hotness_rank"
    ]


def test_output_columns_and_order():
    out = news_features(_daily_news())
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]


def test_source_columns_are_renamed():
    out = news_features(_daily_news())
    assert out["article_count"].tolist() == [7.0, 3.0, 1.0]
    assert out["outlet_count"].tolist() == [5.0, 2.0, 1.0]
    assert out["hotness_rank"].tolist() == [1.0, 2.0, 3.0]
    for stale in ("n_articles", "n_outlets", "rank"):
        assert stale not in out.columns


def test_trade_value_columns_are_excluded():
    """tv_ratio·adv 는 계획 1 FEATURE_COLUMNS 에 이미 있다. 중복 투입 금지."""
    out = news_features(_daily_news())
    assert "tv_ratio" not in out.columns
    assert "adv" not in out.columns


def test_bookkeeping_columns_are_excluded():
    out = news_features(_daily_news())
    assert "in_ops" not in out.columns
    assert "primary_news_id" not in out.columns


def test_row_count_is_preserved():
    out = news_features(_daily_news(3))
    assert len(out) == 3


def test_all_feature_values_are_float():
    out = news_features(_daily_news())
    for col in NEWS_FEATURE_COLUMNS:
        assert out[col].dtype == "float64"


def test_date_is_datetime():
    out = news_features(_daily_news())
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_missing_source_column_fails_loudly():
    bad = _daily_news().drop(columns=["hotness"])
    with pytest.raises(ValueError, match="hotness"):
        news_features(bad)


def test_empty_input_returns_empty_frame_with_columns():
    empty = _daily_news().iloc[0:0]
    out = news_features(empty)
    assert out.empty
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]


def test_load_news_features_reads_sqlite(news_db):
    from swing_it.features.news import load_news_features

    out = load_news_features(news_db, start="2026-08-14", end="2026-08-14")
    assert list(out.columns) == ["code", "date", *NEWS_FEATURE_COLUMNS]
    assert set(out["code"]) == {"034020", "005930", "000660"}
    assert out[out["code"] == "034020"].iloc[0]["article_count"] == 7.0


def test_load_news_features_fails_when_the_db_is_missing(tmp_path):
    from swing_it.features.news import load_news_features

    with pytest.raises(FileNotFoundError, match="뉴스 DB 가 없다"):
        load_news_features(str(tmp_path / "nope.db"), start="2026-01-01", end="2026-12-31")
