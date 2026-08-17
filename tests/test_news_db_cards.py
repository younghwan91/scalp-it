import pandas as pd
import pytest

from scalp_it.adapters.news_db import connect_news, load_article_cards


def test_columns_and_order(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert list(df.columns) == [
        "date", "code", "hotness_rank", "hotness",
        "article_id", "press", "title", "body200", "url",
    ]


def test_column_names_match_the_sibling_loader(news_db):
    """같은 어댑터의 load_articles 와 같은 이름을 쓴다 — 호출부가 헷갈리지 않게."""
    from scalp_it.adapters.news_db import NEWS_COLUMNS

    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    for shared in ("article_id", "date", "code", "press", "title", "body200"):
        assert shared in NEWS_COLUMNS
        assert shared in df.columns


def test_joins_the_representative_article(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    row = df[df["code"] == "034020"].iloc[0]
    assert row["title"] == "체코 원전 2기 최종 계약"
    assert row["press"] == "한국경제"
    assert row["url"] == "https://x/1"
    assert "두코바니" in row["body200"]


def test_row_without_a_primary_article_is_kept_with_nulls(news_db):
    """대표 기사가 없어도 핫함 수치는 카드에 쓴다. LEFT JOIN 이어야 한다."""
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    row = df[df["code"] == "000660"].iloc[0]
    assert pd.isna(row["article_id"])
    assert row["hotness"] == 0.20


def test_rank_is_renamed_to_hotness_rank(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert "rank" not in df.columns
    assert df[df["code"] == "034020"].iloc[0]["hotness_rank"] == 1


def test_aggregate_counts_are_not_returned(news_db):
    """집계 수치는 ②층 피처로 후보 행에 이미 실려 있다. 두 경로로 들여오지 않는다."""
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    for stale in ("n_articles", "n_outlets", "article_count", "article_ratio"):
        assert stale not in df.columns


def test_date_range_is_respected(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-14", "2026-08-14")
    assert df["date"].dt.date.astype(str).unique().tolist() == ["2026-08-14"]
    assert len(df) == 3


def test_ops_only_filters_to_the_operating_universe(news_db):
    df = load_article_cards(
        connect_news(news_db), "2026-08-14", "2026-08-14", ops_only=True
    )
    assert df["code"].tolist() == ["034020"]


def test_date_is_datetime(news_db):
    df = load_article_cards(connect_news(news_db), "2026-08-01", "2026-08-31")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 4


def test_connect_news_fails_loudly_on_a_missing_db(tmp_path):
    """별도 실패 경로를 만들지 않는다. connect_news 가 이미 한다."""
    with pytest.raises(FileNotFoundError, match="뉴스 DB 가 없다"):
        connect_news(str(tmp_path / "nope.db"))
