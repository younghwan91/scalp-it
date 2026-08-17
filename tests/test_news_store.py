import pandas as pd

from scalp_it.news.store import (
    init_db,
    load_daily_news,
    replace_daily_news,
    upsert_article_codes,
    upsert_articles,
)


def _articles(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"news_id": nid, "date": pd.Timestamp(day), "outlet": "매체",
             "title": title, "body": "본문", "keywords": "", "orgs": "", "url": ""}
            for nid, day, title in rows
        ]
    )


def _ranked(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(day), "code": code, "n_articles": 3, "n_outlets": 5,
             "article_ratio": 6.0, "adv": 10000.0, "tv_ratio": 2.0, "hotness": hotness,
             "rank": rank, "in_train": True, "in_ops": rank <= 10,
             "primary_news_id": "n1"}
            for day, code, rank, hotness in rows
        ]
    )


def test_init_creates_three_tables(tmp_path):
    conn = init_db(tmp_path / "news.db")
    names = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='table'")
    }
    assert {"articles", "article_codes", "daily_news"} <= names


def test_articles_roundtrip(tmp_path):
    conn = init_db(tmp_path / "news.db")
    written = upsert_articles(conn, _articles([("a", "2026-08-17", "제목")]))
    assert written == 1
    row = conn.execute("select news_id, date, title from articles").fetchone()
    assert row == ("a", "2026-08-17", "제목")


def test_articles_upsert_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "news.db")
    upsert_articles(conn, _articles([("a", "2026-08-17", "옛 제목")]))
    upsert_articles(conn, _articles([("a", "2026-08-17", "새 제목")]))
    rows = conn.execute("select title from articles").fetchall()
    assert rows == [("새 제목",)]


def test_article_codes_upsert_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "news.db")
    mapped = pd.DataFrame([{"news_id": "a", "date": pd.Timestamp("2026-08-17"),
                            "code": "005930", "match_kind": "name",
                            "outlet": "매체", "title": "제목"}])
    upsert_article_codes(conn, mapped)
    upsert_article_codes(conn, mapped)
    assert conn.execute("select count(*) from article_codes").fetchone()[0] == 1


def test_daily_news_roundtrip_types(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([("2026-08-17", "005930", 1, 2.5)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out.iloc[0]["code"] == "005930"
    assert out.iloc[0]["rank"] == 1
    assert bool(out.iloc[0]["in_ops"]) is True


def test_replace_wipes_the_date_range_before_inserting(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-17", "005930", 1, 2.5),
        ("2026-08-17", "000660", 2, 2.0),
    ]))
    replace_daily_news(conn, _ranked([("2026-08-17", "005930", 1, 9.9)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert out["code"].tolist() == ["005930"]
    assert out.iloc[0]["hotness"] == 9.9


def test_replace_leaves_other_dates_alone(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([("2026-08-10", "005930", 1, 1.0)]))
    replace_daily_news(conn, _ranked([("2026-08-17", "000660", 1, 2.0)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert len(out) == 2


def test_load_respects_date_range(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-10", "005930", 1, 1.0),
        ("2026-08-17", "000660", 1, 2.0),
    ]))
    out = load_daily_news(conn, "2026-08-15", "2026-08-31")
    assert out["code"].tolist() == ["000660"]


def test_ops_only_filters_to_top_ten(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-17", f"{i:06d}", i + 1, float(50 - i)) for i in range(30)
    ]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31", ops_only=True)
    assert len(out) == 10


def test_load_on_empty_db_returns_typed_empty_frame(tmp_path):
    conn = init_db(tmp_path / "news.db")
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert out.empty
    assert "hotness" in out.columns
