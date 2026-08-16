"""로컬 sqlite 적재.

실 DB(TimescaleDB)는 읽기 전용이므로 뉴스 산출물은 전부 여기로 간다.
파일은 ``data/news.db`` 이고 ``.gitignore`` 가 차단한다 — 빅카인즈는
재배포를 금지하며 기사에는 저작권이 있다 (스펙 §8).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    news_id  TEXT PRIMARY KEY,
    date     TEXT NOT NULL,
    outlet   TEXT,
    title    TEXT,
    body     TEXT,
    keywords TEXT,
    orgs     TEXT,
    url      TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);

CREATE TABLE IF NOT EXISTS article_codes (
    news_id    TEXT NOT NULL,
    code       TEXT NOT NULL,
    match_kind TEXT NOT NULL,
    PRIMARY KEY (news_id, code)
);
CREATE INDEX IF NOT EXISTS idx_article_codes_code ON article_codes(code);

CREATE TABLE IF NOT EXISTS daily_news (
    date             TEXT NOT NULL,
    code             TEXT NOT NULL,
    n_articles       INTEGER,
    n_outlets        INTEGER,
    article_ratio    REAL,
    adv              REAL,
    tv_ratio         REAL,
    hotness          REAL,
    "rank"           INTEGER,
    in_ops           INTEGER,
    primary_news_id  TEXT,
    PRIMARY KEY (date, code)
);
"""

_ARTICLE_COLUMNS = ["news_id", "date", "outlet", "title", "body", "keywords", "orgs", "url"]
_DAILY_COLUMNS = [
    "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
    "tv_ratio", "hotness", "rank", "in_ops", "primary_news_id",
]


def init_db(path: str | Path) -> sqlite3.Connection:
    """sqlite 파일을 열고 스키마를 보장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _date_strings(series: pd.Series) -> list[str]:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").tolist()


def upsert_articles(conn: sqlite3.Connection, articles: pd.DataFrame) -> int:
    """기사를 ``news_id`` 기준으로 덮어쓴다."""
    if articles.empty:
        return 0
    df = articles.copy()
    df["date"] = _date_strings(df["date"])
    rows = list(df[_ARTICLE_COLUMNS].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT OR REPLACE INTO articles "
        "(news_id, date, outlet, title, body, keywords, orgs, url) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_article_codes(conn: sqlite3.Connection, mapped: pd.DataFrame) -> int:
    """(기사, 종목) 매핑을 덮어쓴다."""
    if mapped.empty:
        return 0
    rows = list(
        mapped[["news_id", "code", "match_kind"]].itertuples(index=False, name=None)
    )
    conn.executemany(
        "INSERT OR REPLACE INTO article_codes (news_id, code, match_kind) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def replace_daily_news(conn: sqlite3.Connection, ranked: pd.DataFrame) -> int:
    """등장한 날짜들을 통째로 지우고 다시 넣는다.

    부분 갱신을 하지 않는 이유는 핫함 baseline 이 이후에 들어온 데이터에
    따라 달라지기 때문이다. 계산 기준이 다른 행이 섞이면 재현이 깨진다.
    """
    if ranked.empty:
        return 0
    df = ranked.copy()
    df["date"] = _date_strings(df["date"])
    df["in_ops"] = df["in_ops"].astype(int)

    dates = sorted(set(df["date"]))
    conn.executemany("DELETE FROM daily_news WHERE date = ?", [(d,) for d in dates])
    rows = list(df[_DAILY_COLUMNS].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO daily_news "
        "(date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, "
        ' hotness, "rank", in_ops, primary_news_id) '
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def load_daily_news(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    *,
    ops_only: bool = False,
) -> pd.DataFrame:
    """일별 핫함 테이블을 읽는다. ``ops_only`` 면 운영 상위 10건만."""
    sql = (
        "SELECT date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, "
        'hotness, "rank", in_ops, primary_news_id FROM daily_news '
        "WHERE date >= ? AND date <= ?"
    )
    if ops_only:
        sql += " AND in_ops = 1"
    sql += ' ORDER BY date, "rank"'
    rows = conn.execute(sql, (start, end)).fetchall()
    df = pd.DataFrame(rows, columns=_DAILY_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["in_ops"] = df["in_ops"].astype(bool)
    return df
