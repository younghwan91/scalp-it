import sqlite3

import pandas as pd
import pytest

from scalp_it.adapters.news_db import NEWS_COLUMNS, connect_news, load_articles


@pytest.fixture()
def news_db(tmp_path):
    """계획 2 가 만드는 스키마 그대로 — 정규화된 articles + article_codes."""
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE articles (
            news_id  TEXT PRIMARY KEY,
            date     TEXT NOT NULL,
            outlet   TEXT,
            title    TEXT,
            body     TEXT,
            keywords TEXT,
            orgs     TEXT,
            url      TEXT
        );
        CREATE TABLE article_codes (
            news_id    TEXT NOT NULL,
            code       TEXT NOT NULL,
            match_kind TEXT NOT NULL,
            PRIMARY KEY (news_id, code)
        );
        INSERT INTO articles VALUES
            ('bk-1','2026-01-05','한국경제','삼성전자 HBM 공급 계약',
             '2조원 규모 계약을 체결했다.','삼성전자,HBM','삼성전자','http://x/1'),
            ('bk-2','2026-01-06','매일경제','SK하이닉스 증설 발표',
             '청주 M15X 증설을 결정했다.','SK하이닉스,증설','SK하이닉스','http://x/2'),
            ('bk-3','2026-02-01','더벨','삼성전자 후속 보도',
             '앞선 계약의 진행 상황이다.','삼성전자','삼성전자','http://x/3');
        INSERT INTO article_codes VALUES
            ('bk-1','005930','name'),
            ('bk-2','000660','name'),
            ('bk-3','005930','name');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_load_articles_returns_contract_columns(news_db):
    frame = load_articles(connect_news(news_db), "2026-01-01", "2026-12-31")
    assert list(frame.columns) == NEWS_COLUMNS
    assert len(frame) == 3
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_load_articles_respects_date_range(news_db):
    frame = load_articles(connect_news(news_db), "2026-01-01", "2026-01-31")
    assert list(frame["article_id"]) == ["bk-1", "bk-2"]


def test_connect_news_uses_env(news_db, monkeypatch):
    monkeypatch.setenv("SWING_NEWS_DB", news_db)
    frame = load_articles(connect_news(), "2026-01-01", "2026-12-31")
    assert len(frame) == 3
