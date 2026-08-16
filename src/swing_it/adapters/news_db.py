"""뉴스 DB 읽기 어댑터 — 뉴스 SQL 은 이 파일에만 존재한다.

입력은 계획 2 가 만드는 로컬 sqlite ``data/news.db`` 다. 계획 2 는 기사를
``articles`` 에, 기사→종목 매핑을 ``article_codes`` 에 정규화해 넣는다(기사 1건이
여러 종목에 걸릴 수 있기 때문이다). 이 어댑터가 조인과 컬럼명 차이를 흡수하고,
나머지 모듈은 평면화된 ``NEWS_COLUMNS`` 만 안다.

계획 2 컬럼 → 이 계약: ``news_id``→``article_id``, ``outlet``→``press``,
``body``→``body200``(계획 2 가 이미 200자로 잘라 넣는다), ``article_codes.code``→``code``.
같은 기사가 두 종목에 걸리면 종목마다 한 행이 된다 — 스코어링 단위가 (기사, 종목) 이므로 의도된 동작이다.

읽기 전용이다. swing-it 의 score 계층은 news.db 에 쓰지 않는다.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

NEWS_COLUMNS: list[str] = [
    "article_id", "date", "code", "press", "title", "body200", "keywords",
]

_DEFAULT_PATH = "data/news.db"


def connect_news(path: str | None = None) -> sqlite3.Connection:
    """뉴스 DB 를 연다. 생략 시 ``SWING_NEWS_DB``, 그것도 없으면 ``data/news.db``."""
    target = path or os.environ.get("SWING_NEWS_DB") or _DEFAULT_PATH
    if not Path(target).exists():
        raise FileNotFoundError(
            f"뉴스 DB 가 없다: {target}. 계획 2(뉴스 인제스트)를 먼저 돌려야 한다."
        )
    return sqlite3.connect(target)


_SELECT_ARTICLES = """
select a.news_id, a.date, ac.code, a.outlet, a.title, a.body, a.keywords
from articles a
join article_codes ac on ac.news_id = a.news_id
where a.date >= ? and a.date <= ?
order by a.date, a.news_id, ac.code
"""


def load_articles(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    """기간 내 (기사, 종목) 을 컬럼 계약에 맞춰 읽는다.

    계획 2 의 정규화 스키마를 조인해 평면화한다. 종목 매핑이 없는 기사는
    조인에서 빠진다 — 스코어링 단위가 (기사, 종목) 이므로 종목 없는 기사는
    애초에 태울 대상이 아니다.
    """
    rows = conn.execute(_SELECT_ARTICLES, (start, end)).fetchall()
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("article_id", "code", "press", "title", "body200", "keywords"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame
