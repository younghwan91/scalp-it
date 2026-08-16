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


CARD_COLUMNS: list[str] = [
    "date", "code", "hotness_rank", "hotness",
    "article_id", "press", "title", "body200", "url",
]

_SELECT_CARDS = """
select d.date, d.code, d."rank", d.hotness,
       a.news_id, a.outlet, a.title, a.body, a.url
from daily_news d
left join articles a on a.news_id = d.primary_news_id
where d.date >= ? and d.date <= ?
"""


def load_article_cards(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    *,
    ops_only: bool = False,
) -> pd.DataFrame:
    """근거 카드용 — 일별 핫함에 그날의 대표 기사를 붙여 읽는다.

    ``daily_news.primary_news_id → articles.news_id`` 를 LEFT JOIN 한다.
    ``articles`` 에는 ``code`` 가 없다 — 기사 1건이 종목 N개에 매핑되므로
    종목은 ``daily_news`` 를 거쳐야 나온다. 근거 카드는 (종목, 날짜)당 대표
    기사 한 건이면 되므로 이 조인이 맞다.

    대표 기사가 없는 행도 남긴다. 핫함 수치만으로도 카드에 쓸 값이 있고,
    조인 실패로 후보가 조용히 사라지는 것이 더 나쁘다.

    집계 수치(``n_articles`` 등)는 돌려주지 않는다. ②층 피처로 후보 행에
    이미 실려 있고, 같은 값을 두 경로로 들여오면 어긋날 여지만 생긴다.

    ``rank`` 는 SQL 예약어이므로 ``"rank"`` 로 인용하고, 반환할 때
    ``hotness_rank`` 로 개명한다 (``features.news`` 와 같은 이름).

    이 모듈의 기존 ``load_articles`` 와 컬럼 계약을 맞춘다 —
    ``news_id``→``article_id``, ``outlet``→``press``, ``body``→``body200``.
    한 어댑터 안에서 두 함수가 다른 이름을 쓰면 호출부가 헷갈린다.
    ``url`` 은 ``NEWS_COLUMNS`` 에 없지만 근거 카드 링크에 필요해 여기서만 더 뽑는다.
    """
    sql = _SELECT_CARDS + (" and d.in_ops = 1" if ops_only else "")
    sql += ' order by d.date, d."rank"'

    rows = conn.execute(sql, (start, end)).fetchall()
    raw = pd.DataFrame(rows, columns=CARD_COLUMNS)
    if raw.empty:
        return pd.DataFrame(columns=CARD_COLUMNS)

    raw["date"] = pd.to_datetime(raw["date"])
    raw["code"] = raw["code"].astype(str)
    for col in ("hotness_rank", "hotness"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw[CARD_COLUMNS].reset_index(drop=True)
