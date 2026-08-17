"""규칙 기반 뉴스 피처 — LLM 없이 기사에서 뽑는 것 (스펙 §3 ②층).

계획 2 의 ``daily_news`` 를 모델이 먹을 수 있는 형태로 고르고 이름만 바꾼다.
**계산은 거의 하지 않는다.** 60일 baseline 롤링과 shift(1) 은 계획 2 의
``add_hotness`` 가 이미 끝냈다. 여기서 또 롤링하면 이중 계산이고
point-in-time 보증도 흐려진다. 그래서 ``window`` 류 파라미터가 없다.

빠뜨린 컬럼이 둘 있고, 둘 다 의도적이다:

- ``tv_ratio``·``adv`` — 계획 1 ``FEATURE_COLUMNS`` 에 이미 들어 있다. ②층에
  다시 넣으면 "뉴스 피처가 기여했다"가 사실은 "거래대금을 두 번 넣었다"일 수
  있고, 층별 기여도 분리(스펙 §3)가 무너진다.
- ``in_ops``·``primary_news_id`` — 운영 표시와 조인 키다. 피처가 아니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_FEATURE_COLUMNS: list[str] = [
    "article_count",
    "outlet_count",
    "article_ratio",
    "hotness",
    "hotness_rank",
]

# daily_news 원본 컬럼 → 피처 이름.
# rank 는 pandas 메서드·SQL 예약어와 충돌하고, n_* 은 카드에 찍혔을 때 뜻이 덜 통한다.
_SOURCE = {
    "article_count": "n_articles",
    "outlet_count": "n_outlets",
    "article_ratio": "article_ratio",
    "hotness": "hotness",
    "hotness_rank": "rank",
}


def news_features(daily_news: pd.DataFrame) -> pd.DataFrame:
    """``load_daily_news`` 결과를 ``[code, date, *NEWS_FEATURE_COLUMNS]`` 로 바꾼다."""
    required = ["code", "date", *_SOURCE.values()]
    missing = [col for col in required if col not in daily_news.columns]
    if missing:
        raise ValueError(
            f"daily_news 에 컬럼이 없다: {missing}. 계획 2 의 load_daily_news 출력을 넘겨라."
        )

    out = pd.DataFrame(
        {
            "code": daily_news["code"].astype(str),
            "date": pd.to_datetime(daily_news["date"]),
        }
    )
    for feature, source in _SOURCE.items():
        out[feature] = pd.to_numeric(daily_news[source], errors="coerce").astype(float)
    return out[["code", "date", *NEWS_FEATURE_COLUMNS]].reset_index(drop=True)


def load_news_features(
    path: str | Path = "data/news.db",
    *,
    start: str = "1900-01-01",
    end: str = "2999-12-31",
) -> pd.DataFrame:
    """계획 2 의 sqlite 를 열어 뉴스 피처를 만든다.

    ``adapters.news_db.connect_news`` 를 재사용한다. ``init_db`` 를 부르면 파일이
    없을 때 빈 스키마를 만들어 조용히 0행을 돌려주고, 뉴스 층이 통째로 빠진 채로
    비교가 돌아가고도 아무도 모른다. ``connect_news`` 는 없으면 즉시 실패한다.
    """
    from ..adapters.news_db import connect_news
    from ..news.store import load_daily_news

    conn = connect_news(str(path))
    try:
        return news_features(load_daily_news(conn, start, end))
    finally:
        conn.close()
