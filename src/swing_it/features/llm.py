"""LLM 피처를 ``(code, date)`` 격자로 접는 어댑터.

계획 3 의 캐시는 기사 단위이고 모델은 ``(code, date)`` 단위로 학습한다.
그 사이를 잇는 것이 이 파일이며, 계획 4(층별 모델 비교)가 계획 3 에 닿는
유일한 지점이다.

접기 규칙은 "그날 가장 강한 재료 한 건의 행을 통째로 쓴다"이다. 컬럼별로 max 를
따로 집계하면 A기사의 구체성과 B기사의 감정이 섞인, 실재하지 않는 기사가 만들어진다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..adapters.news_db import connect_news, load_articles
from ..score.cache import CACHE_TABLE, load_features_frame
from ..score.prompt import PROMPT_VERSION
from ..score.schema import LLM_FEATURE_COLUMNS

__all__ = [
    "LLM_FEATURE_COLUMNS",
    "LLM_FRAME_COLUMNS",
    "available_models",
    "fold_to_code_date",
    "load_llm_features",
]

# material_type 은 모델 행렬에 넣지 않는다 (9개 원-핫이면 스펙 §6 상한을 혼자 먹는다).
# 한국어 문자열 그대로 근거 카드 표시용으로 딸려 보낸다.
LLM_FRAME_COLUMNS: list[str] = [
    "code", "date", "material_type", "n_articles", *LLM_FEATURE_COLUMNS,
]

_SORT_KEYS = ["code", "date", "_abs_sentiment", "llm_specificity", "article_id"]
_SORT_ORDER = [True, True, False, False, True]


def available_models(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """캐시에 들어 있는 ``(model, prompt_version, 성공 건수)`` 를 건수 내림차순으로."""
    rows = conn.execute(
        f"select model, prompt_version, count(*) from {CACHE_TABLE} "
        "where status = 'ok' group by model, prompt_version "
        "order by count(*) desc, model asc"
    ).fetchall()
    return [(str(model), str(version), int(count)) for model, version, count in rows]


def fold_to_code_date(scores: pd.DataFrame) -> pd.DataFrame:
    """같은 ``(code, date)`` 의 여러 기사를 대표 한 건으로 접는다.

    우선순위: ``|llm_sentiment|`` 최대 → ``llm_specificity`` 최대 →
    ``article_id`` 사전순 최소. 마지막 단계는 재현성을 위한 것이다.
    """
    if scores.empty:
        return pd.DataFrame(columns=LLM_FRAME_COLUMNS)

    work = scores.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["_abs_sentiment"] = work["llm_sentiment"].abs()
    work["n_articles"] = work.groupby(["code", "date"], sort=False)["article_id"].transform("size")

    ordered = work.sort_values(_SORT_KEYS, ascending=_SORT_ORDER, kind="mergesort")
    picked = ordered.groupby(["code", "date"], sort=True, as_index=False).head(1)
    return picked[LLM_FRAME_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _resolve_version(
    conn: sqlite3.Connection, model: str | None, prompt_version: str | None
) -> tuple[str, str]:
    """모델·프롬프트 버전이 생략되면 캐시에서 유일한 조합을 찾는다.

    여러 조합이 섞여 있으면 **고르지 않고 실패한다.** 저가 모델과 상위 모델의
    결과가 한 파일에 있는 게 정상 상태이고(계획 3 Task 7 대조), 아무거나 집으면
    어떤 모델로 학습했는지 모르는 채 결과가 나온다.
    """
    if model is not None:
        return model, prompt_version or PROMPT_VERSION
    combos = available_models(conn)
    if not combos:
        raise ValueError("LLM 피처 캐시가 비어 있다. swing-score 를 먼저 돌려라.")
    if prompt_version is not None:
        combos = [c for c in combos if c[1] == prompt_version]
    if len(combos) > 1:
        listed = ", ".join(f"{m}@{v}({n}건)" for m, v, n in combos)
        raise ValueError(
            f"캐시에 여러 모델이 있다: {listed}. model= 로 명시하라."
        )
    return combos[0][0], combos[0][1]


def load_llm_features(
    cache_path: str | Path = "data/llm_features.db",
    *,
    news_path: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    articles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """LLM 피처를 ``(code, date)`` 격자 프레임으로 읽는다.

    Args:
        cache_path: 계획 3 이 만든 sqlite 캐시. 기본 ``data/llm_features.db``.
        news_path: ``article_id → (code, date)`` 조인용 뉴스 DB. 생략 시
            ``SWING_NEWS_DB``, 그것도 없으면 ``data/news.db``.
        model / prompt_version: 생략하면 캐시에 조합이 하나뿐일 때만 자동 선택한다.
        articles: 이미 읽어둔 기사 프레임(``article_id, date, code``)이 있으면
            넘긴다. news DB 를 다시 열지 않는다.

    Returns:
        컬럼 ``LLM_FRAME_COLUMNS``. ``date`` 는 ``datetime64[ns]``,
        ``LLM_FEATURE_COLUMNS`` 는 전부 수치형.
    """
    conn = sqlite3.connect(Path(cache_path))
    try:
        resolved_model, resolved_version = _resolve_version(conn, model, prompt_version)
        features = load_features_frame(conn, resolved_model, resolved_version)
    finally:
        conn.close()

    if articles is None:
        articles = load_articles(connect_news(news_path), "0001-01-01", "9999-12-31")
    keys = articles[["article_id", "code", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])

    joined = features.merge(keys, on="article_id", how="inner")
    joined = joined[joined["code"].astype(str).str.strip().ne("")]
    return fold_to_code_date(
        joined[["article_id", "code", "date", "material_type", *LLM_FEATURE_COLUMNS]]
    )
