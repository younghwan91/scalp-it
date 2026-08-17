"""LLM 결과 캐시 — 같은 기사를 두 번 태우지 않는다.

캐시 키는 ``(article_id, model, prompt_version)`` 이다. 모델이나 프롬프트가
바뀌면 다른 결과이므로 다른 행이어야 하고, 결과 테이블에 둘 다 적혀 있어야
"이 피처가 무엇으로 뽑혔는지"를 나중에 재현할 수 있다 (스펙 §11).

실패도 저장하되 캐시 히트로 치지 않는다 — 리포트에는 남고, 다음 실행에서는
자동으로 재시도된다.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schema import LLM_FEATURE_COLUMNS, ArticleFeatures, encode_features

CACHE_TABLE = "llm_features"

_SCHEMA_COLUMNS = [
    "material_type", "persistence", "specificity", "freshness",
    "sentiment_direction", "sentiment_strength", "scale_vs_size",
]

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
    article_id          TEXT    NOT NULL,
    model               TEXT    NOT NULL,
    prompt_version      TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    error               TEXT,
    raw_json            TEXT,
    material_type       TEXT,
    persistence         TEXT,
    specificity         INTEGER,
    freshness           TEXT,
    sentiment_direction INTEGER,
    sentiment_strength  INTEGER,
    scale_vs_size       TEXT,
    PRIMARY KEY (article_id, model, prompt_version)
);
"""

_SQLITE_VARIABLE_LIMIT = 900  # 기본 한도 999 보다 여유를 둔다


def open_cache(path: str | Path) -> sqlite3.Connection:
    """캐시 DB 를 열고 없으면 테이블을 만든다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_CREATE)
    conn.commit()
    return conn


def get_cached(
    conn: sqlite3.Connection,
    article_ids: Sequence[str],
    model: str,
    prompt_version: str,
) -> dict[str, ArticleFeatures]:
    """이미 성공적으로 뽑아둔 기사의 피처를 돌려준다."""
    found: dict[str, ArticleFeatures] = {}
    ids = list(article_ids)
    for start in range(0, len(ids), _SQLITE_VARIABLE_LIMIT):
        chunk = ids[start : start + _SQLITE_VARIABLE_LIMIT]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"select article_id, {', '.join(_SCHEMA_COLUMNS)} from {CACHE_TABLE} "
            f"where model = ? and prompt_version = ? and status = 'ok' "
            f"and article_id in ({marks})",
            (model, prompt_version, *chunk),
        ).fetchall()
        for row in rows:
            found[row[0]] = ArticleFeatures(*row[1:])
    return found


def put_result(
    conn: sqlite3.Connection,
    *,
    article_id: str,
    model: str,
    prompt_version: str,
    features: ArticleFeatures | None,
    raw: str | None,
    error: str | None,
) -> None:
    """성공이든 실패든 한 건을 기록한다. 같은 키가 있으면 덮어쓴다."""
    status = "ok" if features is not None else "error"
    values = (
        [getattr(features, name) for name in _SCHEMA_COLUMNS]
        if features is not None
        else [None] * len(_SCHEMA_COLUMNS)
    )
    conn.execute(
        f"insert or replace into {CACHE_TABLE} "
        f"(article_id, model, prompt_version, created_at, status, error, raw_json, "
        f"{', '.join(_SCHEMA_COLUMNS)}) "
        f"values (?, ?, ?, ?, ?, ?, ?, {', '.join('?' * len(_SCHEMA_COLUMNS))})",
        (
            article_id,
            model,
            prompt_version,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status,
            error,
            raw,
            *values,
        ),
    )
    conn.commit()


def load_features_frame(
    conn: sqlite3.Connection, model: str, prompt_version: str
) -> pd.DataFrame:
    """성공 행을 모델 입력 가능한 DataFrame 으로 꺼낸다."""
    columns = ["article_id", *_SCHEMA_COLUMNS]
    rows = conn.execute(
        f"select {', '.join(columns)} from {CACHE_TABLE} "
        f"where model = ? and prompt_version = ? and status = 'ok' order by article_id",
        (model, prompt_version),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=columns)

    if frame.empty:
        for column in LLM_FEATURE_COLUMNS:
            frame[column] = pd.Series(dtype="float64")
        return frame

    encoded = pd.DataFrame(
        [
            encode_features(ArticleFeatures(*row[1:]))
            for row in rows
        ],
        index=frame.index,
    )
    return pd.concat([frame, encoded[LLM_FEATURE_COLUMNS]], axis=1)
