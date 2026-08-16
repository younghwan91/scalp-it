"""배치 러너 — 캐시 스킵 · 재시도 · 부분 실패 집계.

수천 건 배치에서 일부 실패는 정상이다. 실패 하나로 전체를 죽이지 않고, 사유를
캐시에 남겨 다음 실행에서 자동 재시도되게 한다.

다만 연속 실패가 임계치를 넘으면 멈춘다. 키 만료나 모델 ID 오타로 수천 건을
헛돌리는 것과 진짜 부분 실패는 다른 일이다.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .cache import get_cached, put_result
from .prompt import PROMPT_VERSION, build_prompt
from .provider import LLMProvider, ProviderError
from .schema import SchemaError, parse_features


class BatchAborted(RuntimeError):
    """연속 실패가 임계치를 넘었다. 설정 문제일 가능성이 높다."""


@dataclass(frozen=True)
class BatchResult:
    n_input: int
    n_cached: int
    n_called: int
    n_ok: int
    n_failed: int
    failures: list[tuple[str, str]] = field(default_factory=list)


def score_articles(
    articles: pd.DataFrame,
    provider: LLMProvider,
    cache: sqlite3.Connection,
    *,
    prompt_version: str = PROMPT_VERSION,
    max_retries: int = 2,
    backoff: float = 1.0,
    limit: int | None = None,
    abort_after_consecutive: int = 20,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """기사 프레임을 배치로 태워 캐시에 채운다.

    Args:
        limit: **API 호출 건수** 상한. 비용 사고를 막는 안전핀이므로 캐시
            히트는 세지 않는다. ``None`` 이면 무제한.
        max_retries: 기사 한 건당 추가 시도 횟수. 총 시도는 ``max_retries + 1``.
        backoff: 재시도 대기 초. n 번째 재시도는 ``backoff * 2**n`` 초 쉰다.
        sleep: 주입 가능한 대기 함수. 테스트는 no-op 을 넘긴다.
    """
    total = len(articles)
    if total == 0:
        return BatchResult(0, 0, 0, 0, 0, [])

    ids = articles["article_id"].astype(str).tolist()
    cached = get_cached(cache, ids, provider.model, prompt_version)

    n_called = n_ok = n_failed = 0
    consecutive = 0
    failures: list[tuple[str, str]] = []

    for done, (_, row) in enumerate(articles.iterrows(), start=1):
        article_id = str(row["article_id"])
        if article_id in cached:
            if on_progress:
                on_progress(done, total)
            continue
        if limit is not None and n_called >= limit:
            break

        system, user = build_prompt(row.to_dict())
        last_error = ""
        raw: str | None = None
        for attempt in range(max_retries + 1):
            n_called += 1
            try:
                raw = provider.complete(system, user)
                features = parse_features(raw)
            except (ProviderError, SchemaError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    sleep(backoff * (2**attempt))
                continue
            put_result(
                cache,
                article_id=article_id,
                model=provider.model,
                prompt_version=prompt_version,
                features=features,
                raw=raw,
                error=None,
            )
            n_ok += 1
            consecutive = 0
            break
        else:
            put_result(
                cache,
                article_id=article_id,
                model=provider.model,
                prompt_version=prompt_version,
                features=None,
                raw=raw,
                error=last_error,
            )
            n_failed += 1
            failures.append((article_id, last_error))
            consecutive += 1
            if consecutive >= abort_after_consecutive:
                raise BatchAborted(
                    f"연속 {consecutive} 건 실패로 중단한다. 마지막 사유: {last_error}"
                )

        if on_progress:
            on_progress(done, total)

    return BatchResult(
        n_input=total,
        n_cached=len(cached),
        n_called=n_called,
        n_ok=n_ok,
        n_failed=n_failed,
        failures=failures,
    )
