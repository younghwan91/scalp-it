import pandas as pd
import pytest

from scalp_it.score.batch import BatchAborted, score_articles
from scalp_it.score.cache import get_cached, open_cache
from scalp_it.score.prompt import PROMPT_VERSION
from scalp_it.score.provider import FakeProvider, ProviderError

_OK = (
    '{"material_type":"수주","persistence":"multi_quarter","specificity":3,'
    '"freshness":"new","sentiment_direction":1,"sentiment_strength":2,'
    '"scale_vs_size":"medium"}'
)


def _articles(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(n)],
            "date": ["2026-01-05"] * n,
            "code": ["005930"] * n,
            "press": ["한국경제"] * n,
            "title": [f"삼성전자 공급 계약 {i}" for i in range(n)],
            "body200": ["2조원 규모 계약을 체결했다."] * n,
            "keywords": ["삼성전자,수주"] * n,
        }
    )


def _no_sleep(_seconds: float) -> None:
    return None


def test_all_fresh_articles_are_called_once(tmp_path):
    provider = FakeProvider(responses=[_OK] * 3)
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(3), provider, cache, sleep=_no_sleep)
    assert (result.n_input, result.n_cached, result.n_called, result.n_ok) == (3, 0, 3, 3)
    assert result.n_failed == 0
    assert len(provider.calls) == 3


def test_second_run_hits_the_cache(tmp_path):
    cache = open_cache(tmp_path / "llm.db")
    score_articles(_articles(3), FakeProvider(responses=[_OK] * 3), cache, sleep=_no_sleep)

    provider = FakeProvider(responses=[_OK] * 3)
    result = score_articles(_articles(3), provider, cache, sleep=_no_sleep)
    assert (result.n_cached, result.n_called) == (3, 0)
    assert provider.calls == []


def test_transient_failure_is_retried_then_succeeds(tmp_path):
    provider = FakeProvider(responses=[ProviderError("일시 오류"), _OK])
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(1), provider, cache, max_retries=2, sleep=_no_sleep)
    assert (result.n_ok, result.n_failed) == (1, 0)
    assert len(provider.calls) == 2


def test_schema_violation_is_retried(tmp_path):
    provider = FakeProvider(responses=['{"material_type":"미상"}', _OK])
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(1), provider, cache, max_retries=2, sleep=_no_sleep)
    assert result.n_ok == 1
    assert len(provider.calls) == 2


def test_permanent_failure_is_recorded_and_batch_continues(tmp_path):
    provider = FakeProvider(
        responses=[
            ProviderError("깨짐"), ProviderError("깨짐"), ProviderError("깨짐"),
            _OK, _OK,
        ]
    )
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(3), provider, cache, max_retries=2, sleep=_no_sleep)
    assert (result.n_ok, result.n_failed) == (2, 1)
    assert result.failures[0][0] == "a0"
    assert "깨짐" in result.failures[0][1]
    # 실패한 기사는 캐시 히트가 아니므로 다음 실행에서 재시도된다
    assert get_cached(cache, ["a0"], provider.model, PROMPT_VERSION) == {}
    assert set(get_cached(cache, ["a1", "a2"], provider.model, PROMPT_VERSION)) == {"a1", "a2"}


def test_backoff_grows_between_retries(tmp_path):
    slept: list[float] = []
    provider = FakeProvider(responses=[ProviderError("x"), ProviderError("x"), _OK])
    cache = open_cache(tmp_path / "llm.db")
    score_articles(
        _articles(1), provider, cache, max_retries=2, backoff=2.0, sleep=slept.append
    )
    assert slept == [2.0, 4.0]


def test_limit_caps_api_calls(tmp_path):
    provider = FakeProvider(responses=[_OK] * 10)
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(10), provider, cache, limit=4, sleep=_no_sleep)
    assert result.n_called == 4
    assert len(provider.calls) == 4


def test_consecutive_failures_abort_the_batch(tmp_path):
    provider = FakeProvider(responses=[ProviderError("키 만료")] * 100)
    cache = open_cache(tmp_path / "llm.db")
    with pytest.raises(BatchAborted, match="연속"):
        score_articles(
            _articles(50), provider, cache, max_retries=0,
            abort_after_consecutive=3, sleep=_no_sleep,
        )


def test_progress_callback_reports_each_article(tmp_path):
    seen: list[tuple[int, int]] = []
    cache = open_cache(tmp_path / "llm.db")
    score_articles(
        _articles(3), FakeProvider(responses=[_OK] * 3), cache,
        sleep=_no_sleep, on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_empty_input_is_a_no_op(tmp_path):
    provider = FakeProvider()
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(0), provider, cache, sleep=_no_sleep)
    assert (result.n_input, result.n_called, result.n_ok) == (0, 0, 0)
    assert provider.calls == []
