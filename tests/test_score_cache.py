import pandas as pd

from scalp_it.score.cache import get_cached, load_features_frame, open_cache, put_result
from scalp_it.score.schema import LLM_FEATURE_COLUMNS, ArticleFeatures

_F = ArticleFeatures(
    material_type="수주",
    persistence="structural",
    specificity=3,
    freshness="new",
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)


def _conn(tmp_path):
    return open_cache(tmp_path / "llm.db")


def test_open_cache_is_idempotent(tmp_path):
    path = tmp_path / "llm.db"
    open_cache(path).close()
    conn = open_cache(path)
    assert conn.execute("select count(*) from llm_features").fetchone()[0] == 0


def test_put_and_get_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw='{"x":1}', error=None)
    cached = get_cached(conn, ["a1"], "m1", "v1")
    assert cached == {"a1": _F}


def test_error_rows_are_not_cache_hits(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=None, raw=None, error="타임아웃")
    assert get_cached(conn, ["a1"], "m1", "v1") == {}


def test_cache_is_keyed_by_model_and_prompt_version(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="cheap", prompt_version="v1",
               features=_F, raw=None, error=None)
    assert get_cached(conn, ["a1"], "cheap", "v1") == {"a1": _F}
    assert get_cached(conn, ["a1"], "premium", "v1") == {}
    assert get_cached(conn, ["a1"], "cheap", "v2") == {}


def test_put_overwrites_same_key(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=None, raw=None, error="1차 실패")
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw=None, error=None)
    assert get_cached(conn, ["a1"], "m1", "v1") == {"a1": _F}
    assert conn.execute("select count(*) from llm_features").fetchone()[0] == 1


def test_get_cached_handles_large_id_lists(tmp_path):
    """sqlite 변수 한도(999)를 넘는 조회도 나뉘어 처리돼야 한다."""
    conn = _conn(tmp_path)
    ids = [f"a{i}" for i in range(2500)]
    for article_id in ids:
        put_result(conn, article_id=article_id, model="m1", prompt_version="v1",
                   features=_F, raw=None, error=None)
    assert len(get_cached(conn, ids, "m1", "v1")) == 2500


def test_load_features_frame_shape(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw=None, error=None)
    put_result(conn, article_id="a2", model="m1", prompt_version="v1",
               features=None, raw=None, error="실패")
    frame = load_features_frame(conn, "m1", "v1")
    assert list(frame.columns) == [
        "article_id", "material_type", "persistence", "specificity", "freshness",
        "sentiment_direction", "sentiment_strength", "scale_vs_size",
        *LLM_FEATURE_COLUMNS,
    ]
    assert list(frame["article_id"]) == ["a1"]  # 실패 행은 빠진다
    assert frame.iloc[0]["llm_sentiment"] == 3.0
    assert pd.api.types.is_numeric_dtype(frame["llm_specificity"])


def test_load_features_frame_is_empty_when_nothing_matches(tmp_path):
    frame = load_features_frame(_conn(tmp_path), "m1", "v1")
    assert frame.empty
    assert "llm_sentiment" in frame.columns
