import sqlite3

import pandas as pd
import pytest

from swing_it.features.llm import (
    LLM_FEATURE_COLUMNS,
    LLM_FRAME_COLUMNS,
    available_models,
    fold_to_code_date,
    load_llm_features,
)
from swing_it.score.cache import open_cache, put_result
from swing_it.score.schema import ArticleFeatures


def _scores(rows: list[dict]) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "code": "005930",
        "date": pd.Timestamp("2026-01-05"),
        "material_type": "수주",
        "llm_persistence": 1.0,
        "llm_specificity": 2.0,
        "llm_freshness": 2.0,
        "llm_sentiment": 2.0,
        "llm_scale": 2.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_fold_output_contract():
    out = fold_to_code_date(_scores([{}]))
    assert list(out.columns) == LLM_FRAME_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    for column in LLM_FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(out[column])


def test_single_article_passes_through():
    out = fold_to_code_date(_scores([{}]))
    assert len(out) == 1
    assert out.iloc[0]["n_articles"] == 1
    assert out.iloc[0]["llm_sentiment"] == 2.0
    assert out.iloc[0]["material_type"] == "수주"


def test_strongest_absolute_sentiment_wins():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 1.0, "material_type": "테마"},
                {"article_id": "a2", "llm_sentiment": -3.0, "material_type": "루머"},
                {"article_id": "a3", "llm_sentiment": 2.0, "material_type": "수주"},
            ]
        )
    )
    assert len(out) == 1
    assert out.iloc[0]["llm_sentiment"] == -3.0
    assert out.iloc[0]["material_type"] == "루머"
    assert out.iloc[0]["n_articles"] == 3


def test_specificity_breaks_sentiment_ties():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 2.0, "llm_specificity": 1.0},
                {"article_id": "a2", "llm_sentiment": 2.0, "llm_specificity": 3.0},
            ]
        )
    )
    assert out.iloc[0]["llm_specificity"] == 3.0


def test_article_id_breaks_remaining_ties_deterministically():
    rows = [
        {"article_id": "z9", "material_type": "정책"},
        {"article_id": "a1", "material_type": "증설"},
    ]
    assert fold_to_code_date(_scores(rows)).iloc[0]["material_type"] == "증설"
    assert fold_to_code_date(_scores(rows[::-1])).iloc[0]["material_type"] == "증설"


def test_fold_keeps_the_whole_row_not_a_column_wise_max():
    """컬럼별 max 를 섞으면 실재하지 않는 기사가 만들어진다."""
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 3.0, "llm_scale": 0.0},
                {"article_id": "a2", "llm_sentiment": 1.0, "llm_scale": 3.0},
            ]
        )
    )
    assert (out.iloc[0]["llm_sentiment"], out.iloc[0]["llm_scale"]) == (3.0, 0.0)


def test_separate_code_date_pairs_are_independent():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "code": "005930"},
                {"article_id": "a2", "code": "000660"},
                {"article_id": "a3", "code": "005930", "date": pd.Timestamp("2026-01-06")},
            ]
        )
    )
    assert len(out) == 3
    assert list(out["n_articles"]) == [1, 1, 1]


def test_empty_input_returns_empty_contract_frame():
    out = fold_to_code_date(_scores([]).iloc[0:0])
    assert out.empty
    assert list(out.columns) == LLM_FRAME_COLUMNS


# --- load_llm_features (sqlite fixture, 네트워크·API 없음) -------------------

_F = ArticleFeatures(
    material_type="수주",
    persistence="structural",
    specificity=3,
    freshness="new",
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)


@pytest.fixture()
def workspace(tmp_path):
    """계획 2 스키마 그대로 — 정규화된 articles + article_codes."""
    news_path = tmp_path / "news.db"
    conn = sqlite3.connect(news_path)
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
            ('n1','2026-01-05','한국경제','제목1','본문1','키1','삼성전자','http://x/1'),
            ('n2','2026-01-05','매일경제','제목2','본문2','키2','SK하이닉스','http://x/2');
        INSERT INTO article_codes VALUES
            ('n1','005930','name'),
            ('n2','000660','name');
        """
    )
    conn.commit()
    conn.close()

    cache_path = tmp_path / "llm.db"
    cache = open_cache(cache_path)
    for article_id in ("n1", "n2"):
        put_result(cache, article_id=article_id, model="cheap", prompt_version="v1",
                   features=_F, raw=None, error=None)
    cache.close()
    return {"news": str(news_path), "cache": str(cache_path)}


def test_load_llm_features_end_to_end(workspace):
    frame = load_llm_features(workspace["cache"], news_path=workspace["news"])
    assert list(frame.columns) == LLM_FRAME_COLUMNS
    assert sorted(frame["code"]) == ["000660", "005930"]
    assert frame["llm_sentiment"].tolist() == [3.0, 3.0]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_load_llm_features_auto_detects_the_only_model(workspace):
    """모델이 하나뿐이면 인자 없이도 찾아낸다."""
    assert len(load_llm_features(workspace["cache"], news_path=workspace["news"])) == 2


def test_load_llm_features_refuses_to_guess_between_models(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n1", model="premium", prompt_version="v1",
               features=_F, raw=None, error=None)
    cache.close()
    with pytest.raises(ValueError, match="cheap"):
        load_llm_features(workspace["cache"], news_path=workspace["news"])


def test_load_llm_features_honours_explicit_model(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n1", model="premium", prompt_version="v1",
               features=_F, raw=None, error=None)
    cache.close()
    frame = load_llm_features(
        workspace["cache"], news_path=workspace["news"],
        model="premium", prompt_version="v1",
    )
    assert list(frame["code"]) == ["005930"]


def test_load_llm_features_accepts_an_articles_frame(workspace):
    """계획 4 가 이미 기사 프레임을 갖고 있으면 news.db 를 다시 열지 않는다."""
    articles = pd.DataFrame(
        {
            "article_id": ["n1"],
            "date": pd.to_datetime(["2026-01-05"]),
            "code": ["005930"],
        }
    )
    frame = load_llm_features(workspace["cache"], articles=articles)
    assert list(frame["code"]) == ["005930"]


def test_available_models_counts_only_success_rows(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n3", model="cheap", prompt_version="v1",
               features=None, raw=None, error="실패")
    assert available_models(cache) == [("cheap", "v1", 2)]


def test_empty_cache_raises_a_useful_message(tmp_path):
    open_cache(tmp_path / "empty.db").close()
    with pytest.raises(ValueError, match="비어 있다"):
        load_llm_features(tmp_path / "empty.db", articles=pd.DataFrame())
