import pandas as pd

from swing_it.score.prefilter import normalize_title, prefilter_articles, prefilter_summary

_BODY = "삼성전자가 미국 A사와 3년간 2조원 규모의 HBM 공급 계약을 체결했다고 공시했다."


def _articles(rows: list[dict]) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "date": "2026-01-05",
        "code": "005930",
        "press": "한국경제",
        "title": "삼성전자, HBM 대규모 공급 계약",
        "body200": _BODY,
        "keywords": "삼성전자,HBM",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _events(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [c for c, _ in pairs],
            "date": pd.to_datetime([d for _, d in pairs]),
            "adv": [6000.0] * len(pairs),
            "tv_ratio": [4.0] * len(pairs),
        }
    )


def test_normalize_title_strips_prefix_and_punctuation():
    assert normalize_title("[단독] 삼성전자, HBM 공급 계약!") == normalize_title(
        "삼성전자 HBM 공급 계약"
    )
    assert normalize_title("[속보]현대차 수주") == normalize_title("현대차 수주")


def test_clean_article_is_kept():
    out = prefilter_articles(_articles([{}]), _events([("005930", "2026-01-05")]))
    assert bool(out.iloc[0]["keep"]) is True
    assert out.iloc[0]["drop_reason"] == ""
    assert out.iloc[0]["dup_count"] == 1


def test_unmapped_code_is_dropped():
    out = prefilter_articles(
        _articles([{"code": ""}, {"article_id": "a2", "code": None}]),
        _events([("005930", "2026-01-05")]),
    )
    assert list(out["drop_reason"]) == ["no_code", "no_code"]


def test_short_article_is_dropped():
    out = prefilter_articles(
        _articles([{"title": "삼성전자", "body200": ""}]),
        _events([("005930", "2026-01-05")]),
    )
    assert out.iloc[0]["drop_reason"] == "too_short"


def test_article_without_volume_event_is_dropped():
    out = prefilter_articles(_articles([{}]), _events([("000660", "2026-01-05")]))
    assert out.iloc[0]["drop_reason"] == "no_volume_event"


def test_event_on_a_different_day_does_not_count():
    out = prefilter_articles(_articles([{}]), _events([("005930", "2026-01-06")]))
    assert out.iloc[0]["drop_reason"] == "no_volume_event"


def test_duplicate_titles_keep_only_the_first():
    out = prefilter_articles(
        _articles(
            [
                {"article_id": "a1", "press": "한국경제"},
                {
                    "article_id": "a2",
                    "press": "매일경제",
                    "title": "[단독] 삼성전자, HBM 대규모 공급 계약",
                },
                {"article_id": "a3", "press": "더벨", "title": "삼성전자 HBM 대규모 공급계약!"},
            ]
        ),
        _events([("005930", "2026-01-05")]),
    )
    assert list(out["keep"]) == [True, False, False]
    assert list(out["drop_reason"]) == ["", "duplicate", "duplicate"]
    assert list(out["dup_count"]) == [3, 3, 3]


def test_same_title_different_code_is_not_duplicate():
    out = prefilter_articles(
        _articles(
            [
                {"article_id": "a1", "code": "005930"},
                {"article_id": "a2", "code": "000660"},
            ]
        ),
        _events([("005930", "2026-01-05"), ("000660", "2026-01-05")]),
    )
    assert list(out["keep"]) == [True, True]


def test_summary_counts_every_reason():
    articles = _articles(
        [
            {"article_id": "a1"},
            {"article_id": "a2", "code": ""},
            {"article_id": "a3", "title": "짧다", "body200": ""},
            {"article_id": "a4", "code": "000660"},
            {"article_id": "a5"},
        ]
    )
    out = prefilter_articles(articles, _events([("005930", "2026-01-05")]))
    summary = prefilter_summary(out)
    assert summary == {
        "n_input": 5,
        "n_kept": 1,
        "no_code": 1,
        "too_short": 1,
        "no_volume_event": 1,
        "duplicate": 1,
    }


def test_input_row_order_is_preserved():
    articles = _articles([{"article_id": f"a{i}"} for i in range(5)])
    out = prefilter_articles(articles, _events([("005930", "2026-01-05")]))
    assert list(out["article_id"]) == ["a0", "a1", "a2", "a3", "a4"]
