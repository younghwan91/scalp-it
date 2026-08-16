import pandas as pd

from swing_it.news.dedupe import cluster_duplicates, jaccard, normalize_title, title_tokens


def _mapped(rows: list[tuple[str, str, str]], date="2026-08-17", code="005930"):
    return pd.DataFrame(
        [
            {
                "news_id": news_id,
                "date": pd.Timestamp(date),
                "code": code,
                "match_kind": "name",
                "outlet": outlet,
                "title": title,
            }
            for news_id, outlet, title in rows
        ]
    )


def test_normalize_title_drops_bracket_prefix():
    assert normalize_title("[특징주] 삼성전자, 3% 상승") == "삼성전자 3 상승"


def test_title_tokens_drop_one_char_tokens():
    assert title_tokens("삼성전자 3% 상승") == {"삼성전자", "상승"}


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_with_empty_set_is_zero():
    assert jaccard(set(), {"a"}) == 0.0


def test_jaccard_of_identical_sets_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_same_event_from_two_outlets_forms_one_cluster():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 미국 반도체 공장 증설 발표한다"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out["cluster_id"].nunique() == 1
    assert out["is_primary"].sum() == 1


def test_different_events_stay_separate():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 배당 정책 변경 검토"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out["cluster_id"].nunique() == 2


def test_clusters_do_not_cross_codes():
    df = pd.concat([
        _mapped([("n1", "한국경제", "실적 개선 전망")], code="005930"),
        _mapped([("n2", "매일경제", "실적 개선 전망")], code="000660"),
    ], ignore_index=True)
    out = cluster_duplicates(df, threshold=0.6)
    assert len(out) == 2
    assert out["is_primary"].sum() == 2  # 종목이 다르면 각자 대표


def test_clusters_do_not_cross_dates():
    df = pd.concat([
        _mapped([("n1", "한국경제", "실적 개선 전망")], date="2026-08-17"),
        _mapped([("n2", "매일경제", "실적 개선 전망")], date="2026-08-18"),
    ], ignore_index=True)
    out = cluster_duplicates(df, threshold=0.6)
    assert out["is_primary"].sum() == 2


def test_primary_is_lowest_news_id_in_cluster():
    df = _mapped([
        ("n9", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n1", "매일경제", "삼성전자 미국 반도체 공장 증설 발표"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out[out["is_primary"]]["news_id"].tolist() == ["n1"]


def test_all_rows_are_kept():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n3", "머니투데이", "삼성전자 미국 반도체 공장 증설 발표"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert len(out) == 3  # 매체 수를 세야 하므로 행은 지우지 않는다


def test_empty_input_keeps_columns():
    df = _mapped([])
    out = cluster_duplicates(df)
    assert out.empty
    assert "cluster_id" in out.columns and "is_primary" in out.columns
