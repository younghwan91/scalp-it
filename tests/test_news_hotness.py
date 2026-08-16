import numpy as np
import pandas as pd

from swing_it.news.hotness import (
    W_ARTICLES,
    W_OUTLETS,
    W_TRADE,
    add_hotness,
    daily_counts,
    rank_daily,
)


def _clustered(rows: list[dict]) -> pd.DataFrame:
    base = {"news_id": "n", "date": pd.Timestamp("2026-08-17"), "code": "005930",
            "match_kind": "name", "outlet": "매체", "title": "제목",
            "cluster_id": 0, "is_primary": True}
    return pd.DataFrame([{**base, **row} for row in rows])


def _liquidity(codes: list[str], dates: list[str], adv: float, tv_ratio: float):
    return pd.DataFrame(
        [
            {"code": c, "date": pd.Timestamp(d), "adv": adv, "tv_ratio": tv_ratio}
            for c in codes
            for d in dates
        ]
    )


def test_counts_use_clusters_not_rows():
    df = _clustered([
        {"news_id": "n1", "outlet": "A", "cluster_id": 0, "is_primary": True},
        {"news_id": "n2", "outlet": "B", "cluster_id": 0, "is_primary": False},
        {"news_id": "n3", "outlet": "C", "cluster_id": 1, "is_primary": True},
    ])
    out = daily_counts(df)
    assert out.iloc[0]["n_articles"] == 2  # 사건 2건
    assert out.iloc[0]["n_outlets"] == 3   # 매체 3곳


def test_counts_pick_first_primary_as_representative():
    df = _clustered([
        {"news_id": "n5", "outlet": "A", "cluster_id": 0, "is_primary": True},
        {"news_id": "n2", "outlet": "B", "cluster_id": 1, "is_primary": True},
    ])
    out = daily_counts(df)
    assert out.iloc[0]["primary_news_id"] == "n2"


def test_baseline_counts_zero_article_days():
    # 3일 중 2일은 기사가 없다. baseline 은 달력일 평균이어야 한다.
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 3,
         "n_outlets": 3, "primary_news_id": "n1"},
        {"date": pd.Timestamp("2026-01-04"), "code": "A", "n_articles": 4,
         "n_outlets": 4, "primary_news_id": "n2"},
    ])
    liq = _liquidity(["A"], dates, adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=3, min_periods=3)
    row = out[out["date"] == pd.Timestamp("2026-01-04")].iloc[0]
    # 1/1~1/3 의 기사 건수는 3, 0, 0 → 평균 1.0
    assert row["baseline_articles"] == 1.0
    assert row["article_ratio"] == 4.0


def test_baseline_floor_applies_when_history_is_short():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 2,
         "n_outlets": 2, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    assert out.iloc[0]["baseline_articles"] == 0.5
    assert out.iloc[0]["article_ratio"] == 4.0


def test_baseline_excludes_current_day():
    dates = ["2026-01-01", "2026-01-02"]
    counts = pd.DataFrame([
        {"date": pd.Timestamp(d), "code": "A", "n_articles": 2,
         "n_outlets": 1, "primary_news_id": "n"} for d in dates
    ])
    liq = _liquidity(["A"], dates, adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=1, min_periods=1)
    second = out[out["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert second["baseline_articles"] == 2.0  # 전날 값. 당일 2 가 섞이면 lookahead


def test_hotness_matches_weighted_formula():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 2,
         "n_outlets": 4, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=3.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    expected = (
        W_ARTICLES * np.log1p(4.0) + W_OUTLETS * np.log1p(4.0) + W_TRADE * np.log1p(3.0)
    )
    assert np.isclose(out.iloc[0]["hotness"], expected)


def test_missing_liquidity_row_yields_nan_adv():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "Z", "n_articles": 1,
         "n_outlets": 1, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    assert np.isnan(out.iloc[0]["adv"])
    assert out.iloc[0]["tv_ratio"] == 1.0  # 거래대금 급증 없음으로 간주


def test_rank_drops_illiquid_and_missing_price_rows():
    hot = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 5, "n_outlets": 5,
         "article_ratio": 5.0, "adv": 10000.0, "tv_ratio": 3.0, "hotness": 2.0,
         "primary_news_id": "n1"},
        {"date": pd.Timestamp("2026-01-01"), "code": "B", "n_articles": 9, "n_outlets": 9,
         "article_ratio": 9.0, "adv": 100.0, "tv_ratio": 9.0, "hotness": 3.0,
         "primary_news_id": "n2"},
        {"date": pd.Timestamp("2026-01-01"), "code": "C", "n_articles": 9, "n_outlets": 9,
         "article_ratio": 9.0, "adv": np.nan, "tv_ratio": 1.0, "hotness": 3.0,
         "primary_news_id": "n3"},
    ])
    out = rank_daily(hot, adv_floor=5000.0)
    assert out["code"].tolist() == ["A"]


def test_rank_is_per_date_and_flags_train_and_ops():
    rows = []
    for day in ("2026-01-01", "2026-01-02"):
        for i in range(60):
            rows.append({
                "date": pd.Timestamp(day), "code": f"{i:06d}", "n_articles": 1,
                "n_outlets": 1, "article_ratio": 1.0, "adv": 10000.0,
                "tv_ratio": 1.0, "hotness": float(60 - i), "primary_news_id": f"n{i}",
            })
    out = rank_daily(pd.DataFrame(rows), train_top=50, ops_top=10)
    assert len(out) == 100                       # 날짜당 50건
    assert out["rank"].min() == 1
    assert out["rank"].max() == 50
    assert out.groupby("date")["in_ops"].sum().tolist() == [10, 10]


def test_rank_orders_by_hotness_descending():
    rows = [
        {"date": pd.Timestamp("2026-01-01"), "code": c, "n_articles": 1, "n_outlets": 1,
         "article_ratio": 1.0, "adv": 10000.0, "tv_ratio": 1.0, "hotness": h,
         "primary_news_id": "n"}
        for c, h in [("A", 1.0), ("B", 3.0), ("C", 2.0)]
    ]
    out = rank_daily(pd.DataFrame(rows))
    assert out["code"].tolist() == ["B", "C", "A"]


def test_rank_output_columns():
    rows = [{"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 1,
             "n_outlets": 1, "article_ratio": 1.0, "adv": 10000.0, "tv_ratio": 1.0,
             "hotness": 1.0, "primary_news_id": "n1"}]
    out = rank_daily(pd.DataFrame(rows))
    assert list(out.columns) == [
        "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
        "tv_ratio", "hotness", "rank", "in_train", "in_ops", "primary_news_id",
    ]
