import numpy as np
import pandas as pd

from scalp_it.model.layered import fit_final_model
from scalp_it.report.cards import feature_contributions, find_analogs, render_card

COLS = ["tv_ratio", "article_count", "outlet_count", "article_ratio", "llm_specificity"]


def _train(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in COLS})
    df["code"] = [f"{i % 20:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    df["excess_return"] = df["llm_specificity"] * 0.08 + rng.normal(scale=0.01, size=n)
    return df


def _candidate() -> pd.Series:
    return pd.Series(
        {
            "code": "034020",
            "date": pd.Timestamp("2026-08-14"),
            "tv_ratio": 4.6,
            "article_count": 7.0,
            "outlet_count": 5.0,
            "article_ratio": 4.2,
            "llm_specificity": 2.5,
            "prediction": 0.0481,
        }
    )


def _articles() -> pd.DataFrame:
    """load_article_cards 반환 형태 — 집계 수치는 없다. (code, date) 당 대표 기사 1건."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14", "2026-08-14"]),
            "code": ["034020", "005930"],
            "hotness_rank": [1.0, 2.0],
            "hotness": [0.81, 0.44],
            "article_id": ["n1", "n2"],
            "press": ["한국경제", "더벨"],
            "title": ["체코 원전 2기 최종 계약", "무관한 기사"],
            "body200": ["두산에너빌리티가 체코 두코바니 원전 2기 주기기 공급 계약을 체결했다.", "본문"],
            "url": ["https://x/1", "https://x/2"],
        }
    )


def _card(**overrides) -> str:
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    kwargs = {
        "rank": 3,
        "contributions": feature_contributions(model, row, baseline, COLS),
        "analogs": find_analogs(train, row, COLS, k=3),
        "articles": _articles(),
        "gate_verdict": "FAIL",
        "layer_label": "③ + LLM",
        "entry_date": pd.Timestamp("2026-08-17"),
        "exit_date": pd.Timestamp("2026-10-13"),
        "names": {"034020": "두산에너빌리티"},
        "material_types": {"034020": "수주"},
    }
    kwargs.update(overrides)
    return render_card(row, **kwargs)


def test_contribution_is_zero_for_a_feature_already_at_baseline():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate().copy()
    row["llm_specificity"] = baseline["llm_specificity"]
    contrib = feature_contributions(model, row, baseline, COLS)
    assert np.isclose(contrib["llm_specificity"], 0.0)


def test_the_driving_feature_ranks_first():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert contrib.index[0] == "llm_specificity"
    assert contrib.iloc[0] > 0


def test_contributions_are_sorted_descending():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert list(contrib) == sorted(contrib, reverse=True)


def test_analogs_return_k_nearest_with_realised_returns():
    analogs = find_analogs(_train(), _candidate(), COLS, k=3)
    assert list(analogs.columns) == ["code", "date", "excess_return", "distance"]
    assert len(analogs) == 3
    assert analogs["distance"].is_monotonic_increasing


def test_analogs_are_actually_close_in_feature_space():
    train = _train()
    analogs = find_analogs(train, train.iloc[7].copy(), COLS, k=1)
    assert analogs.iloc[0]["distance"] < 1e-9  # 자기 자신이 최근접


def test_card_contains_the_essentials():
    card = _card()
    assert "034020 두산에너빌리티" in card
    assert "2026-08-17" in card  # 진입일
    assert "2026-10-13" in card  # 목표 청산일
    assert "llm_specificity" in card
    assert "체코 원전 2기 최종 계약" in card
    assert "유사 과거 사례" in card


def test_card_shows_article_counts_and_the_representative_article():
    card = _card()
    assert "기사 7건 · 매체 5곳 · 평소 대비 4.2배" in card
    assert "한국경제" in card
    assert "두코바니" in card  # 본문 200자


def test_counts_come_from_the_candidate_row_not_the_article_frame():
    """집계 수치는 ②층 피처다. load_article_cards 는 그것을 돌려주지 않는다."""
    row = _candidate().copy()
    row["article_count"] = 99.0
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="PASS",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "기사 99건" in card


def test_card_omits_the_counts_line_for_a_layer_without_news_features():
    """①층 후보 행에는 article_count 가 없다. 그 줄만 빠지고 카드는 나온다."""
    row = _candidate().drop(["article_count", "outlet_count", "article_ratio"])
    cols = ["tv_ratio", "llm_specificity"]
    train = _train()
    model, baseline = fit_final_model(train, cols)
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, cols),
        analogs=find_analogs(train, row, cols, k=3),
        articles=_articles(),
        gate_verdict="PASS",
        layer_label="① 수급 단독",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "기사 " not in card
    assert "체코 원전 2기 최종 계약" in card


def test_material_type_is_display_only():
    """모델 행렬에 없다. 카드에 문자열로만 딸려온다 (스펙 §6 피처 상한)."""
    assert "재료 유형 | 수주" in _card()
    assert "재료 유형" not in _card(material_types=None)


def test_card_never_hides_a_failed_verdict():
    """스펙 §10 — FAIL 을 숨기지 않는다. FAIL 모델의 카드는 추천이 아니다."""
    card = _card(gate_verdict="FAIL")
    assert "FAIL" in card
    assert "매매 신호가 아니다" in card
    assert "널을 이기지 못했다" in card


def test_passing_card_omits_the_warning_but_keeps_the_disclaimer():
    card = _card(gate_verdict="PASS")
    assert "널을 이기지 못했다" not in card
    assert "매매 신호가 아니다" in card


def test_card_does_not_leak_another_codes_article():
    assert "무관한 기사" not in _card()


def test_card_works_when_there_is_no_representative_article():
    articles = _articles()
    articles.loc[
        articles["code"] == "034020",
        ["article_id", "press", "title", "body200", "url"],
    ] = None
    card = _card(articles=articles)
    assert "대표 기사 없음" in card
    assert "기사 7건 · 매체 5곳" in card  # 집계 수치는 후보 행에서 오므로 살아 있다


def test_card_works_when_the_code_has_no_news_row_at_all():
    card = _card(articles=_articles().iloc[0:0])
    assert "근거 기사 없음" in card


def test_card_falls_back_to_bare_code_without_a_name_map():
    card = _card(names=None)
    assert "034020" in card
