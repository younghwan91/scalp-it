"""휴장일 기사 이월 — 조용한 데이터 손실 방지."""

import pandas as pd
import pytest

from swing_it.news.calendar import align_to_trading_days


@pytest.fixture()
def calendar() -> pd.Series:
    """금(8/14) · 월(8/17) · 화(8/18) 만 거래일. 주말은 휴장."""
    return pd.Series(pd.to_datetime(["2026-08-14", "2026-08-17", "2026-08-18"]))


def _articles(dates: list[str], news_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "code": ["005930"] * len(dates),
            "outlet": ["한국경제"] * len(dates),
            "news_id": news_ids,
            "title": ["제목"] * len(dates),
        }
    )


def test_trading_day_article_keeps_its_date(calendar):
    out = align_to_trading_days(_articles(["2026-08-14"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-14")
    assert out.iloc[0]["published_date"] == pd.Timestamp("2026-08-14")


def test_saturday_article_moves_to_monday(calendar):
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")
    # 원래 게재일은 보존한다 — 근거 카드에 표시해야 한다
    assert out.iloc[0]["published_date"] == pd.Timestamp("2026-08-15")


def test_sunday_article_also_moves_to_monday(calendar):
    out = align_to_trading_days(_articles(["2026-08-16"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_weekend_articles_collapse_onto_the_same_trading_day(calendar):
    out = align_to_trading_days(
        _articles(["2026-08-15", "2026-08-16"], ["n1", "n2"]), calendar
    )
    assert set(out["date"]) == {pd.Timestamp("2026-08-17")}
    # 행은 합쳐지지 않는다. 건수·매체수 집계는 daily_counts 가 한다.
    assert len(out) == 2


def test_article_after_the_last_trading_day_is_dropped(calendar):
    """달력 끝을 넘어가면 이월할 거래일이 없다. 남겨두면 라벨이 안 생긴다."""
    out = align_to_trading_days(_articles(["2026-08-19"], ["n1"]), calendar)
    assert out.empty


def test_forward_only_never_moves_backwards(calendar):
    """뒤로 당기면 lookahead 다 — 토요일 기사를 금요일에 알 수는 없다."""
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert (out["date"] >= out["published_date"]).all()


def test_empty_input_returns_empty_with_columns(calendar):
    empty = _articles([], [])
    out = align_to_trading_days(empty, calendar)
    assert out.empty
    assert "published_date" in out.columns


def test_other_columns_are_preserved(calendar):
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert out.iloc[0]["news_id"] == "n1"
    assert out.iloc[0]["code"] == "005930"
    assert out.iloc[0]["outlet"] == "한국경제"
