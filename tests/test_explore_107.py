"""107번 마감 30분·단일가 — 라벨 창 · 비용 한 번 · 조건 변수가 그 시점 이전 값만 쓰는지.

문서 §1 정의: 시각 T 의 가격 = T 직전 분봉(ts < T, 5분 안) close. 15:20 = 15:19 분봉, 15:30 = 일봉 종가.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "explore_107"))
import common as c  # noqa: E402


def _bars(code: str, closes: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"code": code, "ts": [pd.Timestamp(f"2026-09-02 {t}") for t in closes],
                         "close": list(closes.values())})


def test_price_at_bounds_uses_bar_strictly_before_T():
    # 15:19 분봉이 15:20 가격, 15:20 분봉(있어도 안 씀)·15:30 분봉은 15:20 가격이 아니다.
    mb = _bars("A", {"14:49": 100.0, "14:50": 999.0, "15:19": 110.0, "15:20": 888.0, "15:30": 777.0})
    px = c.price_at_bounds(mb)
    assert px.loc["A", "14:50"] == 100.0
    assert px.loc["A", "15:20"] == 110.0


def test_price_at_bounds_lookback_window():
    # 5분 안에 체결이 없으면 결측 — 14:44 는 14:50 의 5분 창(14:45~14:49) 밖.
    mb = _bars("A", {"14:44": 100.0, "15:19": 110.0})
    px = c.price_at_bounds(mb)
    assert np.isnan(px.loc["A", "14:50"])
    mb2 = _bars("A", {"14:45": 101.0, "14:47": 102.0, "15:19": 110.0})
    assert c.price_at_bounds(mb2).loc["A", "14:50"] == 102.0   # 창 안 마지막 분봉


def _prices():
    cols = {b: 100.0 for b in c.BOUNDS}
    cols["09:00"] = 100.0   # 일봉 시가
    cols["14:50"] = 102.0
    cols["15:20"] = 103.0
    cols["15:30"] = 104.0   # 일봉 종가
    return pd.DataFrame([cols, {**cols, "15:30": 101.0}], index=["K", "Q"])


def test_labels_windows_and_cost_once():
    px = _prices()
    mk = pd.Series({"K": "거래소", "Q": "코스닥"})
    lab = c.labels(px, mk)
    # 다리 A: 14:50 → 15:20, 41.5bp 한 번.
    assert lab.loc["K", "y_a"] == pytest.approx((103 / 102 - 1) * 1e4 - 41.5)
    # 다리 B: 15:20 → 일봉 종가, 시장별 단일가 비용 한 번(거래소 39.8 / 코스닥 41.3).
    assert c.COST_B["거래소"] == pytest.approx(39.8) and c.COST_B["코스닥"] == pytest.approx(41.3)
    assert lab.loc["K", "y_b"] == pytest.approx((104 / 103 - 1) * 1e4 - 39.8)
    assert lab.loc["Q", "y_b"] == pytest.approx((101 / 103 - 1) * 1e4 - 41.3)
    assert c.COST_A == pytest.approx(41.5)


def test_condition_uses_only_prices_known_at_1450():
    px = _prices()
    mk = pd.Series({"K": "거래소", "Q": "코스닥"})
    base = c.labels(px, mk)
    # 15:20·15:30 을 아무리 바꿔도 조건 변수는 그대로다.
    px2 = px.copy()
    px2["15:20"] = 50.0
    px2["15:30"] = 500.0
    assert c.labels(px2, mk).cond.equals(base.cond)
    assert base.loc["K", "cond"] == pytest.approx((102 / 100 - 1) * 1e4)


def test_prior_day_features_are_shifted():
    d = pd.DataFrame({"code": "A", "date": pd.date_range("2026-01-01", periods=25, freq="B"),
                      "close": np.linspace(100, 124, 25), "trade_value": np.arange(25, dtype=float)})
    f = c.prior_day_features(d).set_index("date")
    last = f.index[-1]
    assert f.loc[last, "p_tv"] == 23.0                      # 전일 거래대금
    assert f.loc[last, "p_close"] == pytest.approx(123.0)    # 전일 종가
    assert f.loc[last, "p_ret1"] == pytest.approx(123 / 122 - 1)


def test_interval_returns_cover_whole_day_once():
    px = _prices()
    r = c.interval_returns(px)
    assert len(r.columns) == 14 and r.columns[0] == "09:00→09:30" and r.columns[-1] == "15:20→15:30"
    # 구간 로그수익의 합 = 시가→종가 (겹침·누락 없음)
    tot = np.log1p(r.loc["K"] / 1e4).sum()
    assert tot == pytest.approx(np.log(104 / 100))


def test_tail_bucket_is_within_day():
    x = pd.Series(np.arange(20, dtype=float))
    b = c.tail_bucket(x)
    assert (b == "top").sum() == 2 and (b == "bottom").sum() == 2 and (b == "mid").sum() == 16


def test_universe_prev_day_only():
    f = pd.DataFrame({"code": list("ABCD"), "p_close": [900, 2000, 3000, 4000], "p_tv": [9e9, 3.0, 2.0, 1.0]})
    u = c.universe(f, top_n=2)
    assert list(u.code) == ["B", "C"]   # A 는 전일 종가 < 1,000
