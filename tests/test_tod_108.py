"""108번 시간대 분해 — 라벨 산술을 합성 데이터로 못 박는다(DB 없음).

보는 것: (1) 경계 가격이 "T 직전 마지막 분봉 close" 인가, 09:00 은 open·15:30 은 단일가 close 인가, 15:20~15:29
행은 무시되는가 (2) 창 수익을 곱하면 정확히 시가→종가·시가→익일 시가가 되는가(텔레스코핑) (3) 버킷·조건이
**전일 값만** 쓰는가(당일 일봉을 바꿔도 버킷이 안 바뀐다) (4) 야간이 익일 09:00 open 을 쓰는가.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tod108", Path(__file__).resolve().parents[1] / "scripts" / "minute_108" / "tod.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def _bars(code: str, day: str, *, skip=(), extra=()):
    """09:00~15:19 연속 분봉 + 15:30 단일가. close = 10000 + 분 인덱스. skip 분은 행이 없다."""
    rows = []
    for m in range(9 * 60, 15 * 60 + 20):
        if m in skip:
            continue
        rows.append((code, f"{day} {m // 60:02d}:{m % 60:02d}", 10000.0 + m - 540 - 0.5, 10000.0 + m - 540))
    rows.append((code, f"{day} 15:30", 12000.0, 12345.0))
    for hhmm, o, c in extra:
        rows.append((code, f"{day} {hhmm}", o, c))
    return pd.DataFrame(rows, columns=["code", "ts", "open", "close"]).assign(ts=lambda d: pd.to_datetime(d.ts))


def test_boundary_prices_are_last_close_before_T_open_at_0900_auction_at_1530():
    b = _bars("A", "2026-03-03", skip=(9 * 60 + 4,), extra=(("15:25", 1.0, 1.0), ("15:35", 2.0, 2.0)))
    p = T.boundary_prices(b).loc["A"]
    assert p.p_0900 == 10000.0 - 0.5                     # 09:00 행 open
    assert p.p_0905 == 10003.0 and p.stale_0905 == 2     # 09:04 가 비면 09:03 close, 나이 2분
    assert p.p_0930 == 10029.0 and p.stale_0930 == 1
    assert p.p_1520 == 10000.0 + (15 * 60 + 19 - 540)   # 15:19 close — 15:25 행은 무시
    assert p.p_1530 == 12345.0                            # 15:30 단일가 close (15:35 행 무시)
    assert p.n_bars == 379


def test_missing_1530_auction_leaves_close_nan():
    b = _bars("A", "2026-03-03")
    b = b[b.ts.dt.strftime("%H:%M") != "15:30"]
    p = T.boundary_prices(b).loc["A"]
    assert np.isnan(p.p_1530) and p.p_1520 > 0


def test_windows_telescope_to_open_to_close_and_overnight_uses_next_open():
    p = T.boundary_prices(_bars("A", "2026-03-03"))
    pn = T.boundary_prices(_bars("A", "2026-03-04"))
    w = T.window_returns(p, pn).loc["A"]
    intraday = [f"{a}→{b}" for a, b in zip(T.BOUNDARIES[:-1], T.BOUNDARIES[1:], strict=True)]
    assert len(intraday) == 9
    prod = np.prod([1 + w[k] / 1e4 for k in intraday])
    assert prod == pytest.approx(p.loc["A", "p_1530"] / p.loc["A", "p_0900"])
    assert w["09:00→15:30"] == pytest.approx((p.loc["A", "p_1530"] / p.loc["A", "p_0900"] - 1) * 1e4)
    assert w["15:30→익일09:00"] == pytest.approx((pn.loc["A", "p_0900"] / 12345.0 - 1) * 1e4)
    assert w["15:20→익일09:00"] == pytest.approx((pn.loc["A", "p_0900"] / p.loc["A", "p_1520"] - 1) * 1e4)
    # 야간 다음 날이 없으면 NaN, 일중은 그대로
    w0 = T.window_returns(p, None).loc["A"]
    assert np.isnan(w0["15:30→익일09:00"]) and w0["09:00→15:30"] == w["09:00→15:30"]


def test_universe_uses_prev_day_rank_only_with_fixed_edges():
    n = 1100
    prev = pd.DataFrame({"code": [f"C{i:04d}" for i in range(n)],
                         "close": [500 if i == 0 else 5000 for i in range(n)],
                         "trade_value": np.arange(n, 0, -1, dtype=float)})
    u = T.universe_from_prev_daily(prev).set_index("code")
    assert "C0000" not in u.index                       # 1,000원 미만 제외
    assert u.loc["C0001", "rank"] == 1 and u.loc["C0001", "bucket"] == "1~100"
    assert u.loc["C0100", "bucket"] == "1~100" and u.loc["C0101", "bucket"] == "101~200"
    assert u.loc["C0300", "bucket"] == "201~300" and u.loc["C0301", "bucket"] == "301~400"
    assert u.loc["C0400", "bucket"] == "301~400" and u.loc["C0401", "bucket"] == "401~1000"
    assert u.loc["C1000", "bucket"] == "401~1000" and "C1001" not in u.index


def test_prev_features_come_from_1530_auction_closes_not_daily():
    p2 = T.boundary_prices(_bars("A", "2026-03-02"))
    p1 = T.boundary_prices(_bars("A", "2026-03-03"))
    p1.loc["A", "p_1530"] = 12000.0
    pd_ = T.boundary_prices(_bars("A", "2026-03-04"))
    f = T.prev_features(pd_, p1, p2).loc["A"]
    assert f.ret1_prev == pytest.approx((12000.0 / 12345.0 - 1) * 1e4)
    assert f.gap == pytest.approx((pd_.loc["A", "p_0900"] / 12000.0 - 1) * 1e4)
    f0 = T.prev_features(pd_, None, None).loc["A"]
    assert np.isnan(f0.ret1_prev) and np.isnan(f0.gap)


def _synthetic_world(n_codes=150, days=("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05")):
    codes = [f"C{i:03d}" for i in range(n_codes)]
    P = {}
    rng = np.random.default_rng(0)
    for d in days:
        bars = pd.concat([_bars(c, d) for c in codes], ignore_index=True)
        bars["close"] = bars["close"] * (1 + rng.normal(0, 0.001, len(bars)))
        P[d] = T.boundary_prices(bars)
    daily = pd.concat([pd.DataFrame({"code": codes, "date": d, "close": 5000,
                                     "trade_value": np.arange(n_codes, 0, -1, dtype=float)}) for d in days],
                      ignore_index=True)
    market = pd.Series("거래소", index=codes)
    return list(days), P, daily, market


def test_build_rows_bucket_ignores_same_day_trade_value():
    days, P, daily, market = _synthetic_world()
    rows = T.build_rows(days, set(days[1:]), P, daily, market)
    d = "2026-03-04"
    before = rows[rows.date == d].set_index("code")["bucket"]
    # 당일 거래대금을 뒤집는다 — 전일 순위 버킷은 그대로, 당일 순위(look-ahead 대조)만 바뀌어야 한다
    daily2 = daily.copy()
    daily2.loc[daily2.date == d, "trade_value"] = daily2.loc[daily2.date == d, "trade_value"].to_numpy()[::-1]
    rows2 = T.build_rows(days, set(days[1:]), P, daily2, market)
    after = rows2[rows2.date == d].set_index("code")
    assert after["bucket"].equals(before)
    assert not after["bucket_sameday"].equals(rows[rows.date == d].set_index("code")["bucket_sameday"])
    # 다음 날 유니버스는 바뀐 "전일" 값을 따라간다(그게 전일 기준이라는 뜻)
    nxt = rows2[rows2.date == "2026-03-05"].set_index("code")["bucket"]
    assert not nxt.equals(rows[rows.date == "2026-03-05"].set_index("code")["bucket"])


def test_build_rows_first_sample_day_has_no_prev_features_and_last_has_no_overnight():
    days, P, daily, market = _synthetic_world()
    rows = T.build_rows(days, set(days), P, daily, market)
    first = rows[rows.date == days[0]]
    assert first.empty                                       # 전일 일봉이 없으면 유니버스를 못 만든다
    second = rows[rows.date == days[1]]
    assert second["ret1_prev"].isna().all() and second["gap"].notna().all()
    last = rows[rows.date == days[-1]]
    assert last["15:30→익일09:00"].isna().all() and last["09:00→15:30"].notna().all()
    assert set(rows.columns) >= {"ret_q", "gap_q", "gap_sign", "market", "rank", "bucket"}


def test_cell_stats_reports_day_level_t_and_shares():
    days = pd.bdate_range("2026-03-02", periods=8)
    dates = pd.to_datetime(np.repeat(days, 3))
    f = pd.DataFrame({"date": dates, "grp": "g", "x": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0] * 2})
    s = T.cell_stats(f, "x", "grp").loc["g"]
    assert s.n == 24 and s.n_days == 8 and s.mean_bp == pytest.approx(2.75) and s.day_mean_bp == pytest.approx(2.75)
    assert s.pos_share == 1.0 and s.neg_day_share == 0.0 and s.neg_month_share == 0.0
    assert s.t_hac > 0
