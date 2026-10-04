"""105번 개장 동시호가 탐색 — 라벨 산술을 합성 데이터로 못 박는다.

세 가지만 본다: (1) 라벨이 09:00 open → 지정 분 close 인가 (2) 비용이 종목별로 **한 번만** 빠지는가
(3) 조건·층이 **전일 값만** 쓰는가(당일 일봉을 바꿔도 피처가 안 바뀐다). 97번의 부호 전환이 이 셋 중
하나가 틀려서 났다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "labels105", Path(__file__).resolve().parents[1] / "scripts" / "explore_105" / "labels.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def test_cost_per_market_is_tax_fee_and_two_half_spreads():
    assert L.cost_bp("거래소") == pytest.approx(20 + 3 + 19.1 / 2 + 14.0 / 2)   # 39.55
    assert L.cost_bp("코스닥") == pytest.approx(20 + 3 + 30.7 / 2 + 19.1 / 2)   # 47.9


def _minute(code, day, closes, open0=10000.0, skip=()):
    """09:00~09:15 분봉. closes[m] = 그 분의 close. skip 에 든 분은 행이 없다(체결 없음)."""
    rows = []
    for m, c in enumerate(closes):
        if m in skip:
            continue
        rows.append({"code": code, "ts": pd.Timestamp(f"{day} 09:{m:02d}"),
                     "open": open0 if m == 0 else closes[m - 1], "close": c})
    return pd.DataFrame(rows)


def test_label_is_0900_open_to_horizon_close_minus_cost_once():
    closes = [10000 + 10 * (m + 1) for m in range(16)]      # 09:00 close 10010 … 09:15 close 10160
    hc = L.horizon_closes(_minute("A", "2026-01-05", closes, open0=10000.0))
    assert len(hc) == 1
    assert hc.open0[0] == 10000.0 and hc.c1[0] == 10020 and hc.c5[0] == 10060 and hc.c15[0] == 10160
    ev = hc.assign(p_close=9900.0, cost_bp=40.0)
    lab = L.label(ev)
    assert lab.r5[0] == pytest.approx(10060 / 10000 - 1 - 0.0040)
    assert lab.r15[0] == pytest.approx(10160 / 10000 - 1 - 0.0040)
    # 비용은 한 번: 수익에서 cost 를 더하면 정확히 원수익이 돌아와야 한다.
    assert lab.r1[0] + 0.0040 == pytest.approx(10020 / 10000 - 1)
    assert lab.gap[0] == pytest.approx(10000 / 9900 - 1)


def test_missing_minute_uses_last_close_at_or_before_horizon():
    closes = [10000 + 10 * (m + 1) for m in range(16)]
    hc = L.horizon_closes(_minute("A", "2026-01-05", closes, skip={5, 15}))
    assert hc.c5[0] == closes[4] and hc.c15[0] == closes[14]      # 09:04 · 09:14 의 close


def test_no_0900_bar_means_no_event():
    closes = [10000 + 10 * (m + 1) for m in range(16)]
    hc = L.horizon_closes(_minute("A", "2026-01-05", closes, skip={0}))
    assert hc.empty


def test_horizon_never_reads_past_the_horizon_minute():
    """09:15 뒤의 분봉이 있어도 c15 는 09:15 까지만 본다."""
    closes = [10000.0] * 16
    m = _minute("A", "2026-01-05", closes)
    m = pd.concat([m, pd.DataFrame([{"code": "A", "ts": pd.Timestamp("2026-01-05 09:16"),
                                     "open": 10000.0, "close": 99999.0}])])
    hc = L.horizon_closes(m)
    assert hc.c15[0] == 10000.0


def _daily(n_days=30, codes=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-11-01", periods=n_days)
    rows = []
    for c in codes:
        px = 10000.0
        for d in days:
            px *= 1 + rng.normal(0, 0.02)
            rows.append({"code": c, "date": d, "open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "trade_value": rng.uniform(1e9, 1e10)})
    return pd.DataFrame(rows)


def test_prior_day_features_come_only_from_previous_row():
    d = _daily()
    f = L.prior_day_features(d).set_index(["code", "date"])
    day = pd.Timestamp("2025-12-01")
    prev = d[(d.code == "A") & (d.date < day)].iloc[-1]
    row = f.loc[("A", day)]
    assert row.p_close == prev.close and row.p_high == prev.high and row.p_tv == prev.trade_value
    assert row.p_date == prev.date
    # 당일 행을 아무리 바꿔도 당일 피처는 그대로 — 변동성·수익률 포함.
    d2 = d.copy()
    mask = (d2.code == "A") & (d2.date == day)
    d2.loc[mask, ["close", "high", "trade_value"]] = [1.0, 1.0, 1e15]
    f2 = L.prior_day_features(d2).set_index(["code", "date"])
    pd.testing.assert_series_equal(f2.loc[("A", day)], row)


def test_prior_day_vol_uses_returns_up_to_previous_day_only():
    d = _daily()
    day = pd.Timestamp("2025-12-01")
    f = L.prior_day_features(d).set_index(["code", "date"])
    r = d[d.code == "A"].set_index("date").close.pct_change()
    expect = r[r.index < day].tail(20).std()
    assert f.loc[("A", day)].p_vol20 == pytest.approx(expect)


def test_stale_previous_row_is_dropped():
    d = _daily()
    # A 의 11-24~11-28 를 지워 12-01 의 "전일"이 11-21 이 되게 한다(달력 10일) → 버려야 한다.
    d = d[~((d.code == "A") & (d.date > "2025-11-21") & (d.date < "2025-12-01"))]
    f = L.prior_day_features(d)
    assert not ((f.code == "A") & (f.date == "2025-12-01")).any()


def test_universe_ranks_by_previous_day_value_and_cells_by_previous_return():
    n = 2000
    day = pd.Timestamp("2026-01-05")
    feats = pd.DataFrame({
        "code": [f"S{i:04d}" for i in range(n)], "date": day, "p_date": day - pd.Timedelta(days=1),
        "p_close": np.where(np.arange(n) < 10, 500.0, 5000.0),     # 앞 10개는 1,000원 미만
        "p_high": 5000.0, "p_tv": np.arange(n, dtype=float),        # 거래대금은 뒤로 갈수록 크다
        "p_ret": np.linspace(-0.1, 0.1, n), "p_vol20": np.linspace(0.01, 0.05, n),
    })
    u = L.build_universe(feats)
    assert len(u) == L.TOP_VALUE
    assert u.p_close.min() >= L.MIN_PRICE
    assert u.p_tv.min() == n - L.TOP_VALUE                          # 전일 거래대금 상위 1,300
    assert (u.ret_cell == "top10").sum() == 130 and (u.ret_cell == "bot10").sum() == 130
    assert u[u.ret_cell == "top10"].p_ret.min() > u[u.ret_cell == "mid"].p_ret.max()
    assert u[u.ret_cell == "bot10"].p_ret.max() < u[u.ret_cell == "mid"].p_ret.min()
    assert sorted(u.vol_q.unique()) == [0, 1, 2, 3] and sorted(u.tv_q.unique()) == [0, 1, 2, 3]


def test_limit_up_flag_requires_close_at_high_and_28pct():
    day = pd.Timestamp("2026-01-05")
    feats = pd.DataFrame({
        "code": list("ABCD") + [f"X{i}" for i in range(20)], "date": day,
        "p_date": day - pd.Timedelta(days=1), "p_close": 1300.0,
        "p_high": [1300, 1310, 1300, 1300] + [1400] * 20, "p_tv": np.arange(24, dtype=float),
        "p_ret": [0.30, 0.30, 0.27, 0.299] + [0.0] * 20, "p_vol20": 0.02,
    })
    u = L.build_universe(feats).set_index("code")
    assert bool(u.loc["A", "limit_up"]) and not u.loc["B", "limit_up"] and not u.loc["C", "limit_up"]
    assert bool(u.loc["D", "limit_up"])


def test_build_events_uses_minute_open_not_daily_open_and_excludes_bad_days():
    d = _daily(codes=tuple(f"S{i}" for i in range(12)))
    d["open"] = 1.0                                                  # 일봉 시가는 쓰면 안 된다
    day = "2025-12-01"
    parts = [_minute(c, day, [20000.0] * 16, open0=10000.0) for c in d.code.unique()]
    m = pd.concat(parts)
    mk = pd.DataFrame({"code": d.code.unique(), "market": "거래소"})
    ev = L.build_events(d, m, mk)
    assert (ev.date == pd.Timestamp(day)).all() and len(ev) == 12
    assert (ev.open0 == 10000.0).all()
    assert ev.r5.iloc[0] == pytest.approx(1.0 - 39.55 / 1e4)
    assert L.build_events(d, m, mk, bad_days={day}).empty
