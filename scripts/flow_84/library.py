"""84번 후보 피처 라이브러리 — 83번 피처 + 창 확장 + 호가 미시 + 재료. 사전등록 §2.

    FLOW84_FORWARD=0 ~/of80/.rlvenv/bin/python scripts/flow_84/library.py 2026-08-24 ... 2026-09-15

83번 `build_code` 가 만든 초 격자(같은 결정 행·같은 라벨)를 그대로 쓰고 열만 더한다.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flow_83"))
import features as F83  # noqa: E402

CACHE = Path.home() / "of80" / "flow84"
DART = Path.home() / "of80" / "dart.db"
W8 = (1, 2, 3, 5, 10, 20, 30, 60)
GOOD_MATERIAL = ("공급계약", "실적", "시설투자", "기술_인허가", "주주환원")
BAD_MATERIAL = ("악재",)
LABELS = ("y_F2", "hold_F2", "t_mid10")


@numba.njit
def _trade_runs(sec, side, price, vol, n, big_thr_by_min):
    run = np.zeros(n)            # 초 끝 시점 연속 매수 체결 길이
    since_big = np.full(n, np.nan)
    cur = 0
    last_big = -1
    j = 0
    m = len(sec)
    for s in range(n):
        while j < m and sec[j] == s:
            if side[j] > 0:
                cur += 1
                if price[j] * vol[j] >= big_thr_by_min[s // 60]:
                    last_big = s
            elif side[j] < 0:
                cur = 0
            j += 1
        run[s] = cur
        if last_big >= 0:
            since_big[s] = s - last_big
    return run, since_big


def _extra(tk: pd.DataFrame, qt: pd.DataFrame, f: dict, p: dict) -> dict:
    """83번 격자 위에 얹는 추가 열(전부 t 이하)."""
    N = F83.N
    rs, lag = F83.rsum, F83.lag
    buy, sell = p["buyval"].astype(np.float64), p["sellval"].astype(np.float64)
    bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
    mid = (bid + ask) / 2
    stg = p["strength"].astype(np.float64)
    e = p["e_ofi"].astype(np.float64)
    sec = (tk.ts.dt.hour * 3600 + tk.ts.dt.minute * 60 + tk.ts.dt.second).to_numpy() - F83.T0
    ok = (sec >= 0) & (sec < N)
    tk, sec = tk[ok], sec[ok].astype(np.int64)
    price, vol, side = tk.price.to_numpy(float), tk.volume.to_numpy(float), tk.side.to_numpy(np.int64)
    volsec = np.bincount(sec, weights=vol, minlength=N).astype(float)
    tv = price * vol
    thr = np.full(N // 60 + 1, np.inf)
    for mi in range(N // 60 + 1):
        a, b = np.searchsorted(sec, mi * 60 - 600), np.searchsorted(sec, mi * 60)
        if b - a >= 20:
            thr[mi] = float(np.median(tv[a:b])) * 20.0
    run, since_big = _trade_runs(sec, side, price, vol, N, thr)

    bq1 = np.full(N, np.nan)
    aq1 = np.full(N, np.nan)
    if len(qt):
        qs = (qt.ts.dt.hour * 3600 + qt.ts.dt.minute * 60 + qt.ts.dt.second).to_numpy() - F83.T0
        mq = (qs >= 0) & (qs < N)
        bq1[qs[mq]] = qt.bidqty1.to_numpy(float)[mq]
        aq1[qs[mq]] = qt.askqty1.to_numpy(float)[mq]
    bq1, aq1 = F83.ffill(bq1), F83.ffill(aq1)
    sellvol = np.bincount(sec[side < 0], weights=vol[side < 0], minlength=N).astype(float)

    x: dict[str, np.ndarray] = {}
    tot = buy + sell
    sgn = np.sign(buy - sell)
    for w in W8:
        bw, sw_ = rs(buy, w), rs(sell, w)
        x[f"bv{w}"] = np.log1p(bw)
        x[f"sv{w}"] = np.log1p(sw_)
        x[f"bshare{w}"] = bw / (bw + sw_ + 1)
        x[f"ntr{w}"] = rs(f["n_tr_sec"], w)
        x[f"dS{w}"] = stg - lag(stg, w)
        x[f"ofiw{w}"] = rs(e, w) / (pd.Series(np.nan_to_num((bq1 + aq1) / 2)).rolling(60, min_periods=1).mean().to_numpy() + 1)
        x[f"r{w}"] = (mid / lag(mid, w) - 1) * 1e4
        vw = rs(tot, w) / (rs(volsec, w) + 1e-9)
        x[f"vwapdev{w}"] = np.where(rs(volsec, w) > 0, (mid / vw - 1) * 1e4, 0.0)
        x[f"bq1chg{w}"] = (bq1 - lag(bq1, w)) / (lag(bq1, w) + 1)
        x[f"aq1depl{w}"] = (lag(aq1, w) - aq1) / (lag(aq1, w) + 1)
    for w in (10, 30):
        hv, hs = F83._roll_extreme(mid, w, True)
        lv, ls = F83._roll_extreme(mid, w, False)
        x[f"dhi{w}"], x[f"dlo{w}"] = (mid / hv - 1) * 1e4, (mid / lv - 1) * 1e4
        x[f"sincehi{w}"], x[f"sincelo{w}"] = hs, ls
        x[f"burst{w}"] = rs((f["n_tr_sec"] >= 5).astype(float), w)
        x[f"sgnac{w}"] = rs(sgn * lag(np.nan_to_num(sgn), 1), w) / w
    x["sweep10"] = pd.Series(np.nan_to_num(f["sweep_ticks1"])).rolling(10, min_periods=1).max().to_numpy()
    micro = (bid * aq1 + ask * bq1) / (bq1 + aq1)
    x["micro_dev"] = (micro / mid - 1) * 1e4
    x["micro_dev5"] = pd.Series(x["micro_dev"]).rolling(5, min_periods=1).mean().to_numpy()
    x["l1_imb"] = (bq1 - aq1) / (bq1 + aq1)
    same_bid = bid == lag(bid, 1)
    cancel = np.where(same_bid, np.clip(lag(bq1, 1) - bq1 - sellvol, 0, None), 0.0) / (lag(bq1, 1) + 1)
    x["cancel3"], x["cancel10"] = rs(np.nan_to_num(cancel), 3), rs(np.nan_to_num(cancel), 10)
    x["buy_run"], x["since_bigbuy"] = run, since_big
    v5 = rs(tot, 5)
    x["val_dd5"] = np.log1p(v5) - 2 * np.log1p(lag(v5, 5)) + np.log1p(lag(v5, 10))
    return x


def _material(conn, day: str, codes: list[str]) -> tuple[dict, pd.DataFrame, dict]:
    prev_days = pd.read_sql("select distinct date from daily_bars where date < %s order by date desc limit 3",
                            conn, params=(day,)).date.astype(str).tolist()
    s = sqlite3.connect(DART)
    ds = pd.read_sql("select rcept_dt, code, material_type from disclosures where rcept_dt >= ? and rcept_dt < ?",
                     s, params=(prev_days[-1].replace("-", ""), day.replace("-", "")))
    s.close()
    ds["d"] = pd.to_datetime(ds.rcept_dt).dt.date.astype(str)
    mat = {}
    for c in codes:
        g = ds[ds.code == c]
        mat[c] = {"dis1": float((g.d == prev_days[0]).sum()), "dis3": float(len(g)),
                  "good3": float(g.material_type.isin(GOOD_MATERIAL).sum()),
                  "bad3": float(g.material_type.isin(BAD_MATERIAL).sum())}
    tg = pd.read_sql("select theme_code, flu_rt from theme_groups where snapshot_date = %s", conn, params=(prev_days[0],))
    tm = pd.read_sql("select theme_code, code from theme_members where snapshot_date = %s", conn, params=(prev_days[0],))
    tg["trank"] = tg.flu_rt.rank(ascending=False)
    best = tm.merge(tg, on="theme_code").groupby("code").trank.min()
    for c in codes:
        r = best.get(c, np.nan)
        mat[c]["theme_rank"] = float(r)
        mat[c]["theme_top5"] = float(r <= 5) if r == r else 0.0
    news = pd.read_sql(
        "select t.ticker as code, (a.published_at at time zone 'Asia/Seoul') as ts from news_articles a "
        "join news_article_tickers t on t.article_id = a.id "
        "where a.published_at >= (%s::date - 1) and a.published_at < (%s::date + 1)", conn, params=(day, day))
    prevclose = pd.read_sql("select code, close from daily_bars where date = %s", conn, params=(prev_days[0],))
    return mat, news, dict(zip(prevclose.code, prevclose.close, strict=True))


def build_day(day: str) -> Path:
    forward = os.environ.get("FLOW84_FORWARD") == "1"
    if not forward:
        assert day < F83.FORWARD_START, "탐색이 전진 날짜를 읽으려 했다"
    out = CACHE / f"{day}.pkl"
    if out.exists():
        return out
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    stocks = pd.read_sql("select code from stocks", conn).code.tolist()
    tk = pd.read_sql("select code, ts, seq, price, volume, side, strength, best_bid, best_ask from ticks "
                     "where ts >= %s and ts < %s and code = any(%s) order by code, ts, seq",
                     conn, params=(day, nxt, stocks))
    qt = pd.read_sql("select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
                     "from quotes where ts >= %s and ts < %s and code = any(%s) order by code, ts",
                     conn, params=(day, nxt, stocks))
    qg = dict(tuple(qt.groupby("code")))
    tg = dict(tuple(tk.groupby("code")))
    per = {c: F83.build_code(g, qg.get(c, qt.iloc[:0])) for c, g in tg.items() if len(g) >= 500}
    codes = sorted(per)
    mat, news, prevclose = _material(conn, day, codes)
    conn.close()
    N = F83.N
    cum = np.stack([per[c][0]["cumval"] for c in codes])
    rank = (-cum).argsort(axis=0).argsort(axis=0) + 1
    dayret = np.stack([(per[c][1]["last"] / prevclose.get(c, np.nan) - 1) * 100 for c in codes])
    dayret_rank = (-np.nan_to_num(dayret, nan=-1e9)).argsort(axis=0).argsort(axis=0) + 1
    news_sec = {}
    if len(news):
        news["sec"] = (news.ts.dt.hour * 3600 + news.ts.dt.minute * 60 + news.ts.dt.second - F83.T0).where(
            news.ts.dt.date.astype(str) == day, -10**6)
        news_sec = {c: np.sort(g.sec.to_numpy()) for c, g in news.groupby("code")}
    parts = []
    for ci, c in enumerate(codes):
        f, p = per[c]
        f["cum_rank"], f["log_cumval"], f["sec"] = rank[ci].astype(float), np.log1p(f["cumval"]), np.arange(N, dtype=float)
        s_ = np.arange(N)
        ok = ((f["n_tr_sec"] > 0) & (s_ >= F83.ENTRY_LO) & (s_ <= F83.ENTRY_HI) & (f["cum_rank"] <= F83.TOP_RANK)
              & (p["last"] >= F83.MIN_PRICE) & (p["ask"] > p["bid"]) & np.isfinite(p["ask"]) & np.isfinite(p["bid"]))
        rows = np.nonzero(ok)[0]
        if len(rows) == 0:
            continue
        x = _extra(tg[c], qg.get(c, qt.iloc[:0]), f, p)
        cols = {k: f[k][rows] for k in F83.FEATURES}
        cols.update({k: v[rows] for k, v in x.items()})
        for k, v in mat[c].items():
            cols[k] = np.full(len(rows), v)
        ns = news_sec.get(c, np.array([]))
        cols["news_prev_day"] = np.full(len(rows), float((ns == -10**6).sum()))
        today_ns = ns[ns > -10**6]
        cols["news30"] = (np.searchsorted(today_ns, rows) - np.searchsorted(today_ns, rows - 1800)).astype(float)
        cols["news120"] = (np.searchsorted(today_ns, rows) - np.searchsorted(today_ns, rows - 7200)).astype(float)
        cols["news_today"] = np.searchsorted(today_ns, rows).astype(float)
        cols["news_cov"] = np.full(len(rows), float(day >= "2026-09-04"))
        cols["dayret"], cols["dayret_rank"] = dayret[ci][rows], dayret_rank[ci][rows].astype(float)
        D, kind, win = F83.STRUCTS["F2"]
        y, h = F83._labels(rows, p["bid"], p["ask"], p["buyval"], p["sellval"], p["strength"], p["e_ofi"],
                           p["tick"], D, kind, win, F83.COST)
        mid = (p["bid"] + p["ask"]) / 2
        m10 = np.full(len(rows), np.nan)
        okm = rows + 10 < N
        m10[okm] = (mid[rows[okm] + 10] / mid[rows[okm]] - 1) * 1e4
        df = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in cols.items()})
        df.insert(0, "row", rows.astype(np.int32))
        df.insert(0, "code", c)
        df["y_F2"], df["hold_F2"], df["t_mid10"] = y.astype(np.float32), h.astype(np.int32), m10.astype(np.float32)
        parts.append(df)
    data = pd.concat(parts, ignore_index=True)
    data.insert(0, "day", day)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(data, protocol=5))
    return out


def candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("day", "code", "row", *LABELS)]


if __name__ == "__main__":
    with Pool(min(6, len(sys.argv) - 1)) as pool:
        for r in pool.imap_unordered(build_day, sys.argv[1:]):
            print(r, flush=True)
