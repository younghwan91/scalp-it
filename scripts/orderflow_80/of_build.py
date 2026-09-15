"""80번 탐색: 호가·체결 흐름 특징 × 시장가 진입 N초 뒤 매수1호가 청산 순수익. 날짜별 캐시.

usage: python of_build.py OUTDIR day1 day2 ...
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psycopg2

COST = 0.0023
HS = (5, 15, 30, 60, 120, 300)
T0, T1 = 9 * 3600, 15 * 3600 + 20 * 60  # 09:00~15:20
N = T1 - T0


def _sec(ts):
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy() - T0


def ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(~np.isnan(a)) if (~np.isnan(a)).any() else len(a)
    out[:first] = np.nan
    return out


def roll_sum(a, w):
    c = np.concatenate([[0.0], np.cumsum(a)])
    out = np.full(len(a), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def fwd_max(a, h):
    # max over (t, t+h]
    s = pd.Series(a[::-1]).rolling(h, min_periods=1).max().to_numpy()[::-1]
    out = np.full(len(a), np.nan)
    out[:-1] = s[1:]
    # s[i+1] = max a[i+1 .. i+h]
    return out


def build_code(tk, qt):
    n = N
    sec_t = _sec(tk.ts)
    m = (sec_t >= 0) & (sec_t < n)
    tk = tk[m]
    sec_t = sec_t[m]
    val = (tk.price * tk.volume).to_numpy(float)
    side = tk.side.to_numpy()
    buy = np.bincount(sec_t, weights=np.where(side > 0, val, 0), minlength=n)
    sell = np.bincount(sec_t, weights=np.where(side < 0, val, 0), minlength=n)
    cnt = np.bincount(sec_t, minlength=n)
    last = np.full(n, np.nan)
    last[sec_t] = tk.price.to_numpy(float)  # 마지막 값이 남는다(정렬됨)
    bb = np.full(n, np.nan); ba = np.full(n, np.nan)
    bb[sec_t] = tk.best_bid.to_numpy(float); ba[sec_t] = tk.best_ask.to_numpy(float)

    bq1 = np.full(n, np.nan); aq1 = np.full(n, np.nan)
    bd3 = np.full(n, np.nan); ad3 = np.full(n, np.nan)
    if len(qt):
        sec_q = _sec(qt.ts)
        mq = (sec_q >= 0) & (sec_q < n)
        qt = qt[mq]; sec_q = sec_q[mq]
        # 같은 초면 호가 스냅샷을 틱 best 보다 우선하지 않고, 둘 다 있으면 호가를 쓴다
        b1 = qt.bid1.to_numpy(float); a1 = qt.ask1.to_numpy(float)
        ok = (b1 > 0) & (a1 > 0)
        bb[sec_q[ok]] = b1[ok]; ba[sec_q[ok]] = a1[ok]
        bq1[sec_q[ok]] = qt.bidqty1.to_numpy(float)[ok]
        aq1[sec_q[ok]] = qt.askqty1.to_numpy(float)[ok]
        bd3[sec_q[ok]] = (qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(float) * b1)[ok]
        ad3[sec_q[ok]] = (qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(float) * a1)[ok]
    bb = ffill(bb); ba = ffill(ba); last = ffill(last)
    bq1 = ffill(bq1); aq1 = ffill(aq1); bd3 = ffill(bd3); ad3 = ffill(ad3)
    bb[bb <= 0] = np.nan; ba[ba <= 0] = np.nan
    mid = (bb + ba) / 2

    # OFI(Cont) level1
    pb, pa = np.roll(bb, 1), np.roll(ba, 1)
    pbq, paq = np.roll(bq1, 1), np.roll(aq1, 1)
    e = ((bb >= pb) * bq1 - (bb <= pb) * pbq - (ba <= pa) * aq1 + (ba >= pa) * paq)
    e[0] = 0
    e = np.nan_to_num(e)
    depth1 = np.nan_to_num((bq1 + aq1) / 2)

    tot = buy + sell
    v30 = roll_sum(tot, 30)
    v600 = roll_sum(tot, 600)
    base = np.full(n, np.nan)
    base[30:] = v600[:-30] / 20.0  # 30초 창 20개 평균(직전 30초 제외)
    f = {
        "sec": np.arange(n),
        "cnt": cnt,
        "acc": v30 / (base + 1e6),
        "v30": v30,
        "v600": v600,
        "buyshare30": roll_sum(buy, 30) / (v30 + 1),
        "buyshare10": roll_sum(buy, 10) / (roll_sum(tot, 10) + 1),
        "ret30": mid / np.roll(mid, 30) - 1,
        "ret300": mid / np.roll(mid, 300) - 1,
        "ofi10": roll_sum(e, 10) / (pd.Series(depth1).rolling(60, min_periods=1).mean().to_numpy() + 1),
        "imb3": (bd3 - ad3) / (bd3 + ad3),
        "spread_bp": (ba - bb) / mid * 1e4,
        "bid": bb, "ask": ba,
    }
    ask_e = np.roll(ba, -1)  # 1초 지연 진입
    for h in HS:
        exit_b = np.full(n, np.nan)
        exit_b[: n - 1 - h] = bb[1 + h:]
        f[f"net{h}"] = exit_b / ask_e - 1 - COST
        mx = fwd_max(bb, h)
        mfe = np.full(n, np.nan)
        mfe[: n - 1] = mx[1:]
        f[f"mfe{h}"] = mfe / ask_e - 1 - COST
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    pr = tk.price.to_numpy(float)
    lo_s = pd.Series(pr).groupby(sec_t).min(); hi_s = pd.Series(pr).groupby(sec_t).max()
    lo[lo_s.index.to_numpy()] = lo_s.to_numpy(); hi[hi_s.index.to_numpy()] = hi_s.to_numpy()
    path = {"bid": bb.astype(np.float32), "ask": ba.astype(np.float32),
            "lo": lo.astype(np.float32), "hi": hi.astype(np.float32),
            "bidval1": np.nan_to_num(bq1 * bb).astype(np.float32),
            "bidval3": np.nan_to_num(bd3).astype(np.float32),
            "ofi": np.cumsum(e).astype(np.float32)}
    df = pd.DataFrame(f)
    keep = (cnt > 0) & (df.sec >= 300) & (df.sec <= N - 600) & df.ask.notna() & df.bid.notna() \
        & (df.ask > df.bid) & df.acc.notna()
    df = df[keep]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df, path


def run_day(args):
    out, day = args
    path = f"{out}/{day}.pkl"
    if os.path.exists(f"{out}/{day}.path.pkl"):
        return day, "cached"
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    tk = pd.read_sql(
        "select code, ts, seq, price, volume, side, best_bid, best_ask from ticks "
        "where ts >= %s and ts < %s order by code, ts, seq", conn, params=(day, nxt))
    qt = pd.read_sql(
        "select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
        "from quotes where ts >= %s and ts < %s order by code, ts", conn, params=(day, nxt))
    conn.close()
    parts = []
    paths = {}
    qg = dict(tuple(qt.groupby("code")))
    for code, g in tk.groupby("code"):
        if len(g) < 200:
            continue
        d, paths[code] = build_code(g, qg.get(code, qt.iloc[:0]))
        d.insert(0, "code", code)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "day", day)
    with open(path, "wb") as fh:
        pickle.dump(df, fh, protocol=5)
    with open(f"{out}/{day}.path.pkl", "wb") as fh:
        pickle.dump(paths, fh, protocol=5)
    return day, len(df)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    days = sys.argv[2:]
    with Pool(min(6, len(days))) as p:
        for r in p.imap_unordered(run_day, [(out, d) for d in days]):
            print(r, flush=True)
