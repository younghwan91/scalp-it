"""83번 특징·라벨 — 틱 단위 기관형 체결 + 가속 + 전환점 위치. 사전등록 §2~§4.

    ~/of80/.rlvenv/bin/python scripts/flow_83/features.py 2026-08-24 ... 2026-09-15   # 날짜별 캐시

모든 특징은 결정 초 t 이하 데이터만 쓴다. 라벨만 t+1 이후를 본다.
"""
from __future__ import annotations

import os
import pickle
import sys
from decimal import Decimal
from multiprocessing import Pool
from pathlib import Path

import numba
import numpy as np
import pandas as pd

T0 = 9 * 3600
N = 22800                      # 09:00~15:20 초
ENTRY_LO, ENTRY_HI = 300, 22800 - 600   # 09:05 ~ 15:10
COST = 0.0023
STRUCTS = {"F1": (1, 1, 3), "F2": (1, 2, 0), "F3": (1, 3, 3), "F4": (2, 1, 5)}  # (D, 불꺼짐 종류, 창초) — 사전등록 §4
TOP_RANK = 30
MIN_PRICE = 1000
FORWARD_START = "2026-09-16"
CACHE = Path.home() / "of80" / "flow83"

W = (1, 3, 5, 10)
FEATURES: tuple[str, ...] = (
    *[f"n_tr{w}" for w in W], *[f"buyval{w}" for w in W], *[f"sellval{w}" for w in W],
    *[f"maxbuy{w}" for w in W], "bigbuy_share10", "bigbuy_share3",
    "same_size_cnt10", "same_size_val10", "burst_secs10", "sweep_ticks1", "sweep_ticks3", "persist10",
    *[f"dstr{k}" for k in (1, 3, 5, 10, 30)], *[f"ofi{k}" for k in (1, 3, 5, 10)],
    "bidq1_chg1", "bidq1_chg3", "askq1_depl1", "askq1_depl3", "val_accel5",
    *[f"ret{k}" for k in (1, 3, 5, 10, 30, 300)],
    "dist_hi60", "dist_lo60", "dist_hi300", "dist_lo300", "since_hi60", "since_lo60", "rebound_lo30",
    "spread_bp", "spread_ticks", "imb3", "cum_rank", "log_cumval", "sec",
)


def _tick_np(prices: np.ndarray) -> np.ndarray:
    from kiwoom_client.tick_size import tick_size

    u, inv = np.unique(np.nan_to_num(prices, nan=1000.0), return_inverse=True)
    t = np.array([float(tick_size(Decimal(int(max(p, 1))))) for p in u])
    return t[inv]


def ffill(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    if not ok.any():
        return a
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[: np.argmax(ok)] = np.nan
    return out


def rsum(a: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))])
    out = np.zeros(len(a))
    out[w - 1:] = c[w:] - c[:-w]
    out[: w - 1] = c[1:w]
    return out


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


@numba.njit(cache=True)
def _per_second(sec, price, vol, side, n):
    n_tr = np.zeros(n)
    buyval = np.zeros(n)
    sellval = np.zeros(n)
    maxbuy = np.zeros(n)
    n_buy = np.zeros(n)
    bhi = np.full(n, np.nan)
    blo = np.full(n, np.nan)
    for i in range(len(sec)):
        s = sec[i]
        v = price[i] * vol[i]
        n_tr[s] += 1
        if side[i] > 0:
            buyval[s] += v
            n_buy[s] += 1
            if v > maxbuy[s]:
                maxbuy[s] = v
            if not (bhi[s] == bhi[s]) or price[i] > bhi[s]:
                bhi[s] = price[i]
            if not (blo[s] == blo[s]) or price[i] < blo[s]:
                blo[s] = price[i]
        elif side[i] < 0:
            sellval[s] += v
    return n_tr, buyval, sellval, maxbuy, n_buy, bhi, blo


@numba.njit(cache=True)
def _window_trade_stats(sec, price, vol, side, n, big_thr_by_min):
    """10초 창 같은 수량 반복·큰 체결 매수 비중(10·3초). big_thr 는 분 단위(직전 10분 중앙값×20)."""
    same_cnt = np.zeros(n)
    same_val = np.zeros(n)
    big10 = np.zeros(n)
    big3 = np.zeros(n)
    m = len(sec)
    lo = 0
    hi = 0
    for s in range(n):
        while hi < m and sec[hi] <= s:
            hi += 1
        while lo < hi and sec[lo] < s - 9:
            lo += 1
        if hi == lo:
            continue
        thr = big_thr_by_min[s // 60]
        # 큰 체결
        tot10 = 0.0
        b10 = 0.0
        tot3 = 0.0
        b3 = 0.0
        nb = 0
        vols = np.empty(hi - lo)
        pxs = np.empty(hi - lo)
        for i in range(lo, hi):
            if side[i] > 0:
                v = price[i] * vol[i]
                tot10 += v
                if v >= thr:
                    b10 += v
                if sec[i] >= s - 2:
                    tot3 += v
                    if v >= thr:
                        b3 += v
                vols[nb] = vol[i]
                pxs[nb] = price[i]
                nb += 1
        if tot10 > 0:
            big10[s] = b10 / tot10
        if tot3 > 0:
            big3[s] = b3 / tot3
        if nb >= 2:
            vs = np.sort(vols[:nb])
            best = 1
            run = 1
            bestv = vs[0]
            for j in range(1, nb):
                if vs[j] == vs[j - 1]:
                    run += 1
                else:
                    run = 1
                if run > best:
                    best = run
                    bestv = vs[j]
            same_cnt[s] = best
            same_val[s] = best * bestv * pxs[nb - 1]
    return same_cnt, same_val, big10, big3


@numba.njit(cache=True)
def _roll_extreme(x, w, want_max):
    """최근 w초(포함) 극값과 그 뒤 경과 초 — 단조 덱."""
    n = len(x)
    val = np.full(n, np.nan)
    since = np.full(n, np.nan)
    dq = np.empty(n, np.int64)
    h = 0
    t = 0
    for i in range(n):
        xi = x[i]
        if xi == xi:
            while t > h:
                last = x[dq[t - 1]]
                if (want_max and last <= xi) or ((not want_max) and last >= xi):
                    t -= 1
                else:
                    break
            dq[t] = i
            t += 1
        while t > h and dq[h] <= i - w:
            h += 1
        if t > h:
            val[i] = x[dq[h]]
            since[i] = i - dq[h]
    return val, since


@numba.njit(cache=True)
def _labels(rows, bid, ask, buyval, sellval, stg, e_ofi, tick_at, D, kind, win, cost):
    """시간 청산 없음 — 손절 또는 불 꺼짐(kind 1 매도우위 / 2 체결강도 꺾임 / 3 매도우위+OFI<0), 장 마감."""
    n = len(bid)
    out = np.full(len(rows), np.nan)
    hold = np.zeros(len(rows), np.int64)
    for r in range(len(rows)):
        e = rows[r] + 1
        if e + 1 >= n:
            continue
        a = ask[e]
        b0 = bid[e]
        if not (a > 0 and b0 > 0):
            continue
        stop = b0 - D * tick_at[e]
        peak = stg[e]
        px = bid[n - 1]
        xs = n - 1
        for s in range(e + 1, n):
            if bid[s] <= stop:
                px = bid[s]
                xs = s
                break
            out_now = False
            if kind == 1 or kind == 3:
                bsum = 0.0
                ssum = 0.0
                osum = 0.0
                for j in range(max(e + 1, s - win + 1), s + 1):
                    bsum += buyval[j]
                    ssum += sellval[j]
                    osum += e_ofi[j]
                if s - e >= win:
                    if kind == 1 and bsum <= ssum:
                        out_now = True
                    if kind == 3 and bsum <= ssum and osum < 0:
                        out_now = True
            if kind == 2:
                v = stg[s]
                if v == v:
                    if not (peak == peak) or v > peak:
                        peak = v
                    if v <= peak - 1.0:
                        out_now = True
            if out_now:
                px = bid[s]
                xs = s
                break
        if px > 0:
            out[r] = px / a - 1.0 - cost
            hold[r] = xs - e
    return out, hold


def build_code(tk: pd.DataFrame, qt: pd.DataFrame) -> tuple[dict, dict]:
    sec = (tk.ts.dt.hour * 3600 + tk.ts.dt.minute * 60 + tk.ts.dt.second).to_numpy() - T0
    m = (sec >= 0) & (sec < N)
    tk, sec = tk[m], sec[m].astype(np.int64)
    price = tk.price.to_numpy(np.float64)
    vol = tk.volume.to_numpy(np.float64)
    side = tk.side.to_numpy(np.int64)
    n_tr, buyval, sellval, maxbuy, n_buy, bhi, blo = _per_second(sec, price, vol, side, N)
    tv = price * vol
    # 분 단위 큰 체결 문턱: 직전 10분 체결 1건 금액 중앙값 × 20
    thr = np.full(N // 60 + 1, np.inf)
    for mi in range(N // 60 + 1):
        s0 = mi * 60
        a, b = np.searchsorted(sec, s0 - 600), np.searchsorted(sec, s0)
        if b - a >= 20:
            thr[mi] = float(np.median(tv[a:b])) * 20.0
    same_cnt, same_val, big10, big3 = _window_trade_stats(sec, price, vol, side, N, thr)

    stg = np.full(N, np.nan)
    stg[sec] = tk.strength.to_numpy(np.float64)
    last_px = np.full(N, np.nan)
    last_px[sec] = price
    hi = np.full(N, np.nan)
    s_hi = pd.Series(price).groupby(sec).max()
    hi[s_hi.index.to_numpy()] = s_hi.to_numpy()
    bb = np.full(N, np.nan)
    ba = np.full(N, np.nan)
    bb[sec] = tk.best_bid.to_numpy(np.float64)
    ba[sec] = tk.best_ask.to_numpy(np.float64)
    bq1 = np.full(N, np.nan)
    aq1 = np.full(N, np.nan)
    bd3 = np.full(N, np.nan)
    ad3 = np.full(N, np.nan)
    if len(qt):
        qs = (qt.ts.dt.hour * 3600 + qt.ts.dt.minute * 60 + qt.ts.dt.second).to_numpy() - T0
        mq = (qs >= 0) & (qs < N)
        qt, qs = qt[mq], qs[mq]
        b1 = qt.bid1.to_numpy(np.float64)
        a1 = qt.ask1.to_numpy(np.float64)
        ok = (b1 > 0) & (a1 > 0)
        bb[qs[ok]] = b1[ok]
        ba[qs[ok]] = a1[ok]
        bq1[qs[ok]] = qt.bidqty1.to_numpy(np.float64)[ok]
        aq1[qs[ok]] = qt.askqty1.to_numpy(np.float64)[ok]
        bd3[qs[ok]] = qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(np.float64)[ok] * b1[ok]
        ad3[qs[ok]] = qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(np.float64)[ok] * a1[ok]
    bb, ba, stg = ffill(bb), ffill(ba), ffill(stg)
    bq1, aq1, bd3, ad3 = ffill(bq1), ffill(aq1), ffill(bd3), ffill(ad3)
    bb[bb <= 0] = np.nan
    ba[ba <= 0] = np.nan
    mid = (bb + ba) / 2
    tick_at = _tick_np(ba)

    pb, pa, pbq, paq = lag(bb, 1), lag(ba, 1), lag(bq1, 1), lag(aq1, 1)
    e = np.nan_to_num((bb >= pb) * bq1 - (bb <= pb) * pbq - (ba <= pa) * aq1 + (ba >= pa) * paq)
    depth = pd.Series(np.nan_to_num((bq1 + aq1) / 2)).rolling(60, min_periods=1).mean().to_numpy() + 1

    val = buyval + sellval
    f: dict[str, np.ndarray] = {}
    for w in W:
        f[f"n_tr{w}"] = rsum(n_tr, w)
        f[f"buyval{w}"] = np.log1p(rsum(buyval, w))
        f[f"sellval{w}"] = np.log1p(rsum(sellval, w))
        f[f"maxbuy{w}"] = np.log1p(pd.Series(maxbuy).rolling(w, min_periods=1).max().to_numpy())
    f["bigbuy_share10"], f["bigbuy_share3"] = big10, big3
    f["same_size_cnt10"], f["same_size_val10"] = same_cnt, np.log1p(same_val)
    f["burst_secs10"] = rsum((n_buy >= 5).astype(float), 10)
    sw = np.nan_to_num((bhi - blo) / tick_at)
    f["sweep_ticks1"] = sw
    f["sweep_ticks3"] = pd.Series(sw).rolling(3, min_periods=1).max().to_numpy()
    f["persist10"] = rsum((n_buy > 0).astype(float), 10)
    for k in (1, 3, 5, 10, 30):
        f[f"dstr{k}"] = stg - lag(stg, k)
    for k in (1, 3, 5, 10):
        f[f"ofi{k}"] = rsum(e, k) / depth
    f["bidq1_chg1"] = (bq1 - lag(bq1, 1)) / (lag(bq1, 1) + 1)
    f["bidq1_chg3"] = (bq1 - lag(bq1, 3)) / (lag(bq1, 3) + 1)
    f["askq1_depl1"] = (lag(aq1, 1) - aq1) / (lag(aq1, 1) + 1)
    f["askq1_depl3"] = (lag(aq1, 3) - aq1) / (lag(aq1, 3) + 1)
    f["val_accel5"] = rsum(val, 5) / (rsum(val, 300) / 60 + 1e5)
    for k in (1, 3, 5, 10, 30, 300):
        f[f"ret{k}"] = (mid / lag(mid, k) - 1) * 1e4
    for w in (60, 300):
        hv, hs = _roll_extreme(mid, w, True)
        lv, ls = _roll_extreme(mid, w, False)
        f[f"dist_hi{w}"] = (mid / hv - 1) * 1e4
        f[f"dist_lo{w}"] = (mid / lv - 1) * 1e4
        if w == 60:
            f["since_hi60"], f["since_lo60"] = hs, ls
    lv30, _ = _roll_extreme(mid, 30, False)
    f["rebound_lo30"] = (mid / lv30 - 1) * 1e4
    f["spread_bp"] = (ba - bb) / mid * 1e4
    f["spread_ticks"] = (ba - bb) / tick_at
    f["imb3"] = (bd3 - ad3) / (bd3 + ad3)
    f["cumval"] = np.cumsum(val)
    f["n_tr_sec"] = n_tr
    path = {"bid": bb, "ask": ba, "hi": hi, "tick": tick_at, "last": ffill(last_px),
            "buyval": buyval, "sellval": sellval, "strength": stg, "e_ofi": e}
    return f, path


def bad_days() -> dict:
    """수집이 덜 된 거래일(research/bad_days.json). 원본은 지우지 않고 **여기서 뺀다**.

    부분 수집된 날은 틱이 있어도 하루의 일부뿐이라 표본에 섞이면 조용히 편향된다
    (2026-08-31 은 11:40 에 수집이 죽었고 분커버 41% 였다).
    """
    import json

    f = Path(__file__).resolve().parents[2] / "research" / "bad_days.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))["days"]
    except (OSError, KeyError, ValueError):
        return {}


def build_day(conn, day: str, *, forward_ok: bool = False) -> Path:
    if not forward_ok:
        assert day < FORWARD_START, "학습 코드가 전진 OOS 날짜를 읽으려 했다(§1)"
    bad = bad_days()
    if day in bad:
        print(f"{day} 건너뜀 — 수집 불량(분커버 {bad[day]['cover']:.0%}): {bad[day]['why']}", flush=True)
        return CACHE / f"{day}.pkl"
    out = CACHE / f"{day}.pkl"
    if out.exists():
        return out
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    stocks = pd.read_sql("select code from stocks", conn).code.tolist()
    tk = pd.read_sql("select code, ts, seq, price, volume, side, strength, best_bid, best_ask from ticks "
                     "where ts >= %s and ts < %s and code = any(%s) order by code, ts, seq",
                     conn, params=(day, nxt, stocks))
    qt = pd.read_sql("select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
                     "from quotes where ts >= %s and ts < %s and code = any(%s) order by code, ts",
                     conn, params=(day, nxt, stocks))
    qg = dict(tuple(qt.groupby("code")))
    per: dict[str, tuple[dict, dict]] = {}
    for code, g in tk.groupby("code"):
        if len(g) < 500:
            continue
        per[code] = build_code(g, qg.get(code, qt.iloc[:0]))
    codes = sorted(per)
    cum = np.stack([per[c][0]["cumval"] for c in codes])          # (codes, N)
    rank = (-cum).argsort(axis=0).argsort(axis=0) + 1               # 초마다 누적 체결대금 순위
    rows_all = []
    paths = {}
    for ci, code in enumerate(codes):
        f, p = per[code]
        f["cum_rank"] = rank[ci].astype(float)
        f["log_cumval"] = np.log1p(f["cumval"])
        f["sec"] = np.arange(N, dtype=float)
        s = np.arange(N)
        ok = ((f["n_tr_sec"] > 0) & (s >= ENTRY_LO) & (s <= ENTRY_HI) & (f["cum_rank"] <= TOP_RANK)
              & (p["last"] >= MIN_PRICE) & (p["ask"] > p["bid"]) & np.isfinite(p["ask"]) & np.isfinite(p["bid"]))
        rows = np.nonzero(ok)[0]
        if len(rows) == 0:
            continue
        df = pd.DataFrame({k: f[k][rows].astype(np.float32) for k in FEATURES})
        df.insert(0, "row", rows.astype(np.int32))
        df.insert(0, "code", code)
        for name, (D, kind, win) in STRUCTS.items():
            lab, hold = _labels(rows, p["bid"], p["ask"], p["buyval"], p["sellval"], p["strength"],
                                p["e_ofi"], p["tick"], D, kind, win, COST)
            df[f"y_{name}"] = lab.astype(np.float32)
            df[f"hold_{name}"] = hold.astype(np.int32)
        rows_all.append(df)
        paths[code] = {k: v.astype(np.float32) for k, v in p.items()}
    data = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    data.insert(0, "day", day)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps({"rows": data, "paths": paths}, protocol=5))
    return out


def _job(day: str) -> str:
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    try:
        p = build_day(conn, day, forward_ok=os.environ.get("FLOW83_FORWARD") == "1")
        return f"{day} {p.stat().st_size // 1_000_000}MB"
    finally:
        conn.close()


if __name__ == "__main__":
    with Pool(min(6, len(sys.argv) - 1)) as pool:
        for r in pool.imap_unordered(_job, sys.argv[1:]):
            print(r, flush=True)
