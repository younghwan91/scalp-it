"""80번 탐색 2: 신호 × 진입(시장가/지정가) × 청산(익절·손절·매수벽붕괴·시간) 경로 시뮬레이션.

usage: python of_sim.py START END OUT.pkl
"""
import glob
import itertools
import pickle
import sys
from datetime import date as _date
from multiprocessing import Pool

import numpy as np
import pandas as pd
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.codes import Market

COST = round_trip_cost(_date(2026, 1, 2), Market.KOSDAQ)  # 2026 세율 — 날짜 문맥 없음
COOLDOWN = 60
SIGNALS = [dict(acc=a, bs=b, r=r, spr=15.0)
           for a in (5, 10) for b in (0.75, 0.9) for r in (0.005, 0.01)]
MODES = ("taker", "maker")
MAKER_WAIT = 10
TPS = (0.004, 0.008, 0.015, None)
SLS = (0.003, 0.006)
WALLS = (None, 0.3)
TS = (60, 300)
EXITS = list(itertools.product(TPS, SLS, WALLS, TS))


def sig_key(s):
    return f"acc{s['acc']}_bs{s['bs']}_r{s['r']}_spr{s['spr']:g}"


def cooldown(secs):
    keep, last = [], -10**9
    for s in secs:
        if s - last >= COOLDOWN:
            keep.append(s)
            last = s
    return keep


def sim_exit(entry, e, p, tp, sl, wall, T):
    n = len(p["bid"])
    end = min(e + T, n - 1)
    bid = p["bid"][e + 1:end + 1].astype(float)
    hi = p["hi"][e + 1:end + 1].astype(float)
    if len(bid) == 0 or np.isnan(bid).all():
        return None, None
    L = len(bid)
    cand = []  # (index, price)
    if tp is not None:
        px = entry * (1 + tp)
        hit = np.nan_to_num(hi, nan=-1) > px
        if hit.any():
            i = int(np.argmax(hit)); cand.append((i, px))
    s_hit = np.nan_to_num(bid, nan=np.inf) <= entry * (1 - sl)
    if s_hit.any():
        i = int(np.argmax(s_hit)); cand.append((i, bid[i]))
    if wall is not None:
        w0 = p["bidval1"][e]
        b0 = p["bid"][e]
        wv = p["bidval1"][e + 1:end + 1]
        w_hit = (wv < wall * w0) & (np.nan_to_num(bid, nan=np.inf) <= b0)
        if w0 > 0 and w_hit.any():
            i = int(np.argmax(w_hit)); cand.append((i, bid[i]))
    if cand:
        i, px = min(cand, key=lambda x: x[0])
    else:
        i = L - 1
        px = bid[i] if not np.isnan(bid[i]) else np.nanmax(bid)
    return px / entry - 1 - COST, i + 1


def run_day(path_feat):
    day = path_feat.split("/")[-1][:10]
    df = pickle.load(open(path_feat, "rb"))
    paths = pickle.load(open(path_feat.replace(".pkl", ".path.pkl"), "rb"))
    members = {}
    events = set()
    for s in SIGNALS:
        m = (df.acc >= s["acc"]) & (df.buyshare30 >= s["bs"]) & (df.ret30 > s["r"]) & (df.spread_bp <= s["spr"])
        g = df.loc[m, ["code", "sec"]]
        ev = []
        for code, gg in g.groupby("code"):
            ev += [(code, x) for x in cooldown(gg.sec.tolist())]
        members[sig_key(s)] = ev
        events.update(ev)
    rows = []
    for code, t in events:
        p = paths[code]
        n = len(p["bid"])
        for mode in MODES:
            if mode == "taker":
                e = t + 1
                if e >= n - 2 or np.isnan(p["ask"][e]):
                    continue
                entry = float(p["ask"][e])
            else:
                lim = float(p["bid"][t])
                lo = p["lo"][t + 1:t + 1 + MAKER_WAIT]
                hit = np.nan_to_num(lo, nan=np.inf) < lim
                if np.isnan(lim) or not hit.any():
                    rows.append(dict(day=day, code=code, sec=t, mode=mode, filled=False))
                    continue
                e = t + 1 + int(np.argmax(hit))
                entry = lim
            rec = dict(day=day, code=code, sec=t, mode=mode, filled=True)
            for k, (tp, sl, wall, T) in enumerate(EXITS):
                r, hold = sim_exit(entry, e, p, tp, sl, wall, T)
                rec[k] = r
            rows.append(rec)
    return day, members, pd.DataFrame(rows)


if __name__ == "__main__":
    start, end, out = sys.argv[1:4]
    files = [f for f in sorted(glob.glob("/home/young/of80/cache/*.pkl"))
             if not f.endswith(".path.pkl") and start <= f.split("/")[-1][:10] <= end]
    with Pool(8) as pool:
        res = pool.map(run_day, files)
    pickle.dump(dict(res=res, exits=EXITS, signals=[sig_key(s) for s in SIGNALS]), open(out, "wb"))
    print("saved", out, sum(len(r[2]) for r in res))
