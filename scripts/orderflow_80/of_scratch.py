"""80-2: 집중 종목 스크래치 스캘핑 — 사고 S초 안에 안 오르면 즉시 매도, 오르면 +k호가 익절.

종목: 전일 거래대금(v600 중앙) 상위 N. 종목당 포지션 1개, 청산 후 10초 쿨다운.
usage: python of_scratch.py START END OUT.csv
"""
import glob
import itertools
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

COST = 0.0023
TOPN = 10
COOL = 10
TPK = (2, 3, 4, 6)
SCR = (1, 2, 3, 5)
HOLD = (30, 120)
STOPK = 2
EXITS = list(itertools.product(TPK, SCR, HOLD))


def tick(p):
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return t
    return 1000


SIGNALS = {
    "flow": lambda d: (d.buyshare10 >= 0.7) & (d.ofi10 > 0),
    "flow_acc": lambda d: (d.acc >= 3) & (d.buyshare10 >= 0.7) & (d.ret30 > 0),
    "book": lambda d: (d.imb3 >= 0.6) & (d.ofi10 > 1),
    "any": lambda d: d.buyshare10 >= 0.0,  # 무조건(대조군)
}


def sim(p, sig_secs, tp_k, scr, hold):
    bid, ask, hi = p["bid"], p["ask"], p["hi"]
    n = len(bid)
    out = []
    free = 0
    for t in sig_secs:
        if t < free:
            continue
        e = t + 1
        if e + hold + 1 >= n:
            break
        a = float(ask[e]); b0 = float(bid[e])
        if not (a > b0 > 0):
            continue
        tk = tick(a)
        tp = a + tp_k * tk
        stop = a - STOPK * tk
        px = None; j = None
        for s in range(e + 1, e + hold + 1):
            if hi[s] == hi[s] and hi[s] > tp:           # 지정가 익절(뚫려야 체결)
                px, j = tp, s; break
            if bid[s] <= stop:                           # 손절
                px, j = float(bid[s]), s; break
            if s == e + scr:
                seen = hi[e + 1:s + 1]
                if not (np.nan_to_num(seen, nan=0) > a).any():  # 바로 안 올랐다 → 스크래치
                    px, j = float(bid[s]), s; break
        if px is None:
            j = e + hold; px = float(bid[j])
        out.append((t, px / a - 1 - COST, j - e, tk / a * 1e4))
        free = j + COOL
    return out


def run_day(args):
    f, ranks = args
    day = f.split("/")[-1][:10]
    df = pickle.load(open(f, "rb"))
    paths = pickle.load(open(f.replace(".pkl", ".path.pkl"), "rb"))
    rows = []
    for rank, code in enumerate(ranks.get(day, [])):
        if code not in paths:
            continue
        d = df[df.code == code]
        for sname, fn in SIGNALS.items():
            secs = d.loc[fn(d), "sec"].to_numpy()
            for k, (tp_k, scr, hold) in enumerate(EXITS):
                for t, r, h, tbp in sim(paths[code], secs, tp_k, scr, hold):
                    rows.append((day, code, rank, sname, tp_k, scr, hold, r, h, tbp))
    return pd.DataFrame(rows, columns=["day", "code", "rank", "sig", "tp", "scr", "hold", "net", "held", "tick_bp"])


if __name__ == "__main__":
    start, end, out = sys.argv[1:4]
    files = [f for f in sorted(glob.glob("/home/young/of80/cache/*.pkl")) if not f.endswith(".path.pkl")]
    days = [f.split("/")[-1][:10] for f in files]
    liq = {}
    for f, d in zip(files, days):
        x = pickle.load(open(f, "rb"))
        liq[d] = x.groupby("code").v600.median().sort_values(ascending=False)
    ranks = {}
    for i in range(1, len(days)):
        # 전일 상위이면서 오늘도 데이터가 있는 종목(전일 정보만으로 고른다)
        ranks[days[i]] = [c for c in liq[days[i - 1]].index[:TOPN * 3]][:TOPN]
    todo = [(f, ranks) for f, d in zip(files, days) if start <= d <= end and d in ranks]
    with Pool(8) as pool:
        res = pool.map(run_day, todo)
    t = pd.concat(res, ignore_index=True)
    t.to_csv(out, index=False)
    print("trades", len(t))
