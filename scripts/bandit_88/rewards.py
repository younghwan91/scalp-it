"""88번 팔 보상 사전계산 — 결정 행마다 팔1(시장가 F2)·팔2(대기열 지정가 + 86번 청산) 보상과 종목이 풀리는 초.

    FLOW88_FORWARD=0 ~/of80/.rlvenv/bin/python scripts/bandit_88/rewards.py 2026-08-24 ... 2026-09-15
결과 `~/of80/bandit88/<day>.pkl` = DataFrame(day, code, row, 13피처, y1, free1, y2, free2, filled2)
"""
from __future__ import annotations

import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numba
import numpy as np
import pandas as pd

HOME = Path.home() / "of80"
OUT = HOME / "bandit88"
FORWARD_START = "2026-09-16"
STOP_TICKS, SDROP, CANCEL_SECS, COST = 2, 1.0, 10, 0.0023


@numba.njit
def _limit_exit(rows, fill_sec, P, ptick, bid, stg, n, cost):
    m = len(rows)
    y = np.zeros(m)
    free = np.zeros(m, np.int64)
    for k in range(m):
        e = fill_sec[k]
        if e < 0 or not (P[k] > 0):
            free[k] = rows[k] + 1 + CANCEL_SECS
            continue
        stop = P[k] - STOP_TICKS * ptick[k]
        peak = stg[e]
        px = bid[n - 1]
        xs = n - 1
        for s in range(e + 1, n):
            if bid[s] <= stop:
                px = bid[s]
                xs = s
                break
            v = stg[s]
            if v == v:
                if not (peak == peak) or v > peak:
                    peak = v
                if v <= peak - SDROP:
                    px = bid[s]
                    xs = s
                    break
        y[k] = (px / P[k] - 1.0 - cost) * 1e4
        free[k] = xs + 1
    return y, free


def build(day: str) -> str:
    if os.environ.get("FLOW88_FORWARD") != "1":
        assert day < FORWARD_START
    out = OUT / f"{day}.pkl"
    if out.exists():
        return f"{day} cached"
    feats = pickle.loads((HOME / "flow84_model" / "model.pkl").read_bytes())["features"]
    rows = pickle.loads((HOME / "flow84" / f"{day}.pkl").read_bytes())
    paths = pickle.loads((HOME / "flow83" / f"{day}.pkl").read_bytes())["paths"]
    fills = pickle.loads((HOME / "fills87" / f"{day}.pkl").read_bytes())
    parts = []
    for code, g in rows.groupby("code"):
        if code not in fills:
            continue
        g = g.sort_values("row")
        frows, fsec, fP, ftick = fills[code]
        p = paths[code]
        n = len(p["bid"])
        y2, free2 = _limit_exit(frows, fsec, fP, ftick, p["bid"].astype(np.float64), p["strength"].astype(np.float64),
                                n, COST)
        d = g[["day", "code", "row", *feats]].copy()
        d["y1"] = (g.y_F2.to_numpy(np.float64) * 1e4)
        d["free1"] = g.row.to_numpy(np.int64) + 2 + g.hold_F2.to_numpy(np.int64)
        d["y2"], d["free2"], d["filled2"] = y2, free2, fsec >= 0
        parts.append(d[np.isfinite(d.y1)])
    df = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(df, protocol=5))
    f = df[df.filled2]
    return f"{day} rows {len(df)} · 팔1 평균 {df.y1.mean():+.1f}bp · 팔2 체결 {df.filled2.mean():.0%} 체결시 {f.y2.mean():+.1f}bp"


if __name__ == "__main__":
    with Pool(4) as pool:
        for r in pool.imap_unordered(build, sys.argv[1:]):
            print(r, flush=True)
