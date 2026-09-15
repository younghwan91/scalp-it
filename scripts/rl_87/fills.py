"""87번 — 결정 행마다 대기열 지정가 체결 초 미리 계산. 86번 §3 모형(10초·불꺼짐 취소).

    FLOW87_FORWARD=0 ~/of80/.rlvenv/bin/python scripts/rl_87/fills.py 2026-08-24 ... 2026-09-15
결과 `~/of80/fills87/<day>.pkl` = {code: (rows, fill_sec(-1=미체결), P, tick)}
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flow_83"))
import features as F83  # noqa: E402

OUT = Path.home() / "of80" / "fills87"
CANCEL_SECS, SDROP = 10, 1.0


@numba.njit
def _fills(rows, t_px, t_vol, t_side, t_start, bid, bq1, stg, n):
    m = len(rows)
    fill = np.full(m, -1, np.int64)
    P_out = np.full(m, np.nan)
    for k in range(m):
        j = rows[k] + 1
        if j + 2 >= n:
            continue
        P = bid[j]
        Qa = bq1[j]
        if not (P > 0) or not (Qa == Qa):
            continue
        P_out[k] = P
        peak = stg[j]
        cum = 0.0
        for s in range(j + 1, min(j + CANCEL_SECS, n - 1) + 1):
            hit = False
            for i in range(t_start[s], t_start[s + 1]):
                if t_px[i] < P:
                    hit = True
                    break
                if t_side[i] < 0 and t_px[i] == P:
                    cum += t_vol[i]
            if hit or cum >= Qa + 1:
                fill[k] = s
                break
            v = stg[s]
            if v == v:
                if not (peak == peak) or v > peak:
                    peak = v
                if v <= peak - SDROP:
                    break
    return fill, P_out


def build(day: str) -> str:
    if os.environ.get("FLOW87_FORWARD") != "1":
        assert day < F83.FORWARD_START
    out = OUT / f"{day}.pkl"
    if out.exists():
        return f"{day} cached"
    import psycopg2

    rows84 = pickle.loads((Path.home() / "of80" / "flow84" / f"{day}.pkl").read_bytes())
    paths = pickle.loads((F83.CACHE / f"{day}.pkl").read_bytes())["paths"]
    codes = sorted(rows84.code.unique())
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    tk = pd.read_sql("select code, ts, price, volume, side from ticks where ts >= %s and ts < %s and code = any(%s) "
                     "order by code, ts, seq", conn, params=(day, nxt, codes))
    qt = pd.read_sql("select code, ts, bidqty1 from quotes where ts >= %s and ts < %s and code = any(%s) order by code, ts",
                     conn, params=(day, nxt, codes))
    conn.close()
    tg, qg = dict(tuple(tk.groupby("code"))), dict(tuple(qt.groupby("code")))
    N = F83.N
    res = {}
    for c in codes:
        g = tg.get(c)
        if g is None:
            continue
        sec = (g.ts.dt.hour * 3600 + g.ts.dt.minute * 60 + g.ts.dt.second).to_numpy() - F83.T0
        ok = (sec >= 0) & (sec < N)
        sec = sec[ok].astype(np.int64)
        start = np.searchsorted(sec, np.arange(N + 1))
        bq = np.full(N, np.nan)
        q = qg.get(c)
        if q is not None:
            qs = (q.ts.dt.hour * 3600 + q.ts.dt.minute * 60 + q.ts.dt.second).to_numpy() - F83.T0
            mq = (qs >= 0) & (qs < N)
            bq[qs[mq]] = q.bidqty1.to_numpy(float)[mq]
        bq = F83.ffill(bq)
        p = paths[c]
        rows = np.sort(rows84.loc[rows84.code == c, "row"].to_numpy(np.int64))
        f, P = _fills(rows, g.price.to_numpy(float)[ok], g.volume.to_numpy(float)[ok], g.side.to_numpy(np.int64)[ok],
                      start, p["bid"].astype(np.float64), bq, p["strength"].astype(np.float64), N)
        res[c] = (rows, f, P, p["tick"][np.minimum(rows + 1, N - 1)].astype(np.float64))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(res, protocol=5))
    tot = sum(len(v[0]) for v in res.values())
    fil = sum(int((v[1] >= 0).sum()) for v in res.values())
    return f"{day} rows {tot} filled {fil} ({fil / max(tot, 1):.0%})"


if __name__ == "__main__":
    with Pool(4) as pool:
        for r in pool.imap_unordered(build, sys.argv[1:]):
            print(r, flush=True)
