"""94번 캐시 — 83·84번 가공을 그대로 쓰되 **결정 행 필터만 푼다**(상위30·1,000원·09:05~15:10 제거).
사전등록 docs/research/manju/94 §1. 이어서 지정가 체결 초(86 §3 모형)도 같은 행에 맞춰 계산한다.

    FLOW94_FORWARD=0 ~/of80/.rlvenv/bin/python scripts/rl_94/build.py 2026-08-24 ... 2026-09-15
결과: ~/of80/flow94/<day>.pkl (행+37피처+라벨) · ~/of80/fills94/<day>.pkl (rows, fill_sec, P, tick)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "flow_83"))
sys.path.insert(0, str(ROOT / "scripts" / "flow_84"))
sys.path.insert(0, str(ROOT / "scripts" / "rl_87"))
import features as F83  # noqa: E402
import fills as FL87  # noqa: E402
import library as L84  # noqa: E402

HOME = Path.home() / "of80"
OUT = HOME / "flow94"
OUT_FILLS = HOME / "fills94"
FEATS = json.loads((ROOT / "scripts" / "rl_93" / "features.json").read_text())
FORWARD_START = "2026-09-16"


def build(day: str) -> str:
    if os.environ.get("FLOW94_FORWARD") != "1":
        assert day < FORWARD_START, "학습 캐시가 전진 날짜를 만들려 했다"
    out, outf = OUT / f"{day}.pkl", OUT_FILLS / f"{day}.pkl"
    if out.exists() and outf.exists():
        return f"{day} cached"
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
    mat, news, prevclose = L84._material(conn, day, codes)
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
    parts, fills, paths_out = [], {}, {}
    for ci, c in enumerate(codes):
        f, p = per[c]
        f["cum_rank"], f["log_cumval"], f["sec"] = rank[ci].astype(float), np.log1p(f["cumval"]), np.arange(N, dtype=float)
        # ▼ 94번의 유일한 차이 — 순위·가격·시간 필터를 뺀다(사전등록 §1)
        ok = ((f["n_tr_sec"] > 0) & (p["ask"] > p["bid"]) & np.isfinite(p["ask"]) & np.isfinite(p["bid"]))
        rows = np.nonzero(ok)[0]
        if len(rows) < 50:
            continue
        x = L84._extra(tg[c], qg.get(c, qt.iloc[:0]), f, p)
        cols = {k: f[k][rows] for k in F83.FEATURES}
        cols.update({k: v[rows] for k, v in x.items()})
        for k, v in mat[c].items():
            cols[k] = np.full(len(rows), v)
        ns = news_sec.get(c, np.array([]))
        today_ns = ns[ns > -10**6]
        cols["news_prev_day"] = np.full(len(rows), float((ns == -10**6).sum()))
        cols["news30"] = (np.searchsorted(today_ns, rows) - np.searchsorted(today_ns, rows - 1800)).astype(float)
        cols["news120"] = (np.searchsorted(today_ns, rows) - np.searchsorted(today_ns, rows - 7200)).astype(float)
        cols["news_today"] = np.searchsorted(today_ns, rows).astype(float)
        cols["news_cov"] = np.full(len(rows), float(day >= "2026-09-04"))
        cols["dayret"], cols["dayret_rank"] = dayret[ci][rows], dayret_rank[ci][rows].astype(float)
        df = pd.DataFrame({k: np.asarray(cols[k], np.float32) for k in FEATS})
        df.insert(0, "row", rows.astype(np.int32))
        df.insert(0, "code", c)
        parts.append(df)
        bq = np.full(N, np.nan)
        q = qg.get(c)
        if q is not None:
            qs = (q.ts.dt.hour * 3600 + q.ts.dt.minute * 60 + q.ts.dt.second).to_numpy() - F83.T0
            mq = (qs >= 0) & (qs < N)
            bq[qs[mq]] = q.bidqty1.to_numpy(float)[mq]
        bq = F83.ffill(bq)
        g = tg[c]
        sec = (g.ts.dt.hour * 3600 + g.ts.dt.minute * 60 + g.ts.dt.second).to_numpy() - F83.T0
        okt = (sec >= 0) & (sec < N)
        sec = sec[okt].astype(np.int64)
        start = np.searchsorted(sec, np.arange(N + 1))
        fs, P = FL87._fills(rows.astype(np.int64), g.price.to_numpy(float)[okt], g.volume.to_numpy(float)[okt],
                            g.side.to_numpy(np.int64)[okt], start, p["bid"].astype(np.float64), bq,
                            p["strength"].astype(np.float64), N)
        fills[c] = (rows.astype(np.int64), fs, P, p["tick"][np.minimum(rows + 1, N - 1)].astype(np.float64))
        # 94번은 필터를 풀어 83번 캐시에 없는 종목이 들어온다 — 경로를 여기서 같이 저장한다.
        paths_out[c] = {k: p[k].astype(np.float32) for k in ("bid", "ask", "hi", "tick", "strength")}
    conn.close()
    data = pd.concat(parts, ignore_index=True)
    data.insert(0, "day", day)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FILLS.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps({"rows": data, "paths": paths_out}, protocol=5))
    outf.write_bytes(pickle.dumps(fills, protocol=5))
    return f"{day} rows {len(data):,} codes {data.code.nunique()}"


if __name__ == "__main__":
    with Pool(3) as pool:
        for r in pool.imap_unordered(build, sys.argv[1:]):
            print(r, flush=True)
