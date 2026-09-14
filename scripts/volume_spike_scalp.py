#!/usr/bin/env python3
"""타 트레이더 B 거래량 급등 하루종일 스캘핑 — 사전등록 78번.

``docs/research/manju/78-volume-spike-scalp-preregistration.md`` 가 정본이다. 상수를 바꾸려면
문서부터 바꿔야 하는데, 측정 후에는 바꿀 수 없다.

단계
    scan     날짜별 분봉 → 6셀 트리거·분봉 근사 청산·무작위 대조. 캐시(멱등, 병렬)
    select   IS 6셀 평가·원장 기록·하나 선택
    oos      선택 셀 OOS 게이트(7조건) + 서술
    ticks    선택 셀 16일 틱 검산(서술)
    report   마크다운

안전: DB SELECT 만. 주문 경로 없음. 출력은 data/volume_spike_78/, research/logs/78-*/.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/volume_spike_scalp.py scan --workers 8
    uv run --extra pg python scripts/volume_spike_scalp.py select
    uv run --extra pg python scripts/volume_spike_scalp.py oos
    uv run --extra pg python scripts/volume_spike_scalp.py ticks
    uv run --extra pg python scripts/volume_spike_scalp.py report
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.limits import limit_up_price
from krx_quant_core.stats import trials as trial_ledger
from krx_quant_core.stats.metrics import paired_date_bootstrap
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect
from scalp_it.ticksize import tick_size

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "volume_spike_78"
LOGS_DIR = ROOT / "research" / "logs"
LABEL = "78-volume-spike-scalp"

# ---------------------------------------------------------------- 사전등록 상수
START, IS_END, END = "2025-08-04", "2026-03-31", "2026-08-14"
OOS_START = "2026-04-01"
KS = [3.0, 5.0, 10.0]
RS = [0.005, 0.015]
T_FIRST, T_LAST = 9 * 60 + 5, 15 * 60          # 트리거 분
M_OPEN, M_LAST_EXIT = 9 * 60, 15 * 60 + 19
BASE_WIN, BASE_MIN = 20, 5
MIN_TV_BAR = 1e8
MIN_PREV_TV_MIL = 5000                          # daily_bars 백만원 → 50억
MIN_PREV_CLOSE = 1000
RUNWAY_TICKS = 5
HOLD = 10
NOTIONAL = 1e7
IS_MIN_N = 200
OOS_MIN_N = 30
N_BOOT = 10_000
SEED = 78
N_DSR = 103
TICK_DAYS = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
             "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
             "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
             "2026-09-14"]


def cells() -> list[tuple[float, float]]:
    return [(k, r) for k in KS for r in RS]


def cell_name(c: tuple[float, float]) -> str:
    return f"K={int(c[0])} R={c[1]:.1%}"


def _q(conn: Any, sql: str, params: Any = ()) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def trading_days(conn: Any, lo: str, hi: str) -> list[str]:
    rows = _q(conn, "select distinct date from daily_bars where date >= %s and date <= %s "
                    "order by date", (lo, hi))
    return [r[0].isoformat() for r in rows]


# ================================================================ 청산 시뮬레이션
def tk(p: float) -> float:
    return float(tick_size(float(p)))


def simulate(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, Cff: np.ndarray,
             t: int, lim: float) -> dict[str, Any] | None:
    """분 인덱스 t(0 = 09:00)의 트리거 → §4 분봉 근사. 봉 없는 분은 NaN."""
    j = t + 1
    last = min(t + HOLD, M_LAST_EXIT - M_OPEN)
    if j > last or j >= len(O) or not np.isfinite(O[j]):
        return None
    e = O[j] + tk(O[j])
    if e >= lim - 1e-9:
        return None
    S = e - tk(e - 1e-9)
    hi = H[j:last + 1]
    mfe = (np.nanmax(hi) - e) / tk(e) if np.isfinite(hi).any() else float("nan")

    def run(close_based: bool) -> tuple[float, str, int]:
        c = C[j]
        if c <= S + 1e-9:
            return c - tk(c), "stop", j
        if H[j] < e + tk(e) - 1e-9:
            return c - tk(c), "scratch", j
        for k in range(j + 1, last + 1):
            if not np.isfinite(O[k]):
                if k == last:
                    return Cff[k] - tk(Cff[k]), "time", k
                continue
            if close_based:
                if C[k] <= S + 1e-9:
                    return C[k] - tk(C[k]), "stop", k
            elif L[k] <= S + 1e-9:
                return S - tk(S - 1e-9), "stop", k
            if C[k] < Cff[k - 1] - 1e-9:
                return C[k] - tk(C[k]), "exhaust", k
            if k == last:
                return C[k] - tk(C[k]), "time", k
        return Cff[j] - tk(Cff[j]), "time", j      # j == last

    x, kind, k_exit = run(False)
    xo, kind_o, _ = run(True)
    return {"entry": float(e), "exit": float(x), "kind": kind, "exit_idx": int(k_exit),
            "gross": float(x / e - 1.0), "gross_opt": float(xo / e - 1.0), "kind_opt": kind_o,
            "mfe_ticks": float(mfe)}


# ================================================================ scan
def scan_day(day: str) -> Path:
    path = OUT / "days" / f"{day}.pkl"
    if path.exists():
        return path
    conn = connect()
    try:
        prev = _q(conn, "select max(date) from daily_bars where date < %s", (day,))[0][0]
        drows = _q(conn, "select d.code, d.close, d.volume, d.trade_value, s.market "
                         "from daily_bars d join stocks s on s.code = d.code "
                         "where d.date = %s and s.market in ('코스닥','거래소')", (prev,))
        kosdaq = _q(conn, "select sum(d.trade_value) from daily_bars d join stocks s on "
                          "s.code=d.code where d.date=%s and s.market='코스닥'", (day,))[0][0]
        bars = load_minute_bars(conn, day, day)
    finally:
        conn.close()
    pdf = pd.DataFrame(drows, columns=["code", "pclose", "pvol", "ptv", "market"])
    pdf["code"] = pdf["code"].astype(str).str.strip()
    pdf = pdf[(pdf.ptv >= MIN_PREV_TV_MIL) & (pdf.pclose >= MIN_PREV_CLOSE)].set_index("code")
    out: dict[str, Any] = {"day": day, "kosdaq_jo": float(kosdaq or 0) / 1e6, "trades": [],
                           "n_universe": 0}
    if bars.empty or pdf.empty:
        OUT.joinpath("days").mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))
        return path
    bars = bars[bars.code.isin(pdf.index)].copy()
    bars["m"] = bars.ts.dt.hour * 60 + bars.ts.dt.minute - M_OPEN
    bars = bars[(bars.m >= 0) & (bars.m <= M_LAST_EXIT - M_OPEN)]
    n_min = M_LAST_EXIT - M_OPEN + 1
    for code, g in bars.groupby("code", sort=True):
        out["n_universe"] += 1
        O = np.full(n_min, np.nan); H = O.copy(); L = O.copy(); C = O.copy()
        V = np.zeros(n_min); TV = np.zeros(n_min)
        idx = g.m.to_numpy()
        O[idx], H[idx], L[idx], C[idx] = g.open, g.high, g.low, g.close
        V[idx], TV[idx] = g.volume.fillna(0), g.trade_value.fillna(0)
        Cff = pd.Series(C).ffill().to_numpy()
        info = pdf.loc[code]
        lim = float(limit_up_price(float(info.pclose)))
        floor = float(info.pvol) / 390.0
        csum = np.concatenate([[0.0], np.cumsum(V)])
        mkt = "KOSPI" if info.market == "거래소" else "KOSDAQ"
        base = np.full(n_min, np.nan)
        for t in range(T_FIRST - M_OPEN, T_LAST - M_OPEN + 1):
            lo = max(0, t - BASE_WIN)
            if t - lo >= BASE_MIN:
                base[t] = max((csum[t] - csum[lo]) / (t - lo), floor)
        prevc = np.concatenate([[np.nan], Cff[:-1]])
        ret1 = C / prevc - 1.0
        with np.errstate(invalid="ignore"):
            runway_ok = np.isfinite(C) & (C <= lim - RUNWAY_TICKS * tk(lim) + 1e-9)
        common = (np.isfinite(base) & np.isfinite(C) & (C > O) & (TV >= MIN_TV_BAR) & runway_ok)
        for (K, R) in cells():
            mask = common & (V >= K * np.nan_to_num(base, nan=np.inf)) & (ret1 >= R)
            free_after = -1
            for t in np.flatnonzero(mask):
                if t <= free_after:
                    continue
                res = simulate(O, H, L, C, Cff, int(t), lim)
                if res is None:
                    continue
                free_after = res["exit_idx"]
                sig = (H[t] - L[t]) / O[t] if O[t] > 0 else float("nan")
                imp = sig * math.sqrt(NOTIONAL / TV[t]) if TV[t] > 0 else float("nan")
                # 무작위 대조: 같은 종목·같은 날 09:05~15:00 무작위 분
                rs = np.random.default_rng(zlib.crc32(f"{code}|{day}|{t}|{K}|{R}".encode()))
                nres = None
                for _ in range(20):
                    tn = int(rs.integers(T_FIRST - M_OPEN, T_LAST - M_OPEN + 1))
                    nres = simulate(O, H, L, C, Cff, tn, lim)
                    if nres is not None:
                        break
                out["trades"].append({
                    "day": day, "code": code, "market": mkt, "K": K, "R": R, "t": int(t),
                    "hm": f"{(t + M_OPEN) // 60:02d}{(t + M_OPEN) % 60:02d}",
                    "vol_mult": float(V[t] / base[t]), "ret1": float(ret1[t]),
                    "tv_bar": float(TV[t]), "impact_1w": imp, **res,
                    "null_gross": nres["gross"] if nres else None,
                })
    OUT.joinpath("days").mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return path


def cmd_scan(ns: argparse.Namespace) -> None:
    conn = connect()
    days = trading_days(conn, START, END)
    conn.close()
    todo = [d for d in days if not (OUT / "days" / f"{d}.pkl").exists()]
    print(f"days {len(days)} todo {len(todo)}", flush=True)
    with ProcessPoolExecutor(max_workers=ns.workers) as ex:
        for i, p in enumerate(ex.map(scan_day, todo)):
            if i % 10 == 0:
                print(f"  {i}/{len(todo)} {p.name}", flush=True)


def load_trades() -> pd.DataFrame:
    rows, meta = [], []
    for p in sorted((OUT / "days").glob("*.pkl")):
        d = pickle.loads(p.read_bytes())
        rows.extend(d["trades"])
        meta.append({"day": d["day"], "kosdaq_jo": d["kosdaq_jo"], "n_universe": d["n_universe"]})
    df = pd.DataFrame(rows)
    m = pd.DataFrame(meta)
    cost = {d: {mk: round_trip_cost(date.fromisoformat(d), mk, commission_rate=0.00015)
                for mk in ("KOSPI", "KOSDAQ")} for d in m.day}
    c = np.array([cost[d][mk] for d, mk in zip(df.day, df.market, strict=True)])
    df["cost"] = c
    df["ret"] = df.gross - c
    df["ret_opt"] = df.gross_opt - c
    df["ret_size"] = df.ret - 2 * df.impact_1w
    df["null_ret"] = df.null_gross - c
    df = df.merge(m[["day", "kosdaq_jo"]], on="day", how="left")
    return df


# ================================================================ 통계
def day_cluster_ci(df: pd.DataFrame, col: str = "ret") -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    g = df.groupby("day")[col]
    sums, cnts = g.sum().to_numpy(), g.count().to_numpy()
    n = len(sums)
    pick = rng.integers(0, n, size=(N_BOOT, n))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summ(R: pd.Series) -> dict[str, Any]:
    R = R.dropna()
    return {"n": int(len(R)), "mean": float(R.mean()) if len(R) else None,
            "median": float(R.median()) if len(R) else None,
            "win": float((R > 0).mean()) if len(R) else None}


def cmd_select(ns: argparse.Namespace) -> None:
    df = load_trades()
    isd = df[df.day <= IS_END]
    rows = []
    for c in cells():
        sub = isd[(isd.K == c[0]) & (isd.R == c[1])]
        cfg = {"K": c[0], "R": c[1], "base": "max(mean vol 20m, prev vol/390)",
               "min_tv_bar": MIN_TV_BAR, "runway_ticks": RUNWAY_TICKS, "hold_min": HOLD,
               "exit": "stop e-1tick(stop-first), scratch +1tick entry bar, first lower close",
               "universe": "prev tv>=50억, close>=1000", "window": "0905-1500",
               "is": [START, IS_END], "prereg": "docs/research/manju/78"}
        trial_ledger.record_trial(LABEL, cfg, logs_dir=LOGS_DIR, note="IS cell (prereg 78 §5)")
        s = summ(sub.ret)
        rows.append({"cell": list(c), "name": cell_name(c), **s,
                     "median_opt": float(sub.ret_opt.median()) if len(sub) else None,
                     "mean_opt": float(sub.ret_opt.mean()) if len(sub) else None,
                     "mean_gross": float(sub.gross.mean()) if len(sub) else None,
                     "mean_null": float(sub.null_ret.mean()) if len(sub) else None,
                     "reasons": sub.kind.value_counts().to_dict(),
                     "eligible": s["n"] >= IS_MIN_N})
    el = [r for r in rows if r["eligible"]]
    sel = None
    if el:
        best = max(round(r["mean"], 6) for r in el)
        sel = next(r for r in rows if r["eligible"] and round(r["mean"], 6) == best)
    out = {"rows": rows, "selected": sel["cell"] if sel else None,
           "n_trials_ledger": trial_ledger.count_trials(LABEL, logs_dir=LOGS_DIR)}
    OUT.joinpath("select_is.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"{r['name']:<14} n={r['n']:>7} mean={r['mean']:+.4%} med={r['median']:+.4%} "
              f"win={r['win']:.1%} gross={r['mean_gross']:+.4%} null={r['mean_null']:+.4%}")
    print("selected", sel["name"] if sel else None)


def cmd_oos(ns: argparse.Namespace) -> None:
    sel = json.loads(OUT.joinpath("select_is.json").read_text())
    K, R = sel["selected"]
    df = load_trades()
    cell = df[(df.K == K) & (df.R == R)]
    oos = cell[cell.day >= OOS_START].copy()
    res: dict[str, Any] = {"cell": [K, R], **summ(oos.ret)}
    Rv = oos.ret.to_numpy()
    res["ci"] = day_cluster_ci(oos)
    nd = oos.dropna(subset=["null_ret"])
    a = nd.groupby("day").ret.mean().rename("mean_excess").rename_axis("date").reset_index()
    b = nd.groupby("day").null_ret.mean().rename("mean_excess").rename_axis("date").reset_index()
    pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
    res["null"] = {**pb, "p": 1 - pb["p_greater"], "strat": float(nd.ret.mean()),
                   "null": float(nd.null_ret.mean()), "n_pairs": len(nd)}
    res["dsr"] = deflated_sharpe_from_sample(Rv, N_DSR)
    oos["month"] = oos.day.str[:7]
    res["months"] = oos.groupby("month").ret.mean().to_dict()
    k = max(1, math.ceil(0.01 * len(Rv)))
    res["trim_mean"] = float(np.sort(Rv)[:-k].mean())
    pd_ = res["dsr"].get("prob_deflated_sharpe") or 0.0
    conds = {
        "1 OOS n≥30": res["n"] >= OOS_MIN_N,
        "2 평균>0·중앙>0": res["mean"] > 0 and res["median"] > 0,
        "3 일클러스터 CI하한>0": res["ci"][0] > 0,
        "4 무작위대조 p<0.05": res["null"]["p"] < 0.05,
        "5 DSR확률>0.95(N=103)": pd_ > 0.95,
        "6 월 음수 ≤1": sum(1 for v in res["months"].values() if v < 0) <= 1,
        "7 상위1%제거 평균>0": res["trim_mean"] > 0,
    }
    res["conds"] = conds
    res["verdict"] = ("표본 부족 — 판정 불가" if not conds["1 OOS n≥30"]
                      else "채택" if all(conds.values()) else "기각")
    # 서술
    desc: dict[str, Any] = {}
    desc["gross"] = float(oos.gross.mean())
    desc["opt"] = summ(oos.ret_opt)
    desc["size"] = summ(oos.ret_size)
    desc["reasons"] = {k2: {"n": int(len(v)), "mean": float(v.ret.mean())}
                       for k2, v in oos.groupby("kind")}
    desc["am_pm"] = {k2: summ(v.ret) for k2, v in oos.groupby(oos.hm < "1100")}
    desc["market"] = {k2: summ(v.ret) for k2, v in oos.groupby("market")}
    reg = pd.cut(oos.kosdaq_jo, [0, 5, 8, 10, 1e9], labels=["<5조", "5~8조", "8~10조", "10조↑"])
    desc["regime"] = {str(k2): summ(v.ret) for k2, v in oos.groupby(reg, observed=True)}
    desc["mfe_ticks_q"] = {q: float(oos.mfe_ticks.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)}
    desc["tick_bp_median"] = float((1e4 * oos.entry.map(tk) / oos.entry).median())
    desc["cost_median"] = float(oos.cost.median())
    desc["null_all"] = summ(oos.null_ret)
    desc["all_cells_oos"] = {cell_name(c): {**summ(df[(df.K == c[0]) & (df.R == c[1]) &
                                                      (df.day >= OOS_START)].ret)}
                             for c in cells()}
    res["desc"] = desc
    OUT.joinpath("oos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k2: v for k2, v in res.items() if k2 != "desc"}, ensure_ascii=False,
                     indent=1, default=str))


# ================================================================ 틱 검산(서술)
def tick_day(day: str, K: float, R: float) -> list[dict[str, Any]]:
    conn = connect()
    try:
        prev = _q(conn, "select max(date) from daily_bars where date < %s", (day,))[0][0]
        drows = _q(conn, "select d.code, d.close, d.volume, d.trade_value, s.market from "
                         "daily_bars d join stocks s on s.code=d.code where d.date=%s and "
                         "s.market in ('코스닥','거래소')", (prev,))
        pdf = pd.DataFrame(drows, columns=["code", "pclose", "pvol", "ptv", "market"])
        pdf = pdf[(pdf.ptv >= MIN_PREV_TV_MIL) & (pdf.pclose >= MIN_PREV_CLOSE)].set_index("code")
        codes = [r[0] for r in _q(conn, "select distinct code from ticks where ts >= %s and "
                                        "ts < %s::date + 1", (day, day))]
        out = []
        for code in codes:
            if code not in pdf.index:
                continue
            rows = _q(conn, "select ts, seq, price, volume from ticks where code=%s and ts >= %s "
                            "and ts < %s::date + 1 order by ts, seq", (code, day, day))
            if len(rows) < 100:
                continue
            tdf = pd.DataFrame(rows, columns=["ts", "seq", "price", "volume"])
            tdf["m"] = tdf.ts.dt.hour * 60 + tdf.ts.dt.minute - M_OPEN
            tdf = tdf[(tdf.m >= 0) & (tdf.m <= M_LAST_EXIT - M_OPEN)]
            if tdf.empty:
                continue
            n_min = M_LAST_EXIT - M_OPEN + 1
            g = tdf.groupby("m")
            O = np.full(n_min, np.nan); H = O.copy(); L = O.copy(); C = O.copy()
            V = np.zeros(n_min); TV = np.zeros(n_min); has = np.zeros(n_min, bool)
            ix = g.price.first().index.to_numpy()
            O[ix] = g.price.first(); H[ix] = g.price.max(); L[ix] = g.price.min()
            C[ix] = g.price.last(); V[ix] = g.volume.sum()
            TV[ix] = (tdf.price * tdf.volume).groupby(tdf.m).sum()
            has[ix] = True
            first_m, last_m = int(ix.min()), int(ix.max())
            info = pdf.loc[code]
            lim = float(limit_up_price(float(info.pclose)))
            floor = float(info.pvol) / 390.0
            Cff = pd.Series(C).ffill().to_numpy()
            prevc = np.concatenate([[np.nan], Cff[:-1]])
            ts_arr = tdf.ts.to_numpy()
            px_arr = tdf.price.to_numpy().astype(float)
            free_after = -1
            for t in range(T_FIRST - M_OPEN, min(T_LAST - M_OPEN, last_m - 1) + 1):
                if t - first_m < BASE_WIN or t <= free_after or not has[t]:
                    continue                 # 트리거 앞 20분 연속 수집(첫 체결 분 기준 근사)
                base = max(V[t - BASE_WIN:t].mean(), floor)
                if not (V[t] >= K * base and C[t] > O[t] and TV[t] >= MIN_TV_BAR
                        and C[t] / prevc[t] - 1 >= R and C[t] <= lim - RUNWAY_TICKS * tk(lim)):
                    continue
                t0 = pd.Timestamp(f"{day} 09:00") + pd.Timedelta(minutes=t + 1)
                k0 = int(np.searchsorted(ts_arr, np.datetime64(t0)))
                if k0 >= len(px_arr):
                    continue
                e = px_arr[k0] + tk(px_arr[k0])
                if e >= lim:
                    continue
                S = e - tk(e - 1e-9)
                t_end = t0 + pd.Timedelta(minutes=HOLD)
                t_scr = ts_arr[k0] + np.timedelta64(30, "s")
                surged = False
                x, kind, k_exit = None, None, None
                for k in range(k0 + 1, len(px_arr)):
                    p, ts = px_arr[k], ts_arr[k]
                    if ts >= np.datetime64(t_end):
                        x, kind, k_exit = px_arr[k - 1] - tk(px_arr[k - 1]), "time", k
                        break
                    if not surged and ts > t_scr:
                        pp = px_arr[k - 1]
                        x, kind, k_exit = pp - tk(pp), "scratch", k
                        break
                    if p <= S:
                        x, kind, k_exit = p - tk(p), "stop", k
                        break
                    if p >= e + tk(e):
                        surged = True
                    if surged and p < px_arr[k - 1]:
                        x, kind, k_exit = p - tk(p), "exhaust", k
                        break
                if x is None:
                    continue
                gross = x / e - 1
                cost = round_trip_cost(date.fromisoformat(day),
                                       "KOSPI" if info.market == "거래소" else "KOSDAQ")
                exit_m = int((pd.Timestamp(ts_arr[k_exit]) - pd.Timestamp(f"{day} 09:00"))
                             .total_seconds() // 60)
                free_after = exit_m
                bar = simulate(O, H, L, C, Cff, t, lim)
                out.append({"day": day, "code": code, "hm": f"{(t + 540)//60:02d}{(t + 540) % 60:02d}",
                            "tick_ret": gross - cost, "tick_kind": kind,
                            "bar_ret": (bar["gross"] - cost) if bar else None,
                            "bar_kind": bar["kind"] if bar else None,
                            "hold_s": float((pd.Timestamp(ts_arr[k_exit]) -
                                             pd.Timestamp(ts_arr[k0])).total_seconds())})
        return out
    finally:
        conn.close()


def cmd_ticks(ns: argparse.Namespace) -> None:
    sel = json.loads(OUT.joinpath("select_is.json").read_text())
    K, R = sel["selected"]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=ns.workers) as ex:
        for part in ex.map(tick_day, TICK_DAYS, [K] * len(TICK_DAYS), [R] * len(TICK_DAYS)):
            rows.extend(part)
    df = pd.DataFrame(rows)
    res: dict[str, Any] = {"cell": [K, R], "tick": summ(df.tick_ret) if len(df) else {"n": 0}}
    if len(df):
        res["bar_same_triggers"] = summ(df.bar_ret)
        res["tick_reasons"] = {k: {"n": int(len(v)), "mean": float(v.tick_ret.mean())}
                               for k, v in df.groupby("tick_kind")}
        res["bar_reasons"] = df.bar_kind.value_counts().to_dict()
        res["hold_s_median"] = float(df.hold_s.median())
        res["days"] = int(df.day.nunique())
        res["ci"] = day_cluster_ci(df.rename(columns={"tick_ret": "ret"}))
    OUT.joinpath("ticks.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                     default=str))
    df.to_pickle(OUT / "ticks_trades.pkl")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "select", "oos", "ticks"):
        p = sp.add_parser(name)
        p.add_argument("--workers", type=int, default=8)
    ns = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"scan": cmd_scan, "select": cmd_select, "oos": cmd_oos, "ticks": cmd_ticks}[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
