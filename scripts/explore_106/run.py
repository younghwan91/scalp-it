"""106번 — VI 재개 직후 10·60·300초에 우위가 있는가. simnode 에서만 돈다.

    ssh simnode-local 'cd ~/git/scalp-it && export $(grep -v "^#" ../quant-airflow/.env | xargs) \
        && deploy/research_run.sh .venv/bin/python -W ignore scripts/explore_106/run.py'

1. 날짜마다 틱·호가를 읽어 `vi_reconstruct.reconstruct` 로 VI 후보를 복원한다(bad_days 제외).
2. 복원 VI 가 있는 종목·날의 초 단위 호가 경로를 flow_83 `build_code` 로 만들고(복사 금지),
   재개 첫 초 t0 에서 테이커·메이커(사전확정) · 천장(oracle, maker_pessimistic) 을 잰다.
3. 같은 종목·같은 날의 무작위 초(VI 창 제외)를 풀로, t0 직전 300초 실현변동성 4분위를 맞춘
   매치드 널(`krx_quant_core.stats.matched_null`) 대비 alpha 를 낸다.
4. 기각 기준(사전 고정): 세 지평 모두 테이커·메이커 사전확정 수익의 날짜 클러스터 95% CI 하한 ≤ 0.

산출: data/explore106/vi_candidates.csv · data/explore106/results.json · 표준출력 표.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "explore106"
HORIZONS = (10, 60, 300)
RV_WIN = 300
N_PER = 20
SEED = 106


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V = _load("explore106_vi", HERE / "vi_reconstruct.py")
L = _load("explore106_labels", HERE / "labels.py")
F = _load("flow83_features", ROOT / "scripts" / "flow_83" / "features.py")          # build_code · N · T0
VM = _load("ceiling95_vol_matched", ROOT / "scripts" / "ceiling_95" / "vol_matched.py")  # pre_entry_rv

from krx_quant_core.stats.matched_null import (
    cluster_bootstrap_ci,
    matched_alpha,
    matched_control,
)
from krx_quant_core.stats.trials import record_trial

VI_RULE = {"gap_sec": [V.GAP_MIN, V.GAP_MAX], "jump_min": V.JUMP_MIN,
           "pre_move": [V.PRE_WIN, V.PRE_MOVE_MIN], "quote_min_rate_after_grace": [V.QUOTE_GRACE, V.QUOTE_MIN_RATE],
           "market_min_rate": V.MARKET_MIN_RATE}


def trading_days(conn) -> list[str]:
    bad = set(F.bad_days())
    days = pd.read_sql("select distinct ts::date as d from ticks order by 1", conn).d.astype(str).tolist()
    return [d for d in days if d not in bad]


def load_day(conn, day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    tk = pd.read_sql("select code, ts, seq, price, volume, side, strength, best_bid, best_ask from ticks "
                     "where ts >= %s and ts < %s order by code, ts, seq", conn, params=(day, nxt))
    qt = pd.read_sql("select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
                     "from quotes where ts >= %s and ts < %s order by code, ts", conn, params=(day, nxt))
    for df in (tk, qt):
        df["sec"] = (df.ts.dt.hour * 3600 + df.ts.dt.minute * 60 + df.ts.dt.second).astype(np.int64)
    return tk, qt


def measure_day(day: str, tk: pd.DataFrame, qt: pd.DataFrame, ev: pd.DataFrame
                ) -> tuple[list[dict], list[pd.DataFrame]]:
    """복원 VI 가 있는 종목마다 라벨을 잰다 → (VI 행, 널 풀 조각)."""
    trades: list[dict] = []
    pools: list[pd.DataFrame] = []
    qg = dict(tuple(qt.groupby("code")))
    for code, g in ev.groupby("code"):
        _f, p = F.build_code(tk[tk.code == code], qg.get(code, qt.iloc[:0]))
        bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
        mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)
        rv = VM.pre_entry_rv(mid, RV_WIN)
        e_fill = L.fill_second(bid)
        cols: dict[str, np.ndarray] = {"rv": rv}
        for h in HORIZONS:
            tk_o, mk_o = L.oracle(bid, ask, h)
            cols[f"taker_{h}"] = L.taker_fixed(bid, ask, h)
            cols[f"maker_{h}"] = L.maker_fixed(bid, e_fill, h)
            cols[f"oracle_taker_{h}"] = tk_o
            cols[f"oracle_maker_{h}"] = mk_o
            cols[f"maker_pess_{h}"] = L.maker_pessimistic(bid, e_fill, L.window_max(ask, h))
        # 풀: 세션 안 모든 초에서 **모든 후보 공백**(기각된 것 포함)의 [직전 60초, 재개+600초] 를 뺀다
        pool_ok = np.zeros(F.N, bool)
        pool_ok[F.ENTRY_LO: F.ENTRY_HI] = True
        for r in g.itertuples():
            lo = max(0, r.sec_prev - F.T0 - 60)
            hi = min(F.N, r.sec_resume - F.T0 + 600)
            pool_ok[lo:hi] = False
        pool_ok &= np.isfinite(rv) & np.isfinite(cols[f"taker_{HORIZONS[0]}"])
        pidx = np.nonzero(pool_ok)[0]
        pdf = pd.DataFrame({k: v[pidx] for k, v in cols.items()})
        pdf.insert(0, "sec", pidx)
        pdf.insert(0, "code", code)
        pdf.insert(0, "date", day)
        pools.append(pdf)
        for r in g[g.keep].itertuples():
            t0 = r.sec_resume - F.T0
            if not (0 <= t0 < F.N):
                continue
            row = {"date": day, "code": code, "t0": int(t0), "sec_resume": int(r.sec_resume),
                   "gap": int(r.gap), "jump": float(r.jump), "pre_move": float(r.pre_move),
                   "direction": int(r.direction), "price_resume": float(r.price_resume),
                   "spread_bp": float((ask[t0 + 1] / bid[t0 + 1] - 1) * 1e4)
                   if t0 + 1 < F.N and bid[t0 + 1] > 0 else np.nan}
            row.update({k: float(v[t0]) for k, v in cols.items()})
            trades.append(row)
    return trades, pools


def _stat(x: np.ndarray, dates: np.ndarray) -> dict:
    ok = np.isfinite(x)
    x, d = x[ok] * 1e4, dates[ok]
    if len(x) == 0:
        return {"n": 0}
    lo, hi, _ = cluster_bootstrap_ci(x, d, n_boot=4000, seed=SEED)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "frac_pos": float((x > 0).mean()), "ci_low": lo, "ci_high": hi,
            "n_days": int(pd.unique(d).size)}


def summarize(trades: pd.DataFrame, pool: pd.DataFrame) -> dict:
    out: dict = {"n_vi": len(trades), "n_days": int(trades.date.nunique()),
                 "n_codes": int(trades.code.nunique()), "horizons": {}}
    # 실현변동성 4분위 — 경계는 **풀(무작위 초)** 분위수(95h 와 같다)
    edges = np.quantile(pool.rv.dropna(), [0.25, 0.5, 0.75])
    pool = pool.assign(rv_q=np.searchsorted(edges, pool.rv.to_numpy()))
    trades = trades.assign(rv_q=np.searchsorted(edges, trades.rv.to_numpy()))
    trades.loc[~np.isfinite(trades.rv), "rv_q"] = np.nan
    out["rv_edges_bp"] = [float(e) for e in edges]
    out["rv_q_counts"] = trades.rv_q.value_counts(dropna=False).sort_index().to_dict()
    dates = trades.date.to_numpy()
    for h in HORIZONS:
        rec: dict = {}
        for m in ("taker", "maker", "oracle_taker", "oracle_maker", "maker_pess"):
            col = f"{m}_{h}"
            rec[m] = _stat(trades[col].to_numpy(), dates)
            rec[m]["by_direction"] = {
                str(dv): _stat(trades.loc[trades.direction == dv, col].to_numpy(),
                               dates[(trades.direction == dv).to_numpy()]) for dv in (1, -1)}
        for m in ("taker", "maker"):
            col = f"{m}_{h}"
            tr = trades[np.isfinite(trades[col]) & np.isfinite(trades.rv_q)].copy()
            tr[col] = tr[col] * 1e4
            po = pool[np.isfinite(pool[col])].copy()
            po[col] = po[col] * 1e4
            tr.index = pd.RangeIndex(len(tr))
            po.index = pd.RangeIndex(len(po))
            ctrl = matched_control(tr, po, strata=["date", "code", "rv_q"], n_per=N_PER,
                                   seed=SEED, exclude_self=False)
            a = matched_alpha(tr, po, ctrl, value=col, cluster="date", n_boot=4000, seed=SEED)
            rec[f"{m}_null"] = {"alpha": a.alpha, "ci_low": a.ci_low, "ci_high": a.ci_high,
                                "p_greater": a.p_greater, "trade_mean": a.trade_mean,
                                "control_mean": a.control_mean, "n_trades": a.n_trades,
                                "n_controls": a.n_controls, "n_unmatched": a.n_unmatched}
        out["horizons"][str(h)] = rec
    closed = all(out["horizons"][str(h)][m].get("ci_low", 0) <= 0
                 for h in HORIZONS for m in ("taker", "maker"))
    out["rejection"] = {"rule": "세 지평 모두 테이커·메이커 사전확정 수익의 날짜 클러스터 95% CI 하한 ≤ 0",
                        "closed": bool(closed)}
    return out


def print_report(cands: pd.DataFrame, res: dict) -> None:
    kept = cands[cands.keep]
    print(f"\n후보 공백 {len(cands):,} · 복원 VI {len(kept):,} · 날짜 {kept.date.nunique()} · 종목 {kept.code.nunique()}")
    print("기각 사유:")
    print(cands[~cands.keep].reason.value_counts().to_string())
    print("\n날짜별 VI 수:")
    print(kept.groupby("date").size().to_string())
    hr = (kept.sec_resume // 3600).value_counts().sort_index()
    print("\n재개 시각(시)별:", hr.to_dict())
    print("공백 길이(초) 분위:", kept.gap.quantile([0.1, 0.5, 0.9]).round(0).to_dict())
    print("방향: 상승", int((kept.direction == 1).sum()), "하락", int((kept.direction == -1).sum()))
    print("점프(%) 분위:", (kept.jump * 100).quantile([0.1, 0.5, 0.9]).round(2).to_dict())
    print("\n| H | 테이커 평균 [CI] | 메이커 평균 [CI] (n) | 천장 테이커 | 천장 메이커(뚫림) | 널 alpha 테이커 [CI] | 널 alpha 메이커 [CI] |")
    print("|--:|--|--|--:|--:|--|--|")
    for h in HORIZONS:
        r = res["horizons"][str(h)]
        t, m, nt, nm = r["taker"], r["maker"], r["taker_null"], r["maker_null"]
        ot, om = r["oracle_taker"], r["maker_pess"]
        print(f"| {h} | {t['mean']:+.1f} [{t['ci_low']:+.1f}, {t['ci_high']:+.1f}] (n={t['n']}) "
              f"| {m.get('mean', float('nan')):+.1f} [{m.get('ci_low', float('nan')):+.1f}, {m.get('ci_high', float('nan')):+.1f}] (n={m['n']}) "
              f"| {ot['mean']:+.1f} | {om.get('mean', float('nan')):+.1f} (n={om['n']}) "
              f"| {nt['alpha']:+.1f} [{nt['ci_low']:+.1f}, {nt['ci_high']:+.1f}] "
              f"| {nm['alpha']:+.1f} [{nm['ci_low']:+.1f}, {nm['ci_high']:+.1f}] |")
    print("\n방향별 테이커 평균(bp):")
    for h in HORIZONS:
        bd = res["horizons"][str(h)]["taker"]["by_direction"]
        print(f"  H={h}: 상승 {bd['1'].get('mean', float('nan')):+.1f} (n={bd['1']['n']}) "
              f"· 하락 {bd['-1'].get('mean', float('nan')):+.1f} (n={bd['-1']['n']})")
    print("\n기각 기준 충족:", res["rejection"]["closed"])


def main() -> int:
    import psycopg2

    OUT.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    days = trading_days(conn)
    if sys.argv[1:]:                                   # 점검용: 날짜 몇 개만
        days = [d for d in days if d in sys.argv[1:]]
    print(f"거래일 {len(days)} (bad_days 제외): {days[0]} ~ {days[-1]}", flush=True)
    cands_all, trades_all, pools_all = [], [], []
    for day in days:
        tk, qt = load_day(conn, day)
        ev = V.reconstruct(tk[["code", "sec", "seq", "price"]], qt[["code", "sec"]])
        ev.insert(0, "date", day)
        cands_all.append(ev)
        kept = ev[ev.keep]
        print(f"{day}: 틱 {len(tk):,} · 호가 {len(qt):,} · 후보 {len(ev)} · VI {len(kept)}", flush=True)
        if len(kept):
            tr, po = measure_day(day, tk, qt, ev[ev.code.isin(kept.code)])
            trades_all.extend(tr)
            pools_all.extend(po)
    conn.close()
    cands = pd.concat(cands_all, ignore_index=True)
    cands.to_csv(OUT / "vi_candidates.csv", index=False)
    trades = pd.DataFrame(trades_all)
    pool = pd.concat(pools_all, ignore_index=True)
    trades.to_csv(OUT / "vi_trades.csv", index=False)
    res = summarize(trades, pool)
    res["days"] = days
    res["cost_bp"] = L.COST * 1e4
    res["vi_rule"] = VI_RULE
    res["n_candidates"] = len(cands)
    res["reject_reasons"] = cands[~cands.keep].reason.value_counts().to_dict()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    for h in HORIZONS:
        for m in ("taker_fixed", "maker_fixed_through"):
            record_trial("106-vi-resume",
                         {"horizon": h, "method": m, "cost_bp": L.COST * 1e4, "vi_rule": VI_RULE,
                          "null": f"same code-day random seconds, rv{RV_WIN} quartile matched, n_per={N_PER}"},
                         logs_dir=ROOT / "research" / "logs", note="106 VI 재개 탐색 — 사전확정 청산")
    print_report(cands, res)
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
