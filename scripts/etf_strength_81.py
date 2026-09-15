#!/usr/bin/env python3
"""81번 — 매도세 면제 ETF 체결강도 가속 진입·꺾임 청산. 사전등록 docs/research/manju/81.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/etf_strength_81.py                 # 매일 크론
    uv run --extra pg python scripts/etf_strength_81.py --codes 000660 --start 2026-09-08  # 배선 점검(주식)

정의·격자·게이트는 문서 §1~§6 이 정본이다. 여기 숫자를 바꾸려면 문서를 먼저 봐라 — 바꾸면 안 된다.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import os
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ETFS = ("069500", "122630", "102110", "360750", "133690", "379800", "114800",
        "233740", "229200", "396500", "091160", "251340", "381180", "305720")
COST = 0.00015 * 2
QS = (0.95, 0.99)
SDROPS = (3.0, 10.0)
TS = (60, 300)
CELLS = list(itertools.product(QS, SDROPS, TS))
IS_DAYS = 5
COOL = 60
N_TRIALS = 1258
SEED = 81
N_BOOT = 10_000


def _load_build():
    spec = importlib.util.spec_from_file_location(
        "of_build", ROOT / "scripts" / "orderflow_80" / "of_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def etf_tick(p: float) -> float:
    return 1.0 if p < 2000 else 5.0


def stock_tick(p: float) -> float:
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return float(t)
    return 1000.0


TICK = etf_tick


def build_day(conn, day: str, codes: tuple[str, ...], cache: Path) -> dict | None:
    path = cache / f"{day}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    ob = _load_build()
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    tk = pd.read_sql(
        "select code, ts, seq, price, volume, side, strength, best_bid, best_ask from ticks "
        "where code = any(%s) and ts >= %s and ts < %s order by code, ts, seq",
        conn, params=(list(codes), day, nxt))
    if tk.empty:
        return None
    qt = pd.read_sql(
        "select code, ts, bid1, ask1, bidqty1, bidqty2, bidqty3, askqty1, askqty2, askqty3 "
        "from quotes where code = any(%s) and ts >= %s and ts < %s order by code, ts",
        conn, params=(list(codes), day, nxt))
    qg = dict(tuple(qt.groupby("code")))
    out = {}
    for code, g in tk.groupby("code"):
        if len(g) < 200:
            continue
        feat, p = ob.build_code(g, qg.get(code, qt.iloc[:0]))
        feat = feat[feat.strength.notna()][["sec", "dstr30", "bid", "ask"]]
        out[code] = (feat, p)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out, protocol=5))
    return out


def exit_at(p: dict, e: int, sdrop: float, T: int) -> tuple[float, int]:
    bid, stg = p["bid"], p["strength"]
    end = min(e + T, len(bid) - 1)
    peak = stg[e]
    for s in range(e + 1, end + 1):
        if stg[s] == stg[s]:
            peak = max(peak, stg[s]) if peak == peak else stg[s]
            if stg[s] <= peak - sdrop:
                return float(bid[s]), s
    return float(bid[end]), end


def trade(p: dict, t: int, sdrop: float, T: int) -> float | None:
    e = t + 1
    if e >= len(p["ask"]) - 2:
        return None
    a = float(p["ask"][e])
    if not (a == a and a > 0):
        return None
    px, _ = exit_at(p, e, sdrop, T)
    if not (px == px and px > 0):
        return None
    return px / a - 1 - COST


def run_cell(days: list[str], data: dict[str, dict], cell: tuple, *, control: bool = False) -> pd.DataFrame:
    q, sdrop, T = cell
    rng = np.random.default_rng(SEED)
    rows = []
    for i, day in enumerate(days):
        for code, (feat, p) in data[day].items():
            prev = [data[d][code][0] for d in days[max(0, i - 5):i] if code in data[d]]
            if not prev:
                continue                                   # 직전 거래일 없음 → 신호 없음(§3)
            thr = float(pd.concat(prev).dstr30.quantile(q))
            tickv = np.array([TICK(x) for x in feat.ask.to_numpy()])
            ok = ((feat.ask - feat.bid) / tickv <= 1.01) & (feat.dstr30 >= thr)
            free = -1
            live_secs = feat.sec.to_numpy()
            for t in feat.loc[ok, "sec"].to_numpy():
                if t < free:
                    continue
                free = t + COOL
                r = trade(p, int(t), sdrop, T)
                if r is None:
                    continue
                rec = {"day": day, "code": code, "sec": int(t), "ret": r}
                if control:
                    near = live_secs[(np.abs(live_secs - t) <= 300) & (live_secs != t)]
                    rc = trade(p, int(rng.choice(near)), sdrop, T) if len(near) else None
                    rec["ret_ctrl"] = rc
                rows.append(rec)
    return pd.DataFrame(rows, columns=["day", "code", "sec", "ret", *(["ret_ctrl"] if control else [])])


def daymean(df: pd.DataFrame) -> float:
    return float(df.groupby("day").ret.mean().mean()) if len(df) else float("nan")


def gates(df: pd.DataFrame, oos_days: list[str]) -> dict:
    from krx_quant_core.stats.metrics import paired_date_bootstrap
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    R = df.ret.to_numpy()
    n = len(R)
    res: dict = {"n": n, "oos_days": len(oos_days)}
    if n == 0:
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    res["mean"] = float(R.mean())
    res["median"] = float(np.median(R))
    rng = np.random.default_rng(SEED)
    by = {d: df.loc[df.day == d, "ret"].to_numpy() for d in oos_days}
    means = []
    for _ in range(N_BOOT):
        arr = np.concatenate([by[oos_days[k]] for k in rng.integers(0, len(oos_days), len(oos_days))])
        if arr.size:
            means.append(arr.mean())
    res["ci_low"] = float(np.percentile(means, 2.5))
    k = max(1, math.ceil(0.01 * n))
    res["trim1_mean"] = float(np.sort(R)[:-k].mean()) if n > k else float("nan")
    wk = df.assign(week=pd.to_datetime(df.day).dt.isocalendar().week).groupby("week").ret.mean()
    res["neg_weeks"] = f"{int((wk < 0).sum())}/{len(wk)}"
    dsr = deflated_sharpe_from_sample(R, N_TRIALS)
    res["dsr_prob"] = float(dsr["prob_deflated_sharpe"])
    c = df.dropna(subset=["ret_ctrl"])
    if len(c):
        a = c.groupby("day").ret.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
        b = c.groupby("day").ret_ctrl.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
        pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
        res["ctrl_p"] = 1 - pb["p_greater"]
        res["ctrl_mean"] = float(c.ret_ctrl.mean())
    checks = {
        "1 평균>0": res["mean"] > 0,
        "2 CI하한>0": res["ci_low"] > 0,
        "3 상위1%제거>0": res["trim1_mean"] > 0,
        "4 음수주≤1/3": int((wk < 0).sum()) <= len(wk) // 3,
        "5 DSR>0.95": res["dsr_prob"] > 0.95,
        "6 대조 p<0.05": res.get("ctrl_p", 1.0) < 0.05,
        "7 n≥30·일≥10": n >= 30 and len(oos_days) >= 10,
    }
    res["checks"] = checks
    if not checks["7 n≥30·일≥10"]:
        res["verdict"] = "표본 부족 — 판정 불가"
    else:
        res["verdict"] = "채택" if all(checks.values()) else "기각"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="81번 ETF 체결강도 스캘핑")
    ap.add_argument("--codes", default=",".join(ETFS))
    ap.add_argument("--start", default="2026-09-16")
    ap.add_argument("--end", default=date.today().isoformat())  # noqa: DTZ011
    ap.add_argument("--cache", default="data/etf81")
    ap.add_argument("--out", default="data/etf81_report.md")
    ap.add_argument("--record-trials", action="store_true",
                    help="IS 가 확정된 첫 판정 때 원장에 8셀 기록(한 번만)")
    ap.add_argument("--stock-ticks", action="store_true",
                    help="배선 점검 전용 — 주식 호가단위로 스프레드 틱수를 잰다(판정에 쓰지 말 것)")
    a = ap.parse_args(argv)
    global TICK
    if a.stock_ticks:
        TICK = stock_tick

    import psycopg2

    codes = tuple(c.strip() for c in a.codes.split(",") if c.strip())
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    cur = conn.cursor()
    cur.execute("select distinct ts::date from ticks where code = any(%s) and ts >= %s and ts < %s::date + 1 "
                "order by 1", (list(codes), a.start, a.end))
    days_all = [r[0].isoformat() for r in cur.fetchall()]
    data = {}
    for d in days_all:
        x = build_day(conn, d, codes, Path(a.cache))
        if x:
            data[d] = x
    conn.close()
    days = sorted(data)
    lines = [f"# 81번 ETF 체결강도 — 리포트 ({a.end})", "",
             f"- 종목 {', '.join(codes)} · 데이터 거래일 {len(days)}일 ({days[0] if days else '-'} ~ {days[-1] if days else '-'})"]
    if len(days) <= IS_DAYS:
        lines += [f"- IS {IS_DAYS}거래일 미달 — 선택 전. **표본 부족 — 판정 불가**"]
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0
    is_days, oos_days = days[:IS_DAYS], days[IS_DAYS:]
    rows = []
    for cell in CELLS:
        d = run_cell(days, data, cell)
        di = d[d.day.isin(is_days)]
        rows.append((cell, len(di), daymean(di)))
    lines += ["", "## IS 8셀", "", "| q | s | T | IS n | IS 일평균(bp) |", "|--:|--:|--:|--:|--:|"]
    lines += [f"| {c[0]} | {c[1]:g} | {c[2]} | {n} | {m * 1e4:+.1f} |" for c, n, m in rows]
    elig = [r for r in rows if r[1] >= 20 and r[2] == r[2]]
    if not elig:
        lines += ["", "**기각(IS 에서 신호 없음)** — 자격 셀(IS n≥20) 없음."]
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0
    best = sorted(elig, key=lambda r: (-round(r[2], 6), -r[0][0]))[0][0]
    if a.record_trials:
        from krx_quant_core.stats.trials import record_trial
        for c, n, m in rows:
            record_trial("81-etf-strength", {"q": c[0], "s": c[1], "T": c[2], "prereg": "docs/research/manju/81",
                                             "objective": "IS daymean net 3bp"},
                         logs_dir=ROOT / "research" / "logs", note="IS trial (prereg 81 §5)")
    oos = run_cell(days, data, best, control=True)
    oos = oos[oos.day.isin(oos_days)]
    g = gates(oos, oos_days)
    lines += ["", f"## 선택 셀 q={best[0]} s={best[1]:g} T={best[2]} — OOS {len(oos_days)}거래일", ""]
    for k in ("n", "mean", "median", "ci_low", "trim1_mean", "neg_weeks", "dsr_prob", "ctrl_mean", "ctrl_p"):
        if k in g:
            v = g[k]
            lines.append(f"- {k}: {v * 1e4:+.1f}bp" if k in ("mean", "median", "ci_low", "trim1_mean", "ctrl_mean")
                         else f"- {k}: {v}")
    if "checks" in g:
        lines += ["", "게이트: " + " · ".join(f"[{k}: {'✓' if v else '✗'}]" for k, v in g["checks"].items())]
    lines += ["", f"**판정: {g['verdict']}**"]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
