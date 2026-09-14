#!/usr/bin/env python3
"""만쥬 진입 시점 — 상한가 3호가 근접 vs 잠김 확인 후 (사전등록 79번).

``docs/research/manju/79-manju-entry-timing-preregistration.md`` 가 판정 기준의 정본이다.

단계
====

    replay   팔×청산(T1·T2 × live·B)마다 75번 ``replay_day`` 를 설정 래퍼로 호출해 캐시.
             T4 는 T2 캐시에 사후 필터(후속 60초 수익률 < +2%)라 따로 안 돌린다.
    extras   T2 후보의 후속 60초 수익률, T1 후보의 "대장 180초 안 3초 잠김 도달" 계산(DB SELECT).
    select   IS 6 시행 평가·원장 기록·1개 선택(§6).
    oos      선택 시행 + 기준선 T1×E-live 를 OOS 에서 한 번, 6조건 게이트(§8).
    describe 패리티·체결률·MFE·퍼널·국면 분할(§9).

재사용(복사 금지): 리플레이·비용·단계·대조군·부트스트랩은 ``tick_replay_optimize``,
B 청산 설정은 ``algo_flow_reversal.b_setup_factory`` 를 import 한다.
안전: DB SELECT 만, 주문 경로 없음. 출력은 ``data/entry_timing_79/`` 와
``research/logs/79-entry-timing/`` 뿐.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/entry_timing_79.py replay --arm T1 --exit live
    uv run --extra pg python scripts/entry_timing_79.py extras
    uv run --extra pg python scripts/entry_timing_79.py select
    uv run --extra pg python scripts/entry_timing_79.py oos
    uv run --extra pg python scripts/entry_timing_79.py describe
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from krx_quant_core.market.limits import limit_up_price
from krx_quant_core.stats import trials as trial_ledger
from krx_quant_core.stats.metrics import paired_date_bootstrap
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import algo_flow_reversal as afr  # noqa: E402
import tick_replay_optimize as tro  # noqa: E402

from scalp_it.intraday.store import connect  # noqa: E402

ROOT = tro.ROOT
OUT = ROOT / "data" / "entry_timing_79"
LOGS_DIR = ROOT / "research" / "logs"
LABEL = "79-entry-timing"

IS_DAYS, OOS_DAYS, ALL_DAYS = tro.IS_DAYS, tro.OOS_DAYS, tro.ALL_DAYS
LOCK_SECS = 3.0                      # §2 L
T4_MAX_RET60 = 0.02                  # §2 X
ARMS = ["T1", "T2", "T4"]            # 동률 순서
EXITS = ["live", "B"]
CELL_P = (None, True, 0.0, None, None)          # §4 주 집합
NATIVE = {"live": tro.LIVE_CELL, "B": afr.CELL_B}
IS_MIN_N = 10
OOS_MIN_N = 30
N_BOOT = 10_000
SEED = 79
N_PRIOR = 30 + 48 + 16
HOLD_S = 180


# ================================================================ 1. 리플레이
def setup_factory(orig: Any, arm: str, exit_: str) -> Any:
    base = afr.b_setup_factory(orig) if exit_ == "B" else orig

    def setup(day: str) -> tuple[Any, Any, Any]:
        live_cfg, run_cfg, guard_cfg = base(day)
        if arm == "T2":
            kw = {"limitup_near_ticks": 0, "limitup_lock_secs": LOCK_SECS, "vi_near_pct": 0.0}
            live_cfg, run_cfg = replace(live_cfg, **kw), replace(run_cfg, **kw)
        return live_cfg, run_cfg, guard_cfg
    return setup


def cache_dir(arm: str, exit_: str) -> Path:
    return OUT / f"{'T2' if arm == 'T4' else arm}_{exit_}"


def cmd_replay(ns: argparse.Namespace) -> None:
    assert ns.arm in ("T1", "T2") and ns.exit in EXITS
    tro.live_setup = setup_factory(tro.live_setup, ns.arm, ns.exit)
    tro.OUT = cache_dir(ns.arm, ns.exit)
    for d in (ns.days or ALL_DAYS):
        tro.replay_day(d, mode="stored", force=ns.force)


def load(arm: str, exit_: str, day: str) -> dict[str, Any]:
    with gzip.open(cache_dir(arm, exit_) / "stored" / f"{day}.json.gz", "rt",
                   encoding="utf-8") as fh:
        return json.load(fh)


def caches(arm: str, exit_: str) -> dict[str, dict]:
    out = {d: load(arm, exit_, d) for d in ALL_DAYS}
    if arm == "T4":
        ex = extras()
        for d, c in out.items():
            c["candidates"] = [x for x in c["candidates"]
                               if (r := ex["ret60"].get(key(x))) is not None and r < T4_MAX_RET60]
    return out


def key(c: dict) -> str:
    return f"{c['ts']}|{c['leader']}|{c['follower']}"


# ================================================================ 2. 부가 계산
def _last_px(conn: Any, code: str, t: datetime, day: str) -> float | None:
    cur = conn.cursor()
    cur.execute("select price from ticks where code = %s and ts >= %s and ts <= %s "
                "and price is not null order by ts desc, seq desc limit 1", (code, day, t))
    r = cur.fetchone()
    cur.close()
    return float(r[0]) if r else None


def _reached_lock(conn: Any, leader: str, t: datetime, day: str) -> bool | None:
    cur = conn.cursor()
    cur.execute("select close from daily_bars where code = %s and date < %s and close > 0 "
                "order by date desc limit 1", (leader, date.fromisoformat(day)))
    r = cur.fetchone()
    if not r:
        cur.close()
        return None
    lim = limit_up_price(float(r[0]))
    cur.execute("select ts, price from ticks where code = %s and ts >= %s and ts <= %s "
                "and price is not null order by ts, seq", (leader, t, t + timedelta(seconds=HOLD_S)))
    since = None
    for ts, px in cur.fetchall():
        if float(px) >= lim:
            since = since or ts
            if (ts - since).total_seconds() >= LOCK_SECS:
                cur.close()
                return True
        else:
            since = None
    cur.close()
    return False


def extras(force: bool = False) -> dict[str, Any]:
    path = OUT / "extras.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    conn = connect(os.environ.get("KR_QUANT_DB"))
    ret60: dict[str, float | None] = {}
    lock: dict[str, bool | None] = {}
    for d in ALL_DAYS:
        for c in load("T2", "live", d)["candidates"] + load("T2", "B", d)["candidates"]:
            k = key(c)
            if k in ret60:
                continue
            t = datetime.fromisoformat(c["ts"])
            p0 = _last_px(conn, c["follower"], t - timedelta(seconds=60), d)
            ret60[k] = (c["price"] / p0 - 1.0) if p0 else None
        for c in load("T1", "live", d)["candidates"]:
            k = key(c)
            if k not in lock:
                lock[k] = _reached_lock(conn, c["leader"], datetime.fromisoformat(c["ts"]), d)
    conn.close()
    out = {"ret60": ret60, "t1_leader_lock180": lock}
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def cmd_extras(ns: argparse.Namespace) -> None:
    ex = extras(force=ns.force)
    print("ret60", len(ex["ret60"]), "lock", len(ex["t1_leader_lock180"]))


# ================================================================ 3. 평가
def frame(arm: str, exit_: str, days: list[str], cell: tuple = CELL_P,
          cs: dict | None = None) -> pd.DataFrame:
    cs = cs or caches(arm, exit_)
    df = tro.trades(cs, days, cell)
    return df


def funnel(cs: dict, days: list[str], cell: tuple = CELL_P) -> dict[str, int]:
    n_c = n_s = n_f = n_ft = 0
    for d in days:
        for c in cs[d]["candidates"]:
            n_c += 1
            if tro.stage(c, cell, cs[d]["kosdaq_jo"]) != "trade":
                continue
            n_s += 1
            if tro.trade_return(c, "through") is not None:
                n_f += 1
            if tro.trade_return(c, "touch") is not None:
                n_ft += 1
    return {"cands": n_c, "stage_pass": n_s, "filled_through": n_f, "filled_touch": n_ft,
            "fill_rate": (n_f / n_s) if n_s else None}


def trial_cfg(arm: str, exit_: str) -> dict[str, Any]:
    return {"arm": arm, "exit": exit_, "lock_secs": LOCK_SECS if arm != "T1" else None,
            "t4_max_ret60": T4_MAX_RET60 if arm == "T4" else None, "cell": tro.cell_name(CELL_P),
            "objective": "IS daymean net PRIMARY through", "prereg": "docs/research/manju/79"}


def cmd_select(ns: argparse.Namespace) -> None:
    rows = []
    for arm in ARMS:
        for ex in EXITS:
            cs = caches(arm, ex)
            df = frame(arm, ex, IS_DAYS, cs=cs)
            obj = tro.daymean_objective(df)
            trial_ledger.record_trial(LABEL, trial_cfg(arm, ex), logs_dir=LOGS_DIR,
                                      note="IS trial (prereg 79 §6)")
            rows.append({"arm": arm, "exit": ex, "n": len(df), "obj": obj,
                         "mean": float(df.ret.mean()) if len(df) else None,
                         "median": float(df.ret.median()) if len(df) else None,
                         "funnel": funnel(cs, IS_DAYS),
                         "eligible": len(df) >= IS_MIN_N and obj == obj})
    el = [r for r in rows if r["eligible"]]
    sel = None
    if el:
        best = max(round(r["obj"], 6) for r in el)
        sel = next(r for r in el if round(r["obj"], 6) == best)  # rows 는 동률 순서대로
    out = {"rows": rows, "selected": None if sel is None else [sel["arm"], sel["exit"]]}
    (OUT / "select_is.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    for r in rows:
        print(r["arm"], r["exit"], r["n"], r["obj"], r["median"], r["funnel"], r["eligible"])
    print("selected", out["selected"])


def n_trials() -> tuple[int, dict[str, int]]:
    extra = {}
    for p in sorted(LOGS_DIR.glob("7[78]*/TRIALS.jsonl")):
        extra[p.parent.name] = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln)
    return len(ARMS) * len(EXITS) + N_PRIOR + sum(extra.values()), extra


def gate(df: pd.DataFrame, arm: str, exit_: str, cs: dict) -> dict[str, Any]:
    res: dict[str, Any] = {"n": len(df)}
    if not len(df):
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    R = df.ret.to_numpy()
    tro.SEED, tro.N_BOOT = SEED, N_BOOT
    res.update(mean=float(R.mean()), median=float(np.median(R)), win=float((R > 0).mean()),
               ci=tro.day_cluster_ci(df, OOS_DAYS), days=int(df.day.nunique()))
    nd = df.dropna(subset=["null_ret"])
    res["null"] = None
    if len(nd):
        a = nd.groupby("day").ret.mean().rename("mean_excess").rename_axis("date").reset_index()
        b = nd.groupby("day").null_ret.mean().rename("mean_excess").rename_axis("date").reset_index()
        try:
            pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
            res["null"] = {"n_pairs": len(nd), "strat": float(nd.ret.mean()),
                           "null": float(nd.null_ret.mean()), **pb, "p": 1.0 - pb["p_greater"]}
        except ValueError as e:
            res["null_err"] = str(e)
    N, extra = n_trials()
    res["N"], res["N_extra"] = N, extra
    res["dsr"] = deflated_sharpe_from_sample(R, N)
    full = tro.trades(cs, ALL_DAYS, CELL_P)
    weeks = {w: (float(full.loc[full.day.isin(ds), "ret"].mean()) if full.day.isin(ds).any()
                 else None, int(full.day.isin(ds).sum())) for w, ds in tro.WEEKS.items()}
    res["weeks"] = weeks
    res["neg_weeks"] = sum(1 for m, _ in weeks.values() if m is not None and m < 0)
    pd_ = (res["dsr"] or {}).get("prob_deflated_sharpe") or 0.0
    conds = {
        "1 OOS n≥30": len(df) >= OOS_MIN_N,
        "2 평균>0·중앙>0": res["mean"] > 0 and res["median"] > 0,
        "3 일클러스터 CI하한>0": res["ci"][0] > 0,
        "4 무작위대조 p<0.05": bool(res["null"]) and res["null"]["p"] < 0.05,
        f"5 DSR확률>0.95(N={N})": pd_ > 0.95,
        "6 음수 주 ≤1": res["neg_weeks"] <= 1,
    }
    res["conds"] = conds
    if not conds["1 OOS n≥30"]:
        res["verdict"] = "표본 부족 — 판정 불가"
    else:
        res["verdict"] = "채택(즉시 라이브 반영)" if all(conds.values()) else "기각"
    return res


def cmd_oos(ns: argparse.Namespace) -> None:
    sel = json.loads((OUT / "select_is.json").read_text(encoding="utf-8"))["selected"]
    todo = [("baseline T1×live", "T1", "live")]
    if sel is not None:
        todo.insert(0, (f"selected {sel[0]}×{sel[1]}", sel[0], sel[1]))
    out: dict[str, Any] = {"selected": sel}
    for label, arm, ex in todo:
        cs = caches(arm, ex)
        df = frame(arm, ex, OOS_DAYS, cs=cs)
        g = gate(df, arm, ex, cs)
        g["funnel"] = funnel(cs, OOS_DAYS)
        nat = frame(arm, ex, OOS_DAYS, NATIVE[ex], cs=cs)
        g["native"] = {"cell": tro.cell_name(NATIVE[ex]), "n": len(nat),
                       "mean": float(nat.ret.mean()) if len(nat) else None,
                       "median": float(nat.ret.median()) if len(nat) else None}
        g["gross_mean"] = (float(tro.trades(cs, OOS_DAYS, CELL_P, cost="gross").ret.mean())
                           if len(df) else None)
        g["reasons"] = df.reason.value_counts().to_dict() if len(df) else {}
        out[label] = g
        print(label, {k: g.get(k) for k in ("n", "mean", "median", "ci", "verdict", "native",
                                             "gross_mean")})
        print("   conds", g.get("conds"), "null", (g.get("null") or {}).get("p"),
              "dsr", (g.get("dsr") or {}).get("prob_deflated_sharpe"), "weeks", g.get("weeks"))
    if sel is None:
        out["verdict"] = "표본 부족 — 판정 불가"
    else:
        out["verdict"] = out[f"selected {sel[0]}×{sel[1]}"]["verdict"]
    (OUT / "oos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
    print("VERDICT", out["verdict"])


# ================================================================ 4. 서술
def cmd_describe(ns: argparse.Namespace) -> None:
    desc: dict[str, Any] = {}
    # 패리티: 새 T1 캐시 vs 75·76 캐시
    par = {}
    for ex, loader in (("live", afr.load_a), ("B", afr.load_b)):
        rows = []
        for d in ALL_DAYS:
            new = {(c["ts"], c["leader"], c["follower"], c["entry"]) for c in load("T1", ex, d)["candidates"]}
            old = {(c["ts"], c["leader"], c["follower"], c["entry"]) for c in loader(d)["candidates"]}
            same_fill = sum(1 for a, b in zip(load("T1", ex, d)["candidates"], loader(d)["candidates"])
                            if (a["through"] or {}).get("exit_ts") == (b["through"] or {}).get("exit_ts"))
            rows.append({"day": d, "new": len(new), "old": len(old), "common": len(new & old),
                         "same_exit_ts": same_fill})
        par[ex] = rows
    desc["parity"] = par
    ex_ = extras()
    lk = [v for v in ex_["t1_leader_lock180"].values() if v is not None]
    desc["t1_leader_lock180"] = {"n": len(lk), "share": (sum(lk) / len(lk)) if lk else None}
    stats = {}
    for arm in ARMS:
        for ex in EXITS:
            cs = caches(arm, ex)
            recs = []
            for d in ALL_DAYS:
                jo = cs[d]["kosdaq_jo"]
                for c in cs[d]["candidates"]:
                    if tro.stage(c, CELL_P, jo) != "trade":
                        continue
                    r = tro.trade_return(c, "through")
                    if r is None:
                        continue
                    p = c["through"]
                    hold = ((datetime.fromisoformat(p["exit_ts"]) - datetime.fromisoformat(p["fill_ts"]))
                            .total_seconds() if p["exit_ts"] and p["fill_ts"] else None)
                    recs.append({"day": d, "ret": r, "gross": tro.trade_return(c, "through", "gross"),
                                 "mfe": p["mfe"], "mae": p["mae"], "reason": p["reason"],
                                 "hold": hold, "jo": jo, "is": d in IS_DAYS})
            df = pd.DataFrame(recs)
            s: dict[str, Any] = {"n": len(df), "funnel_all": funnel(cs, ALL_DAYS),
                                 "funnel_is": funnel(cs, IS_DAYS), "funnel_oos": funnel(cs, OOS_DAYS),
                                 "per_day_trades": {d: int((df.day == d).sum()) if len(df) else 0
                                                    for d in ALL_DAYS}}
            if len(df):
                m = df.mfe
                s.update(mfe_median=float(m.median()), mfe_zero=float((m <= 1e-12).mean()),
                         mfe_ge05=float((m >= 0.005).mean()), mfe_ge1=float((m >= 0.01).mean()),
                         mfe_ge3=float((m >= 0.03).mean()), mae_median=float(df.mae.median()),
                         hold_median=float(df.hold.median()),
                         reasons={k: [int(v), float(df.loc[df.reason == k, "ret"].mean())]
                                  for k, v in df.reason.value_counts().items()},
                         all16_mean=float(df.ret.mean()), all16_median=float(df.ret.median()),
                         all16_gross=float(df.gross.mean()))
                reg = {}
                for lab, lo, hi in (("<5조", 0, 5), ("5~8조", 5, 8), ("≥8조", 8, 1e9)):
                    sub = df[(df.jo >= lo) & (df.jo < hi)]
                    reg[lab] = {"n": len(sub), "days": int(sub.day.nunique()),
                                "mean": float(sub.ret.mean()) if len(sub) else None,
                                "median": float(sub.ret.median()) if len(sub) else None}
                s["regime"] = reg
            stats[f"{arm}×{ex}"] = s
            print(arm, ex, {k: s.get(k) for k in ("n", "mfe_median", "mfe_zero", "mfe_ge05",
                                                   "mfe_ge3", "hold_median", "funnel_all")})
    desc["arms"] = stats
    (OUT / "describe.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    print(json.dumps({"parity": par, "lock180": desc["t1_leader_lock180"]}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--arm", required=True)
    p.add_argument("--exit", required=True)
    p.add_argument("--days", nargs="*")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("extras")
    p.add_argument("--force", action="store_true")
    for name in ("select", "oos", "describe"):
        sub.add_parser(name)
    ns = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"replay": cmd_replay, "extras": cmd_extras, "select": cmd_select, "oos": cmd_oos,
     "describe": cmd_describe}[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
