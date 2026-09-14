#!/usr/bin/env python3
"""만쥬 재료 확인 — 대장에 신선한 호재 재료가 있을 때 짝꿍이 나은가. 사전등록 77번.

``docs/research/manju/77-material-leader-preregistration.md`` 가 정본이다.

단계
    collect   72번 ``pair_live_rules_sanity.collect`` 를 import 해 거래를 캐시(별도 실행)
    s1        표본 1(분봉·72번): 플래그 → IS 선택 → OOS 게이트 → 서술
    s2        표본 2(틱·75번 캐시): MNEWS 플래그 → 16일 게이트 → 서술

안전: DB·sqlite SELECT 만. 주문 경로 없음. 출력은 data/material_leader_77/, research/logs/77-*/.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/material_leader_sanity.py s1
    uv run --extra pg python scripts/material_leader_sanity.py s2
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.stats import trials as trial_ledger
from krx_quant_core.stats.metrics import paired_date_bootstrap
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tick_replay_optimize as tro  # noqa: E402

from scalp_it.adapters.db import load_stock_master  # noqa: E402
from scalp_it.dart.classify import classify_report, is_correction  # noqa: E402
from scalp_it.intraday.store import connect  # noqa: E402
from scalp_it.ticksize import tick_size  # noqa: E402
from scalp_it.universe import MATERIAL_WEIGHTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "material_leader_77"
LOGS_DIR = ROOT / "research" / "logs"
LABEL = "77-material-leader"
KST = ZoneInfo("Asia/Seoul")

# ---------------------------------------------------------------- 사전등록 상수
STRONG = {k for k, w in MATERIAL_WEIGHTS.items() if w >= 0.7}
S1_IS_END, S1_OOS_START = "2026-03-31", "2026-04-01"
USER_WINDOW_START = "2026-07-01"
VARIANTS = {"M3d": 3, "M1d": 1}                 # 동률이면 앞(M3d)
NEWS_CONF, NEWS_HOURS = 70, 72
IS_MIN = 20
OOS_MIN_N = 30
N_BOOT = 10_000
SEED = 77
N_DSR = 103
S2_WEEKS = [tro.ALL_DAYS[0:5], tro.ALL_DAYS[5:10], tro.ALL_DAYS[10:15], tro.ALL_DAYS[15:16]]
CELL_WIDE = (None, False, 0.0, None, None)


# ================================================================ 재료 원천
def _sqlite_rows(path: Path) -> list[tuple]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
        return c.execute("select rcept_no, rcept_dt, code, report_nm, material_type, "
                         "is_correction from disclosures").fetchall()


def strong_disclosures(conn: Any) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for path in (ROOT / "data" / "dart_2024.db", ROOT / "data" / "dart.db"):  # 뒤가 우선
        for rno, rdt, code, nm, mt, corr in _sqlite_rows(path):
            rows[str(rno)] = {"rcept_no": str(rno), "d": rdt, "code": str(code).strip(),
                              "nm": nm or "", "type": mt, "corr": bool(corr), "src": path.name}
    cur = conn.cursor()
    cur.execute("select url, ticker, title, published_at from disclosures where source='dart'")
    for url, code, title, pub in cur.fetchall():
        m = re.search(r"rcept_no=(\d+)", url or "")
        if not m or m.group(1) in rows or not code:
            continue
        rows[m.group(1)] = {"rcept_no": m.group(1),
                            "d": pub.astimezone(KST).strftime("%Y%m%d"),
                            "code": str(code).strip(), "nm": title or "",
                            "type": classify_report(title or ""), "corr": False, "src": "timescale"}
    cur.close()
    df = pd.DataFrame(rows.values())
    df["corr"] = df["corr"] | df.nm.map(is_correction)
    df = df[df.type.isin(STRONG) & ~df["corr"]].copy()
    df["d"] = pd.to_datetime(df.d, format="%Y%m%d").dt.date
    return df


class Calendar:
    def __init__(self, conn: Any) -> None:
        cur = conn.cursor()
        cur.execute("select distinct date from daily_bars where date >= '2023-12-01' order by date")
        self.days = [r[0] for r in cur.fetchall()]
        cur.close()

    def k_back(self, d: date, k: int) -> date:
        i = bisect.bisect_left(self.days, d)
        return self.days[max(0, i - k)]


def dart_index(df: pd.DataFrame) -> dict[str, list[tuple[date, str]]]:
    out: dict[str, list[tuple[date, str]]] = {}
    for r in df.sort_values("d").itertuples():
        out.setdefault(r.code, []).append((r.d, r.type))
    return out


def dart_flags(idx: dict, cal: Calendar, code: str, d: date) -> dict[str, Any]:
    lst = idx.get(code, [])
    res: dict[str, Any] = {"same": any(x == d for x, _ in lst)}
    for name, k in VARIANTS.items():
        lo, hi = cal.k_back(d, k), d - timedelta(days=1)
        hits = [t for x, t in lst if lo <= x <= hi]
        res[name] = bool(hits)
        res[name + "_types"] = hits
    return res


def bucket(fl: dict[str, Any], var: str) -> str:
    if fl[var]:
        return "ON"
    return "SAME" if fl["same"] else "OFF"


# ================================================================ 통계
def day_ci(df: pd.DataFrame, col: str = "r") -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    g = df.groupby("day")[col]
    s, c = g.sum().to_numpy(), g.count().to_numpy()
    pick = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    m = s[pick].sum(1) / c[pick].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def on_vs_off(df: pd.DataFrame, col: str = "r") -> dict[str, Any]:
    """거래일 복원추출 → (ON 평균 − OFF 평균). p = P(차이 ≤ 0)."""
    rng = np.random.default_rng(SEED)
    days = sorted(df.day.unique())
    on = df[df.b == "ON"].groupby("day")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    off = df[df.b == "OFF"].groupby("day")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    pick = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    os_, oc = on["sum"].to_numpy()[pick].sum(1), on["count"].to_numpy()[pick].sum(1)
    fs, fc = off["sum"].to_numpy()[pick].sum(1), off["count"].to_numpy()[pick].sum(1)
    ok = (oc > 0) & (fc > 0)
    diff = os_[ok] / oc[ok] - fs[ok] / fc[ok]
    obs = float(df[df.b == "ON"][col].mean() - df[df.b == "OFF"][col].mean())
    return {"diff": obs, "p": float((diff <= 0).mean()), "ci": [float(np.percentile(diff, 2.5)),
                                                               float(np.percentile(diff, 97.5))]}


def permutation(df: pd.DataFrame, group: pd.Series, col: str = "r") -> dict[str, Any]:
    """그룹(월·주) 안에서 ON 라벨을 같은 개수로 재배정. p = P(무작위 ON 평균 ≥ 실제)."""
    rng = np.random.default_rng(SEED)
    sub = df[df.b.isin(["ON", "OFF"])]
    grp = group.loc[sub.index]
    obs = float(sub[sub.b == "ON"][col].mean())
    tot = np.zeros(N_BOOT)
    n_on = int((sub.b == "ON").sum())
    for _, g in sub.groupby(grp):
        k = int((g.b == "ON").sum())
        if k == 0:
            continue
        vals = g[col].to_numpy()
        idx = np.argsort(rng.random((N_BOOT, len(vals))), axis=1)[:, :k]
        tot += vals[idx].sum(1)
    null = tot / n_on
    return {"obs": obs, "null_mean": float(null.mean()), "p": float((null >= obs).mean())}


def summ(R: pd.Series) -> dict[str, Any]:
    R = R.dropna()
    if not len(R):
        return {"n": 0}
    return {"n": int(len(R)), "mean": float(R.mean()), "median": float(R.median()),
            "win": float((R > 0).mean())}


# ================================================================ 표본 1
def s1_frame(conn: Any, cal: Calendar, idx: dict) -> pd.DataFrame:
    data = pd.read_pickle(OUT / "pair_live_72.pkl")
    tr = data["trades"].copy()
    master = load_stock_master(conn)
    mkt = dict(zip(master.code.astype(str).str.strip(), master.market, strict=False))
    tr["day"] = pd.to_datetime(tr.date).dt.strftime("%Y-%m-%d")
    e_adj = tr.entry + tr.entry.map(lambda p: tick_size(float(p)))
    x_adj = tr.exit - tr.exit.map(lambda p: tick_size(float(p)))
    cost = [round_trip_cost(date.fromisoformat(d), "KOSPI" if mkt.get(c) == "거래소" else "KOSDAQ")
            for d, c in zip(tr.day, tr.fol, strict=True)]
    tr["r"] = x_adj / e_adj - 1 - np.array(cost)
    tr["net64"] = tr.ret - 0.0064
    fl = [dart_flags(idx, cal, c, date.fromisoformat(d)) for c, d in zip(tr.lead, tr.day, strict=True)]
    for key in ("same", *VARIANTS, *(v + "_types" for v in VARIANTS)):
        tr[key] = [f[key] for f in fl]
    return tr


def gate_s1(oos: pd.DataFrame) -> dict[str, Any]:
    on = oos[oos.b == "ON"]
    res: dict[str, Any] = {"on": summ(on.r), "off": summ(oos[oos.b == "OFF"].r),
                           "same": summ(oos[oos.b == "SAME"].r)}
    if not len(on):
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    res["ci"] = day_ci(on)
    res["vs_off"] = on_vs_off(oos)
    month = oos.day.str[:7]
    res["perm"] = permutation(oos, month)
    res["dsr"] = deflated_sharpe_from_sample(on.r.to_numpy(), N_DSR)
    res["months"] = on.groupby(on.day.str[:7]).r.mean().to_dict()
    pdsr = res["dsr"].get("prob_deflated_sharpe") or 0.0
    s = res["on"]
    conds = {
        "1 ON n≥30": s["n"] >= OOS_MIN_N,
        "2 평균>0·중앙>0": s["mean"] > 0 and s["median"] > 0,
        "3 일클러스터 CI하한>0": res["ci"][0] > 0,
        "4 ON>OFF p<0.05": res["vs_off"]["p"] < 0.05,
        "5 무작위(월내 재배정) p<0.05": res["perm"]["p"] < 0.05,
        "6 DSR확률>0.95(N=103)": pdsr > 0.95,
        "7 월 음수 ≤1": sum(1 for v in res["months"].values() if v < 0) <= 1,
    }
    res["conds"] = conds
    res["verdict"] = ("표본 부족 — 판정 불가" if not conds["1 ON n≥30"]
                      else "채택" if all(conds.values()) else "기각")
    return res


def cmd_s1(ns: argparse.Namespace) -> None:
    conn = connect()
    cal = Calendar(conn)
    sd = strong_disclosures(conn)
    idx = dart_index(sd)
    tr = s1_frame(conn, cal, idx)
    conn.close()
    tr.to_pickle(OUT / "s1_trades.pkl")
    is_ = tr[tr.day <= S1_IS_END]
    rows = []
    for var, k in VARIANTS.items():
        d = is_.assign(b=is_.apply(lambda r, v=var: bucket(r, v), axis=1))
        on, off = d[d.b == "ON"].r, d[d.b == "OFF"].r
        obj = float(on.mean() - off.mean()) if len(on) and len(off) else float("nan")
        trial_ledger.record_trial(LABEL, {"sample": "S1 minute pair72", "flag": var, "K_days": k,
                                          "types": sorted(STRONG), "no_correction": True,
                                          "same_day": "excluded bucket",
                                          "cost": "entry+1tick exit-1tick round_trip_cost",
                                          "is_end": S1_IS_END, "prereg": "docs/research/manju/77"},
                                  logs_dir=LOGS_DIR, note="IS trial (prereg 77 §4-1)")
        rows.append({"var": var, "on": summ(on), "off": summ(off),
                     "same": summ(d[d.b == "SAME"].r), "obj": obj,
                     "eligible": len(on) >= IS_MIN and len(off) >= IS_MIN and obj == obj and obj > 0})
    el = [r for r in rows if r["eligible"]]
    sel = None
    if el:
        best = max(round(r["obj"], 6) for r in el)
        sel = next(r["var"] for r in rows if r["eligible"] and round(r["obj"], 6) == best)
    use = sel or "M3d"
    tr["b"] = tr.apply(lambda r: bucket(r, use), axis=1)
    oos = tr[tr.day >= S1_OOS_START]
    g = gate_s1(oos)
    if sel is None:
        g["verdict"] = "기각(IS 신호 없음)"
    # 서술
    desc: dict[str, Any] = {
        "n_trades": len(tr), "n_is": len(is_), "n_oos": len(oos),
        "excl": pd.read_pickle(OUT / "pair_live_72.pkl")["excl"],
        "all_net64_72": summ(tr.net64), "all_r": summ(tr.r),
        "buckets_all": {b: summ(v.r) for b, v in tr.groupby("b")},
        "buckets_oos_net64": {b: summ(v.net64) for b, v in oos.groupby("b")},
        "user_window": {b: summ(v.r) for b, v in tr[tr.day >= USER_WINDOW_START].groupby("b")},
        "exit_kind_oos": {f"{b}|{k}": int(len(v)) for (b, k), v in oos.groupby(["b", "exit_kind"])},
        "by_type_all": {},
        "n_strong_disclosures": len(sd),
        "variant_oos": {},
    }
    for var in VARIANTS:
        bb = oos.apply(lambda r, v=var: bucket(r, v), axis=1)
        desc["variant_oos"][var] = {b: summ(oos[bb == b].r) for b in ("ON", "OFF", "SAME")}
    on_all = tr[tr.b == "ON"]
    for t in sorted(STRONG):
        m = on_all[use + "_types"].map(lambda xs, t=t: t in xs)
        desc["by_type_all"][t] = summ(on_all[m].r)
    out = {"is_rows": rows, "selected": sel, "used": use, "gate": g, "desc": desc}
    OUT.joinpath("s1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


# ================================================================ 표본 2
def news_index(conn: Any) -> dict[str, list[tuple[datetime, dict[str, Any]]]]:
    cur = conn.cursor()
    cur.execute("select ticker, judged_at, source_type, source_id, event_type, confidence, "
                "rationale from news_judgments where sentiment_direction = 1 and "
                "price_impact_likely and not is_stale_repeat and confidence >= %s", (NEWS_CONF,))
    out: dict[str, list] = {}
    for tick, ja, st, sid, ev, conf, rat in cur.fetchall():
        if not tick:
            continue
        t = ja.astimezone(KST).replace(tzinfo=None)
        out.setdefault(str(tick).strip(), []).append(
            (t, {"source_type": st, "source_id": sid, "event_type": ev, "confidence": conf,
                 "rationale": (rat or "")[:160], "judged_at": t.isoformat()}))
    cur.close()
    return out


def news_hit(nidx: dict, code: str, ts: datetime) -> list[dict[str, Any]]:
    lo = ts - timedelta(hours=NEWS_HOURS)
    return [m for t, m in nidx.get(code, []) if lo <= t < ts]


def s2_frame(source: str, cal: Calendar, idx: dict, nidx: dict, dart_var: str,
             cell: tuple = CELL_WIDE) -> pd.DataFrame:
    rows = []
    for d in tro.ALL_DAYS:
        if source == "A":
            path = ROOT / "data" / "tick_replay_75" / "stored" / f"{d}.json.gz"
        else:
            path = ROOT / "data" / "algo_flow_76" / "replay_B" / "stored" / f"{d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            cache = json.load(fh)
        for c in cache["candidates"]:
            if tro.stage(c, cell, cache["kosdaq_jo"]) != "trade":
                continue
            r = tro.trade_return(c)
            if r is None:
                continue
            ts = datetime.fromisoformat(c["ts"])
            fl = dart_flags(idx, cal, c["leader"], date.fromisoformat(d))
            nh = news_hit(nidx, c["leader"], ts)
            dart_on = fl[dart_var]
            on = dart_on or bool(nh)
            b = "ON" if on else ("SAME" if fl["same"] else "OFF")
            rows.append({"day": d, "ts": c["ts"], "leader": c["leader"], "follower": c["follower"],
                         "r": r, "null_r": tro.trade_return(c, "through", "primary", key="null"),
                         "reason": c["through"]["reason"], "dart_on": dart_on,
                         "news_on": bool(nh), "news": nh[:2], "same": fl["same"], "b": b})
    return pd.DataFrame(rows)


def cmd_s2(ns: argparse.Namespace) -> None:
    s1 = json.loads(OUT.joinpath("s1.json").read_text())
    dart_var = s1["used"]
    conn = connect()
    cal = Calendar(conn)
    idx = dart_index(strong_disclosures(conn))
    nidx = news_index(conn)
    conn.close()
    trial_ledger.record_trial(LABEL, {"sample": "S2 tick replay75 A-wide", "flag": "MNEWS",
                                      "dart": dart_var, "news_conf": NEWS_CONF,
                                      "news_hours": NEWS_HOURS, "days": "16 all",
                                      "prereg": "docs/research/manju/77"},
                              logs_dir=LOGS_DIR, note="S2 evaluation (prereg 77 §5)")
    df = s2_frame("A", cal, idx, nidx, dart_var)
    df.to_pickle(OUT / "s2_trades.pkl")
    on = df[df.b == "ON"]
    res: dict[str, Any] = {"dart_var": dart_var, "n_all": len(df), "on": summ(on.r),
                           "off": summ(df[df.b == "OFF"].r), "same": summ(df[df.b == "SAME"].r)}
    if len(on):
        res["ci"] = day_ci(on)
        res["vs_off"] = on_vs_off(df)
        nd = on.dropna(subset=["null_r"])
        a = nd.groupby("day").r.mean().rename("mean_excess").rename_axis("date").reset_index()
        bb = nd.groupby("day").null_r.mean().rename("mean_excess").rename_axis("date").reset_index()
        try:
            pb = paired_date_bootstrap(a, bb, n_boot=N_BOOT, seed=SEED)
            res["null"] = {**pb, "p": 1 - pb["p_greater"], "strat": float(nd.r.mean()),
                           "null": float(nd.null_r.mean()), "n_pairs": len(nd)}
        except ValueError:
            res["null"] = None
        wk = df.day.map({d: i for i, w in enumerate(S2_WEEKS) for d in w})
        res["perm_week"] = permutation(df, wk)
        res["dsr"] = deflated_sharpe_from_sample(on.r.to_numpy(), N_DSR)
        res["weeks"] = {i: (float(on[on.day.isin(w)].r.mean()) if on.day.isin(w).any() else None)
                        for i, w in enumerate(S2_WEEKS)}
        pdsr = res["dsr"].get("prob_deflated_sharpe") or 0.0
        conds = {
            "1 ON n≥30": res["on"]["n"] >= OOS_MIN_N,
            "2 평균>0·중앙>0": res["on"]["mean"] > 0 and res["on"]["median"] > 0,
            "3 일클러스터 CI하한>0": res["ci"][0] > 0,
            "4 ON>OFF p<0.05": res["vs_off"]["p"] < 0.05,
            "5 무작위대조(75 대조군) p<0.05": bool(res["null"]) and res["null"]["p"] < 0.05,
            "6 DSR확률>0.95(N=103)": pdsr > 0.95,
            "7 주 음수 ≤1": sum(1 for v in res["weeks"].values() if v is not None and v < 0) <= 1,
        }
        res["conds"] = conds
        res["verdict"] = ("표본 부족 — 판정 불가" if not conds["1 ON n≥30"]
                          else "채택" if all(conds.values()) else "기각")
    else:
        res["verdict"] = "표본 부족 — 판정 불가"
    desc: dict[str, Any] = {
        "dart_only": summ(df[df.dart_on & ~df.news_on].r),
        "news_only": summ(df[df.news_on & ~df.dart_on].r),
        "both": summ(df[df.news_on & df.dart_on].r),
        "on_leaders": on.leader.value_counts().to_dict(),
        "on_days": on.day.value_counts().to_dict(),
        "is8": {b: summ(v.r) for b, v in df[df.day.isin(tro.IS_DAYS)].groupby("b")},
        "oos8": {b: summ(v.r) for b, v in df[df.day.isin(tro.OOS_DAYS)].groupby("b")},
        "news_window_days": {b: summ(v.r) for b, v in df[df.day >= "2026-09-07"].groupby("b")},
        "reasons_on": on.reason.value_counts().to_dict(),
        "examples": [x for lst in on[on.news_on].news.head(3) for x in lst[:1]],
    }
    b_df = s2_frame("B", cal, idx, nidx, dart_var)
    desc["B_wide"] = {b: summ(v.r) for b, v in b_df.groupby("b")}
    live = s2_frame("A", cal, idx, nidx, dart_var, cell=tro.LIVE_CELL)
    desc["A_live_cell"] = {b: summ(v.r) for b, v in live.groupby("b")}
    res["desc"] = desc
    OUT.joinpath("s2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("s1")
    sp.add_parser("s2")
    ns = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"s1": cmd_s1, "s2": cmd_s2}[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
