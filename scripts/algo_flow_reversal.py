#!/usr/bin/env python3
"""만쥬 원형 복원 + 알고리즘 체결 패턴(1초 몰아치기·비슷한 수량 반복) — 사전등록 76번.

``docs/research/manju/76-manju-pure-algo-flow-reversal-preregistration.md`` 가 판정 기준의
정본이다. 여기 상수를 바꾸려면 문서부터 바꿔야 하는데, 측정 후에는 바꿀 수 없다.

단계
====

    replay     설정 B(만쥬 원형: 가격손절·본절탈출 끔) 로 75번 ``replay_day`` 를 그대로 돌려
               후보·가드 청산을 캐시(``data/algo_flow_76/replay_B/stored``). 멱등.
    features   후보마다 대장·후속·대조군 종목의 신호 직전 틱으로 P1/P2/REV 원시값 계산,
               같은 종목 무작위 시점도 함께(서술용). ``ts < 신호초`` 만 쓴다. 멱등.
    select     IS 후보 분위수로 θ 고정 → B-넓음 IS 에서 16 시행 평가·원장 기록·1개 선택.
    oos        선택 특징을 OOS 에 한 번 적용, 7조건 게이트(B-필터 ON).
    describe   B vs A, 패턴 빈도, 예시, 국면 일수.
    report     마크다운.

재사용(복사 금지): 리플레이·비용·필터 단계·대조군은 ``scripts/tick_replay_optimize.py`` 를
import 한다. 안전: DB SELECT 만, 주문 경로 없음, 출력은 ``data/algo_flow_76/`` 와
``research/logs/76-algo-flow/`` 뿐.

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/algo_flow_reversal.py replay
    uv run --extra pg python scripts/algo_flow_reversal.py features
    uv run --extra pg python scripts/algo_flow_reversal.py select
    uv run --extra pg python scripts/algo_flow_reversal.py oos
    uv run --extra pg python scripts/algo_flow_reversal.py describe
    uv run --extra pg python scripts/algo_flow_reversal.py report
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import random
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from krx_quant_core.stats import trials as trial_ledger
from krx_quant_core.stats.metrics import paired_date_bootstrap
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tick_replay_optimize as tro  # noqa: E402

from scalp_it.intraday.store import connect  # noqa: E402
from scalp_it.market_regime import evaluate_gate  # noqa: E402

ROOT = tro.ROOT
OUT = ROOT / "data" / "algo_flow_76"
LOGS_DIR = ROOT / "research" / "logs"
LABEL = "76-algo-flow"

# ---------------------------------------------------------------- 사전등록 상수
IS_DAYS, OOS_DAYS, ALL_DAYS = tro.IS_DAYS, tro.OOS_DAYS, tro.ALL_DAYS
OOS_WEEKS = [["2026-09-03", "2026-09-04"],
             ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"],
             ["2026-09-14"]]
CELL_A = tro.LIVE_CELL                               # (30, depth on, minsl 3, 1000, 5조)
CELL_B = (20.0, True, 0.0, "1000", 5.0)              # 주 판정 집합(B-필터)
CELL_B8 = (20.0, True, 0.0, "1000", 8.0)             # 만쥬 8조 관점(병기)
CELL_A8 = (30.0, True, 3.0, "1000", 8.0)
CELL_WIDE = (None, False, 0.0, None, None)           # B-넓음(선택용)
COMBOS = {"S": (3, 10), "L": (10, 30)}               # (W1, W2)
BASELINE_S = 120
REV_PRIOR_S = 30
MIN_REP_VOL = 10
REP_TOL = 0.05
P1_MIN_CNT = 3
P2_MIN_K = 4
P2_MIN_SECS = 3
P2_MAX_CV = 1.0
Q = 75
FEATURES = ["P1", "P2", "P12", "REV"]
ROLES = ["follower", "leader"]                       # 동률 순서: 후속 < 대장
N_PRIOR = 48 + 30
IS_MIN_N = 10
OOS_MIN_N = 30
N_BOOT = 10_000
SEED = 76
LOOKBACK_S = BASELINE_S + max(w1 for w1, _ in COMBOS.values())  # 130


def trials_list() -> list[tuple[str, str, str]]:
    return [(f, c, r) for f in FEATURES for c in COMBOS for r in ROLES]


def trial_name(t: tuple[str, str, str]) -> str:
    return f"{t[0]}·{t[1]}·{'후속' if t[2] == 'follower' else '대장'}"


# ================================================================ 1. 리플레이(B)
def b_setup_factory(orig: Any) -> Any:
    def b_setup(day: str) -> tuple[Any, Any, Any]:
        live_cfg, run_cfg, guard_cfg = orig(day)
        live_cfg = replace(live_cfg, stop_loss=None, min_sl_ticks=0.0, max_roundtrip_bp=20.0)
        run_cfg = replace(run_cfg, stop_loss=None)
        guard_cfg = replace(guard_cfg, scratch_ticks=0, scratch_secs=0.0,
                            profit_giveback_ratio=0.0)
        return live_cfg, run_cfg, guard_cfg
    return b_setup


def _use_b_cache() -> None:
    if not getattr(tro, "_b76", False):
        tro.live_setup = b_setup_factory(tro.live_setup)
        tro.OUT = OUT / "replay_B"
        tro._b76 = True


def load_b(day: str) -> dict[str, Any]:
    with gzip.open(OUT / "replay_B" / "stored" / f"{day}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_a(day: str) -> dict[str, Any]:
    with gzip.open(ROOT / "data" / "tick_replay_75" / "stored" / f"{day}.json.gz", "rt",
                   encoding="utf-8") as fh:
        return json.load(fh)


def cmd_replay(ns: argparse.Namespace) -> None:
    _use_b_cache()
    for d in (ns.days or ALL_DAYS):
        tro.replay_day(d, mode="stored", force=ns.force)


# ================================================================ 2. 특징
def _ticks(conn: Any, code: str, t: datetime) -> list[tuple]:
    """[t−LOOKBACK, t) 틱(ts, seq, price, volume, side) — 신호 초는 통째로 뺀다."""
    return tro._q(conn, "select ts, seq, price, volume, side from ticks where code = %s "
                        "and ts >= %s and ts < %s order by ts, seq",
                  (code, t - timedelta(seconds=LOOKBACK_S), t))


def raw_features(rows: list[tuple], t: datetime) -> dict[str, dict[str, Any]]:
    """창 조합별 원시값. rows 는 ts < t 만."""
    tt = t.timestamp()
    off = np.array([tt - tro._as_dt(r[0]).timestamp() for r in rows])   # 1..130 (초 전)
    px = np.array([float(r[2] or 0) for r in rows])
    vol = np.array([float(abs(r[3] or 0)) for r in rows])
    side = np.array([int(r[4] or 0) for r in rows])
    out: dict[str, dict[str, Any]] = {}
    for name, (w1, w2) in COMBOS.items():
        f: dict[str, Any] = {}
        # P1 ------------------------------------------------------------------
        last = off <= w1
        base = (off > w1) & (off <= w1 + BASELINE_S)
        base_rate = float(base.sum()) / BASELINE_S
        cnt = collections.Counter()
        net = collections.defaultdict(float)
        for o, p, v, s in zip(off[last], px[last], vol[last], side[last], strict=True):
            if s == 1:
                cnt[o] += 1
            net[o] += s * p * v
        if cnt:
            best = max(cnt.values())
            bsec = min(o for o, c in cnt.items() if c == best)   # 가장 늦은 초 = off 최소
            f.update(burst_cnt=int(best), burst_off=float(bsec), burst_net=float(net[bsec]))
        else:
            f.update(burst_cnt=0, burst_off=None, burst_net=0.0)
        f["base_rate"] = base_rate
        f["p1_ratio"] = f["burst_cnt"] / max(1.0, base_rate)
        # P2 ------------------------------------------------------------------
        win = (off <= w2) & (side == 1) & (vol >= MIN_REP_VOL)
        v2, o2 = vol[win], off[win]
        k, secs, cv, vref = 0, 0, math.inf, None
        if v2.size:
            for ref in sorted(set(v2.tolist()), reverse=True):
                m = np.abs(v2 - ref) <= max(1.0, REP_TOL * ref)
                if int(m.sum()) > k:
                    k, vref = int(m.sum()), ref
                    ss = np.unique(o2[m])
                    secs = int(ss.size)
                    if ss.size >= 3:
                        g = np.diff(np.sort(ss))
                        cv = float(g.std() / g.mean()) if g.mean() > 0 else math.inf
                    else:
                        cv = math.inf
        f.update(p2_count=k, p2_secs=secs, p2_gap_cv=cv, p2_vref=vref,
                 n_buy_rep=int(v2.size))
        # REV ------------------------------------------------------------------
        sv = side * vol
        f["net_recent"] = float(sv[off <= w2].sum())
        f["net_prior"] = float(sv[(off > w2) & (off <= w2 + REV_PRIOR_S)].sum())
        f["buy_notional"] = float((px * vol)[(off <= w2) & (side == 1)].sum())
        out[name] = f
    return out


def _rand_moment(day: str, key: str) -> datetime:
    rng = random.Random(f"{SEED}|{key}")
    start = datetime.fromisoformat(f"{day} 09:00:00")
    return start + timedelta(seconds=rng.randrange(0, 6 * 3600 + 20 * 60 + 1))


def feature_day(day: str, force: bool = False) -> Path:
    path = OUT / "features" / f"{day}.json.gz"
    if path.exists() and not force:
        print(f"[캐시] {path}")
        return path
    cache = load_b(day)
    conn = connect(os.environ.get("KR_QUANT_DB"))
    recs = []
    for i, c in enumerate(cache["candidates"]):
        t = datetime.fromisoformat(c["ts"])
        rec: dict[str, Any] = {"idx": i, "ts": c["ts"], "leader": c["leader"],
                               "follower": c["follower"]}
        for role in ("leader", "follower"):
            rec[role] = raw_features(_ticks(conn, c[role], t), t)
            key = f"{day}|{c['ts']}|{c['leader']}|{c['follower']}|{role}"
            rt = _rand_moment(day, key)
            rec[f"rand_{role}"] = raw_features(_ticks(conn, c[role], rt), rt)
            rec[f"rand_{role}_ts"] = rt.isoformat()
        rec["null"] = (raw_features(_ticks(conn, c["null_code"], t), t)
                       if c.get("null_code") else None)
        recs.append(rec)
    conn.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path.with_suffix(".tmp"), "wt", encoding="utf-8") as fh:
        json.dump({"day": day, "features": recs}, fh, default=str)
    path.with_suffix(".tmp").replace(path)
    print(f"[{day}] 특징 {len(recs)}건 → {path}", flush=True)
    return path


def load_feat(day: str) -> list[dict[str, Any]]:
    with gzip.open(OUT / "features" / f"{day}.json.gz", "rt", encoding="utf-8") as fh:
        recs = json.load(fh)["features"]
    # 캐시의 "leader"/"follower" 키는 특징 dict 로 덮였다 — 코드는 B 캐시에서 붙인다.
    for rec, c in zip(recs, load_b(day)["candidates"], strict=True):
        assert rec["ts"] == c["ts"]
        rec["codes"] = {"leader": c["leader"], "follower": c["follower"]}
    return recs


def cmd_features(ns: argparse.Namespace) -> None:
    for d in (ns.days or ALL_DAYS):
        feature_day(d, force=ns.force)


# ================================================================ 3. 문턱·플래그
def thresholds() -> dict[str, dict[str, float]]:
    """θ1·θ2 = IS 전 후보(필터 전) 분위수 — 수익률 열을 읽지 않는다."""
    th: dict[str, dict[str, float]] = {}
    for role in ROLES:
        for combo in COMBOS:
            p1, p2 = [], []
            for d in IS_DAYS:
                for rec in load_feat(d):
                    p1.append(rec[role][combo]["p1_ratio"])
                    p2.append(rec[role][combo]["p2_count"])
            th[f"{role}|{combo}"] = {"theta1": float(np.percentile(p1, Q)) if p1 else math.inf,
                                     "theta2": float(np.percentile(p2, Q)) if p2 else math.inf,
                                     "n_is_cands": len(p1)}
    return th


def flags(f: dict[str, Any], th: dict[str, float]) -> dict[str, bool]:
    p1 = (f["burst_cnt"] >= P1_MIN_CNT and f["p1_ratio"] >= th["theta1"]
          and f["burst_net"] > 0)
    cv = f["p2_gap_cv"]
    cv = math.inf if cv is None or cv == "inf" else float(cv)
    p2 = (f["p2_count"] >= max(P2_MIN_K, th["theta2"]) and f["p2_secs"] >= P2_MIN_SECS
          and cv <= P2_MAX_CV)
    rev = f["net_prior"] < 0 and f["net_recent"] > 0 and (p1 or p2)
    return {"P1": p1, "P2": p2, "P12": p1 and p2, "REV": rev}


def candidate_frame(days: list[str], cell: tuple, th: dict, *, source: str = "B") -> pd.DataFrame:
    """셀을 통과해 through 체결된 거래 + 16 시행 플래그."""
    rows = []
    for d in days:
        cache = load_b(d) if source == "B" else load_a(d)
        feats = load_feat(d) if source == "B" else None
        for i, c in enumerate(cache["candidates"]):
            if tro.stage(c, cell, cache["kosdaq_jo"]) != "trade":
                continue
            r = tro.trade_return(c)
            if r is None:
                continue
            row = {"day": d, "ts": c["ts"], "leader": c["leader"], "follower": c["follower"],
                   "ret": r, "null_ret": tro.trade_return(c, "through", "primary", key="null"),
                   "gross": tro.trade_return(c, cost="gross"),
                   "touch": tro.trade_return(c, "touch"), "reason": c["through"]["reason"],
                   "hold_s": _hold(c["through"])}
            if feats is not None:
                fr = feats[i]
                assert fr["ts"] == c["ts"] and fr["codes"]["follower"] == c["follower"]
                for f, combo, role in trials_list():
                    row[f"{f}|{combo}|{role}"] = flags(fr[role][combo],
                                                       th[f"{role}|{combo}"])[f]
            rows.append(row)
    return pd.DataFrame(rows)


def _hold(p: dict) -> float | None:
    if not p.get("fill_ts") or not p.get("exit_ts"):
        return None
    return (datetime.fromisoformat(p["exit_ts"]) - datetime.fromisoformat(p["fill_ts"])).total_seconds()


# ================================================================ 4. IS 선택
def cmd_select(ns: argparse.Namespace) -> None:
    th = thresholds()
    df = candidate_frame(IS_DAYS, CELL_WIDE, th)
    rows = []
    for t in trials_list():
        col = "|".join(t)
        on, off = df.loc[df[col], "ret"], df.loc[~df[col], "ret"]
        obj = (float(on.mean()) - float(off.mean())) if len(on) and len(off) else float("nan")
        cfg = {"feature": t[0], "combo": t[1], "W1W2": list(COMBOS[t[1]]), "role": t[2],
               "q": Q, "theta": th[f"{t[2]}|{t[1]}"], "selection_set": "B-wide IS",
               "config": "B manju-pure", "prereg": "docs/research/manju/76"}
        trial_ledger.record_trial(LABEL, cfg, logs_dir=LOGS_DIR, note="IS trial (prereg 76 §6)")
        elig = len(on) >= IS_MIN_N and len(off) >= IS_MIN_N and obj == obj and obj > 0
        rows.append({"trial": list(t), "name": trial_name(t), "n_on": len(on), "n_off": len(off),
                     "mean_on": float(on.mean()) if len(on) else None,
                     "mean_off": float(off.mean()) if len(off) else None,
                     "obj": obj, "eligible": elig})
    el = [r for r in rows if r["eligible"]]
    sel = None
    if el:
        best = max(round(r["obj"], 6) for r in el)
        sel = next(r for r in rows if r["eligible"] and round(r["obj"], 6) == best)
    out = {"thresholds": th, "n_is_trades": len(df), "rows": rows,
           "selected": sel["trial"] if sel else None,
           "n_trials_ledger": trial_ledger.count_trials(LABEL, logs_dir=LOGS_DIR)}
    (OUT / "select_is.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                   default=str), encoding="utf-8")
    print(json.dumps(th, indent=1))
    print(f"IS B-넓음 거래 {len(df)} · 선택 {sel['name'] if sel else '없음'}")
    for r in rows:
        print(f"  {r['name']:<14} on={r['n_on']:>3} off={r['n_off']:>3} obj={r['obj']:+.4%} "
              f"{'자격' if r['eligible'] else ''}")


# ================================================================ 5. OOS
def _day_ci(df: pd.DataFrame, days: list[str]) -> tuple[float, float]:
    tro.SEED = SEED                      # 75번 부트스트랩 함수를 seed 76 으로
    tro.N_BOOT = N_BOOT
    return tro.day_cluster_ci(df, days)


def _paired(a: pd.Series, b: pd.Series) -> dict[str, Any] | None:
    """일평균 a − b, 날짜 짝 부트스트랩. a·b 는 day 인덱스 시리즈."""
    x = a.rename("mean_excess").rename_axis("date").reset_index()
    y = b.rename("mean_excess").rename_axis("date").reset_index()
    try:
        pb = paired_date_bootstrap(x, y, n_boot=N_BOOT, seed=SEED)
    except ValueError:
        return None
    return {**pb, "p": 1.0 - pb["p_greater"]}


def evaluate(df: pd.DataFrame, col: str) -> dict[str, Any]:
    on = df[df[col]] if len(df) else df
    off = df[~df[col]] if len(df) else df
    res: dict[str, Any] = {"n": len(on), "n_off": len(off), "n_all": len(df)}
    if len(off):
        res["off_mean"], res["off_median"] = float(off.ret.mean()), float(off.ret.median())
    if len(df):
        res["all_mean"], res["all_median"] = float(df.ret.mean()), float(df.ret.median())
    if not len(on):
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    R = on.ret.to_numpy()
    res.update(mean=float(R.mean()), median=float(np.median(R)), win=float((R > 0).mean()),
               ci=_day_ci(on, OOS_DAYS), days=int(on.day.nunique()))
    res["vs_off"] = (_paired(on.groupby("day").ret.mean(), off.groupby("day").ret.mean())
                     if len(off) else None)
    nd = on.dropna(subset=["null_ret"])
    res["vs_null"] = (_paired(nd.groupby("day").ret.mean(), nd.groupby("day").null_ret.mean())
                      if len(nd) else None)
    if res["vs_null"]:
        res["vs_null"].update(n_pairs=len(nd), strat=float(nd.ret.mean()),
                              null=float(nd.null_ret.mean()))
    res["dsr"] = deflated_sharpe_from_sample(R, len(trials_list()) + N_PRIOR)
    res["dsr96"] = deflated_sharpe_from_sample(R, len(trials_list()) + N_PRIOR + 2)
    res["dsr334"] = deflated_sharpe_from_sample(R, len(trials_list()) + N_PRIOR + 240)
    lowo = []
    for wk in OOS_WEEKS:
        rest = on[~on.day.isin(wk)]
        lowo.append(float(rest.ret.mean()) if len(rest) else None)
    res["lowo"] = lowo
    res["gross_mean"] = float(on.gross.mean())
    res["reasons"] = on.reason.value_counts().to_dict()
    pdsr = (res["dsr"] or {}).get("prob_deflated_sharpe") or 0.0
    conds = {
        "1 OOS n≥30": len(on) >= OOS_MIN_N,
        "2 평균>0·중앙>0": res["mean"] > 0 and res["median"] > 0,
        "3 일클러스터 CI하한>0": res["ci"][0] > 0,
        "4 ON>OFF p<0.05": bool(res["vs_off"]) and res["vs_off"]["p"] < 0.05,
        "5 무작위대조 p<0.05": bool(res["vs_null"]) and res["vs_null"]["p"] < 0.05,
        "6 DSR확률>0.95(N=94)": pdsr > 0.95,
        "7 주 빼기 음수 ≤1": sum(1 for m in lowo if m is not None and m < 0) <= 1,
    }
    res["conds"] = conds
    if not conds["1 OOS n≥30"]:
        res["verdict"] = "표본 부족 — 판정 불가"
    else:
        res["verdict"] = "채택(그림자 비교 단계로)" if all(conds.values()) else "기각"
    return res


def cmd_oos(ns: argparse.Namespace) -> None:
    sel = json.loads((OUT / "select_is.json").read_text(encoding="utf-8"))
    th = sel["thresholds"]
    out: dict[str, Any] = {"selected": sel["selected"]}
    if sel["selected"] is None:
        out["verdict"] = "기각(IS 에서 신호 없음)"
    else:
        col = "|".join(sel["selected"])
        for label, cell in (("B-필터", CELL_B), ("B-넓음", CELL_WIDE), ("B-필터·8조", CELL_B8)):
            df = candidate_frame(OOS_DAYS, cell, th)
            g = evaluate(df, col)
            g["trades"] = df[["day", "ts", "leader", "follower", "ret", "null_ret", "reason",
                              col]].to_dict("records") if len(df) else []
            out[label] = g
        out["verdict"] = out["B-필터"]["verdict"]
    (OUT / "oos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
    for k in ("B-필터", "B-넓음", "B-필터·8조"):
        if k in out:
            g = out[k]
            print(k, {x: g.get(x) for x in ("n", "n_off", "mean", "median", "ci", "verdict")})
            if g.get("conds"):
                print("  ", g["conds"], "vs_off p", (g["vs_off"] or {}).get("p"),
                      "vs_null p", (g["vs_null"] or {}).get("p"),
                      "dsr", g["dsr"].get("prob_deflated_sharpe"))


# ================================================================ 6. 서술
def _summ(df: pd.DataFrame) -> dict[str, Any]:
    if not len(df):
        return {"n": 0}
    return {"n": len(df), "mean": float(df.ret.mean()), "median": float(df.ret.median()),
            "win": float((df.ret > 0).mean()), "gross": float(df.gross.mean()),
            "touch": float(df.touch.dropna().mean()) if df.touch.notna().any() else None,
            "hold_med": float(df.hold_s.dropna().median()) if df.hold_s.notna().any() else None,
            "reasons": {k: [int(v), float(df.loc[df.reason == k, "ret"].mean())]
                        for k, v in df.reason.value_counts().items()}}


def _tick_seq(conn: Any, code: str, t: datetime, secs: int) -> list[dict[str, Any]]:
    rows = tro._q(conn, "select ts, seq, price, volume, side from ticks where code = %s "
                        "and ts >= %s and ts < %s order by ts, seq",
                  (code, t - timedelta(seconds=secs), t))
    by: dict[str, dict[str, Any]] = {}
    for ts, _seq, p, v, s in rows:
        k = tro._as_dt(ts).strftime("%H:%M:%S")
        e = by.setdefault(k, {"sec": k, "buy": 0, "sell": 0, "vols": [], "px": p})
        if s == 1:
            e["buy"] += 1
            e["vols"].append(f"+{abs(v)}")
        elif s == -1:
            e["sell"] += 1
            e["vols"].append(f"-{abs(v)}")
        e["px"] = p
    return list(by.values())


def cmd_describe(ns: argparse.Namespace) -> None:
    sel = json.loads((OUT / "select_is.json").read_text(encoding="utf-8"))
    th = sel["thresholds"]
    D: dict[str, Any] = {}
    # 국면 일수
    conn = connect(os.environ.get("KR_QUANT_DB"))
    reg = []
    for d in ALL_DAYS:
        prev_jo = load_b(d)["kosdaq_jo"]
        try:
            same = float(evaluate_gate(conn, date.fromisoformat(d), with_leader=False).turnover_jo)
        except ValueError:
            same = None
        reg.append({"day": d, "prev_jo": prev_jo, "same_jo": same})
    D["regime"] = reg
    # 후보 수 동일성(A vs B)
    D["cand_counts"] = {d: [len(load_a(d)["candidates"]), len(load_b(d)["candidates"])]
                        for d in ALL_DAYS}
    # B vs A
    views = {}
    for name, cell, src in (("A 라이브(5조)", CELL_A, "A"), ("B 만쥬원형(5조)", CELL_B, "B"),
                            ("A 라이브(8조)", CELL_A8, "A"), ("B 만쥬원형(8조)", CELL_B8, "B"),
                            ("B 청산·필터 전부 끔", CELL_WIDE, "B"),
                            ("A 청산·필터 전부 끔", CELL_WIDE, "A")):
        df = candidate_frame(ALL_DAYS, cell, th, source=src)
        views[name] = {"all": _summ(df), "IS": _summ(df[df.day.isin(IS_DAYS)] if len(df) else df),
                       "OOS": _summ(df[df.day.isin(OOS_DAYS)] if len(df) else df),
                       "per_day": df.groupby("day").size().to_dict() if len(df) else {}}
    D["views"] = views
    # 빈도: 후보 시점 vs 같은 종목 무작위 시점 vs 같은 시각 다른 종목
    prev = {}
    for f, combo, role in trials_list():
        k = f"{role}|{combo}"
        s = r = nn = 0
        tot = totn = 0
        for d in ALL_DAYS:
            for rec in load_feat(d):
                tot += 1
                s += flags(rec[role][combo], th[k])[f]
                r += flags(rec[f"rand_{role}"][combo], th[k])[f]
                if rec["null"] is not None:
                    totn += 1
                    nn += flags(rec["null"][combo], th[k])[f]
        prev["|".join((f, combo, role))] = {"n": tot, "signal": s / tot if tot else None,
                                            "rand_same_code": r / tot if tot else None,
                                            "n_null": totn, "null_code": nn / totn if totn else None}
    D["prevalence"] = prev
    # 원시값 분포(후보 vs 무작위)
    dist = {}
    for role in ROLES:
        for combo in COMBOS:
            vals = collections.defaultdict(list)
            for d in ALL_DAYS:
                for rec in load_feat(d):
                    for src, key in (("signal", role), ("rand", f"rand_{role}")):
                        vals[f"{src}_p1"].append(rec[key][combo]["burst_cnt"])
                        vals[f"{src}_p2"].append(rec[key][combo]["p2_count"])
            dist[f"{role}|{combo}"] = {k: [float(np.percentile(v, q)) for q in (50, 75, 90)]
                                       for k, v in vals.items()}
    D["dist"] = dist
    # 예시: P1·P2 각각 가장 강한 후보 3건(대장·후속 무관, S 조합 P1 / L 조합 P2)
    ex = {"P1": [], "P2": []}
    pool = []
    for d in ALL_DAYS:
        for rec in load_feat(d):
            for role in ROLES:
                pool.append((d, rec, role))
    p1s = sorted([x for x in pool if flags(x[1][x[2]]["S"], th[f"{x[2]}|S"])["P1"]],
                 key=lambda x: -x[1][x[2]]["S"]["p1_ratio"])
    p2s = sorted([x for x in pool if flags(x[1][x[2]]["L"], th[f"{x[2]}|L"])["P2"]],
                 key=lambda x: -x[1][x[2]]["L"]["p2_count"])

    def pick(lst: list, n: int = 3) -> list:
        seen, got = set(), []
        for d, rec, role in lst:
            code = rec["codes"][role]
            if code in seen:
                continue
            seen.add(code)
            got.append((d, rec, role))
            if len(got) == n:
                break
        return got

    for kind, lst, combo, secs in (("P1", pick(p1s), "S", 15), ("P2", pick(p2s), "L", 32)):
        for d, rec, role in lst:
            t = datetime.fromisoformat(rec["ts"])
            code = rec["codes"][role]
            ex[kind].append({"day": d, "ts": rec["ts"], "role": role, "code": code,
                             "leader": rec["codes"]["leader"],
                             "follower": rec["codes"]["follower"],
                             "feat": rec[role][combo], "seq": _tick_seq(conn, code, t, secs)})
    conn.close()
    D["examples"] = ex
    D["thresholds"] = th
    (OUT / "describe.json").write_text(json.dumps(D, ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    print(json.dumps({k: D[k] for k in ("regime", "cand_counts")}, ensure_ascii=False))
    for k, v in views.items():
        print(k, {s: {x: v[s].get(x) for x in ("n", "mean", "median")} for s in ("all", "OOS")})


# ================================================================ 7. 리포트
def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.2f}%"


def cmd_report(ns: argparse.Namespace) -> None:
    sel = json.loads((OUT / "select_is.json").read_text(encoding="utf-8"))
    oos = json.loads((OUT / "oos.json").read_text(encoding="utf-8"))
    D = json.loads((OUT / "describe.json").read_text(encoding="utf-8"))
    L: list[str] = []
    reg = D["regime"]
    L += ["### 국면 — 16일 코스닥 거래대금", "",
          "| 날짜 | 구분 | 전일(게이트 기준) | 당일 |", "|---|---|--:|--:|"]
    for r in reg:
        L.append(f"| {r['day']} | {'IS' if r['day'] in IS_DAYS else 'OOS'} | {r['prev_jo']:.2f}조 | "
                 f"{'—' if r['same_jo'] is None else f'{r['same_jo']:.2f}조'} |")
    n8p = sum(r["prev_jo"] >= 8 for r in reg)
    n8s = sum(1 for r in reg if r["same_jo"] is not None and r["same_jo"] >= 8)
    n5p = sum(r["prev_jo"] >= 5 for r in reg)
    L += ["", f"전일 ≥8조 **{n8p}/16일** · 당일 ≥8조 **{n8s}/16일** · 전일 ≥5조 {n5p}/16일", ""]
    cc = D["cand_counts"]
    same = all(a == b for a, b in cc.values())
    L += [f"후보 수 A 캐시 = B 리플레이: {'전 일자 동일' if same else cc}", ""]
    L += ["### B(만쥬 원형) vs A(라이브) — 16일 서술", "",
          "| 관점 | n | 평균 | 중앙 | 승률 | 세전 | touch | 보유 중앙(초) | OOS n | OOS 평균 | OOS 중앙 |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, v in D["views"].items():
        a, o = v["all"], v["OOS"]
        if not a["n"]:
            L.append(f"| {k} | 0 | | | | | | | 0 | | |")
            continue
        L.append(f"| {k} | {a['n']} | {_pct(a['mean'])} | {_pct(a['median'])} | {a['win'] * 100:.0f}% | "
                 f"{_pct(a['gross'])} | {_pct(a['touch'])} | {a['hold_med'] or 0:.0f} | {o['n']} | "
                 f"{_pct(o.get('mean'))} | {_pct(o.get('median'))} |")
    L += ["", "청산 사유(16일, 건수·평균 순수익):", ""]
    for k, v in D["views"].items():
        if v["all"]["n"]:
            L.append(f"- {k}: " + " · ".join(f"{r} {c}건({_pct(m)})" for r, (c, m)
                                             in v["all"]["reasons"].items()))
    L += ["", "### 문턱(IS 전 후보 75% 분위수, 수익률 미사용)", "",
          "| 역할·조합 | θ1 (p1_ratio) | θ2 (p2_count) | P1 실효 | P2 실효 |", "|---|--:|--:|---|---|"]
    for k, v in sel["thresholds"].items():
        L.append(f"| {k} | {v['theta1']:.2f} | {v['theta2']:.1f} | cnt≥3 ∧ ratio≥{v['theta1']:.2f} ∧ net>0 | "
                 f"k≥{max(P2_MIN_K, v['theta2']):.1f} ∧ 초≥3 ∧ CV≤1 |")
    L += ["", f"### IS 선택 — B-넓음 IS {sel['n_is_trades']}건, 16 시행", "",
          "| 시행 | ON n | OFF n | ON 평균 | OFF 평균 | 목적(ON−OFF) | 자격 |", "|---|--:|--:|--:|--:|--:|---|"]
    for r in sel["rows"]:
        L.append(f"| {r['name']} | {r['n_on']} | {r['n_off']} | {_pct(r['mean_on'])} | {_pct(r['mean_off'])} | "
                 f"{_pct(r['obj'])} | {'예' if r['eligible'] else '아니오'} |")
    L += ["", f"선택: **{trial_name(tuple(sel['selected'])) if sel['selected'] else '없음'}** · 원장 시행 수 "
              f"{sel['n_trials_ledger']}", ""]
    if sel["selected"]:
        L += ["### OOS — 선택 특징 한 번 적용", "",
              "| 집합 | ON n | 평균 | 중앙 | 승률 | 일클러스터 CI | OFF n | OFF 평균 | ON−OFF | p(ON≤OFF) | 대조군 평균 | p(≤대조) | DSR(94) | DSR(96) | DSR(334) | 판정 |",
              "|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
        for k in ("B-필터", "B-넓음", "B-필터·8조"):
            g = oos[k]
            if not g["n"]:
                L.append(f"| {k} | 0 | | | | | {g['n_off']} | {_pct(g.get('off_mean'))} | | | | | | | | {g['verdict']} |")
                continue
            vo, vn = g["vs_off"] or {}, g["vs_null"] or {}
            L.append(
                f"| {k}{' (주)' if k == 'B-필터' else ' (보조)'} | {g['n']} | {_pct(g['mean'])} | {_pct(g['median'])} | "
                f"{g['win'] * 100:.0f}% | [{_pct(g['ci'][0])}, {_pct(g['ci'][1])}] | {g['n_off']} | "
                f"{_pct(g.get('off_mean'))} | {_pct(vo.get('mean_diff'))} | "
                f"{'—' if not vo else f'{vo['p']:.3f}'} | {_pct(vn.get('null'))} | "
                f"{'—' if not vn else f'{vn['p']:.3f}'} | {g['dsr']['prob_deflated_sharpe']:.4f} | "
                f"{g['dsr96']['prob_deflated_sharpe']:.4f} | {g['dsr334']['prob_deflated_sharpe']:.4f} | "
                f"**{g['verdict']}{'' if k == 'B-필터' else '(참고)'}** |")
        g = oos["B-필터"]
        if g.get("conds"):
            L += ["", "게이트(주 판정 B-필터 ON):", ""]
            L += [f"- {'✓' if ok else '✗'} {c}" for c, ok in g["conds"].items()]
            L += ["", "주 빼기 평균(09-03주 / 09-07주 / 09-14주 제외): "
                  + " · ".join(_pct(x) for x in g["lowo"])]
        for k in ("B-필터", "B-넓음"):
            if oos[k].get("reasons"):
                L.append(f"\n{k} ON 청산 사유: " + " · ".join(f"{a} {b}" for a, b in oos[k]["reasons"].items()))
    L += ["", "### 패턴 빈도 — 후보 시점 vs 무작위 시점 (16일 전 후보, 서술)", "",
          "| 특징·조합·역할 | 후보 n | 후보 시점 ON | 같은 종목 무작위 초 ON | 같은 시각 다른 종목 ON |",
          "|---|--:|--:|--:|--:|"]
    for t in trials_list():
        p = D["prevalence"]["|".join(t)]
        L.append(f"| {trial_name(t)} | {p['n']} | {p['signal'] * 100:.1f}% | {p['rand_same_code'] * 100:.1f}% | "
                 f"{'—' if p['null_code'] is None else f'{p['null_code'] * 100:.1f}%'} |")
    L += ["", "원시값 분위(50/75/90) — burst_cnt(한 초 최대 매수 체결 수), p2_count(비슷한 수량 반복 건수):", "",
          "| 역할·조합 | 후보 burst | 무작위 burst | 후보 p2 | 무작위 p2 |", "|---|---|---|---|---|"]
    for k, v in D["dist"].items():
        f = lambda a: "/".join(f"{x:.0f}" for x in a)  # noqa: E731
        L.append(f"| {k} | {f(v['signal_p1'])} | {f(v['rand_p1'])} | {f(v['signal_p2'])} | {f(v['rand_p2'])} |")
    L += ["", "### 예시 — 신호 직전 초별 체결(+매수 공격 / −매도 공격, 숫자=체결 수량 주)", ""]
    for kind, lst in D["examples"].items():
        for e in lst:
            ft = e["feat"]
            head = (f"**{kind}** {e['day']} {e['ts'][11:19]} · {'대장' if e['role'] == 'leader' else '후속'} "
                    f"{e['code']} (대장 {e['leader']} → 후속 {e['follower']}) · ")
            if kind == "P1":
                head += (f"한 초 최대 매수 {ft['burst_cnt']}건, 기준 {ft['base_rate']:.2f}건/초, "
                         f"ratio {ft['p1_ratio']:.1f}, 그 초 순체결대금 {ft['burst_net'] / 1e6:+.1f}백만원")
            else:
                head += (f"수량 ≈{ft['p2_vref']:.0f}주 반복 {ft['p2_count']}건, {ft['p2_secs']}개 초, "
                         f"간격 CV {float(ft['p2_gap_cv']):.2f}")
            L += [head, "", "```"]
            for s in e["seq"]:
                vols = " ".join(s["vols"])
                if len(vols) > 150:
                    vols = vols[:150] + " …"
                L.append(f"{s['sec']} 매수{s['buy']:>3} 매도{s['sell']:>3} @{s['px']:>7,}  {vols}")
            L += ["```", ""]
    text = "\n".join(L)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "features"):
        s = sub.add_parser(name)
        s.add_argument("days", nargs="*")
        s.add_argument("--force", action="store_true")
    for name in ("select", "oos", "describe", "report"):
        sub.add_parser(name)
    ns = p.parse_args(argv)
    {"replay": cmd_replay, "features": cmd_features, "select": cmd_select, "oos": cmd_oos,
     "describe": cmd_describe, "report": cmd_report}[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
