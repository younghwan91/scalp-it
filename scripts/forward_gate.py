"""83·84·85 전진 판정 공통 — 매일은 건수만, 조건이 차면 최종 평가 한 번(OOS 잠금, research/runs/*).

    # 매일 크론(수익 수치는 계산하지 않는다)
    FLOW_FORWARD=1 ~/of80/.rlvenv/bin/python scripts/forward_gate.py --study flow83
    # 전진 거래일 ≥ 10 · 거래 ≥ 30 이 되면 같은 명령이 스스로 final 평가를 한 번 돌리고 잠근다.

판정 게이트는 각 사전등록 §6(83번 §6 과 같다): 평균>0 · 날짜 부트스트랩 CI 하한>0 · 상위 1% 제거>0 ·
음수일 ≤ 1/3 · DSR>0.95 · 무작위 진입 대조 p<0.05 · 거래일≥10·거래≥30.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home() / "of80"
FORWARD_START = "2026-09-16"
SEED = 83
N_BOOT = 10_000
STUDIES = {
    "flow83": {"model": HOME / "flow83_model" / "model.pkl", "n_trials": 1303,
               "doc": "docs/research/manju/83-institutional-flow-turning-point-preregistration.md"},
    "flow84": {"model": HOME / "flow84_model" / "model.pkl", "n_trials": 1558,
               "doc": "docs/research/manju/84-feature-optimization-preregistration.md"},
    "tiger86": {"model": ROOT / "scripts" / "tiger_86" / "engine.py", "n_trials": 1619,
                "doc": "docs/research/manju/86-tiger-limit-queue-preregistration.md"},
    "rl87": {"model": HOME / "rl87" / "selected.json", "n_trials": 1679,
             "doc": "docs/research/manju/87-rl-limit-queue-preregistration.md"},
    "bandit88": {"model": HOME / "bandit88_run" / "state.pt", "n_trials": 1680,
                 "doc": "docs/research/manju/88-neural-bandit-preregistration.md"},
    "bandit89": {"model": HOME / "bandit89_run" / "state.pt", "n_trials": 1681,
                 "doc": "docs/research/manju/89-two-head-limit-bandit-preregistration.md"},
    "bandit91": {"model": HOME / "bandit91_run" / "state.pt", "n_trials": 2094,
                 "doc": "docs/research/manju/91-limit-bandit-30feat-preregistration.md"},
    "rl85": {"model": HOME / "rl85" / "selected.json", "n_trials": 1618,
             "doc": "docs/research/manju/85-rl-wave-riding-preregistration.md"},
}
MIN_DAYS, MIN_TRADES = 10, 30


def forward_days(conn) -> list[str]:
    cur = conn.cursor()
    cur.execute("select distinct ts::date from ticks where ts >= %s order by 1", (FORWARD_START,))
    now = dt.datetime.now()
    return [r[0].isoformat() for r in cur.fetchall() if r[0] < now.date() or now.hour >= 16]


def build_caches(conn, days: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "flow_83"))
    sys.path.insert(0, str(ROOT / "scripts" / "flow_84"))
    import features as F83
    import library as L84

    os.environ["FLOW84_FORWARD"] = "1"
    os.environ["FLOW87_FORWARD"] = "1"
    import importlib.util
    spec = importlib.util.spec_from_file_location("fills87", ROOT / "scripts" / "rl_87" / "fills.py")
    fills87 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fills87)
    os.environ["FLOW88_FORWARD"] = "1"
    spec88 = importlib.util.spec_from_file_location("rewards88", ROOT / "scripts" / "bandit_88" / "rewards.py")
    rewards88 = importlib.util.module_from_spec(spec88)
    spec88.loader.exec_module(rewards88)
    for d in days:
        F83.build_day(conn, d, forward_ok=True)
        L84.build_day(d)
        fills87.build(d)
        rewards88.build(d)


def trades_for(study: str, days: list[str]) -> pd.DataFrame:
    """정책 거래 목록(day, code, row/sec, hold, ret) + 무작위 대조 ret_ctrl."""
    rng = np.random.default_rng(SEED)
    if study in ("flow83", "flow84"):
        import lightgbm as lgb
        sys.path.insert(0, str(ROOT / "scripts" / "flow_83"))
        from train import policy

        m = pickle.loads(STUDIES[study]["model"].read_bytes())
        cache = HOME / ("flow83" if study == "flow83" else "flow84")
        rows = []
        for d in days:
            obj = pickle.loads((cache / f"{d}.pkl").read_bytes())
            rows.append(obj["rows"] if isinstance(obj, dict) else obj)
        rows = pd.concat(rows, ignore_index=True)
        pred = lgb.Booster(model_str=m["booster"]).predict(rows[m["features"]].to_numpy(np.float32))
        t = policy(rows, pred, m["thr_bp"], m["struct"])
        by = {k: g[f"y_{m['struct']}"].dropna().to_numpy() for k, g in rows.groupby(["day", "code"])}
        t["ret_ctrl"] = [float(rng.choice(by[(r.day, r.code)])) if len(by.get((r.day, r.code), [])) else np.nan
                         for r in t.itertuples()]
        return t
    if study == "tiger86":
        import psycopg2
        sys.path.insert(0, str(ROOT / "scripts" / "tiger_86"))
        import engine as T

        conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
        cur = conn.cursor()
        cur.execute("select max(ts::date) from ticks where ts < %s", (FORWARD_START,))
        warm_day = cur.fetchone()[0].isoformat()
        _, prev_q = T.simulate_day(conn, warm_day, None)          # 첫 전진일 09:35 전 문턱용 전일 분위
        out = []
        for d in days:
            t, prev_q = T.simulate_day(conn, d, prev_q, forward_ok=True, control_seed=SEED)
            if len(t):
                out.append(t[t.filled])
        conn.close()
        t = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["day", "code", "row", "ret", "hold"])
        t["ret_ctrl"] = t.get("ctrl_day_code_mean", np.nan)   # 같은 종목·날 무작위 진입의 체결 거래 평균
        return t
    if study == "bandit88":
        import importlib.util
        spec = importlib.util.spec_from_file_location("bandit88_mod", ROOT / "scripts" / "bandit_88" / "bandit.py")
        B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(B)
        B.OUT = HOME / "bandit88_forward"          # 학습 상태(17일 선행 끝)에서 이어서 온라인 학습
        t = B.run(days, forward_ok=True, state_in=HOME / "bandit88_run" / "state.pt")
        t = t[t.filled].rename(columns={"ret_bp": "ret"})
        t["ret"] = t["ret"] / 1e4
        ctrl = []
        for r in t.itertuples():                  # §5: 같은 종목·날, 같은 팔, 무작위 결정 행
            df = pickle.loads((HOME / "bandit88" / f"{r.day}.pkl").read_bytes())
            g = df[df.code == r.code]
            pick = g.sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]
            y = pick.y1 if r.arm == 1 else (pick.y2 if pick.filled2 else np.nan)
            ctrl.append(y / 1e4)
        t["ret_ctrl"] = ctrl
        return t
    if study in ("bandit89", "bandit91"):
        import importlib.util
        if study == "bandit89":
            spec = importlib.util.spec_from_file_location("bandit89_mod", ROOT / "scripts" / "bandit_89" / "bandit.py")
            B = importlib.util.module_from_spec(spec)
            sys.modules["bandit89_mod"] = B
            spec.loader.exec_module(B)
            B.OUT = HOME / "bandit89_forward"
            t = B.run(days, forward_ok=True, state_in=HOME / "bandit89_run" / "state.pt")
        else:
            spec = importlib.util.spec_from_file_location("bandit91_mod", ROOT / "scripts" / "bandit_91" / "bandit.py")
            B = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B)
            t = B.run(days, forward_ok=True, state_in=HOME / "bandit91_run" / "state.pt",
                      out=HOME / "bandit91_forward")
        t = t[t.filled].rename(columns={"ret_bp": "ret"})
        t["ret"] = t["ret"] / 1e4
        ctrl = []
        for r in t.itertuples():                  # §5: 같은 종목·날 무작위 결정 행, 같은 지정가 모형(체결된 것만)
            df = pickle.loads((HOME / "bandit88" / f"{r.day}.pkl").read_bytes())
            g = df[(df.code == r.code) & df.filled2]
            ctrl.append(float(g.sample(1, random_state=int(rng.integers(0, 2**31))).y2.iloc[0]) / 1e4 if len(g) else np.nan)
        t["ret_ctrl"] = ctrl
        return t
    # rl85 · rl87
    import importlib.util

    import torch
    name = "rl_87" if study == "rl87" else "rl_85"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", ROOT / "scripts" / name / "dqn.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_mod"] = R
    spec.loader.exec_module(R)
    if study == "rl87":
        R.COST, R.Q = R.R85.COST, R.R85.Q

    sel = json.loads(STUDIES[study]["model"].read_text())
    # 학습 스크립트를 직접 실행해 저장한 prep 은 Scaler 가 __main__ 소속으로 피클돼 있다 — 여기서 이어 준다.
    scaler_cls = R.R85.Scaler if study == "rl87" else R.Scaler

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "Scaler":
                return scaler_cls
            return super().find_class(module, name)

    with open(R.OUT / f"seed{sel['seed']}" / "prep.pkl", "rb") as fh:
        prep = _Unpickler(fh).load()
    parts = R.load_days(days, prep["features"], forward_ok=True)
    bk = R.make_book(parts, prep["scaler"])
    q = R.Q(len(prep["features"]) + R.EXTRA)
    q.load_state_dict(torch.load(R.OUT / f"seed{sel['seed']}" / f"ckpt_{sel['steps'] // 1_000_000:02d}M.pt"))
    t = R.greedy(q, bk)
    key = {k: i for i, k in enumerate(bk.keys)}
    ctrl = []
    for r in t.itertuples():
        i = key[(r.day, r.code)]
        s, e = bk.start[i], bk.end[i]
        secs = bk.sec[s:e]
        cand = np.nonzero(secs + r.hold <= secs[-1])[0]
        if not len(cand):
            ctrl.append(np.nan)
            continue
        a = int(rng.choice(cand))
        b = min(int(np.searchsorted(secs, secs[a] + r.hold)), len(secs) - 1)
        if study == "rl87":   # §4: 무작위 초에 같은 대기열 체결 모형
            fs = bk.fill_sec[s + a]
            if fs < 0:
                ctrl.append(np.nan)
                continue
            a2 = int(np.searchsorted(secs, fs))
            b = min(int(np.searchsorted(secs, fs + r.hold)), len(secs) - 1)
            ctrl.append(float(bk.sell[s + b] / bk.P[s + a] - 1 - R.COST) if a2 < len(secs) else np.nan)
            continue
        ctrl.append(float(bk.sell[s + b] / bk.buy[s + a] - 1 - R.COST))
    t["ret_ctrl"] = ctrl
    return t


def gates(t: pd.DataFrame, n_trials: int) -> tuple[dict, str]:
    from krx_quant_core.stats.metrics import paired_date_bootstrap
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    R = t.ret.to_numpy()
    days = sorted(t.day.unique())
    rng = np.random.default_rng(SEED)
    by = {d: t.loc[t.day == d, "ret"].to_numpy() for d in days}
    means = [np.concatenate([by[days[k]] for k in rng.integers(0, len(days), len(days))]).mean()
             for _ in range(N_BOOT)]
    dm = t.groupby("day").ret.mean()
    c = t.dropna(subset=["ret_ctrl"])
    a = c.groupby("day").ret.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    b = c.groupby("day").ret_ctrl.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
    k = max(1, math.ceil(0.01 * len(R)))
    g = {"n": len(R), "days": len(days), "mean_bp": R.mean() * 1e4, "median_bp": float(np.median(R)) * 1e4,
         "ci_low_bp": float(np.percentile(means, 2.5)) * 1e4, "trim1_bp": float(np.sort(R)[:-k].mean()) * 1e4,
         "neg_days": f"{int((dm < 0).sum())}/{len(dm)}",
         "dsr_prob": float(deflated_sharpe_from_sample(R, n_trials)["prob_deflated_sharpe"]),
         "ctrl_mean_bp": float(c.ret_ctrl.mean()) * 1e4, "ctrl_p": 1 - pb["p_greater"]}
    checks = {"1 평균>0": g["mean_bp"] > 0, "2 CI하한>0": g["ci_low_bp"] > 0, "3 상위1%제거>0": g["trim1_bp"] > 0,
              "4 음수일≤1/3": int((dm < 0).sum()) <= len(dm) // 3, "5 DSR>0.95": g["dsr_prob"] > 0.95,
              "6 대조 p<0.05": g["ctrl_p"] < 0.05, "7 일≥10·n≥30": len(days) >= MIN_DAYS and len(R) >= MIN_TRADES}
    g["checks"] = checks
    return g, ("채택" if all(checks.values()) else "기각")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=sorted(STUDIES), required=True)
    a = ap.parse_args()
    import psycopg2

    spec = STUDIES[a.study]
    report = ROOT / "data" / f"{a.study}_forward.md"
    final_mark = ROOT / "research" / "runs" / a.study / "FINAL"
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    days = forward_days(conn)
    build_caches(conn, days)
    conn.close()
    lines = [f"# {a.study} 전진 — {dt.date.today()}", "", f"- 전진 거래일 {len(days)}일 (판정 조건 ≥{MIN_DAYS}일·≥{MIN_TRADES}거래)"]
    if final_mark.exists():
        lines += ["- **최종 판정 완료** — research/runs 기록 참조. 다시 평가하지 않는다."]
    elif not days or not spec["model"].exists():
        lines += ["- 전진 데이터 또는 모델 없음 — 대기"]
    else:
        # 건수는 정책 거래 수만 본다. 수익 수치는 최종 평가 전까지 계산·기록하지 않는다(OOS 잠금).
        t = trades_for(a.study, days)
        n = len(t)
        lines += [f"- 누적 정책 거래 {n}건 (수익 수치는 최종 평가 때 한 번만 계산)"]
        if len(days) >= MIN_DAYS and n >= MIN_TRADES:
            from krx_quant_core.runtime import DataSpec, start_run

            with start_run(a.study, {"study": a.study, "gate": "prereg §6"}, repo_root=ROOT,
                           data=DataSpec(FORWARD_START, days[-1], "forward-final"), final=True) as run:
                g, verdict = gates(t, spec["n_trials"])
                run.log_result({"verdict": verdict, **{k: v for k, v in g.items() if k != "checks"},
                                "checks": {k: bool(v) for k, v in g["checks"].items()}})
            lines += ["", "## 최종 판정", "", *[f"- {k}: {v}" for k, v in g.items() if k != "checks"],
                      "- 게이트: " + " · ".join(f"[{k}: {'✓' if v else '✗'}]" for k, v in g["checks"].items()),
                      "", f"**판정: {verdict}**"]
            t.to_csv(report.with_suffix(".final_trades.csv"), index=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
