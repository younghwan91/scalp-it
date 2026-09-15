"""83번 전진 OOS — 매일 크론. 09-16 이후 거래일만. 사전등록 §1·§6.

    FLOW83_FORWARD=1 ~/of80/.rlvenv/bin/python scripts/flow_83/forward.py
"""
from __future__ import annotations

import datetime as dt
import math
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

from features import CACHE, FORWARD_START, build_day
from train import policy

MODEL = Path.home() / "of80" / "flow83_model" / "model.pkl"
REPORT = Path.home() / "git" / "scalp-it" / "data" / "flow83_forward.md"
N_TRIALS = 1303
SEED = 83
N_BOOT = 10_000


def control(t: pd.DataFrame, rows: pd.DataFrame, struct: str) -> pd.Series:
    """같은 종목·날·구조, 진입 초만 무작위 — §6-6."""
    rng = np.random.default_rng(SEED)
    y = f"y_{struct}"
    by = {k: g[y].dropna().to_numpy() for k, g in rows.groupby(["day", "code"])}
    return pd.Series([float(rng.choice(by[(r.day, r.code)])) if len(by.get((r.day, r.code), [])) else np.nan
                      for r in t.itertuples()], index=t.index)


def gates(t: pd.DataFrame) -> tuple[dict, str]:
    from krx_quant_core.stats.metrics import paired_date_bootstrap
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    R = t.ret.to_numpy()
    n, days = len(R), sorted(t.day.unique())
    if n < 2:
        return {"n": n, "days": len(days)}, "표본 부족 — 판정 불가"
    rng = np.random.default_rng(SEED)
    by = {d: t.loc[t.day == d, "ret"].to_numpy() for d in days}
    means = [np.concatenate([by[days[k]] for k in rng.integers(0, len(days), len(days))]).mean()
             for _ in range(N_BOOT)]
    dm = t.groupby("day").ret.mean()
    c = t.dropna(subset=["ret_ctrl"])
    a = c.groupby("day").ret.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    b = c.groupby("day").ret_ctrl.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
    k = max(1, math.ceil(0.01 * n))
    g = {"n": n, "days": len(days), "mean_bp": R.mean() * 1e4, "median_bp": float(np.median(R)) * 1e4,
         "ci_low_bp": float(np.percentile(means, 2.5)) * 1e4, "trim1_bp": float(np.sort(R)[:-k].mean()) * 1e4,
         "neg_days": f"{int((dm < 0).sum())}/{len(dm)}",
         "dsr_prob": float(deflated_sharpe_from_sample(R, N_TRIALS)["prob_deflated_sharpe"]),
         "ctrl_mean_bp": float(c.ret_ctrl.mean()) * 1e4, "ctrl_p": 1 - pb["p_greater"],
         "hold_med_s": float(t.hold.median())}
    checks = {"1 평균>0": g["mean_bp"] > 0, "2 CI하한>0": g["ci_low_bp"] > 0, "3 상위1%제거>0": g["trim1_bp"] > 0,
              "4 음수일≤1/3": int((dm < 0).sum()) <= len(dm) // 3, "5 DSR>0.95": g["dsr_prob"] > 0.95,
              "6 대조 p<0.05": g["ctrl_p"] < 0.05, "7 일≥10·n≥30": len(days) >= 10 and n >= 30}
    g["checks"] = checks
    verdict = ("표본 부족 — 판정 불가" if not checks["7 일≥10·n≥30"]
               else "채택" if all(checks.values()) else "기각")
    return g, verdict


def main() -> None:
    m = pickle.loads(MODEL.read_bytes())
    bst = lgb.Booster(model_str=m["booster"])
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    cur = conn.cursor()
    cur.execute("select distinct ts::date from ticks where ts >= %s order by 1", (FORWARD_START,))
    today = dt.date.today()
    days = [r[0].isoformat() for r in cur.fetchall() if r[0] < today or dt.datetime.now().hour >= 16]
    for d in days:
        build_day(conn, d, forward_ok=True)
    conn.close()
    rows = pd.concat([pickle.loads((CACHE / f"{d}.pkl").read_bytes())["rows"] for d in days],
                     ignore_index=True) if days else pd.DataFrame()
    lines = [f"# 83번 전진 OOS — {today}", "",
             f"- 모델: 구조 {m['struct']} · 분위 {m['q']} · 문턱 {m['thr_bp']:.1f}bp",
             f"- 전진 거래일 {len(days)}일"]
    if len(rows):
        s = m["struct"]
        pred = bst.predict(rows[m["features"]].to_numpy(np.float32))
        t = policy(rows, pred, m["thr_bp"], s)
        t["ret_ctrl"] = control(t, rows, s)
        g, verdict = gates(t)
        lines += ["", "| 항목 | 값 |", "|---|---|"]
        lines += [f"| {k} | {v:.2f} |" if isinstance(v, float) else f"| {k} | {v} |"
                  for k, v in g.items() if k != "checks"]
        if "checks" in g:
            lines += ["", "게이트: " + " · ".join(f"[{k}: {'✓' if v else '✗'}]" for k, v in g["checks"].items())]
        lines += ["", f"**판정: {verdict}**", "", "## 일별", "",
                  t.groupby("day").ret.agg(["size", "mean"]).assign(mean=lambda x: x["mean"] * 1e4)
                  .round(1).to_markdown()]
        t.to_csv(REPORT.with_suffix(".trades.csv"), index=False)
    else:
        lines += ["", "**표본 부족 — 판정 불가** (전진 데이터 없음)"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
