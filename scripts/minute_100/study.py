"""102번 분봉 대규모 탐색 — 학습·매치드 널·판정. 사전등록 docs/research/manju/102 §1~§4.

    ~/of80/.rlvenv/bin/python scripts/minute_100/study.py --phase val      # 검증 구간만(최종은 안 연다)
    ~/of80/.rlvenv/bin/python scripts/minute_100/study.py --phase final    # 최종 구간, 한 번만(FINAL 잠금)

분할·대조·게이트는 전부 사전등록에서 고정됐다. 여기서 바꾸면 그건 탐색이다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _envload  # noqa: E402,F401

CACHE = Path.home() / "of80" / "min100"
OUT = Path.home() / "of80" / "min100_study"
TRAIN = ("2025-08-01", "2026-04-30")
VAL = ("2026-05-01", "2026-07-31")
FINAL = ("2026-08-01", "2026-09-16")
TOP_FRAC = 0.01                 # 판정 구간 — 사전등록 §4. 10%·5% 는 보고만.
MIN_CELL = 5
N_BOOT, SEED = 2000, 102
N_TRIALS = 2450                 # 83~98 누적 + 이 연구 1회 (§4-6)
MIN_DAYS, MIN_ROWS = 20, 500
NON_FEATS = ("y", "code", "ts", "date", "vol_q", "tv_q", "cell")


def bad_days() -> set[str]:
    f = ROOT / "research" / "bad_days.json"
    try:
        return set(json.loads(f.read_text(encoding="utf-8"))["days"])
    except (OSError, KeyError, ValueError):
        return set()


def split(days: list[str]) -> tuple[list[str], list[str], list[str]]:
    """날짜 분할 — 겹치지 않고, 최종은 사전등록 OOS 와 같다."""
    tr = [d for d in days if TRAIN[0] <= d <= TRAIN[1]]
    va = [d for d in days if VAL[0] <= d <= VAL[1]]
    fi = [d for d in days if FINAL[0] <= d <= FINAL[1]]
    assert not (set(tr) & set(va)) and not (set(va) & set(fi)) and not (set(tr) & set(fi))
    return tr, va, fi


def load(days: list[str]) -> pd.DataFrame:
    bad = bad_days()
    parts = []
    for d in days:
        if d in bad:
            continue
        f = CACHE / f"{d}.pkl"
        if f.exists():
            df = pickle.loads(f.read_bytes())
            df["date"] = d
            parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def prior_day_cells(conn, dates: list[str]) -> pd.DataFrame:
    """(date, code) → 전일 거래대금 4분위 · 전일 20일 실현변동성 4분위. **둘 다 개장 전 정보.**"""
    lo = (pd.Timestamp(min(dates)) - pd.Timedelta(days=45)).date().isoformat()
    b = pd.read_sql("select code, date, close, trade_value from daily_bars where date >= %s and date <= %s "
                    "order by code, date", conn, params=(lo, max(dates)))
    b["date"] = pd.to_datetime(b.date)
    g = b.groupby("code", sort=False)
    b["ret"] = g.close.pct_change()
    b["vol20"] = g.ret.transform(lambda s: s.rolling(20, min_periods=10).std())
    b["p_vol"] = g.vol20.shift(1)
    b["p_tv"] = g.trade_value.shift(1)
    b = b.dropna(subset=["p_vol", "p_tv"])
    b["date"] = b.date.dt.strftime("%Y-%m-%d")
    b = b[b.date.isin(set(dates))]
    b["vol_q"] = b.groupby("date").p_vol.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    b["tv_q"] = b.groupby("date").p_tv.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    return b[["date", "code", "vol_q", "tv_q"]]


def null_alpha(df: pd.DataFrame) -> pd.Series:
    """매치드 널 — 같은 셀(날짜×분×전일변동성4분위×전일거래대금4분위)의 **다른** 행 평균 대비 초과. §3."""
    key = ["date", "minute", "vol_q", "tv_q"]
    grp = df.groupby(key, sort=False).y
    n = grp.transform("size")
    s = grp.transform("sum")
    alpha = df.y - (s - df.y) / (n - 1).where(n > 1)
    return alpha.where(n >= MIN_CELL)


def feats_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATS]


def fit(train: pd.DataFrame, val: pd.DataFrame):
    import lightgbm as lgb

    f = feats_of(train)
    dtr = lgb.Dataset(train[f], train.y)
    dva = lgb.Dataset(val[f], val.y, reference=dtr)
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": SEED}
    m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, f


def top_slice(df: pd.DataFrame, pred: np.ndarray, frac: float) -> pd.DataFrame:
    k = max(1, math.ceil(frac * len(df)))
    idx = np.argsort(-pred)[:k]
    return df.iloc[idx].copy()


def gates(top: pd.DataFrame, n_trials: int) -> tuple[dict, str]:
    """사전등록 §4 — 상위 1% 행에 대해 7개 AND."""
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    R = top.y.to_numpy() / 1e4
    days = sorted(top.date.unique())
    rng = np.random.default_rng(SEED)
    by = {d: top.loc[top.date == d, "y"].to_numpy() / 1e4 for d in days}
    boot = [np.concatenate([by[days[k]] for k in rng.integers(0, len(days), len(days))]).mean()
            for _ in range(N_BOOT)]
    a = top.dropna(subset=["alpha"])
    ab = {d: a.loc[a.date == d, "alpha"].to_numpy() / 1e4 for d in sorted(a.date.unique())}
    adays = list(ab)
    aboot = ([np.concatenate([ab[adays[k]] for k in rng.integers(0, len(adays), len(adays))]).mean()
              for _ in range(N_BOOT)] if adays else [np.nan])
    dm = top.groupby("date").y.mean()
    k = max(1, math.ceil(0.01 * len(R)))
    g = {"n": int(len(R)), "days": len(days), "mean_bp": float(R.mean()) * 1e4,
         "median_bp": float(np.median(R)) * 1e4, "ci_low_bp": float(np.percentile(boot, 2.5)) * 1e4,
         "alpha_mean_bp": float(a.alpha.mean()) if len(a) else float("nan"),
         "alpha_ci_low_bp": float(np.percentile(aboot, 2.5)) * 1e4, "alpha_n": int(len(a)),
         "trim1_bp": float(np.sort(R)[:-k].mean()) * 1e4 if len(R) > k else float("nan"),
         "neg_days": f"{int((dm < 0).sum())}/{len(dm)}",
         "dsr_prob": float(deflated_sharpe_from_sample(R, n_trials)["prob_deflated_sharpe"]) if len(R) > 2 else 0.0}
    checks = {"1 평균>0": g["mean_bp"] > 0, "2 CI하한>0": g["ci_low_bp"] > 0,
              "3 널alpha CI하한>0": g["alpha_ci_low_bp"] > 0, "4 상위1%제거>0": g["trim1_bp"] > 0,
              "5 음수일≤1/3": int((dm < 0).sum()) <= len(dm) // 3, "6 DSR>0.95": g["dsr_prob"] > 0.95,
              "7 일≥20·행≥500": len(days) >= MIN_DAYS and len(R) >= MIN_ROWS}
    g["checks"] = checks
    return g, ("채택" if all(checks.values()) else "기각")


def report(df: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    out = {"label": label, "n": int(len(df)), "days": int(df.date.nunique()),
           "baseline_mean_bp": float(df.y.mean()), "baseline_frac_pos": float((df.y > 0).mean())}
    from scipy.stats import spearmanr

    out["spearman"] = float(spearmanr(pred, df.y).correlation)
    for frac, nm in ((0.10, "top10"), (0.05, "top5"), (0.01, "top1")):
        t = top_slice(df, pred, frac)
        out[nm] = {"n": int(len(t)), "mean_bp": float(t.y.mean()), "median_bp": float(t.y.median()),
                   "frac_pos": float((t.y > 0).mean()),
                   "alpha_mean_bp": float(t.alpha.mean()) if t.alpha.notna().any() else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "final"], required=True)
    a = ap.parse_args()
    import psycopg2

    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(p.stem for p in CACHE.glob("*.pkl"))
    tr_d, va_d, fi_d = split(days)
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    cells = prior_day_cells(conn, tr_d + va_d + fi_d)
    tr = load(tr_d).merge(cells, on=["date", "code"], how="inner")
    va = load(va_d).merge(cells, on=["date", "code"], how="inner")
    print(f"학습 {len(tr):,}행/{tr.date.nunique()}일 · 검증 {len(va):,}행/{va.date.nunique()}일", flush=True)
    model, feats = fit(tr, va)
    va["alpha"] = null_alpha(va)
    pv = model.predict(va[feats], num_iteration=model.best_iteration)
    rep_val = report(va, pv, "검증")
    imp = sorted(zip(feats, model.feature_importance("gain"), strict=True), key=lambda x: -x[1])
    rep_val["importance_top10"] = [(f, float(v)) for f, v in imp[:10]]
    rep_val["best_iteration"] = int(model.best_iteration)
    (OUT / "val.json").write_text(json.dumps(rep_val, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep_val.items() if k != "importance_top10"}, ensure_ascii=False, indent=1))
    print("중요도 상위:", rep_val["importance_top10"][:6])
    if a.phase == "val":
        print("\n최종 구간은 열지 않았다(--phase final 로 한 번만).")
        return 0

    from krx_quant_core.runtime import DataSpec, start_run
    from krx_quant_core.stats.trials import record_trial

    record_trial("102-minute-largescale", {"model": "lgbm", "top": TOP_FRAC, "hold": 30, "cost_bp": 41.5},
                 logs_dir=ROOT / "research" / "logs", note="사전등록 102 최종 판정 1회")
    with start_run("min100", {"prereg": "docs/research/manju/102", "phase": "final"}, repo_root=ROOT,
                   data=DataSpec(FINAL[0], FINAL[1], "final"), seed=SEED, final=True, allow_dirty=True) as run:
        fi = load(fi_d).merge(cells, on=["date", "code"], how="inner")
        fi["alpha"] = null_alpha(fi)
        pf = model.predict(fi[feats], num_iteration=model.best_iteration)
        rep_fin = report(fi, pf, "최종")
        top = top_slice(fi, pf, TOP_FRAC)
        g, verdict = gates(top, N_TRIALS)
        rep_fin["gates"] = g
        rep_fin["verdict"] = verdict
        (OUT / "final.json").write_text(json.dumps(rep_fin, ensure_ascii=False, indent=1, default=str))
        run.log_result({"verdict": verdict, "gates": {k: v for k, v in g.items() if k != "checks"},
                        "checks": g["checks"]})
        print(json.dumps(rep_fin, ensure_ascii=False, indent=1, default=str))
        print("\n판정:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
