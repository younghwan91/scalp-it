"""84번 선별·전진 선택·최종 재선택. 사전등록 §4.

    ~/of80/.rlvenv/bin/python scripts/flow_84/select.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flow_83"))
from train import BLOCKS, policy  # noqa: E402

from library import CACHE, candidates  # noqa: E402

OUT = Path.home() / "of80" / "flow84_model"
START = ["sec", "log_cumval", "spread_bp", "cum_rank", "n_tr10", "dist_lo300", "ret300",
         "same_size_val10", "sweep_ticks3", "dist_hi300"]           # 83번 중요도 상위 10
FAST = dict(objective="regression", num_leaves=31, learning_rate=0.05, min_data_in_leaf=2000,
            feature_fraction=0.8, seed=84, verbose=-1, num_threads=16)
FAST_ROUNDS = 300
FINAL = dict(objective="regression", num_leaves=63, learning_rate=0.03, min_data_in_leaf=1000,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
             seed=83, verbose=-1, num_threads=16)
FINAL_ROUNDS = 800
QUANTS = (0.99, 0.995, 0.999, 0.9995, 0.9999)
MAX_FEATS = 40
MIN_GAIN_BP = 0.5


def load() -> pd.DataFrame:
    fs = sorted(glob.glob(str(CACHE / "*.pkl")))
    assert all(Path(f).name[:10] < "2026-09-16" for f in fs)
    return pd.concat([pickle.loads(Path(f).read_bytes()) for f in fs], ignore_index=True)


def block_masks(df: pd.DataFrame):
    days = sorted(df.day.unique())
    for a, b in BLOCKS:
        test = ((df.day >= a) & (df.day <= b)).to_numpy()
        ti = days.index(min(d for d in days if d >= a))
        tj = days.index(max(d for d in days if d <= b))
        purge = set(days[max(0, ti - 1):ti] + days[tj + 1:tj + 2])
        yield test, ~test & ~df.day.isin(purge).to_numpy()


def daymean(t: pd.DataFrame) -> float:
    return float(t.groupby("day").ret.mean().mean()) if len(t) else float("nan")


def cv_score(df, feats, y, params, rounds, q, masks):
    X = df[feats].to_numpy(np.float32)
    ok = np.isfinite(y)
    vals, n = [], 0
    for test, train in masks:
        bst = lgb.train(params, lgb.Dataset(X[train & ok], y[train & ok]), num_boost_round=rounds)
        pt = bst.predict(X[test])
        t = policy(df[test].rename(columns={"y_F2": "y_F2", "hold_F2": "hold_F2"}), pt, float(np.quantile(pt, q)), "F2")
        n += len(t)
        vals.append(daymean(t))
    v = [x for x in vals if x == x]
    return (float(np.mean(v)) * 1e4 if v else float("nan")), n, [x * 1e4 for x in vals]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    masks = list(block_masks(df))
    y = df.y_F2.to_numpy(np.float64) * 1e4
    log: dict = {"rows": len(df)}

    # 1. 단일 선별
    cands = candidates(df)
    tm = df.t_mid10.to_numpy(np.float64)
    screen = []
    for c in cands:
        x = df[c].to_numpy(np.float64)
        rhos = []
        for test, _ in masks:
            m = test & np.isfinite(x) & np.isfinite(tm)
            if m.sum() < 1000 or np.nanstd(x[m]) == 0:
                rhos.append(np.nan)
                continue
            rhos.append(pd.Series(x[m]).rank().corr(pd.Series(tm[m]).rank()))
        r = np.array(rhos)
        same_sign = np.isfinite(r).all() and (np.all(r > 0) or np.all(r < 0))
        screen.append({"feat": c, "mean_rho": float(np.nanmean(r)), "rhos": [round(float(v), 4) for v in r],
                       "pass": bool(same_sign and abs(np.nanmean(r)) >= 0.01)})
    passed = sorted([s for s in screen if s["pass"]], key=lambda s: -abs(s["mean_rho"]))
    keep: list[str] = []
    sample = df.sample(min(len(df), 300_000), random_state=84)
    for s in passed:
        if all(abs(sample[s["feat"]].corr(sample[k], method="spearman")) <= 0.95 for k in keep):
            keep.append(s["feat"])
    log["screen_candidates"], log["screen_passed"], log["screen_kept"] = len(cands), len(passed), keep
    log["screen"] = screen
    print("선별:", len(cands), "→ 통과", len(passed), "→ 중복 제거", len(keep), flush=True)

    # 2. 전진 선택
    feats = list(START)
    base, n0, blk = cv_score(df, feats, y, FAST, FAST_ROUNDS, 0.999, masks)
    steps = [{"feat": None, "score_bp": base, "n": n0, "blocks": blk, "accepted": True}]
    print("시작", round(base, 2), n0, flush=True)
    tries = 0
    for c in keep:
        if c in feats or len(feats) >= MAX_FEATS:
            continue
        tries += 1
        sc, n, blk = cv_score(df, [*feats, c], y, FAST, FAST_ROUNDS, 0.999, masks)
        acc = sc == sc and sc >= base + MIN_GAIN_BP
        steps.append({"feat": c, "score_bp": sc, "n": n, "blocks": blk, "accepted": bool(acc)})
        print(("채택 " if acc else "기각 ") + c, round(sc, 2), n, flush=True)
        if acc:
            feats.append(c)
            base = sc
        (OUT / "select_log.json").write_text(json.dumps({**log, "steps": steps, "feats": feats},
                                                        ensure_ascii=False, indent=1))
    log.update({"steps": steps, "feats": feats, "forward_tries": tries})

    # 3. 최종 재선택(83번 §5 그대로, F2 구조 고정)
    table = []
    for q in QUANTS:
        sc, n, blk = cv_score(df, feats, y, FINAL, FINAL_ROUNDS, q, masks)
        table.append({"q": q, "cv_daymean_bp": sc, "cv_trades": n, "blocks": blk})
        print("최종 q", q, round(sc, 2), n, flush=True)
    elig = [r for r in table if r["cv_trades"] >= 100 and r["cv_daymean_bp"] == r["cv_daymean_bp"]]
    best = sorted(elig, key=lambda r: (-round(r["cv_daymean_bp"], 6), r["cv_trades"]))[0] if elig else None
    log["final_table"], log["selected"] = table, best
    if best:
        X = df[feats].to_numpy(np.float32)
        ok = np.isfinite(y)
        bst = lgb.train(FINAL, lgb.Dataset(X[ok], y[ok]), num_boost_round=FINAL_ROUNDS)
        thr = float(np.quantile(bst.predict(X), best["q"]))
        pickle.dump({"booster": bst.model_to_string(), "features": feats, "struct": "F2", "q": best["q"],
                     "thr_bp": thr}, open(OUT / "model.pkl", "wb"))
        log["final"] = {"q": best["q"], "thr_bp": thr, "n_feats": len(feats)}
    log["n_trials_added"] = len(cands) + tries + len(QUANTS)
    (OUT / "select_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log[k] for k in ("screen_passed", "feats", "final_table", "selected", "final",
                                          "n_trials_added") if k in log}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
