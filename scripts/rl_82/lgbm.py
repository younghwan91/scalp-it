"""82번 모델 A — LightGBM 지도학습 상한. 사전등록 §4-A.

    ~/of80/.rlvenv/bin/python scripts/rl_82/lgbm.py            # TRAIN 학습 + VAL 문턱 선택 → model_a.pkl
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import trade_ret
from dataset import Scaler, load

OUT = Path.home() / "of80" / "rl82"
QUANTS = (0.9, 0.95, 0.99, 0.995, 0.999)
COOL = 60
PARAMS = dict(objective="regression", num_leaves=31, learning_rate=0.05, min_data_in_leaf=500,
              feature_fraction=0.8, seed=82, verbose=-1, num_threads=16)


def policy_trades(eps, preds, thr: float) -> pd.DataFrame:
    """예측 ≥ 문턱이면 진입, 80번 §7 청산(라벨이 곧 그 거래의 순수익). 종목당 60초 쿨다운."""
    rows = []
    for ep, pr in zip(eps, preds, strict=True):
        free = -1
        for i in np.nonzero(pr >= thr)[0]:
            if ep.secs[i] < free or not np.isfinite(ep.label[i]):
                continue
            free = ep.secs[i] + COOL
            rows.append((ep.day, ep.code, int(ep.secs[i]), float(ep.label[i]), int(i)))
    return pd.DataFrame(rows, columns=["day", "code", "sec", "ret", "i"])


def daymean(t: pd.DataFrame) -> float:
    return float(t.groupby("day").ret.mean().mean()) if len(t) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = load("train"), load("val")
    sc = Scaler(tr)
    Xtr = np.concatenate([sc(e.X) for e in tr])
    ytr = np.concatenate([e.label for e in tr])
    m = np.isfinite(ytr)
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr[m], ytr[m] * 1e4), num_boost_round=500)
    ptr = booster.predict(Xtr[m])
    pv = [booster.predict(sc(e.X)) for e in va]
    allv = np.concatenate(pv)
    rep = {"train_rows": int(m.sum()), "train_corr": float(np.corrcoef(ptr, ytr[m])[0, 1])}
    yv = np.concatenate([e.label for e in va])
    mv = np.isfinite(yv)
    rep["val_corr"] = float(np.corrcoef(allv[mv], yv[mv])[0, 1])
    best = None
    rep["val"] = {}
    for q in QUANTS:
        thr = float(np.quantile(allv, q))
        t = policy_trades(va, pv, thr)
        dm = daymean(t)
        rep["val"][str(q)] = {"thr_bp": thr, "n": len(t), "daymean_bp": dm * 1e4,
                              "gross_bp": (t.ret.mean() + 0.0023) * 1e4 if len(t) else None}
        if len(t) >= 20 and (best is None or dm > best[1]):
            best = (q, dm, thr)
    rep["selected"] = None if best is None else {"q": best[0], "thr_bp": best[2]}
    pickle.dump({"booster": booster.model_to_string(), "scaler": sc, "selected": rep["selected"]},
                open(OUT / "model_a.pkl", "wb"))
    imp = dict(zip(booster.feature_name(), booster.feature_importance("gain").round(0).tolist(), strict=True))
    rep["importance"] = imp
    (OUT / "model_a_val.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
