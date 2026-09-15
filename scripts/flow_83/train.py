"""83번 학습 — 블록 교차검증으로 구조·문턱 선택, 17일 전체 재학습. 사전등록 §5.

    ~/of80/.rlvenv/bin/python scripts/flow_83/train.py
"""
from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CACHE, FEATURES, FORWARD_START, STRUCTS

OUT = Path.home() / "of80" / "flow83_model"
QUANTS = (0.99, 0.995, 0.999, 0.9995, 0.9999)
PARAMS = dict(objective="regression", num_leaves=63, learning_rate=0.03, min_data_in_leaf=1000,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              seed=83, verbose=-1, num_threads=16)
ROUNDS = 800
BLOCKS = [("2026-08-24", "2026-08-28"), ("2026-08-31", "2026-09-02"), ("2026-09-03", "2026-09-07"),
          ("2026-09-08", "2026-09-10"), ("2026-09-11", "2026-09-15")]


def load_rows() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(CACHE / "*.pkl"))):
        day = Path(f).name[:10]
        assert day < FORWARD_START, "학습이 전진 OOS 날짜를 읽으려 했다"
        parts.append(pickle.loads(Path(f).read_bytes())["rows"])
    return pd.concat(parts, ignore_index=True)


def policy(df: pd.DataFrame, pred: np.ndarray, thr: float, struct: str) -> pd.DataFrame:
    """예측 ≥ 문턱이면 진입. 종목당 포지션 1개 — 청산 초 다음부터 재진입."""
    d = df[["day", "code", "row", f"y_{struct}", f"hold_{struct}"]].assign(pred=pred)
    d = d[(d.pred >= thr) & d[f"y_{struct}"].notna()].sort_values(["day", "code", "row"])
    out = []
    for (day, code), g in d.groupby(["day", "code"], sort=False):
        free = -1
        for row, y, h in zip(g.row.to_numpy(), g[f"y_{struct}"].to_numpy(), g[f"hold_{struct}"].to_numpy(),
                             strict=True):
            if row < free:
                continue
            out.append((day, code, int(row), float(y), int(h)))
            free = row + 1 + h + 1
    return pd.DataFrame(out, columns=["day", "code", "row", "ret", "hold"])


def daymean(t: pd.DataFrame) -> float:
    return float(t.groupby("day").ret.mean().mean()) if len(t) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    days = sorted(df.day.unique())
    X = df[list(FEATURES)].to_numpy(np.float32)
    log: dict = {"rows": len(df), "days": len(days), "cv": {}}
    score: dict[tuple[str, float], list[float]] = {}
    ntr: dict[tuple[str, float], int] = {}
    for s in STRUCTS:
        y = df[f"y_{s}"].to_numpy(np.float64) * 1e4
        for bi, (a, b) in enumerate(BLOCKS):
            test = (df.day >= a) & (df.day <= b)
            ti = days.index(min(d for d in days if d >= a))
            tj = days.index(max(d for d in days if d <= b))
            purge = set(days[max(0, ti - 1):ti] + days[tj + 1:tj + 2])
            train = ~test & ~df.day.isin(purge) & np.isfinite(y)
            bst = lgb.train(PARAMS, lgb.Dataset(X[train.to_numpy()], y[train.to_numpy()]), num_boost_round=ROUNDS)
            dt = df[test]
            pt = bst.predict(X[test.to_numpy()])
            for q in QUANTS:
                thr = float(np.quantile(pt, q))
                t = policy(dt, pt, thr, s)
                score.setdefault((s, q), []).append(daymean(t))
                ntr[(s, q)] = ntr.get((s, q), 0) + len(t)
                log["cv"].setdefault(f"{s}_q{q}", []).append(
                    {"block": bi, "n": len(t), "daymean_bp": daymean(t) * 1e4 if len(t) else None,
                     "corr": float(np.corrcoef(pt[np.isfinite(y[test.to_numpy()])],
                                               y[test.to_numpy()][np.isfinite(y[test.to_numpy()])])[0, 1])})
            print(s, bi, "done", flush=True)
    table = []
    for (s, q), v in score.items():
        vv = [x for x in v if x == x]
        table.append({"struct": s, "q": q, "cv_daymean_bp": float(np.mean(vv)) * 1e4 if vv else None,
                      "cv_trades": ntr[(s, q)], "pos_blocks": sum(1 for x in vv if x > 0), "blocks": len(vv)})
    log["table"] = sorted(table, key=lambda r: -(r["cv_daymean_bp"] if r["cv_daymean_bp"] is not None else -1e9))
    elig = [r for r in table if r["cv_trades"] >= 100 and r["cv_daymean_bp"] is not None]
    best = sorted(elig, key=lambda r: (-round(r["cv_daymean_bp"], 6), r["cv_trades"]))[0] if elig else None
    log["selected"] = best
    if best is not None:
        s, q = best["struct"], best["q"]
        y = df[f"y_{s}"].to_numpy(np.float64) * 1e4
        ok = np.isfinite(y)
        bst = lgb.train(PARAMS, lgb.Dataset(X[ok], y[ok]), num_boost_round=ROUNDS)
        pall = bst.predict(X)
        thr = float(np.quantile(pall, q))
        log["final"] = {"struct": s, "q": q, "thr_bp": thr}
        pickle.dump({"booster": bst.model_to_string(), "features": list(FEATURES), "struct": s, "q": q,
                     "thr_bp": thr}, open(OUT / "model.pkl", "wb"))
        imp = sorted(zip(FEATURES, bst.feature_importance("gain"), strict=True), key=lambda x: -x[1])[:15]
        log["top_features"] = [(k, round(float(v))) for k, v in imp]
    (OUT / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log[k] for k in ("table", "selected", "final", "top_features") if k in log},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
