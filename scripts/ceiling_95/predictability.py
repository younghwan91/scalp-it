"""96번 예측가능성 탐침 — **분 단위 지정가 결과가 사전에 예측 가능한가**를 잰다. 판정이 아니라 기술이다.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/predictability.py

왜: 95번에서 오라클 천장을 재 보니 10~60초 보유는 완벽한 예지력으로도 음수였고, 5분 이상 +
지정가 진입에서만 천장이 열렸다. 천장이 열린다는 건 "먹을 게 남아 있다"는 뜻일 뿐, 그걸
**미리 고를 수 있다**는 뜻이 아니다. 그래서 사후 최적(오라클)을 버리고 **고정 규칙 두 가지**로
라벨을 박은 뒤, 기존 피처만으로 그 라벨을 맞힐 수 있는지 시간 분할로 본다.

- 라벨 A(고정 보유): 진입 bid[t+1] → t+1+300 초의 bid 로 청산
- 라벨 B(브래킷): 진입 bid[t+1] → [t+2, t+1+600] 안에서 ±50bp 먼저 닿는 쪽, 안 닿으면 창 끝 bid
- 둘 다 비용 23bp 를 **한 번** 뺀다. 체결은 낙관(지정가 100% 체결) 가정 — 95번 메이커 낙관과 같다.

미래를 보는 건 라벨뿐이다. 피처는 캐시에 이미 있는 결정 시점 값만 쓰고, 캐시에 들어 있는
기존 라벨(y_F*/hold_F*)과 식별자(day/code/row)는 입력에서 뺀다. 학습/검증은 날짜로 가른다.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0023
HOLD_A = 300                       # 규칙 A 고정 보유(초)
HORIZON_B = 600                    # 규칙 B 최대 보유(초)
TP = SL = 0.0050                   # ±50bp — 비용(23bp)의 두 배쯤, 스프레드에 묻히지 않는 최소폭
CACHE = Path.home() / "of80" / "flow83"
TRAIN_DAYS = ("2026-08-24", "2026-09-10")          # 양끝 포함
VALID_DAYS = ("2026-09-11", "2026-09-14", "2026-09-15")
DROP_PREFIX = ("y_F", "hold_F")                    # 캐시에 들어 있는 기존 라벨 — 입력이 되면 누수다
DROP_COLS = ("day", "code", "row")                 # 식별자·인덱스(시각은 sec 로 따로 있다)


def feature_columns(cols) -> list[str]:
    """입력으로 쓸 컬럼만 남긴다 — 라벨과 식별자를 지운다."""
    return [c for c in cols if c not in DROP_COLS and not c.startswith(DROP_PREFIX)]


def label_fixed(bid: np.ndarray, rows: np.ndarray, hold: int = HOLD_A) -> np.ndarray:
    """규칙 A — bid[t+1] 에 사서 bid[t+1+hold] 에 판다. 순수익(비율). 못 보면 NaN."""
    n = len(bid)
    out = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        t = rows[i]
        if t + 1 + hold >= n:
            continue
        p = bid[t + 1]
        q = bid[t + 1 + hold]
        if p > 0 and q > 0:
            out[i] = q / p - 1.0 - COST
    return out


def label_bracket(bid: np.ndarray, rows: np.ndarray, tp: float = TP, sl: float = SL,
                  horizon: int = HORIZON_B) -> np.ndarray:
    """규칙 B — bid[t+1] 에 사서 [t+2, t+1+horizon] 에서 먼저 닿는 쪽으로 나간다.

    한 초에 익절선과 손절선을 동시에 닿을 수는 없으므로(초당 bid 하나) 우선순위는 **시간순**이다.
    창 안에서 아무 데도 안 닿으면 창 끝 bid 로 청산한다. 비용은 어느 경우든 한 번만 뺀다.
    """
    n = len(bid)
    out = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        t = rows[i]
        end = t + 1 + horizon
        if end >= n:
            continue
        p = bid[t + 1]
        if not (p > 0):
            continue
        up = p * (1.0 + tp)
        dn = p * (1.0 - sl)
        r = np.nan
        for s in range(t + 2, end + 1):
            b = bid[s]
            if not (b > 0):
                continue
            if b >= up:
                r = tp
                break
            if b <= dn:
                r = -sl
                break
        if np.isnan(r):
            q = bid[end]
            if not (q > 0):
                continue
            r = q / p - 1.0
        out[i] = r - COST
    return out


try:                                   # 초 단위 루프 — 파이썬으로는 하루치도 못 돈다
    import numba

    label_fixed = numba.njit(cache=True)(label_fixed)
    label_bracket = numba.njit(cache=True)(label_bracket)
except Exception:                      # noqa: BLE001
    pass


def load_day(path: str) -> pd.DataFrame:
    """하루치 캐시 → 피처 + 라벨 A/B 가 붙은 DataFrame."""
    d = pickle.load(open(path, "rb"))
    rows = d["rows"]
    ya = np.full(len(rows), np.nan)
    yb = np.full(len(rows), np.nan)
    code_arr = rows["code"].to_numpy()
    row_arr = rows["row"].to_numpy().astype(np.int64)
    for code, p in d["paths"].items():
        m = np.nonzero(code_arr == code)[0]
        if not len(m):
            continue
        bid = p["bid"].astype(np.float64)
        ya[m] = label_fixed(bid, row_arr[m], HOLD_A)
        yb[m] = label_bracket(bid, row_arr[m], TP, SL, HORIZON_B)
    out = rows.copy()
    out["lab_a"] = ya
    out["lab_b"] = yb
    return out


def top_slice(pred: np.ndarray, actual: np.ndarray, q: float) -> dict:
    """예측 상위 q 분위의 실제 수익(bp) — 여기서 갈리지 않으면 예측은 쓸모가 없다."""
    thr = np.quantile(pred, 1.0 - q)
    sel = pred >= thr
    a = actual[sel] * 1e4
    return {"frac": float(sel.mean()), "n": int(sel.sum()),
            "mean_bp": float(a.mean()), "median_bp": float(np.median(a)),
            "frac_pos": float((a > 0).mean())}


def fit_eval(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], label: str) -> dict:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    xtr = tr[feats].to_numpy(np.float32)
    xva = va[feats].to_numpy(np.float32)
    ytr = tr[label].to_numpy(np.float64)
    yva = va[label].to_numpy(np.float64)
    model = lgb.train(
        {"objective": "regression", "metric": "l2", "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 1, "verbose": -1, "num_threads": 0, "seed": 7},
        lgb.Dataset(xtr, ytr, feature_name=feats), num_boost_round=400)
    pred = model.predict(xva)
    base_mean = float(yva.mean() * 1e4)
    rec = {
        "n_train": int(len(tr)), "n_valid": int(len(va)),
        "baseline_random_entry": {"mean_bp": base_mean, "median_bp": float(np.median(yva) * 1e4),
                                  "frac_pos": float((yva > 0).mean())},
        "spearman": float(spearmanr(pred, yva).statistic),
        "top": {}, "importance_gain_top15": [],
    }
    for q in (0.10, 0.05, 0.01):
        t = top_slice(pred, yva, q)
        t["lift_vs_all_bp"] = t["mean_bp"] - base_mean
        rec["top"][f"{q:.2f}"] = t
    gain = model.feature_importance("gain")
    order = np.argsort(gain)[::-1][:15]
    rec["importance_gain_top15"] = [{"feature": feats[i], "gain": float(gain[i])} for i in order]
    return rec


def main(argv: list[str] | None = None) -> int:
    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    if not files:
        print(f"캐시가 없다: {CACHE}")
        return 2
    keep = [f for f in files
            if TRAIN_DAYS[0] <= Path(f).stem <= TRAIN_DAYS[1] or Path(f).stem in VALID_DAYS]
    frames = []
    for f in keep:
        df = load_day(f)
        frames.append(df)
        print(f"  {Path(f).stem}  n={len(df):,}  라벨A 유효 {np.isfinite(df.lab_a).mean():.1%}")
    all_df = pd.concat(frames, ignore_index=True)
    feats = feature_columns([c for c in all_df.columns if c not in ("lab_a", "lab_b")])
    is_va = all_df.day.isin(VALID_DAYS).to_numpy()
    out = {"cost_bp": COST * 1e4, "hold_a_sec": HOLD_A, "horizon_b_sec": HORIZON_B,
           "tp_bp": TP * 1e4, "sl_bp": SL * 1e4, "n_features": len(feats),
           "train_days": [Path(f).stem for f in keep if Path(f).stem not in VALID_DAYS],
           "valid_days": list(VALID_DAYS), "labels": {}}
    for label in ("lab_a", "lab_b"):
        ok = np.isfinite(all_df[label].to_numpy())
        tr = all_df[ok & ~is_va]
        va = all_df[ok & is_va]
        rec = fit_eval(tr, va, feats, label)
        out["labels"][label] = rec
        name = "A(300초 고정)" if label == "lab_a" else "B(±50bp 브래킷·600초)"
        b = rec["baseline_random_entry"]
        print(f"\n■ 라벨 {name}  학습 {rec['n_train']:,} · 검증 {rec['n_valid']:,}")
        print(f"   무작위 진입 기준선: 평균 {b['mean_bp']:+7.1f}bp · 중앙 {b['median_bp']:+7.1f}bp · "
              f"양(+) {b['frac_pos']:.1%}")
        print(f"   스피어만(예측 vs 실현) {rec['spearman']:+.4f}")
        for q, t in rec["top"].items():
            print(f"   예측 상위 {float(q):.0%} ({t['n']:,}): 평균 {t['mean_bp']:+7.1f}bp "
                  f"(기준선 대비 {t['lift_vs_all_bp']:+6.1f}bp) · 중앙 {t['median_bp']:+7.1f}bp · "
                  f"양(+) {t['frac_pos']:.1%}")
        print("   중요도(gain) 상위 15: "
              + ", ".join(r["feature"] for r in rec["importance_gain_top15"]))
    Path("data").mkdir(exist_ok=True)
    Path("data/predictability96.json").write_text(json.dumps(out, indent=1))
    print("\n→ data/predictability96.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
