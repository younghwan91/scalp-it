"""98번 심화 — 거래 단위 자료로 선택 규칙의 낙관을 분해한다. 판정이 아니라 기술이다.

    ~/of80/.rlvenv/bin/python scripts/selection_98/replay_val.py rl96 rl93 rl92   # 먼저 재생
    ~/of80/.rlvenv/bin/python scripts/selection_98/deep.py rl96 rl93 rl92

`selection_bias.py` 가 로그만으로 잰 것(선택 프리미엄·거래수와 gap 의 상관)을 여기서 네 가지로
확장한다. 넷 다 **거래 단위 자료**를 요구해서 `val_log.json` 만으로는 못 한다.

1. 귀무분포 — 거래의 **날짜 딱지만 섞어** 실력 차이를 지우고 같은 규칙을 다시 돌린다. 거래 수
   불균형은 그대로 두므로, 나오는 최댓값 분포가 곧 "아무 신호가 없을 때 이 규칙이 만드는 값"이다.
   풀링 평균은 섞어도 안 변하니(거래 집합이 같다) 일 가중치가 만드는 몫만 정확히 떨어져 나온다.
2. 분산 분해 — 하루 거래 k 건일 때 일평균의 표준오차는 σ/√k 이고, 3 일 일평균의 표준오차는
   (1/3)·√(Σ σ²/k_d) 다. 거래 표에서 σ 를 직접 재서 "거래 ≥30" 문턱이 왜 부족한지 숫자로 본다.
3. 대안 규칙 — (a) 일평균 최대(현행) (b) 풀링 평균 최대 (c) 거래 가중 일평균 최대
   (d) 하루 10 건 이상인 날만 세는 일평균 최대. 넷을 **풀링 평균이라는 공통 잣대**로 비교한다.
4. "최대를 고르면 나쁜 걸 고르는가" — 현행 규칙이 고른 체크포인트의 풀링 평균이 후보 중앙값보다
   나쁠 확률을, 관측과 귀무분포 양쪽에서 센다.

학습 코드는 아무것도 바꾸지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRADES_ROOT = Path.home() / "of80" / "sel98"
MIN_N = 30                    # pick() 의 거래 하한
MIN_PER_DAY = 10              # 대안 규칙 (d) 의 하루 하한
N_REP = 2000                  # 귀무분포 반복
OUT_JSON = Path("data/selection98_deep.json")

_spec = importlib.util.spec_from_file_location(
    "selection98", Path(__file__).resolve().parent / "selection_bias.py")
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

RULES = {"daymean": "(a) 일평균 최대 — 현행",
         "pooled": "(b) 풀링 평균 최대",
         "wdaymean": "(c) 거래 가중 일평균 최대",
         "daymean_min10": f"(d) 일평균 최대 · 하루 {MIN_PER_DAY}건 이상인 날만"}


def score(t: pd.DataFrame, rule: str) -> float:
    """거래 표 → 그 규칙의 점수(bp). 비교 가능하려면 넷이 같은 표를 먹어야 한다."""
    if not len(t):
        return float("nan")
    if rule == "pooled" or rule == "wdaymean":
        # 거래 가중 일평균은 Σ(k_d·m_d)/Σk_d = 풀링 평균이다. 이름만 다른 같은 값이라는 게 요점이다.
        return float(t.ret.mean()) * 1e4
    if rule == "daymean":
        return float(t.groupby("day").ret.mean().mean()) * 1e4
    if rule == "daymean_min10":
        g = t.groupby("day").ret.agg(["mean", "size"])
        g = g[g["size"] >= MIN_PER_DAY]
        return float(g["mean"].mean()) * 1e4 if len(g) else float("nan")
    raise ValueError(rule)


def day_mean_se(sigma_bp: float, k: int) -> float:
    """하루 거래가 k 건일 때 그날 평균의 표준오차."""
    return float(sigma_bp / np.sqrt(k))


def daymean_se(sigma_bp: float, counts) -> float:
    """3 일 일평균(= 일별 평균의 단순평균)의 표준오차 — 얇은 하루가 전체를 지배한다."""
    c = np.asarray(counts, np.float64)
    return float(np.sqrt(np.sum(sigma_bp**2 / c)) / len(c))


def permute_days(t: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """거래는 그대로 두고 **날짜 딱지만** 섞는다 — 하루 거래 수는 보존, 실력 차이는 소멸."""
    out = t.copy()
    out["day"] = rng.permutation(t["day"].to_numpy())
    return out


def null_max(trades: dict, rule: str, n_rep: int = N_REP,
             rng: np.random.Generator | None = None) -> dict:
    """귀무 반복마다 후보 전체를 다시 채점하고 최댓값을 모은다 — 선택 규칙 자체의 분포."""
    rng = rng or np.random.default_rng(98)
    keys = [k for k, t in trades.items() if len(t) >= MIN_N]
    mx = np.empty(n_rep)
    pm = np.empty(n_rep)
    for b in range(n_rep):
        vals = np.array([score(permute_days(trades[k], rng), rule) for k in keys])
        vals = vals[np.isfinite(vals)]
        mx[b] = vals.max() if len(vals) else np.nan
        pm[b] = vals.mean() if len(vals) else np.nan
    return {"max": mx, "pool_mean": pm, "n_cand": len(keys)}


def describe(study: str) -> dict | None:
    f = TRADES_ROOT / study / "trades.pkl"
    if not f.exists():
        print(f"{study}: 재생 결과가 없다 — replay_val.py 를 먼저 돌려라")
        return None
    trades: dict = pickle.load(open(f, "rb"))
    cand = {k: t for k, t in trades.items() if len(t) >= MIN_N}
    if len(cand) < 2:
        return {"n_ckpt": len(trades), "n_cand": len(cand), "note": "후보가 모자라 잴 게 없다"}
    rec: dict = {"n_ckpt": len(trades), "n_cand": len(cand), "min_n": MIN_N,
                 "min_per_day": MIN_PER_DAY, "n_rep": N_REP}

    # ── 3. 네 규칙이 각각 무엇을 고르고, 그 선택의 풀링 평균은 얼마인가 ──────────────
    tab = {r: {k: score(t, r) for k, t in cand.items()} for r in RULES}
    pooled_all = np.array([tab["pooled"][k] for k in cand])
    rec["pool"] = {"pooled_mean": float(pooled_all.mean()),
                   "pooled_median": float(np.median(pooled_all)),
                   "pooled_max": float(pooled_all.max()), "pooled_min": float(pooled_all.min())}
    rec["rules"] = {}
    for r in RULES:
        vals = tab[r]
        fin = {k: v for k, v in vals.items() if v == v}
        if not fin:
            rec["rules"][r] = {"note": "고를 수 있는 후보가 없다"}
            continue
        key = max(fin, key=lambda k: fin[k])          # 동점은 사실상 없다(연속값)
        rec["rules"][r] = {
            "label": RULES[r], "n_eligible": len(fin),
            "seed": int(key[0]), "steps": int(key[1]), "val_n": int(len(cand[key])),
            "score_bp": float(fin[key]),
            "pooled_bp": float(tab["pooled"][key]),         # 공통 잣대
            "daymean_bp": float(tab["daymean"][key]),
            "pooled_pctile": float((pooled_all <= tab["pooled"][key]).mean()),
            "day_counts": sorted(cand[key].groupby("day").size().tolist()),
        }

    # ── 2. 분산 분해 — σ 를 거래 표에서 직접 잰다 ───────────────────────────────────
    sig = np.array([float(t.ret.std(ddof=1)) * 1e4 for t in cand.values() if len(t) > 1])
    sigma = float(np.median(sig))
    counts = np.concatenate([cand[k].groupby("day").size().to_numpy() for k in cand])
    rec["variance"] = {
        "sigma_per_trade_bp_median": sigma,
        "sigma_per_trade_bp_p10": float(np.quantile(sig, 0.10)),
        "sigma_per_trade_bp_p90": float(np.quantile(sig, 0.90)),
        "day_count_median": float(np.median(counts)),
        "day_count_p10": float(np.quantile(counts, 0.10)),
        "day_count_min": int(counts.min()),
        "se_by_k": {str(k): day_mean_se(sigma, k) for k in (1, 3, 5, 10, 30, 100, 300)},
        # 규칙이 허용하는 최악의 쏠림: 3일 총 30건이 (28,1,1) 로 몰린 경우 vs 고르게 (10,10,10)
        "daymean_se_even_10_10_10": daymean_se(sigma, [10, 10, 10]),
        "daymean_se_skewed_28_1_1": daymean_se(sigma, [28, 1, 1]),
    }
    thin = {k: v for k, v in rec["rules"].items() if "day_counts" in v}
    rec["variance"]["selected_daymean_se"] = {
        r: daymean_se(sigma, v["day_counts"]) for r, v in thin.items()}

    # ── 1. 귀무분포 — 날짜 딱지만 섞는다 ───────────────────────────────────────────
    rng = np.random.default_rng(98)
    obs_max = max(v for v in tab["daymean"].values() if v == v)
    obs_pool_mean = float(np.mean([v for v in tab["daymean"].values() if v == v]))
    nl = null_max(cand, "daymean", N_REP, rng)
    prem_null = nl["max"] - nl["pool_mean"]
    obs_prem = obs_max - obs_pool_mean
    rec["null"] = {
        "obs_selected_daymean_bp": float(obs_max), "obs_pool_mean_bp": obs_pool_mean,
        "obs_premium_bp": float(obs_prem),
        "null_premium_mean_bp": float(prem_null.mean()),
        "null_premium_p50_bp": float(np.median(prem_null)),
        "null_premium_p95_bp": float(np.quantile(prem_null, 0.95)),
        "null_premium_p99_bp": float(np.quantile(prem_null, 0.99)),
        "p_value": float((prem_null >= obs_prem).mean()),
        "null_max_p95_bp": float(np.quantile(nl["max"], 0.95)),
    }

    # ── 4. 최대를 고르면 나쁜 걸 고르는가 ─────────────────────────────────────────
    sel_key = max((k for k in cand if tab["daymean"][k] == tab["daymean"][k]),
                  key=lambda k: tab["daymean"][k])
    med_pooled = float(np.median(pooled_all))
    # 귀무 반복마다 "일평균 최대"로 고른 뒤 그 후보의 풀링 평균이 중앙값보다 나쁜지 센다.
    worse = 0
    for _ in range(N_REP):
        vals = {k: score(permute_days(cand[k], rng), "daymean") for k in cand}
        fin = {k: v for k, v in vals.items() if v == v}
        k = max(fin, key=lambda k: fin[k])
        worse += tab["pooled"][k] < med_pooled
    rec["worse_than_median"] = {
        "selected_pooled_bp": float(tab["pooled"][sel_key]),
        "candidate_pooled_median_bp": med_pooled,
        "observed_is_worse": bool(tab["pooled"][sel_key] < med_pooled),
        "null_prob_worse": worse / N_REP,
    }
    return rec


def main(argv: list[str]) -> int:
    out: dict = {"studies": {}}
    for study in argv or ["rl96"]:
        rec = describe(study)
        if rec is None:
            continue
        out["studies"][study] = rec
        print(f"\n■ {study}  체크포인트 {rec['n_ckpt']} · 후보(거래≥{MIN_N}) {rec['n_cand']}")
        if "note" in rec:
            print(f"   {rec['note']}")
            continue
        p, v, n, w = rec["pool"], rec["variance"], rec["null"], rec["worse_than_median"]
        print(f"   후보 풀 풀링평균: 평균 {p['pooled_mean']:+7.1f}bp · 중앙 {p['pooled_median']:+7.1f}bp "
              f"· 최대 {p['pooled_max']:+7.1f}bp")
        print("   ── 네 규칙이 고르는 것 (공통 잣대 = 풀링 평균)")
        for r, e in rec["rules"].items():
            if "label" not in e:
                print(f"      {RULES[r]:<34} {e['note']}")
                continue
            print(f"      {e['label']:<34} seed {e['seed']} · {e['steps']:>10,} · 거래 {e['val_n']:>6,} "
                  f"(일별 {e['day_counts']}) · 자기 점수 {e['score_bp']:+8.1f}bp · "
                  f"**풀링 {e['pooled_bp']:+7.1f}bp** ({e['pooled_pctile']:.0%} 분위)")
        print(f"   ── 분산: 거래당 표준편차 중앙 {v['sigma_per_trade_bp_median']:.0f}bp "
              f"(p10 {v['sigma_per_trade_bp_p10']:.0f} · p90 {v['sigma_per_trade_bp_p90']:.0f})")
        print("      하루 k 건일 때 그날 평균 표준오차: "
              + " · ".join(f"k={k} {s:.0f}bp" for k, s in v["se_by_k"].items()))
        print(f"      3일 일평균 표준오차: 고르게(10,10,10) {v['daymean_se_even_10_10_10']:.0f}bp · "
              f"쏠리면(28,1,1) {v['daymean_se_skewed_28_1_1']:.0f}bp")
        for r, s in v["selected_daymean_se"].items():
            print(f"      {RULES[r]:<34} 선택본의 일평균 표준오차 {s:.1f}bp")
        print(f"   ── 귀무(날짜 딱지 섞기, {rec['n_rep']}회): 관측 프리미엄 {n['obs_premium_bp']:+.1f}bp · "
              f"귀무 평균 {n['null_premium_mean_bp']:+.1f}bp · p95 {n['null_premium_p95_bp']:+.1f}bp "
              f"· p99 {n['null_premium_p99_bp']:+.1f}bp → p = {n['p_value']:.3f}")
        print(f"   ── 최대 선택이 중앙보다 나쁜가: 관측 {'예' if w['observed_is_worse'] else '아니오'} "
              f"(선택 풀링 {w['selected_pooled_bp']:+.1f}bp vs 후보 중앙 "
              f"{w['candidate_pooled_median_bp']:+.1f}bp) · 귀무에서 그럴 확률 {w['null_prob_worse']:.1%}")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n→ {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
