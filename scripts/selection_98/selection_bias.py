"""98번 — 우리 체크포인트 선택 규칙이 얼마나 낙관적인지 잰다. 판정이 아니라 기술이다.

    ~/of80/.rlvenv/bin/python scripts/selection_98/selection_bias.py

왜: `rl_*/dqn.py` 의 `pick()` 은 모두 같다 — **거래 ≥30 이고 일평균이 NaN 이 아닌 체크포인트 중
`val_daymean_bp` 최대**. 시드 3 개 × 평가 20 회면 후보가 60 개다. 60 개 중 최댓값은 실력이 아니라
잡음이어도 뽑힌다. 96번에서 나온 `{"val_n": 91, "val_daymean_bp": -4.16, "val_mean_bp": -27.97}`
처럼 **하루가 평균을 끌어올린** 체크포인트가 보이면 그게 그 신호다.

세 가지를 잰다.
 1. 선택 프리미엄 — 뽑힌 값이 같은 후보 풀에서 무작위로 고른 값보다 몇 bp 높은가. 후보 수 m 을
    늘려 가며 "최댓값의 기댓값"을 정확히 계산하면, 프리미엄 중 얼마가 **탐색 자체**의 몫인지 보인다.
 2. `val_n` 과 일평균의 관계 — 거래가 적을수록 일평균이 극단으로 가는가(잡음 선택의 서명).
 3. `val_mean_bp`(거래 풀링 평균)로 골랐으면 무엇이 뽑혔을까, 그 차이는 얼마인가.

`val_daymean_bp − val_mean_bp` 는 **하루에 몇 건 넣었는지와 그날 성과의 공분산**이다. 큰 양수는
"많이 번 날에 거래가 적었다"는 뜻이고, 그건 실력이 아니라 표본이 작았다는 뜻이다. 이 벌어짐을
`gap` 으로 부른다.

코드는 아무것도 바꾸지 않는다 — 96번은 사전등록이 잠겨 있다. 여기서 나온 숫자는 앞으로의 설계에 쓴다.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

RL_ROOT = Path.home() / "of80"
MIN_N = 30                             # pick() 의 거래 하한 — 여기가 다르면 다른 규칙이다
OUT_JSON = Path("data/selection98.json")


def eligible(rows: list[dict]) -> list[dict]:
    """`pick()` 의 필터 그대로 — 거래 ≥30 이고 일평균이 NaN 이 아닌 행."""
    return [r for r in rows
            if r.get("val_n", 0) >= MIN_N and r.get("val_daymean_bp") == r.get("val_daymean_bp")]


def argmax_by(rows: list[dict], key: str) -> dict | None:
    """`pick()` 과 같은 순회·같은 동점 처리(strict >, 먼저 만난 쪽이 이긴다)."""
    best = None
    for r in rows:
        v = r.get(key)
        if v is None or v != v:
            continue
        if best is None or v > best[key]:
            best = r
    return best


def gap_bp(row: dict) -> float:
    """일평균 − 풀링 평균. 거래가 적은 날이 좋았을수록 커진다."""
    return float(row["val_daymean_bp"] - row["val_mean_bp"])


def expected_max(values: np.ndarray, m: int) -> float:
    """후보 n 개에서 **비복원으로 m 개**를 뽑았을 때 최댓값의 기댓값 — 순서통계량으로 정확히 센다.

    표본이 아니라 조합 가중치라 시드에 안 흔들린다. m=1 이면 평균, m=n 이면 최댓값이다.
    그 사이 곡선이 "후보를 늘리면 공짜로 얼마가 올라가는가"를 보여 준다.
    """
    v = np.sort(np.asarray(values, np.float64))
    n = len(v)
    if m >= n:
        return float(v[-1])
    tot = math.comb(n, m)
    w = np.array([math.comb(i, m - 1) for i in range(n)], np.float64) / tot
    return float((v * w).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def load_study(name: str) -> list[dict]:
    """한 연구의 모든 시드·모든 평가 회차를 `pick()` 이 보는 순서대로 이어 붙인다."""
    rows: list[dict] = []
    for f in sorted(glob.glob(str(RL_ROOT / name / "seed*" / "val_log.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return rows


def describe(name: str) -> dict | None:
    rows = load_study(name)
    if not rows:
        return None
    el = eligible(rows)
    if len(el) < 2:
        return {"n_ckpt": len(rows), "n_eligible": len(el), "note": "후보가 모자라 잴 게 없다"}
    day = np.array([r["val_daymean_bp"] for r in el], np.float64)
    pooled = np.array([r["val_mean_bp"] for r in el], np.float64)
    nn = np.array([r["val_n"] for r in el], np.float64)
    gaps = day - pooled
    sel_day = argmax_by(el, "val_daymean_bp")           # 실제로 쓰는 규칙
    sel_pool = argmax_by(el, "val_mean_bp")             # 대안 규칙
    rec = {
        "n_ckpt": len(rows), "n_eligible": len(el),
        "pool": {"daymean_mean": float(day.mean()), "daymean_median": float(np.median(day)),
                 "daymean_sd": float(day.std(ddof=1)), "daymean_max": float(day.max()),
                 "pooled_mean": float(pooled.mean()), "pooled_median": float(np.median(pooled)),
                 "gap_median": float(np.median(gaps)), "gap_p90": float(np.quantile(gaps, 0.9))},
        "selected_by_daymean": {k: sel_day[k] for k in
                                ("seed", "steps", "val_n", "val_daymean_bp", "val_mean_bp")},
        "selected_by_pooled": {k: sel_pool[k] for k in
                               ("seed", "steps", "val_n", "val_daymean_bp", "val_mean_bp")},
        "same_checkpoint": bool(sel_day["seed"] == sel_pool["seed"]
                                and sel_day["steps"] == sel_pool["steps"]),
    }
    # 재현 확인 — 그 연구가 실제로 남긴 selected.json 과 같은 걸 골랐는지. 다르면 내 재현이 틀린 것이다.
    f = RL_ROOT / name / "selected.json"
    if f.exists():
        was = json.loads(f.read_text())
        rec["reproduces_selected_json"] = bool(was["seed"] == sel_day["seed"]
                                               and was["steps"] == sel_day["steps"])
    # 1. 선택 프리미엄 — 무작위 한 개(=풀 평균) 대비, 그리고 후보 수 m 별 최댓값 기댓값
    rec["premium_vs_random_bp"] = float(sel_day["val_daymean_bp"] - day.mean())
    rec["premium_vs_median_bp"] = float(sel_day["val_daymean_bp"] - np.median(day))
    ms = sorted({1, 2, 3, 5, 10, 20, len(el)})
    rec["expected_max_curve"] = {str(m): expected_max(day, m) for m in ms if m <= len(el)}
    rec["selected_gap_bp"] = gap_bp(sel_day)
    rec["selected_gap_pctile"] = float((gaps <= gap_bp(sel_day)).mean())
    # 2. 거래 수와 일평균의 관계 — 적을수록 극단으로 가는가
    med = float(np.median(day))
    rec["rho_n_vs_daymean"] = spearman(nn, day)
    rec["rho_n_vs_absdev"] = spearman(nn, np.abs(day - med))
    rec["rho_n_vs_absgap"] = spearman(nn, np.abs(gaps))
    q = np.quantile(nn, [0.25, 0.5, 0.75])
    rec["by_n_quartile"] = []
    edges = [nn.min() - 1, *q, nn.max() + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (nn > lo) & (nn <= hi)
        if not sel.any():
            continue
        rec["by_n_quartile"].append(
            {"n_lo": float(nn[sel].min()), "n_hi": float(nn[sel].max()), "k": int(sel.sum()),
             "daymean_mean": float(day[sel].mean()), "daymean_sd": float(day[sel].std(ddof=1))
             if sel.sum() > 1 else None,
             "daymean_max": float(day[sel].max()), "absgap_median": float(np.median(np.abs(gaps[sel])))})
    # 3. 두 규칙의 차이 — 무엇을 잃고 무엇을 얻나
    rec["rule_delta"] = {
        "daymean_bp": float(sel_pool["val_daymean_bp"] - sel_day["val_daymean_bp"]),
        "pooled_bp": float(sel_pool["val_mean_bp"] - sel_day["val_mean_bp"]),
        "val_n": int(sel_pool["val_n"] - sel_day["val_n"]),
    }
    return rec


def main(argv: list[str] | None = None) -> int:
    studies = argv or ["rl85", "rl87", "rl92", "rl93", "rl94", "rl96"]
    out: dict[str, dict] = {"min_n": MIN_N, "studies": {}}
    for name in studies:
        rec = describe(name)
        if rec is None:
            print(f"{name}: val_log 가 없다")
            continue
        out["studies"][name] = rec
        rp = rec.get("reproduces_selected_json")
        mark = "" if rp is None else ("  [selected.json 재현 ✓]" if rp else "  [재현 실패 — 확인 필요]")
        print(f"\n■ {name}  체크포인트 {rec['n_ckpt']} · 후보(거래≥{MIN_N}) {rec['n_eligible']}{mark}")
        if "note" in rec:
            print(f"   {rec['note']}")
            continue
        p = rec["pool"]
        s, t = rec["selected_by_daymean"], rec["selected_by_pooled"]
        print(f"   후보 풀 일평균: 평균 {p['daymean_mean']:+8.1f}bp · 중앙 {p['daymean_median']:+8.1f}bp "
              f"· 표준편차 {p['daymean_sd']:8.1f}bp")
        print(f"   선택(일평균 최대): seed {s['seed']} · {s['steps']:,} steps · 거래 {s['val_n']:,} · "
              f"일평균 {s['val_daymean_bp']:+8.1f}bp · 풀링 {s['val_mean_bp']:+8.1f}bp")
        print(f"   → 선택 프리미엄 {rec['premium_vs_random_bp']:+8.1f}bp(무작위 평균 대비) · "
              f"{rec['premium_vs_median_bp']:+8.1f}bp(중앙 대비)")
        print("   최댓값 기댓값(후보 m 개): "
              + " · ".join(f"m={m} {v:+.1f}" for m, v in rec["expected_max_curve"].items()))
        print(f"   선택된 것의 gap(일평균−풀링) {rec['selected_gap_bp']:+8.1f}bp "
              f"(후보 중 {rec['selected_gap_pctile']:.0%} 분위) · 후보 중앙 gap {p['gap_median']:+.1f}bp")
        print(f"   거래수 vs 일평균 ρ {rec['rho_n_vs_daymean']:+.3f} · "
              f"vs |일평균−중앙| ρ {rec['rho_n_vs_absdev']:+.3f} · vs |gap| ρ {rec['rho_n_vs_absgap']:+.3f}")
        for b in rec["by_n_quartile"]:
            sd = f"{b['daymean_sd']:7.1f}" if b["daymean_sd"] is not None else "      -"
            print(f"      거래 {b['n_lo']:>7,.0f}~{b['n_hi']:>7,.0f} ({b['k']:>2}개): 일평균 평균 "
                  f"{b['daymean_mean']:+8.1f}bp · 표준편차 {sd}bp · 최대 {b['daymean_max']:+8.1f}bp · "
                  f"|gap| 중앙 {b['absgap_median']:7.1f}bp")
        print(f"   풀링 평균으로 골랐다면: seed {t['seed']} · {t['steps']:,} steps · 거래 {t['val_n']:,} · "
              f"일평균 {t['val_daymean_bp']:+8.1f}bp · 풀링 {t['val_mean_bp']:+8.1f}bp "
              f"({'같은' if rec['same_checkpoint'] else '다른'} 체크포인트)")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n→ {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
