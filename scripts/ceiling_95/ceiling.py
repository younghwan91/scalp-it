"""95번 천장 측정 — **완벽한 예지력(오라클)** 이 비용을 넘는지 본다. 전략이 아니라 산술 검사다.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/ceiling.py            # flow83 일별 캐시 전부

왜: 83~94 에서 모든 정책이 −30bp 언저리로 수렴한다. 이는 "신호가 없다"일 수도, "신호가 있어도
비용이 더 크다"일 수도 있다. 둘은 처방이 다르다. 그래서 **매도 시점을 사후에 최적으로 고른**
상한을 잰다. 오라클조차 못 넘으면 그 진입 방식은 산술적으로 끝이고, 더 좋은 모형은 답이 아니다.

- 테이커 천장: t+1 매도1호가 매수 → [t+2, t+1+H] 매수1호가 **최댓값** 매도 − 23bp
- 메이커 천장: t+1 매수1호가 매수(체결 가정, 대기열 무시 — 낙관) → 같은 구간 매도1호가 최댓값 − 23bp

두 값 모두 실전에서 도달 불가능한 상한이다. 판정이 아니라 기술(description)이므로 사전등록 대상이 아니다.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

COST = 0.0023
HORIZONS = (10, 30, 60, 300, 600, 1800)
LO, HI = 300, 22800 - 600          # 09:05~15:10 (flow83 진입 구간과 같다)
CACHE = Path.home() / "of80" / "flow83"


def oracle(bid: np.ndarray, ask: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """→ (테이커 순수익, 메이커 순수익) — 초 t 진입 결정 기준, 사후 최적 매도."""
    n = len(bid)
    best_bid = np.full(n, np.nan, np.float64)
    best_ask = np.full(n, np.nan, np.float64)
    for j in range(2, h + 2):                     # [t+2, t+1+h]
        sb = np.full(n, np.nan)
        sa = np.full(n, np.nan)
        sb[: n - j] = bid[j:]
        sa[: n - j] = ask[j:]
        best_bid = np.fmax(best_bid, sb)
        best_ask = np.fmax(best_ask, sa)
    entry_taker = np.full(n, np.nan)
    entry_maker = np.full(n, np.nan)
    entry_taker[: n - 1] = ask[1:]
    entry_maker[: n - 1] = bid[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = best_bid / entry_taker - 1 - COST
        mk = best_ask / entry_maker - 1 - COST
    bad = ~(entry_taker > 0) | ~(entry_maker > 0) | ~(best_bid > 0) | ~(best_ask > 0)
    tk[bad] = np.nan
    mk[bad] = np.nan
    return tk, mk


def main(days: list[str] | None = None) -> int:
    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    if days:
        files = [f for f in files if Path(f).stem in days]
    if not files:
        print(f"캐시가 없다: {CACHE}")
        return 2
    acc: dict[int, dict[str, list]] = {h: {"tk": [], "mk": [], "burst": []} for h in HORIZONS}
    for f in files:
        d = pickle.load(open(f, "rb"))
        rows = d["rows"]
        for code, p in d["paths"].items():
            bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
            # 체결 폭발 세기: 그 종목·그 초의 10초 매수체결대금(캐시 행에 있는 값만 쓴다)
            sub = rows[rows.code == code]
            burst = np.zeros(len(bid))
            if len(sub):
                burst[sub.row.to_numpy()] = sub.buyval10.to_numpy()
            for h in HORIZONS:
                tk, mk = oracle(bid, ask, h)
                sl = slice(LO, HI)
                ok = np.isfinite(tk[sl]) & np.isfinite(mk[sl])
                acc[h]["tk"].append(tk[sl][ok])
                acc[h]["mk"].append(mk[sl][ok])
                acc[h]["burst"].append(burst[sl][ok])
    out = {"days": len(files), "cost_bp": COST * 1e4, "horizons": {}}
    for h in HORIZONS:
        tk = np.concatenate(acc[h]["tk"]) * 1e4
        mk = np.concatenate(acc[h]["mk"]) * 1e4
        bu = np.concatenate(acc[h]["burst"])
        top = bu >= np.quantile(bu[bu > 0], 0.99) if (bu > 0).any() else np.zeros(len(bu), bool)
        rec = {
            "n_sec": int(len(tk)),
            "taker": {"median": float(np.median(tk)), "mean": float(tk.mean()),
                      "p90": float(np.quantile(tk, 0.90)), "frac_pos": float((tk > 0).mean())},
            "maker": {"median": float(np.median(mk)), "mean": float(mk.mean()),
                      "p90": float(np.quantile(mk, 0.90)), "frac_pos": float((mk > 0).mean())},
            "burst_top1pct": {"n": int(top.sum()),
                              "taker_median": float(np.median(tk[top])) if top.any() else None,
                              "maker_median": float(np.median(mk[top])) if top.any() else None},
        }
        out["horizons"][str(h)] = rec
        print(f"H={h:>2}s  n={rec['n_sec']:,}")
        for k in ("taker", "maker"):
            r = rec[k]
            print(f"   {k:<5} 중앙 {r['median']:+7.1f}bp · 평균 {r['mean']:+7.1f}bp · "
                  f"상위10% {r['p90']:+7.1f}bp · 양(+) 비중 {r['frac_pos']:.1%}")
        b = rec["burst_top1pct"]
        if b["taker_median"] is not None:
            print(f"   체결폭발 상위1%({b['n']:,}초): 테이커 중앙 {b['taker_median']:+.1f}bp · "
                  f"메이커 중앙 {b['maker_median']:+.1f}bp")
    Path("data").mkdir(exist_ok=True)
    Path("data/ceiling95.json").write_text(json.dumps(out, indent=1))
    print("\n→ data/ceiling95.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
