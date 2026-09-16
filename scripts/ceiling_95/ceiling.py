"""95번 천장 측정 — **완벽한 예지력(오라클)** 이 비용을 넘는지 본다. 전략이 아니라 산술 검사다.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/ceiling.py            # flow83 일별 캐시 전부

왜: 83~94 에서 모든 정책이 −30bp 언저리로 수렴한다. 이는 "신호가 없다"일 수도, "신호가 있어도
비용이 더 크다"일 수도 있다. 둘은 처방이 다르다. 그래서 **매도 시점을 사후에 최적으로 고른**
상한을 잰다. 오라클조차 못 넘으면 그 진입 방식은 산술적으로 끝이고, 더 좋은 모형은 답이 아니다.

- 테이커 천장: t+1 매도1호가 매수 → [t+2, t+1+H] 매수1호가 **최댓값** 매도 − 23bp
- 메이커 천장(낙관): t+1 매수1호가 매수(체결 가정, 대기열 무시) → 같은 구간 매도1호가 최댓값 − 23bp
- 메이커 천장(비관): 같은 지정가가 **가격이 뚫고 내려와야만** 체결됐다고 본다(10초 안). 이는 역선택이
  가장 심한 부분집합이다 — 실제 체결(대기열 소진 포함)은 이보다 낫다. 두 값이 실전을 위아래로 감싼다.

세 값 모두 실전 도달 불가능한 상·하한이다. 판정이 아니라 기술(description)이므로 사전등록 대상이 아니다.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

COST = 0.0023
FILL_SEC = 10                      # 지정가를 이 초 안에 못 채우면 취소(86 §3 과 같다)
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


def fill_second(bid: np.ndarray) -> np.ndarray:
    """지정가 bid[t+1] 이 **뚫려서** 체결된 초. 없으면 −1. (t 당 한 번, 보유 구간과 무관)"""
    n = len(bid)
    e = np.full(n, -1, np.int64)
    for t in range(n - FILL_SEC - 3):
        P = bid[t + 1]
        if not (P > 0):
            continue
        for s in range(t + 2, t + 2 + FILL_SEC):
            if bid[s] > 0 and bid[s] < P:            # 뚫고 내려왔다 → 우리 주문은 확실히 체결
                e[t] = s
                break
    return e


try:
    import numba

    fill_second = numba.njit(fill_second)            # 초 단위 이중 루프 — 파이썬으로는 못 돈다
except Exception:                                     # noqa: BLE001
    pass


def window_max(x: np.ndarray, h: int) -> np.ndarray:
    """w[i] = max(x[i : i+h]) — 뒤가 모자라면 NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    if n > h:
        from numpy.lib.stride_tricks import sliding_window_view

        out[: n - h + 1] = np.nanmax(sliding_window_view(x, h), axis=1)
    return out


def maker_pessimistic(bid: np.ndarray, e: np.ndarray, maxask: np.ndarray) -> np.ndarray:
    """역선택이 최대인 부분집합의 순수익 — 뚫려서 체결된 초부터 사후 최적 매도. 미체결은 NaN."""
    n = len(bid)
    out = np.full(n, np.nan)
    ok = e >= 0
    idx = np.nonzero(ok)[0]
    nxt = e[idx] + 1
    good = nxt < n
    idx, nxt = idx[good], nxt[good]
    P = bid[idx + 1]
    best = maxask[nxt]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = best / P - 1 - COST
    r[~(P > 0) | ~(best > 0)] = np.nan
    out[idx] = r
    return out


def main(days: list[str] | None = None) -> int:
    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    if days:
        files = [f for f in files if Path(f).stem in days]
    if not files:
        print(f"캐시가 없다: {CACHE}")
        return 2
    acc: dict[int, dict[str, list]] = {h: {"tk": [], "mk": [], "mp": [], "burst": []} for h in HORIZONS}
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
            e_fill = fill_second(bid)
            for h in HORIZONS:
                tk, mk = oracle(bid, ask, h)
                sl = slice(LO, HI)
                ok = np.isfinite(tk[sl]) & np.isfinite(mk[sl])
                acc[h]["tk"].append(tk[sl][ok])
                acc[h]["mk"].append(mk[sl][ok])
                mp = maker_pessimistic(bid, e_fill, window_max(ask, h))
                acc[h]["mp"].append(mp[sl][ok])
                acc[h]["burst"].append(burst[sl][ok])
    out = {"days": len(files), "cost_bp": COST * 1e4, "horizons": {}}
    for h in HORIZONS:
        tk = np.concatenate(acc[h]["tk"]) * 1e4
        mk = np.concatenate(acc[h]["mk"]) * 1e4
        mp = np.concatenate(acc[h]["mp"]) * 1e4
        mpf = mp[np.isfinite(mp)]
        bu = np.concatenate(acc[h]["burst"])
        top = bu >= np.quantile(bu[bu > 0], 0.99) if (bu > 0).any() else np.zeros(len(bu), bool)
        rec = {
            "n_sec": int(len(tk)),
            "taker": {"median": float(np.median(tk)), "mean": float(tk.mean()),
                      "p90": float(np.quantile(tk, 0.90)), "frac_pos": float((tk > 0).mean())},
            "maker": {"median": float(np.median(mk)), "mean": float(mk.mean()),
                      "p90": float(np.quantile(mk, 0.90)), "frac_pos": float((mk > 0).mean())},
            "maker_pessimistic": {"n": int(len(mpf)), "fill_rate": float(len(mpf) / max(len(mp), 1)),
                                  "median": float(np.median(mpf)) if len(mpf) else None,
                                  "mean": float(mpf.mean()) if len(mpf) else None,
                                  "frac_pos": float((mpf > 0).mean()) if len(mpf) else None},
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
        q = rec["maker_pessimistic"]
        if q["median"] is not None:
            print(f"   maker(비관·뚫림체결 {q['fill_rate']:.1%}) 중앙 {q['median']:+7.1f}bp · "
                  f"평균 {q['mean']:+7.1f}bp · 양(+) 비중 {q['frac_pos']:.1%}")
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
