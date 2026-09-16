"""95h번 변동성 매칭 대조 — **딥 체결 우위가 진짜인지, 그냥 변동성이 큰 순간이었는지.**

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/vol_matched.py --horizon 300

왜: 95 의 대조군은 "같은 종목·같은 날의 무작위 초"였다. 딥 체결은 **정의상 가격이 움직이는 순간**에
일어나므로, 변동성 국면이 안 맞춰진 대조와 비교하면 이기는 게 당연할 수 있다. 창 안 최댓값은 σ√H 로
커지기 때문이다.

그래서 **진입 직전까지의 실현 변동성**(t 이하 300초 초당 수익률 표준편차 — 결정 시점 정보만 쓴다)으로
버킷을 나누고, **같은 버킷 안에서** 딥 체결과 무작위 초를 비교한다. 버킷 안에서 차이가 사라지면
+14bp 는 변동성이었고, 남으면 딥 체결 자체가 정보다. **이 한 숫자가 96번의 사전 확률을 정한다.**
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / "of80" / "flow83"
RV_WIN = 300
N_BUCKET = 5


def pre_entry_rv(mid: np.ndarray, win: int = RV_WIN) -> np.ndarray:
    """초 t 까지(포함)의 실현 변동성 bp — **미래를 안 본다**."""
    r = pd.Series(mid).pct_change()
    return r.rolling(win, min_periods=30).std().to_numpy() * 1e4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--out", default="data/vol_matched95.json")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ceiling95", Path(__file__).parent / "ceiling.py")
    C = importlib.util.module_from_spec(spec)
    sys.modules["ceiling95"] = C
    spec.loader.exec_module(C)

    dip_r, dip_v, all_r, all_v = [], [], [], []
    for f in sorted(glob.glob(str(CACHE / "*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for _code, p in d["paths"].items():
            bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
            mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)
            rv = pre_entry_rv(mid)
            e = C.fill_second(bid)
            maxask = C.window_max(ask, a.horizon)
            dip = C.maker_pessimistic(bid, e, maxask)        # 딥 체결만
            _tk, mk = C.oracle(bid, ask, a.horizon)          # 모든 초(= 무작위 진입)
            sl = slice(C.LO, C.HI)
            for arr, rs, vs in ((dip, dip_r, dip_v), (mk, all_r, all_v)):
                x, v = arr[sl], rv[sl]
                ok = np.isfinite(x) & np.isfinite(v)
                rs.append(x[ok])
                vs.append(v[ok])
    dip_r = np.concatenate(dip_r) * 1e4
    dip_v = np.concatenate(dip_v)
    all_r = np.concatenate(all_r) * 1e4
    all_v = np.concatenate(all_v)

    # 버킷 경계는 **무작위 초 전체**로 정한다 — 딥 쪽으로 경계가 끌려가면 비교가 깨진다.
    edges = np.quantile(all_v, np.linspace(0, 1, N_BUCKET + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    print(f"보유 {a.horizon}초 · 딥 체결 {len(dip_r):,} · 무작위 {len(all_r):,}")
    print(f"{'변동성버킷(bp)':>18}{'무작위 중앙':>12}{'딥 중앙':>10}{'차이':>9}"
          f"{'무작위 평균':>12}{'딥 평균':>10}{'차이':>9}{'딥 n':>11}")
    for i in range(N_BUCKET):
        lo, hi = edges[i], edges[i + 1]
        md = (dip_v >= lo) & (dip_v < hi)
        ma = (all_v >= lo) & (all_v < hi)
        if md.sum() < 100 or ma.sum() < 100:
            continue
        rec = {"bucket": i, "lo": float(lo), "hi": float(hi),
               "rand_median": float(np.median(all_r[ma])), "dip_median": float(np.median(dip_r[md])),
               "rand_mean": float(all_r[ma].mean()), "dip_mean": float(dip_r[md].mean()),
               "n_dip": int(md.sum()), "n_rand": int(ma.sum())}
        rec["d_median"] = rec["dip_median"] - rec["rand_median"]
        rec["d_mean"] = rec["dip_mean"] - rec["rand_mean"]
        rows.append(rec)
        lab = f"{lo if np.isfinite(lo) else 0:.0f}~{hi if np.isfinite(hi) else 9999:.0f}"
        print(f"{lab:>18}{rec['rand_median']:>+12.1f}{rec['dip_median']:>+10.1f}{rec['d_median']:>+9.1f}"
              f"{rec['rand_mean']:>+12.1f}{rec['dip_mean']:>+10.1f}{rec['d_mean']:>+9.1f}{rec['n_dip']:>11,}")
    pooled_md = float(np.median(dip_r) - np.median(all_r))
    wmean = float(np.average([r["d_mean"] for r in rows], weights=[r["n_dip"] for r in rows])) if rows else float("nan")
    wmed = float(np.average([r["d_median"] for r in rows], weights=[r["n_dip"] for r in rows])) if rows else float("nan")
    print(f"\n매칭 전 중앙값 차이 {pooled_md:+.1f}bp → 변동성 매칭 후 {wmed:+.1f}bp (평균 차이 {wmean:+.1f}bp)")
    print("남으면 딥 체결이 정보, 사라지면 변동성이었다.")
    Path("data").mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps({"horizon": a.horizon, "pooled_d_median": pooled_md,
                                       "matched_d_median": wmed, "matched_d_mean": wmean,
                                       "buckets": rows}, indent=1))
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
