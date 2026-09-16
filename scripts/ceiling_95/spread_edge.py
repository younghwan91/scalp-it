"""95e번 스프레드 우위 — **양쪽 다 지정가**로 돌면 비용을 넘는 자리가 얼마나 있나. 기술, 판정 아님.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/spread_edge.py

왜: [예측가능성 탐침](../../docs/research/manju/95-predictability-probe.md)이 "모형이 배운 건 방향이 아니라
**비용 회피**(시각·유동성·스프레드가 gain 의 70%)"라고 했다. 그렇다면 방향을 맞히려 들지 말고,
**스프레드 자체가 비용보다 넓은 자리**를 세는 게 맞는 질문이다.

왕복을 전부 지정가로 하면(매수1호가에 사고 매도1호가에 판다) 총수익은
    스프레드 − 23bp − 역선택
이다. 스프레드는 t 시점에 **알 수 있는 값**이다 — 예측이 아니다. 그래서 이 측정은 "얼마나 자주 그 자리가
있는가"와 "거기서 실제로 어떻게 되는가"를 같이 센다.

- 진입: `t+1` 매수1호가 지정가. 체결은 **뚫림**(가격이 우리 아래로 내려옴, 10초 안)만 인정 — 역선택 최대.
- 청산: 체결 뒤 `H` 초 안에 **매도1호가에 지정가 매도**가 체결되는가 = 그 사이 매도1호가 위로 가격이 뚫렸는가.
  못 팔면 창 끝 매수1호가에 시장가로 던진다(현실적인 최악).
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

COST = 0.0023
FILL_SEC = 10
HORIZONS = (60, 300, 600)
SPREAD_BINS = ((0.0, 15.0), (15.0, 25.0), (25.0, 40.0), (40.0, 70.0), (70.0, 1e9))
LO, HI = 300, 22800 - 900
CACHE = Path.home() / "of80" / "flow83"


def round_trip(bid: np.ndarray, ask: np.ndarray, e: np.ndarray, h: int) -> np.ndarray:
    """양쪽 지정가 왕복의 순수익. 체결 초 `e` 에서 시작해 `h` 초 안에 매도1호가에 팔리면 그 값.

    팔리는 조건은 진입과 대칭이다 — 가격이 **우리 매도 지정가 위로 뚫고 올라가야** 한다.
    못 팔면 창 끝 매수1호가에 시장가 청산(지정가 전략의 현실적 최악).
    """
    n = len(bid)
    out = np.full(n, np.nan)
    for t in range(n - h - FILL_SEC - 3):
        s = e[t]
        if s < 0:
            continue
        P = bid[t + 1]
        S = ask[s]                                  # 체결 직후의 매도1호가에 매도 지정가를 건다
        if not (P > 0) or not (S > 0):
            continue
        end = min(s + h, n - 1)
        sold = -1
        for j in range(s + 1, end + 1):
            if ask[j] > S:                          # 매도1호가가 올라갔다 = 우리 물량이 팔렸다
                sold = j
                break
        px = S if sold >= 0 else bid[end]
        if px > 0:
            out[t] = px / P - 1 - COST
    return out


try:
    import numba

    round_trip = numba.njit(round_trip)
except Exception:  # noqa: BLE001
    pass


def main(days: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("ceiling95", Path(__file__).parent / "ceiling.py")
    C = importlib.util.module_from_spec(spec)
    sys.modules["ceiling95"] = C
    spec.loader.exec_module(C)

    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    if days:
        files = [f for f in files if Path(f).stem in days]
    acc = {h: {"r": [], "sp": [], "sold": []} for h in HORIZONS}
    for f in files:
        d = pickle.load(open(f, "rb"))
        for _code, p in d["paths"].items():
            bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
            good = (bid > 0) & (ask > 0)
            sp = np.where(good, (ask / bid - 1) * 1e4, np.nan)
            e = C.fill_second(bid)
            for h in HORIZONS:
                r = round_trip(bid, ask, e, h)
                sl = slice(LO, HI)
                ok = np.isfinite(r[sl]) & np.isfinite(sp[sl])
                acc[h]["r"].append(r[sl][ok])
                acc[h]["sp"].append(sp[sl][ok])
    out = {"days": len(files), "cost_bp": COST * 1e4, "horizons": {}}
    for h in HORIZONS:
        r = np.concatenate(acc[h]["r"]) * 1e4
        sp = np.concatenate(acc[h]["sp"])
        rec = {"n": int(len(r)), "median": float(np.median(r)), "mean": float(r.mean()),
               "frac_pos": float((r > 0).mean()), "bins": {}}
        print(f"H={h:>4}s  체결 {len(r):,}건 · 중앙 {rec['median']:+.1f}bp · "
              f"평균 {rec['mean']:+.1f}bp · 양(+) {rec['frac_pos']:.1%}")
        for lo, hi in SPREAD_BINS:
            m = (sp >= lo) & (sp < hi)
            lab = f"{lo:g}~{hi:g}" if hi < 1e8 else f"{lo:g}+"
            b = {"n": int(m.sum()), "share": float(m.mean()),
                 "median": float(np.median(r[m])) if m.any() else None,
                 "mean": float(r[m].mean()) if m.any() else None,
                 "frac_pos": float((r[m] > 0).mean()) if m.any() else None}
            rec["bins"][lab] = b
            if b["n"]:
                print(f"     스프레드 {lab:>8}bp: 중앙 {b['median']:+7.1f} · 평균 {b['mean']:+7.1f} · "
                      f"양(+) {b['frac_pos']:.1%} ({b['n']:,}건, 전체의 {b['share']:.1%})")
        out["horizons"][str(h)] = rec
    Path("data").mkdir(exist_ok=True)
    Path("data/spread_edge95.json").write_text(json.dumps(out, indent=1))
    print("\n→ data/spread_edge95.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
