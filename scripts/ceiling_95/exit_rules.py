"""95f번 청산 규칙 격자 — **RL 이 넘어야 할 하한선**을 긋는다. 탐색이고 판정이 아니다.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/exit_rules.py

[95e](spread_edge.py)에서 같은 진입 모집단(가격이 지정가를 뚫고 내려와 체결된 초)의 결과가
청산 방식만으로 **−23.0bp(매도1호가 고정) ~ +21.3bp(사후 최적)** 로 갈렸다. 그 44bp 가 전부
청산 타이밍이다. 그렇다면 **사람이 쓸 법한 고정 규칙**은 그중 얼마를 먹나 — 그게 96번 RL 의 하한선이다.

규칙: 체결가 `P` 에서 시작해
- **익절**: `P + k틱` 에 지정가 매도. 매도1호가가 그 위로 올라가면 체결로 본다(95e 와 같은 대칭 판정).
- **손절**: 매수1호가가 `P − s틱` 이하로 내려가면 그 자리에서 시장가 청산. `s=0` 은 손절 없음.
- **시한**: `H` 초 안에 둘 다 안 걸리면 창 끝 매수1호가에 시장가.

**이 격자는 시행 원장에 들어간다** — 여기서 고른 칸을 그대로 승격시키지 않는다. 사전등록된 96번의
하한선을 긋는 용도이고, 규칙 자체를 쓰려면 새 번호로 사전등록해야 한다.
"""
from __future__ import annotations

import glob
import importlib.util
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

COST = 0.0023
LO, HI = 300, 22800 - 1200
TAKES = (1, 2, 3, 4, 6, 8)          # 익절 틱
STOPS = (0, 2, 4, 8)                # 손절 틱(0 = 없음)
HORIZONS = (60, 300, 600)
CACHE = Path.home() / "of80" / "flow83"


def run_rule(bid, ask, tick, e, k, s, h):
    """→ 체결된 진입마다의 순수익(미체결은 NaN). 규칙은 전부 **사전확정**이다 — 미래를 안 본다."""
    n = len(bid)
    out = np.full(n, np.nan)
    for t in range(n - h - 14):
        f = e[t]
        if f < 0:
            continue
        P = bid[t + 1]
        tk = tick[f]
        if not (P > 0) or not (tk > 0):
            continue
        target = P + k * tk
        stop = P - s * tk if s > 0 else -1.0
        end = min(f + h, n - 1)
        px = -1.0
        for j in range(f + 1, end + 1):
            if stop > 0 and bid[j] > 0 and bid[j] <= stop:
                px = bid[j]                      # 손절 — 시장가로 던진다
                break
            if ask[j] > target:                  # 매도1호가가 우리 위로 올라갔다 = 팔렸다
                px = target
                break
        if px < 0:
            px = bid[end]
        if px > 0:
            out[t] = px / P - 1 - COST
    return out


try:
    import numba

    run_rule = numba.njit(run_rule)
except Exception:  # noqa: BLE001
    pass


def main(days: list[str] | None = None) -> int:
    spec = importlib.util.spec_from_file_location("ceiling95", Path(__file__).parent / "ceiling.py")
    C = importlib.util.module_from_spec(spec)
    sys.modules["ceiling95"] = C
    spec.loader.exec_module(C)

    files = sorted(glob.glob(str(CACHE / "*.pkl")))
    if days:
        files = [f for f in files if Path(f).stem in days]
    cells = list(itertools.product(TAKES, STOPS, HORIZONS))
    acc: dict[tuple, list] = {c: [] for c in cells}
    for f in files:
        d = pickle.load(open(f, "rb"))
        for _code, p in d["paths"].items():
            bid = p["bid"].astype(np.float64)
            ask = p["ask"].astype(np.float64)
            tick = p["tick"].astype(np.float64)
            e = C.fill_second(bid)
            if (e >= 0).sum() < 10:
                continue
            for k, s, h in cells:
                r = run_rule(bid, ask, tick, e, k, s, h)[LO:HI]
                acc[(k, s, h)].append(r[np.isfinite(r)])
    rows = []
    for c, parts in acc.items():
        r = np.concatenate(parts) * 1e4 if parts else np.array([])
        if not len(r):
            continue
        rows.append({"take_ticks": c[0], "stop_ticks": c[1], "horizon": c[2], "n": int(len(r)),
                     "median": float(np.median(r)), "mean": float(r.mean()),
                     "frac_pos": float((r > 0).mean())})
    rows.sort(key=lambda x: -x["mean"])
    print(f"격자 {len(rows)}칸 (시행 원장에 들어간다) — 평균 상위 12칸")
    print(f"{'익절틱':>6}{'손절틱':>7}{'시한':>7}{'건수':>12}{'평균bp':>9}{'중앙bp':>9}{'양(+)':>8}")
    for r in rows[:12]:
        print(f"{r['take_ticks']:>6}{r['stop_ticks']:>7}{r['horizon']:>7}{r['n']:>12,}"
              f"{r['mean']:>+9.1f}{r['median']:>+9.1f}{r['frac_pos']:>8.1%}")
    best = rows[0] if rows else None
    print(f"\n최고 칸 평균 {best['mean']:+.1f}bp — 95e 고정 규칙 −23.0bp, 오라클 천장 +21.3bp 사이 어디인가")
    # 격자는 시행이다 — 원장에 남겨 Deflated Sharpe 의 N 이 커지게 한다(안 남기면 다음 판정이 거짓말이 된다).
    try:
        from krx_quant_core.stats.trials import record_trial

        for r in rows:
            record_trial("95f-exit-rules", {k: r[k] for k in ("take_ticks", "stop_ticks", "horizon")},
                         logs_dir=Path(__file__).resolve().parents[2] / "research" / "logs",
                         note="탐색 격자 — 96번 RL 의 하한선 측정")
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] 시행 원장 기록 실패: {exc}")
    Path("data").mkdir(exist_ok=True)
    Path("data/exit_rules95.json").write_text(json.dumps({"cells": rows}, indent=1))
    print("→ data/exit_rules95.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
