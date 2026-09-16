"""95b번 ETF 천장 — **증권거래세가 없는** 종목에서 같은 상한을 잰다. 판정 아님, 기술.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/etf_ceiling.py 2026-09-16

왜: 95번이 찾은 블로커는 **비용 23bp** 였고, 그 중 20bp 는 매도 증권거래세·농특세다.
ETF 는 이 세금이 **면제**다(수수료만 남는다). 즉 블로커의 87% 가 ETF 에서는 존재하지 않는다.
주식에서 불가능했던 수초 보유가 ETF 에서도 불가능한지는 **다시 재야** 안다.

비용 가정(왕복):
- `--cost-bp 3.0`  기본 수수료 0.015% × 2 = 3bp (세금 0)
- `--cost-bp 0.72` 비대면 우대 수수료 0.0036% × 2 (계좌 실수수료 확인 전까지 참고용)

가격 격자·오라클 정의는 95번(`ceiling.py`)과 **같은 함수**를 쓴다 — 주식 결과와 직접 비교하기 위해서다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _envload  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("ceiling95", Path(__file__).parent / "ceiling.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

T0, N = 9 * 3600, 22800
ETFS = ("069500", "122630", "102110", "360750", "133690", "379800", "114800", "233740",
        "229200", "396500", "091160", "251340", "381180", "305720")


def day_grid(conn, day: str, codes: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """호가를 초 격자에 채운다(직전 값 유지). 95번 `paths` 와 같은 모양의 bid/ask 만 만든다."""
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    qt = pd.read_sql(
        "select code, ts, bid1, ask1 from quotes where ts >= %s and ts < %s and code = any(%s) order by code, ts",
        conn, params=(day, nxt, list(codes)))
    out: dict[str, dict[str, np.ndarray]] = {}
    for code, g in qt.groupby("code"):
        sec = (g.ts.dt.hour * 3600 + g.ts.dt.minute * 60 + g.ts.dt.second).to_numpy() - T0
        ok = (sec >= 0) & (sec < N)
        if ok.sum() < 1000:
            continue
        bid = np.full(N, np.nan)
        ask = np.full(N, np.nan)
        bid[sec[ok]] = g.bid1.to_numpy(float)[ok]
        ask[sec[ok]] = g.ask1.to_numpy(float)[ok]
        s = pd.Series(bid).ffill().to_numpy()
        a = pd.Series(ask).ffill().to_numpy()
        out[code] = {"bid": s, "ask": a}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--cost-bp", type=float, default=3.0, help="왕복 비용(bp). ETF 는 증권거래세가 없다")
    a = ap.parse_args()
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    days = a.days or [pd.Timestamp.today().date().isoformat()]
    C.COST = a.cost_bp / 1e4                       # 95번과 같은 오라클 함수를 비용만 바꿔 쓴다
    acc = {h: {"tk": [], "mk": [], "mp": []} for h in C.HORIZONS}
    spreads = []
    for day in days:
        grid = day_grid(conn, day, ETFS)
        print(f"{day}: 종목 {len(grid)}")
        for code, p in grid.items():
            bid, ask = p["bid"], p["ask"]
            good = (bid > 0) & (ask > 0)
            if good.sum() < 1000:
                continue
            spreads.append(np.median((ask[good] / bid[good] - 1) * 1e4))
            e_fill = C.fill_second(bid)
            for h in C.HORIZONS:
                tk, mk = C.oracle(bid, ask, h)
                sl = slice(C.LO, C.HI)
                ok = np.isfinite(tk[sl]) & np.isfinite(mk[sl])
                acc[h]["tk"].append(tk[sl][ok])
                acc[h]["mk"].append(mk[sl][ok])
                acc[h]["mp"].append(C.maker_pessimistic(bid, e_fill, C.window_max(ask, h))[sl][ok])
    if not spreads:
        print("ETF 호가 데이터가 없다 — 수집 일수를 더 쌓아야 한다")
        return 2
    print(f"\nETF 스프레드 중앙값 {np.median(spreads):.2f}bp · 왕복 비용 가정 {a.cost_bp:.2f}bp")
    out = {"days": days, "cost_bp": a.cost_bp, "spread_bp_median": float(np.median(spreads)), "horizons": {}}
    for h in C.HORIZONS:
        tk = np.concatenate(acc[h]["tk"]) * 1e4
        mk = np.concatenate(acc[h]["mk"]) * 1e4
        mp = np.concatenate(acc[h]["mp"]) * 1e4
        mp = mp[np.isfinite(mp)]
        rec = {"n_sec": int(len(tk)),
               "taker_median": float(np.median(tk)), "taker_frac_pos": float((tk > 0).mean()),
               "maker_median": float(np.median(mk)), "maker_frac_pos": float((mk > 0).mean()),
               "maker_pess_median": float(np.median(mp)) if len(mp) else None,
               "maker_pess_frac_pos": float((mp > 0).mean()) if len(mp) else None}
        out["horizons"][str(h)] = rec
        pess = (f" · 메이커비관 {rec['maker_pess_median']:+7.1f}bp({rec['maker_pess_frac_pos']:.0%})"
                if rec["maker_pess_median"] is not None else "")
        print(f"H={h:>4}s n={rec['n_sec']:,}  테이커 {rec['taker_median']:+7.1f}bp({rec['taker_frac_pos']:.0%}) · "
              f"메이커 {rec['maker_median']:+7.1f}bp({rec['maker_frac_pos']:.0%}){pess}")
    Path("data").mkdir(exist_ok=True)
    Path("data/ceiling95_etf.json").write_text(json.dumps(out, indent=1))
    print("\n→ data/ceiling95_etf.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
