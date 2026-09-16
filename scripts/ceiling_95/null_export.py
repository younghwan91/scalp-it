"""95g번 매치드 널 입력 내보내기 — **+21.3bp 가 "딥에서 샀다"인지 "변동성 큰 순간이었다"인지** 가른다.

    ~/of80/.rlvenv/bin/python scripts/ceiling_95/null_export.py --horizon 300 --sample 200000

왜: 우리 대조군은 "같은 종목·같은 날의 무작위 초"다. 그건 **변동성 국면이 안 맞춰져 있다**.
딥 체결은 정의상 가격이 움직이는 순간에 일어나므로, 무작위 초보다 좋아 보이는 게 당연할 수 있다.
daytrade-it 의 매치드 널(day × 진입 분 × 전일 변동성 4분위 × 전일 거래대금 4분위, 둘 다 **개장 전**
정보로만 만든다)로 돌리면 그 둘이 갈린다. 결과가 96번의 사전 확률을 직접 바꾼다.

출력(그쪽 입력 규격): `code,date,entry_ts,exit_ts,ret_bp,tag` — `ret_bp` 는 **비용 차감 전(gross)**.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

T0 = 9 * 3600
CACHE = Path.home() / "of80" / "flow83"
TICK_BINS = ((0.0, 7.0, "tick_narrow"), (7.0, 12.0, "tick_mid"), (12.0, 1e9, "tick_wide"))


def _hhmmss(sec: int) -> str:
    s = sec + T0
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=300, help="보유 초")
    ap.add_argument("--sample", type=int, default=200_000, help="내보낼 행 수 상한(균등 추출)")
    ap.add_argument("--out", default="data/null_input_dipfill.csv")
    ap.add_argument("--seed", type=int, default=95)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ceiling95", Path(__file__).parent / "ceiling.py")
    C = importlib.util.module_from_spec(spec)
    sys.modules["ceiling95"] = C
    spec.loader.exec_module(C)

    rows: list[tuple] = []
    for f in sorted(glob.glob(str(CACHE / "*.pkl"))):
        day = Path(f).stem
        d = pickle.load(open(f, "rb"))
        for code, p in d["paths"].items():
            bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
            tick = p["tick"].astype(np.float64)
            mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)
            e = C.fill_second(bid)
            maxask = C.window_max(ask, a.horizon)
            r = C.maker_pessimistic(bid, e, maxask)          # 비용 차감 **후** 값이다
            idx = np.nonzero(np.isfinite(r))[0]
            idx = idx[(idx >= C.LO) & (idx < C.HI)]
            for t in idx:
                m = mid[t]
                if not (m > 0) or not (tick[t] > 0):
                    continue
                tb = tick[t] / m * 1e4
                tag = next(n for lo, hi, n in TICK_BINS if lo <= tb < hi)
                gross = (r[t] + C.COST) * 1e4                # 그쪽 규격은 gross 다
                rows.append((code, day, _hhmmss(int(e[t])),
                             _hhmmss(int(min(e[t] + a.horizon, len(bid) - 1))),
                             round(float(gross), 2), f"dipfill_{tag}"))
    rng = np.random.default_rng(a.seed)
    if len(rows) > a.sample:
        pick = rng.choice(len(rows), a.sample, replace=False)
        rows = [rows[i] for i in sorted(pick)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "date", "entry_ts", "exit_ts", "ret_bp", "tag"])
        w.writerows(rows)
    print(f"{len(rows):,}행 → {out}  (보유 {a.horizon}초, ret_bp 는 비용 차감 전)")
    for _lo, _hi, n in TICK_BINS:
        k = sum(1 for r in rows if r[5].endswith(n))
        if k:
            mean = float(np.mean([r[4] for r in rows if r[5].endswith(n)]))
            print(f"   {n:>12}: {k:,}행 · gross 평균 {mean:+.1f}bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
