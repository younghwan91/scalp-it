"""상따 당일청산형 사전등록(59번) 측정.

06번 문서 5차 확정(verify_sangtta.py)의 이벤트 정의를 그대로 재사용하고
청산만 바꾼다: 15:20 이전 마지막 분봉 종가(잠기지 않았으면) 또는 익일 시가
(잠겼으면 — 상한가라 당일 매도 불가). build_events()는 수정하지 않는다.

읽기 전용, DB만 읽는다, 실주문 없음.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv
from scalp_it.intraday.loader import available_codes, load_minute_bars
from verify_sangtta import (
    build_events, _full_codes, _describe,
    TRIGGER, TV_FLOOR, COST, COST_HI, LIMIT_AT_ENTRY,
)

CLOSE_CUTOFF = pd.Timestamp("15:20").time()


def add_sameday_exit(ev: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """이벤트에 당일청산형 청산가·잠김여부를 붙인다. ev·bars 는 수정하지 않는다."""
    b = bars[["code", "ts", "close"]].copy()
    b["date"] = b["ts"].dt.normalize()
    b = b[b["ts"].dt.time < CLOSE_CUTOFF]
    last = (b.sort_values(["code", "date", "ts"])
              .groupby(["code", "date"], sort=False).tail(1)
              .set_index(["code", "date"])["close"].rename("close_1519"))
    ev = ev.join(last, on=["code", "date"])
    ev["locked_1519"] = ev["close_1519"] >= ev["prev_close"] * LIMIT_AT_ENTRY
    ev["exit_sameday"] = np.where(ev["locked_1519"], ev["n_open"], ev["close_1519"])
    ev["ret_sameday"] = (ev["exit_sameday"] - ev["entry"]) / ev["entry"]
    return ev.dropna(subset=["ret_sameday"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-only", action="store_true",
                     help="분봉이 --full-start~--full-end 를 덮는 종목만")
    ap.add_argument("--full-start", default="2025-08-01")
    ap.add_argument("--full-end", default="2026-08-13")
    ap.add_argument("--cache", default=None,
                     help="이벤트 테이블 pickle 경로 (재실행 시간 절약)")
    args = ap.parse_args()
    if args.cache and os.path.exists(args.cache):
        ev = pd.read_pickle(args.cache)
        print(f"캐시에서 이벤트 {len(ev):,}건 로드: {args.cache}", file=sys.stderr)
        return _analyze(ev)

    c = connect()
    codes = available_codes(c)
    print(f"분봉 보유 {len(codes)} 종목", file=sys.stderr)
    if args.full_only:
        codes = _full_codes(c, args.full_start, args.full_end)
        print(f"완비({args.full_start}~{args.full_end}) 종목만: {len(codes)}", file=sys.stderr)
    daily = load_ohlcv(c, "2025-06-01", "2026-08-21")
    daily = daily[daily["code"].isin(codes)]
    print(f"일봉 {len(daily):,} 행", file=sys.stderr)

    BATCH = 40
    parts = []
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i + BATCH]
        bars = load_minute_bars(c, "2025-08-01", "2026-08-21", codes=chunk)
        bars_ev = bars.rename(columns={"trade_value": "trade_value_bar"})
        sub = build_events(bars_ev, daily[daily["code"].isin(chunk)])
        if len(sub):
            sub = add_sameday_exit(sub, bars)
            parts.append(sub)
        print(f"  배치 {i//BATCH+1}/{(len(codes)+BATCH-1)//BATCH}: "
              f"{len(bars):,}행 -> 이벤트 {len(sub):,}", file=sys.stderr)
        del bars, bars_ev
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if args.cache:
        ev.to_pickle(args.cache)
    return _analyze(ev)


def _analyze(ev: pd.DataFrame) -> int:
    print(f"\n=== 상따 당일청산형 이벤트 {len(ev):,}건 ===")
    if len(ev) < 200:
        print("표본 부족. 종료.")
        return 1

    n_locked = int(ev["locked_1519"].sum())
    print(f"  잠김(강제 오버나이트) {n_locked:,}건 ({ev['locked_1519'].mean():.1%})  "
          f"당일매도 가능 {len(ev)-n_locked:,}건")

    print("\n  블렌드(잠기면 익일시가, 아니면 15:19 종가):")
    _describe("전체", ev["ret_sameday"])
    _describe("상위1%제거", ev["ret_sameday"][ev["ret_sameday"] <= ev["ret_sameday"].quantile(0.99)])
    print("\n  당일 매도분만(잠기지 않은 서브셋 — 이번 가설의 핵심):")
    sold = ev.loc[~ev["locked_1519"], "ret_sameday"]
    _describe("당일매도", sold)
    print("\n  잠긴 서브셋만(참고 — 기존 오버나이트 기각 결과와 정의상 동일해야 함):")
    _describe("잠김", ev.loc[ev["locked_1519"], "ret_sameday"])
    print("\n  참고: 06번 확정 전량 오버나이트 결과 net34 -2.09% (n=1,809) 와 비교")

    ev = ev.sort_values("date").reset_index(drop=True)
    folds = np.array_split(np.arange(len(ev)), 3)
    print("\n  시간순 3폴드:")
    neg_folds = 0
    for i, f in enumerate(folds, 1):
        r = ev.loc[f, "ret_sameday"]
        net = r.mean() - COST
        neg_folds += net < 0
        print(f"    폴드{i} n={len(f):>4} net34 {net:+.2%} "
              f"({ev.loc[f,'date'].min().date()}~{ev.loc[f,'date'].max().date()})")

    r_all = ev["ret_sameday"]
    r_top = r_all[r_all <= r_all.quantile(0.99)]
    cond = {
        "1 블렌드 net34>0": r_all.mean() - COST > 0,
        "2 음수폴드<=1": neg_folds <= 1,
        "3 상위1%제거 net34>0": r_top.mean() - COST > 0,
        "4 당일매도분만 net34>0": sold.mean() - COST > 0 if len(sold) else False,
        "5 블렌드 승률>50%": (r_all > 0).mean() > 0.5,
    }
    print("\n  채택 5조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    print(f"  판정: {'채택(추가 조사)' if all(cond.values()) else '기각'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
