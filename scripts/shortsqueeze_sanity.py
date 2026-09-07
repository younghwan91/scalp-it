"""숏스퀴즈 셋업 사전등록(64번) 측정.

52주 신고가 + 외인·기관 3일 연속 순매수 -> 익일 시가 매수, 당일 종가 매도.
읽기 전용, DB만 읽는다, 실주문 없음.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv, load_supply_demand

TV_FLOOR = 5000.0          # 50억, daily_bars_adjusted.trade_value 는 백만원
COST = 0.0034
COST_HI = 0.0055
LOOKBACK = 250             # 52주 신고가
CONSEC = 3                 # 연속 순매수 일수
START, END = "2022-11-01", "2026-08-21"   # 250거래일(52주) lookback 여유 위해 14개월 앞당김
SD_START = "2024-01-01"
LIMIT_SUSPECT = 0.295


def build_events() -> pd.DataFrame:
    c = connect()
    daily = load_ohlcv(c, START, END)
    sd = load_supply_demand(c, SD_START, END)

    d = daily.sort_values(["code", "date"]).copy()
    d["is_high250"] = d.groupby("code")["close"].transform(
        lambda s: s >= s.rolling(LOOKBACK, min_periods=LOOKBACK).max())
    d["next_open"] = d.groupby("code")["open"].shift(-1)
    d["next_close"] = d.groupby("code")["close"].shift(-1)
    d["next_ret_day"] = (d["next_close"] - d["next_open"]) / d["next_open"]

    s = sd.sort_values(["code", "date"]).copy()
    s["net_buy"] = (s["foreign_"] > 0) & (s["institution"] > 0)
    s["consec"] = (s.groupby("code")["net_buy"]
                    .transform(lambda x: x.groupby((~x).cumsum()).cumcount() + 1) * s["net_buy"])
    s["trigger_sd"] = s["consec"] >= CONSEC

    m = d.merge(s[["code", "date", "trigger_sd"]], on=["code", "date"], how="inner")
    m = m[(m["date"] >= SD_START) & m["is_high250"] & m["trigger_sd"]
          & (m["trade_value"] >= TV_FLOOR)]
    ev = m.dropna(subset=["next_open", "next_close"]).copy()
    ev["ret"] = ev["next_ret_day"]
    ev["limit_suspect"] = ev["next_ret_day"] >= LIMIT_SUSPECT
    return ev


def _describe(tag: str, r: pd.Series) -> None:
    if len(r) == 0:
        print(f"  {tag}: n=0")
        return
    w, l = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / l.mean()) if len(w) and len(l) and l.mean() != 0 else np.nan
    print(f"  {tag}: n={len(r):>4} 평균 {r.mean():+.2%} net34 {r.mean()-COST:+.2%} "
          f"net55 {r.mean()-COST_HI:+.2%} 중앙 {r.median():+.2%} 승률 {(r>0).mean():.1%} "
          f"손익비 {pl:.2f}")


def main() -> int:
    ev = build_events()
    print(f"=== 숏스퀴즈 이벤트 {len(ev):,}건, 종목 {ev['code'].nunique()}, "
          f"거래일 {ev['date'].nunique()} ===", file=sys.stderr)
    if len(ev) < 30:
        print("표본 부족(< 30건). 종료.")
        return 1

    print("\n전부 매수:")
    _describe("전체", ev["ret"])
    r_top = ev["ret"][ev["ret"] <= ev["ret"].quantile(0.99)]
    _describe("상위1%제거", r_top)
    ex = ev.loc[~ev["limit_suspect"], "ret"]
    _describe("상한가의심제외", ex)
    print(f"  상한가 의심(다음날 종가등락 >= 29.5%) {ev['limit_suspect'].sum()}건 "
          f"({ev['limit_suspect'].mean():.1%})")

    ev = ev.sort_values("date")
    ev["quarter"] = ev["date"].dt.to_period("Q")
    print("\n분기별:")
    neg_q = 0
    for q, g in ev.groupby("quarter"):
        net = g["ret"].mean() - COST
        neg_q += net < 0
        print(f"  {q} n={len(g):>4} net34 {net:+.2%}")
    n_q = ev["quarter"].nunique()

    cond = {
        "1 net34>0": ev["ret"].mean() - COST > 0,
        "2 음수분기<=3": neg_q <= 3,
        "3 상위1%제거 net34>0": r_top.mean() - COST > 0,
        "4 상한가의심제외 net34>0": (ex.mean() - COST > 0) if len(ex) else False,
        "5 승률>50%": (ev["ret"] > 0).mean() > 0.5,
    }
    print(f"\n채택 5조건 (총 {n_q}분기, 음수 {neg_q}): " +
          "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    print(f"판정: {'채택(추가 조사)' if all(cond.values()) else '기각'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
