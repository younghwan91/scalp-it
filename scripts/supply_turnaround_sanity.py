"""수급 턴어라운드 사전등록(61번) 측정.

이벤트: D-2,D-1 연속 2음봉(외인+기관 순매도) -> D 에 외인+기관 순매수가
D-2 순매도의 50%+ 재흡수, D 거래대금 >= 500억. 진입 D+1 시가, 청산
D+1 로부터 10거래일 후 종가.

읽기 전용, DB만 읽는다, 실주문 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv, load_supply_demand

TV_FLOOR = 50_000.0     # 500억 (daily_bars_adjusted.trade_value 백만원 단위)
HOLD_DAYS = 10          # D+1 시가 진입 후 10거래일째 종가 청산
COST = 0.0034
COST_HI = 0.0055
START, END = "2024-01-01", "2026-08-28"


def build_events() -> pd.DataFrame:
    c = connect()
    daily = load_ohlcv(c, "2023-12-01", END)
    sd = load_supply_demand(c, "2023-12-01", END)

    d = daily.sort_values(["code", "date"]).copy()
    sd = sd.sort_values(["code", "date"]).copy()
    sd["flow"] = sd["foreign_"] + sd["institution"]

    m = d.merge(sd[["code", "date", "flow"]], on=["code", "date"], how="inner")
    m = m.sort_values(["code", "date"]).reset_index(drop=True)
    g = m.groupby("code", sort=False)

    m["is_down"] = m["close"] < m["open"]
    m["flow_neg"] = m["flow"] < 0
    m["down2"] = m["is_down"] & m["flow_neg"]

    m["down2_lag1"] = g["down2"].shift(1)
    m["down2_lag2"] = g["down2"].shift(2)
    m["flow_lag2"] = g["flow"].shift(2)   # D-2 (첫 음봉) 순매도량

    trig = (
        m["down2_lag1"].fillna(False)
        & m["down2_lag2"].fillna(False)
        & (m["flow"] >= 0.5 * m["flow_lag2"].abs())
        & (m["trade_value"] >= TV_FLOOR)
        & (m["flow_lag2"] < 0)
    )
    ev = m.loc[trig].copy()

    d = d.reset_index(drop=True)
    g2 = d.groupby("code", sort=False)
    d["n1_open"] = g2["open"].shift(-1)
    d["exit_close"] = g2["close"].shift(-(HOLD_DAYS + 1))
    key = d.set_index(["code", "date"])[["n1_open", "exit_close", "trade_value"]].rename(
        columns={"trade_value": "tv_rank"})

    ev = ev.join(key, on=["code", "date"], rsuffix="_fwd")
    ev = ev.dropna(subset=["n1_open", "exit_close"])
    ev["ret"] = (ev["exit_close"] - ev["n1_open"]) / ev["n1_open"]
    return ev


def _describe(tag: str, r: pd.Series) -> None:
    w, l = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / l.mean()) if len(w) and len(l) and l.mean() != 0 else np.nan
    print(f"  {tag}: n={len(r):>4} 평균 {r.mean():+.2%} net34 {r.mean()-COST:+.2%} "
          f"net55 {r.mean()-COST_HI:+.2%} 중앙 {r.median():+.2%} 승률 {(r>0).mean():.1%} "
          f"손익비 {pl:.2f}")


def main() -> int:
    ev = build_events()
    print(f"=== 수급 턴어라운드 이벤트 {len(ev):,}건 / 종목 {ev['code'].nunique()} / "
          f"거래일 {ev['date'].nunique()} ===")
    if len(ev) < 30:
        print("표본 부족. 종료.")
        return 1

    _describe("전체", ev["ret"])
    r = ev["ret"]
    _describe("상위1%제거", r[r <= r.quantile(0.99)])

    ev2 = ev.sort_values("date").reset_index(drop=True)
    folds = np.array_split(np.arange(len(ev2)), 3)
    neg_folds = 0
    print("  시간순 3폴드:")
    for i, f in enumerate(folds, 1):
        rr = ev2.loc[f, "ret"]
        net = rr.mean() - COST
        neg_folds += net < 0
        print(f"    폴드{i} n={len(f):>4} net34 {net:+.2%} "
              f"({ev2.loc[f,'date'].min()}~{ev2.loc[f,'date'].max()})")

    tv_med = ev["tv_rank"].median()
    top_tv = ev.loc[ev["tv_rank"] >= tv_med, "ret"]
    _describe("거래대금 상위50%", top_tv)

    cond = {
        "1 net34>0": r.mean() - COST > 0,
        "2 음수폴드<=1": neg_folds <= 1,
        "3 상위1%제거 net34>0": r[r <= r.quantile(0.99)].mean() - COST > 0,
        "4 승률>50%": (r > 0).mean() > 0.5,
        "5 거래대금상위50% net34>0": top_tv.mean() - COST > 0,
    }
    print("\n  채택 5조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    print(f"  판정: {'채택(추가 조사)' if all(cond.values()) else '기각'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
