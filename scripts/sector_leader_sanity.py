"""섹터 대장 선정 사전등록(62번) 측정.

지배 섹터(당일 신고가 종목 breadth>=3)의 등락률 1위 종목을 사서 10거래일 뒤
파는 것이, 같은 기간 유니버스 무조건부 수익률보다 나은지 잰다.

읽기 전용, DB만 읽는다, 실주문 없음.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv, load_stock_master

BREADTH_MIN = 3
NEWHIGH_WINDOW = 250
HOLD_DAYS = 10
TV_FLOOR = 5000.0   # 50억, 백만원 단위 (daily_bars_adjusted 관행)
COST = 0.0034
COST_HI = 0.0055


def _describe(tag: str, r: pd.Series) -> None:
    w, l = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / l.mean()) if len(w) and len(l) and l.mean() != 0 else np.nan
    print(f"  {tag}: n={len(r):>5} 평균 {r.mean():+.2%} net34 {r.mean()-COST:+.2%} "
          f"net55 {r.mean()-COST_HI:+.2%} 중앙 {r.median():+.2%} 승률 {(r>0).mean():.1%} "
          f"손익비 {pl:.2f}")


def main() -> int:
    c = connect()
    daily = load_ohlcv(c, "2015-01-01", "2026-08-28")
    master = load_stock_master(c)
    print(f"일봉 {len(daily):,}행 종목 {daily['code'].nunique()}", file=sys.stderr)

    sector = master.loc[master["sector"].notna(), ["code", "sector"]]
    d = daily.merge(sector, on="code", how="inner")
    print(f"섹터 있는 종목만: {len(d):,}행 종목 {d['code'].nunique()} 섹터 {d['sector'].nunique()}",
          file=sys.stderr)

    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", sort=False)["close"]
    d["prior_high"] = g.transform(lambda s: s.shift(1).rolling(NEWHIGH_WINDOW, min_periods=NEWHIGH_WINDOW).max())
    d["new_high"] = d["close"] >= d["prior_high"]
    d["prev_close"] = g.shift(1)
    d["ret_day"] = d["close"] / d["prev_close"] - 1
    d["fwd_close"] = d.groupby("code", sort=False)["close"].shift(-HOLD_DAYS)
    d["ret_fwd"] = d["fwd_close"] / d["close"] - 1
    d = d.dropna(subset=["prior_high", "prev_close"])

    # 섹터 breadth: 그 섹터에서 당일 신고가 찍은 종목 수
    breadth = d[d["new_high"]].groupby(["date", "sector"]).size().rename("breadth")
    d = d.join(breadth, on=["date", "sector"])
    d["breadth"] = d["breadth"].fillna(0)

    dominant_days = d[(d["new_high"]) & (d["breadth"] >= BREADTH_MIN)]
    print(f"지배 섹터-일 신고가 이벤트: {len(dominant_days):,}건, "
          f"섹터-일 조합 {dominant_days.groupby(['date','sector']).ngroups:,}", file=sys.stderr)

    leaders = (dominant_days.sort_values(["date", "sector", "ret_day"], ascending=[True, True, False])
                             .groupby(["date", "sector"], sort=False).head(1))
    leaders = leaders[leaders["trade_value"] >= TV_FLOOR]
    leaders = leaders.dropna(subset=["ret_fwd"])
    print(f"대장 선정(유동성 50억+, 라벨 있음) {len(leaders):,}건", file=sys.stderr)

    if len(leaders) < 30:
        print("표본 부족. 종료.")
        return 1

    baseline = d.dropna(subset=["ret_fwd"])
    baseline = baseline[baseline["date"].isin(leaders["date"].unique())]

    print(f"\n=== 대장 선정 이벤트 {len(leaders):,}건 (breadth>={BREADTH_MIN}) ===")
    _describe("대장 매수", leaders["ret_fwd"])
    _describe("기준선(같은 날짜, 섹터무관 전체)", baseline["ret_fwd"])
    _describe("대장 상위1%제거", leaders["ret_fwd"][leaders["ret_fwd"] <= leaders["ret_fwd"].quantile(0.99)])

    leaders = leaders.copy()
    leaders["year"] = pd.to_datetime(leaders["date"]).dt.year
    print("\n연도별:")
    neg_years = 0
    for y, gy in leaders.groupby("year"):
        net = gy["ret_fwd"].mean() - COST
        neg_years += net < 0
        print(f"  {y}: n={len(gy):>4} net34 {net:+.2%}")

    r = leaders["ret_fwd"]
    r_top = r[r <= r.quantile(0.99)]
    base_net = baseline["ret_fwd"].mean() - COST
    cond = {
        "1 net34>0": r.mean() - COST > 0,
        "2 음수해<=2": neg_years <= 2,
        "3 상위1%제거 net34>0": r_top.mean() - COST > 0,
        "4 승률>50%": (r > 0).mean() > 0.5,
        "5 기준선보다 나음": (r.mean() - COST) > base_net,
    }
    print(f"\n기준선 net34 {base_net:+.2%}")
    print("채택 5조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    print(f"판정: {'채택(추가 조사)' if all(cond.values()) else '기각'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
