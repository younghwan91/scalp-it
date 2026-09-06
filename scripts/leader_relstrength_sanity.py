"""대장 vs 2등주 상대강도 사전등록(63번) 측정.

지수 급락일(시장지수 일수익률 <= -1.0%)에 직전 20거래일 고점을 돌파하는
종목들을 섹터별로 묶어 당일 등락률 1위(대장) 대 2·3위(2·3등주)의 5거래일
forward 수익을 비교한다.

읽기 전용, DB만 읽는다, 실주문 없음.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv, load_stock_master

START, END = "2024-01-26", "2026-08-21"
INDEX_DROP = -0.01       # 지수 급락일: 당일 지수 수익률 <= -1.0%
BOX_WINDOW = 20          # 박스 상단 = 직전 20거래일 최고종가(당일 제외)
TV_FLOOR = 5000.0        # 50억, daily_bars_adjusted 백만원 단위
HORIZON = 5              # 진입 후 5거래일 뒤 종가 청산 (판정용)
HORIZON_REF = 10         # 참고 병기, 판정에 안 씀
COST = 0.0034
COST_HI = 0.0055


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
    c = connect()
    idx = pd.read_sql if False else None  # placeholder to avoid unused import lint

    cur = c.cursor()
    cur.execute(
        "select code, date, close from sector_index where code in ('001','101') "
        "and date >= %s and date <= %s order by code, date", (START, END))
    idx_rows = cur.fetchall()
    cur.close()
    index_df = pd.DataFrame(idx_rows, columns=["code", "date", "close"])
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values(["code", "date"])
    index_df["ret"] = index_df.groupby("code")["close"].pct_change()
    index_df["market"] = index_df["code"].map({"001": "KOSPI", "101": "KOSDAQ"})
    crash_days = {
        m: set(index_df.loc[(index_df["market"] == m) & (index_df["ret"] <= INDEX_DROP), "date"])
        for m in ("KOSPI", "KOSDAQ")
    }
    print(f"급락일(<=−1.0%) KOSPI {len(crash_days['KOSPI'])}일 / "
          f"KOSDAQ {len(crash_days['KOSDAQ'])}일", file=sys.stderr)

    master = load_stock_master(c)
    master = master[~master["is_delisted"]][["code", "market", "sector"]]
    master["market"] = master["market"].map({"거래소": "KOSPI", "코스닥": "KOSDAQ"})

    daily = load_ohlcv(c, "2023-11-01", END)  # 여유분 앞당겨 20일 박스 계산
    daily = daily.merge(master, on="code", how="inner")
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    g = daily.groupby("code", sort=False)
    daily["prior_high20"] = g["close"].transform(lambda s: s.shift(1).rolling(BOX_WINDOW, min_periods=BOX_WINDOW).max())
    daily["ret_day"] = g["close"].pct_change()
    daily["fwd5"] = g["close"].shift(-HORIZON) / daily["close"] - 1
    daily["fwd10"] = g["close"].shift(-HORIZON_REF) / daily["close"] - 1

    daily = daily[(daily["date"] >= START) & (daily["date"] <= END)]
    daily["is_crash"] = daily.apply(
        lambda r: r["date"] in crash_days.get(r["market"], set()), axis=1)
    daily = daily[daily["is_crash"]]
    daily = daily.dropna(subset=["prior_high20", "fwd5"])
    daily = daily[daily["trade_value"] >= TV_FLOOR]
    daily["breakout"] = daily["close"] > daily["prior_high20"]
    ev = daily[daily["breakout"]].copy()
    print(f"급락일 x 박스돌파 후보 {len(ev):,}건 (거래대금 50억+, 섹터 매핑됨)", file=sys.stderr)

    ev = ev.dropna(subset=["sector"])
    ev["rank"] = ev.groupby(["date", "sector"])["ret_day"].rank(ascending=False, method="first")
    grp_size = ev.groupby(["date", "sector"])["rank"].transform("count")
    ev = ev[grp_size >= 2]   # 대장-2등 비교가 가능한 그룹만

    leader = ev[ev["rank"] == 1]
    followers = ev[ev["rank"].isin([2, 3])]

    print(f"\n=== 대장(1위) {len(leader):,}건 vs 2·3등 {len(followers):,}건 ===")
    if len(leader) < 30:
        print("표본 부족(대장 그룹 n<30). 참고치로만 본다.")
    _describe("대장 fwd5", leader["fwd5"])
    _describe("대장 fwd5 상위1%제거", leader["fwd5"][leader["fwd5"] <= leader["fwd5"].quantile(0.99)])
    _describe("2·3등 fwd5", followers["fwd5"])
    _describe("대장 fwd10(참고)", leader["fwd10"].dropna())

    leader_s = leader.sort_values("date")
    folds = np.array_split(np.arange(len(leader_s)), 3) if len(leader_s) >= 3 else []
    neg_folds = 0
    print("\n  시간순 3폴드(대장 fwd5):")
    for i, f in enumerate(folds, 1):
        r = leader_s.iloc[f]["fwd5"]
        net = r.mean() - COST
        neg_folds += net < 0
        lo, hi = (leader_s.iloc[f]["date"].min(), leader_s.iloc[f]["date"].max()) if len(f) else (None, None)
        print(f"    폴드{i} n={len(f):>4} net34 {net:+.2%}  ({lo}~{hi})")

    lead_r = leader["fwd5"]
    lead_top = lead_r[lead_r <= lead_r.quantile(0.99)]
    cond = {
        "1 대장 net34>0": lead_r.mean() - COST > 0,
        "2 대장>2,3등": lead_r.mean() > followers["fwd5"].mean(),
        "3 음수폴드<=1": neg_folds <= 1,
        "4 상위1%제거 net34>0": lead_top.mean() - COST > 0,
        "5 대장 승률>50%": (lead_r > 0).mean() > 0.5,
    }
    print("\n  채택 5조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    print(f"  판정: {'채택(추가 조사)' if all(cond.values()) else '기각'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
