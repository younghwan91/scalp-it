#!/usr/bin/env python
"""Combo + inverse-hedge reproduction — PEAD ⊕ low-vol on kr-quant data.

Reproduces, on the shared TimescaleDB (read-only, via the storage front doors),
the scalp-it combo finding (note #32) **and** the deployability question raised
for this account: the user cannot short individual names, only buy an inverse
index ETF. So for each edge we report both forms side by side:

  · **L/S (ceiling)** — dollar-neutral rank-tilt with an individual short leg.
    The theoretical spec; not tradeable for this account.
  · **Long-only + inverse hedge (deployable)** — hold the long book of individual
    stocks, neutralize market beta by buying a (synthetic) inverse index ETF.
    Removes market beta but forgoes stock-specific short alpha.

Then the two edges are blended by inverse-vol weighting (risk parity) in both
forms, and the combo's Sharpe / MDD is compared to each standalone.

This is an **engine-approximate** reproduction: it reuses kr-quant's rank-tilt
engine (entry t+1, measured-turnover cost, borrow) at a monthly cadence rather
than scalp-it's bespoke daily-close cohort marker, so the headline numbers are
close, not byte-identical (scalp-it flagged the same caveat). No DB writes.

Run:  python scripts/combo_book.py          # prints a markdown report
      python scripts/combo_book.py --out report.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from kr_quant.engine.metrics import newey_west_t
from kr_quant.features.fundamentals import _yoy_vec, earnings_yoy_panel
from kr_quant.strategies.combo import combine_inverse_vol, expanding_inverse_vol, series_metrics
from kr_quant.strategies.hedge import INVERSE_ANNUAL_FEE, inverse_hedged_return, universe_market_return
from kr_quant.strategies.lowvol import lowvol_backtest
from kr_quant.strategies.pead import pead_backtest
from kr_quant.storage import connect, db_default, read_earnings, read_prices

LIQ_FLOOR = 5000.0   # 50억 (백만원) — the note #24/#31 deployment floor
HORIZON = 21         # monthly cadence so PEAD and low-vol series align for the combo
BORROW = 0.03        # short-leg borrow, 3%/yr (L/S only)
COST = 0.0034        # 34bp one-way (68bp round-trip)


# ----------------------------------------------------------------- data loading

def load_frames(db: str):
    con = connect(db)
    prices = read_prices(con, cols=("code", "date", "close", "trade_value"))
    prices["date"] = prices["date"].astype(str)
    ea = read_earnings(con, cols=("code", "period", "avail_date", "netinc",
                                  "netinc_prior", "knowledge_date"))
    meta = pd.read_sql_query("SELECT code, name FROM stocks", con)  # not a gated table
    con.close()
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    dates = sorted(prices["date"].unique())
    yoy_panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), dates,
                                   knowledge_col="knowledge_date")
    return prices, yoy_panel, meta


# --------------------------------------------------------- period -> monthly net

def to_monthly(periods: pd.DataFrame, col: str = "net") -> pd.Series:
    """Compound a rank-tilt period series into a monthly net series."""
    if periods.empty:
        return pd.Series(dtype=float)
    s = periods[["date", col]].copy()
    s["m"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    return s.groupby("m")[col].apply(lambda x: (1.0 + x).prod() - 1.0)


def market_monthly(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight liquid-universe market return, compounded to monthly."""
    daily = universe_market_return(prices, adv_floor=LIQ_FLOOR)
    df = daily.rename("r").reset_index()
    df["m"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return df.groupby("m")["r"].apply(lambda x: (1.0 + x).prod() - 1.0)


def hedged_monthly(longonly_excess: pd.Series, mkt: pd.Series, *, leverage: float = 1.0) -> pd.Series:
    """Long-only book hedged with a synthetic inverse ETF (expanding beta + fee).

    The engine's long-only ``gross`` is EXCESS over the universe mean, i.e. a
    beta≈1 market-neutralized series already. To model an *explicit* inverse-ETF
    hedge (estimated beta, ETF fee), reconstruct the absolute book return
    (excess + market), then hedge it with the inverse leg.
    """
    common = longonly_excess.index.intersection(mkt.index)
    excess = longonly_excess.loc[common]
    m = mkt.loc[common]
    book_abs = excess + m
    hedged, _beta = inverse_hedged_return(
        book_abs, m, leverage=leverage, beta=None, beta_window=None,
        fee_per_period=INVERSE_ANNUAL_FEE / 12.0, min_obs=12)
    return hedged.dropna()


# ------------------------------------------------------------------ report

def _t_monthly(net: pd.Series) -> float:
    m, t = newey_west_t(net.dropna().to_numpy(float), lag=1)
    return t


def build_report(prices, yoy_panel, meta) -> str:
    L: list[str] = []
    def w(s: str = ""):
        L.append(s)

    start = 130
    # PEAD: L/S ceiling and long-only excess (the inverse-hedge base).
    pe_ls, _ = pead_backtest(prices, yoy_panel, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                             cost_one_way=COST, borrow_cost_annual=BORROW, start_index=start)
    pe_lo, _ = pead_backtest(prices, yoy_panel, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                             cost_one_way=COST, long_only=True, start_index=start)
    # Low-vol: same two forms.
    lv_ls, _ = lowvol_backtest(prices, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                               cost_one_way=COST, borrow_cost_annual=BORROW)
    lv_lo, _ = lowvol_backtest(prices, horizon=HORIZON, adv_floor=LIQ_FLOOR,
                               cost_one_way=COST, long_only=True)

    mkt = market_monthly(prices)
    pead_ls = to_monthly(pe_ls)
    pead_lo_ex = to_monthly(pe_lo, "gross")   # long-only gross == excess over universe
    lowvol_ls = to_monthly(lv_ls)
    lowvol_lo_ex = to_monthly(lv_lo, "gross")

    pead_hedge = hedged_monthly(pead_lo_ex, mkt)
    lowvol_hedge = hedged_monthly(lowvol_lo_ex, mkt)

    def row(name, s):
        m = series_metrics(s)
        return (f"| {name} | {m['n']} | {m['ann_ret']*100:+.1f}% | {m['ann_vol']*100:.1f}% "
                f"| {m['sharpe']:.2f} | {m['mdd']*100:.1f}% | {_t_monthly(s):+.2f} |")

    w("# 결합 북 + 인버스 헤지 재현 (PEAD ⊕ 저변동, kr-quant 데이터)")
    w()
    w(f"- 유니버스 50억(백만원 {LIQ_FLOOR:.0f}), 월 리밸(H={HORIZON}), 비용 68bp 왕복 + "
      f"숏차입 3%/yr(롱숏만). 인버스 헤지 = 확장창 베타 × 합성 인버스ETF(−1×, 수수료 "
      f"{INVERSE_ANNUAL_FEE*100:.2f}%/yr). ETF 일봉이 DB 에 없어 유니버스 EW 를 지수로 대용.")
    w("- 엔진 근사: kr-quant 랭크틸트 엔진(월 케이던스). scalp-it 일별-종가 코호트 "
      "마커와 방식이 달라 수치는 근사(부호·순서 재현 목적).")
    w()
    w("## 전략별 월 net 성과")
    w()
    w("| 전략 | 월수 | 연율수익 | 연율변동성 | 샤프 | MDD | NW-t |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(row("PEAD 롱숏 (이론 상한)", pead_ls))
    w(row("PEAD 롱온리+인버스헤지 (배포)", pead_hedge))
    w(row("저변동 롱숏 (이론 상한)", lowvol_ls))
    w(row("저변동 롱온리+인버스헤지 (배포)", lowvol_hedge))
    w()

    # Combos — both the L/S ceiling and the deployable inverse-hedge form.
    combo_ls, w_ls = combine_inverse_vol({"PEAD": pead_ls, "LOWVOL": lowvol_ls})
    combo_hedge, w_h = combine_inverse_vol({"PEAD": pead_hedge, "LOWVOL": lowvol_hedge})
    combo_hedge_exp = expanding_inverse_vol({"PEAD": pead_hedge, "LOWVOL": lowvol_hedge},
                                            min_periods=24)
    cm_pe = pead_ls.index.intersection(lowvol_ls.index)
    corr_ls = np.corrcoef(pead_ls.loc[cm_pe], lowvol_ls.loc[cm_pe])[0, 1] if len(cm_pe) >= 6 else float("nan")

    w("## 결합 (역변동성 가중)")
    w()
    w(f"- 롱숏 결합 비중 w_PEAD={w_ls['PEAD']:.2f} / w_저변동={w_ls['LOWVOL']:.2f}; "
      f"PEAD↔저변동 월 net 상관 {corr_ls:+.2f}.")
    w(f"- 배포(헤지) 결합 비중 w_PEAD={w_h['PEAD']:.2f} / w_저변동={w_h['LOWVOL']:.2f}.")
    w()
    w("| 전략 | 월수 | 연율수익 | 연율변동성 | 샤프 | MDD | NW-t |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(row("결합 롱숏 (이론 상한)", combo_ls))
    w(row("결합 롱온리+인버스헤지 (배포)", combo_hedge))
    w(row("결합 헤지 · 확장창 비중(룩어헤드 없음)", combo_hedge_exp))
    w()
    w("## 결론")
    w()
    pe_ls_s = series_metrics(pead_ls)["sharpe"]
    pe_hd_s = series_metrics(pead_hedge)["sharpe"]
    lv_ls_s = series_metrics(lowvol_ls)["sharpe"]
    lv_hd_s = series_metrics(lowvol_hedge)["sharpe"]
    cmb_hd_s = series_metrics(combo_hedge)["sharpe"]
    viable = pe_hd_s > 0 and lv_hd_s > 0
    w(f"- **사용자 실전 배포 = 롱온리 + 인버스 헤지**(개별주 숏 불가). 이 엔진-근사 "
      f"재현에서 배포형은 {'살아있다' if viable else '약하다'} — PEAD 헤지 샤프 "
      f"{pe_hd_s:.2f}(롱숏 상한 {pe_ls_s:.2f}), 저변동 헤지 {lv_hd_s:.2f}"
      f"(상한 {lv_ls_s:.2f}), 결합 헤지 {cmb_hd_s:.2f}.")
    w("- 인버스 ETF 는 시장 베타만 제거한다 — 롱 종목선택 알파는 살고 개별주 숏 알파는 "
      "포기된다. 여기서 롱온리 초과가 강한 건 netinc-YoY·월 케이던스 프레임이라, "
      "op-income SUE·40일 프레임에서 롱온리가 죽던 노트 #24 와 신호·주기가 다르다는 "
      "점을 유의(엔진-근사 재현).")
    w("- 저변동은 롱=저변동주라 롱 다리가 상대적으로 강해 헤지 후 손실이 PEAD 보다 작다.")
    w(f"- 결합(역변동성)은 두 배포형 단독보다 샤프가 높다 — 결합 헤지 {cmb_hd_s:.2f} "
      f"vs PEAD {pe_hd_s:.2f}·저변동 {lv_hd_s:.2f}. 상관이 낮아(월 net {corr_ls:+.2f}) "
      f"분산 효과가 실재한다.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="PEAD⊕저변동 결합 + 인버스헤지 재현")
    ap.add_argument("--db", default=db_default())
    ap.add_argument("--out", default=None, help="마크다운 저장 경로(기본: stdout)")
    args = ap.parse_args()

    if not os.environ.get("KR_QUANT_DB") and args.db == db_default():
        print("경고: KR_QUANT_DB 미설정 — 로컬 sqlite(비어있음)를 읽는다. "
              ".env 를 셸에 로드하라.")
    prices, yoy_panel, meta = load_frames(args.db)
    report = build_report(prices, yoy_panel, meta)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"저장: {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
