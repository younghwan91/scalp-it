"""PEAD 이벤트 구성 + 규모 집계 (사전등록 23번).

읽기 전용. earnings.avail_date 를 정보 도착 시점(PIT)으로 사용한다.
- 시그널: 영업이익 YoY 서프라이즈 = (op_income - op_income_prior)/|op_income_prior|.
  earnings.op_income_prior 는 '직전연도 동일분기'(YoY)임을 확인함(2017Q1 prior == 2016Q1 값).
- 유니버스: avail_date 시점 거래중 + 직전 20거래일 평균 거래대금 >= 500(백만원=5억) + 스팩 제외 + 상폐주 포함.
- 진입: avail_date 익일(해당 종목 실제 거래일) 시가. 보유 +20/+40/+60 거래일.

이 파일은 verify_pead.py 가 build_events() 로 재사용한다.
DB: 환경변수 KR_QUANT_DB.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2

LIQ_FLOOR_MWON = 500  # 직전 20거래일 평균 거래대금 하한 (백만원) = 5억원
LIQ_WINDOW = 20
HORIZONS = (20, 40, 60)


def _conn():
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        raise SystemExit("KR_QUANT_DB 환경변수 없음")
    return psycopg2.connect(dsn)


def load_raw():
    c = _conn()
    earn = pd.read_sql(
        "select code, period, avail_date, op_income, op_income_prior, "
        "netinc, netinc_prior, revenue, revenue_prior from earnings",
        c, parse_dates=["avail_date"],
    )
    bars = pd.read_sql(
        "select code, date, open, close, trade_value from daily_bars_adjusted "
        "order by code, date",
        c, parse_dates=["date"],
    )
    raw = pd.read_sql(
        "select code, date, open, close from daily_bars order by code, date",
        c, parse_dates=["date"],
    )
    spac_stocks = pd.read_sql(
        "select code from stocks where name like '%%스팩%%' "
        "or name like '%%기업인수목적%%'", c)
    spac_del = pd.read_sql(
        "select code from delisted_stocks where name like '%%스팩%%' "
        "or name like '%%기업인수목적%%'", c)
    c.close()
    spac = set(spac_stocks["code"]) | set(spac_del["code"])
    return earn, bars, raw, spac


def build_events(earn=None, bars=None, raw=None, spac=None, horizons=HORIZONS):
    """이벤트 프레임 반환. 각 행 = (code, period, avail_date, 시그널, 진입/청산 가격, 라벨, 진단 플래그)."""
    if earn is None:
        earn, bars, raw, spac = load_raw()

    # 종목별 (수정주가) 배열 사전
    adj = {}
    for code, g in bars.groupby("code", sort=False):
        g = g.sort_values("date")
        adj[code] = (
            g["date"].values,
            g["open"].values.astype(float),
            g["trade_value"].values.astype(float),
        )
    # 종목별 원주가 (상한가 판정: 익일 시가 / 전일 종가)
    rawd = {}
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("date")
        rawd[code] = (
            g["date"].values,
            g["open"].values.astype(float),
            g["close"].values.astype(float),
        )

    rows = []
    for r in earn.itertuples(index=False):
        code = r.code
        avail = np.datetime64(pd.Timestamp(r.avail_date), "ns")
        oi, oip = r.op_income, r.op_income_prior
        has_sig = (oi is not None) and (oip is not None) and pd.notna(oi) and pd.notna(oip) and oip != 0
        if code not in adj:
            continue
        dates, opens, tvs = adj[code]
        # 진입 = avail_date 초과 첫 거래일
        ei = int(np.searchsorted(dates, avail, side="right"))
        if ei >= len(dates):
            continue
        entry_open = opens[ei]
        # 유동성: 진입 직전 20거래일(=avail_date 까지 존재하는 바) 평균 거래대금
        lo = max(0, ei - LIQ_WINDOW)
        if ei - lo < LIQ_WINDOW:
            liq = np.nan
        else:
            liq = float(np.nanmean(tvs[lo:ei]))
        # 라벨: +N 거래일 뒤 익일 시가 대비 (진입 시가 -> exit 시가)
        labels = {}
        for h in horizons:
            xi = ei + h
            labels[h] = (opens[xi] / entry_open - 1.0) if (xi < len(dates) and entry_open > 0) else np.nan

        # 상한가 판정 (원주가): 진입일 시가가 전일종가 대비 +29% 이상이면 체결불가 후보
        entry_limit_up = False
        avail_day_limit_up = False
        if code in rawd:
            rd, ro, rc = rawd[code]
            rei = int(np.searchsorted(rd, avail, side="right"))
            if 0 < rei < len(rd):
                prev_close = rc[rei - 1]
                if prev_close > 0 and ro[rei] / prev_close >= 1.29:
                    entry_limit_up = True
            # avail_date 당일(<=avail) 마지막 거래일 종가가 상한가 마감이었는가(오염 진단)
            ai = int(np.searchsorted(rd, avail, side="right")) - 1
            if ai > 0:
                pc = rc[ai - 1]
                if pc > 0 and rc[ai] / pc >= 1.29:
                    avail_day_limit_up = True

        sue_raw = (float(oi) - float(oip)) / abs(float(oip)) if has_sig else np.nan
        rows.append({
            "code": code, "period": r.period, "avail_date": pd.Timestamp(r.avail_date),
            "sue_raw": sue_raw, "entry_open": entry_open, "liq": liq,
            "entry_limit_up": entry_limit_up, "avail_day_limit_up": avail_day_limit_up,
            **{f"ret{h}": labels[h] for h in horizons},
        })
    ev = pd.DataFrame(rows)
    ev["year"] = ev["avail_date"].dt.year
    ev["spac"] = ev["code"].isin(spac)
    ev["pass_liq"] = ev["liq"] >= LIQ_FLOOR_MWON
    ev["in_universe"] = ev["pass_liq"] & (~ev["spac"]) & ev["sue_raw"].notna()
    return ev


def main():
    earn, bars, raw, spac = load_raw()
    ev = build_events(earn, bars, raw, spac)
    print(f"총 earnings 행: {len(earn):,}, adj bars 종목: {bars['code'].nunique():,}, 스팩 코드: {len(spac)}")
    print(f"이벤트(가격 매칭됨): {len(ev):,}")
    print(f"  시그널 유효(op_income & prior): {ev['sue_raw'].notna().sum():,}")
    print(f"  유동성 통과(>= {LIQ_FLOOR_MWON}백만): {ev['pass_liq'].sum():,}")
    print(f"  스팩: {ev['spac'].sum():,}")
    print(f"  최종 유니버스: {ev['in_universe'].sum():,}")
    print(f"  진입일 상한가(체결불가): {(ev['in_universe'] & ev['entry_limit_up']).sum():,}")
    print(f"  +40 라벨 존재(유니버스&non-limit): "
          f"{(ev['in_universe'] & ~ev['entry_limit_up'] & ev['ret40'].notna()).sum():,}")

    # 중복 점검 (#3): 같은 분기 같은 종목 여러 행?
    dup = ev.groupby(["code", "period"]).size()
    print(f"  (code,period) 중복 행: {(dup > 1).sum()} 조합")

    print("\n분기별(유니버스 & non-limit & ret40 유효) N:")
    q = ev[ev["in_universe"] & ~ev["entry_limit_up"] & ev["ret40"].notna()].copy()
    q["quarter"] = q["period"]
    tab = q.groupby("quarter").size()
    for qtr, n in tab.items():
        print(f"  {qtr}: {n}")
    print(f"\n유효 분기 수: {tab.shape[0]}, 분기당 중앙 N: {int(tab.median())}, 합계: {tab.sum():,}")


if __name__ == "__main__":
    main()
