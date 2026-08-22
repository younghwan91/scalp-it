"""돌파 매매 사전등록 — **이벤트 규모만** 일봉으로 센다. 수익률은 계산하지 않는다.

`docs/research/manju/09-breakout-preregistration.md` 의 "이벤트 규모" 절이 이 스크립트의
출력이다. 분봉 도착 전에 표본 크기·월별 분포·분봉 보유 종목 수를 알아야 설계(폴드 수,
임계)를 정할 수 있고, 분봉이 없는 종목 목록을 수집 큐에 넣어야 하기 때문이다.

**이 파일에서 수익률·종가·익일 가격을 읽는 코드는 금지다.** 사전등록 전에 결과를 보는
것이 이 프로젝트의 최대 금기다. 읽는 열은 open/high/trade_value 와 돌파선 계산용 과거 high 뿐.

실행:
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/breakout_events.py [--out scratchpad/breakout_need.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

import pandas as pd

# ---- 사전 고정 파라미터 (문서와 동일해야 한다) -----------------------------------
LOOKBACK = 250          # 52주 신고가 ≈ 250거래일. 2차 사전등록의 `+신고가` 와 같은 값
LOOKBACK_REF = 20       # 참고용 건수만 센다. 규칙 후보가 아니다
ADV_FLOOR_MN = 5_000    # 직전 20일 평균 거래대금 ≥ 50억 (백만원 단위). 프로젝트 기존 하한
START = "2025-08-01"    # 분봉 수집 시작일
END = "2026-08-21"
INDEP_GAP = 5           # "독립 이벤트" = 같은 종목 직전 5거래일 안에 이벤트 없음
NEED_LEAD_DAYS = 10     # 수집 큐: 최초 이벤트일 − 10일(달력)


def _connect():
    import psycopg2

    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        sys.exit("KR_QUANT_DB 가 없다.")
    return psycopg2.connect(dsn)


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    # 250거래일 선행 구간을 넉넉히 — 2024-06 부터
    daily = pd.read_sql(
        "select code, date, open, high, trade_value from daily_bars_adjusted "
        "where date >= '2024-06-01' and date <= %(end)s order by code, date",
        conn, params={"end": END},
    )
    stocks = pd.read_sql("select code, name, market from stocks", conn)
    mb = pd.read_sql("select distinct code from minute_bars", conn)
    return daily, stocks, set(mb["code"].str.strip())


def compute(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("code", sort=False)
    daily = daily.copy()
    daily["prior_high"] = g["high"].transform(
        lambda s: s.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).max())
    daily["prior_high_ref"] = g["high"].transform(
        lambda s: s.shift(1).rolling(LOOKBACK_REF, min_periods=LOOKBACK_REF).max())
    daily["adv20"] = g["trade_value"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    w = daily[(daily["date"] >= pd.Timestamp(START).date()) & (daily["date"] <= pd.Timestamp(END).date())].copy()
    w["has_hist"] = w["prior_high"].notna()
    w["ev_any"] = w["has_hist"] & (w["high"] > w["prior_high"])
    w["ev_gap"] = w["ev_any"] & (w["open"] > w["prior_high"])       # 시가부터 돌파선 위 → 갭돌파
    w["ev_intraday"] = w["ev_any"] & (w["open"] <= w["prior_high"])  # 장중 돌파 (주 이벤트)
    w["in_univ"] = w["adv20"] >= ADV_FLOOR_MN
    w["ev_ref20"] = w["prior_high_ref"].notna() & (w["high"] > w["prior_high_ref"]) & (w["open"] <= w["prior_high_ref"])
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="분봉 미보유 종목 JSON 저장 경로")
    args = ap.parse_args()

    conn = _connect()
    daily, stocks, mb_codes = load(conn)
    conn.close()

    spac = set(stocks.loc[stocks["name"].str.contains("스팩", na=False), "code"])
    listed = set(stocks["code"])
    w = compute(daily)
    w = w[~w["code"].isin(spac)]

    ev = w[w["ev_any"]]
    n_all = len(ev)
    ev_u = ev[ev["in_univ"]]
    ev_main = ev_u[ev_u["ev_intraday"]].copy()
    ev_gap = ev_u[ev_u["ev_gap"]]
    ref20 = w[w["ev_ref20"] & w["in_univ"]]

    # 독립 이벤트 — 같은 종목 직전 INDEP_GAP 거래일 안에 주 이벤트 없음
    # (거래일 간격은 그 종목의 일봉 행 순서로 센다)
    w_sorted = w.sort_values(["code", "date"]).copy()
    w_sorted["rn"] = w_sorted.groupby("code").cumcount()
    w_sorted["ev_main_u"] = w_sorted["ev_intraday"] & w_sorted["in_univ"]
    prev_rn = w_sorted.loc[w_sorted["ev_main_u"]].groupby("code")["rn"].shift(1)
    indep = (w_sorted.loc[w_sorted["ev_main_u"], "rn"] - prev_rn)
    n_indep = int(((indep.isna()) | (indep > INDEP_GAP)).sum())

    ev_main["month"] = pd.to_datetime(ev_main["date"]).dt.strftime("%Y-%m")
    monthly = ev_main.groupby("month").agg(n=("code", "size"), codes=("code", "nunique"))
    per_code = ev_main.groupby("code").size()
    codes_main = set(ev_main["code"])
    codes_with_mb = codes_main & mb_codes
    codes_delisted = codes_main - listed
    ev_with_mb = int(ev_main["code"].isin(mb_codes).sum())

    print("## 이벤트 규모 — 일봉 측정 (수익률 없음)\n")
    print(f"기간 {START} ~ {END}, 돌파선 = 직전 {LOOKBACK}거래일 최고가(shift 1), 스팩 제외\n")
    print("| 단계 | (종목,일자) 건수 | 종목 수 |")
    print("|---|---:|---:|")
    print(f"| 고가 > 직전 {LOOKBACK}일 최고가 (전체) | {n_all:,} | {ev['code'].nunique():,} |")
    print(f"| + 직전 20일 평균 거래대금 ≥ 50억 | {len(ev_u):,} | {ev_u['code'].nunique():,} |")
    print(f"|   └ 갭돌파 (시가가 이미 돌파선 위) — **제외** | {len(ev_gap):,} | {ev_gap['code'].nunique():,} |")
    print(f"|   └ **장중 돌파 (주 이벤트)** | **{len(ev_main):,}** | **{len(codes_main):,}** |")
    print(f"|      └ 독립 이벤트 (직전 {INDEP_GAP}거래일 내 이벤트 없음) | {n_indep:,} | — |")
    print(f"|      └ 그중 `minute_bars` 에 분봉 있는 종목 | {ev_with_mb:,} | {len(codes_with_mb):,} |")
    print(f"|      └ 현재 상장 목록(`stocks`)에 없는 종목 | {int(ev_main['code'].isin(codes_delisted).sum()):,} | {len(codes_delisted):,} |")
    print(f"| (참고, 규칙 후보 아님) 직전 {LOOKBACK_REF}일 고가 장중 돌파 + 거래대금 하한 | {len(ref20):,} | {ref20['code'].nunique():,} |")
    print()
    print(f"250일 이력 부족으로 판정 불가한 (종목,일자): {int((~w['has_hist']).sum()):,} "
          f"(신규 상장주 등)\n")
    print("### 월별 분포 — 주 이벤트\n")
    print("| 월 | 건수 | 종목 수 |")
    print("|---|---:|---:|")
    for m, r in monthly.iterrows():
        print(f"| {m} | {int(r['n']):,} | {int(r['codes']):,} |")
    print()
    d = per_code.describe()
    print("### 종목당 이벤트 수 (주 이벤트)\n")
    print(f"중앙값 {d['50%']:.0f} · 평균 {d['mean']:.1f} · p75 {d['75%']:.0f} · p95 {per_code.quantile(0.95):.0f} · 최대 {d['max']:.0f}\n")

    per_day = ev_main.groupby("date").size()
    n_days = w["date"].nunique()
    print("### 하루 건수 (주 이벤트)\n")
    print(f"거래일 {n_days}일 중 이벤트 있는 날 {len(per_day)}일 · 하루 중앙값 {per_day.median():.0f} · "
          f"p90 {per_day.quantile(0.9):.0f} · 최대 {per_day.max()} (만쥬 실제 빈도는 하루 1~2건)\n")

    need = {}
    first = ev_main.groupby("code")["date"].min()
    for code, dt in first.items():
        if code not in mb_codes:
            need[code] = (pd.Timestamp(dt) - timedelta(days=NEED_LEAD_DAYS)).strftime("%Y%m%d")
    print(f"분봉이 없는 종목: {len(need):,}개 (그 종목들의 주 이벤트 {int(ev_main['code'].isin(need).sum()):,}건)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(need, f, indent=0, sort_keys=True)
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
