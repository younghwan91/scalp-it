"""장중 궤적→익일 갭 사전등록(16번) — **유니버스 규모만** 센다. 수익률은 계산하지 않는다.

`docs/research/manju/16-intraday-trajectory-preregistration.md` 의 "이벤트 규모" 절이 이 스크립트의 출력이다.
이 시험의 "이벤트" 는 트리거가 아니라 **유니버스 (종목,일자) 그 자체**다 — 당일 거래대금 ≥ 50억 · 보통주 ·
스팩/리츠 제외 · 전일종가 있음 · 상한가/하한가 마감 제외 · 그날 분봉(15:19 봉) 있음.

**이 파일에서 D+1 가격·수익률을 읽는 코드는 금지다.** 읽는 열은 당일 OHLC·거래대금과 전일종가(상·하한가 판정)뿐.
`daily_bars`(원주가) 로 호가단위 정확 상한가를, `daily_bars_adjusted` 로 1.295 근사를 판정하고 둘을 OR 로 합친다.

실행:
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/traj_events.py [--out <scratch>/traj_events.out]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 사전 고정 파라미터 (문서와 동일해야 한다) -----------------------------------
START = "2025-08-01"            # 분봉 시작일
END = "2026-08-21"              # 분봉 마지막 수집일
TV_FLOOR_MN = 5_000             # 당일 거래대금 ≥ 50억 (백만원) — 프로젝트 공통 하한
TV_REF_MN = 50_000              # 참고 분리 보고: 500억 (상따·종베 v2 하한). 판정 아님
LIMIT_UP_RATIO = 1.295          # 수정주가 근사 상한가
LIMIT_DN_RATIO = 0.705          # 수정주가 근사 하한가
NEW_LISTING_FROM = "2025-01-01" # 첫 거래일이 이후면 "신규주" 로 표시 (포함, 분리 보고만)
BIG2 = ("005930", "000660")


def _load_env(path: str = "/home/young/Documents/git/kr-quant/.env") -> None:
    if os.environ.get("KR_QUANT_DB") or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def connect():
    import psycopg2

    _load_env()
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        sys.exit("KR_QUANT_DB 가 없다.")
    return psycopg2.connect(dsn)


# ---- 호가단위 (limitup_timing.py 와 동일) ------------------------------------------

def tick_size(price: float) -> int:
    """KRX 호가단위 (2023-01-25 이후 코스피·코스닥 공통)."""
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def limit_up_price(prev_close: float) -> float:
    raw = prev_close * 1.3
    t = tick_size(raw)
    return math.floor(raw / t + 1e-9) * t


def limit_dn_price(prev_close: float) -> float:
    raw = prev_close * 0.7
    t = tick_size(raw)
    return math.ceil(raw / t - 1e-9) * t


# ---- 로드 -----------------------------------------------------------------------

def load_daily(conn, start: str = START, end: str = END) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """원주가·수정주가 일봉(전일종가용으로 한 달 선행)과 종목 마스터. D+1 행은 읽지 않는다."""
    lead = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    raw = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars "
        "where date >= %(lead)s and date <= %(end)s order by code, date",
        conn, params={"lead": lead, "end": end})
    adj = pd.read_sql(
        "select code, date, close from daily_bars_adjusted "
        "where date >= %(lead)s and date <= %(end)s order by code, date",
        conn, params={"lead": lead, "end": end})
    first = pd.read_sql("select code, min(date) as first_date from daily_bars group by code", conn)
    st = pd.read_sql("select code, name, market from stocks where kind = 'A'", conn)
    dl = pd.read_sql("select code, name, market from delisted_stocks", conn)
    st["listed_now"] = True
    dl["listed_now"] = False
    master = pd.concat([st, dl[~dl["code"].isin(st["code"])]], ignore_index=True)
    master = master.merge(first, on="code", how="left")
    for df in (raw, adj, master):
        for c in ("date", "first_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    return raw, adj, master


def minute_days(conn, start: str = START, end: str = END) -> pd.DataFrame:
    """15:19 이하 봉이 있는 (종목,일자) 와 그 봉 수. 피처 계약(15:19 까지)이 성립하는 날만 유니버스다."""
    df = pd.read_sql(
        "select code, ts::date as date, count(*) as n_bars, "
        "bool_or(ts::time = '15:19') as has_1519 "
        "from minute_bars where ts >= %(s)s and ts < %(e)s and ts::time <= '15:19' group by 1, 2",
        conn, params={"s": start, "e": (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")})
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.strip()
    return df


# ---- 유니버스 -------------------------------------------------------------------

def build_universe(raw: pd.DataFrame, adj: pd.DataFrame, master: pd.DataFrame, mdays: pd.DataFrame,
                   start: str = START, end: str = END) -> pd.DataFrame:
    """일봉 조건 → 분봉 조건 순으로 단계 플래그를 붙인 (종목,일자) 표. 수익률 열 없음.

    단계 플래그: ``common``(보통주) ``known``(마스터에 있음) ``spac_reit`` ``has_prev`` ``tv_ok``
    ``limit_up`` ``limit_dn`` ``has_min``. 유니버스 = 전부 통과 & ~limit_up & ~limit_dn.
    """
    d = raw.copy()
    g = d.groupby("code", sort=False)
    d["prev_close_raw"] = g["close"].shift(1)
    a = adj.rename(columns={"close": "close_adj"}).copy()
    a["prev_close_adj"] = a.groupby("code", sort=False)["close_adj"].shift(1)
    d = d.merge(a, on=["code", "date"], how="left")
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))].copy()

    d = d.merge(master[["code", "name", "market", "listed_now", "first_date"]], on="code", how="left")
    d["known"] = d["name"].notna()
    d["common"] = d["code"].str.len().eq(6) & d["code"].str.endswith("0")
    nm = d["name"].fillna("")
    d["spac_reit"] = nm.str.contains("스팩|SPAC|리츠|REIT", case=False, regex=True)
    d["has_prev"] = d["prev_close_raw"].notna() & (d["prev_close_raw"] > 0)
    d["tv_ok"] = d["trade_value"] >= TV_FLOOR_MN
    d["tv_ref"] = d["trade_value"] >= TV_REF_MN

    ratio = d["close_adj"] / d["prev_close_adj"]
    d["ret_day"] = ratio - 1.0
    lu_tick = d["prev_close_raw"].map(lambda p: limit_up_price(p) if np.isfinite(p) and p > 0 else np.nan)
    ld_tick = d["prev_close_raw"].map(lambda p: limit_dn_price(p) if np.isfinite(p) and p > 0 else np.nan)
    d["limit_up_tick"] = (d["close"] == lu_tick)
    d["limit_up"] = d["limit_up_tick"] | (ratio >= LIMIT_UP_RATIO)
    d["limit_dn"] = (d["close"] == ld_tick) | (ratio <= LIMIT_DN_RATIO)
    d["is_new"] = d["first_date"] >= pd.Timestamp(NEW_LISTING_FROM)
    d["big2"] = d["code"].isin(BIG2)

    md = mdays.rename(columns={"n_bars": "n_bars_1519"})
    d = d.merge(md, on=["code", "date"], how="left")
    d["has_1519"] = d["has_1519"].fillna(False).astype(bool)
    d["n_bars_1519"] = d["n_bars_1519"].fillna(0).astype(int)
    d["has_min"] = d["has_1519"]

    d["daily_ok"] = d["known"] & d["common"] & ~d["spac_reit"] & d["has_prev"] & d["tv_ok"]
    d["in_univ"] = d["daily_ok"] & d["has_min"] & ~d["limit_up"] & ~d["limit_dn"]
    d["month"] = d["date"].dt.strftime("%Y-%m")
    return d


# ---- 보고 -----------------------------------------------------------------------

def report(d: pd.DataFrame) -> str:
    L: list[str] = []
    L.append("# 장중 궤적→익일 갭 — 유니버스 규모 (수익률 없음)\n")
    L.append(f"기간 {START} ~ {END} · 거래일 {d['date'].nunique()} · 거래대금 하한 {TV_FLOOR_MN/100:.0f}억 (참고 {TV_REF_MN/100:.0f}억)\n")
    L.append("| 단계 | (종목,일자) | 종목 수 |")
    L.append("|---|---:|---:|")
    steps = [
        ("일봉 전체 (기간 내)", d),
        ("마스터에 있음(stocks·delisted_stocks)", d[d["known"]]),
        ("+ 보통주(코드 끝 0)", d[d["known"] & d["common"]]),
        ("+ 스팩·리츠 제외", d[d["known"] & d["common"] & ~d["spac_reit"]]),
        ("+ 전일종가 있음 (상장일 D0 제외)", d[d["known"] & d["common"] & ~d["spac_reit"] & d["has_prev"]]),
        (f"+ 당일 거래대금 ≥ {TV_FLOOR_MN/100:.0f}억 = **일봉 유니버스**", d[d["daily_ok"]]),
        ("+ 그날 분봉(15:19 봉) 있음", d[d["daily_ok"] & d["has_min"]]),
        ("  └ 상한가 마감 — **제외**", d[d["daily_ok"] & d["has_min"] & d["limit_up"]]),
        ("  └ 하한가 마감 — **제외**", d[d["daily_ok"] & d["has_min"] & d["limit_dn"]]),
        ("**= 유니버스 (측정 대상)**", d[d["in_univ"]]),
        (f"  참고: 그중 거래대금 ≥ {TV_REF_MN/100:.0f}억", d[d["in_univ"] & d["tv_ref"]]),
        ("  참고: 그중 신규주(첫 거래일 ≥ 2025-01-01)", d[d["in_univ"] & d["is_new"]]),
        ("  참고: 그중 삼성전자·SK하이닉스", d[d["in_univ"] & d["big2"]]),
        ("  참고: 그중 당일 등락률 ≥ +25%", d[d["in_univ"] & (d["ret_day"] >= 0.25)]),
        ("  참고: 그중 현재 비상장", d[d["in_univ"] & ~d["listed_now"]]),
    ]
    for name, sub in steps:
        L.append(f"| {name} | {len(sub):,} | {sub['code'].nunique():,} |")
    L.append("")
    du = d[d["daily_ok"]]
    cov = du["has_min"].mean() if len(du) else float("nan")
    L.append(f"분봉 커버리지: 일봉 유니버스 {len(du):,} 중 분봉 있음 {int(du['has_min'].sum()):,} (**{cov:.1%}**) · "
             f"종목 {du['code'].nunique():,} 중 {du.loc[du['has_min'], 'code'].nunique():,}. "
             f"분봉 없는 (종목,일자)의 거래대금 중앙 {du.loc[~du['has_min'], 'trade_value'].median()/100:,.0f}억 vs 있는 쪽 "
             f"{du.loc[du['has_min'], 'trade_value'].median()/100:,.0f}억 · 등락률 평균 "
             f"{du.loc[~du['has_min'], 'ret_day'].mean():+.2%} vs {du.loc[du['has_min'], 'ret_day'].mean():+.2%} "
             f"(분봉이 이벤트 기준 수집이라 생기는 편향의 크기)\n")
    lu = d[d["daily_ok"] & d["has_min"] & d["limit_up"]]
    L.append(f"상한가 판정: 호가단위 정확 일치 {int(lu['limit_up_tick'].sum()):,} · 1.295 근사만 "
             f"{int((~lu['limit_up_tick']).sum()):,} (합 {len(lu):,})\n")
    u = d[d["in_univ"]]
    L.append("### 월별 (유니버스)\n")
    L.append("| 월 | 거래일 | 일봉 유니버스 | 분봉 있음 | 상한가 제외 | 하한가 제외 | **유니버스** | 하루 평균 | 종목 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, sub in d.groupby("month"):
        du_m = sub[sub["daily_ok"]]
        dm = du_m[du_m["has_min"]]
        um = sub[sub["in_univ"]]
        L.append(f"| {m} | {sub['date'].nunique()} | {len(du_m):,} | {len(dm):,} | {int(dm['limit_up'].sum()):,} | "
                 f"{int(dm['limit_dn'].sum()):,} | **{len(um):,}** | {len(um)/max(sub['date'].nunique(),1):,.0f} | {um['code'].nunique():,} |")
    L.append("")
    per_day = u.groupby("date").size()
    L.append(f"하루 건수: 중앙 {per_day.median():.0f} · p10 {per_day.quantile(.1):.0f} · p90 {per_day.quantile(.9):.0f} · "
             f"최소 {per_day.min()} · 최대 {per_day.max()}\n")
    per_code = u.groupby("code").size()
    L.append(f"종목당 일수: 중앙 {per_code.median():.0f} · p90 {per_code.quantile(.9):.0f} · 최대 {per_code.max()} · "
             f"상위 5: {', '.join(f'{c} {n}' for c, n in per_code.sort_values(ascending=False).head(5).items())}\n")
    L.append(f"15:19 까지 봉 수: 중앙 {u['n_bars_1519'].median():.0f} · p10 {u['n_bars_1519'].quantile(.1):.0f} · "
             f"100개 미만 {int((u['n_bars_1519'] < 100).sum()):,}건 (피처는 계산되지만 해상도 낮음 — 제외하지 않고 보고)\n")
    bins = [-1, -.1, -.05, 0, .05, .10, .20, .25, .295, 10]
    dist = pd.cut(u["ret_day"], bins).value_counts().sort_index()
    L.append("유니버스 당일 등락률 분포: " + " · ".join(f"{k!s} {v:,}" for k, v in dist.items()) + "\n")
    L.append(f"시장: {u['market'].value_counts().to_dict()}\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="유니버스 표 pickle 저장 경로 (verify 가 재사용)")
    args = ap.parse_args()
    conn = connect()
    raw, adj, master = load_daily(conn)
    md = minute_days(conn)
    conn.close()
    d = build_universe(raw, adj, master, md)
    text = report(d)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.cache:
        d.to_pickle(args.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
