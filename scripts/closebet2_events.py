"""종가베팅 v2 사전등록 — **이벤트 규모만** 일봉·수급으로 센다. 수익률은 계산하지 않는다.

`docs/research/manju/14-closebet-v2-preregistration.md` 의 "이벤트 규모" 절이 이 스크립트의 출력이다.
기존 종가베팅(tv_ratio·close_pos)과 달리 선택 변수가 **만쥬의 실제 변수** — 외국인+기관 순매수,
업종 지수 강세, 신고가 마감 — 이고, 상한가 마감은 선택에서 제외한다.

**이 파일에서 익일 가격·수익률을 읽는 코드는 금지다.** 읽는 열은 당일 OHLC·거래대금, 당일 수급,
당일 업종지수, 그리고 신고가선 계산용 과거 고가뿐. D+1 행은 어디서도 참조하지 않는다.

실행:
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/closebet2_events.py [--need-out scratchpad/cbet2_need.json] [--ref]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

# ---- 사전 고정 파라미터 (문서와 동일해야 한다) -----------------------------------
TV_FLOOR_MN = 50_000      # 당일 거래대금 ≥ 500억 (백만원). 08번 문서 상따 하한과 같은 기존 값
TV_FLOOR_REF_MN = 5_000   # 참고: 기존 종가베팅 하한 50억 — 커버리지 비교용으로만 센다
LOOKBACK = 250            # 신고가 = 직전 250거래일 최고가(shift 1). 2차·돌파와 같은 값
BOX_WINDOW = 30           # 피처용 횡보 폭 — 규칙 아님
LIMIT_UP = 1.295          # 상한가 마감 판정: 종가 ≥ 전일종가 × 1.295
TOP_N = 2                 # 하루 선택 건수 — 기존 종가베팅과 동일. 순위 = 외국인+기관 순매수 금액 내림차순
MAIN_START = "2024-02-01" # 업종 지수(2024-01-26~)·수급 전 종목(2024~) 둘 다 있는 구간
END = "2026-08-21"
REF_START = "2017-01-01"  # 보조 측정(판정 불사용): 630종목·동일가중 섹터 근사
MINUTE_START = "2025-08-01"
NEED_LEAD_DAYS = 10

# 업종명 → 업종지수 코드. stocks.sector 이름과 sector_index.name 이 같은 KRX 분류라 이름으로 맞춘다.
# 한 시장에 같은 이름이 둘인 지수는 없다(확인: 거래소 005~030, 코스닥 103~141).
MARKET_INDEX = {"거래소": "001", "코스닥": "101"}


def _connect():
    import psycopg2

    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        sys.exit("KR_QUANT_DB 가 없다.")
    return psycopg2.connect(dsn)


def load(conn, start: str):
    # 250거래일 선행 구간을 넉넉히 — start 의 14개월 전부터
    lead = (pd.Timestamp(start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    daily = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars_adjusted "
        "where date >= %(lead)s and date <= %(end)s order by code, date",
        conn, params={"lead": lead, "end": END},
    )
    sd = pd.read_sql(
        "select code, date, foreign_, institution, penfnd_etc, source from supply_demand "
        "where date >= %(start)s and date <= %(end)s",
        conn, params={"start": start, "end": END},
    )
    stocks = pd.read_sql("select code, name, market, sector from stocks where kind = 'A'", conn)
    si = pd.read_sql(
        "select code, name, date, close from sector_index where date >= '2024-01-01' order by code, date", conn)
    for df in (daily, sd, si):
        df["date"] = pd.to_datetime(df["date"])
    return daily, sd, stocks, si


def option_expiry_days(trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """KOSPI200 옵션 최종거래일 = 매월 둘째 목요일, 휴장이면 직전 거래일(KRX 규정, 순차 앞당김)."""
    out = set()
    tds = pd.DatetimeIndex(sorted(set(trading_days)))
    for y in range(tds.min().year, tds.max().year + 1):
        for m in range(1, 13):
            first = pd.Timestamp(year=y, month=m, day=1)
            # 둘째 목요일
            offset = (3 - first.weekday()) % 7
            second_thu = first + pd.Timedelta(days=offset + 7)
            cand = tds[tds <= second_thu]
            if len(cand):
                out.add(cand[-1])
    return out


def compute(daily: pd.DataFrame, sd: pd.DataFrame, stocks: pd.DataFrame, si: pd.DataFrame,
            start: str, sector_mode: str) -> pd.DataFrame:
    d = daily.copy()
    g = d.groupby("code", sort=False)
    d["prev_close"] = g["close"].shift(1)
    d["hist"] = g.cumcount()                       # 상장 후 경과 거래일 (0 = 상장일). 데이터 시작 2016-09 이후 기준
    d["ret_day"] = d["close"] / d["prev_close"] - 1
    d["prior_high"] = g["high"].transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=1).max())
    d["has_hist250"] = g["high"].transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).max()).notna()
    d["box_width"] = g["close"].transform(
        lambda s: s.shift(1).rolling(BOX_WINDOW, min_periods=BOX_WINDOW).max()
        / s.shift(1).rolling(BOX_WINDOW, min_periods=BOX_WINDOW).min() - 1)
    span = d["high"] - d["low"]
    d["close_pos"] = ((d["close"] - d["low"]) / span.where(span != 0)).fillna(0.5)
    d["vwap"] = (d["trade_value"] * 1e6 / d["volume"].where(d["volume"] > 0))
    d["nh250"] = d["close"] > d["prior_high"]         # 상장일(prior_high NaN)은 False
    d["bull"] = d["close"] > d["open"]
    d["limit_up"] = d["close"] >= d["prev_close"] * LIMIT_UP
    # 상장일 상한가(2023-06-26 이전 규정: 시가×1.3)만 근사. 이후 규정(공모가×4)은 공모가가 없어 판정 불가
    d.loc[d["hist"] == 0, "limit_up"] = (d["close"] >= d["open"] * LIMIT_UP) & (d["date"] < "2023-06-26")

    w = d[d["date"] >= pd.Timestamp(start)].copy()

    # 유니버스: 보통주(코드 끝 0)·스팩 제외·현재 상장(stocks, 업종 필요) — 폐지 종목은 업종이 없어 빠진다
    st = stocks.copy()
    st["is_spac"] = st["name"].str.contains("스팩", na=False)
    w = w.merge(st[["code", "name", "market", "sector", "is_spac"]], on="code", how="left")
    w["listed_now"] = w["name"].notna()
    w["is_spac"] = w["is_spac"].fillna(False).astype(bool)
    w["common"] = w["code"].str.endswith("0")

    # 수급: 외국인 + 기관계 순매수 **수량** × 당일 VWAP ≈ 순매수 금액(억)
    w = w.merge(sd, on=["code", "date"], how="left")
    w["has_sd"] = w["foreign_"].notna() & w["institution"].notna()
    w["fi_amt"] = (w["foreign_"] + w["institution"]) * w["vwap"] / 1e8
    w["frgn_amt"] = w["foreign_"] * w["vwap"] / 1e8
    w["fi_share"] = w["fi_amt"] / (w["trade_value"] / 100)   # 백만 → 억

    # 섹터: 업종지수(2024-01-26~) 또는 동일가중 평균(보조 측정)
    if sector_mode == "index":
        si = si.copy()
        si["ret"] = si.groupby("code")["close"].pct_change()
        mk = si[si["code"].isin(MARKET_INDEX.values())][["code", "date", "ret"]]
        mk = mk.rename(columns={"code": "mkt_code", "ret": "mkt_ret"})
        w["mkt_code"] = w["market"].map(MARKET_INDEX)
        w = w.merge(mk, on=["mkt_code", "date"], how="left")
        # 업종명 + 시장 → 지수
        kospi = si[si["code"].str.startswith("0")][["name", "code"]].drop_duplicates("name")
        kosdaq = si[si["code"].str.startswith("1")][["name", "code"]].drop_duplicates("name")
        w["sec_code"] = np.where(
            w["market"] == "거래소",
            w["sector"].map(dict(zip(kospi["name"], kospi["code"]))),
            w["sector"].map(dict(zip(kosdaq["name"], kosdaq["code"]))))
        sec = si[["code", "date", "ret"]].rename(columns={"code": "sec_code", "ret": "sec_ret"})
        w = w.merge(sec, on=["sec_code", "date"], how="left")
    else:
        # 동일가중: 같은 (시장, 업종) 의 당일 등락률 평균 / 시장 = 같은 시장 전 종목 평균 (현재 업종 분류 → 경미한 lookahead)
        base = w[w["listed_now"] & w["common"] & ~w["is_spac"]]
        sec = base.groupby(["market", "sector", "date"])["ret_day"].mean().rename("sec_ret").reset_index()
        mk = base.groupby(["market", "date"])["ret_day"].mean().rename("mkt_ret").reset_index()
        w = w.merge(sec, on=["market", "sector", "date"], how="left").merge(mk, on=["market", "date"], how="left")
    w["has_sec"] = w["sec_ret"].notna() & w["mkt_ret"].notna()
    w["sec_rel"] = w["sec_ret"] - w["mkt_ret"]
    w["sec_strong"] = w["has_sec"] & (w["sec_rel"] > 0)

    opex = option_expiry_days(pd.DatetimeIndex(w["date"].unique()))
    w["opex"] = w["date"].isin(opex)
    w["is_new"] = ~w["has_hist250"]
    return w


def select_events(w: pd.DataFrame, floor_mn: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """후보 풀(pool) → 상한가 제외(pool_ok) → 하루 상위 TOP_N(sel). 수익률은 어디에도 없다."""
    base = w[w["listed_now"] & w["common"] & ~w["is_spac"] & (w["trade_value"] >= floor_mn)].copy()
    base = base[base["has_sd"]]
    pool = base[(base["fi_amt"] > 0) & (base["sec_strong"] | base["nh250"]) & base["bull"]].copy()
    pool_ok = pool[~pool["limit_up"]].copy()
    # 순위 = 외국인+기관 순매수 **절대 금액** 내림차순 — 증권사 HTS "외국인/기관 순매수 상위" 화면이 금액 정렬이고
    # 만쥬가 15:19 에 보는 것이 그것이다. 비율(순매수/거래대금)은 피처로만 둔다.
    pool_ok["rank"] = pool_ok.groupby("date")["fi_amt"].rank(ascending=False, method="first")
    sel = pool_ok[pool_ok["rank"] <= TOP_N].copy()
    return base, pool, sel


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def yearly_table(base, pool, sel, label_year) -> pd.DataFrame:
    rows = []
    for y, grp in sel.groupby(label_year(sel)):
        b = base[label_year(base) == y]
        p = pool[label_year(pool) == y]
        n_days = b["date"].nunique()
        rows.append({
            "기간": y,
            "거래일": n_days,
            "유니버스 (종목,일) 일평균": round(len(b) / max(n_days, 1), 1),
            "후보 풀": len(p),
            "풀 중 상한가 마감": f"{int(p['limit_up'].sum()):,} ({p['limit_up'].mean()*100:.1f}%)",
            "선택": len(grp),
            "선택 종목 수": grp["code"].nunique(),
            "하루 선택 중앙/최대": f"{grp.groupby('date').size().median():.0f} / {grp.groupby('date').size().max()}",
            "신규주(<250일)": f"{int(grp['is_new'].sum())} ({grp['is_new'].mean()*100:.1f}%)",
            "섹터만/신고가만/둘다": f"{int((grp['sec_strong'] & ~grp['nh250']).sum())}/"
                                 f"{int((~grp['sec_strong'] & grp['nh250']).sum())}/"
                                 f"{int((grp['sec_strong'] & grp['nh250']).sum())}",
            "옵션만기일 선택": int(grp["opex"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-out", default=None, help="분봉 없는 (종목,일자) → {code: YYYYMMDD} JSON")
    ap.add_argument("--ref", action="store_true", help="보조 측정(2017~2023, 630종목·동일가중 섹터)도 센다")
    args = ap.parse_args()

    conn = _connect()
    daily, sd, stocks, si = load(conn, REF_START if args.ref else MAIN_START)

    print("## 이벤트 규모 — 일봉·수급 측정 (수익률 없음)\n")
    print(f"주 정의: 거래대금 ≥ {TV_FLOOR_MN/100:.0f}억 · 외국인+기관 순매수 > 0 · (업종지수 > 시장지수 OR 종가 > 직전 {LOOKBACK}일 고가) · "
          f"양봉 · 상한가 마감 제외 · 하루 상위 {TOP_N}건(외국인+기관 순매수 금액)\n")

    # ---- 수급 커버리지 (유니버스 결정 근거) ----
    d = daily[daily["date"] >= pd.Timestamp(REF_START if args.ref else MAIN_START)]
    d = d[d["code"].isin(stocks["code"]) & d["code"].str.endswith("0")]
    cov = d.merge(sd[["code", "date", "foreign_"]], on=["code", "date"], how="left")
    cov["y"] = cov["date"].dt.year
    print("### 수급(`supply_demand`) 커버리지 — 유니버스 (종목,일자) 중 수급 행이 있는 비율\n")
    print("| 연도 | 거래대금≥50억 | 그중 수급 有 | 거래대금≥500억 | 그중 수급 有 |")
    print("|---|---:|---:|---:|---:|")
    for y, grp in cov.groupby("y"):
        a = grp[grp["trade_value"] >= TV_FLOOR_REF_MN]
        b = grp[grp["trade_value"] >= TV_FLOOR_MN]
        print(f"| {y} | {len(a):,} | {a['foreign_'].notna().mean()*100:.1f}% | {len(b):,} | {b['foreign_'].notna().mean()*100:.1f}% |")
    print()

    # ---- 주 측정 ----
    w = compute(daily, sd, stocks, si, MAIN_START, "index")
    base, pool, sel = select_events(w, TV_FLOOR_MN)
    yr = lambda df: df["date"].dt.year.astype(str)
    print(f"### 주 측정 {MAIN_START} ~ {END} (업종지수·전 종목 수급)\n")
    univ = w[w["listed_now"] & w["common"] & ~w["is_spac"] & (w["trade_value"] >= TV_FLOOR_MN)]
    print(f"유니버스 (종목,일자) {len(univ):,} · 그중 수급 없음 {int((~univ['has_sd']).sum()):,} · "
          f"업종지수 없음 {int((~univ['has_sec']).sum()):,} (업종 공란·2024-12 이전 부동산/IT서비스/오락문화/코스닥통신)\n")
    print("| 단계 | (종목,일자) | 종목 수 |")
    print("|---|---:|---:|")
    print(f"| 유니버스: 보통주·스팩 제외·현재 상장·거래대금 ≥ 500억·수급 有 | {len(base):,} | {base['code'].nunique():,} |")
    s1 = base[base["fi_amt"] > 0]
    print(f"| + 외국인+기관 순매수 > 0 | {len(s1):,} | {s1['code'].nunique():,} |")
    s2 = s1[s1["sec_strong"] | s1["nh250"]]
    print(f"| + (업종지수 > 시장지수 OR 신고가 마감) | {len(s2):,} | {s2['code'].nunique():,} |")
    print(f"| + 양봉 (종가 > 시가) = **후보 풀** | {len(pool):,} | {pool['code'].nunique():,} |")
    lu = pool[pool["limit_up"]]
    print(f"|   └ 상한가 마감 — **제외** | {len(lu):,} ({len(lu)/max(len(pool),1)*100:.1f}%) | {lu['code'].nunique():,} |")
    print(f"|   └ 제외 후 풀 | {len(pool)-len(lu):,} | {pool[~pool['limit_up']]['code'].nunique():,} |")
    print(f"| **하루 상위 {TOP_N}건 (외국인+기관 순매수 금액) = 선택** | **{len(sel):,}** | **{sel['code'].nunique():,}** |")
    print()
    print("### 연도별 (주 측정)\n")
    print(md(yearly_table(base, pool, sel, yr)))
    print()
    per_code = sel.groupby("code").size().sort_values(ascending=False)
    print(f"선택 종목당 건수: 중앙 {per_code.median():.0f} · p95 {per_code.quantile(.95):.0f} · 최대 {per_code.max()} "
          f"(상위 5: {', '.join(f'{c} {n}' for c, n in per_code.head(5).items())}) — 같은 종목 재선택은 측정 시 별도 보고\n")
    # 등락률 분포 (상한가 근접 오염 점검용 — 수익률 아님)
    bins = [-1, 0, .05, .10, .20, .25, .295, 10]
    dist = pd.cut(sel["ret_day"], bins).value_counts().sort_index()
    print("선택 건 당일 등락률 분포: " + " · ".join(f"{k!s} {v}" for k, v in dist.items()) + "\n")
    pool_ok = pool[~pool["limit_up"]]
    per_day = pool_ok.groupby("date").size()
    print(f"제외 후 풀 하루 건수: 중앙 {per_day.median():.0f} · p10 {per_day.quantile(.1):.0f} · p90 {per_day.quantile(.9):.0f} · "
          f"선택 {TOP_N}건 미만인 날 {int((per_day < TOP_N).sum())}/{pool_ok['date'].nunique()}\n")
    # 연기금 방향 일치(피처 참고): 선택 건 중 연기금 순매수 > 0 비율
    pf = sel["penfnd_etc"]
    print(f"선택 건 중 연기금 순매수 > 0: {int((pf > 0).sum())}/{int(pf.notna().sum())} (연기금 값 없음 {int(pf.isna().sum())})\n")
    sel_s = sel.sort_values(["code", "date"])
    gap = (sel_s["date"] - sel_s.groupby("code")["date"].shift(1)).dt.days
    print(f"선택 건 중 같은 종목을 직전 7일(달력) 안에 또 선택한 비율: {(gap <= 7).mean()*100:.1f}% · "
          f"선택 거래대금 중앙 {sel['trade_value'].median()/100:,.0f}억 · 순매수 금액 중앙 {sel['fi_amt'].median():,.0f}억 · "
          f"순매수/거래대금 중앙 {sel['fi_share'].median():.3f} (풀 중앙 {pool_ok['fi_share'].median():.3f}) · "
          f"시장 {sel['market'].value_counts().to_dict()}\n")
    print(f"(진단용, 채택 판정 불사용) 제외 후 풀 전체: {len(pool_ok):,}건 · 종목 {pool_ok['code'].nunique():,} · "
          f"신규주 {int(pool_ok['is_new'].sum()):,} ({pool_ok['is_new'].mean()*100:.1f}%) · "
          f"섹터만/신고가만/둘다 {int((pool_ok['sec_strong'] & ~pool_ok['nh250']).sum()):,}/"
          f"{int((~pool_ok['sec_strong'] & pool_ok['nh250']).sum()):,}/{int((pool_ok['sec_strong'] & pool_ok['nh250']).sum()):,}\n")

    # ---- 분봉 필요 목록 ----
    sel_m = sel[sel["date"] >= pd.Timestamp(MINUTE_START)]
    codes = sorted(sel_m["code"].unique())
    have = pd.read_sql(
        "select code, ts::date as d from minute_bars where code = any(%(codes)s) and ts::time = '15:19' group by 1, 2",
        conn, params={"codes": codes})
    conn.close()
    have_set = set(zip(have["code"], pd.to_datetime(have["d"])))
    sel_m = sel_m.assign(has_min=[(c, dt) in have_set for c, dt in zip(sel_m["code"], sel_m["date"])])
    print(f"### 분봉 보유 — 선택 건 중 {MINUTE_START} 이후 {len(sel_m):,}건: 분봉 有 {int(sel_m['has_min'].sum()):,} · "
          f"없음 {int((~sel_m['has_min']).sum()):,} (종목 {sel_m.loc[~sel_m['has_min'], 'code'].nunique()})\n")
    if args.need_out:
        need = (sel_m[~sel_m["has_min"]].groupby("code")["date"].min() - timedelta(days=NEED_LEAD_DAYS))
        need_json = {c: dt.strftime("%Y%m%d") for c, dt in need.items()}
        with open(args.need_out, "w") as f:
            json.dump(need_json, f, indent=0, sort_keys=True)
        print(f"→ {args.need_out} ({len(need_json)} 종목)\n")

    # ---- 보조 측정 (판정 불사용) ----
    if args.ref:
        w2 = compute(daily, sd, stocks, si, REF_START, "eqw")
        w2 = w2[w2["date"] < pd.Timestamp(MAIN_START)]
        b2, p2, s2_ = select_events(w2, TV_FLOOR_MN)
        print(f"### 보조 측정 {REF_START} ~ {MAIN_START} (630종목 수급 · 동일가중 섹터 근사 · 판정에 쓰지 않음)\n")
        print(md(yearly_table(b2, p2, s2_, yr)))
        print()


if __name__ == "__main__":
    main()
