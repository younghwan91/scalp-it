"""신규주 유니버스 사전등록 — **이벤트 규모만** 일봉으로 센다. 수익률은 계산하지 않는다.

`docs/research/manju/15-newlisting-preregistration.md` 의 "이벤트 규모" 절이 이 스크립트의
출력이다. 신규주는 표본이 적을 것이므로, 사전등록(폴드 수·최소 건수·임계)을 정하기 전에
건수·월별 분포·상한가 비율·분봉 보유 비율을 알아야 한다.

**이 파일에서 수익률·익일 가격을 읽는 코드는 금지다.** 읽는 열은 open/high/close/trade_value 다.
close 는 (1) 전일종가(상따 기준선·상한가 판정) (2) 당일 상한가 마감 여부에만 쓴다 —
진입가와 비교해 수익을 내는 계산은 없다.

정의 (문서와 동일해야 한다):
  상장일     `stocks` 에 상장일 컬럼이 없다 → `daily_bars` 의 첫 거래일로 추정.
             DB 가 2016-09-09 시작이므로 첫 거래일 ≥ 2016-10-01 인 종목만 "신규 상장" 으로 인정.
  경과일 d   상장일 = D0 (d=0). 일봉 행 순서로 센다. 유니버스 = d < N (N=60).
  제외       스팩·리츠 (이름), 이름을 모르는 종목(상장 목록·폐지 목록 모두 없음)은 건수만 보고.
  (A) 신규주 돌파  d ≥ 1. 돌파선 = 상장일 ~ D−1 일봉 고가의 최대(원주가). 당일 고가 > 돌파선.
             시가 ≤ 돌파선 (장중 돌파) 만 주 이벤트, 시가 > 돌파선은 갭돌파 → 제외·보고.
             유니버스 하한: 상장 후 D−1 까지 일평균 거래대금 ≥ 50억.
  (B) 신규주 상따  d ≥ 0. 기준선 = 전일종가 (D0 은 시초가). 당일 고가 > 기준선 × 1.25.
             유니버스 하한: 당일 거래대금 ≥ 50억 (일봉에서는 당일 합계로 상한 집계; 분봉 단계에서
             트리거 시점 누적 ≥ 50억 으로 대체).
  상한가 마감  d ≥ 1: 종가 ≥ 전일종가 × 1.295. D0: 2023-06-26 이전 상장은 시초가 × 1.295 (따상),
             이후 상장은 가격제한이 공모가 × 4 라 공모가 없이 판정 불가 → "고가 = 종가" 를 대리로 보고.

실행:
    uv run --extra pg python scripts/newlisting_events.py [--out scratchpad/newlist_need.json]
    (KR_QUANT_DB 가 없으면 ../kr-quant/.env 에서 읽는다)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---- 사전 고정 파라미터 (문서와 동일해야 한다) -----------------------------------
N_DAYS = 60                         # 상장 후 N 거래일 미만 (D0 ~ D59)
LISTING_CUTOFF = date(2016, 10, 1)  # DB 시작(2016-09-09) 직후 첫 등장은 기존 상장주일 수 있다
TV_FLOOR_MN = 5_000                 # 50억 (백만원). 프로젝트 공통 하한
SANGTTA_RATIO = 1.25
LIMIT_RATIO = 1.295
D0_RULE_CHANGE = date(2023, 6, 26)  # 신규상장일 가격제한 60~400% 시행일
RECENT_START = date(2025, 8, 1)     # 분봉 수집 구간
END = date(2026, 8, 21)
NEED_LEAD_DAYS = 10
INDEP_GAP = 5                       # 독립 이벤트 = 같은 종목 직전 5거래일 안에 이벤트 없음 (09번과 동일)
MANJU_CODES = {"317450": "명인제약", "439260": "대한조선", "125490": "한라캐스트", "380550": "뉴로핏"}


def _load_env(path: str = "/home/young/Documents/git/kr-quant/.env") -> None:
    if os.environ.get("KR_QUANT_DB"):
        return
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _connect():
    import psycopg2

    _load_env()
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        sys.exit("KR_QUANT_DB 가 없다.")
    return psycopg2.connect(dsn)


def load(conn):
    # 첫 거래일 → 신규 상장 후보만 일봉을 읽는다 (전체 570만 행을 안 읽기 위해)
    first = pd.read_sql("select code, min(date) as first_date from daily_bars group by code", conn)
    first["code"] = first["code"].str.strip()
    cand = first[first["first_date"] >= LISTING_CUTOFF]
    codes = tuple(cand["code"].tolist())
    daily = pd.read_sql(
        "select code, date, open, high, close, trade_value from daily_bars "
        "where code in %(codes)s order by code, date",
        conn, params={"codes": codes},
    )
    daily["code"] = daily["code"].str.strip()
    stocks = pd.read_sql("select code, name, market from stocks", conn)
    stocks["code"] = stocks["code"].str.strip()
    delisted = pd.read_sql("select code, name from delisted_stocks", conn)
    delisted["code"] = delisted["code"].str.strip()
    mb = pd.read_sql(
        "select code, ts::date as date from minute_bars where code in %(codes)s group by 1, 2",
        conn, params={"codes": codes},
    )
    mb["code"] = mb["code"].str.strip()
    sd = pd.read_sql(
        "select code, min(date) as sd_first from supply_demand where code in %(codes)s group by code",
        conn, params={"codes": codes},
    ) if _table_exists(conn, "supply_demand") else pd.DataFrame(columns=["code", "sd_first"])
    return first, daily, stocks, delisted, mb, sd


def _table_exists(conn, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("select 1 from information_schema.tables where table_name=%s", (name,))
    return cur.fetchone() is not None


def classify(first: pd.DataFrame, stocks: pd.DataFrame, delisted: pd.DataFrame) -> pd.DataFrame:
    names = pd.concat([stocks[["code", "name"]], delisted[["code", "name"]]]).drop_duplicates("code")
    f = first.merge(names, on="code", how="left")
    f["listed_now"] = f["code"].isin(set(stocks["code"]))
    nm = f["name"].fillna("")
    f["is_spac"] = nm.str.contains("스팩|SPAC", case=False, regex=True)
    f["is_reit"] = nm.str.contains("리츠|REIT", case=False, regex=True)
    f["is_etp"] = nm.str.contains(r"KODEX|TIGER|KBSTAR|ARIRANG|ETN|ETF|HANARO|SOL |ACE |KOSEF|TIMEFOLIO|PLUS ", case=False, regex=True)
    f["name_unknown"] = f["name"].isna()
    f["is_new"] = f["first_date"] >= LISTING_CUTOFF
    f["keep"] = f["is_new"] & ~f["is_spac"] & ~f["is_reit"] & ~f["is_etp"]
    return f


def compute(daily: pd.DataFrame, keep_codes: set[str]) -> pd.DataFrame:
    d = daily[daily["code"].isin(keep_codes)].drop_duplicates(["code", "date"]).sort_values(["code", "date"]).copy()
    g = d.groupby("code", sort=False)
    d["d"] = g.cumcount()
    d["prev_close"] = g["close"].shift(1)
    d["prior_max_high"] = g["high"].transform(lambda s: s.shift(1).cummax())
    d["adv_hist"] = g["trade_value"].transform(lambda s: s.shift(1).expanding().mean())
    d["d0_open"] = g["open"].transform("first")
    d["listing_date"] = g["date"].transform("first")
    w = d[d["d"] < N_DAYS].copy()

    # (A) 신규주 돌파 — 상장 후 최고가 장중 첫 돌파 (d ≥ 1)
    w["A_any"] = (w["d"] >= 1) & (w["high"] > w["prior_max_high"])
    w["A_gap"] = w["A_any"] & (w["open"] > w["prior_max_high"])
    w["A_intraday"] = w["A_any"] & (w["open"] <= w["prior_max_high"])
    w["A_univ"] = w["adv_hist"] >= TV_FLOOR_MN
    w["A"] = w["A_intraday"] & w["A_univ"]

    # (B) 신규주 상따 — 전일종가(D0: 시초가) × 1.25 돌파
    w["B_ref"] = np.where(w["d"] == 0, w["open"], w["prev_close"])
    w["B_any"] = w["high"] > w["B_ref"] * SANGTTA_RATIO
    w["B_univ"] = w["trade_value"] >= TV_FLOOR_MN
    w["B"] = w["B_any"] & w["B_univ"]

    # 상한가 마감 (보고용)
    d1 = w["d"] >= 1
    w["limit_close"] = np.where(d1, w["close"] >= w["prev_close"] * LIMIT_RATIO, False)
    pre = (w["d"] == 0) & (w["listing_date"] < D0_RULE_CHANGE)
    w.loc[pre, "limit_close"] = w.loc[pre, "close"] >= w.loc[pre, "open"] * LIMIT_RATIO
    w["d0_limit_undet"] = (w["d"] == 0) & (w["listing_date"] >= D0_RULE_CHANGE)
    w["close_eq_high"] = w["close"] >= w["high"]
    w["month"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m")
    w["year"] = pd.to_datetime(w["date"]).dt.year
    return w


def n_indep(ev: pd.DataFrame) -> int:
    """같은 종목 직전 이벤트와 경과일 차이 > INDEP_GAP (또는 첫 이벤트) 인 건수."""
    e = ev.sort_values(["code", "d"])
    gap = e["d"] - e.groupby("code")["d"].shift(1)
    return int((gap.isna() | (gap > INDEP_GAP)).sum())


def _pct(a, b):
    return f"{a / b:.1%}" if b else "—"


def report_block(title: str, ev: pd.DataFrame, mb_pairs: set | None):
    print(f"\n#### {title}\n")
    print("| 구간 | 건수 | 종목 수 | 상한가 마감 | 판정 불가(D0, 2023-06-26 이후) | 고가=종가(D0 대리) | 분봉 있는 (종목,일자) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, m in [("D0", ev["d"] == 0), ("D1~D20", (ev["d"] >= 1) & (ev["d"] <= 20)),
                     (f"D21~D{N_DAYS-1}", ev["d"] >= 21), ("합계", ev["d"] >= 0)]:
        e = ev[m]
        n = len(e)
        lim = int(e["limit_close"].sum())
        und = int(e["d0_limit_undet"].sum())
        ceh = int(e.loc[e["d0_limit_undet"], "close_eq_high"].sum())
        cov = "—"
        if mb_pairs is not None:
            have = sum((c, dt) in mb_pairs for c, dt in zip(e["code"], e["date"]))
            cov = f"{have:,} ({_pct(have, n)})"
        print(f"| {label} | {n:,} | {e['code'].nunique():,} | {lim:,} ({_pct(lim, max(n - und, 0))} of 판정가능) | {und:,} | {ceh:,} | {cov} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="분봉 미보유 (종목→수집 시작일) JSON")
    args = ap.parse_args()

    conn = _connect()
    first, daily, stocks, delisted, mb, sd = load(conn)
    conn.close()

    f = classify(first, stocks, delisted)
    newl = f[f["is_new"]]
    print("## 이벤트 규모 — 일봉 측정 (수익률 없음)\n")
    print(f"상장일 추정 = `daily_bars` 첫 거래일 (≥ {LISTING_CUTOFF}). N = {N_DAYS} 거래일. 거래대금 하한 {TV_FLOOR_MN/100:.0f}억.\n")
    print("### 신규 상장 종목 수 (첫 거래일 기준)\n")
    print("| 첫 거래일 연도 | 전체 | 스팩 | 리츠 | ETF/ETN류 | 이름 모름 | **유니버스(제외 후)** | 현재 상장 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    newl = newl.assign(y=pd.to_datetime(newl["first_date"]).dt.year)
    for y, grp in newl.groupby("y"):
        k = grp[grp["keep"]]
        print(f"| {y} | {len(grp)} | {int(grp['is_spac'].sum())} | {int(grp['is_reit'].sum())} | {int(grp['is_etp'].sum())} | "
              f"{int(grp['name_unknown'].sum())} | **{len(k)}** | {int(k['listed_now'].sum())} |")
    k_all = newl[newl["keep"]]
    print(f"| 합계 | {len(newl)} | {int(newl['is_spac'].sum())} | {int(newl['is_reit'].sum())} | {int(newl['is_etp'].sum())} | "
          f"{int(newl['name_unknown'].sum())} | **{len(k_all)}** | {int(k_all['listed_now'].sum())} |")
    print(f"\n`stocks` 에 있으나 이름 모름·폐지 목록 없음인 신규주 {int(newl['name_unknown'].sum())}개는 이름 필터를 못 거쳤다(유니버스에 남김).")
    if len(sd):
        sdm = k_all.merge(sd, on="code", how="left")
        has = sdm["sd_first"].notna()
        lag = (pd.to_datetime(sdm.loc[has, "sd_first"]) - pd.to_datetime(sdm.loc[has, "first_date"])).dt.days
        print(f"`supply_demand` 가 있는 신규주: {int(has.sum())}/{len(sdm)} · 첫 수급일 − 상장일 중앙값 {lag.median():.0f}일 "
              f"(2024 이후 상장만: {int(has[sdm['first_date'] >= date(2024,1,1)].sum())}/{int((sdm['first_date'] >= date(2024,1,1)).sum())})")

    w = compute(daily, set(k_all["code"]))
    mb_pairs = set(zip(mb["code"], mb["date"]))
    mb_codes = set(mb["code"])

    for (title, start) in [("최근 1년 (2025-08-01 ~ 2026-08-21)", RECENT_START), ("전 기간 (2016-10 ~ 2026-08-21)", LISTING_CUTOFF)]:
        ww = w[(w["date"] >= start) & (w["date"] <= END)]
        n_listings = ww.loc[ww["d"] == 0, "code"].nunique()
        n_days_univ = len(ww)
        print(f"\n### {title}\n")
        print(f"유니버스 (종목,일자) = {n_days_univ:,} (상장일이 구간 안인 종목 {n_listings}개 + 구간 전 상장·d<{N_DAYS} 잔여일)\n")
        print("| 단계 | (종목,일자) 건수 | 종목 수 |")
        print("|---|---:|---:|")
        A_any = ww[ww["A_any"]]; A_u = ww[ww["A_any"] & ww["A_univ"]]
        print(f"| (A) 고가 > 상장 후 최고가, d≥1 (전체) | {len(A_any):,} | {A_any['code'].nunique():,} |")
        print(f"| (A) + 상장 후 평균 거래대금 ≥ 50억 | {len(A_u):,} | {A_u['code'].nunique():,} |")
        print(f"|   └ 갭돌파 (시가가 이미 돌파선 위) — 제외 | {int((A_u['A_gap']).sum()):,} | {A_u.loc[A_u['A_gap'],'code'].nunique():,} |")
        Aev = ww[ww["A"]]
        print(f"|   └ **(A) 장중 돌파 (주 이벤트)** | **{len(Aev):,}** | **{Aev['code'].nunique():,}** |")
        B_any = ww[ww["B_any"]]; Bev = ww[ww["B"]]
        print(f"| (B) 고가 > 기준선 × 1.25 (전체) | {len(B_any):,} | {B_any['code'].nunique():,} |")
        print(f"|   └ **(B) + 당일 거래대금 ≥ 50억 (주 이벤트)** | **{len(Bev):,}** | **{Bev['code'].nunique():,}** |")
        both = ww[ww["A"] & ww["B"]]
        print(f"| (A)∩(B) 같은 (종목,일자) | {len(both):,} | {both['code'].nunique():,} |")
        report_block("(A) 신규주 돌파 — 경과일 구간별", Aev, mb_pairs if start == RECENT_START else None)
        report_block("(B) 신규주 상따 — 경과일 구간별", Bev, mb_pairs if start == RECENT_START else None)

        if start == RECENT_START:
            print("\n#### 월별 (최근 1년)\n")
            print("| 월 | 신규 상장 | (A) 건수 | (A) 종목 | (B) 건수 | (B) 종목 | (B) D0 |")
            print("|---|---:|---:|---:|---:|---:|---:|")
            months = sorted(ww["month"].unique())
            for m in months:
                x = ww[ww["month"] == m]
                a = x[x["A"]]; b = x[x["B"]]
                print(f"| {m} | {x.loc[x['d']==0,'code'].nunique()} | {len(a)} | {a['code'].nunique()} | {len(b)} | {b['code'].nunique()} | {int((b['d']==0).sum())} |")
            print("\n#### 분기별 (최근 1년)\n")
            print("| 분기 | (A) 건수 | (B) 건수 |")
            print("|---|---:|---:|")
            ww2 = ww.assign(q=pd.to_datetime(ww["date"]).dt.to_period("Q").astype(str))
            for q, x in ww2.groupby("q"):
                print(f"| {q} | {int(x['A'].sum())} | {int(x['B'].sum())} |")
            for nm_, ev in [("A", Aev), ("B", Bev)]:
                pc = ev.groupby("code").size()
                pdy = ev.groupby("date").size()
                if len(pc):
                    print(f"\n({nm_}) 종목당 건수 중앙 {pc.median():.0f} · 최대 {pc.max()} · 이벤트 있는 날 {len(pdy)} · 하루 최대 {pdy.max()} · "
                          f"독립 이벤트(직전 {INDEP_GAP}거래일 내 같은 종목 이벤트 없음) {n_indep(ev):,}")
            # 만쥬 4종목 확인 — 이벤트 플래그만
            print("\n#### 만쥬 신규주 4종목 — 정의상 이벤트 플래그 (수익률 아님)\n")
            print("| 종목 | 상장일 | (A) 이벤트일(d) | (B) 이벤트일(d) | 분봉 |")
            print("|---|---|---|---|---|")
            for c, nm in MANJU_CODES.items():
                x = w[w["code"] == c]
                if x.empty:
                    print(f"| {nm} {c} | 유니버스 밖 | | | |"); continue
                la = ", ".join(f"{r.date}(d{r.d})" for r in x[x["A"]].itertuples()) or "없음"
                lb = ", ".join(f"{r.date}(d{r.d})" for r in x[x["B"]].itertuples()) or "없음"
                print(f"| {nm} {c} | {x['listing_date'].iloc[0]} | {la} | {lb} | {'有' if c in mb_codes else '없음'} |")
        else:
            delisted_codes = set(f.loc[~f["listed_now"], "code"])
            for nm_, ev in [("A", Aev), ("B", Bev)]:
                print(f"({nm_}) 독립 이벤트 {n_indep(ev):,} · 현재 비상장 종목의 이벤트 {int(ev['code'].isin(delisted_codes).sum()):,}건 "
                      f"({ev.loc[ev['code'].isin(delisted_codes),'code'].nunique()}종목)")
            print("\n#### 연도별 (전 기간)\n")
            print("| 연도 | 신규 상장(유니버스) | (A) 건수 | (A) 종목 | (A) 상한가 마감 | (B) 건수 | (B) 종목 | (B) D0 | (B) 상한가 마감(판정가능분) |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for y, x in ww.groupby("year"):
                a = x[x["A"]]; b = x[x["B"]]
                bdet = b[~b["d0_limit_undet"]]
                print(f"| {y} | {x.loc[x['d']==0,'code'].nunique()} | {len(a)} | {a['code'].nunique()} | {_pct(int(a['limit_close'].sum()), len(a))} | "
                      f"{len(b)} | {b['code'].nunique()} | {int((b['d']==0).sum())} | {_pct(int(bdet['limit_close'].sum()), len(bdet))} |")

    # 수집 큐 — 최근 1년 (A)∪(B) 이벤트 중 분봉 없는 (종목, 최초이벤트일 − 10일)
    rec = w[(w["date"] >= RECENT_START) & (w["date"] <= END) & (w["A"] | w["B"])]
    need = {}
    first_ev = rec.groupby("code")["date"].min()
    for code, dt in first_ev.items():
        if code not in mb_codes:
            need[code] = (pd.Timestamp(dt) - timedelta(days=NEED_LEAD_DAYS)).strftime("%Y%m%d")
    n_ev_need = int(rec["code"].isin(need).sum())
    have_pairs = sum((c, dt) in mb_pairs for c, dt in zip(rec["code"], rec["date"]))
    print(f"\n### 분봉 커버리지 (최근 1년, (A)∪(B) {len(rec):,}건 / {rec['code'].nunique()}종목)\n")
    print(f"- (종목,일자) 단위로 분봉 있음: {have_pairs:,} ({_pct(have_pairs, len(rec))})")
    print(f"- 종목 단위로 `minute_bars` 에 전혀 없음: {len(need)}종목, 이벤트 {n_ev_need:,}건 ({_pct(n_ev_need, len(rec))})")
    partial = rec[rec["code"].isin(mb_codes)]
    miss_partial = sum((c, dt) not in mb_pairs for c, dt in zip(partial["code"], partial["date"]))
    print(f"- 종목은 있으나 해당 일자 분봉 없음: {miss_partial:,}건 (수집 시작일이 이벤트 뒤인 경우 — 재수집 필요)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(need, fh, indent=0, sort_keys=True)
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
