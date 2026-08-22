"""짝꿍매매 — 이벤트 규모 측정 (수익률 계산 없음).

`docs/research/manju/12-pair-preregistration.md` 사전등록의 부록이다. **트리거 시점까지의
정보와 표본 크기만** 센다. 진입 이후 가격은 어떤 형태로도 읽지 않는다 — 유일한 예외는
진입 분(T+1) 시가가 상한가라 체결 불가인 건을 제외하기 위해 그 한 봉의 시가를 보는 것이다.

측정하는 것
1. 대장 트리거: 분봉 종가가 처음 상한가와 같아진 분(T). 상한가 = 전일 원주가 종가
   (`daily_bars`) × 1.3 을 호가단위로 내림. 트리거 시점 누적 거래대금 ≥ 300억, T ≤ 15:13.
2. 후속 후보 (테마 무시): T 시점 등락률 +10~15%, 누적 거래대금 ≥ 50억, 분봉 있음.
3. 클러스터 일치 후보: 위 중 대장과의 **잔차 상관 순위가 유니버스 상위 30 이내 & 상관 > 0**
   (직전 60거래일 일봉 로그수익률을 소속 시장지수(KOSPI 001 / KOSDAQ 101)에 회귀한 잔차,
   겹치는 날 ≥ 40). 절대 임계 0.4 는 기술통계로만 병기한다.
4. 분봉 없는 후속 후보 (일봉 근사: 당일 고가 ≥ +10%, 일 거래대금 ≥ 50억, 상관 상위 30)
   → `pair_need.json` ({code: 최초이벤트일-10일}).
5. 월별 분포, 잠김 시각 구간, 점상 여부, 제외 사유별 건수.

DB 는 읽기만 한다. 실행:

    uv run --extra pg python scripts/pair_events.py --start 2025-08-04 --end 2026-08-21 \\
        --out /path/pair_events.md --need /path/pair_need.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from limitup_timing import _load_env, limit_up_price

from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect

# ── 사전 고정 파라미터 (문서 12번과 같아야 한다) ────────────────────────────
LEAD_CUM_TV_MIN = 300e8      # 대장: 트리거 시점까지 누적 거래대금 ≥ 300억 (원)
FOL_CUM_TV_MIN = 50e8        # 후속: 트리거 시점까지 누적 거래대금 ≥ 50억 (원)
FOL_RET_LO, FOL_RET_HI = 0.10, 0.15   # 후속 등락률 구간 (T 시점, 전일 원주가 종가 대비)
CORR_WINDOW = 60             # 잔차 상관 창 (거래일, t-60 ~ t-1)
CORR_MIN_OVERLAP = 40        # 겹치는 날 최소
TOP_K = 30                   # "같은 테마" = 대장의 잔차 상관 상위 30 종목 (유니버스 ~2,600 의 약 1%) & 상관 > 0
CORR_MIN = 0.4               # 보조 기술통계용 임계 (판정에 쓰지 않는다)
LAST_TRIGGER_MINUTE = 15 * 60 + 13   # T ≤ 15:13 (T+6 시가 = 15:19 가 존재해야 함)
DAILY_FOL_HIGH_MIN = 0.10    # 분봉 없는 후속 후보 근사: 일 고가 ≥ +10%
DAILY_FOL_TV_MIN = 50e8      # 〃 일 거래대금 ≥ 50억 (daily_bars.trade_value 는 백만원)
NEED_LEAD_DAYS = 10          # need 파일: 최초 이벤트일 − 10일

_CLOSE_MINUTE = 15 * 60 + 30
BUCKETS = [
    ("09:00(점상)", 9 * 60, 9 * 60 + 1),
    ("09:01~10:00", 9 * 60 + 1, 10 * 60),
    ("10:00~12:00", 10 * 60, 12 * 60),
    ("12:00~14:00", 12 * 60, 14 * 60),
    ("14:00~15:13", 14 * 60, LAST_TRIGGER_MINUTE + 1),
]
MARKET_INDEX = {"거래소": "001", "코스닥": "101"}


def _q(conn, sql: str, params=()) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    return df


def load_daily_raw(conn, start: str, end: str) -> pd.DataFrame:
    df = _q(conn, "select code, date, open, high, low, close, trade_value from daily_bars "
                  "where date >= %s and date <= %s order by code, date", (start, end))
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "trade_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    return df


def load_daily_adj_close(conn, start: str, end: str) -> pd.DataFrame:
    df = _q(conn, "select code, date, close from daily_bars_adjusted "
                  "where date >= %s and date <= %s", (start, end))
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.pivot(index="date", columns="code", values="close").sort_index()


def load_index_close(conn, start: str, end: str) -> pd.DataFrame:
    df = _q(conn, "select code, date, close from sector_index where code in ('001','101') "
                  "and date >= %s and date <= %s", (start, end))
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="code", values="close").sort_index().astype(float)


def residual_corr_with(px: pd.DataFrame, idx: pd.DataFrame, market_of: dict, date: pd.Timestamp,
                       leader: str) -> pd.Series:
    """``date`` 직전 CORR_WINDOW 거래일의 잔차 수익률로 대장과 전 종목의 상관을 계산한다.

    잔차 = 종목 로그수익률 − (α + β × 소속 시장지수 로그수익률). 겹치는 날 < 40 이면 NaN.
    """
    days = px.index[px.index < date][-CORR_WINDOW:]
    if len(days) < CORR_MIN_OVERLAP or leader not in px.columns:
        return pd.Series(dtype=float)
    r = np.log(px.loc[days]).diff().iloc[1:]
    m = np.log(idx.reindex(days)).diff().iloc[1:]
    resid = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for mkt, icode in MARKET_INDEX.items():
        cols = [c for c in r.columns if market_of.get(c, "거래소") == mkt]
        if not cols:
            continue
        sub = r[cols]
        mm = m[icode]
        valid = sub.notna()
        x = mm.to_numpy()[:, None] * np.ones((1, len(cols)))
        x = np.where(valid, x, np.nan)
        y = sub.to_numpy()
        xm = np.nanmean(x, axis=0)
        ym = np.nanmean(y, axis=0)
        cov = np.nanmean((x - xm) * (y - ym), axis=0)
        var = np.nanmean((x - xm) ** 2, axis=0)
        beta = np.where(var > 0, cov / var, 0.0)
        alpha = ym - beta * xm
        resid[cols] = y - (alpha + beta * x)
    if leader not in resid.columns or resid[leader].notna().sum() < CORR_MIN_OVERLAP:
        return pd.Series(dtype=float)
    lead = resid[leader]
    overlap = resid.notna().mul(lead.notna(), axis=0).sum()
    corr = resid.corrwith(lead)
    corr[overlap < CORR_MIN_OVERLAP] = np.nan
    corr = corr.drop(index=leader, errors="ignore")
    return corr


def scan_date(day_bars: pd.DataFrame, raw_day: pd.DataFrame) -> tuple[list[dict], dict, pd.DataFrame]:
    """하루치 분봉(전 종목)에서 대장 트리거와 후속 후보 스냅샷을 만든다.

    Returns:
        triggers: 대장 트리거 목록 (dict)
        excl: 제외 사유 카운트
        snap: 종목별 분 단위 상태 (code, minute → close, cum_tv) — 후속 판정용 wide 프레임
    """
    excl: dict[str, int] = {}
    bars = day_bars[day_bars["ts"].dt.hour * 60 + day_bars["ts"].dt.minute <= _CLOSE_MINUTE].copy()
    bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
    prev = raw_day.set_index("code")["prev_close"]
    raw_close = raw_day.set_index("code")["close"]

    # 분봉이 원주가인지 확인: 15:30 봉 종가 == 일봉 원주가 종가 (0.2% 허용)
    last = bars.groupby("code")["close"].last()
    ok_raw = {}
    for code, c in last.items():
        rc = raw_close.get(code, np.nan)
        ok_raw[code] = bool(np.isfinite(rc) and rc > 0 and abs(c / rc - 1) <= 0.002)
    bars = bars[bars["code"].map(ok_raw).fillna(False).astype(bool)]
    n_adj = sum(1 for v in ok_raw.values() if not v)
    if n_adj:
        excl["분봉·일봉 종가 불일치(수정주가 의심) 종목-일"] = n_adj

    bars["cum_tv"] = bars.groupby("code")["trade_value"].cumsum()
    lim_by_code = {c: limit_up_price(p) for c, p in prev.items() if np.isfinite(p) and p > 0}
    bars["limit"] = bars["code"].map(lim_by_code)
    bars = bars[bars["limit"].notna()]

    close_w = bars.pivot(index="minute", columns="code", values="close").ffill()
    open_w = bars.pivot(index="minute", columns="code", values="open")
    cumtv_w = bars.pivot(index="minute", columns="code", values="cum_tv").ffill()
    limit_s = bars.groupby("code")["limit"].first()
    prev_s = prev.reindex(close_w.columns)

    triggers: list[dict] = []
    # 대장: 봉 종가 == 상한가 인 첫 분
    at_limit = (close_w - limit_s).abs() < 1e-6
    at_limit = at_limit & bars.pivot(index="minute", columns="code", values="close").notna()  # ffill 된 분은 제외
    high_w = bars.pivot(index="minute", columns="code", values="high")
    touched = (high_w - limit_s) >= -1e-6
    for code in at_limit.columns:
        col = at_limit[code]
        if not col.any():
            continue
        T = int(col.idxmax())
        first_touch = int(touched[code].idxmax()) if touched[code].any() else T
        cum = float(cumtv_w.at[T, code])
        if T > LAST_TRIGGER_MINUTE:
            excl["대장 잠김 15:13 이후"] = excl.get("대장 잠김 15:13 이후", 0) + 1
            continue
        if cum < LEAD_CUM_TV_MIN:
            excl["대장 누적 거래대금 < 300억"] = excl.get("대장 누적 거래대금 < 300억", 0) + 1
            continue
        o = open_w.at[T, code] if T in open_w.index else np.nan
        triggers.append({
            "lead": code, "T": T, "first_touch": first_touch,
            "touch_to_lock": T - first_touch,
            "lead_cum_tv": cum,
            "gap_lock": bool(T == 9 * 60 and np.isfinite(o) and abs(o - limit_s[code]) < 1e-6),
            "lead_limit": float(limit_s[code]), "lead_prev_close": float(prev_s[code]),
        })

    snap = {"close": close_w, "open": open_w, "cum_tv": cumtv_w, "limit": limit_s, "prev": prev_s,
            "has_bar": bars.pivot(index="minute", columns="code", values="close").notna()}
    return triggers, excl, snap


def follower_candidates(snap: dict, trig: dict) -> tuple[pd.DataFrame, dict]:
    """T 시점 후속 후보(테마 무시). 제외 사유도 센다."""
    T = trig["T"]
    lead = trig["lead"]
    close_w, open_w, cum_w = snap["close"], snap["open"], snap["cum_tv"]
    excl: dict[str, int] = {}
    if T not in close_w.index:
        return pd.DataFrame(), excl
    c = close_w.loc[T].drop(index=lead, errors="ignore")
    ret = c / snap["prev"].reindex(c.index) - 1
    cum = cum_w.loc[T].reindex(c.index)
    cand = pd.DataFrame({"ret": ret, "cum_tv": cum, "limit": snap["limit"].reindex(c.index)})
    cand = cand[(cand["ret"] >= FOL_RET_LO) & (cand["ret"] <= FOL_RET_HI)]
    cand = cand[cand["cum_tv"] >= FOL_CUM_TV_MIN]
    n0 = len(cand)
    # 진입 분 T+1 봉 존재 & 시가 < 상한가
    T1 = T + 1
    if T1 in open_w.index:
        o1 = open_w.loc[T1].reindex(cand.index)
    else:
        o1 = pd.Series(np.nan, index=cand.index)
    no_bar = o1.isna()
    at_lim = (o1 - cand["limit"]).abs() < 1e-6
    excl["후속 T+1 봉 없음(VI·무체결)"] = int(no_bar.sum())
    excl["후속 T+1 시가 = 상한가(체결 불가)"] = int((at_lim & ~no_bar).sum())
    cand = cand[~no_bar & ~at_lim]
    excl["_n_raw"] = n0
    return cand, excl


SANITY_PAIRS = [
    # (일자, 대장, 후속) — 알려진 테마 짝. 상관 프록시가 깨지지 않았는지 눈으로 확인하는 용도.
    ("2025-10-14", "108490", "090360"),   # 로보티즈 → 로보스타
    ("2026-01-05", "247540", "086520"),   # 에코프로비엠 → 에코프로
    ("2026-01-05", "005930", "000660"),   # 삼성전자 → SK하이닉스 (지수 회귀가 흡수 — 낮게 나와야 정상)
]


def sanity(conn) -> None:
    """알려진 짝에 대해 잔차 상관과 순위를 찍는다. 프록시 자체의 점검이지 측정이 아니다."""
    px = load_daily_adj_close(conn, "2025-06-01", "2026-08-21")
    idx = load_index_close(conn, "2025-06-01", "2026-08-21")
    st = _q(conn, "select code, market, name from stocks")
    mk = dict(zip(st["code"], st["market"]))
    nm = dict(zip(st["code"], st["name"]))
    for d, lead, fol in SANITY_PAIRS:
        corr = residual_corr_with(px, idx, mk, pd.Timestamp(d), lead)
        rk = corr.rank(ascending=False, method="min")
        print(f"{d} {nm.get(lead, lead)} → {nm.get(fol, fol)}: corr {corr.get(fol, float('nan')):.3f}, "
              f"rank {rk.get(fol, float('nan')):.0f} / {int(corr.notna().sum())}")
        top = corr.sort_values(ascending=False).head(8)
        print("   상위 8:", ", ".join(f"{nm.get(k, k)} {v:.2f}" for k, v in top.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="짝꿍매매 이벤트 규모 (읽기 전용, 수익률 없음)")
    ap.add_argument("--start", default="2025-08-04")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--out", required=False)
    ap.add_argument("--need", required=False)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--sanity", action="store_true", help="알려진 테마 짝의 상관·순위만 찍고 끝")
    args = ap.parse_args(argv)

    _load_env()
    conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
    if args.sanity:
        sanity(conn)
        return 0
    if not (args.out and args.need):
        ap.error("--out 과 --need 가 필요하다")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    warm = (start - timedelta(days=130)).strftime("%Y-%m-%d")

    print("일봉 로드 …", file=sys.stderr)
    raw = load_daily_raw(conn, warm, args.end)
    px = load_daily_adj_close(conn, warm, args.end)
    idx = load_index_close(conn, warm, args.end)
    stocks = _q(conn, "select code, market, sector, name from stocks")
    market_of = dict(zip(stocks["code"], stocks["market"]))
    name_of = dict(zip(stocks["code"], stocks["name"]))
    minute_codes = set(_q(conn, "select distinct code from minute_bars")["code"].astype(str))
    trading_days = [d for d in idx.index if start <= d <= end]

    all_trig: list[dict] = []
    all_pairs: list[dict] = []
    excl_total: dict[str, int] = {}
    corr_all: list[float] = []
    rank_all: list[float] = []
    need: dict[str, str] = {}
    need_lead: dict[str, str] = {}
    n_daily_lead_no_minute = 0
    n_daily_fol_cluster = 0
    n_daily_fol_cluster_has_min = 0

    for i, d in enumerate(trading_days):
        ds = d.strftime("%Y-%m-%d")
        raw_day = raw[raw["date"] == d]
        # 일봉 기준 대장 이벤트(원주가 고가 ≥ 상한가) 중 분봉 없는 종목 → need
        lim_day = raw_day["prev_close"].map(lambda p: limit_up_price(p) if np.isfinite(p) else np.nan)
        lead_daily = raw_day[(raw_day["high"] >= lim_day - 1e-6) & (raw_day["trade_value"] * 1e6 >= LEAD_CUM_TV_MIN)]
        for code in lead_daily["code"]:
            if code not in minute_codes:
                n_daily_lead_no_minute += 1
                need_lead.setdefault(code, (d - timedelta(days=NEED_LEAD_DAYS)).strftime("%Y%m%d"))

        day_bars = load_minute_bars(conn, ds, ds)
        if day_bars.empty:
            continue
        trigs, excl, snap = scan_date(day_bars, raw_day)
        for k, v in excl.items():
            excl_total[k] = excl_total.get(k, 0) + v
        for t in trigs:
            t["date"] = d
            corr = residual_corr_with(px, idx, market_of, d, t["lead"])
            cand, ex2 = follower_candidates(snap, t)
            for k, v in ex2.items():
                if not k.startswith("_"):
                    excl_total[k] = excl_total.get(k, 0) + v
            cand["corr"] = corr.reindex(cand.index)
            if len(corr):
                rank_s = corr.rank(ascending=False, method="min")
                cand["rank"] = rank_s.reindex(cand.index)
            else:
                cand["rank"] = np.nan
            cand["cluster"] = (cand["rank"] <= TOP_K) & (cand["corr"] > 0)
            corr_all.extend(cand["corr"].dropna().tolist())
            rank_all.extend(cand["rank"].dropna().tolist())
            t["n_fol_raw"] = int(ex2.get("_n_raw", 0))
            t["n_fol"] = len(cand)
            t["n_fol_cluster"] = int(cand["cluster"].sum())
            t["n_fol_corr04"] = int((cand["corr"] >= CORR_MIN).sum())
            t["n_fol_corr_nan"] = int(cand["corr"].isna().sum())
            for code, row in cand.iterrows():
                all_pairs.append({"date": d, "lead": t["lead"], "fol": code, "T": t["T"],
                                  "ret": row["ret"], "cum_tv": row["cum_tv"], "corr": row["corr"],
                                  "rank": row["rank"], "cluster": bool(row["cluster"])})
            # 분봉 없는 후속 후보 (일봉 근사) — 클러스터 일치만
            if len(corr):
                rank_s = corr.rank(ascending=False, method="min")
                cl = corr[(rank_s <= TOP_K) & (corr > 0)].index
                fd = raw_day[raw_day["code"].isin(cl)]
                fd = fd[(fd["high"] / fd["prev_close"] - 1 >= DAILY_FOL_HIGH_MIN)
                        & (fd["trade_value"] * 1e6 >= DAILY_FOL_TV_MIN)]
                n_daily_fol_cluster += len(fd)
                for code in fd["code"]:
                    if code in minute_codes:
                        n_daily_fol_cluster_has_min += 1
                    else:
                        need.setdefault(code, (d - timedelta(days=NEED_LEAD_DAYS)).strftime("%Y%m%d"))
                t["n_fol_daily_cluster"] = len(fd)
            else:
                t["n_fol_daily_cluster"] = 0
            all_trig.append(t)
        if i % 20 == 0:
            print(f"  {ds} 누적 트리거 {len(all_trig)}", file=sys.stderr)

    trig = pd.DataFrame(all_trig)
    pairs = pd.DataFrame(all_pairs)
    report = build_report(trig, pairs, excl_total, corr_all, rank_all, args, {
        "daily_lead_no_minute": n_daily_lead_no_minute,
        "daily_fol_cluster": n_daily_fol_cluster,
        "daily_fol_cluster_has_min": n_daily_fol_cluster_has_min,
        "need_fol": len(need), "need_lead": len(need_lead),
        "minute_codes": len(minute_codes),
    }, name_of)
    pathlib.Path(args.out).write_text(report, encoding="utf-8")
    merged = dict(need)
    for k, v in need_lead.items():
        merged[k] = min(merged.get(k, v), v)
    pathlib.Path(args.need).write_text(json.dumps(merged, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(report)
    return 0


def _bucket(m: int) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= m < hi:
            return label
    return "기타"


def build_report(trig: pd.DataFrame, pairs: pd.DataFrame, excl: dict, corr_all: list, rank_all: list,
                 args, meta: dict, name_of: dict) -> str:
    L: list[str] = []
    L.append(f"# 짝꿍매매 이벤트 규모 — {args.start} ~ {args.end} (수익률 없음)\n")
    L.append(f"분봉 보유 종목 {meta['minute_codes']}개. 파라미터: 대장 누적 거래대금 ≥ {LEAD_CUM_TV_MIN/1e8:.0f}억, "
             f"T ≤ 15:13, 후속 등락률 [{FOL_RET_LO:.0%}, {FOL_RET_HI:.0%}], 후속 누적 거래대금 ≥ {FOL_CUM_TV_MIN/1e8:.0f}억, "
             f"잔차 상관 창 {CORR_WINDOW}일 · 겹침 ≥ {CORR_MIN_OVERLAP} · 클러스터 = 대장 상관 상위 {TOP_K} & 상관 > 0.\n")
    if trig.empty:
        L.append("트리거 0건.")
        return "\n".join(L)
    n = len(trig)
    L.append("## 1. 대장 트리거\n")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| 대장 트리거(첫 잠김 분) 건수 | **{n}** ({trig['lead'].nunique()} 종목, {trig['date'].nunique()} 거래일) |")
    L.append(f"| 점상(09:00 봉 시가=상한가) | {int(trig['gap_lock'].sum())} ({trig['gap_lock'].mean():.0%}) |")
    L.append(f"| 처음 닿은 분 = 잠긴 분 (touch_to_lock=0) | {int((trig['touch_to_lock']==0).sum())} ({(trig['touch_to_lock']==0).mean():.0%}) |")
    L.append(f"| touch_to_lock 중앙값 / p75 / p90 (분) | {trig['touch_to_lock'].median():.0f} / {trig['touch_to_lock'].quantile(.75):.0f} / {trig['touch_to_lock'].quantile(.9):.0f} |")
    L.append(f"| 대장 누적 거래대금 중앙값 / p25 / p75 (억) | {trig['lead_cum_tv'].median()/1e8:,.0f} / {trig['lead_cum_tv'].quantile(.25)/1e8:,.0f} / {trig['lead_cum_tv'].quantile(.75)/1e8:,.0f} |")
    L.append(f"| 일봉 기준 대장 이벤트(고가≥상한가 & 일 거래대금≥300억) 중 분봉 없는 건 | {meta['daily_lead_no_minute']} (종목 {meta['need_lead']}) |")
    L.append("")
    L.append("### 잠김 시각 구간\n")
    trig["bucket"] = trig["T"].map(_bucket)
    L.append("| 구간 | 트리거 | 비율 | 후속 후보(테마무시) 평균 | 클러스터 일치 평균 | ≥1 클러스터 후보 트리거 |")
    L.append("|---|---|---|---|---|---|")
    for label, _, _ in BUCKETS:
        g = trig[trig["bucket"] == label]
        if g.empty:
            L.append(f"| {label} | 0 | 0% | — | — | — |")
            continue
        L.append(f"| {label} | {len(g)} | {len(g)/n:.0%} | {g['n_fol'].mean():.1f} | {g['n_fol_cluster'].mean():.2f} | {int((g['n_fol_cluster']>0).sum())} |")
    L.append("")
    L.append("### 월별\n")
    trig["month"] = trig["date"].dt.strftime("%Y-%m")
    L.append("| 월 | 트리거 | 후속 후보 합(테마무시) | 클러스터 일치 페어 | ≥1 클러스터 후보 트리거 | 분봉없는 클러스터 후보(일봉근사) |")
    L.append("|---|---|---|---|---|---|")
    for m, g in trig.groupby("month"):
        L.append(f"| {m} | {len(g)} | {int(g['n_fol'].sum())} | {int(g['n_fol_cluster'].sum())} | {int((g['n_fol_cluster']>0).sum())} | {int(g['n_fol_daily_cluster'].sum()) - 0} |")
    L.append(f"| **합계** | **{n}** | **{int(trig['n_fol'].sum())}** | **{int(trig['n_fol_cluster'].sum())}** | **{int((trig['n_fol_cluster']>0).sum())}** | {int(trig['n_fol_daily_cluster'].sum())} |")
    L.append("")
    L.append("## 2. 후속 후보 수 분포 (트리거당)\n")
    L.append("| 기준 | 0 | 1 | 2~5 | 6~10 | 11+ | 평균 | 중앙값 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, col in (("테마 무시 (+10~15% · 50억 · 분봉 · 체결가능)", "n_fol"),
                       (f"클러스터 일치 (대장 상관 상위 {TOP_K} & >0) — 주 정의", "n_fol_cluster"),
                       ("(참고) 잔차 상관 ≥ 0.4", "n_fol_corr04")):
        s = trig[col]
        L.append(f"| {label} | {int((s==0).sum())} | {int((s==1).sum())} | {int(((s>=2)&(s<=5)).sum())} | "
                 f"{int(((s>=6)&(s<=10)).sum())} | {int((s>=11).sum())} | {s.mean():.2f} | {s.median():.0f} |")
    L.append("")
    L.append(f"테마 무시 후보 중 상관 계산 불가(겹침<40일·신규상장 등): {int(trig['n_fol_corr_nan'].sum())} 쌍.\n")
    if corr_all:
        c = pd.Series(corr_all)
        L.append("### 테마 무시 후보의 대장과의 잔차 상관 분포\n")
        L.append("| n | p10 | p25 | p50 | p75 | p90 | ≥0.4 비율 | ≥0.3 | ≥0.5 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        L.append(f"| {len(c)} | {c.quantile(.1):.2f} | {c.quantile(.25):.2f} | {c.median():.2f} | {c.quantile(.75):.2f} | "
                 f"{c.quantile(.9):.2f} | {(c>=0.4).mean():.1%} | {(c>=0.3).mean():.1%} | {(c>=0.5).mean():.1%} |")
        L.append("")
        L.append("(절대 임계 0.3/0.4/0.5 는 판정에 쓰지 않는다. 주 정의는 순위(상위 30)다. 분포를 보이기 위한 기술통계.)\n")
    if rank_all:
        rk = pd.Series(rank_all)
        L.append("### 테마 무시 후보의 대장 상관 순위 분포 (유니버스 내, 1 = 가장 높음)\n")
        L.append("| n | p10 | p25 | p50 | p75 | ≤10 비율 | ≤30 (주 정의) | ≤100 |")
        L.append("|---|---|---|---|---|---|---|---|")
        L.append(f"| {len(rk)} | {rk.quantile(.1):.0f} | {rk.quantile(.25):.0f} | {rk.median():.0f} | {rk.quantile(.75):.0f} | "
                 f"{(rk<=10).mean():.1%} | {(rk<=30).mean():.1%} | {(rk<=100).mean():.1%} |")
        L.append("")
    L.append(f"## 3. 페어 (주 정의: 대장 상관 상위 {TOP_K} 클러스터 일치, 트리거당 상관 최고 1건)\n")
    if not pairs.empty:
        cl = pairs[pairs["cluster"]]
        top1 = cl.sort_values(["date", "lead", "corr"], ascending=[True, True, False]).groupby(["date", "lead"]).head(1)
        L.append("| 항목 | 값 |")
        L.append("|---|---|")
        L.append(f"| 테마 무시 페어 (분봉 동시 보유·체결 가능) | {len(pairs)} |")
        L.append(f"| 클러스터 일치 페어 | **{len(cl)}** |")
        L.append(f"| 주 규칙 거래 수 (트리거당 상관 최고 1건) | **{len(top1)}** |")
        L.append(f"| 〃 후속 종목 수 / 대장 종목 수 | {top1['fol'].nunique()} / {top1['lead'].nunique()} |")
        L.append(f"| 〃 같은 후속주 재등장 최대 횟수 | {top1['fol'].value_counts().max()} |")
        L.append(f"| 〃 후속 등락률 중앙값 (T 시점) | {top1['ret'].median():.1%} |")
        L.append(f"| 〃 후속 누적 거래대금 중앙값 (억) | {top1['cum_tv'].median()/1e8:,.0f} |")
        L.append(f"| 〃 상관 중앙값 | {top1['corr'].median():.2f} |")
        L.append(f"| 테마 무시 비교용 거래 수 (트리거당 누적 거래대금 최고 1건) | {pairs.sort_values('cum_tv', ascending=False).groupby(['date','lead']).head(1).shape[0]} |")
        L.append("")
        vc = top1["fol"].value_counts().head(8)
        L.append("주 규칙 후속주 최다 등장 (종목명 — 횟수): " + ", ".join(f"{name_of.get(k, k)} — {v}" for k, v in vc.items()) + "\n")
    L.append("## 4. 분봉 커버리지 (클러스터 일치 후속 후보, 일봉 근사)\n")
    tot = meta["daily_fol_cluster"]
    L.append(f"일봉 근사(당일 고가 ≥ +10% · 일 거래대금 ≥ 50억 · 대장 상관 상위 {TOP_K}) 후속 후보 {tot} 건 중 분봉 있음 "
             f"{meta['daily_fol_cluster_has_min']} ({(meta['daily_fol_cluster_has_min']/tot if tot else 0):.0%}). "
             f"분봉 없는 후속 종목 {meta['need_fol']}개, 분봉 없는 대장 종목 {meta['need_lead']}개 → `pair_need.json`.\n")
    L.append("## 5. 제외 사유\n")
    L.append("| 사유 | 건수 |")
    L.append("|---|---|")
    for k, v in sorted(excl.items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
