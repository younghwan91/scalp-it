"""108번 일중 수익의 시간대 분해 — **기술(description)이며 판정이 아니다.**

    /home/young/git/scalp-it/.venv/bin/python scripts/minute_108/tod.py \\
        --from 2025-08-01 --to 2026-09-16 --extra-from 2026-09-17 --extra-to 2026-10-02 \\
        --out data/research108

질문 하나: 전일 거래대금 상위 300 의 **일중 −28bp(97번)는 하루 중 어디에 있는가.** 균등한가, 특정 창에
몰려 있는가, 몰려 있다면 그 창을 **뺀 나머지** 가 비용(41.5 / 23bp)을 넘는 버킷이 있는가.

정의(코드 전에 고정, 바꾸지 않는다):
- 창 경계 10개: 09:00 · 09:05 · 09:30 · 10:00 · 11:00 · 13:00 · 14:00 · 15:00 · 15:20 · 15:30, 그리고 야간(15:30 → 익일 09:00).
- 경계 T 의 가격 = **T 직전 마지막 분봉의 close**(ts < T 인 마지막 체결가). 09:00 = 09:00 분봉 `open`(개장 단일가),
  15:30 = 15:30 분봉 `close`(마감 단일가). 15:20 = 15:19 분봉 close(연속 세션 마지막 체결). 분봉은 체결이 있는 분에만
  있으므로 직전 분이 비면 더 앞의 분으로 간다(그 나이는 `stale_*` 로 보고). 창 수익은 서로 곱해 **정확히 시가→종가**
  가 된다(텔레스코핑).
- `daily_bars.close` 는 2026-09-14 부터 시간외(16:00~20:00) 체결이 섞여 15:30 단일가와 다르다(일치율 13%). 그래서
  종가·전일 종가·갭은 전부 **분봉 15:30 close** 로 만든다. 일봉은 전일 거래대금 순위(유니버스)와 1,000원 하한에만 쓴다.
- 유니버스·버킷: **전일** 거래대금 순위(전일 종가 ≥ 1,000원) 1~100 / 101~200 / 201~300 / 301~400 / 401~1000.
  당일 값은 어디에도 쓰지 않는다(97 의 부호 전환이 당일 순위에서 났다). 대조로 "당일 순위" 버킷을 따로 계산해
  look-ahead 가 숫자를 어떻게 바꾸는지 보고만 한다.
- 조건부(전부 d 의 09:00 이전에 아는 값): 전일 수익률(d−1 15:30 / d−2 15:30) 5분위, 갭(d 09:00 / d−1 15:30)
  부호·크기 3분위. 분위는 그날 유니버스(1~1000) 안에서.
- 통계: 풀링 평균·중앙·양(+) 비율, **일평균 열의 Newey-West t**(lag 5, `krx_quant_core.stats.metrics.newey_west_t`
  via `backtest.drift.drift_summary`), 음수일·음수월 비율. 비용은 **빼지 않는다** — 표에 41.5bp(연속 체결) /
  23bp(양쪽 단일가 체결 시 세금+수수료만, 스프레드 가정 미검증) 두 기준선을 병기한다.
- 불량일(`research/bad_days.json`)은 표본 행에서 뺀다. 단 그 날의 09:00·15:30 단일가는 이웃 날의 야간·갭 계산에
  쓴다(단일가는 분봉 수집 불량과 무관한 하나의 값이다).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from itertools import pairwise
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _envload  # noqa: F401
from krx_quant_core.backtest.drift import drift_summary, rank_buckets
from krx_quant_core.stats.trials import record_trial

ROOT = Path(__file__).resolve().parents[2]
BAD_DAYS = ROOT / "research" / "bad_days.json"
CACHE = Path.home() / "of80" / "tod108"

COST_CONT_BP = 41.5      # 연속 체결 왕복: 세금 20 + 수수료 3 + 스프레드 18.5
COST_AUCTION_BP = 23.0   # 양쪽 단일가 체결 가정: 세금 20 + 수수료 3 (스프레드 0 가정 — 미검증)
MIN_PRICE = 1000
TOP_N = 1000
HAC_LAG = 5

BOUNDARIES = ["09:00", "09:05", "09:30", "10:00", "11:00", "13:00", "14:00", "15:00", "15:20", "15:30"]
BUCKET_EDGES = [(1, 100, "1~100"), (101, 200, "101~200"), (201, 300, "201~300"),
                (301, 400, "301~400"), (401, 1000, "401~1000")]
BUCKETS = [b[2] for b in BUCKET_EDGES]
GROUPS = BUCKETS + ["1~300", "1~1000"]          # 보고 집단: 버킷 5 + 합집단 2
NEXT_OPEN = "익일09:00"
# 합성 창(사전 고정): "어떤 창을 빼면 남는 것이 양수인가" 를 보는 용도. 늘리지 않는다.
COMPOSITES = [("09:00", "15:30"), ("09:05", "15:30"), ("09:30", "15:30"), ("10:00", "15:30"),
              ("11:00", "15:30"), ("13:00", "15:30"), ("14:00", "15:30"), ("15:00", "15:30"),
              ("15:30", NEXT_OPEN), ("15:20", NEXT_OPEN), ("15:00", NEXT_OPEN), ("14:00", NEXT_OPEN),
              ("13:00", NEXT_OPEN), ("09:30", NEXT_OPEN), ("09:05", NEXT_OPEN), ("09:00", NEXT_OPEN)]


def _mod(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def pcol(hhmm: str) -> str:
    return "p_" + hhmm.replace(":", "")


WINDOWS = [f"{a}→{b}" for a, b in pairwise(BOUNDARIES)] + [f"15:30→{NEXT_OPEN}"]
COMPOSITE_NAMES = [f"{a}→{b}" for a, b in COMPOSITES]


# ─────────────────────────── 순수 함수(테스트 대상) ───────────────────────────

def boundary_prices(bars: pd.DataFrame) -> pd.DataFrame:
    """하루 분봉(code·ts·open·close) → 종목별 경계 가격.

    p_0900 = 09:00 분봉 open · p_1530 = 15:30 분봉 close · 그 사이 경계 T = ts < T 인 마지막 분봉 close.
    stale_T = 그 마지막 분봉이 T 보다 몇 분 앞인가(1 이면 직전 분). 15:20~15:29 분봉(일부 월에만 있음)은 어떤
    경계에도 안 쓴다(15:20 경계는 ts < 15:20, 15:30 은 15:30 행만).
    """
    b = bars.sort_values(["code", "ts"], kind="stable").copy()
    ts = pd.to_datetime(b["ts"])
    b["m"] = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    m = b["m"].to_numpy()
    out = {}
    out["p_0900"] = b[m == _mod("09:00")].groupby("code", sort=True)["open"].first()
    for t in BOUNDARIES[1:-1]:
        last = b[m < _mod(t)].groupby("code", sort=True)
        out[pcol(t)] = last["close"].last()
        out["stale_" + t.replace(":", "")] = _mod(t) - last["m"].last()
    out["p_1530"] = b[m == _mod("15:30")].groupby("code", sort=True)["close"].last()
    out["n_bars"] = b[m < _mod("15:20")].groupby("code", sort=True).size()
    res = pd.DataFrame(out)
    res.index.name = "code"
    for c in res.columns:
        if c.startswith("p_"):
            res.loc[~(res[c] > 0), c] = np.nan
    return res


def universe_from_prev_daily(prev_daily: pd.DataFrame) -> pd.DataFrame:
    """**전일** 일봉(code·close·trade_value) → 순위 1..TOP_N 과 버킷. 당일 값은 받지도 않는다."""
    d = prev_daily[(prev_daily["close"] >= MIN_PRICE) & prev_daily["trade_value"].notna()].copy()
    d["rank"] = d["trade_value"].rank(ascending=False, method="first").astype(int)
    d = d[d["rank"] <= TOP_N]
    d["bucket"] = pd.cut(d["rank"], bins=[0] + [hi for _, hi, _ in BUCKET_EDGES], labels=BUCKETS).astype(str)
    return d[["code", "rank", "bucket", "trade_value"]].rename(columns={"trade_value": "prev_value"})


def window_returns(p: pd.DataFrame, p_next: pd.DataFrame | None) -> pd.DataFrame:
    """경계 가격 → 창 수익(bp). 연속 창 9개 + 야간 + 합성 창. 곱하면 정확히 시가→종가·시가→익일 시가."""
    out = pd.DataFrame(index=p.index)
    nxt = (p_next["p_0900"].reindex(p.index) if p_next is not None and len(p_next)
           else pd.Series(np.nan, index=p.index))
    price = {t: p[pcol(t)] for t in BOUNDARIES}
    price[NEXT_OPEN] = nxt
    for a, b in pairwise(BOUNDARIES):
        out[f"{a}→{b}"] = (price[b] / price[a] - 1) * 1e4
    out[f"15:30→{NEXT_OPEN}"] = (price[NEXT_OPEN] / price["15:30"] - 1) * 1e4
    for a, b in COMPOSITES:
        out[f"{a}→{b}"] = (price[b] / price[a] - 1) * 1e4
    return out


def prev_features(p_d: pd.DataFrame, p_1: pd.DataFrame | None, p_2: pd.DataFrame | None) -> pd.DataFrame:
    """d 09:00 에 아는 값만: 전일 수익률(d−1 15:30 / d−2 15:30), 갭(d 09:00 / d−1 15:30). bp."""
    out = pd.DataFrame(index=p_d.index)
    nan = pd.Series(np.nan, index=p_d.index)
    c1 = p_1["p_1530"].reindex(p_d.index) if p_1 is not None and len(p_1) else nan
    c2 = p_2["p_1530"].reindex(p_d.index) if p_2 is not None and len(p_2) else nan
    out["ret1_prev"] = (c1 / c2 - 1) * 1e4
    out["gap"] = (p_d["p_0900"] / c1 - 1) * 1e4
    return out


def add_conditions(rows: pd.DataFrame) -> pd.DataFrame:
    """그날 유니버스 안 분위: 전일 수익률 5분위(q0 = 최고), |갭| 3분위(g0 = 최대), 갭 부호."""
    rows = rows.copy()
    rows["ret_q"] = rank_buckets(rows, "ret1_prev", n=5, labels=["q0(최고)", "q1", "q2", "q3", "q4(최저)"])
    rows["abs_gap"] = rows["gap"].abs()
    rows["gap_q"] = rank_buckets(rows, "abs_gap", n=3, labels=["g0(큼)", "g1", "g2(작음)"])
    rows["gap_sign"] = np.select([rows["gap"] > 0, rows["gap"] < 0], ["갭업", "갭다운"], default="0")
    rows.loc[rows["gap"].isna(), "gap_sign"] = np.nan
    return rows


def cell_stats(frame: pd.DataFrame, value: str, by: str, hac_lag: int = HAC_LAG) -> pd.DataFrame:
    """집단별 풀링 평균·중앙·n·일수·월수·음수월 비율·일평균 HAC t(코어) + 양(+) 비율·일평균·음수일 비율."""
    s = drift_summary(frame, value, by=by, date_col="date", hac_lag=hac_lag)
    ok = frame[value].notna()
    f = frame[ok]
    extra = f.groupby(by, observed=True)[value].agg(pos_share=lambda x: float((x > 0).mean()))
    daily = f.groupby([by, "date"], observed=True)[value].mean().reset_index()
    dstat = daily.groupby(by, observed=True)[value].agg(day_mean_bp="mean", neg_day_share=lambda x: float((x < 0).mean()))
    out = s.join(extra).join(dstat)
    out.index.name = by
    return out


# ─────────────────────────── DB ───────────────────────────

def fetch_day(conn, day: str) -> pd.DataFrame:
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    bars = pd.read_sql(
        "select code, ts, open, close from minute_bars where ts >= %s and ts < %s and ts::time <= '15:30'",
        conn, params=(day, nxt))
    if bars.empty:
        return pd.DataFrame()
    bars["ts"] = pd.to_datetime(bars["ts"])
    return boundary_prices(bars)


def load_boundaries(conn, days: list[str]) -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for day in days:
        f = CACHE / f"{day}.pkl"
        if f.exists():
            out[day] = pickle.loads(f.read_bytes())
        else:
            p = fetch_day(conn, day)
            f.write_bytes(pickle.dumps(p, protocol=5))
            out[day] = p
            print(f"{day} 종목 {len(p):,}", flush=True)
    return out


def build_rows(days: list[str], sample_days: set[str], P: dict[str, pd.DataFrame],
               daily: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """날짜별 행: 유니버스(전일 순위) × 창 수익 × 조건. 당일 순위 버킷은 look-ahead 대조용으로만 붙인다."""
    daily_by_date = {d: g for d, g in daily.groupby("date", sort=True)}
    parts = []
    for i, d in enumerate(days):
        if d not in sample_days or d not in P or P[d].empty:
            continue
        prev = days[i - 1] if i >= 1 else None
        if prev is None or prev not in daily_by_date:
            continue
        uni = universe_from_prev_daily(daily_by_date[prev])
        p_d = P[d]
        p_n = P[days[i + 1]] if i + 1 < len(days) else None
        p_1 = P.get(days[i - 1]) if i >= 1 else None
        p_2 = P.get(days[i - 2]) if i >= 2 else None
        w = window_returns(p_d, p_n)
        f = prev_features(p_d, p_1, p_2)
        stale = p_d[[c for c in p_d.columns if c.startswith("stale_")]]
        r = uni.set_index("code").join(p_d[["p_0900", "p_1530", "n_bars"]], how="inner").join(w).join(f).join(stale)
        r = r[r["p_0900"].notna()]
        # look-ahead 대조: **당일** 거래대금 순위(쓰면 안 되는 값). 보고용.
        if d in daily_by_date:
            same = universe_from_prev_daily(daily_by_date[d]).set_index("code")["bucket"]
            r["bucket_sameday"] = same.reindex(r.index)
        else:
            r["bucket_sameday"] = np.nan
        r["date"] = pd.Timestamp(d)
        r["market"] = market[~market.index.duplicated()].reindex(r.index).fillna("미상").to_numpy()
        parts.append(add_conditions(r.reset_index()))
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────── 보고 ───────────────────────────

def with_groups(rows: pd.DataFrame) -> pd.DataFrame:
    """버킷 5 + 합집단(1~300, 1~1000) 행을 겹쳐 쌓는다(집단 열 `grp`)."""
    a = rows.assign(grp=rows["bucket"])
    b = rows[rows["rank"] <= 300].assign(grp="1~300")
    c = rows.assign(grp="1~1000")
    return pd.concat([a, b, c], ignore_index=True)


def table_by_group(rows_g: pd.DataFrame, cols: list[str], period: str) -> pd.DataFrame:
    out = []
    for w in cols:
        s = cell_stats(rows_g, w, "grp").reset_index()
        s.insert(0, "window", w)
        s.insert(0, "period", period)
        out.append(s)
    t = pd.concat(out, ignore_index=True)
    t["grp"] = pd.Categorical(t["grp"], GROUPS, ordered=True)
    return t.sort_values(["period", "window", "grp"], kind="stable")


def table_conditional(rows_g: pd.DataFrame, cond: str, cols: list[str]) -> pd.DataFrame:
    out = []
    for g, sub in rows_g.groupby("grp", observed=True):
        sub = sub.dropna(subset=[cond])
        if sub.empty:
            continue
        for w in cols:
            s = cell_stats(sub, w, cond).reset_index()
            s.insert(0, "window", w)
            s.insert(0, "grp", g)
            out.append(s)
    return pd.concat(out, ignore_index=True)


def fmt_matrix(t: pd.DataFrame, col: str, cols: list[str], groups: list[str], digits: int = 1) -> str:
    """markdown: 행 = 창, 열 = 집단."""
    piv = t.pivot(index="window", columns="grp", values=col).reindex(index=cols, columns=groups)
    head = "| 창 | " + " | ".join(groups) + " |\n|---|" + "--:|" * len(groups) + "\n"
    body = ""
    for w in cols:
        body += f"| {w} | " + " | ".join(
            "—" if pd.isna(v) else (f"{v:+.{digits}f}" if digits else f"{v:+.0f}") for v in piv.loc[w]) + " |\n"
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default="2025-08-01")
    ap.add_argument("--to", dest="d1", default="2026-09-16")
    ap.add_argument("--extra-from", default="2026-09-17")
    ap.add_argument("--extra-to", default="2026-10-02")
    ap.add_argument("--out", default="data/research108")
    a = ap.parse_args()
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    conn.set_session(readonly=True)
    bad = set(json.loads(BAD_DAYS.read_text(encoding="utf-8"))["days"])
    lo = (pd.Timestamp(a.d0) - pd.Timedelta(days=10)).date().isoformat()
    hi = a.extra_to or a.d1
    days = pd.read_sql("select distinct ts::date d from minute_bars where ts >= %s and ts < %s::date + 1 order by d",
                       conn, params=(lo, hi)).d.astype(str).tolist()
    daily = pd.read_sql("select code, date, close, trade_value from daily_bars where date >= %s and date <= %s",
                        conn, params=(lo, hi))
    daily["date"] = daily["date"].astype(str)
    market = pd.read_sql("select code, market from stocks", conn).set_index("code")["market"]
    P = load_boundaries(conn, days)
    conn.close()

    main_days = {d for d in days if a.d0 <= d <= a.d1 and d not in bad}
    extra_days = {d for d in days if a.extra_from and a.extra_from <= d <= a.extra_to and d not in bad}
    rows = build_rows(days, main_days | extra_days, P, daily, market)
    rows["period"] = np.where(rows["date"].dt.strftime("%Y-%m-%d").isin(main_days), "main", "extra")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if _has_parquet():
        rows.to_parquet(out / "rows.parquet")
    else:
        rows.to_pickle(out / "rows.pkl")

    cols = WINDOWS + [c for c in COMPOSITE_NAMES if c not in WINDOWS]
    rm = rows[rows.period == "main"]
    periods = {"main": rm, "extra": rows[rows.period == "extra"],
               "2025": rm[rm.date.dt.year == 2025], "2026": rm[rm.date.dt.year == 2026]}
    tabs = [table_by_group(with_groups(df), cols, name) for name, df in periods.items() if len(df)]
    T = pd.concat(tabs, ignore_index=True)
    T.to_csv(out / "windows_by_bucket.csv", index=False)

    rg = with_groups(rm)
    cond = {}
    for c in ("ret_q", "gap_sign", "gap_q"):
        cond[c] = table_conditional(rg, c, WINDOWS)
        cond[c].to_csv(out / f"conditional_{c}.csv", index=False)

    # 월별 부호(1~300)
    top = rm[rm["rank"] <= 300].copy()
    top["month"] = top.date.dt.to_period("M").astype(str)
    monthly = top.groupby("month")[WINDOWS].mean()
    monthly.to_csv(out / "monthly_top300.csv")

    # 시장별(1~300): 단일가 다리들
    mk = table_conditional(rg[rg.grp == "1~300"], "market", ["15:20→15:30", f"15:30→{NEXT_OPEN}", f"15:20→{NEXT_OPEN}"])
    mk.to_csv(out / "market_top300.csv", index=False)

    # look-ahead 대조: 당일 순위로 버킷을 만들면 어떻게 되나(97 의 오류 재현). 보고용.
    la = rm.dropna(subset=["bucket_sameday"]).assign(grp=lambda x: x["bucket_sameday"])
    la_t = cell_stats(la, "09:00→15:30", "grp").reset_index()
    la_t.to_csv(out / "lookahead_sameday_rank.csv", index=False)

    # 가장 좋은 (집단 × 창) — 주 구간, 사전 고정 창 집합 안에서. 셀 수를 같이 적는다.
    Tm = T[T.period == "main"].copy()
    ncells = len(Tm)
    best = Tm.sort_values("mean_bp", ascending=False).iloc[0]
    best_t = Tm.sort_values("t_hac", ascending=False).iloc[0]
    stale_cols = [c for c in rows.columns if c.startswith("stale_")]
    summary = {
        "period_main": [a.d0, a.d1], "period_extra": [a.extra_from, a.extra_to],
        "n_days_main": int(rm.date.nunique()), "n_days_extra": int(rows[rows.period == "extra"].date.nunique()),
        "n_rows_main": len(rm), "n_rows_extra": int((rows.period == "extra").sum()),
        "bad_days_excluded": sorted(bad), "n_cells_main": int(ncells),
        "coverage_by_bucket": {
            b: {"rows": int((rm.bucket == b).sum()),
                "has_1530": float(rm.loc[rm.bucket == b, "p_1530"].notna().mean()),
                "has_next_open": float(rm.loc[rm.bucket == b, f"15:30→{NEXT_OPEN}"].notna().mean()),
                "stale_gt5_share": float((rm.loc[rm.bucket == b, stale_cols] > 5).any(axis=1).mean())}
            for b in BUCKETS},
        "best_mean": {k: _j(best[k]) for k in ("grp", "window", "mean_bp", "day_mean_bp", "median_bp", "t_hac", "n",
                                               "n_days", "pos_share", "neg_day_share", "neg_month_share")},
        "best_t": {k: _j(best_t[k]) for k in ("grp", "window", "mean_bp", "t_hac", "n")},
        "cost_lines_bp": {"continuous": COST_CONT_BP, "auction_both_sides_unverified": COST_AUCTION_BP},
        "lookahead_check": {
            "prev_rank_1~300_open_to_close_bp": float(Tm[(Tm.grp == "1~300") & (Tm.window == "09:00→15:30")].mean_bp.iloc[0]),
            "sameday_rank_1~100_open_to_close_bp": float(la_t[la_t.grp == "1~100"].mean_bp.iloc[0]),
            "sameday_rank_401~1000_open_to_close_bp": float(la_t[la_t.grp == "401~1000"].mean_bp.iloc[0]),
        },
    }
    config = {"study": "108-intraday-tod", "boundaries": BOUNDARIES, "composites": COMPOSITE_NAMES,
              "buckets": BUCKETS, "period": [a.d0, a.d1], "hac_lag": HAC_LAG, "cost_bp": [COST_CONT_BP, COST_AUCTION_BP],
              "judgement": "none (description)"}
    summary["trial_fingerprint"] = record_trial("108-intraday-tod", config, logs_dir=ROOT / "research" / "logs",
                                                note="기술(description) — 창·버킷 사전 고정, 판정 없음")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # stdout 요약
    print(f"\n주 구간 {a.d0}~{a.d1}: {summary['n_days_main']}일 · {summary['n_rows_main']:,}행 · 셀 {ncells}")
    print("\n평균 bp (주 구간)\n" + fmt_matrix(Tm, "mean_bp", cols, GROUPS))
    print("\nNW t (일평균, lag 5)\n" + fmt_matrix(Tm, "t_hac", cols, GROUPS, digits=2))
    print(f"\n최대 평균 셀: {best.grp} × {best.window} = {best.mean_bp:+.1f}bp (t={best.t_hac:+.2f}, n={int(best.n):,})")
    print(f"look-ahead 대조: 전일 순위 1~300 일중 {summary['lookahead_check']['prev_rank_1~300_open_to_close_bp']:+.1f} "
          f"vs 당일 순위 1~100 일중 {summary['lookahead_check']['sameday_rank_1~100_open_to_close_bp']:+.1f}bp")
    print(f"→ {out}/")
    return 0


def _j(v):
    return float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
