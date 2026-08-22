"""시가 베팅 사전등록 — 이벤트 **규모만** 잰다. 수익률은 계산하지 않는다.

    uv run --extra pg python scripts/openbet_events.py \
        --env-file ../kr-quant/.env --out-dir /path/to/scratch

이 스크립트가 D+1 일봉에서 읽는 값은 **시가 하나**다. D+1 고가·저가·종가는
로드 후 즉시 버린다 — 사전등록 전에 결과를 보지 않기 위한 구조적 장치다.
(회귀 검사: ``_strip_outcome`` 이후 프레임에 D+1 의 high/low/close 가 없다.)

출력:
- 표준출력: 규모 표 (마크다운)
- ``{out-dir}/openbet_events_summary.json``: 같은 내용 JSON
- ``{out-dir}/openbet_need.json``: 분봉 없는 이벤트 종목 {code: 최초이벤트일-10일}

DB 쓰기 없음. 읽기만 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv
from scalp_it.closebet.features import close_bet_features

# ── 전일(D) 조건 — 종가베팅 1차 확정 규칙 그대로. 새 값을 고르지 않았다. ──
TV_RATIO_MIN = 3.0
CLOSE_POS_MIN = 0.8
TV_FLOOR = 5000.0  # 백만원 = 50억
TOP_N = 2          # 하루 상위 건수 (tv_ratio 내림차순)

# ── 당일(D+1) 갭 — 주 정의 구간. 3% 는 08번 문서 만쥬 기대 갭(+3~7%)의 하한. ──
GAP_MIN = 0.03
GAP_LIMIT = 0.29   # 이 이상이면 상한가 시가 — 체결 불가로 제외
LIMIT_UP_RET = 0.29  # 전일 상한가 마감 판정 (프로젝트 공통 임계)

GAP_BINS = [-1.0, 0.0, 0.03, 0.05, 0.10, 0.15, 0.29, 9.0]
GAP_LABELS = ["<0%", "0~3%", "3~5%", "5~10%", "10~15%", "15~29%", "≥29%(상한가)"]

MINUTE_START = "2025-08-01"


def _load_env(path: str | None) -> None:
    if not path:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _strip_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """D+1 행에서 시가만 남긴다. 여기 지나면 결과(고가·저가·종가)는 없다."""
    keep = ["code", "date", "open"]
    return df[keep].copy()


def build_events(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feats = close_bet_features(ohlcv)
    feats = feats.merge(ohlcv[["code", "date", "close", "trade_value"]], on=["code", "date"])

    cond = (
        (feats["tv_ratio"] >= TV_RATIO_MIN)
        & (feats["close_pos"] >= CLOSE_POS_MIN)
        & (feats["trade_value"] >= TV_FLOOR)
    )
    prev = feats.loc[cond, ["code", "date", "close", "tv_ratio", "close_pos", "ret_day", "trade_value"]].copy()
    prev = prev.rename(columns={"date": "d0", "close": "close_d0"})
    prev["prev_limit_up"] = prev["ret_day"] >= LIMIT_UP_RET

    # D+1 = 전체 시장 달력상 다음 거래일. 그 종목에 그날 봉이 없으면 제외(거래정지 등).
    cal = pd.Series(sorted(ohlcv["date"].unique()))
    nxt = dict(zip(cal[:-1], cal[1:]))
    prev["d1"] = prev["d0"].map(nxt)
    n_no_next_cal = int(prev["d1"].isna().sum())  # 마지막 날짜(미래) 등
    prev = prev[prev["d1"].notna()]

    opens = _strip_outcome(ohlcv).rename(columns={"date": "d1", "open": "open_d1"})
    ev = prev.merge(opens, on=["code", "d1"], how="left")
    n_no_bar = int(ev["open_d1"].isna().sum())
    ev = ev[ev["open_d1"].notna() & (ev["open_d1"] > 0)].copy()

    ev["gap"] = ev["open_d1"] / ev["close_d0"] - 1.0
    ev["gap_bin"] = pd.cut(ev["gap"], bins=GAP_BINS, labels=GAP_LABELS, right=False)
    ev["is_main"] = (ev["gap"] >= GAP_MIN) & (ev["gap"] < GAP_LIMIT)

    meta = {
        "n_prev_cond": len(prev) + n_no_next_cal,
        "n_no_next_calendar": n_no_next_cal,
        "n_no_next_bar": n_no_bar,
        "n_with_open": len(ev),
    }
    return ev, meta


def cap_top_n(ev: pd.DataFrame, n: int) -> pd.DataFrame:
    return (
        ev.sort_values(["d1", "tv_ratio"], ascending=[True, False])
        .groupby("d1", sort=False)
        .head(n)
    )


def minute_coverage(conn) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("select code, min(ts)::date, max(ts)::date, count(*) from minute_bars group by code")
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=["code", "m_first", "m_last", "m_rows"])
    df["code"] = df["code"].astype(str).str.strip()
    df["m_first"] = pd.to_datetime(df["m_first"])
    df["m_last"] = pd.to_datetime(df["m_last"])
    return df


def _counts_by(ev: pd.DataFrame, key: str) -> dict:
    return {str(k): int(v) for k, v in ev.groupby(key).size().items()}


def summarize(ev: pd.DataFrame, meta: dict, mcov: pd.DataFrame, last_year_start: str, end: str) -> dict:
    out: dict = {"meta": meta, "rule": {
        "prev_day": f"tv_ratio>={TV_RATIO_MIN} & close_pos>={CLOSE_POS_MIN} & trade_value>={TV_FLOOR:.0f}백만",
        "gap_main": f"{GAP_MIN:.0%} <= gap < {GAP_LIMIT:.0%}",
        "top_n": TOP_N,
    }}
    ly = ev[(ev["d1"] >= last_year_start) & (ev["d1"] <= end)]

    def _block(e: pd.DataFrame) -> dict:
        main = e[e["is_main"]]
        capped = cap_top_n(main, TOP_N)
        per_day = main.groupby("d1").size()
        return {
            "n_prev_cond_with_open": len(e),
            "n_prev_limit_up": int(e["prev_limit_up"].sum()),
            "gap_bins_all": {str(k): int(v) for k, v in e["gap_bin"].value_counts(sort=False).items()},
            "gap_bins_prev_limit_up": {str(k): int(v) for k, v in e.loc[e["prev_limit_up"], "gap_bin"].value_counts(sort=False).items()},
            "gap_bins_prev_not_limit_up": {str(k): int(v) for k, v in e.loc[~e["prev_limit_up"], "gap_bin"].value_counts(sort=False).items()},
            "n_main": len(main),
            "n_main_prev_limit_up": int(main["prev_limit_up"].sum()),
            "n_main_capped": len(capped),
            "n_days_main": int(main["d1"].nunique()),
            "per_day_mean": float(per_day.mean()) if len(per_day) else 0.0,
            "per_day_median": float(per_day.median()) if len(per_day) else 0.0,
            "per_day_max": int(per_day.max()) if len(per_day) else 0,
            "n_codes_main": int(main["code"].nunique()),
            "n_codes_main_capped": int(capped["code"].nunique()),
            "top_codes_capped": {str(k): int(v) for k, v in capped["code"].value_counts().head(10).items()},
            "gap_quantiles_main": {q: float(main["gap"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(main) else {},
        }

    out["all_11y"] = _block(ev)
    out["last_1y"] = _block(ly)

    main = ev[ev["is_main"]]
    capped = cap_top_n(main, TOP_N)
    out["by_year_main"] = _counts_by(main.assign(y=main["d1"].dt.year), "y")
    out["by_year_main_capped"] = _counts_by(capped.assign(y=capped["d1"].dt.year), "y")
    out["by_year_prev_cond"] = _counts_by(ev.assign(y=ev["d1"].dt.year), "y")
    lym = ly[ly["is_main"]]
    lyc = cap_top_n(lym, TOP_N)
    out["by_month_main_last_1y"] = _counts_by(lym.assign(m=lym["d1"].dt.strftime("%Y-%m")), "m")
    out["by_month_main_capped_last_1y"] = _counts_by(lyc.assign(m=lyc["d1"].dt.strftime("%Y-%m")), "m")

    # 분봉 보유 — 분봉 구간(2025-08-01~) 이벤트 기준
    mw = ev[(ev["is_main"]) & (ev["d1"] >= MINUTE_START)].copy()
    mw = mw.merge(mcov, on="code", how="left")
    mw["has_minute_code"] = mw["m_rows"].notna()
    mw["has_minute_day"] = mw["has_minute_code"] & (mw["d1"] >= mw["m_first"]) & (mw["d1"] <= mw["m_last"])
    codes_w = mw["code"].nunique()
    out["minute_window"] = {
        "start": MINUTE_START,
        "n_events_main": len(mw),
        "n_codes": int(codes_w),
        "n_codes_with_minute": int(mw.loc[mw["has_minute_code"], "code"].nunique()),
        "n_events_code_has_minute": int(mw["has_minute_code"].sum()),
        "n_events_day_covered": int(mw["has_minute_day"].sum()),
        "n_minute_codes_total": len(mcov),
    }
    mwc = cap_top_n(mw, TOP_N)
    out["minute_window"]["capped"] = {
        "n_events": len(mwc),
        "n_codes": int(mwc["code"].nunique()),
        "n_codes_with_minute": int(mwc.loc[mwc["has_minute_code"], "code"].nunique()),
        "n_events_day_covered": int(mwc["has_minute_day"].sum()),
    }

    need = (
        mw[~mw["has_minute_code"]].groupby("code")["d1"].min()
        .apply(lambda d: (d - timedelta(days=10)).strftime("%Y%m%d")).to_dict()
    )
    out["need"] = {str(k): v for k, v in sorted(need.items())}
    out["minute_window"]["capped"]["n_need_codes"] = int(mwc.loc[~mwc["has_minute_code"], "code"].nunique())
    return out


def render(s: dict) -> str:
    L: list[str] = []
    a, y = s["all_11y"], s["last_1y"]
    L.append(f"전일 조건: {s['rule']['prev_day']} / 주 정의 갭: {s['rule']['gap_main']} / 하루 상위 {s['rule']['top_n']}건")
    L.append(f"전일 조건 충족 (종목,일자): {s['meta']['n_prev_cond']:,}  · 다음 거래일 봉 없음 제외 {s['meta']['n_no_next_bar']:,} · 달력 끝 {s['meta']['n_no_next_calendar']}")
    L.append("")
    L.append("| 항목 | 11년 | 최근 1년 |")
    L.append("|---|---|---|")
    L.append(f"| 전일 조건 충족 (D+1 시가 있음) | {a['n_prev_cond_with_open']:,} | {y['n_prev_cond_with_open']:,} |")
    L.append(f"| └ 전일 상한가 마감 | {a['n_prev_limit_up']:,} | {y['n_prev_limit_up']:,} |")
    L.append(f"| 주 정의 (갭 3~29%) | {a['n_main']:,} | {y['n_main']:,} |")
    L.append(f"| └ 전일 상한가 마감 | {a['n_main_prev_limit_up']:,} | {y['n_main_prev_limit_up']:,} |")
    L.append(f"| 하루 상위 {TOP_N}건 후 | {a['n_main_capped']:,} | {y['n_main_capped']:,} |")
    L.append(f"| 이벤트 있는 날 | {a['n_days_main']:,} | {y['n_days_main']:,} |")
    L.append(f"| 하루 건수 평균/중앙/최대 | {a['per_day_mean']:.2f}/{a['per_day_median']:.0f}/{a['per_day_max']} | {y['per_day_mean']:.2f}/{y['per_day_median']:.0f}/{y['per_day_max']} |")
    L.append(f"| 종목 수 (주 정의 / 상위{TOP_N}) | {a['n_codes_main']:,} / {a['n_codes_main_capped']:,} | {y['n_codes_main']:,} / {y['n_codes_main_capped']:,} |")
    L.append("")
    L.append("갭 구간별 건수 (전일 조건 충족 전체; 시가/전일종가 − 1)")
    L.append("")
    L.append("| 갭 구간 | 11년 전체 | 11년 전일상한가 | 11년 전일비상한가 | 최근 1년 전체 |")
    L.append("|---|---|---|---|---|")
    for lab in GAP_LABELS:
        L.append(f"| {lab} | {a['gap_bins_all'].get(lab,0):,} | {a['gap_bins_prev_limit_up'].get(lab,0):,} | {a['gap_bins_prev_not_limit_up'].get(lab,0):,} | {y['gap_bins_all'].get(lab,0):,} |")
    L.append("")
    gq = a["gap_quantiles_main"]
    L.append("주 정의 갭 분위 (11년): " + ", ".join(f"p{int(q*100)} {v:+.2%}" for q, v in gq.items()))
    L.append("")
    L.append("| 연도 (D+1) | 전일조건 | 주 정의 | 상위2 |")
    L.append("|---|---|---|---|")
    for yv in sorted(s["by_year_prev_cond"]):
        L.append(f"| {yv} | {s['by_year_prev_cond'][yv]:,} | {s['by_year_main'].get(yv,0):,} | {s['by_year_main_capped'].get(yv,0):,} |")
    L.append("")
    L.append("| 월 (최근 1년) | 주 정의 | 상위2 |")
    L.append("|---|---|---|")
    for m in sorted(s["by_month_main_last_1y"]):
        L.append(f"| {m} | {s['by_month_main_last_1y'][m]:,} | {s['by_month_main_capped_last_1y'].get(m,0):,} |")
    L.append("")
    L.append(f"상위{TOP_N} 최다 등장 종목 (11년): " + ", ".join(f"{k}×{v}" for k, v in a["top_codes_capped"].items()))
    L.append(f"상위{TOP_N} 최다 등장 종목 (1년): " + ", ".join(f"{k}×{v}" for k, v in y["top_codes_capped"].items()))
    mw = s["minute_window"]
    L.append("")
    L.append(f"분봉 구간({mw['start']}~) 주 정의 이벤트 {mw['n_events_main']:,}건 · 종목 {mw['n_codes']:,} · "
             f"minute_bars 보유 종목 {mw['n_codes_with_minute']:,} (전체 수집 {mw['n_minute_codes_total']:,}종목) · "
             f"이벤트 기준 종목 보유 {mw['n_events_code_has_minute']:,} / 이벤트일 범위 내 {mw['n_events_day_covered']:,}")
    c = mw["capped"]
    L.append(f"  상위{TOP_N}: 이벤트 {c['n_events']:,} · 종목 {c['n_codes']:,} · 보유 종목 {c['n_codes_with_minute']:,} · 이벤트일 범위 내 {c['n_events_day_covered']:,}")
    L.append(f"분봉 필요 종목 {len(s['need']):,}개 (상위{TOP_N}만이면 {c['n_need_codes']:,}개) → openbet_need.json")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2016-09-09")
    p.add_argument("--end", default="2026-08-21")
    p.add_argument("--last-year-start", default="2025-08-22")
    p.add_argument("--dsn", default=None)
    p.add_argument("--env-file", default=None, help="KR_QUANT_DB 를 담은 .env")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    _load_env(args.env_file)
    conn = connect(args.dsn)
    ohlcv = load_ohlcv(conn, args.start, args.end)
    ev, meta = build_events(ohlcv)
    del ohlcv
    mcov = minute_coverage(conn)
    conn.close()

    s = summarize(ev, meta, mcov, args.last_year_start, args.end)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "openbet_events_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    (out / "openbet_need.json").write_text(json.dumps(s["need"], ensure_ascii=False, indent=1))
    print(render(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
