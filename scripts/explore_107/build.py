"""107번 마감 30분·단일가 탐색 — 날짜별 캐시. **simnode 에서만** 돌린다(104 규칙 9).

    deploy/research_run.sh .venv/bin/python -W ignore scripts/explore_107/build.py [날짜 ...]

하루마다: 전일 거래대금 상위 1,300(전일 종가 ≥ 1,000원) 종목의 경계 시각 가격(분봉 T 직전 close)
+ 일봉 시가·종가 + 전일 피처(수익률·20일 변동성·거래대금) + 시장 구분 → ``~/of80/close107/<날짜>.pkl``.
분봉은 경계 시각 직전 5분만 읽는다(하루 전체를 읽는 것의 1/5 — 수집·ssh 를 안 건드리게).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _envload  # noqa: E402,F401
import common as c  # noqa: E402

CACHE = Path.home() / "of80" / "close107"
START, END = "2025-08-01", "2026-10-02"


def bad_days() -> set[str]:
    f = ROOT / "research" / "bad_days.json"
    return set(json.loads(f.read_text(encoding="utf-8"))["days"])


def window_timestamps(day: str) -> list[str]:
    """경계 시각 T 마다 T−5 ~ T−1 분 — 분봉에서 읽을 timestamp 목록."""
    out = []
    for t in c.PRICE_TIMES:
        tm = int(t[:2]) * 60 + int(t[3:])
        for m in range(tm - c.LOOKBACK, tm):
            out.append(f"{day} {m // 60:02d}:{m % 60:02d}")
    return out


def build_day(conn, day: str, feats: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    uni = c.universe(feats[feats.date == day])
    if uni.empty:
        return pd.DataFrame()
    codes = uni.code.tolist()
    mb = pd.read_sql(
        "select code, ts, close from minute_bars where ts = any(%s::timestamp[]) and code = any(%s)",
        conn, params=(window_timestamps(day), codes))
    daily = pd.read_sql("select code, open, close from daily_bars where date = %s and code = any(%s)",
                        conn, params=(day, codes))
    if mb.empty or daily.empty:
        return pd.DataFrame()
    px = c.assemble_day(c.price_at_bounds(mb), daily)
    out = px.join(uni.set_index("code")[["p_ret1", "p_vol20", "p_tv"]], how="inner")
    out["market"] = market.reindex(out.index)
    out["date"] = day
    return out.reset_index(names="code")


def main(days: list[str]) -> int:
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    CACHE.mkdir(parents=True, exist_ok=True)
    lo = (pd.Timestamp(START) - pd.Timedelta(days=60)).date().isoformat()
    daily = pd.read_sql("select code, date, close, trade_value from daily_bars where date >= %s and date <= %s",
                        conn, params=(lo, END))
    daily["date"] = pd.to_datetime(daily.date)
    feats = c.prior_day_features(daily)
    feats["date"] = feats.date.dt.strftime("%Y-%m-%d")
    market = pd.read_sql("select code, market from stocks", conn).set_index("code").market
    if not days:
        days = pd.read_sql("select distinct ts::date d from minute_bars where ts >= %s and ts < %s order by d",
                           conn, params=(START, f"{END} 23:59")).d.astype(str).tolist()
    bad = bad_days()
    for day in days:
        out = CACHE / f"{day}.pkl"
        if out.exists() or day in bad:
            continue
        df = build_day(conn, day, feats, market)
        if len(df):
            out.write_bytes(pickle.dumps(df, protocol=5))
        print(f"{day} {len(df):,}행 · 14:50 결측 {df['14:50'].isna().mean():.1%}" if len(df) else f"{day} 0행",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
