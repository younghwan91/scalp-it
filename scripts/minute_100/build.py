"""100번 분봉 대규모 탐색 — **표본을 20배로 키운다.** 사전등록 전 기술 단계.

    ~/of80/.rlvenv/bin/python scripts/minute_100/build.py            # 날짜별 캐시

왜: 83~96 의 모든 연구가 **틱 17거래일**로 돌았다. 신호가 약하면 17일로는 못 본다. `minute_bars` 에
**2,089종목 × 13.5개월(2025-08-01~2026-09-16, 1억 1,329만 행)** 이 있는데 한 번도 쓴 적이 없다.

질문은 하나다: **30분 보유에서 왕복 41.5bp 를 넘는 신호가 있는가.**

- 결정 시점: 매 5분(09:05~14:30). 09:00~09:05 와 14:30 이후는 뺀다(동시호가·마감 효과).
- 라벨: 30분 뒤 종가 / 현재 종가 − 1 − 비용. 비용은 **41.5bp**(세금 20 + 수수료 3 + 스프레드 18.5).
- 피처는 전부 결정 분 t **이하**만 본다.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _envload  # noqa: E402,F401

COST = 0.00415                 # 왕복 실비용 — daytrade-it 실측 스프레드 포함
HORIZON = 30                   # 분
STEP = 5                       # 결정 간격(분)
LO_MIN, HI_MIN = 9 * 60 + 5, 14 * 60 + 30
TOP_N = 400                    # 그날 거래대금 상위(전일 기준) — 유동성 하한
CACHE = Path.home() / "of80" / "min100"


def features_for_day(df: pd.DataFrame) -> pd.DataFrame:
    """한 종목의 하루 분봉 → 결정 분별 피처 + 라벨. 전부 t 이하만 본다(라벨 제외)."""
    d = df.sort_values("ts").reset_index(drop=True)
    c = d.close.to_numpy(float)
    v = d.trade_value.to_numpy(float)
    hi, lo = d.high.to_numpy(float), d.low.to_numpy(float)
    n = len(c)
    out: dict[str, np.ndarray] = {}
    s = pd.Series(c)
    for w in (1, 3, 5, 10, 30, 60):
        out[f"ret{w}"] = (s / s.shift(w) - 1).to_numpy() * 1e4
    vv = pd.Series(v)
    for w in (5, 30):
        out[f"val{w}"] = np.log1p(vv.rolling(w, min_periods=1).sum().to_numpy())
        out[f"valrel{w}"] = (vv.rolling(w, min_periods=1).mean()
                             / (vv.expanding(min_periods=1).mean() + 1)).to_numpy()
    r1 = s.pct_change()
    for w in (10, 30):
        out[f"rv{w}"] = r1.rolling(w, min_periods=5).std().to_numpy() * 1e4
    cum_hi = np.maximum.accumulate(hi)
    cum_lo = np.minimum.accumulate(lo)
    out["dd_hi"] = (c / cum_hi - 1) * 1e4
    out["up_lo"] = (c / cum_lo - 1) * 1e4
    rng = cum_hi - cum_lo
    out["pos_day"] = np.where(rng > 0, (c - cum_lo) / rng, 0.5)
    vwap = (pd.Series(c * v).expanding().sum() / (vv.expanding().sum() + 1e-9)).to_numpy()
    out["vwapdev"] = np.where(vwap > 0, (c / vwap - 1) * 1e4, 0.0)
    for w in (10, 30):
        rr = pd.Series(c).rolling(w, min_periods=2)
        out[f"pos{w}"] = ((c - rr.min().to_numpy())
                          / np.where(rr.max().to_numpy() - rr.min().to_numpy() > 0,
                                     rr.max().to_numpy() - rr.min().to_numpy(), np.nan))
    out["minute"] = (d.ts.dt.hour * 60 + d.ts.dt.minute).to_numpy(float)
    out["gap"] = np.full(n, (c[0] / d.prev_close.iloc[0] - 1) * 1e4 if d.prev_close.iloc[0] > 0 else 0.0)
    fwd = np.full(n, np.nan)
    if n > HORIZON:
        fwd[: n - HORIZON] = c[HORIZON:] / c[: n - HORIZON] - 1 - COST
    out["y"] = fwd * 1e4
    f = pd.DataFrame(out)
    f["code"] = d.code.iloc[0]
    f["ts"] = d.ts
    keep = ((f.minute >= LO_MIN) & (f.minute <= HI_MIN)
            & (((f.minute - LO_MIN) % STEP) == 0) & f.y.notna())
    return f[keep].replace([np.inf, -np.inf], np.nan)


def build_day(conn, day: str) -> pd.DataFrame:
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    prev = pd.read_sql(
        "select code, close from daily_bars where date = (select max(date) from daily_bars where date < %s)",
        conn, params=(day,))
    pc = dict(zip(prev.code, prev.close, strict=True))
    liq = pd.read_sql(
        "select code from daily_bars where date = (select max(date) from daily_bars where date < %s) "
        "and close >= 1000 order by trade_value desc nulls last limit %s", conn, params=(day, TOP_N))
    codes = [c for c in liq.code if c in pc]
    if not codes:
        return pd.DataFrame()
    mb = pd.read_sql(
        "select code, ts, open, high, low, close, volume, trade_value from minute_bars "
        "where ts >= %s and ts < %s and code = any(%s) order by code, ts",
        conn, params=(day, nxt, codes))
    if mb.empty:
        return pd.DataFrame()
    mb["ts"] = pd.to_datetime(mb.ts)
    mb["prev_close"] = mb.code.map(pc)
    parts = [features_for_day(g) for _, g in mb.groupby("code", sort=False) if len(g) > HORIZON + 60]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(days: list[str]) -> int:
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    CACHE.mkdir(parents=True, exist_ok=True)
    if not days:
        days = pd.read_sql("select distinct ts::date d from minute_bars order by d", conn).d.astype(str).tolist()
    for day in days:
        out = CACHE / f"{day}.pkl"
        if out.exists():
            continue
        df = build_day(conn, day)
        if len(df):
            out.write_bytes(pickle.dumps(df, protocol=5))
        print(f"{day} {len(df):,}행", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
