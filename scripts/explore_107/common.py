"""107번 마감 30분·단일가 탐색 — 순수 함수(DB 없음). 테스트가 이 파일을 본다.

문서 docs/research/manju/107-close-window-exploration.md §1~§2 를 그대로 옮긴 것이다.
여기서 정의가 바뀌면 그건 탐색이다 — 문서가 먼저고 코드가 따른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 시각 경계(분) — 09:00 은 일봉 시가, 15:30 은 일봉 종가, 나머지는 "T 직전 분봉 close"
BOUNDS_MIN = [9 * 60 + 30 * k for k in range(13)] + [15 * 60 + 20]   # 09:00, 09:30, …, 15:00, 15:20
BOUNDS = [f"{m // 60:02d}:{m % 60:02d}" for m in BOUNDS_MIN]          # 14개
INTERVALS = [(BOUNDS[i], BOUNDS[i + 1]) for i in range(len(BOUNDS) - 1)] + [("15:20", "15:30")]
LOOKBACK = 5                 # T 직전 5분 안의 마지막 분봉이 없으면 결측
ENTRY_A, EXIT_A = "14:50", "15:20"   # 다리 A: 연속 세션
# 분봉에서 뽑는 시각 = 30분 경계(09:00 제외) + 14:50(판정 셀 진입)
PRICE_TIMES = sorted(set(BOUNDS[1:]) | {ENTRY_A})
ENTRY_B = "15:20"                    # 다리 B: 15:20 매수 → 일봉 종가(마감 단일가) 매도
COND_AT = "14:50"                    # 조건 변수: 09:00→14:50 수익률

# ── 비용(bp). 97 §4(b) daytrade-it 실측.
TAX_FEE = 20.0 + 3.0
SPREAD_CONT = 18.5                   # 연속 세션 왕복 스프레드
COST_A = TAX_FEE + SPREAD_CONT       # 41.5
SPREAD_CLOSE_AUCTION = {"거래소": 15.1, "코스닥": 18.1}
COST_B = {m: TAX_FEE + SPREAD_CONT / 2 + s / 2 for m, s in SPREAD_CLOSE_AUCTION.items()}  # 39.8 / 41.3

TOP_N = 1_300
MIN_PRICE = 1_000
TAIL = 0.10                           # 상·하위 10%


def price_at_bounds(mb: pd.DataFrame) -> pd.DataFrame:
    """한 날의 분봉(code, ts, close) → code × 경계시각 가격. 경계 T 의 가격은 **ts < T** 인 마지막 분봉 close
    (T−LOOKBACK 분 이상). 09:00 열은 여기서 만들지 않는다(일봉 시가로 채운다)."""
    d = mb[["code", "ts", "close"]].copy()
    d["ts"] = pd.to_datetime(d.ts)
    d["minute"] = d.ts.dt.hour * 60 + d.ts.dt.minute
    cols = {}
    for b in PRICE_TIMES:
        bm = int(b[:2]) * 60 + int(b[3:])
        w = d[(d.minute < bm) & (d.minute >= bm - LOOKBACK)]
        cols[b] = w.sort_values("ts").groupby("code").close.last()
    return pd.DataFrame(cols, index=sorted(d.code.unique()))


def assemble_day(px: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """경계 가격 + 그날 일봉(code, open, close) → 가격 행렬(09:00 = open, 15:30 = close)."""
    out = px.join(daily.set_index("code")[["open", "close"]], how="inner")
    out = out.rename(columns={"open": "09:00", "close": "15:30"})
    return out[["09:00", *PRICE_TIMES, "15:30"]]


def interval_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """구간 수익(bp, 비용 전). 열 이름 'a→b'."""
    out = {}
    for a, b in INTERVALS:
        out[f"{a}→{b}"] = (prices[b] / prices[a] - 1) * 1e4
    return pd.DataFrame(out, index=prices.index)


def labels(prices: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """판정 라벨(bp, 비용 **한 번** 차감).
    y_a = 15:20 / 14:50 − 1 − 41.5bp.  y_b = 15:30(일봉 종가) / 15:20 − 1 − COST_B[시장].
    cond = 09:00(일봉 시가) → 14:50 수익률(bp) — 14:50 에 아는 값."""
    cost_b = market.reindex(prices.index).map(COST_B)
    y_a = (prices[EXIT_A] / prices[ENTRY_A] - 1) * 1e4 - COST_A
    y_b = (prices["15:30"] / prices[ENTRY_B] - 1) * 1e4 - cost_b
    cond = (prices[COND_AT] / prices["09:00"] - 1) * 1e4
    return pd.DataFrame({"y_a": y_a, "y_b": y_b, "cond": cond, "cost_b": cost_b}, index=prices.index)


def tail_bucket(x: pd.Series, tail: float = TAIL) -> pd.Series:
    """그날 유니버스 안에서 상위/하위 tail → 'top' / 'bottom' / 'mid'. 결측은 NaN."""
    r = x.rank(pct=True, method="first")
    out = pd.Series(np.where(r > 1 - tail, "top", np.where(r <= tail, "bottom", "mid")), index=x.index, dtype=object)
    return out.where(x.notna())


def quartile(x: pd.Series) -> pd.Series:
    """그날 유니버스 안 4분위(0~3). 매치드 널 셀 키용."""
    return pd.qcut(x.rank(method="first"), 4, labels=False).astype("float")


def prior_day_features(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉(code, date, close, trade_value) → 각 (date, code) 의 **전일** 값: p_ret1 · p_vol20 · p_tv · p_close.
    전부 shift(1) — 그날 값은 섞이지 않는다."""
    d = daily.sort_values(["code", "date"]).copy()
    g = d.groupby("code", sort=False)
    d["ret1"] = g.close.pct_change()
    d["vol20"] = g.ret1.transform(lambda s: s.rolling(20, min_periods=10).std())
    d["p_ret1"] = g.ret1.shift(1)
    d["p_vol20"] = g.vol20.shift(1)
    d["p_tv"] = g.trade_value.shift(1)
    d["p_close"] = g.close.shift(1)
    return d[["date", "code", "p_ret1", "p_vol20", "p_tv", "p_close"]]


def universe(feat_day: pd.DataFrame, top_n: int = TOP_N, min_price: float = MIN_PRICE) -> pd.DataFrame:
    """하루치 전일 피처 → 전일 종가 ≥ min_price 중 전일 거래대금 상위 top_n."""
    f = feat_day[(feat_day.p_close >= min_price) & feat_day.p_tv.notna()]
    return f.nlargest(top_n, "p_tv", keep="first")
