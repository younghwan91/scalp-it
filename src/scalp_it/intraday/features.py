"""장중 궤적 피처 — 8차 사전등록(`docs/research/manju/06-closebet-preregistration.md`).

**계약은 하나다: 모든 피처는 15:19 까지의 정보만 쓴다.**
장은 15:19 에 접속매매가 끊기고(15:20~15:29 봉은 아예 없다) 15:30 단일가가
종가가 된다. 종가베팅은 15:19 근처에 주문을 내므로, 15:30 봉은 **결과**이지
**판단 재료**가 아니다. 이 파일의 어떤 피처도 15:30 을 보지 않으며
`tests/test_intraday_features.py` 가 "15:30 봉을 넣어도 피처가 그대로"임을
회귀 테스트로 강제한다. 15:30 종가는 ``p1530`` 으로 따로 실어 보낸다.

피처 정의는 사전등록에 고정돼 있다. **추가·변경 금지.**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 반환 프레임의 피처 컬럼. 순서가 계약이다.
INTRADAY_FEATURE_COLUMNS: list[str] = [
    "late_ret",
    "high_time_pct",
    "tv_pm_share",
    "n_shock",
    "max_dd",
    "recovery",
    "trend_r2",
    "close_vs_vwap",
]

#: ``p1530`` 은 피처가 아니라 라벨 재료다. 그래서 피처 목록 뒤에 붙인다.
OUTPUT_COLUMNS: list[str] = [
    "code", "date", "p1519", "tv1519", *INTRADAY_FEATURE_COLUMNS, "p1530",
]

_MINUTE_COLUMNS = ["code", "ts", "open", "high", "low", "close", "volume", "trade_value"]

#: 09:00 첫 봉부터 15:19 마지막 접속매매 봉까지의 분 수.
_OPEN_MINUTE = 9 * 60
_CUTOFF_MINUTE = 15 * 60 + 19
_PM_MINUTE = 13 * 60
_LATE_MINUTE = 15 * 60
_CLOSE_MINUTE = 15 * 60 + 30
_SPAN_MINUTES = _CUTOFF_MINUTE - _OPEN_MINUTE

#: 1분 수익률이 이 값을 넘으면 충격 1회 (VI 근사).
_SHOCK_THRESHOLD = 0.03


def _safe_div(num: float, den: float) -> float:
    """0 나눗셈을 NaN 으로 만든다 (inf 를 만들지 않는다)."""
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num) / float(den)


def _trend_r2(close: np.ndarray) -> float:
    """종가를 봉 순번에 회귀한 R² 에 기울기 부호를 곱한다.

    봉이 3개 미만이면 0 이다 — 2점은 항상 R²=1 이라 '일관성'을 재지 못한다.
    종가가 완전히 평평하면 분산이 0 이라 상관을 정의할 수 없으므로 역시 0.
    """
    n = close.size
    if n < 3:
        return 0.0
    if np.ptp(close) == 0:
        return 0.0
    x = np.arange(n, dtype=float)
    slope, _ = np.polyfit(x, close, 1)
    corr = float(np.corrcoef(x, close)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr * corr * float(np.sign(slope))


def _one_day(day: pd.DataFrame) -> dict | None:
    """한 (종목, 일자) 의 피처. 15:19 까지의 봉이 없으면 ``None``."""
    minute = day["_minute"].to_numpy()
    session = day[minute <= _CUTOFF_MINUTE]
    if session.empty:
        return None

    close = session["close"].to_numpy(dtype=float)
    high = session["high"].to_numpy(dtype=float)
    volume = session["volume"].to_numpy(dtype=float)
    tv = session["trade_value"].to_numpy(dtype=float)
    smin = session["_minute"].to_numpy()

    p1519 = float(close[-1])
    tv1519 = float(np.nansum(tv))

    # 1. late_ret — 15:00 봉이 없으면 그 이후 첫 봉을 기준으로 잡는다.
    late_idx = np.flatnonzero(smin >= _LATE_MINUTE)
    if late_idx.size:
        base = close[late_idx[0]]
        late_ret = _safe_div(p1519 - base, base)
    else:
        late_ret = float("nan")

    # 2. high_time_pct — 고가 시각의 위치. 봉 순번이 아니라 시계로 잰다
    #    (결측 봉이 있어도 '언제였나'가 흔들리지 않는다).
    hi_at = smin[int(np.argmax(high))]
    high_time_pct = float(np.clip((hi_at - _OPEN_MINUTE) / _SPAN_MINUTES, 0.0, 1.0))

    # 3. tv_pm_share — 오후장(13:00 이후) 거래대금 비중.
    tv_pm_share = _safe_div(float(np.nansum(tv[smin >= _PM_MINUTE])), tv1519)

    # 4. n_shock — 1분 수익률 절대값 3% 초과 봉 수.
    if close.size >= 2:
        prev = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(prev != 0, close[1:] / prev - 1.0, np.nan)
        n_shock = int(np.nansum(np.abs(step) > _SHOCK_THRESHOLD))
    else:
        n_shock = 0

    # 5·6. max_dd 와 그 저점으로부터의 회복. 둘은 같은 저점을 공유해야 하므로
    #      한 번에 계산한다.
    running_max = np.maximum.accumulate(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max != 0, 1.0 - close / running_max, np.nan)
    if np.isfinite(dd).any():
        trough = int(np.nanargmax(dd))
        max_dd = float(dd[trough])
    else:
        trough, max_dd = 0, float("nan")
    # 낙폭이 없으면 회복도 없다 — 저점이 15:19 종가 자신이라 0 이 정의값이다.
    recovery = _safe_div(p1519 - close[trough], close[trough]) if max_dd > 0 else 0.0

    vwap = _safe_div(float(np.nansum(close * volume)), float(np.nansum(volume)))

    row = {
        "p1519": p1519,
        "tv1519": tv1519,
        "late_ret": late_ret,
        "high_time_pct": high_time_pct,
        "tv_pm_share": tv_pm_share,
        "n_shock": n_shock,
        "max_dd": max_dd,
        "recovery": recovery,
        "trend_r2": _trend_r2(close),
        "close_vs_vwap": _safe_div(p1519 - vwap, vwap),
    }

    # p1530 은 피처가 아니다. 단일가 봉이 없으면 NaN 으로 남긴다.
    tail = day[day["_minute"] >= _CLOSE_MINUTE]
    row["p1530"] = float(tail["close"].to_numpy(dtype=float)[-1]) if len(tail) else float("nan")
    return row


def intraday_features(bars: pd.DataFrame) -> pd.DataFrame:
    """분봉에서 장중 궤적 피처 8개를 만든다.

    Args:
        bars: ``code, ts, open, high, low, close, volume, trade_value``.
            ``trade_value`` 는 수집원(키움 ka10080)에 거래대금 필드가 없어
            ``close × volume`` 근사다 — 비중(``tv_pm_share``)에만 쓰므로
            수준이 아니라 하루 안의 상대 비율만 의미가 있다.

    Returns:
        (종목, 일자) 한 행씩. 컬럼은 :data:`OUTPUT_COLUMNS`.
        15:19 까지의 봉이 하나도 없는 (종목, 일자) 는 결과에서 빠진다.
    """
    missing = [c for c in _MINUTE_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars 에 컬럼이 없다: {missing}")
    if bars.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = bars.loc[:, _MINUTE_COLUMNS].copy()
    df["code"] = df["code"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "trade_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ts"].notna() & df["close"].notna()]
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["date"] = df["ts"].dt.normalize()
    df["_minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df.sort_values(["code", "ts"], kind="stable")

    records = []
    for (code, date), day in df.groupby(["code", "date"], sort=True):
        row = _one_day(day)
        if row is None:
            continue
        records.append({"code": code, "date": date, **row})

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.DataFrame.from_records(records)
    return out.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
