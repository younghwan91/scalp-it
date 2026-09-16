"""96번 분 단위 피처 — 83·84 의 1~30초 창으로는 **분 단위 맥락이 상태에 없다**. 사전등록 96 §1.

    ~/of80/.rlvenv/bin/python scripts/minute_96/features.py 2026-08-24 ... 2026-09-16   # 날짜별 캐시

83번 일별 캐시(`~/of80/flow83/*.pkl`)의 `paths`(초 격자 배열)만 쓴다 — 원시 틱을 다시 읽지 않는다.
모든 열은 결정 초 t **이하**만 본다. 라벨은 만들지 않는다(96 은 RL 이 청산을 정한다).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 22800
CACHE83 = Path.home() / "of80" / "flow83"
CACHE = Path.home() / "of80" / "flow96"

RET_W = (60, 120, 600, 1800)
STR_W = (60, 300, 600)
VWAP_W = (300, 600)
RANGE_W = (600, 1800)
RV_W = (300, 600)
OFI_W = (300, 600)

FEATURES: tuple[str, ...] = (
    *[f"ret{w}" for w in RET_W], *[f"dstr{w}" for w in STR_W], *[f"vwapdev{w}" for w in VWAP_W],
    "dd_hi", *[f"rangepos{w}" for w in RANGE_W], *[f"rv{w}" for w in RV_W],
    *[f"cofi{w}" for w in OFI_W], "bsr300", "tod",
)


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _rsum(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(np.nan_to_num(x)).rolling(w, min_periods=1).sum().to_numpy()


def minute_features(p: dict) -> dict[str, np.ndarray]:
    """초 격자 배열 묶음 → 분 단위 피처. 키는 `FEATURES` 와 같다."""
    bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)
    last = p["last"].astype(np.float64)
    buy, sell = p["buyval"].astype(np.float64), p["sellval"].astype(np.float64)
    stg = p["strength"].astype(np.float64)
    e = p["e_ofi"].astype(np.float64)
    val = np.nan_to_num(buy) + np.nan_to_num(sell)
    n = len(bid)
    x: dict[str, np.ndarray] = {}

    for w in RET_W:
        x[f"ret{w}"] = (mid / _lag(mid, w) - 1) * 1e4
    for w in STR_W:
        x[f"dstr{w}"] = stg - _lag(stg, w)
    for w in VWAP_W:
        # 체결량이 격자에 없어 **체결대금 가중 평균가**로 VWAP 를 근사한다(정의를 문서에 고정).
        vw = _rsum(np.nan_to_num(last) * val, w) / (_rsum(val, w) + 1e-9)
        x[f"vwapdev{w}"] = np.where(_rsum(val, w) > 0, (mid / vw - 1) * 1e4, 0.0)

    hi_so_far = np.fmax.accumulate(np.where(np.isfinite(mid), mid, -np.inf))
    hi_so_far[~np.isfinite(hi_so_far)] = np.nan
    x["dd_hi"] = (mid / hi_so_far - 1) * 1e4
    s = pd.Series(mid)
    for w in RANGE_W:
        lo = s.rolling(w, min_periods=1).min().to_numpy()
        hi = s.rolling(w, min_periods=1).max().to_numpy()
        x[f"rangepos{w}"] = np.where(hi > lo, (mid - lo) / (hi - lo), 0.5)
    r1 = pd.Series(mid / _lag(mid, 1) - 1)
    for w in RV_W:
        x[f"rv{w}"] = r1.rolling(w, min_periods=10).std().to_numpy() * 1e4
    for w in OFI_W:
        x[f"cofi{w}"] = _rsum(e, w)
    b3, s3 = _rsum(buy, 300), _rsum(sell, 300)
    x["bsr300"] = b3 / (b3 + s3 + 1)
    x["tod"] = np.arange(n, dtype=np.float64) / N
    return {k: np.nan_to_num(x[k], nan=0.0, posinf=0.0, neginf=0.0) for k in FEATURES}


def build_day(day: str) -> pd.DataFrame:
    """그날 83 캐시의 결정 행에 맞춘 분 단위 피처 DataFrame(day·code·row + FEATURES)."""
    obj = pickle.loads((CACHE83 / f"{day}.pkl").read_bytes())
    rows, paths = obj["rows"], obj["paths"]
    parts = []
    for code, g in rows.groupby("code"):
        if code not in paths:
            continue
        x = minute_features(paths[code])
        r = g.row.to_numpy(np.int64)
        df = pd.DataFrame({k: x[k][r].astype(np.float32) for k in FEATURES})
        df.insert(0, "row", r.astype(np.int32))
        df.insert(0, "code", code)
        df.insert(0, "day", day)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(days: list[str]) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    for day in days:
        out = CACHE / f"{day}.pkl"
        if out.exists():
            continue
        df = build_day(day)
        out.write_bytes(pickle.dumps(df, protocol=5))
        print(f"{day} {len(df):,}행")
    return 0


if __name__ == "__main__":
    days = sys.argv[1:] or sorted(p.name[:10] for p in CACHE83.glob("*.pkl"))
    raise SystemExit(main(days))
