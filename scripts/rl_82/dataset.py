"""82번 데이터셋 — 80번 캐시(초 격자 특징+호가 경로)를 에피소드 배열로. 사전등록 §1~§3.

에피소드 = (날짜, 종목). 결정 행 = 체결이 있는 초(09:05~15:10, 80번 build_code 의 keep).
각 결정 행에 시뮬레이터 체결가를 미리 붙인다: buy = 다음 초 매도1호가, sell = 다음 초 매수1호가.
"""
from __future__ import annotations

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

CACHE = Path.home() / "of80" / "cache2"
TRAIN = ("2026-08-24", "2026-09-03")
VAL = ("2026-09-04", "2026-09-07")
OOS = ("2026-09-08", "2026-09-15")
FEATS = ("acc", "buyshare10", "buyshare30", "ret30", "ret300", "ofi10", "imb3", "spread_bp",
         "strength", "dstr10", "dstr30", "dstr60", "v30", "v600", "spread_ticks", "sec")
LOG_FEATS = {"acc", "v30", "v600"}
COST = 0.0023


@dataclass
class Episode:
    day: str
    code: str
    secs: np.ndarray      # int32 결정 초
    X: np.ndarray         # float32 (n, 16) 원시 특징
    buy: np.ndarray       # float64 다음 초 매도1호가
    sell: np.ndarray      # float64 다음 초 매수1호가
    label: np.ndarray     # float32 80번 §7 청산 순수익(모델 A 목표)
    hold: np.ndarray      # int32 그 청산까지 보유 초(무작위 대조용)


@numba.njit(cache=True)
def _strength_exit_labels(secs, bid, ask, stg, sdrop, tmax, cost):
    n = len(secs)
    out = np.full(n, np.nan, np.float32)
    hold = np.zeros(n, np.int32)
    L = len(bid)
    for i in range(n):
        e = secs[i] + 1
        if e >= L - 1:
            continue
        a = ask[e]
        if not (a > 0):
            continue
        peak = stg[e]
        end = min(e + tmax, L - 1)
        px = bid[end]
        xs = end
        for s in range(e + 1, end + 1):
            v = stg[s]
            if v == v:
                if not (peak == peak) or v > peak:
                    peak = v
                if v <= peak - sdrop:
                    px = bid[s]
                    xs = s
                    break
        if px > 0:
            out[i] = px / a - 1.0 - cost
            hold[i] = xs - e
    return out, hold


def _in(day: str, rng: tuple[str, str]) -> bool:
    return rng[0] <= day <= rng[1]


def load(split: str, *, allow_oos: bool = False) -> list[Episode]:
    rng = {"train": TRAIN, "val": VAL, "oos": OOS}[split]
    if split == "oos" and not allow_oos:
        raise PermissionError("OOS 는 최종 평가(evaluate.py --final)만 읽는다 — 사전등록 §1")
    eps: list[Episode] = []
    for f in sorted(glob.glob(str(CACHE / "*.pkl"))):
        if f.endswith(".path.pkl"):
            continue
        day = Path(f).name[:10]
        if not _in(day, rng):
            continue
        if split != "oos":
            assert day <= "2026-09-07", "학습 코드가 OOS 날짜를 읽으려 했다"
        df = pickle.loads(Path(f).read_bytes())
        paths = pickle.loads(Path(f.replace(".pkl", ".path.pkl")).read_bytes())
        df = df[df.strength.notna() & (df.sec <= 22800 - 600)]
        for code, g in df.groupby("code"):
            p = paths[code]
            secs = g.sec.to_numpy(np.int64)
            if len(secs) < 50:
                continue
            nxt = np.minimum(secs + 1, len(p["bid"]) - 1)
            buy = p["ask"][nxt].astype(np.float64)
            sell = p["bid"][nxt].astype(np.float64)
            ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
            g, secs, buy, sell = g[ok], secs[ok], buy[ok], sell[ok]
            if len(secs) < 50:
                continue
            X = np.stack([g[c].to_numpy(np.float64) for c in FEATS], axis=1)
            for j, c in enumerate(FEATS):
                if c in LOG_FEATS:
                    X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
            lab, hold = _strength_exit_labels(secs, p["bid"].astype(np.float64), p["ask"].astype(np.float64),
                                        p["strength"].astype(np.float64), 3.0, 300, COST)
            eps.append(Episode(day, str(code), secs.astype(np.int32), X.astype(np.float32),
                               buy, sell, lab, hold))
    return eps


class Scaler:
    """TRAIN 분위수 클리핑(1~99%) + 표준화 — 사전등록 §3."""

    def __init__(self, eps: list[Episode]) -> None:
        X = np.concatenate([e.X for e in eps])
        X = X[np.isfinite(X).all(axis=1)]
        self.lo, self.hi = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(axis=0), Xc.std(axis=0) + 1e-8

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)
