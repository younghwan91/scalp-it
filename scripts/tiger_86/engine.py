"""86번 호랑이 규칙 · 대기열 지정가 진입 엔진. 사전등록 docs/research/manju/86 §2~§4.

    ~/of80/.rlvenv/bin/python scripts/tiger_86/engine.py 2026-08-24 ... 2026-09-15   # 17일 기술(판정 아님)
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flow_83"))
import features as F83  # noqa: E402

COST = 0.0023
CANCEL_SECS = 10
STOP_TICKS = 2
SDROP = 1.0
SAME_SIZE_MIN = 3
Q_SIG = 0.99
WARMUP_SEC = 2100          # 09:35 전에는 전일 분위수


@numba.njit
def _run(sig_rows, t_sec, t_px, t_vol, t_side, t_start, bid, ask, bq1, stg, tick, n, cost, taker):
    """→ (ret, hold, filled, fill_delay) per signal. taker=True 면 t+1 매도1호가 시장가 진입(비교용)."""
    m = len(sig_rows)
    ret = np.full(m, np.nan)
    hold = np.zeros(m, np.int64)
    filled = np.zeros(m, np.bool_)
    delay = np.full(m, -1)
    free = -1
    for k in range(m):
        row = sig_rows[k]
        if row < free:
            continue
        j = row + 1
        if j + 2 >= n:
            continue
        P = bid[j]
        Qa = bq1[j]
        if not (P > 0) or not (Qa == Qa):
            continue
        peak = stg[j]
        e = -1
        if taker:
            e = j
            P = ask[j]
            if not (P > 0):
                continue
        else:
            cum = 0.0
            for s in range(j + 1, min(j + CANCEL_SECS, n - 1) + 1):
                a = t_start[s]
                b = t_start[s + 1]
                hit = False
                for i in range(a, b):
                    if t_px[i] < P:
                        hit = True
                        break
                    if t_side[i] < 0 and t_px[i] == P:
                        cum += t_vol[i]
                if hit or cum >= Qa + 1:
                    e = s
                    break
                v = stg[s]
                if v == v:
                    if not (peak == peak) or v > peak:
                        peak = v
                    if v <= peak - SDROP:
                        break
            if e < 0:
                free = j + CANCEL_SECS + 1
                continue
        filled[k] = True
        delay[k] = e - j
        peak = stg[e]
        stop = P - STOP_TICKS * tick[j]
        px = bid[n - 1]
        xs = n - 1
        for s in range(e + 1, n):
            if bid[s] <= stop:
                px = bid[s]
                xs = s
                break
            v = stg[s]
            if v == v:
                if not (peak == peak) or v > peak:
                    peak = v
                if v <= peak - SDROP:
                    px = bid[s]
                    xs = s
                    break
        ret[k] = px / P - 1.0 - cost
        hold[k] = xs - e
        free = xs + 1
    return ret, hold, filled, delay


def day_thresholds(rows: pd.DataFrame, prev_q: float | None) -> np.ndarray:
    """분 단위: 그날 그 분까지 유니버스 결정 행 dstr30 의 99% 분위. 09:35 전은 전일 값."""
    r = rows.sort_values("row")
    secs = r.row.to_numpy()
    vals = r.dstr30.to_numpy(np.float64)
    out = np.full(F83.N // 60 + 1, np.inf)
    for mi in range(len(out)):
        s0 = mi * 60
        if s0 < WARMUP_SEC:
            out[mi] = prev_q if prev_q is not None else np.inf
            continue
        k = np.searchsorted(secs, s0)
        v = vals[:k]
        v = v[np.isfinite(v)]
        if len(v) >= 1000:
            out[mi] = float(np.quantile(v, Q_SIG))
    return out


def signals(rows: pd.DataFrame, thr_by_min: np.ndarray) -> pd.Series:
    thr = thr_by_min[rows.row.to_numpy() // 60]
    return ((rows.dstr30.to_numpy() >= thr) & (rows.same_size_cnt10.to_numpy() >= SAME_SIZE_MIN)
            & (rows.buyval3.to_numpy() > rows.sellval3.to_numpy()))


def simulate_day(conn, day: str, prev_q: float | None, *, forward_ok: bool = False, control_seed: int | None = None):
    """→ (거래 DataFrame, 그날 전체 dstr30 99% 분위 — 다음 날 워밍업용)."""
    if not forward_ok:
        assert day < F83.FORWARD_START
    obj = pickle.loads((F83.CACHE / f"{day}.pkl").read_bytes())
    rows, paths = obj["rows"], obj["paths"]
    thr = day_thresholds(rows, prev_q)
    rows = rows.assign(sig=signals(rows, thr))
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    codes = sorted(rows.loc[rows.sig, "code"].unique())
    out = []
    if codes:
        tk = pd.read_sql("select code, ts, price, volume, side from ticks where ts >= %s and ts < %s and code = any(%s) "
                         "order by code, ts, seq", conn, params=(day, nxt, codes))
        qt = pd.read_sql("select code, ts, bidqty1 from quotes where ts >= %s and ts < %s and code = any(%s) order by code, ts",
                         conn, params=(day, nxt, codes))
        tg, qg = dict(tuple(tk.groupby("code"))), dict(tuple(qt.groupby("code")))
        N = F83.N
        for c in codes:
            g = tg[c]
            sec = (g.ts.dt.hour * 3600 + g.ts.dt.minute * 60 + g.ts.dt.second).to_numpy() - F83.T0
            ok = (sec >= 0) & (sec < N)
            sec = sec[ok].astype(np.int64)
            px, vol, side = g.price.to_numpy(float)[ok], g.volume.to_numpy(float)[ok], g.side.to_numpy(np.int64)[ok]
            start = np.searchsorted(sec, np.arange(N + 1))
            bq = np.full(N, np.nan)
            q = qg.get(c)
            if q is not None:
                qs = (q.ts.dt.hour * 3600 + q.ts.dt.minute * 60 + q.ts.dt.second).to_numpy() - F83.T0
                mq = (qs >= 0) & (qs < N)
                bq[qs[mq]] = q.bidqty1.to_numpy(float)[mq]
            bq = F83.ffill(bq)
            p = paths[c]
            rc = rows[rows.code == c]
            sig_rows = np.sort(rc.loc[rc.sig, "row"].to_numpy(np.int64))
            args = (sec, px, vol, side, start, p["bid"].astype(np.float64), p["ask"].astype(np.float64), bq, p["strength"].astype(np.float64),
                    p["tick"].astype(np.float64), N, COST)
            r, h, f, dl = _run(sig_rows, *args, False)
            rt, ht, ft, _ = _run(sig_rows, *args, True)
            rec = pd.DataFrame({"day": day, "code": c, "row": sig_rows, "ret": r, "hold": h, "filled": f,
                                "delay": dl, "ret_taker": rt})
            if control_seed is not None:
                # §5 대조: 같은 종목·같은 날 결정 행 중 무작위 초(신호 수만큼), 같은 체결 모형·청산
                rng = np.random.default_rng(control_seed + len(out))
                pool = rc.row.to_numpy(np.int64)
                cr = np.sort(rng.choice(pool, size=min(len(pool), len(sig_rows)), replace=False))
                rc_, _, fc, _ = _run(cr, *args, False)
                rec["ctrl_day_code_mean"] = float(np.nanmean(rc_[fc])) if fc.any() else np.nan
            out.append(rec)
    vals = rows.dstr30.to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), (float(np.quantile(vals, Q_SIG)) if len(vals) else None)


def main(days: list[str]) -> None:
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    prev_q = None
    res = []
    for d in days:
        t, prev_q = simulate_day(conn, d, prev_q)
        if len(t):
            res.append(t)
            f = t[t.filled]
            print(d, f"신호 {len(t)} 체결 {len(f)} ({len(f) / len(t):.0%}) · 지정가 순 {f.ret.mean() * 1e4:+.1f}bp · "
                  f"같은 신호 시장가 순 {t.ret_taker.mean() * 1e4:+.1f}bp", flush=True)
    conn.close()
    a = pd.concat(res, ignore_index=True)
    f = a[a.filled]
    print("\n== 17일 합계(기술 — 판정 아님)")
    print(f"신호 {len(a)} · 체결 {len(f)} ({len(f) / len(a):.0%}) · 체결 지연 중앙 {f.delay.median():.0f}초")
    print(f"지정가: 세전 {(f.ret.mean() + COST) * 1e4:+.1f}bp · 순 {f.ret.mean() * 1e4:+.1f}bp · 중앙 {f.ret.median() * 1e4:+.1f} · "
          f"승률 {(f.ret > 0).mean():.0%} · 보유 중앙 {f.hold.median():.0f}초 · 일별 양수 "
          f"{(f.groupby('day').ret.mean() > 0).sum()}/{f.day.nunique()}")
    print(f"같은 신호 시장가: 순 {a.ret_taker.mean() * 1e4:+.1f}bp")
    miss = a[~a.filled]
    print(f"미체결 신호가 시장가였다면: 순 {miss.ret_taker.mean() * 1e4:+.1f}bp (역선택 확인용)")
    Path.home().joinpath("of80", "tiger86_explore.csv").write_text(a.to_csv(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
