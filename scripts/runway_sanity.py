"""상단 런웨이(라운드피겨·전고점) 필터 sanity. 사전등록 `docs/research/manju/58`.

`accel_sanity_minute.py`(56·57번)과 같은 트리거(4분 +2% 서지)·같은 국면별
층화(20일씩)·같은 검정 절차 위에서, "그 순간 위가 얼마나 열려 있었나"(라운드
피겨 갭·전고점 갭)만 새 신호로 바꿔 잰다. 2차 VI 는 범위 밖(사전등록 §0).

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/runway_sanity.py
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    _load_env,
    _session_mask,
    bootstrap_diff_ci,
    epoch_seconds,
    forward_labels_all,
    quintile_means,
)

from scalp_it.market_regime import (  # noqa: E402
    COLD,
    DEAD,
    HOT,
    MILD,
    classify_regime,
    fetch_market_turnover,
)

from accel_sanity_minute import (  # noqa: E402
    _ret_series,
    _select_dates,
)

# ---------------------------------------------------------------- 고정 파라미터 (58번 사전등록)
WINDOW_MIN = 4.0
SURGE_RET = 0.02
COOLDOWN_SEC = WINDOW_MIN * 60.0
SESSION_START = "09:00:00"
SESSION_END = "15:20:00"
HORIZONS_MIN = (5, 10, 20)
HORIZONS_SEC = tuple(int(n * 60) for n in HORIZONS_MIN)
MIN_GROUP_N = 100
RNG_SEED = 20260906
N_BOOT = 1000
PRIOR_HIGH_LOOKBACK_DAYS = 60
REGIMES = (HOT, MILD, COLD, DEAD)


def round_step(p: float) -> float:
    if p < 1_000:
        return 100.0
    if p < 5_000:
        return 500.0
    if p < 10_000:
        return 1_000.0
    if p < 50_000:
        return 5_000.0
    if p < 100_000:
        return 10_000.0
    return 50_000.0


def round_gap_pct(p: float) -> float:
    step = round_step(p)
    n = np.floor(p / step)
    next_level = (n + 1) * step
    if abs(p - n * step) < 1e-9:
        next_level = (n + 1) * step  # 이미 배수여도 위쪽 다음 레벨 기준
    return float((next_level - p) / p)


def find_surge_triggers(ts_sec: np.ndarray, price: np.ndarray, hms: np.ndarray) -> list[int]:
    in_sess = _session_mask(hms)
    out: list[int] = []
    last_ts = -np.inf
    w_sec = WINDOW_MIN * 60.0
    for i in np.flatnonzero(in_sess):
        t = ts_sec[i]
        if t - last_ts < COOLDOWN_SEC:
            continue
        r = _ret_series(ts_sec, price, i, w_sec)
        if r is not None and r >= SURGE_RET:
            out.append(int(i))
            last_ts = t
    return out


class Accum:
    def __init__(self) -> None:
        self.n_codes = 0
        self.n_triggers = 0
        self.round_gap = {n: [] for n in HORIZONS_SEC}
        self.prior_gap = {n: [] for n in HORIZONS_SEC}
        self.close = {n: [] for n in HORIZONS_SEC}
        self.mfe = {n: [] for n in HORIZONS_SEC}


def _prior_high(conn_cache: dict, code: str, date) -> float | None:
    return conn_cache.get(code, {}).get(date)


def process_code(
    acc: Accum, ts_sec: np.ndarray, hms: np.ndarray, price: np.ndarray,
    prior_high: float | None,
) -> None:
    acc.n_codes += 1
    triggers = find_surge_triggers(ts_sec, price, hms)
    if not triggers or prior_high is None:
        return
    fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS_SEC}
    for i in triggers:
        p = float(price[i])
        if p <= 0:
            continue
        acc.n_triggers += 1
        rgap = round_gap_pct(p)
        pgap = float((prior_high - p) / p)
        for n in HORIZONS_SEC:
            mfe_i, close_i = fwd[n][0][i], fwd[n][1][i]
            if np.isfinite(close_i):
                acc.close[n].append(close_i)
                acc.mfe[n].append(mfe_i)
                acc.round_gap[n].append(rgap)
                acc.prior_gap[n].append(pgap)


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def _report_feature(acc: Accum, feature: dict, label: str) -> list[str]:
    L = [f"### {label}", ""]
    ref_n = HORIZONS_SEC[1]
    sig = _arr(feature[ref_n])
    ret = _arr(acc.close[ref_n])
    if sig.size < 5:
        L.append("_표본 부족(<5) — 분위 생략._")
        L.append("")
        return L
    qm = quintile_means(sig, ret)
    L.append("| 분위 | n | 신호평균(%) | ret_close(N=600s) 평균(%) |")
    L.append("|--:|--:|--:|--:|")
    vals = []
    for q, row in enumerate(qm):
        cnt, smean, rmean = row
        vals.append(rmean)
        L.append(f"| Q{q+1} | {cnt} | {smean*100:.3f} | {rmean*100:+.3f} |")
    mono_ok = len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]) and vals[4] > vals[0]
    L.append("")

    # 문턱(중앙값) 기준 pass(gap>=median, 런웨이 열림) vs fail
    verdict_rows = []
    med = float(np.median(sig))
    L.append(f"중앙값 문턱({med*100:.3f}%) pass(≥) vs fail(<):")
    L.append("")
    L.append("| N | pass n | fail n | pass close | fail close | 차(95%CI) | 게이트 |")
    L.append("|--:|--:|--:|--:|--:|--:|---|")
    for n in HORIZONS_SEC:
        sig_n = _arr(feature[n])
        close_n = _arr(acc.close[n])
        mask = sig_n >= med
        p_c, f_c = close_n[mask], close_n[~mask]
        gated = p_c.size >= MIN_GROUP_N and f_c.size >= MIN_GROUP_N
        d, lo, hi = bootstrap_diff_ci(p_c, f_c, n_boot=N_BOOT, seed=RNG_SEED)
        pc = p_c.mean() if p_c.size else float("nan")
        fc = f_c.mean() if f_c.size else float("nan")
        L.append(
            f"| {n} | {p_c.size} | {f_c.size} | {pc*100:+.3f} | {fc*100:+.3f} | "
            f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | "
            f"{'OK' if gated else f'표본부족(<{MIN_GROUP_N})'} |"
        )
        if gated:
            verdict_rows.append((n, np.isfinite(d) and lo > 0))
    L.append("")
    dir_ok = any(ok for _n, ok in verdict_rows)
    if not verdict_rows:
        L.append("**판정: 표본 부족.**")
    elif dir_ok and mono_ok:
        L.append("**판정: 신호 있음(가설 방향) — 런웨이 클수록 forward 좋음.**")
    else:
        L.append(f"**판정: 미확인.** 방향OK={dir_ok}, 단조성OK={mono_ok}")
    L.append("")
    return L


def build_report(pooled: Accum, per_regime: dict[str, Accum], selected: dict) -> str:
    L = ["# 상단 런웨이(라운드피겨·전고점) 필터 sanity", ""]
    L.append("> 사전등록 `docs/research/manju/58`. accel(56·57번)과 같은 트리거·같은"
              " 국면별 층화. 2차 VI 는 범위 밖.")
    L.append("")
    L.append("선택 날짜 수: " + ", ".join(f"{k} {len(v)}일" for k, v in selected.items()))
    L.append("")

    for title, acc in [("전체 풀링", pooled)] + [(f"국면: {r}", per_regime[r]) for r in REGIMES]:
        L.append(f"## {title}")
        L.append("")
        L.append(f"- 종목 {acc.n_codes} · 트리거 {acc.n_triggers:,}건")
        L.append("")
        if acc.n_triggers == 0:
            L.append("_트리거 없음._")
            L.append("")
            continue
        L += _report_feature(acc, acc.round_gap, "(a) 라운드피겨 갭")
        L += _report_feature(acc, acc.prior_gap, "(b) 전고점 갭(60거래일)")
    return "\n".join(L)


def _load_date_minute(conn, date) -> dict[str, tuple]:
    cur = conn.cursor()
    cur.execute(
        "select code, ts, close from minute_bars where ts >= %s and ts < %s order by code, ts",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    out: dict[str, tuple] = {}
    for code, group in itertools.groupby(rows, key=lambda r: r[0]):
        g = list(group)
        ts = [r[1] for r in g]
        price = np.array([float(r[2]) if r[2] is not None else np.nan for r in g])
        import pandas as pd

        ts_pd = pd.to_datetime(ts)
        ts_sec = epoch_seconds(ts_pd)
        hms = ts_pd.strftime("%H:%M:%S").to_numpy()
        out[str(code).strip()] = (ts_sec, hms, price)
    return out


def _load_prior_highs(conn, codes: list[str], date) -> dict[str, float]:
    """``date`` **이전 영업일까지** ``PRIOR_HIGH_LOOKBACK_DAYS``일 고가 최대값."""
    if not codes:
        return {}
    cur = conn.cursor()
    cur.execute(
        """
        select code, max(high) from (
            select code, high, row_number() over (
                partition by code order by date desc
            ) as rn
            from daily_bars
            where code = any(%s) and date < %s
        ) t
        where rn <= %s
        group by code
        """,
        (codes, date, PRIOR_HIGH_LOOKBACK_DAYS),
    )
    rows = cur.fetchall()
    cur.close()
    return {str(c).strip(): float(h) for c, h in rows if h is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="런웨이(라운드피겨·전고점) 필터 sanity (읽기 전용)")
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("select distinct date(ts) from minute_bars order by 1")
        all_dates = [r[0] for r in cur.fetchall()]
        cur.close()

        selected = _select_dates(conn, all_dates)
        print("[runway_sanity] 선택된 날짜 수:", {k: len(v) for k, v in selected.items()})

        pooled = Accum()
        per_regime: dict[str, Accum] = {r: Accum() for r in REGIMES}
        for regime, dates in selected.items():
            for date in dates:
                per_code = _load_date_minute(conn, date)
                prior_highs = _load_prior_highs(conn, list(per_code.keys()), date)
                for code, (ts_sec, hms, price) in per_code.items():
                    ph = prior_highs.get(code)
                    process_code(per_regime[regime], ts_sec, hms, price, ph)
                    process_code(pooled, ts_sec, hms, price, ph)
    finally:
        conn.close()

    report = build_report(pooled, per_regime, selected)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "runway_sanity.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[runway_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
