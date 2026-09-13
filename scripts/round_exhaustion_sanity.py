"""라운드피겨 변곡(매도 소진) sanity. 사전등록 `docs/research/manju/70`.

`quotes`(10호가 잔량)로 이 프로젝트 처음으로 **호가** 기반 신호를 잰다. 라운드
가격 근접 + 매도 잔량(total_ask_qty) 급감을 신호로, forward 체결가(ticks)
수익을 라벨로 쓴다. 10일치뿐이라 확정 배선 판단에는 못 쓴다(사전등록 §6).

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/round_exhaustion_sanity.py
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    _load_env,
    _session_mask,
    bootstrap_diff_ci,
    epoch_seconds,
    forward_labels_all,
    quintile_means,
)
from runway_sanity import round_gap_pct  # noqa: E402

NEAR_PCT = 0.005
COOLDOWN_SEC = 30.0
EXHAUST_WINDOW_SEC = 30.0
HORIZONS = (30, 60, 120)
# 고정 가정값(값은 바꾸지 않는다). 일자별 거래세 스케줄은 krx_quant_core.costs 참고.
COST_BP = (40.0, 50.0, 60.0)
MIN_GROUP_N = 100
RNG_SEED = 20260906
N_BOOT = 1000
DATES = (
    "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
    "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
)


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def _load_quotes_day(conn, date: str) -> dict[str, tuple]:
    cur = conn.cursor()
    cur.execute(
        "select code, ts, ask1, total_ask_qty from quotes "
        "where ts >= %s and ts < %s and ask1 is not null and ask1 > 0 "
        "order by code, ts",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    out: dict[str, tuple] = {}
    for code, group in itertools.groupby(rows, key=lambda r: r[0]):
        g = list(group)
        ts_pd = pd.to_datetime([r[1] for r in g])
        ts_sec = epoch_seconds(ts_pd)
        hms = ts_pd.strftime("%H:%M:%S").to_numpy()
        ask1 = np.array([float(r[2]) for r in g])
        taq = np.array([float(r[3]) if r[3] is not None else np.nan for r in g])
        out[str(code).strip()] = (ts_sec, hms, ask1, taq)
    return out


def _load_ticks_day(conn, date: str) -> dict[str, tuple]:
    cur = conn.cursor()
    cur.execute(
        "select code, ts, price from ticks where ts >= %s and ts < %s order by code, ts, seq",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    out: dict[str, tuple] = {}
    for code, group in itertools.groupby(rows, key=lambda r: r[0]):
        g = list(group)
        ts_pd = pd.to_datetime([r[1] for r in g])
        ts_sec = epoch_seconds(ts_pd)
        price = np.array([float(r[2]) if r[2] is not None else np.nan for r in g])
        out[str(code).strip()] = (ts_sec, price)
    return out


def find_triggers(ts_sec, hms, ask1, taq) -> tuple[list[int], np.ndarray]:
    """근접 조건(라운드 0.5% 이내) 통과 인덱스(쿨다운 적용) + exhaust_ratio 배열."""
    n = ts_sec.shape[0]
    exhaust = np.full(n, np.nan)
    # exhaust_ratio: taq(t) / mean(taq, [t-30s, t))  — 슬라이딩 윈도 합/카운트
    s = 0.0
    cnt = 0
    left = 0
    for i in range(n):
        lo = ts_sec[i] - EXHAUST_WINDOW_SEC
        while left < i and ts_sec[left] <= lo:
            if np.isfinite(taq[left]):
                s -= taq[left]
                cnt -= 1
            left += 1
        if cnt > 0 and np.isfinite(taq[i]):
            base = s / cnt
            if base > 0:
                exhaust[i] = taq[i] / base
        if np.isfinite(taq[i]):
            s += taq[i]
            cnt += 1
    in_sess = _session_mask(hms)
    near = np.array([round_gap_pct(p) <= NEAR_PCT for p in ask1])
    eligible = in_sess & near & np.isfinite(exhaust)
    out: list[int] = []
    last_ts = -np.inf
    for i in np.flatnonzero(eligible):
        t = ts_sec[i]
        if t - last_ts < COOLDOWN_SEC:
            continue
        out.append(int(i))
        last_ts = t
    return out, exhaust


def _price_at(ts_sec: np.ndarray, price: np.ndarray, t: float) -> tuple[int, float] | None:
    """``t`` 이하 마지막 체결 인덱스·가격. ticks 는 quotes 보다 성기므로 근사."""
    idx = int(np.searchsorted(ts_sec, t, side="right")) - 1
    if idx < 0:
        return None
    return idx, float(price[idx])


class Accum:
    def __init__(self) -> None:
        self.n_codes = 0
        self.n_candidates = 0
        self.n_labeled = 0
        self.exhaust_all: list[float] = []
        self.close = {n: [] for n in HORIZONS}
        self.mfe = {n: [] for n in HORIZONS}
        self.exhaust_by_n = {n: [] for n in HORIZONS}


def process_code(acc: Accum, quote_data, tick_data) -> None:
    ts_sec, hms, ask1, taq = quote_data
    acc.n_codes += 1
    idxs, exhaust = find_triggers(ts_sec, hms, ask1, taq)
    if not idxs or tick_data is None:
        acc.n_candidates += len(idxs)
        return
    acc.n_candidates += len(idxs)
    tick_ts, tick_price = tick_data
    if tick_ts.shape[0] < 2:
        return
    fwd = {n: forward_labels_all(tick_ts, tick_price, n) for n in HORIZONS}
    for i in idxs:
        found = _price_at(tick_ts, tick_price, float(ts_sec[i]))
        if found is None:
            continue
        j, _entry_price = found
        acc.exhaust_all.append(float(exhaust[i]))
        acc.n_labeled += 1
        for n in HORIZONS:
            mfe_i, close_i = fwd[n][0][j], fwd[n][1][j]
            if np.isfinite(close_i):
                acc.close[n].append(close_i)
                acc.mfe[n].append(mfe_i)
                acc.exhaust_by_n[n].append(float(exhaust[i]))


def build_report(acc: Accum) -> str:
    L = ["# 라운드피겨 변곡(매도 소진) sanity — 2026-08-24~09-04", ""]
    L.append("> 사전등록 `docs/research/manju/70`. 대장 자신의 forward 수익 대리"
              "(후속주 페어링 범위 밖). 10일치뿐 — 확정 배선 판단에는 못 쓴다.")
    L.append("")
    L.append(f"- 종목 {acc.n_codes} · 근접+쿨다운 통과 후보 {acc.n_candidates:,}건 "
              f"· 체결가 라벨 성공 {acc.n_labeled:,}건")
    exhaust_all = _arr(acc.exhaust_all)
    L.append("")
    if exhaust_all.size:
        pcts = [5, 25, 50, 75, 95]
        vals = np.percentile(exhaust_all, pcts)
        L.append("## 0. exhaust_ratio 분포")
        L.append("")
        L.append("| 분위 | " + " | ".join(f"P{p}" for p in pcts) + " |")
        L.append("|---|" + "---|" * len(pcts))
        L.append("| ratio | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        L.append("")

    L.append("## 1. exhaust_ratio 중앙값 문턱 pass(소진, ≤중앙값) vs fail(>중앙값)")
    L.append("")
    L.append(
        "| N | pass n | fail n | pass close | fail close | 차 close(95%CI) | "
        "pass MFE | fail MFE | 차 MFE(95%CI) | net_close 50bp | 게이트 |"
    )
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    verdict_rows = []
    med = float(np.median(exhaust_all)) if exhaust_all.size else float("nan")
    for n in HORIZONS:
        ex_n = _arr(acc.exhaust_by_n[n])
        close_n = _arr(acc.close[n])
        mfe_n = _arr(acc.mfe[n])
        if ex_n.size == 0:
            L.append(f"| {n} | 0 | 0 | - | - | - | - | - | - | - | 표본없음 |")
            continue
        mask = ex_n <= med
        p_c, f_c = close_n[mask], close_n[~mask]
        p_m, f_m = mfe_n[mask], mfe_n[~mask]
        gated = p_c.size >= MIN_GROUP_N and f_c.size >= MIN_GROUP_N
        d, lo, hi = bootstrap_diff_ci(p_c, f_c, n_boot=N_BOOT, seed=RNG_SEED)
        dm, lom, him = bootstrap_diff_ci(p_m, f_m, n_boot=N_BOOT, seed=RNG_SEED)
        pc_mean = p_c.mean() if p_c.size else float("nan")
        fc_mean = f_c.mean() if f_c.size else float("nan")
        pm_mean = p_m.mean() if p_m.size else float("nan")
        fm_mean = f_m.mean() if f_m.size else float("nan")
        net = pc_mean * 10000 - 50.0 if p_c.size else float("nan")
        gate_txt = "OK" if gated else f"표본부족(<{MIN_GROUP_N})"
        L.append(
            f"| {n} | {p_c.size} | {f_c.size} | {pc_mean*100:+.3f} | {fc_mean*100:+.3f} | "
            f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | {pm_mean*100:+.3f} | {fm_mean*100:+.3f} | "
            f"{dm*100:+.3f} [{lom*100:+.3f},{him*100:+.3f}] | {net:+.1f}bp | {gate_txt} |"
        )
        if gated:
            dir_ok = np.isfinite(d) and lo > 0 and np.isfinite(dm) and lom > 0
            net_ok = np.isfinite(net) and net >= 0
            verdict_rows.append((n, dir_ok, net_ok))
    L.append("")
    L.append("net_close 50bp = pass close 평균 − 50bp.")
    L.append("")

    L.append("## 2. exhaust_ratio 5분위 → close 60s (Q1=가장 소진)")
    L.append("")
    mono_ok = False
    ex60 = _arr(acc.exhaust_by_n[60])
    close60 = _arr(acc.close[60])
    if ex60.size >= 5:
        qm = quintile_means(ex60, close60)
        L.append("| 분위 | n | exhaust평균 | ret_close_60 평균(%) |")
        L.append("|--:|--:|--:|--:|")
        vals = []
        for q, row in enumerate(qm):
            cnt, smean, rmean = row
            vals.append(rmean)
            L.append(f"| Q{q+1} | {cnt} | {smean:.3f} | {rmean*100:+.3f} |")
        # 가설: 소진(Q1, exhaust 작음)일수록 forward 좋아야 함 → Q1 > Q5
        if len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]):
            mono_ok = vals[0] > vals[4]
    else:
        L.append("_표본 부족(<5) — 분위 생략._")
    L.append("")

    L.append("## 판정 (사전등록 §5)")
    L.append("")
    hit = [n for n, dir_ok, net_ok in verdict_rows if dir_ok and net_ok and mono_ok]
    if hit:
        L.append(f"**신호 있음(잠정) → 정식 채택 검토.** N={hit} 에서 방향·비용후·단조성 충족.")
    elif not verdict_rows:
        L.append("**표본 부족 — 판정 불가.**")
    else:
        L.append("**미확인.** 방향(부트CI)·비용후(net≥0)·단조성(Q1>Q5) 을 모두 만족하는 N 이 없다.")
    L.append("")
    L.append("_주의: 10일치뿐(56·57번과 같은 저유동성/짧은 창 캐베어트). "
              "대장 자신의 forward 수익 대리 측정._")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="라운드피겨 변곡(매도 소진) sanity")
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    acc = Accum()
    try:
        for date in DATES:
            quotes_by_code = _load_quotes_day(conn, date)
            if not quotes_by_code:
                continue
            ticks_by_code = _load_ticks_day(conn, date)
            for code, qdata in quotes_by_code.items():
                process_code(acc, qdata, ticks_by_code.get(code))
    finally:
        conn.close()

    report = build_report(acc)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "round_exhaustion_sanity.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[round_exhaustion_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
