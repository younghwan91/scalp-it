"""라운드피겨 매도소진 — 후속주(2등주) forward sanity. 사전등록 `docs/research/manju/71`.

70번(다른 워크트리 main 브랜치 `scripts/round_exhaustion_sanity.py`)은 대장 자신의
forward만 쟀다. 이 스크립트는 같은 근접·소진 신호(L 기준)를 쓰되, L이 그 순간
자기 섹터(27종) 안에서 진짜 1위(대장)인 이벤트만 남기고 2위 종목(2등주)의
forward 체결가로 라벨을 낸다. 트리거·소진 계산 로직은 70번 스크립트와 동일한
정의를 이 워크트리에 있는 파일만으로 재구현했다(두 브랜치가 아직 병합 전이라
import 할 수 없다).

읽기 전용, DB만 읽는다, 실주문 없음.

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/round_exhaustion_follower_sanity.py
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

from scalp_it.adapters.db import load_stock_master  # noqa: E402

NEAR_PCT = 0.005
COOLDOWN_SEC = 30.0
EXHAUST_WINDOW_SEC = 30.0
HORIZONS = (30, 60, 120)
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
    """70번과 동일한 정의: 근접(라운드 0.5% 이내) 통과 인덱스(쿨다운) + exhaust_ratio."""
    n = ts_sec.shape[0]
    exhaust = np.full(n, np.nan)
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
    idx = int(np.searchsorted(ts_sec, t, side="right")) - 1
    if idx < 0:
        return None
    return idx, float(price[idx])


def _sector_rank(sector_codes: list[str], ticks_by_code: dict, t: float) -> list[tuple[str, float]]:
    """섹터 내 종목들의 t 시점까지 누적수익률(시가 대비) 내림차순. 틱 없는 종목은 제외."""
    out = []
    for c in sector_codes:
        data = ticks_by_code.get(c)
        if data is None:
            continue
        ts_sec, price = data
        found = _price_at(ts_sec, price, t)
        if found is None:
            continue
        j, p_t = found
        if j < 0 or not np.isfinite(price[0]) or price[0] <= 0:
            continue
        ret = p_t / float(price[0]) - 1.0
        out.append((c, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


class Accum:
    def __init__(self) -> None:
        self.n_events_raw = 0
        self.n_leader_confirmed = 0
        self.n_labeled = 0
        self.exhaust_all: list[float] = []
        self.close = {n: [] for n in HORIZONS}
        self.mfe = {n: [] for n in HORIZONS}
        self.exhaust_by_n = {n: [] for n in HORIZONS}


def process_day(acc: Accum, quotes_by_code: dict, ticks_by_code: dict, sector_of: dict[str, str]) -> None:
    sector_groups: dict[str, list[str]] = {}
    for c in quotes_by_code:
        s = sector_of.get(c)
        if s:
            sector_groups.setdefault(s, []).append(c)

    follower_fwd_cache: dict[str, dict[int, tuple]] = {}

    def follower_fwd(code: str) -> dict[int, tuple]:
        if code not in follower_fwd_cache:
            data = ticks_by_code.get(code)
            if data is None:
                follower_fwd_cache[code] = {}
            else:
                ts_sec, price = data
                follower_fwd_cache[code] = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS}
        return follower_fwd_cache[code]

    for code, qdata in quotes_by_code.items():
        ts_sec, hms, ask1, taq = qdata
        idxs, exhaust = find_triggers(ts_sec, hms, ask1, taq)
        acc.n_events_raw += len(idxs)
        if not idxs:
            continue
        sector = sector_of.get(code)
        if not sector:
            continue
        group = sector_groups.get(sector, [])
        if len(group) < 2:
            continue
        for i in idxs:
            t = float(ts_sec[i])
            ranked = _sector_rank(group, ticks_by_code, t)
            if len(ranked) < 2 or ranked[0][0] != code:
                continue  # L이 이 순간 섹터 1위가 아니면 버린다
            follower_code = ranked[1][0]
            acc.n_leader_confirmed += 1
            fwd = follower_fwd(follower_code)
            if not fwd:
                continue
            f_ts_sec, f_price = ticks_by_code[follower_code]
            found = _price_at(f_ts_sec, f_price, t)
            if found is None:
                continue
            j, _entry = found
            acc.exhaust_all.append(float(exhaust[i]))
            acc.n_labeled += 1
            for n in HORIZONS:
                mfe_arr, close_arr = fwd[n]
                if j < close_arr.shape[0] and np.isfinite(close_arr[j]):
                    acc.close[n].append(float(close_arr[j]))
                    acc.mfe[n].append(float(mfe_arr[j]))
                    acc.exhaust_by_n[n].append(float(exhaust[i]))


def build_report(acc: Accum) -> str:
    L = ["# 라운드피겨 매도소진 — 후속주(2등주) forward sanity — 2026-08-24~09-04", ""]
    L.append("> 사전등록 `docs/research/manju/71`. 70번(대장 자신 forward, net -46.7bp 기각)의 "
              "후속 시험 — 신호는 대장(L) 기준, 라벨은 L이 섹터 1위로 확인된 순간의 2등주 forward.")
    L.append("")
    L.append(f"- 근접+쿨다운 통과 후보(전체, 대장확인 전) {acc.n_events_raw:,}건")
    L.append(f"- 그중 L이 섹터 1위로 확인된 이벤트 {acc.n_leader_confirmed:,}건")
    L.append(f"- 2등주 체결가 라벨 성공 {acc.n_labeled:,}건")
    exhaust_all = _arr(acc.exhaust_all)
    L.append("")
    if exhaust_all.size:
        pcts = [5, 25, 50, 75, 95]
        vals = np.percentile(exhaust_all, pcts)
        L.append("## 0. exhaust_ratio 분포 (L 기준, 라벨 성공 표본)")
        L.append("")
        L.append("| 분위 | " + " | ".join(f"P{p}" for p in pcts) + " |")
        L.append("|---|" + "---|" * len(pcts))
        L.append("| ratio | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        L.append("")

    L.append("## 1. exhaust_ratio 중앙값 문턱 pass(소진, ≤중앙값) vs fail(>중앙값) — 2등주 forward")
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

    L.append("## 2. exhaust_ratio 5분위 → 2등주 close 60s (Q1=가장 소진)")
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
    L.append("_주의: 10일치·443종목 부분집합 내 27종 섹터 근사(§6 한계). "
              "동시 동조와 진짜 전이를 이 설계로는 못 가른다 — 70번(대장 자신 net -46.7bp)과 "
              "나란히 놓고 해석할 것._")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="라운드피겨 매도소진 — 후속주 forward sanity")
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    acc = Accum()
    try:
        master = load_stock_master(conn)
        sector_of = dict(zip(master["code"].astype(str).str.strip(), master["sector"]))
        for date in DATES:
            quotes_by_code = _load_quotes_day(conn, date)
            if not quotes_by_code:
                continue
            ticks_by_code = _load_ticks_day(conn, date)
            process_day(acc, quotes_by_code, ticks_by_code, sector_of)
    finally:
        conn.close()

    report = build_report(acc)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "round_exhaustion_follower_sanity.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[round_exhaustion_follower_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
