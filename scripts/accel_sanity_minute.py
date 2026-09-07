"""체결 가속도(accel) 필터 — 분봉 일반화·국면별 재검증. 사전등록 `docs/research/manju/57`.

`56`(틱, 10일·전부 저유동성 국면)의 "미확인 — 방향이 가설과 반대"가 국면 탓인지
보려고 이미 있는 `minute_bars`(2025-08-01~2026-08-28, 262거래일)를 국면별로
층화 추출해 같은 절차를 반복한다. **56번을 대체하지 않는다** — 분봉은 1분
해상도라 56번의 30초 신호를 근사할 뿐이다(사전등록 §0·§2).

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/accel_sanity_minute.py
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

# ---------------------------------------------------------------- 고정 파라미터 (57번 사전등록)
WINDOW_MIN = 4.0                 # 서지 관측 창(분)
HALF_MIN = WINDOW_MIN / 2.0      # 2분
SURGE_RET = 0.02                 # 4분 +2%
COOLDOWN_SEC = WINDOW_MIN * 60.0
HORIZONS_MIN = (5, 10, 20)       # 분 단위 지평
HORIZONS_SEC = tuple(int(n * 60) for n in HORIZONS_MIN)
COST_BP = (40.0, 50.0, 60.0)
REF_THRESHOLDS = (1.0,)
MIN_GROUP_N = 100
RNG_SEED = 20260906
N_BOOT = 1000
MAX_DAYS_PER_REGIME = 20
REGIMES = (HOT, MILD, COLD, DEAD)


def _ret_series(ts_sec: np.ndarray, price: np.ndarray, i: int, window_sec: float) -> float | None:
    t = ts_sec[i]
    ref_t = t - window_sec
    j = i
    while j >= 0 and ts_sec[j] > ref_t:
        j -= 1
    if j < 0:
        return None
    ref = float(price[j])
    if ref <= 0:
        return None
    return float(price[i]) / ref - 1.0


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


def classify_accel(ts_sec: np.ndarray, price: np.ndarray, i: int) -> tuple[str, float]:
    half_sec = HALF_MIN * 60.0
    recent = _ret_series(ts_sec, price, i, half_sec)
    t = ts_sec[i]
    ref_t = t - half_sec
    j = i
    while j >= 0 and ts_sec[j] > ref_t:
        j -= 1
    prior = _ret_series(ts_sec, price, j, half_sec) if j >= 0 else None
    if recent is None or prior is None or recent <= 0:
        return "reject", float("nan")
    if prior <= 0:
        return "always_pass", float("nan")
    return "normal", recent / prior


class Accum:
    def __init__(self) -> None:
        self.n_codes = 0
        self.n_triggers = 0
        self.n_reject = 0
        self.n_always_pass = 0
        self.normal_ratio_all: list[float] = []
        self.normal_ratio = {n: [] for n in HORIZONS_SEC}
        self.normal_close = {n: [] for n in HORIZONS_SEC}
        self.normal_mfe = {n: [] for n in HORIZONS_SEC}
        self.always_close = {n: [] for n in HORIZONS_SEC}
        self.always_mfe = {n: [] for n in HORIZONS_SEC}


def process_code(acc: Accum, ts_sec: np.ndarray, hms: np.ndarray, price: np.ndarray) -> None:
    acc.n_codes += 1
    triggers = find_surge_triggers(ts_sec, price, hms)
    if not triggers:
        return
    fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS_SEC}
    for i in triggers:
        acc.n_triggers += 1
        branch, ratio = classify_accel(ts_sec, price, i)
        if branch == "reject":
            acc.n_reject += 1
            continue
        if branch == "always_pass":
            acc.n_always_pass += 1
            close_bucket, mfe_bucket, ratio_bucket = acc.always_close, acc.always_mfe, None
        else:
            acc.normal_ratio_all.append(ratio)
            close_bucket, mfe_bucket, ratio_bucket = acc.normal_close, acc.normal_mfe, acc.normal_ratio
        for n in HORIZONS_SEC:
            mfe_i, close_i = fwd[n][0][i], fwd[n][1][i]
            if np.isfinite(close_i):
                close_bucket[n].append(close_i)
                mfe_bucket[n].append(mfe_i)
                if ratio_bucket is not None:
                    ratio_bucket[n].append(ratio)


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def candidate_thresholds(ratios: np.ndarray) -> list[tuple[str, float]]:
    if ratios.size == 0:
        return [(f"{t:.2f}(참조)", t) for t in REF_THRESHOLDS]
    out = []
    for label, pctl in (("P25", 25), ("P50", 50), ("P75", 75)):
        out.append((f"{label}={np.percentile(ratios, pctl):.3f}", float(np.percentile(ratios, pctl))))
    for t in REF_THRESHOLDS:
        out.append((f"{t:.2f}(참조)", t))
    return out


def build_section(acc: Accum, title: str) -> tuple[list[str], dict]:
    L = [f"## {title}", ""]
    L.append(f"- 종목 {acc.n_codes} · 서지 트리거 {acc.n_triggers:,}건")
    L.append(f"- reject {acc.n_reject:,}건 · always_pass {acc.n_always_pass:,}건")
    normal_ratios_all = _arr(acc.normal_ratio_all)
    L.append(f"- normal(ratio 분위 대상) {normal_ratios_all.size:,}건")
    L.append("")
    if acc.n_triggers == 0:
        L.append("_트리거 없음._")
        L.append("")
        return L, {"verdict_rows": [], "mono_ok": False, "n_triggers": 0}

    cands = candidate_thresholds(normal_ratios_all)
    L.append(
        "| τ | N(초) | pass n | fail n | pass close | fail close | 차 close(95%CI) | "
        "pass MFE | fail MFE | 차 MFE(95%CI) | net_close 50bp | 게이트 |"
    )
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    verdict_rows = []
    for label, tau in cands:
        for n in HORIZONS_SEC:
            ratio_n = _arr(acc.normal_ratio[n])
            close_n = _arr(acc.normal_close[n])
            mfe_n = _arr(acc.normal_mfe[n])
            normal_mask = ratio_n >= tau if ratio_n.size else np.array([], dtype=bool)
            pass_close = list(close_n[normal_mask]) if ratio_n.size else []
            pass_close += acc.always_close[n]
            pass_mfe = list(mfe_n[normal_mask]) if ratio_n.size else []
            pass_mfe += acc.always_mfe[n]
            fail_mask = ~normal_mask if ratio_n.size else np.array([], dtype=bool)
            fail_close = list(close_n[fail_mask]) if ratio_n.size else []
            fail_mfe = list(mfe_n[fail_mask]) if ratio_n.size else []

            p_c, f_c = _arr(pass_close), _arr(fail_close)
            p_m, f_m = _arr(pass_mfe), _arr(fail_mfe)
            gated = p_c.size >= MIN_GROUP_N and f_c.size >= MIN_GROUP_N
            gate_txt = "OK" if gated else f"표본부족(<{MIN_GROUP_N})"

            d, lo, hi = bootstrap_diff_ci(p_c, f_c, n_boot=N_BOOT, seed=RNG_SEED)
            dm, lom, him = bootstrap_diff_ci(p_m, f_m, n_boot=N_BOOT, seed=RNG_SEED)
            pc_mean = p_c.mean() if p_c.size else float("nan")
            fc_mean = f_c.mean() if f_c.size else float("nan")
            pm_mean = p_m.mean() if p_m.size else float("nan")
            fm_mean = f_m.mean() if f_m.size else float("nan")
            net = pc_mean * 10000 - 50.0 if p_c.size else float("nan")
            L.append(
                f"| {label} | {n} | {p_c.size} | {f_c.size} | {pc_mean*100:+.3f} | {fc_mean*100:+.3f} | "
                f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | {pm_mean*100:+.3f} | {fm_mean*100:+.3f} | "
                f"{dm*100:+.3f} [{lom*100:+.3f},{him*100:+.3f}] | {net:+.1f}bp | {gate_txt} |"
            )
            if gated:
                dir_ok = np.isfinite(d) and lo > 0 and np.isfinite(dm) and lom > 0
                net_ok = np.isfinite(net) and net >= 0
                verdict_rows.append((label, n, dir_ok, net_ok, d))
    L.append("")

    mono_ok = False
    ref_n = HORIZONS_SEC[1] if len(HORIZONS_SEC) > 1 else HORIZONS_SEC[0]
    ratio_ref = _arr(acc.normal_ratio[ref_n])
    close_ref = _arr(acc.normal_close[ref_n])
    if ratio_ref.size >= 5:
        qm = quintile_means(ratio_ref, close_ref)
        vals = [row[2] for row in qm]
        if len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]):
            mono_ok = vals[4] > vals[0]
        L.append(f"5분위(N={ref_n}s) close 평균(%): " + ", ".join(f"{v*100:+.3f}" for v in vals))
        L.append("")

    hit = [(label, n) for label, n, dir_ok, net_ok, _d in verdict_rows if dir_ok and net_ok and mono_ok]
    neg_dir = [d for _l, _n, _ok1, _ok2, d in verdict_rows if np.isfinite(d) and d < 0]
    if hit:
        L.append(f"**판정: 신호 있음(가설 방향).** {', '.join(f'{l}(N={n})' for l, n in hit)}")
    elif not verdict_rows:
        L.append("**판정: 표본 부족.**")
    elif len(neg_dir) > len(verdict_rows) / 2:
        L.append(f"**판정: 미확인 — 방향이 가설과 반대**(차이 음수 {len(neg_dir)}/{len(verdict_rows)}셀).")
    else:
        L.append("**판정: 미확인.**")
    L.append("")
    return L, {"verdict_rows": verdict_rows, "mono_ok": mono_ok, "n_triggers": acc.n_triggers}


def _select_dates(conn, all_dates: list) -> dict[str, list]:
    buckets: dict[str, list] = {r: [] for r in REGIMES}
    for d in all_dates:
        tj = fetch_market_turnover(conn, d)
        if tj is None:
            continue
        regime = classify_regime(tj)
        buckets[regime].append(d)
    selected: dict[str, list] = {}
    for regime, dates in buckets.items():
        dates = sorted(dates)
        if len(dates) <= MAX_DAYS_PER_REGIME:
            selected[regime] = dates
        else:
            idx = np.linspace(0, len(dates) - 1, MAX_DAYS_PER_REGIME).round().astype(int)
            selected[regime] = [dates[i] for i in sorted(set(idx.tolist()))]
    return selected


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="accel 필터 분봉 국면별 재검증 (읽기 전용)")
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
        print("[accel_sanity_minute] 선택된 날짜 수:", {k: len(v) for k, v in selected.items()})

        pooled = Accum()
        per_regime: dict[str, Accum] = {r: Accum() for r in REGIMES}
        for regime, dates in selected.items():
            for date in dates:
                per_code = _load_date_minute(conn, date)
                for code, (ts_sec, hms, price) in per_code.items():
                    process_code(per_regime[regime], ts_sec, hms, price)
                    process_code(pooled, ts_sec, hms, price)
    finally:
        conn.close()

    L = ["# 체결 가속도(accel) 필터 — 분봉 국면별 재검증", ""]
    L.append("> 사전등록 `docs/research/manju/57`. 56번(틱 10일)의 '미확인 — 방향 반대'가"
              " 저유동성 국면 탓인지 국면별로 재확인한다.")
    L.append("")
    L.append("선택 날짜 수: " + ", ".join(f"{k} {len(v)}일" for k, v in selected.items()))
    L.append("")

    section, meta = build_section(pooled, "전체 풀링")
    L += section
    for regime in REGIMES:
        section, meta = build_section(per_regime[regime], f"국면: {regime}")
        L += section

    report = "\n".join(L)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "accel_sanity_minute.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[accel_sanity_minute] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
