"""체결 가속도(accel) 필터 문턱 sanity — 사전등록 `docs/research/manju/56`.

`PairDetector.accel_min_ratio`(`src/scalp_it/realtime/pair_detector.py`)는
배선돼 있으나 문턱값이 없다(타 트레이더 B B-1, (비공개 — 공개본 미포함) "아직 안 붙인 것"). 대장 서지
트리거(30초 +1.5%) 순간의 accel_ratio(뒤 15초/앞 15초 수익률) 분포를 재고,
그 분위를 후보 문턱으로 삼아 pass/fail 두 군의 forward 수익을 비교한다.

**대장 자신의 forward 수익으로 잰다 — 후속주(2등주) 페어링은 범위 밖이다**
(사전등록 §1). 10거래일 전부 저유동성 국면((비공개 — 공개본 미포함))이라 확정 배선 판단에는
쓰지 않는다(사전등록 §0·§6).

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/accel_sanity.py --from 2026-08-24 --to 2026-09-04
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    HORIZONS,
    _load_env,
    _session_mask,
    bootstrap_diff_ci,
    forward_labels_all,
    load_date_ticks,
    quintile_means,
)

# ---------------------------------------------------------------- 고정 파라미터 (56번 사전등록)
SURGE_WINDOW = 30.0            # PairConfig 기본값 그대로
SURGE_RET = 0.015              # 〃
HALF = SURGE_WINDOW / 2.0      # 15초
COOLDOWN = SURGE_WINDOW        # 트리거 후 같은 종목 억제(초)
# 고정 가정값(값은 바꾸지 않는다). 일자별 거래세 스케줄은 krx_quant_core.costs 참고.
COST_BP = (40.0, 50.0, 60.0)
REF_THRESHOLDS = (1.0,)        # 분위 후보에 항상 추가하는 참조값
MIN_GROUP_N = 100              # pass/fail 최소표본 게이트 (N=60 라벨)
RNG_SEED = 20260906
N_BOOT = 1000


def _ret_series(ts_sec: np.ndarray, price: np.ndarray, i: int, window: float) -> float | None:
    """``pair_detector._ret`` 근사 — [t-window, t] 구간 수익률. 과거 체결이 없으면 None."""
    t = ts_sec[i]
    ref_t = t - window
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
    """서지 트리거(30초 +1.5%↑) 인덱스, 세션 내·쿨다운 적용."""
    in_sess = _session_mask(hms)
    out: list[int] = []
    last_ts = -np.inf
    for i in np.flatnonzero(in_sess):
        t = ts_sec[i]
        if t - last_ts < COOLDOWN:
            continue
        r = _ret_series(ts_sec, price, i, SURGE_WINDOW)
        if r is not None and r >= SURGE_RET:
            out.append(int(i))
            last_ts = t
    return out


def classify_accel(ts_sec: np.ndarray, price: np.ndarray, i: int) -> tuple[str, float]:
    """(갈래, ratio). 갈래 ∈ {"reject", "always_pass", "normal"}. normal 만 ratio 유효."""
    recent = _ret_series(ts_sec, price, i, HALF)
    prior_end_idx = i
    # prior 구간은 [t-2*HALF, t-HALF] — recent 와 같은 함수를 t-HALF 시점 기준으로 재호출
    t = ts_sec[i]
    ref_t = t - HALF
    j = prior_end_idx
    while j >= 0 and ts_sec[j] > ref_t:
        j -= 1
    prior = _ret_series(ts_sec, price, j, HALF) if j >= 0 else None
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
        # normal 갈래: 분위·문턱 비교에 ratio 와 라벨을 지평별로 **정렬 유지**해서 담는다
        # (forward_label 이 없는(미래 체결 없음) 트리거는 지평마다 다르게 빠질 수 있어
        # 공유 리스트 하나로 모으면 인덱스가 어긋난다)
        self.normal_ratio_all: list[float] = []  # 분포(§0)용 — forward 유효 여부 무관
        self.normal_ratio = {n: [] for n in HORIZONS}
        self.normal_close = {n: [] for n in HORIZONS}
        self.normal_mfe = {n: [] for n in HORIZONS}
        self.always_close = {n: [] for n in HORIZONS}
        self.always_mfe = {n: [] for n in HORIZONS}


def process_code(acc: Accum, data, code: str) -> None:
    ts_sec, hms, price, side, volume, strength = data
    acc.n_codes += 1
    triggers = find_surge_triggers(ts_sec, price, hms)
    if not triggers:
        return
    fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS}
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
        for n in HORIZONS:
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


def build_report(acc: Accum, date_from: str, date_to: str) -> str:
    L = []
    L.append(f"# 체결 가속도(accel) 필터 sanity — {date_from} ~ {date_to}")
    L.append("")
    L.append("> 사전등록 `docs/research/manju/56`. 대장 자신의 forward 수익으로 잰다"
              "(후속주 페어링은 범위 밖). 10거래일 전부 저유동성 국면((비공개 — 공개본 미포함)) — 확정 배선"
              " 판단에는 쓰지 않는다.")
    L.append("")
    L.append(f"- 종목 {acc.n_codes} · 서지 트리거 {acc.n_triggers:,}건")
    L.append(f"- reject(뒤 절반 상승 아님, 문턱 무관 항상 기각) {acc.n_reject:,}건 "
              f"({acc.n_reject / max(acc.n_triggers, 1) * 100:.1f}%)")
    L.append(f"- always_pass(앞 잠잠·뒤만 튐, 문턱 무관 항상 통과) {acc.n_always_pass:,}건 "
              f"({acc.n_always_pass / max(acc.n_triggers, 1) * 100:.1f}%)")
    normal_ratios_all = _arr(acc.normal_ratio_all)
    L.append(f"- normal(둘 다 양수, ratio 분위 대상) {normal_ratios_all.size:,}건")
    L.append("")

    if normal_ratios_all.size:
        L.append("## 0. ratio 분포 (normal 갈래, forward 유효 여부 무관)")
        L.append("")
        pcts = [5, 25, 50, 75, 95]
        vals = np.percentile(normal_ratios_all, pcts)
        L.append("| 분위 | " + " | ".join(f"P{p}" for p in pcts) + " |")
        L.append("|---|" + "---|" * len(pcts))
        L.append("| ratio | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        L.append("")

    cands = candidate_thresholds(normal_ratios_all)
    L.append("## 1. 후보 문턱별 pass vs fail (forward, %)")
    L.append("")
    L.append(
        "| τ | N | pass n | fail n | pass close 평균 | fail close 평균 | "
        "차 close(부트95%CI) | pass MFE 평균 | fail MFE 평균 | 차 MFE(부트95%CI) | net_close 50bp | 게이트 |"
    )
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")

    verdict_rows = []
    for label, tau in cands:
        for n in HORIZONS:
            ratio_n = _arr(acc.normal_ratio[n])   # normal_close[n] 와 같은 순서로 쌓인 ratio
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
                verdict_rows.append((label, n, dir_ok, net_ok))
    L.append("")
    L.append("net_close 50bp = pass close 평균 − 50bp. 40bp = 표값 +10bp, 60bp = 표값 −10bp.")
    L.append("")

    L.append("## 2. normal 갈래 ratio 5분위 → close 60s")
    L.append("")
    mono_ok = False
    ratio_60 = _arr(acc.normal_ratio[60])
    close_60 = _arr(acc.normal_close[60])
    if ratio_60.size >= 5:
        qm = quintile_means(ratio_60, close_60)
        L.append("| 분위 | n | ratio평균 | ret_close_60 평균(%) |")
        L.append("|--:|--:|--:|--:|")
        vals = []
        for q, row in enumerate(qm):
            cnt, smean, rmean = row
            vals.append(rmean)
            L.append(f"| Q{q+1} | {cnt} | {smean:.3f} | {rmean*100:+.3f} |")
        if len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]):
            mono_ok = vals[4] > vals[0]
    else:
        L.append("_normal 표본 부족(<5) — 분위 생략._")
    L.append("")

    L.append("## 판정 (사전등록 §5)")
    L.append("")
    hit = [(label, n) for label, n, dir_ok, net_ok in verdict_rows if dir_ok and net_ok and mono_ok]
    if hit:
        best = ", ".join(f"{label}(N={n})" for label, n in hit)
        L.append(f"**신호 있음 → 문턱 후보로 채택 검토.** 방향·비용후·단조성 모두 충족: {best}.")
    elif not verdict_rows:
        L.append("**표본 부족 — 판정 불가.** 모든 후보 τ 에서 pass·fail 군 중 하나가 "
                  f"최소표본({MIN_GROUP_N}) 미달.")
    else:
        L.append("**미확인.** 판정에 들어간 후보 중 방향(부트CI 0 초과)·비용후(net≥0)·단조성"
                  "(Q5>Q1) 을 모두 만족하는 (τ,N) 이 없다. §0 캐베어트(저유동성 국면) 때문에"
                  " '신호 없음'이 아니라 '이 10일 데이터로는 불충분'으로 읽는다.")
    L.append("")
    L.append("_주의: 10거래일 전부 저유동성 국면((비공개 — 공개본 미포함)). 대장 자신의 forward 수익 대리 측정"
              "(후속주 페어링 범위 밖). 활황 장세 재검증 전 실배선 금지(56번 §6)._")
    return "\n".join(L)


def _resolve_range(args) -> list[str]:
    if args.date:
        return [args.date]
    d0 = dt.date.fromisoformat(args.date_from)
    d1 = dt.date.fromisoformat(args.date_to)
    out = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="accel 필터 문턱 sanity (읽기 전용)")
    ap.add_argument("--date", default=None, help="단일 날짜 YYYY-MM-DD")
    ap.add_argument("--from", dest="date_from", default="2026-08-24")
    ap.add_argument("--to", dest="date_to", default="2026-09-04")
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    dates = _resolve_range(args)
    acc = Accum()
    conn = connect()
    try:
        for date in dates:
            per_code = load_date_ticks(conn, date)
            for code, data in per_code.items():
                process_code(acc, data, code)
    finally:
        conn.close()

    if acc.n_codes == 0:
        print("[accel_sanity] ticks 없음. 종료.")
        return 0

    report = build_report(acc, dates[0], dates[-1])
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"accel_sanity_{dates[0].replace('-', '')}_{dates[-1].replace('-', '')}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[accel_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
