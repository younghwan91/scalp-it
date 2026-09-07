"""틱 스캘핑 sanity — 트레일링 스탑 청산으로 라벨 재정의.

`tick_sanity.py`/`tick_sanity_confirm.py` 는 고정 지평 N초 뒤 **종가**로 무조건
청산하는 라벨을 쓴다. 10일 풀링 결과 MFE(최고 유리 이동)는 트리거군이 크게
나오는데(또는 최소한 존재하는데) close 는 그걸 못 살리는 패턴이 반복됐다 —
"진입 타이밍은 맞는데 고정 시각 청산이 이익을 깎아먹는다"는 가설을 여기서
검증한다.

청산 규칙을 "진입 후 최고가 대비 `TRAIL_PCT` 하락하면 즉시 청산(트레일링
스탑), 그 전에 `MAX_HOLD`초가 지나면 그 시점 마지막 체결가로 청산"으로 바꾼다.
신호(트리거) 정의·파라미터(W,P,B,K,쿨다운)는 `docs/research/manju/28` 그대로
가져온다 — 바뀐 건 청산 규칙뿐.

**탐색적 실험이다 — 사전등록 아님.** TRAIL_PCT·MAX_HOLD 는 이번 실험용으로
임의 선택했다. 여기서 신호가 나와도 "채택"이 아니라 "청산 방식을 사전등록해서
다시 확정 검증 필요"로 읽어야 한다.

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/trail_stop_sanity.py --from 2026-08-24 --to 2026-09-04
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    BUY_RATIO_THR,
    MIN_COUNT,
    STRENGTH_PCTL,
    WINDOWS,
    _arr,
    _load_env,
    _session_mask,
    bootstrap_diff_ci,
    find_trigger_indices,
    load_date_ticks,
    percentile_threshold,
    trailing_features,
)

COOLDOWN = 120
TRAIL_PCT = 0.0040     # 진입 후 최고가 대비 40bp 하락하면 청산 (15bp는 틱노이즈에 거의 다 털림 — 08-24~09-04 실측)
MAX_HOLD = 300         # 최대 보유 300초(트레일링 스탑에 안 걸리면 이 시점 청산)
COST_BP = 50.0
BASELINE_PER_CODE = 300
RNG_SEED = 20260824


def trailing_stop_exit(
    ts_sec: np.ndarray, price: np.ndarray, i: int, *, trail_pct: float, max_hold: int
) -> tuple[float, float] | None:
    """진입 i 이후: 최고가 대비 trail_pct 하락 시 청산, 아니면 max_hold초 뒤 청산.

    반환 (청산수익률, 보유초). 이후 체결이 전혀 없으면 None.
    """
    entry = float(price[i])
    if entry <= 0:
        return None
    peak = entry
    deadline = ts_sec[i] + max_hold
    n = ts_sec.shape[0]
    j = i + 1
    last = None
    while j < n and ts_sec[j] <= deadline:
        p = float(price[j])
        if p > peak:
            peak = p
        if p <= peak * (1.0 - trail_pct):
            return p / entry - 1.0, float(ts_sec[j] - ts_sec[i])
        last = (p, ts_sec[j])
        j += 1
    if last is None:
        return None
    p, t = last
    return p / entry - 1.0, float(t - ts_sec[i])


class Acc:
    def __init__(self) -> None:
        self.trig_ret: dict[int, list] = {w: [] for w in WINDOWS}
        self.trig_hold: dict[int, list] = {w: [] for w in WINDOWS}
        self.base_ret: dict[int, list] = {w: [] for w in WINDOWS}
        self.n_trig = {w: 0 for w in WINDOWS}
        self.n_codes = 0
        self.n_ticks = 0
        self.rng = np.random.default_rng(RNG_SEED)


def process_code(acc: Acc, data, code: str) -> None:
    ts_sec, hms, price, side, volume, strength = data
    acc.n_codes += 1
    acc.n_ticks += ts_sec.shape[0]
    in_sess = _session_mask(hms)
    for w in WINDOWS:
        ms, br, cnt = trailing_features(ts_sec, side, volume, strength, w)
        thr = percentile_threshold(ms, cnt, MIN_COUNT, STRENGTH_PCTL)
        eligible = in_sess & (cnt >= MIN_COUNT)
        trig_idx = find_trigger_indices(
            np.where(eligible, ms, np.nan), br, cnt, ts_sec,
            strength_thr=thr, buy_thr=BUY_RATIO_THR,
            min_count=MIN_COUNT, cooldown=COOLDOWN,
        )
        trig_set = set(trig_idx)
        acc.n_trig[w] += len(trig_idx)
        for i in trig_idx:
            lab = trailing_stop_exit(ts_sec, price, i, trail_pct=TRAIL_PCT, max_hold=MAX_HOLD)
            if lab is not None:
                acc.trig_ret[w].append(lab[0])
                acc.trig_hold[w].append(lab[1])
        pool = np.array(
            [i for i in np.flatnonzero(eligible) if i not in trig_set], dtype=np.int64
        )
        if pool.size:
            k = min(BASELINE_PER_CODE, pool.size)
            for i in acc.rng.choice(pool, k, replace=False):
                lab = trailing_stop_exit(ts_sec, price, i, trail_pct=TRAIL_PCT, max_hold=MAX_HOLD)
                if lab is not None:
                    acc.base_ret[w].append(lab[0])


def build_report(acc: Acc, label: str) -> str:
    L = []
    L.append(f"# 틱 스캘핑 sanity — 트레일링 스탑 청산 — {label}")
    L.append("")
    L.append(
        f"> `tick_sanity.py` 변형: N초 후 종가 청산 → 최고가 대비 {TRAIL_PCT*10000:.0f}bp 하락 시 "
        f"트레일링 스탑 청산(최대 보유 {MAX_HOLD}s). 신호·게이트·쿨다운은 사전등록 28번 그대로. "
        f"**탐색적 실험 — 사전등록 아님.**"
    )
    L.append("")
    L.append(f"- 종목 {acc.n_codes} · 체결 {acc.n_ticks:,} 행")
    L.append("")
    L.append("| W | 트리거n | 트리거 청산수익 평균/중앙 | 비트리거 청산수익 평균/중앙 | 차(부트 95%CI) | 평균 보유(초) | net 50bp |")
    L.append("|--:|--:|--:|--:|--:|--:|--:|")
    dir_ok = False
    net_ok = False
    for w in WINDOWS:
        t = _arr(acc.trig_ret[w])
        b = _arr(acc.base_ret[w])
        hold = _arr(acc.trig_hold[w])
        d, lo, hi = bootstrap_diff_ci(t, b)
        tc = t.mean() if t.size else float("nan")
        bc = b.mean() if b.size else float("nan")
        tc_med = np.median(t) if t.size else float("nan")
        bc_med = np.median(b) if b.size else float("nan")
        net = (tc * 10000 - COST_BP) if t.size else float("nan")
        if np.isfinite(d) and lo > 0:
            dir_ok = True
        if np.isfinite(net) and net >= 0:
            net_ok = True
        L.append(
            f"| {w} | {t.size} | {tc*100:+.3f}/{tc_med*100:+.3f} | {bc*100:+.3f}/{bc_med*100:+.3f} | "
            f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | {hold.mean() if hold.size else float('nan'):.1f} "
            f"| {net:+.1f}bp |"
        )
    L.append("")
    L.append("## 판정")
    L.append("")
    if dir_ok and net_ok:
        L.append("**신호 있음(트레일링 청산 기준) → 청산 방식을 사전등록해 재검증 필요.** 방향(+CI)·비용후(net≥0) 충족.")
    else:
        miss = []
        if not dir_ok:
            miss.append("방향(부트 CI가 0 초과 없음)")
        if not net_ok:
            miss.append("비용후 net<0")
        L.append(f"**신호 약함/미확인.** 미충족: {', '.join(miss)}.")
    L.append("")
    L.append(
        f"_주의: TRAIL_PCT={TRAIL_PCT*10000:.0f}bp, MAX_HOLD={MAX_HOLD}s 는 이번 실험용 임의값. "
        "여기서 신호가 나와도 이 값 자체를 사전등록해 별도 확정 필요._"
    )
    return "\n".join(L)


def _dates_from_range(start: str, end: str) -> list[str]:
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    out = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="틱 트레일링 스탑 청산 sanity (읽기 전용)")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    dates = _dates_from_range(args.date_from, args.date_to)
    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    acc = Acc()
    used_dates: list[str] = []
    try:
        for date in dates:
            tick_data = load_date_ticks(conn, date)
            if not tick_data:
                print(f"[trail_stop_sanity] {date}: ticks 없음. 스킵.")
                continue
            used_dates.append(date)
            for code, tdata in tick_data.items():
                process_code(acc, tdata, code)
    finally:
        conn.close()

    if not used_dates:
        print("[trail_stop_sanity] 유효한 날짜 없음. 종료.")
        return 0

    label = f"{used_dates[0]} ~ {used_dates[-1]} ({len(used_dates)}일 풀링)"
    report = build_report(acc, label)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{used_dates[0].replace('-', '')}_{used_dates[-1].replace('-', '')}"
    out = out_dir / f"trail_stop_sanity_{tag}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[trail_stop_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
