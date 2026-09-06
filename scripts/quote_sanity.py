"""틱 스캘핑 sanity — 호가(quotes) 조인 트리거 재정의.

`tick_sanity.py` 의 매수비율(체결 side 기반, **후행**)을 `quotes` 의 매수/매도
총잔량 임밸런스(호가 기반, **선행**)로 바꿔 트리거를 다시 정의한다. 강도(체결
활발함) 게이트는 그대로 두고, "매수 우세"만 체결이 아니라 각 틱 시각 직전의
최신 호가 스냅샷에서 읽는다 — 진짜 만쥬처럼 "매수벽이 두꺼워지는" 순간을
가격이 튀기 전에 잡으려는 시도.

세션 창은 기본 09:00~10:00 으로 좁힌다 — 만쥬 물량 대부분이 개장 직후 한
시간에 나온다는 가설(사용자 확인) 반영. `--session-end` 로 되돌릴 수 있다.

`ticks` · `quotes` 둘 다 읽기만 한다. 산출물은
`data/quote_sanity_<from>_<to>.md` + stdout.

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/quote_sanity.py --from 2026-08-24 --to 2026-09-04
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (  # noqa: E402
    COST_BP,
    HORIZONS,
    MIN_COUNT,
    N_BOOT,
    RNG_SEED,
    STRENGTH_PCTL,
    WINDOWS,
    Accum,
    _arr,
    _load_env,
    _placeholder,
    _record_top,
    bootstrap_diff_ci,
    epoch_seconds,
    find_trigger_indices,
    forward_labels_all,
    load_date_ticks,
    percentile_threshold,
    quintile_means,
    trailing_features,
)

IMBALANCE_THR = 0.60          # 호가 매수우위 하한 (총매수잔량 비율)
COOLDOWN = max(HORIZONS)
SESSION_START = "09:00:00"
SESSION_END = "10:00:00"      # 기본값 — 만쥬 물량 대부분이 개장 첫 시간에 나온다는 가설
BASELINE_PER_CODE = 300
QUINTILE_CAP = 4000


def load_date_quote_imbalance(conn, date: str) -> dict[str, tuple]:
    """하루치 전 종목 호가를 한 번의 쿼리로 읽어 종목별 (ts_sec, imbalance) 로 쪼갠다.

    imbalance = total_bid_qty / (total_bid_qty + total_ask_qty). 둘 다 0이면 nan.
    """
    cur = conn.cursor()
    ph = _placeholder(conn)
    cur.execute(
        f"select code, ts, total_bid_qty, total_ask_qty from quotes "
        f"where ts >= {ph} and ts < {ph} order by code, ts",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return {}
    out: dict[str, tuple] = {}
    for code, group in itertools.groupby(rows, key=lambda r: r[0]):
        g = list(group)
        ts_sec = epoch_seconds_from_list([r[1] for r in g])
        bid = np.array([float(r[2]) if r[2] is not None else 0.0 for r in g])
        ask = np.array([float(r[3]) if r[3] is not None else 0.0 for r in g])
        tot = bid + ask
        imb = np.where(tot > 0, bid / np.where(tot > 0, tot, 1.0), np.nan)
        out[str(code).strip()] = (ts_sec, imb)
    return out


def epoch_seconds_from_list(ts_list) -> np.ndarray:
    import pandas as pd

    return epoch_seconds(pd.to_datetime(ts_list))


def attach_quote_imbalance(
    tick_ts_sec: np.ndarray, quote_ts_sec: np.ndarray, quote_imb: np.ndarray
) -> np.ndarray:
    """각 체결 시각에 **그 시각 이전 최신** 호가 임밸런스를 붙인다(선행 정보 누출 없음)."""
    idx = np.searchsorted(quote_ts_sec, tick_ts_sec, side="right") - 1
    out = np.full(tick_ts_sec.shape[0], np.nan)
    valid = idx >= 0
    out[valid] = quote_imb[idx[valid]]
    return out


def _session_mask(hms: np.ndarray) -> np.ndarray:
    return (hms >= SESSION_START) & (hms <= SESSION_END)


def process_code_quote(acc: Accum, tick_data, quote_data, code: str) -> None:
    ts_sec, hms, price, side, volume, strength = tick_data
    q_ts_sec, q_imb = quote_data
    acc.n_codes += 1
    acc.n_ticks += ts_sec.shape[0]
    in_sess = _session_mask(hms)
    imbalance = attach_quote_imbalance(ts_sec, q_ts_sec, q_imb)
    fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS}
    for w in WINDOWS:
        ms, _br, cnt = trailing_features(ts_sec, side, volume, strength, w)
        thr = percentile_threshold(ms, cnt, MIN_COUNT, STRENGTH_PCTL)
        eligible = in_sess & (cnt >= MIN_COUNT)
        trig_idx = find_trigger_indices(
            np.where(eligible, ms, np.nan), imbalance, cnt, ts_sec,
            strength_thr=thr, buy_thr=IMBALANCE_THR,
            min_count=MIN_COUNT, cooldown=COOLDOWN,
        )
        trig_set = set(trig_idx)
        acc.n_trig[w] += len(trig_idx)
        for i in trig_idx:
            for n in HORIZONS:
                mfe_i, close_i = fwd[n][0][i], fwd[n][1][i]
                if np.isfinite(close_i):
                    acc.trig_mfe[w][n].append(mfe_i)
                    acc.trig[w][n].append(close_i)
            if w == 60:
                _record_top(acc, code, ts_sec, hms, price, i, ms[i], imbalance[i])
        pool = np.array(
            [i for i in np.flatnonzero(eligible) if i not in trig_set], dtype=np.int64
        )
        if pool.size:
            k = min(BASELINE_PER_CODE, pool.size)
            for i in acc.rng.choice(pool, k, replace=False):
                for n in HORIZONS:
                    close_i = fwd[n][1][i]
                    if np.isfinite(close_i):
                        acc.base_mfe[w][n].append(fwd[n][0][i])
                        acc.base[w][n].append(close_i)
        allpool = np.flatnonzero(eligible & np.isfinite(fwd[60][1]))
        if allpool.size > QUINTILE_CAP:
            allpool = acc.rng.choice(allpool, QUINTILE_CAP, replace=False)
        for i in allpool:
            acc.qsig_str[w].append(float(ms[i]))
            acc.qsig_buy[w].append(float(imbalance[i]) if np.isfinite(imbalance[i]) else float("nan"))
            for n in HORIZONS:
                acc.qret[w][n].append(float(fwd[n][1][i]))


def build_report(acc: Accum, label: str, pooled_days: int) -> str:
    L = []
    L.append(f"# 틱 스캘핑 sanity — 호가 조인 트리거 — {label}")
    L.append("")
    L.append(
        f"> `tick_sanity.py` 변형: 매수비율(체결,후행) → 호가 매수우위(quotes,선행). "
        f"**{pooled_days}거래일 풀링.** 세션 {SESSION_START}~{SESSION_END}. "
        f"W∈{list(WINDOWS)}s, 강도{int(STRENGTH_PCTL)}분위, 호가매수우위≥{IMBALANCE_THR}, "
        f"n≥{MIN_COUNT}, 쿨다운{COOLDOWN}s, N∈{list(HORIZONS)}s, 비용{list(map(int, COST_BP))}bp."
    )
    L.append("")
    L.append(f"- 종목 {acc.n_codes} · 체결 {acc.n_ticks:,} 행")
    for w in WINDOWS:
        L.append(f"- W={w}s 트리거 {acc.n_trig[w]:,} 건")
    L.append("")
    L.append("## 1. 트리거군 vs 비트리거군 (forward N초, %)")
    L.append("")
    L.append(
        "| W | N | 트리거n | 트리거 close 평균/중앙 | 비트리거 close 평균/중앙 | "
        "차 close(부트 95%CI) | 트리거 MFE 평균/중앙 | 비트리거 MFE 평균/중앙 | "
        "차 MFE(부트 95%CI) | net_close 50bp |"
    )
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    verdict_net_ok = False
    verdict_dir_ok = False
    for w in WINDOWS:
        for n in HORIZONS:
            t = _arr(acc.trig[w][n])
            b = _arr(acc.base[w][n])
            tm = _arr(acc.trig_mfe[w][n])
            bm = _arr(acc.base_mfe[w][n])
            d, lo, hi = bootstrap_diff_ci(t, b)
            dm, lom, him = bootstrap_diff_ci(tm, bm)
            tc = t.mean() if t.size else float("nan")
            bc = b.mean() if b.size else float("nan")
            tc_med = np.median(t) if t.size else float("nan")
            bc_med = np.median(b) if b.size else float("nan")
            tm_mean = tm.mean() if tm.size else float("nan")
            tm_med = np.median(tm) if tm.size else float("nan")
            bm_mean = bm.mean() if bm.size else float("nan")
            bm_med = np.median(bm) if bm.size else float("nan")
            net = (tc * 10000 - 50.0) if t.size else float("nan")
            close_dir_ok = np.isfinite(d) and lo > 0
            mfe_dir_ok = np.isfinite(dm) and lom > 0
            if close_dir_ok and mfe_dir_ok:
                verdict_dir_ok = True
            if np.isfinite(net) and net >= 0:
                verdict_net_ok = True
            L.append(
                f"| {w} | {n} | {t.size} | {tc*100:+.3f}/{tc_med*100:+.3f} | {bc*100:+.3f}/{bc_med*100:+.3f} | "
                f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | {tm_mean*100:+.3f}/{tm_med*100:+.3f} | "
                f"{bm_mean*100:+.3f}/{bm_med*100:+.3f} | {dm*100:+.3f} [{lom*100:+.3f},{him*100:+.3f}] "
                f"| {net:+.1f}bp |"
            )
    L.append("")
    L.append("net_close 50bp = 트리거 close 평균 − 50bp. 40bp = 표값 +10bp, 60bp = 표값 −10bp.")
    L.append("방향 판정은 close·MFE 둘 다의 부트 CI 하한이 0 초과인 (W,N)이 있어야 충족.")
    L.append("")
    L.append("## 2. 신호 분위별 forward 수익 (close %, 5분위)")
    L.append("")
    mono_ok = True
    for w in WINDOWS:
        for label2, sig in ((f"체결강도(W={w})", acc.qsig_str[w]), (f"호가매수우위(W={w})", acc.qsig_buy[w])):
            L.append(f"### {label2} 5분위 → close 60s")
            L.append("| 분위 | n | 신호평균 | ret_close_60 평균(%) |")
            L.append("|--:|--:|--:|--:|")
            qm = quintile_means(_arr(sig), _arr(acc.qret[w][60]))
            vals = []
            for q, row in enumerate(qm):
                cnt, smean, rmean = row
                vals.append(rmean)
                L.append(f"| Q{q+1} | {cnt} | {smean:.2f} | {rmean*100:+.3f} |")
            if not (len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]) and vals[4] > vals[0]):
                mono_ok = False
            L.append("")
    L.append("## 3. 트리거 상위 20건 눈으로 (W=60, 강도순, 다음 60초 궤적 %)")
    L.append("")
    L.append("| 종목 | 시각 | 강도 | 호가매수우위 | 진입가 | +10s | +20s | +30s | +40s | +50s | +60s |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    top = sorted(acc.top, key=lambda x: x["strength"], reverse=True)[:20]
    for r in top:
        tj = " | ".join(f"{v:+.2f}" for v in r["traj"])
        L.append(
            f"| {r['code']} | {r['time']} | {r['strength']:.1f} | {r['buy_ratio']:.2f} "
            f"| {int(r['entry'])} | {tj} |"
        )
    L.append("")
    L.append("## 판정")
    L.append("")
    if verdict_dir_ok and verdict_net_ok and mono_ok:
        L.append("**신호 있음 → 채택 검토.** 방향(+CI)·비용후(net≥0)·단조성 모두 충족(풀링 확정).")
    else:
        miss = []
        if not verdict_dir_ok:
            miss.append("방향(부트 CI가 0 초과 없음)")
        if not verdict_net_ok:
            miss.append("비용후 net_close<0")
        if not mono_ok:
            miss.append("단조성 미충족")
        L.append(f"**신호 약함/미확인.** 미충족: {', '.join(miss)}. (풀링 확정 판정)")
    L.append("")
    L.append(
        "_주의: 탐색적 재정의(사전등록 아님). 세션·임계값(0.60)은 이번 실험용으로 임의 선택 — "
        "채택 판단 전 별도 사전등록 필요._"
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
    global SESSION_START, SESSION_END
    ap = argparse.ArgumentParser(description="틱+호가 조인 sanity (읽기 전용)")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--session-start", default=SESSION_START)
    ap.add_argument("--session-end", default=SESSION_END)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    SESSION_START = args.session_start
    SESSION_END = args.session_end

    dates = _dates_from_range(args.date_from, args.date_to)
    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    acc = Accum()
    used_dates: list[str] = []
    try:
        for date in dates:
            tick_data = load_date_ticks(conn, date)
            if not tick_data:
                print(f"[quote_sanity] {date}: ticks 없음. 스킵.")
                continue
            quote_data = load_date_quote_imbalance(conn, date)
            if not quote_data:
                print(f"[quote_sanity] {date}: quotes 없음. 스킵.")
                continue
            used_dates.append(date)
            for code, tdata in tick_data.items():
                qdata = quote_data.get(code)
                if qdata is None:
                    continue
                process_code_quote(acc, tdata, qdata, code)
    finally:
        conn.close()

    if not used_dates:
        print("[quote_sanity] 유효한 날짜 없음. 종료.")
        return 0

    label = f"{used_dates[0]} ~ {used_dates[-1]} ({len(used_dates)}일 풀링)"
    report = build_report(acc, label, len(used_dates))
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{used_dates[0].replace('-', '')}_{used_dates[-1].replace('-', '')}"
    out = out_dir / f"quote_sanity_{tag}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[quote_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
