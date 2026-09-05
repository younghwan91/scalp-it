"""틱 스캘핑 1차 sanity check — 사전등록 `docs/research/manju/28`.

만쥬 가설: **체결강도 급증 + 매수 체결 우세** 순간에 진입하면 다음 수 분간 오른다.
`ticks` 하루치로 트리거군 vs 비트리거군의 forward N초 수익(MFE·종가)을 비교한다.

**확정 판정이 아니다** — 하루치 "신호 방향 존재 여부" sanity check. DB 는 읽기만 하고,
결과는 `data/tick_sanity_YYYYMMDD.md` + stdout 로만 낸다.

사용:
  export $(grep -v '^#' ../kr-quant/.env | xargs)
  uv run --extra pg python scripts/tick_sanity.py --date today
  uv run --extra pg python scripts/tick_sanity.py --date 2026-08-25

파라미터는 사전등록(28번)에서 고정: W∈{30,60}s, 강도 80분위, 매수비율≥0.60,
활동게이트 n≥5, 쿨다운=max(N)=120s, N∈{30,60,120}s, 왕복비용 40/50/60bp.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import os
import pathlib
import sys

import numpy as np
import pandas as pd

from scalp_it.intraday.store import _placeholder

# ---------------------------------------------------------------- 고정 파라미터
WINDOWS = (30, 60)              # 트리거 트레일링 윈도(초)
HORIZONS = (30, 60, 120)       # 라벨 지평(초)
STRENGTH_PCTL = 80.0           # (종목·당일) 트레일링 강도 분위 기준
BUY_RATIO_THR = 0.60           # 매수체결 비율 하한
MIN_COUNT = 5                  # 활동 게이트: 윈도 내 체결 건수 하한
COOLDOWN = max(HORIZONS)       # 트리거 후 같은 종목 억제(초) — 라벨 겹침 방지
COST_BP = (40.0, 50.0, 60.0)  # 왕복 비용 시나리오(bp)
SESSION_START = "09:00:00"     # 트리거 후보 시간 창
SESSION_END = "15:20:00"
BASELINE_PER_CODE = 300        # 종목별 비트리거 표본 상한
QUINTILE_CAP = 4000            # 종목별 분위 모집단 상한(메모리)
RNG_SEED = 20260824
N_BOOT = 1000


# ============================================================ 순수 계산 함수


def trailing_features(
    ts_sec: np.ndarray,
    side: np.ndarray,
    volume: np.ndarray,
    strength: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """각 틱 i 로 끝나는 직전 ``window`` 초의 (평균 체결강도, 매수비율, 건수).

    윈도 = ``ts_sec[i]-window < ts <= ts_sec[i]``. ``ts_sec`` 는 오름차순 가정.
    매수비율 = Σ(volume where side==+1) / Σ(volume). 거래량 0 이면 nan.
    """
    n = ts_sec.shape[0]
    mean_strength = np.full(n, np.nan)
    buy_ratio = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int64)
    buy_vol = np.where(side > 0, volume, 0).astype(np.float64)
    s_sum = 0.0
    v_sum = 0.0
    bv_sum = 0.0
    left = 0
    for i in range(n):
        s_sum += float(strength[i])
        v_sum += float(volume[i])
        bv_sum += float(buy_vol[i])
        lo = ts_sec[i] - window
        while ts_sec[left] <= lo:
            s_sum -= float(strength[left])
            v_sum -= float(volume[left])
            bv_sum -= float(buy_vol[left])
            left += 1
        c = i - left + 1
        count[i] = c
        if c > 0:
            mean_strength[i] = s_sum / c
        if v_sum > 0:
            buy_ratio[i] = bv_sum / v_sum
    return mean_strength, buy_ratio, count


def percentile_threshold(
    mean_strength: np.ndarray, count: np.ndarray, min_count: int, pctl: float
) -> float:
    """활동게이트를 통과한 틱들의 평균강도 분포에서 ``pctl`` 분위값. 없으면 +inf."""
    valid = mean_strength[(count >= min_count) & np.isfinite(mean_strength)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, pctl))


def find_trigger_indices(
    mean_strength: np.ndarray,
    buy_ratio: np.ndarray,
    count: np.ndarray,
    ts_sec: np.ndarray,
    *,
    strength_thr: float,
    buy_thr: float,
    min_count: int,
    cooldown: int,
) -> list[int]:
    """트리거 조건(강도≥thr & 매수비율≥thr & 건수≥게이트) + 쿨다운을 적용한 인덱스."""
    cond = (
        (count >= min_count)
        & np.isfinite(mean_strength)
        & (mean_strength >= strength_thr)
        & np.isfinite(buy_ratio)
        & (buy_ratio >= buy_thr)
    )
    out: list[int] = []
    last_ts = -np.inf
    for i in np.flatnonzero(cond):
        t = ts_sec[i]
        if t - last_ts < cooldown:
            continue
        out.append(int(i))
        last_ts = t
    return out


def forward_label(
    ts_sec: np.ndarray, price: np.ndarray, i: int, horizon: int
) -> tuple[float, float] | None:
    """트리거 i 이후 (t, t+horizon] 의 (MFE, 종가수익). 이후 체결이 없으면 None.

    MFE = max(price)/entry - 1, 종가수익 = last(price)/entry - 1.
    """
    entry = float(price[i])
    if entry <= 0:
        return None
    hi = ts_sec[i] + horizon
    j = i + 1
    n = ts_sec.shape[0]
    fmax = -np.inf
    flast = None
    while j < n and ts_sec[j] <= hi:
        p = float(price[j])
        fmax = max(fmax, p)
        flast = p
        j += 1
    if flast is None:
        return None
    return fmax / entry - 1.0, flast / entry - 1.0


def forward_labels_all(
    ts_sec: np.ndarray, price: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """모든 틱 i 에 대한 (MFE, 종가수익) 배열 — ``forward_label`` 의 벡터화판.

    구간 (t, t+horizon] 의 max·last 를 단조 두 포인터 + 슬라이딩 최대(deque)로 O(n).
    미래 체결이 없으면 nan. 대량 틱에서 O(n^2) 를 피하려고 이걸 쓴다.
    """
    from collections import deque

    n = ts_sec.shape[0]
    mfe = np.full(n, np.nan)
    close = np.full(n, np.nan)
    dq: deque[int] = deque()   # (i, r] 구간 인덱스, price 내림차순
    r = 0
    for i in range(n):
        hi = ts_sec[i] + horizon
        while r < n and ts_sec[r] <= hi:
            if r > i:
                while dq and price[dq[-1]] <= price[r]:
                    dq.pop()
                dq.append(r)
            r += 1
        while dq and dq[0] <= i:
            dq.popleft()
        entry = float(price[i])
        last_idx = r - 1
        if entry > 0 and last_idx > i:
            close[i] = price[last_idx] / entry - 1.0
            if dq:
                mfe[i] = price[dq[0]] / entry - 1.0
    return mfe, close


def bootstrap_diff_ci(
    treat: np.ndarray, base: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """평균차(treat-base) 와 부트스트랩 95% CI. 표본이 비면 nan."""
    if treat.size == 0 or base.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diff = float(treat.mean() - base.mean())
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        t = rng.choice(treat, treat.size, replace=True)
        c = rng.choice(base, base.size, replace=True)
        diffs[b] = t.mean() - c.mean()
    return diff, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def quintile_means(
    signal: np.ndarray, ret: np.ndarray
) -> list[tuple[int, float, float]]:
    """``signal`` 5분위별 (건수, 신호평균, 수익평균). 유한값만 사용."""
    m = np.isfinite(signal) & np.isfinite(ret)
    s, r = signal[m], ret[m]
    if s.size < 5:
        return []
    edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(s, edges)
    out = []
    for q in range(5):
        sel = bins == q
        if sel.any():
            out.append((int(sel.sum()), float(s[sel].mean()), float(r[sel].mean())))
        else:
            out.append((0, float("nan"), float("nan")))
    return out


# ============================================================ DB · 로딩


def _load_env(path: str | None) -> None:
    if os.environ.get("KR_QUANT_DB"):
        return
    for cand in (path, "../kr-quant/.env", "/home/young/Documents/git/kr-quant/.env"):
        if cand and pathlib.Path(cand).exists():
            for line in pathlib.Path(cand).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


def epoch_seconds(ts) -> np.ndarray:
    """pandas 시각열 → **초 단위** epoch int64.

    ``ts.astype("int64") // 1_000_000_000`` 는 틀린다 — pandas 2 는 datetime64 의
    단위를 보존하고, psycopg2 가 주는 ``datetime`` 은 ``datetime64[us]`` 로 잡혀
    그 나눗셈이 **1,000초 단위**를 준다. 2026-08-28 실측에서 3분치 997틱이 전부
    같은 값(1787907)으로 뭉쳐 트레일링 윈도도 지평(N=30/60/120)도 전부 같은
    결과를 냈다. 단위를 명시해 변환한다.
    """
    return ts.to_numpy(dtype="datetime64[s]").astype("int64")


def _codes_for_date(conn, date: str) -> list[str]:
    cur = conn.cursor()
    ph = _placeholder(conn)
    cur.execute(
        f"select distinct code from ticks where ts >= {ph} and ts < {ph} order by code",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes


def _rows_to_arrays(rows) -> tuple:
    """(ts, price, side, volume, strength) 행들을 시각순 numpy 배열로.

    (ts_sec, ts_str, price, side, volume, strength). ``_load_code_day`` 와
    ``load_date_ticks`` (종목별로 쪼갠 뒤) 양쪽에서 같은 변환을 쓴다 — 두 번
    따로 짜면 ``epoch_seconds`` 단위 버그 같은 걸 한쪽만 고치고 잊기 쉽다.
    """
    ts = pd.to_datetime([r[0] for r in rows])
    ts_sec = epoch_seconds(ts)
    hms = ts.strftime("%H:%M:%S").to_numpy()
    price = np.array([float(r[1]) if r[1] is not None else np.nan for r in rows])
    side = np.array([int(r[2]) if r[2] is not None else 0 for r in rows])
    volume = np.array([float(r[3]) if r[3] is not None else 0.0 for r in rows])
    strength = np.array([float(r[4]) if r[4] is not None else np.nan for r in rows])
    return ts_sec, hms, price, side, volume, strength


def _load_code_day(conn, code: str, date: str):
    """한 종목·하루 체결을 시각순 numpy 배열로. (ts_sec, ts_str, price, side, volume, strength)."""
    cur = conn.cursor()
    ph = _placeholder(conn)
    cur.execute(
        f"select ts, price, side, volume, strength from ticks "
        f"where code = {ph} and ts >= {ph} and ts < {ph} order by ts, seq",
        (code, f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None
    return _rows_to_arrays(rows)


def load_date_ticks(conn, date: str) -> dict[str, tuple]:
    """하루치 전 종목 체결을 **한 번의 쿼리**로 읽어 종목별로 쪼갠다.

    ``_codes_for_date`` + 종목별 ``_load_code_day`` 를 N일 풀링에 그대로 쓰면
    쿼리 수가 (일수 × 종목수)로 불어난다(예: 10일 × 1,500종목 ≈ 15,000회).
    `tick_sanity_confirm.py` 처럼 여러 날을 한 프로세스에서 도는 경우 이 함수로
    하루당 쿼리 1회로 줄인다. 하루 한 번만 도는 `tick_sanity.py` 자체 흐름은
    `_load_code_day` 그대로 둔다(바꿀 이유가 없다).
    """
    cur = conn.cursor()
    ph = _placeholder(conn)
    cur.execute(
        f"select code, ts, price, side, volume, strength from ticks "
        f"where ts >= {ph} and ts < {ph} order by code, ts, seq",
        (f"{date} 00:00:00", f"{date} 23:59:59"),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return {}
    out: dict[str, tuple] = {}
    for code, group in itertools.groupby(rows, key=lambda r: r[0]):
        out[str(code).strip()] = _rows_to_arrays([r[1:] for r in group])
    return out


# ============================================================ 집계


class Accum:
    """종목을 가로지르며 트리거·비트리거·분위·상위 궤적을 모은다."""

    def __init__(self) -> None:
        self.trig = {w: {n: [] for n in HORIZONS} for w in WINDOWS}
        self.trig_mfe = {w: {n: [] for n in HORIZONS} for w in WINDOWS}
        self.base = {w: {n: [] for n in HORIZONS} for w in WINDOWS}
        self.base_mfe = {w: {n: [] for n in HORIZONS} for w in WINDOWS}
        self.qsig_str = {w: [] for w in WINDOWS}   # 분위: 강도
        self.qsig_buy = {w: [] for w in WINDOWS}   # 분위: 매수비율
        self.qret = {w: {n: [] for n in HORIZONS} for w in WINDOWS}
        self.top: list[dict] = []                  # 상위 트리거 궤적(W=60 기준)
        self.n_codes = 0
        self.n_ticks = 0
        self.n_trig = {w: 0 for w in WINDOWS}
        self.rng = np.random.default_rng(RNG_SEED)


def _session_mask(hms: np.ndarray) -> np.ndarray:
    return (hms >= SESSION_START) & (hms <= SESSION_END)


def process_code(acc: Accum, data, code: str) -> None:
    ts_sec, hms, price, side, volume, strength = data
    acc.n_codes += 1
    acc.n_ticks += ts_sec.shape[0]
    in_sess = _session_mask(hms)
    # forward 라벨은 윈도 W 와 무관 — 지평별 한 번만 벡터화(O(n))
    fwd = {n: forward_labels_all(ts_sec, price, n) for n in HORIZONS}
    for w in WINDOWS:
        ms, br, cnt = trailing_features(ts_sec, side, volume, strength, w)
        thr = percentile_threshold(ms, cnt, MIN_COUNT, STRENGTH_PCTL)
        # 트리거는 장중 후보만
        eligible = in_sess & (cnt >= MIN_COUNT)
        trig_idx = find_trigger_indices(
            np.where(eligible, ms, np.nan), br, cnt, ts_sec,
            strength_thr=thr, buy_thr=BUY_RATIO_THR,
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
                _record_top(acc, code, ts_sec, hms, price, i, ms[i], br[i])
        # 비트리거 표본(활동게이트 통과·장중·트리거 아님)
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
        # 분위 모집단(장중·활동게이트, 상한 표본; 60s close 유효한 것만)
        allpool = np.flatnonzero(eligible & np.isfinite(fwd[60][1]))
        if allpool.size > QUINTILE_CAP:
            allpool = acc.rng.choice(allpool, QUINTILE_CAP, replace=False)
        for i in allpool:
            acc.qsig_str[w].append(float(ms[i]))
            acc.qsig_buy[w].append(float(br[i]))
            for n in HORIZONS:
                acc.qret[w][n].append(float(fwd[n][1][i]))


def _record_top(acc, code, ts_sec, hms, price, i, ms_i, br_i) -> None:
    entry = float(price[i])
    if entry <= 0:
        return
    hi = ts_sec[i] + 60
    traj = []
    for mark in (10, 20, 30, 40, 50, 60):
        tgt = ts_sec[i] + mark
        j = i
        last = entry
        while j < ts_sec.shape[0] and ts_sec[j] <= tgt:
            last = float(price[j])
            j += 1
        traj.append((last / entry - 1.0) * 100 if ts_sec[i] < tgt <= hi else 0.0)
    acc.top.append(
        {
            "code": code,
            "time": str(hms[i]),
            "strength": float(ms_i),
            "buy_ratio": float(br_i),
            "entry": entry,
            "traj": traj,
        }
    )


# ============================================================ 리포트


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def build_report(acc: Accum, date: str, *, pooled_days: int | None = None) -> str:
    """``acc`` 로 sanity 리포트를 만든다.

    ``pooled_days`` 를 주면 여러 거래일을 풀링한 확정 판정 문구로 낸다
    (`tick_sanity_confirm.py` 가 씀). ``None`` 이면 하루치 sanity 문구.
    """
    texts = (
        {
            "scope": f"**{pooled_days}거래일 풀링 확정 판정.** 사전등록 §0의 '며칠~몇 주 더 쌓은 뒤' 조건 충족.",
            "verdict_hit": "**신호 있음 → 채택 검토.** 방향(+CI)·비용후(net≥0)·단조성 모두 충족(풀링 확정).",
            "caveat": "풀링 확정 판정",
            "footer": (
                "_주의: 풀링 판정. 날짜 간 파라미터 재조정 없음(28번 고정값 그대로). "
                "일자별 개별 sanity 는 `data/tick_sanity_YYYYMMDD.md` 참고._"
            ),
        }
        if pooled_days
        else {
            "scope": "**하루치 sanity, 확정 아님.**",
            "verdict_hit": "**신호 있음(잠정) → 며칠 더 쌓아 확정.** 방향(+CI)·비용후(net≥0)·단조성 모두 충족.",
            "caveat": "하루치라 t·CI 참고, 방향·효과크기 위주",
            "footer": "_주의: 하루치 sanity. 다중검정·과최적화 경계. 초기일 데이터는 thrash 섞일 수 있음._",
        }
    )
    L = []
    L.append(f"# 틱 스캘핑 sanity — {date}")
    L.append("")
    L.append(f"> 사전등록 `docs/research/manju/28`. {texts['scope']} "
             f"W∈{list(WINDOWS)}s, 강도{int(STRENGTH_PCTL)}분위, 매수비율≥{BUY_RATIO_THR}, "
             f"n≥{MIN_COUNT}, 쿨다운{COOLDOWN}s, N∈{list(HORIZONS)}s, 비용{list(map(int,COST_BP))}bp.")
    L.append("")
    L.append(f"- 종목 {acc.n_codes} · 체결 {acc.n_ticks:,} 행")
    for w in WINDOWS:
        L.append(f"- W={w}s 트리거 {acc.n_trig[w]:,} 건")
    L.append("")
    # 1) 트리거 vs 비트리거
    L.append("## 1. 트리거군 vs 비트리거군 (forward N초, %)")
    L.append("")
    L.append("| W | N | 트리거n | 트리거 close 평균 | 비트리거 close 평균 | 차(부트 95%CI) | 트리거 MFE 평균 | net_close 50bp |")
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    verdict_net_ok = False
    verdict_dir_ok = False
    for w in WINDOWS:
        for n in HORIZONS:
            t = _arr(acc.trig[w][n])
            b = _arr(acc.base[w][n])
            tm = _arr(acc.trig_mfe[w][n])
            d, lo, hi = bootstrap_diff_ci(t, b)
            tc = t.mean() if t.size else float("nan")
            bc = b.mean() if b.size else float("nan")
            net = (tc * 10000 - 50.0) if t.size else float("nan")
            if np.isfinite(d) and lo > 0:
                verdict_dir_ok = True
            if np.isfinite(net) and net >= 0:
                verdict_net_ok = True
            L.append(
                f"| {w} | {n} | {t.size} | {tc*100:+.3f} | {bc*100:+.3f} | "
                f"{d*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] | {tm.mean()*100 if tm.size else float('nan'):+.3f} "
                f"| {net:+.1f}bp |"
            )
    L.append("")
    L.append("net_close 50bp = 트리거 close 평균 − 50bp. 다른 비용 시나리오: **40bp = 표값 +10bp, 60bp = 표값 −10bp**.")
    L.append("")
    # 2) 분위 단조성
    L.append("## 2. 신호 분위별 forward 수익 (W=30, close %, 5분위)")
    L.append("")
    mono_ok = False
    for label, sig in (("체결강도", acc.qsig_str[30]), ("매수비율", acc.qsig_buy[30])):
        L.append(f"### {label} 5분위 → close 60s")
        L.append("| 분위 | n | 신호평균 | ret_close_60 평균(%) |")
        L.append("|--:|--:|--:|--:|")
        qm = quintile_means(_arr(sig), _arr(acc.qret[30][60]))
        vals = []
        for q, row in enumerate(qm):
            cnt, smean, rmean = row
            vals.append(rmean)
            L.append(f"| Q{q+1} | {cnt} | {smean:.2f} | {rmean*100:+.3f} |")
        if len(vals) == 5 and np.isfinite(vals[0]) and np.isfinite(vals[4]) and vals[4] > vals[0]:
            mono_ok = True
        L.append("")
    # 3) 상위 20 눈으로
    L.append("## 3. 트리거 상위 20건 눈으로 (W=60, 강도순, 다음 60초 궤적 %)")
    L.append("")
    L.append("| 종목 | 시각 | 강도 | 매수비율 | 진입가 | +10s | +20s | +30s | +40s | +50s | +60s |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    top = sorted(acc.top, key=lambda x: x["strength"], reverse=True)[:20]
    for r in top:
        tj = " | ".join(f"{v:+.2f}" for v in r["traj"])
        L.append(
            f"| {r['code']} | {r['time']} | {r['strength']:.1f} | {r['buy_ratio']:.2f} "
            f"| {int(r['entry'])} | {tj} |"
        )
    L.append("")
    # 판정
    L.append("## 판정 (사전등록 4절)")
    L.append("")
    if verdict_dir_ok and verdict_net_ok and mono_ok:
        L.append(texts["verdict_hit"])
    else:
        miss = []
        if not verdict_dir_ok:
            miss.append("방향(부트 CI가 0 초과 없음)")
        if not verdict_net_ok:
            miss.append("비용후 net_close<0")
        if not mono_ok:
            miss.append("단조성 미충족")
        L.append(f"**신호 약함/미확인.** 미충족: {', '.join(miss)}. ({texts['caveat']})")
    L.append("")
    L.append(texts["footer"])
    return "\n".join(L)


# ============================================================ main


def _resolve_date(date: str) -> str:
    if date in ("today", "TODAY"):
        return dt.date.today().isoformat()  # noqa: DTZ011 — 로컬(KST) 당일
    return date


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="틱 스캘핑 1차 sanity check (읽기 전용)")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD 또는 today")
    ap.add_argument("--env", default=None, help=".env 경로(선택)")
    ap.add_argument("--out-dir", default="data", help="리포트 출력 폴더")
    args = ap.parse_args(argv)

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    date = _resolve_date(args.date)
    conn = connect()
    try:
        codes = _codes_for_date(conn, date)
        if not codes:
            print(f"[tick_sanity] {date}: ticks 없음. 종료.")
            return 0
        acc = Accum()
        for code in codes:
            data = _load_code_day(conn, code, date)
            if data is None:
                continue
            process_code(acc, data, code)
    finally:
        conn.close()

    report = build_report(acc, date)
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tick_sanity_{date.replace('-', '')}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[tick_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
