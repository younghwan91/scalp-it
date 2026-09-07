"""상한가 도달 시각 진단 — 종가베팅이 고른 상한가 마감 종목을 분봉으로 들여다본다.

`docs/research/manju/06-closebet-preregistration.md` "남은 질문 — 상한가를 정말 못 사는가"
에 답하기 위한 **진단**이다. 전략 채택·파라미터 탐색이 아니다.

측정하는 것:
1. 종가베팅 규칙(유니버스 50억 · tv_ratio>=3 · close_pos>=0.8 · 일 상위 2건)이 고른
   (종목, 일자) 중 당일 상한가 마감 건에 대해 분봉에서 **처음 상한가에 닿은 시각**,
   **상한가가 풀린 적이 있는지**, **도달 이후 거래량·거래대금**.
2. 도달 시각 구간별 건수·비율.
3. 도달 시각 구간별 익일 시가 수익률 (일봉, 왕복 34bp 차감).
4. 15:00 이후 도달분의 도달 이후 거래대금 분포.

상한가 정의 (두 단계):
- 1차(일봉, 근사): ``close / prev_close >= 1.295``. 일봉은 수정주가라 호가단위 내림을
  정확히 못 한다.
- 2차(분봉, 정확): 전 거래일 마지막 분봉 종가(원주가) × 1.3 을 **호가단위로 내림**한
  값을 상한가로 삼고, 당일 15:30 분봉 종가가 정확히 그 값이면 상한가 마감.
  일봉 판정과 분봉 판정이 어긋나거나 수정주가/원주가 비율이 두 날 사이에 달라지면
  (배당락·액면분할 등) 제외하고 건수를 보고한다.

분봉 구조: 09:00~15:19 접속매매 1분봉, 15:20~15:29 없음, 15:30 단일가(종가) 봉.
따라서 "동시호가에 처음 도달" = 15:19 까지 고가가 상한가 미만이었는데 15:30 봉에서
상한가 체결.

DB 는 읽기만 한다. 실행:

    uv run python scripts/limitup_timing.py --start 2025-08-01 --end 2026-08-21 \\
        --out /path/to/limitup_timing.md
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from scalp_it.adapters.db import load_ohlcv
from scalp_it.cli_cbet_watch import (
    ADV_FLOOR,
    ADV_WINDOW,
    CLOSE_POS_MIN,
    COST_ROUND_TRIP,
    TOP_N,
    TV_RATIO_MIN,
)
from scalp_it.closebet import close_bet_features, close_bet_labels, select_close_bet
from scalp_it.intraday.loader import available_codes, load_minute_bars
from scalp_it.intraday.store import connect
from scalp_it.ticksize import tick_size  # 정의는 한 곳에만 — 값은 그대로다

#: 일봉 근사 상한가 판정 문턱 (수정주가라 정확한 호가단위 내림이 불가능).
DAILY_LIMIT_RATIO = 1.295
#: 수정주가/원주가 비율이 전일·당일 사이에 이보다 벌어지면 배당락·분할 등으로 보고 제외.
ADJ_DRIFT_TOL = 0.002
#: 15:30 이후 봉(15:35 시간외 등)은 보지 않는다.
_CLOSE_MINUTE = 15 * 60 + 30
_CUTOFF_MINUTE = 15 * 60 + 19

#: 도달 시각 구간. (라벨, 시작분 포함, 끝분 미포함). 15:30 은 단일가 봉.
BUCKETS = [
    ("09:00~10:00", 9 * 60, 10 * 60),
    ("10:00~12:00", 10 * 60, 12 * 60),
    ("12:00~14:00", 12 * 60, 14 * 60),
    ("14:00~15:00", 14 * 60, 15 * 60),
    ("15:00~15:19", 15 * 60, _CUTOFF_MINUTE + 1),
    ("동시호가(15:30)", _CLOSE_MINUTE, _CLOSE_MINUTE + 1),
]


def _load_env() -> None:
    """``KR_QUANT_DB`` 가 없으면 관례대로 ../kr-quant/.env 를 시도한다(있으면 유지)."""
    if os.environ.get("KR_QUANT_DB"):
        return
    try:
        import _envload  # noqa: F401  import 만으로 환경변수 주입
    except Exception as exc:                                   # noqa: BLE001
        print(f"[경고] .env 자동 로드 실패({exc}) — KR_QUANT_DB 를 직접 export 하라.",
              file=sys.stderr)


def limit_up_price(prev_close: float) -> float:
    """전일종가 × 1.3 을 **그 가격대의** 호가단위로 내림한다.

    호가단위는 내림 전 값(prev×1.3)의 가격대로 정한다. 경계 근처에서 내림 후
    한 단계 아래 가격대로 떨어지는 일은 없다 — 내림은 그 단위의 배수로만 움직인다.
    """
    raw = prev_close * 1.3
    tick = tick_size(raw)
    return math.floor(raw / tick + 1e-9) * tick


def bucket_of(minute: int | None) -> str | None:
    if minute is None:
        return None
    for label, lo, hi in BUCKETS:
        if lo <= minute < hi:
            return label
    return None


def select_trades(ohlcv: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, int]:
    """사전등록 규칙 그대로 선택한다. 선택 논리는 :func:`select_close_bet` 하나다."""
    features = close_bet_features(ohlcv, adv_window=ADV_WINDOW)
    labels = close_bet_labels(ohlcv)
    # 유니버스(당일 거래대금 50억)는 피처에 없으니 일봉에서 붙인다.
    features = features.merge(
        ohlcv[["code", "date", "open", "high", "low", "close", "trade_value"]],
        on=["code", "date"],
        how="left",
    )
    prev_close = ohlcv.sort_values(["code", "date"]).groupby("code")["close"].shift(1)
    features = features.merge(
        ohlcv.assign(prev_close=prev_close)[["code", "date", "prev_close"]],
        on=["code", "date"],
        how="left",
    )
    rules = {
        "trade_value": (ADV_FLOOR, None),
        "tv_ratio": (TV_RATIO_MIN, None),
        "close_pos": (CLOSE_POS_MIN, None),
    }
    selected, n_excluded = select_close_bet(
        features, labels, rules=rules, top_n=TOP_N, exit_col="ret_open"
    )
    selected = selected[(selected["date"] >= pd.Timestamp(start)) & (selected["date"] <= pd.Timestamp(end))]
    return selected.reset_index(drop=True), n_excluded


def inspect_day(
    conn, code: str, date: pd.Timestamp, daily_prev_close: float, daily_close: float,
    daily_limit_approx: bool,
) -> dict:
    """한 (종목, 일자) 의 분봉을 읽어 상한가 도달 정보를 만든다."""
    out: dict = {"code": code, "date": date, "status": None}
    lo = (date - timedelta(days=10)).strftime("%Y-%m-%d")
    bars = load_minute_bars(conn, lo, date.strftime("%Y-%m-%d"), [code])
    if bars.empty:
        out["status"] = "분봉 없음"
        return out
    bars = bars[bars["ts"].dt.hour * 60 + bars["ts"].dt.minute <= _CLOSE_MINUTE]
    day = bars[bars["ts"].dt.normalize() == date]
    prev = bars[bars["ts"].dt.normalize() < date]
    if day.empty:
        out["status"] = "당일 분봉 없음"
        return out
    if prev.empty:
        out["status"] = "전일 분봉 없음"
        return out
    prev_day = prev["ts"].dt.normalize().max()
    prev = prev[prev["ts"].dt.normalize() == prev_day]
    prev_close_min = float(prev.iloc[-1]["close"])
    close_min = float(day.iloc[-1]["close"])
    last_minute = int(day.iloc[-1]["ts"].hour * 60 + day.iloc[-1]["ts"].minute)

    # 수정주가/원주가 비율이 전일·당일 같은지. 다르면 배당락·분할이거나 15:30 봉 누락으로
    # 분봉 종가 != 일봉 종가인 경우다. 어느 쪽이든 상한가 판정이 흔들리므로 제외.
    f_prev = daily_prev_close / prev_close_min if prev_close_min else np.nan
    f_day = daily_close / close_min if close_min else np.nan
    out["adj_prev"] = f_prev
    out["adj_day"] = f_day
    if not (np.isfinite(f_prev) and np.isfinite(f_day)) or abs(f_prev / f_day - 1) > ADJ_DRIFT_TOL:
        out["status"] = "일봉·분봉 종가비 괴리"
        return out

    minute = (day["ts"].dt.hour * 60 + day["ts"].dt.minute).to_numpy()
    high = day["high"].to_numpy(dtype=float)
    close = day["close"].to_numpy(dtype=float)
    low = day["low"].to_numpy(dtype=float)
    vol = day["volume"].to_numpy(dtype=float)
    tv = day["trade_value"].to_numpy(dtype=float)
    is_auction = minute == _CLOSE_MINUTE

    out["prev_close_min"] = prev_close_min
    out["close_min"] = close_min
    out["last_minute"] = last_minute
    out["has_1530"] = bool(last_minute == _CLOSE_MINUTE)
    out["minute_ratio"] = close_min / prev_close_min
    out["tv_auction"] = float(tv[is_auction].sum()) if is_auction.any() else np.nan
    out["vol_auction"] = float(vol[is_auction].sum()) if is_auction.any() else np.nan
    out["tv_day_min"] = float(tv.sum())

    # 상한가 가격 결정. (a) 원주가면 전일종가×1.3 호가단위 내림과 종가가 정확히 일치한다.
    # (b) 분봉도 수정주가인 종목(이후 증자·분할)은 호가단위에 안 맞는다 — 그때는
    #     "종가 == 당일 최고가 이고 일봉 근사 비율 >= 1.295" 를 상한가로 본다 (근사).
    limit_tick = limit_up_price(prev_close_min)
    out["limit_price_tick"] = limit_tick
    day_high = float(high.max())
    if abs(close_min - limit_tick) < 1e-6:
        limit, method = limit_tick, "tick"
    elif daily_limit_approx and abs(close_min - day_high) < 1e-6:
        limit, method = close_min, "close=high(근사)"
    else:
        limit, method = limit_tick, "tick"
    out["limit_price"] = limit
    out["limit_method"] = method
    out["minute_limit_close"] = bool(abs(close_min - limit) < 1e-6)

    hit = np.where(high >= limit - 1e-6)[0]
    if hit.size == 0:
        out["first_minute"] = None
        out["lock_minute"] = None
        out["status"] = "ok"
        return out
    i0 = int(hit[0])
    out["first_minute"] = int(minute[i0])
    after = slice(i0, None)
    out["n_bars_after"] = int(len(minute) - i0)
    below_close = close < limit - 1e-6
    out["broke_close_below"] = bool(below_close[after].any())
    out["broke_low_below"] = bool((low[after] < limit - 1e-6).any())
    out["n_bars_below_close"] = int(below_close[after].sum())
    out["vol_after"] = float(vol[after].sum())
    out["tv_after"] = float(tv[after].sum())
    at_limit = (close[after] >= limit - 1e-6) & (low[after] >= limit - 1e-6)
    out["tv_at_limit_locked"] = float(tv[after][at_limit].sum())
    out["vol_at_limit_locked"] = float(vol[after][at_limit].sum())
    out["tv_after_ex_auction"] = float(tv[after][~is_auction[after]].sum())

    # 최종 잠김 시각: 이 봉부터 15:30 까지 봉 종가가 한 번도 상한가 아래로 안 내려간
    # 마지막 구간의 시작. 종가가 상한가가 아니면 정의되지 않는다.
    if out["minute_limit_close"]:
        below_idx = np.where(below_close)[0]
        j0 = int(below_idx[-1]) + 1 if below_idx.size else i0
        j0 = max(j0, i0)
        out["lock_minute"] = int(minute[j0])
        lk = slice(j0, None)
        out["n_bars_after_lock"] = int(len(minute) - j0)
        out["tv_after_lock"] = float(tv[lk].sum())
        out["tv_after_lock_ex_auction"] = float(tv[lk][~is_auction[lk]].sum())
        out["vol_after_lock"] = float(vol[lk].sum())
    else:
        out["lock_minute"] = None
    out["status"] = "ok"
    return out


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float("nan")


def _fmt_eok(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{x / 1e8:,.2f}"


def build_report(sel: pd.DataFrame, res: pd.DataFrame, meta: dict) -> str:
    """마크다운 리포트 문자열."""
    L: list[str] = []
    L.append("# 상한가 도달 시각 진단 — 종가베팅 선택분\n")
    L.append(f"기간 {meta['start']} ~ {meta['end']} · 생성 {meta['generated']}\n")
    L.append("## 표본\n")
    L.append("| 항목 | 건수 |")
    L.append("|---|---|")
    for k, v in meta["counts"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("상한가 정의: (1) 일봉 근사 `close/prev_close >= 1.295` → (2) 분봉 원주가로 "
             "`전일 마지막 분봉 종가 × 1.3` 을 호가단위(KRX 2023~ 공통) 내림한 값과 당일 15:30 봉 "
             "종가가 **정확히 일치**. 둘이 어긋나거나 수정주가 비율이 전일·당일 0.2% 이상 다르면 제외.\n")

    ok = res[res["status"] == "ok"].copy()
    lim = ok[ok["is_limit"]].copy()

    n_lim = len(lim)
    late_first = lim[lim["first_minute"] >= 15 * 60]
    late_lock = lim[lim["lock_minute"] >= 15 * 60]

    # 2. 도달 시각 분포 — 처음 닿은 시각 / 최종 잠김 시각 두 가지
    L.append("## 2. 상한가 도달 시각 분포 (상한가 마감 건)\n")
    L.append("두 시각을 따로 본다. **처음 도달** = 고가가 상한가에 닿은 첫 분. "
             "**최종 잠김** = 그 봉부터 15:30 까지 봉 종가가 한 번도 상한가 아래로 안 내려간 마지막 구간의 시작 "
             "(처음 도달 후 풀렸다가 다시 잠긴 경우 재잠김 시각). 잔량이 얇은지는 최종 잠김 이후가 관건이다.\n")
    L.append("| 구간 | 처음 도달 건수 | 비율 | 그중 풀린 적 있음(봉 종가<상한가) | (봉 저가<상한가) | 최종 잠김 건수 | 비율 |")
    L.append("|---|---|---|---|---|---|---|")
    for label, _, _ in BUCKETS:
        g = lim[lim["bucket"] == label]
        gl = lim[lim["lock_bucket"] == label]
        n, nl = len(g), len(gl)
        brk_c = int(g["broke_close_below"].sum()) if n else 0
        brk_l = int(g["broke_low_below"].sum()) if n else 0
        brk_txt = f"{brk_c} ({brk_c / n:.0%}) | {brk_l} ({brk_l / n:.0%})" if n else "— | —"
        L.append(f"| {label} | {n} | {n / n_lim:.1%} | {brk_txt} | {nl} | {nl / n_lim:.1%} |")
    n_none = int(lim["first_minute"].isna().sum())
    L.append(f"| (도달 기록 없음) | {n_none} | {n_none / n_lim:.1%} | — | — | — | — |")
    L.append(f"| **합계** | **{n_lim}** | 100% | | | **{n_lim}** | 100% |")
    L.append("")
    L.append("'풀림' = 처음 도달한 봉 이후 15:30 까지 (15:30 봉 포함) 봉 종가(또는 저가)가 상한가 아래로 내려간 봉이 하나라도 있음. "
             "'동시호가(15:30)' 행의 최종 잠김 = 15:19 봉 종가는 상한가 아래였는데 15:30 단일가가 상한가로 찍힌 건.\n")

    # 3. 구간별 익일 시가 수익률
    L.append("## 3. 도달 시각 구간별 익일 시가 수익률 (일봉, 왕복 34bp 차감)\n")

    def _row(label: str, g: pd.DataFrame) -> str:
        if g.empty:
            return f"| {label} | 0 | — | — | — | — | — |"
        r = g["ret_open"] - COST_ROUND_TRIP
        return (f"| {label} | {len(g)} | {r.mean():+.2%} | {r.median():+.2%} | {(r > 0).mean():.0%} "
                f"| {_q(r, .25):+.2%} | {_q(r, .75):+.2%} |")

    for title, col in (("처음 도달 시각 기준", "bucket"), ("최종 잠김 시각 기준", "lock_bucket")):
        L.append(f"**{title}**\n")
        L.append("| 구간 | n | 평균 net | 중앙 net | 승률 | p25 | p75 |")
        L.append("|---|---|---|---|---|---|---|")
        for label, _, _ in BUCKETS:
            L.append(_row(label, lim[lim[col] == label]))
        L.append("")
    L.append("**참조군**\n")
    L.append("| 구간 | n | 평균 net | 중앙 net | 승률 | p25 | p75 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(_row("상한가 마감 전체", lim))
    L.append(_row("  처음 도달 후 한 번도 안 풀림", lim[~lim["broke_close_below"].astype(bool)]))
    L.append(_row("  처음 도달 후 풀린 적 있음", lim[lim["broke_close_below"].astype(bool)]))
    L.append(_row("상한가 아님 (분봉 확인됨)", ok[~ok["is_limit"]]))
    L.append(_row("선택 전체 (분봉 확인됨)", ok))
    L.append(_row("선택 전체 (일봉 기준, 분봉 무관)", sel))
    L.append("")

    # 4. 도달 이후 거래대금 분포
    L.append("## 4. 도달 이후 거래대금 (억원) — 체결 존재의 대용\n")
    L.append("`tv_after` = 처음 도달한 봉부터 15:30 봉까지 거래대금 합계(도달 봉 포함). "
             "`tv_lock` = 최종 잠김 봉부터 15:30 까지 합계(15:30 봉 포함). "
             "`tv_1530` = 15:30 단일가 봉 거래대금 — **종가 동시호가에서 실제 체결된 금액**. "
             "`share_1530` = tv_1530 / 당일 분봉 거래대금 합계.\n")
    L.append("| 구간 (처음 도달 기준) | n | tv_after 중앙 | p25 | p75 | tv_lock 중앙 | p25 | p75 | tv_1530 중앙 | p25 | p75 | share_1530 중앙 | tv_1530≤0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def _row4(label: str, g: pd.DataFrame) -> str:
        if g.empty:
            return f"| {label} | 0 | — | — | — | — | — | — | — | — | — | — | — |"
        share = g["tv_auction"] / g["tv_day_min"]
        zero = int((g["tv_auction"].fillna(0) <= 0).sum())
        tl = g["tv_after_lock"] if "tv_after_lock" in g else pd.Series(dtype=float)
        return (f"| {label} | {len(g)} | {_fmt_eok(g['tv_after'].median())} | {_fmt_eok(_q(g['tv_after'], .25))} "
                f"| {_fmt_eok(_q(g['tv_after'], .75))} | {_fmt_eok(tl.median())} | {_fmt_eok(_q(tl, .25))} | {_fmt_eok(_q(tl, .75))} "
                f"| {_fmt_eok(g['tv_auction'].median())} | {_fmt_eok(_q(g['tv_auction'], .25))} "
                f"| {_fmt_eok(_q(g['tv_auction'], .75))} | {share.median():.2%} | {zero} |")

    for label, _, _ in BUCKETS:
        L.append(_row4(label, lim[lim["bucket"] == label]))
    L.append(_row4("**15:00 이후 처음 도달 합계**", late_first))
    L.append(_row4("15:00 이전 처음 도달 합계", lim[lim["first_minute"] < 15 * 60]))
    L.append(_row4("상한가 마감 전체", lim))
    L.append(_row4("(참고) 상한가 아님 — 장중 상한가 터치 후 밀린 건만 tv_after 있음", ok[~ok["is_limit"]]))
    L.append("")
    L.append("| 구간 (최종 잠김 기준) | n | tv_lock 중앙 | p25 | p75 | tv_1530 중앙 | p25 | p75 | share_1530 중앙 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for label, _, _ in BUCKETS:
        g = lim[lim["lock_bucket"] == label]
        if g.empty:
            L.append(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue
        share = g["tv_auction"] / g["tv_day_min"]
        L.append(f"| {label} | {len(g)} | {_fmt_eok(g['tv_after_lock'].median())} | {_fmt_eok(_q(g['tv_after_lock'], .25))} "
                 f"| {_fmt_eok(_q(g['tv_after_lock'], .75))} | {_fmt_eok(g['tv_auction'].median())} "
                 f"| {_fmt_eok(_q(g['tv_auction'], .25))} | {_fmt_eok(_q(g['tv_auction'], .75))} | {share.median():.2%} |")
    g = late_lock
    if not g.empty:
        share = g["tv_auction"] / g["tv_day_min"]
        L.append(f"| **15:00 이후 최종 잠김 합계** | {len(g)} | {_fmt_eok(g['tv_after_lock'].median())} | {_fmt_eok(_q(g['tv_after_lock'], .25))} "
                 f"| {_fmt_eok(_q(g['tv_after_lock'], .75))} | {_fmt_eok(g['tv_auction'].median())} "
                 f"| {_fmt_eok(_q(g['tv_auction'], .25))} | {_fmt_eok(_q(g['tv_auction'], .75))} | {share.median():.2%} |")
    L.append("")
    L.append("15:30 단일가 봉 거래대금 / 당일 분봉 거래대금 합계 (상한가 마감 vs 상한가 아님):\n")
    for label, g in (("상한가 마감", lim), ("상한가 아님", ok[~ok["is_limit"]])):
        if g.empty:
            continue
        share = (g["tv_auction"] / g["tv_day_min"]).dropna()
        L.append(f"- {label}: n={len(share)} share_1530 중앙 {share.median():.2%} · p25 {_q(share, .25):.2%} · p75 {_q(share, .75):.2%} · 평균 {share.mean():.2%}"
                 f" · tv_1530 중앙 {_fmt_eok(g['tv_auction'].median())}억")
    L.append("")

    # 1. 개별 건 (15:00 이후 처음 도달 또는 최종 잠김)
    L.append("## 1. 개별 건 — 15:00 이후 처음 도달했거나 15:00 이후 최종 잠긴 상한가 마감 종목\n")
    L.append("| 종목 | 일자 | 처음 도달 | 최종 잠김 | 풀림(종가) | 풀림(저가) | 잠김후 봉수 | tv_after(억) | tv_lock(억) | tv_1530(억) | share_1530 | 익일시가 net |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    late_any = lim[(lim["first_minute"] >= 15 * 60) | (lim["lock_minute"] >= 15 * 60)]

    def _hm(m: float) -> str:
        return "—" if pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}"

    for _, r in late_any.sort_values(["date", "lock_minute"]).iterrows():
        L.append(f"| {r['code']} | {pd.Timestamp(r['date']).date()} | {_hm(r['first_minute'])} | {_hm(r['lock_minute'])} "
                 f"| {'예' if r['broke_close_below'] else '아니오'} | {'예' if r['broke_low_below'] else '아니오'} | {int(r['n_bars_after_lock'])} "
                 f"| {_fmt_eok(r['tv_after'])} | {_fmt_eok(r['tv_after_lock'])} | {_fmt_eok(r['tv_auction'])} | {r['tv_auction'] / r['tv_day_min']:.2%} "
                 f"| {r['ret_open'] - COST_ROUND_TRIP:+.2%} |")
    L.append("")
    L.append("## 제외·불일치 내역\n")
    for k, v in meta["exclusions"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## 미측정 (이 데이터로 알 수 없는 것)\n")
    L.append("- 15:20~15:30 동시호가의 **매수 잔량 대비 체결량** (호가창 데이터 없음). 15:30 봉 거래대금은 "
             "'그 가격에 얼마가 체결됐는가' 이지 '내 주문이 체결됐을 확률' 이 아니다 — 매수 잔량이 체결량의 "
             "수십 배면 시간우선순위상 거의 못 산다.")
    L.append("- 15:19 이전에 이미 잠긴 종목은 동시호가 직전 주문이 매수 잔량 맨 뒤에 선다. 따라서 '15:19 전 "
             "도달 + 안 풀림' 건의 15:30 체결은 대부분 먼저 걸어둔 주문의 것으로 봐야 한다.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="종가베팅 선택분의 상한가 도달 시각 진단 (읽기 전용)")
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--out", default=None, help="마크다운 결과 경로")
    ap.add_argument("--csv", default=None, help="건별 결과 CSV 경로")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args(argv)

    _load_env()
    conn = connect(args.dsn)
    try:
        warm_start = (pd.Timestamp(args.start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        # 라벨(D+1 시가)을 위해 끝도 며칠 더 읽는다.
        warm_end = (pd.Timestamp(args.end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        print(f"일봉 로딩 {warm_start} ~ {warm_end} ...", file=sys.stderr)
        ohlcv = load_ohlcv(conn, warm_start, warm_end)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        print(f"  {len(ohlcv):,}행 · {ohlcv['code'].nunique():,}종목", file=sys.stderr)

        sel, n_excl = select_trades(ohlcv, args.start, args.end)
        sel["daily_ratio"] = sel["close"] / sel["prev_close"]
        sel["daily_limit"] = sel["daily_ratio"] >= DAILY_LIMIT_RATIO
        print(f"선택 {len(sel):,}건 · 일봉 상한가 마감 {int(sel['daily_limit'].sum()):,}건", file=sys.stderr)

        codes = set(available_codes(conn))
        sel["has_minute_code"] = sel["code"].isin(codes)
        print(f"분봉 보유 종목 {len(codes):,} · 선택분 중 분봉 종목 {int(sel['has_minute_code'].sum()):,}건", file=sys.stderr)

        rows = []
        todo = sel[sel["has_minute_code"]]
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            info = inspect_day(
                conn, str(r["code"]), pd.Timestamp(r["date"]), float(r["prev_close"]), float(r["close"]),
                bool(r["daily_limit"]),
            )
            info["ret_open"] = float(r["ret_open"])
            info["daily_ratio"] = float(r["daily_ratio"])
            info["daily_limit"] = bool(r["daily_limit"])
            info["tv_ratio"] = float(r["tv_ratio"])
            info["trade_value_daily"] = float(r["trade_value"])
            rows.append(info)
            if i % 100 == 0:
                print(f"  분봉 검사 {i}/{len(todo)}", file=sys.stderr)
        res = pd.DataFrame(rows)
    finally:
        conn.close()

    # 상한가 판정 합치기: 일봉 근사 + 분봉 정확 판정이 일치해야 '상한가 마감'
    ok_mask = res["status"] == "ok"
    agree = ok_mask & (res["daily_limit"] == res["minute_limit_close"].fillna(False))
    mismatch = ok_mask & ~agree
    res.loc[mismatch, "status"] = "상한가 판정 불일치"
    res["is_limit"] = res["daily_limit"] & res["minute_limit_close"].fillna(False) & (res["status"] == "ok")
    res["bucket"] = res["first_minute"].map(lambda m: bucket_of(None if pd.isna(m) else int(m)))
    res["lock_bucket"] = res["lock_minute"].map(lambda m: bucket_of(None if pd.isna(m) else int(m)))
    # 상한가 아님인데 장중 상한가를 찍었던 건은 '도달' 이 있어도 버킷 통계에 안 들어간다 (is_limit 로 거른다)

    mismatch_detail = res[mismatch][["code", "date", "daily_ratio", "minute_ratio", "daily_limit", "minute_limit_close", "limit_price_tick", "close_min", "has_1530"]]

    counts = {
        "종가베팅 선택 (일봉, 전 종목)": len(sel),
        "  그중 일봉 근사 상한가 마감 (close/prev>=1.295)": int(sel["daily_limit"].sum()),
        "  다음날 봉 없어 선택 전 제외": n_excl,
        "분봉 보유 종목 수": len(codes),
        "선택분 중 분봉 보유 종목 건": int(sel["has_minute_code"].sum()),
        "  그중 분봉 검사 통과(status=ok)": int((res["status"] == "ok").sum()),
        "  그중 상한가 마감 (일봉·분봉 일치)": int(res["is_limit"].sum()),
        "  그중 상한가 아님": int(((res["status"] == "ok") & ~res["is_limit"]).sum()),
        "  상한가 가격 결정 방식: 호가단위 내림 정확 일치 / 종가=고가 근사(분봉도 수정주가)": (
            f"{int((res['is_limit'] & (res['limit_method'] == 'tick')).sum())} / "
            f"{int((res['is_limit'] & (res['limit_method'] != 'tick')).sum())}"
        ),
        "상한가 마감 표본 / 일봉 기준 전체 상한가 마감": f"{int(res['is_limit'].sum())} / {int(sel['daily_limit'].sum())} = {int(res['is_limit'].sum()) / max(1, int(sel['daily_limit'].sum())):.1%}",
    }
    exclusions = {k: int(v) for k, v in res["status"].value_counts().items() if k != "ok"}
    if not mismatch_detail.empty:
        exclusions["불일치 상세"] = "; ".join(
            f"{r.code}@{pd.Timestamp(r.date).date()} 일봉비 {r.daily_ratio:.4f} 분봉비 {r.minute_ratio:.4f} 호가내림 상한가 {r.limit_price_tick} 분봉종가 {r.close_min} 15:30봉 {'유' if r.has_1530 else '무'}"
            for r in mismatch_detail.itertuples()
        )
    # 15:30 봉이 없는 날(수집 누락)도 보고
    n_no1530 = int(((res["status"] == "ok") & ~res["has_1530"].fillna(False).astype(bool)).sum())
    exclusions["(참고) status=ok 인데 15:30 봉 없음"] = n_no1530

    meta = {
        "start": args.start,
        "end": args.end,
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "counts": counts,
        "exclusions": exclusions,
    }
    report = build_report(sel, res, meta)
    print(report)
    if args.out:
        pathlib.Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n저장: {args.out}", file=sys.stderr)
    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"CSV: {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
