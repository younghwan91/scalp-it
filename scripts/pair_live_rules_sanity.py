"""짝꿍매매 — 현재 라이브 규칙 근사 백테스트. 사전등록 `docs/research/manju/72`.

12번(`verify_pair.py`)의 인프라(`pair_events.py` 로더·`limitup_timing.py`)를 재사용하되
세 가지를 지금 라이브 규칙에 맞게 바꾼다:
  트리거   "잠김"(종가==상한가) 대신 "3틱 근접"(고가 >= 상한가 - 3틱)
  후속주   60일 잔차상관 상위 30 대신 같은 27종 섹터 내 등락률 1위
  청산     고정 5분 보유 대신 대장 이탈(근접 이탈)·재난선(-2%)·타임스탑(3분) 중 먼저

읽기 전용, DB만 읽는다, 실주문 없음.

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/pair_live_rules_sanity.py --start 2025-08-04 --end 2026-08-21
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from limitup_timing import _load_env, limit_up_price
from pair_events import _q, load_daily_raw

from scalp_it.adapters.db import load_stock_master
from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect
from scalp_it.ticksize import tick_size

LEAD_CUM_TV_MIN = 300e8
FOL_CUM_TV_MIN = 50e8
FOL_RET_LO, FOL_RET_HI = 0.10, 0.15
NEAR_TICKS = 3
DISASTER = 0.02
HOLD_MIN = 3                 # 타임스탑 3분(180초 근사)
LAST_TRIGGER_MINUTE = 15 * 60 + 13
# 고정 가정값(값은 바꾸지 않는다). 일자별 거래세 스케줄은 krx_quant_core.costs 참고.
COST = 0.0064
COST_LO, COST_HI = 0.0034, 0.0094
N_FOLDS = 4
_CLOSE_MINUTE = 15 * 60 + 30
_OPEN_MINUTE = 9 * 60


def scan_date_near(day_bars: pd.DataFrame, raw_day: pd.DataFrame) -> tuple[list[dict], dict]:
    """근접(3틱) 트리거 + 스냅샷. `pair_events.scan_date`의 근접판."""
    excl: dict[str, int] = {}
    bars = day_bars[day_bars["ts"].dt.hour * 60 + day_bars["ts"].dt.minute <= _CLOSE_MINUTE].copy()
    bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
    prev = raw_day.set_index("code")["prev_close"]
    raw_close = raw_day.set_index("code")["close"]

    last = bars.groupby("code")["close"].last()
    ok_raw = {}
    for code, c in last.items():
        rc = raw_close.get(code, np.nan)
        ok_raw[code] = bool(np.isfinite(rc) and rc > 0 and abs(c / rc - 1) <= 0.002)
    bars = bars[bars["code"].map(ok_raw).fillna(False).astype(bool)]

    bars["cum_tv"] = bars.groupby("code")["trade_value"].cumsum()
    lim_by_code = {c: limit_up_price(p) for c, p in prev.items() if np.isfinite(p) and p > 0}
    bars["limit"] = bars["code"].map(lim_by_code)
    bars = bars[bars["limit"].notna()]
    if bars.empty:
        return [], excl

    close_w = bars.pivot(index="minute", columns="code", values="close").ffill()
    open_w = bars.pivot(index="minute", columns="code", values="open")
    high_w = bars.pivot(index="minute", columns="code", values="high")
    low_w = bars.pivot(index="minute", columns="code", values="low")
    cumtv_w = bars.pivot(index="minute", columns="code", values="cum_tv").ffill()
    limit_s = bars.groupby("code")["limit"].first()
    prev_s = prev.reindex(close_w.columns)
    near_thresh = pd.Series({c: limit_s[c] - tick_size(limit_s[c]) * NEAR_TICKS for c in limit_s.index})

    triggers: list[dict] = []
    for code in high_w.columns:
        if code not in near_thresh.index:
            continue
        thr = near_thresh[code]
        col = high_w[code]
        hit = col >= thr - 1e-6
        if not hit.any():
            continue
        T = int(hit.idxmax())
        if T > LAST_TRIGGER_MINUTE:
            excl["대장 근접 15:13 이후"] = excl.get("대장 근접 15:13 이후", 0) + 1
            continue
        cum = float(cumtv_w.at[T, code]) if T in cumtv_w.index and code in cumtv_w.columns else np.nan
        if not np.isfinite(cum) or cum < LEAD_CUM_TV_MIN:
            excl["대장 누적 거래대금 < 300억"] = excl.get("대장 누적 거래대금 < 300억", 0) + 1
            continue
        triggers.append({
            "lead": code, "T": T, "lead_cum_tv": cum,
            "lead_limit": float(limit_s[code]), "lead_near_thresh": float(thr),
            "lead_prev_close": float(prev_s[code]),
        })

    snap = {"close": close_w, "open": open_w, "high": high_w, "low": low_w,
            "cum_tv": cumtv_w, "limit": limit_s, "near_thresh": near_thresh, "prev": prev_s}
    return triggers, snap  # type: ignore[return-value]


def follower_by_sector(snap: dict, trig: dict, sector_of: dict[str, str]) -> tuple[str | None, dict]:
    """T 시점 같은 섹터 내 +10~15% 등락률 후보 중 등락률 1위. (code, meta) 또는 (None, {})."""
    T, lead = trig["T"], trig["lead"]
    close_w, open_w, cum_w = snap["close"], snap["open"], snap["cum_tv"]
    if T not in close_w.index:
        return None, {}
    lead_sector = sector_of.get(lead)
    if not lead_sector:
        return None, {}
    same_sector = [c for c in close_w.columns if c != lead and sector_of.get(c) == lead_sector]
    if not same_sector:
        return None, {}
    c = close_w.loc[T, same_sector]
    ret = c / snap["prev"].reindex(same_sector) - 1
    cum = cum_w.loc[T, same_sector] if T in cum_w.index else pd.Series(np.nan, index=same_sector)
    cand = pd.DataFrame({"ret": ret, "cum_tv": cum, "limit": snap["limit"].reindex(same_sector)})
    cand = cand[(cand["ret"] >= FOL_RET_LO) & (cand["ret"] <= FOL_RET_HI) & (cand["cum_tv"] >= FOL_CUM_TV_MIN)]
    if cand.empty:
        return None, {}
    T1 = T + 1
    if T1 not in open_w.index:
        return None, {}
    o1 = open_w.loc[T1].reindex(cand.index)
    cand = cand[o1.notna() & ((o1 - cand["limit"]).abs() > 1e-6)]
    if cand.empty:
        return None, {}
    fol = cand["ret"].idxmax()
    return fol, {"fol_ret": float(cand.at[fol, "ret"]), "fol_cum_tv": float(cand.at[fol, "cum_tv"])}


def _next_open(open_s: pd.Series, minute: int) -> tuple[float, int]:
    s = open_s.dropna()
    s = s[s.index >= minute]
    if s.empty:
        return float("nan"), -1
    return float(s.iloc[0]), int(s.index[0])


def outcome_live(snap: dict, trig: dict, fol: str) -> dict | None:
    T, lead = trig["T"], trig["lead"]
    open_w, close_w = snap["open"], snap["close"]
    high_w, low_w = snap["high"], snap["low"]
    if T + 1 not in open_w.index or fol not in open_w.columns:
        return None
    entry = open_w.at[T + 1, fol] if fol in open_w.columns else np.nan
    if not np.isfinite(entry) or entry <= 0:
        return None
    fol_limit = float(snap["limit"].get(fol, np.nan))

    lead_close = close_w[lead] if lead in close_w.columns else pd.Series(dtype=float)
    lead_thresh = trig["lead_near_thresh"]
    k_exit = None
    for k in range(T + 1, T + HOLD_MIN + 1):
        c = lead_close.get(k, np.nan)
        if np.isfinite(c) and c < lead_thresh - 1e-6:
            k_exit = k
            break

    disaster_minute = None
    if fol in low_w.columns:
        for k in range(T + 1, T + HOLD_MIN + 1):
            lo = low_w[fol].get(k, np.nan)
            if np.isfinite(lo) and lo <= entry * (1 - DISASTER):
                disaster_minute = k
                break

    if disaster_minute is not None and (k_exit is None or disaster_minute <= k_exit):
        exit_px = entry * (1 - DISASTER)
        kind = "disaster"
        exit_minute = disaster_minute
    elif k_exit is not None:
        exit_px, exit_actual = _next_open(open_w[fol], k_exit + 1)
        kind = "lead_exit"
        exit_minute = k_exit
        if not np.isfinite(exit_px):
            exit_px, _ = _next_open(open_w[fol], T + HOLD_MIN + 1)
            kind = "lead_exit_fallback_time"
    else:
        exit_px, exit_actual = _next_open(open_w[fol], T + HOLD_MIN + 1)
        kind = "time"
        exit_minute = T + HOLD_MIN

    if not np.isfinite(exit_px):
        return None

    win_hi = high_w[fol].reindex(range(T + 1, T + HOLD_MIN + 1)).dropna() if fol in high_w.columns else pd.Series(dtype=float)
    fol_hit_limit = bool(np.isfinite(fol_limit) and ((win_hi >= fol_limit - 1e-6).any() or exit_px >= fol_limit - 1e-6))

    return {
        "entry": entry, "exit": exit_px, "exit_kind": kind, "exit_minute": exit_minute,
        "ret": exit_px / entry - 1.0, "fol_hit_limit": fol_hit_limit,
    }


def collect(conn, start: str, end: str) -> dict:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    warm = (s - timedelta(days=10)).strftime("%Y-%m-%d")
    raw = load_daily_raw(conn, warm, end).sort_values(["code", "date"])
    master = load_stock_master(conn)
    sector_of = dict(zip(master["code"].astype(str).str.strip(), master["sector"]))
    idx_days = _q(conn, "select distinct date from daily_bars where date >= %s and date <= %s order by date",
                  (start, end))["date"]
    trading_days = [pd.Timestamp(d) for d in idx_days]

    rows: list[dict] = []
    excl: dict[str, int] = {}

    def _add(k, v=1):
        excl[k] = excl.get(k, 0) + v

    for i, d in enumerate(trading_days):
        ds = d.strftime("%Y-%m-%d")
        raw_day = raw[raw["date"] == d]
        day_bars = load_minute_bars(conn, ds, ds)
        if day_bars.empty:
            continue
        trigs, snap = scan_date_near(day_bars, raw_day)
        if not trigs:
            continue
        for t in trigs:
            fol, meta = follower_by_sector(snap, t, sector_of)
            if fol is None:
                _add("후속주 없음(섹터·등락률·거래대금 조건)")
                continue
            out = outcome_live(snap, t, fol)
            if out is None:
                _add("청산가 계산 불가(T+1 봉 없음 등)")
                continue
            rows.append({"date": d, "lead": t["lead"], "T": t["T"], "fol": fol,
                         "lead_cum_tv": t["lead_cum_tv"], **meta, **out})
        if i % 20 == 0:
            print(f"  {ds} 누적 거래 {len(rows)}", file=sys.stderr)
    return {"trades": pd.DataFrame(rows), "excl": excl}


def _fold_stats(r: pd.Series, n_folds: int) -> list[float]:
    folds = np.array_split(np.arange(len(r)), n_folds)
    return [float(r.iloc[f].mean() - COST) for f in folds if len(f)]


def build_report(data: dict) -> str:
    tr = data["trades"]
    L = ["# 짝꿍매매 — 현재 라이브 규칙 근사 백테스트 결과", ""]
    L.append("> 사전등록 `docs/research/manju/72`.")
    L.append("")
    L.append(f"- 후속주 거래 {len(tr):,}건")
    L.append("- 제외 사유: " + ", ".join(f"{k} {v}" for k, v in data["excl"].items()))
    L.append("")
    if len(tr) < 30:
        L.append("**표본 부족(<30) — 판정 불가.**")
        return "\n".join(L)

    r = tr["ret"]
    net = r - COST
    L.append("## 1단계 — 전부 진입")
    L.append("")
    L.append(f"- 평균 {r.mean():+.3%} / net64 {net.mean():+.3%} / 중앙 net64 {net.median():+.3%} "
              f"/ 승률 {(r>0).mean():.1%} / n={len(r)}")
    top1 = net[net <= net.quantile(0.99)]
    L.append(f"- 상위1%제거 net64 평균 {top1.mean():+.3%} (n={len(top1)})")
    ex_limit = net[~tr["fol_hit_limit"]]
    L.append(f"- 진입후 3분내 상한가 도달 제외 net64 평균 {ex_limit.mean():+.3%} (n={len(ex_limit)})")
    tr_sorted = tr.sort_values("date")
    folds = _fold_stats(net.loc[tr_sorted.index], N_FOLDS)
    neg_folds = sum(1 for f in folds if f < 0)
    L.append(f"- 시간순 {N_FOLDS}폴드 net64: " + ", ".join(f"{f:+.3%}" for f in folds) + f" (음수 {neg_folds}/{N_FOLDS})")
    L.append("")
    L.append("## 청산 사유 분포")
    L.append("")
    L.append(tr["exit_kind"].value_counts().to_string())
    L.append("")

    cond = {
        "1 net64 평균·중앙 > 0": net.mean() > 0 and net.median() > 0,
        "2 음수폴드<=1": neg_folds <= 1,
        "3 상위1%제거 net64>0": top1.mean() > 0,
        "4 상한가도달제외 net64>0": ex_limit.mean() > 0 if len(ex_limit) else False,
        "5 승률>50%": (r > 0).mean() > 0.5,
    }
    L.append("## 판정 (사전등록 §5)")
    L.append("")
    L.append("  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()))
    L.append("")
    L.append(f"**{'채택(추가 조사 가치)' if all(cond.values()) else '기각'}**")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-04")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default="data/pair_live_rules_sanity.md")
    args = ap.parse_args(argv)

    _load_env(None)
    if args.cache and pathlib.Path(args.cache).exists():
        data = pd.read_pickle(args.cache)
    else:
        conn = connect()
        try:
            data = collect(conn, args.start, args.end)
        finally:
            conn.close()
        if args.cache:
            pd.to_pickle(data, args.cache)

    report = build_report(data)
    print(report)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n[pair_live_rules_sanity] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
