"""105번 개장 동시호가 → 첫 1·5·15분 탐색 — DB 읽기 · 6셀 통계 · 보고용 분해. 기술, 판정 아님.

    deploy/research_run.sh .venv/bin/python -W ignore scripts/explore_105/run.py --out ~/of80/explore105

판정 셀 6개와 기각 기준은 docs/research/manju/105 §2 에 코드보다 먼저 적혔다. 여기서는 그 셀만 매치드 널로
재고, 나머지(1분·갭 10분위·상한가성·시장)는 보고만 한다. 라벨 산술은 labels.py(테스트 있음)에만 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _envload  # noqa: F401
import labels as L
from krx_quant_core.stats import cluster_bootstrap_ci, matched_alpha, matched_control
from krx_quant_core.stats.trials import record_trial

START, END = "2025-08-01", "2026-10-02"
STRATA = ("date", "vol_q", "tv_q")
N_BOOT, SEED = 2000, 105
JUDGE_HOLDS = (5, 15)
CELLS = ("top10", "bot10", "mid")


def bad_days() -> set[str]:
    """불량일 — 추적 목록 + 자동 등록분(scripts/_bad_days.py 가 정문)."""
    import _bad_days

    return set(_bad_days.load())


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 20일 변동성에 앞 거래일 ~30개가 필요하다 → 일봉은 START 보다 45일 앞부터.
    lo = (pd.Timestamp(START) - pd.Timedelta(days=45)).date().isoformat()
    daily = pd.read_sql(
        "select code, date, open, high, low, close, trade_value from daily_bars "
        "where date >= %s and date <= %s order by code, date", conn, params=(lo, END))
    minute = pd.read_sql(
        "select code, ts, open, close from minute_bars "
        "where ts >= %s and ts < %s and ts::time >= '09:00' and ts::time <= '09:15'",
        conn, params=(START, (pd.Timestamp(END) + pd.Timedelta(days=1)).date().isoformat()))
    # 전일 원주가 종가용 마감 분봉(15:19~15:30). daily_bars.close 는 수정주가라 분봉 09:00 open 과 단위가 다르다.
    minute_eod = pd.read_sql(
        "select code, ts, close from minute_bars "
        "where ts >= %s and ts < %s and ts::time >= '15:19' and ts::time <= '15:30'",
        conn, params=((pd.Timestamp(START) - pd.Timedelta(days=10)).date().isoformat(), END))
    markets = pd.read_sql("select code, market from stocks where market in ('거래소','코스닥')", conn)
    return daily, minute, markets, minute_eod


def summarize(r: np.ndarray) -> dict:
    return {"n": len(r), "mean": float(r.mean()), "median": float(np.median(r)),
            "frac_pos": float((r > 0).mean())}


def cell_stats(ev: pd.DataFrame, cell: str, h: int) -> dict:
    """한 판정 셀: 무조건부 요약 · 매치드 널 alpha(날짜 클러스터 CI) · 월별 부호. 풀은 유니버스 전체."""
    col = f"r{h}"
    pool = ev.dropna(subset=[col])
    tr = pool[pool["ret_cell"] == cell]
    r = tr[col].to_numpy() * 1e4
    out = {"cell": cell, "hold": h, **summarize(r)}
    lo, hi, _ = cluster_bootstrap_ci(r, tr["date"].to_numpy(), n_boot=N_BOOT, seed=SEED)
    out.update(raw_ci_low=lo, raw_ci_high=hi)
    ctl = matched_control(tr, pool, strata=STRATA, n_per=1, seed=SEED, exclude_self=True)
    ma = matched_alpha(tr, pool, ctl, value=col, cluster="date", n_boot=N_BOOT, seed=SEED)
    out.update(alpha=ma.alpha * 1e4, alpha_ci_low=ma.ci_low * 1e4, alpha_ci_high=ma.ci_high * 1e4,
               control_mean=ma.control_mean * 1e4, n_matched=ma.n_trades, n_unmatched=ma.n_unmatched,
               n_days=ma.n_clusters)
    mo = tr.groupby(tr["date"].dt.strftime("%Y-%m"))[col].mean() * 1e4
    out.update(months=len(mo), neg_months=int((mo < 0).sum()),
               monthly={k: round(float(v), 1) for k, v in mo.items()})
    return out


def breakdown(ev: pd.DataFrame, by: str, holds=L.HOLDS) -> pd.DataFrame:
    rows = []
    for k, g in ev.groupby(by, sort=True, observed=True):
        for h in holds:
            r = g[f"r{h}"].dropna().to_numpy() * 1e4
            if len(r):
                rows.append({by: k, "hold": h, **summarize(r)})
    return pd.DataFrame(rows)


def fmt_bp(x: float) -> str:
    return f"{x:+.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "of80" / "explore105"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    daily, minute, markets, minute_eod = load(conn)
    bad = bad_days()
    ev = L.build_events(daily, minute, markets, minute_eod, bad_days=bad)
    ev.to_pickle(out / "events.pkl")
    print(f"사건 {len(ev):,}건 · {ev.date.min().date()}~{ev.date.max().date()} · 거래일 {ev.date.nunique()} "
          f"· 종목 {ev.code.nunique():,} · 제외일 {len(bad)} · 평균 비용 {ev.cost_bp.mean():.1f}bp")
    print(f"일당 유니버스 중앙값 {ev.groupby('date').size().median():.0f} (상위 1,300 중 09:00 체결 있는 것)")

    # ── 판정 셀 6개 ──────────────────────────────────────────────────────────────
    config_base = {"start": START, "end": END, "top_value": L.TOP_VALUE, "min_price": L.MIN_PRICE,
                   "ret_tail": L.RET_TAIL, "strata": list(STRATA), "n_boot": N_BOOT, "seed": SEED,
                   # 1차 실행(원장에 남음)은 전일 종가를 daily_bars(수정주가)로 썼다 — 폐기. 2차는 분봉 15:30.
                   "prev_close_source": "minute_1530",
                   "cost": {"tax": L.TAX_BP, "fee": L.FEE_BP, "open_spread": L.OPEN_SPREAD_BP,
                            "cont_spread": L.CONT_SPREAD_BP}, "bad_days": sorted(bad)}
    results = []
    for cell in CELLS:
        for h in JUDGE_HOLDS:
            record_trial("105-open-auction", {**config_base, "cell": cell, "hold": h},
                         logs_dir=ROOT / "research" / "logs", note="탐색 6셀")
            results.append(cell_stats(ev, cell, h))
    (out / "cells.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n## 판정 셀 (bp, 비용 차감 후)")
    print("| 셀 | 보유 | n | 평균 | 중앙 | 양(+) | 원CI | 대조 평균 | 널 alpha | alpha CI | 음수월 |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for s in results:
        print(f"| {s['cell']} | {s['hold']}분 | {s['n']:,} | {fmt_bp(s['mean'])} | {fmt_bp(s['median'])} | "
              f"{s['frac_pos']:.1%} | [{fmt_bp(s['raw_ci_low'])}, {fmt_bp(s['raw_ci_high'])}] | "
              f"{fmt_bp(s['control_mean'])} | **{fmt_bp(s['alpha'])}** | "
              f"[{fmt_bp(s['alpha_ci_low'])}, {fmt_bp(s['alpha_ci_high'])}] | "
              f"{s['neg_months']}/{s['months']} |")
    passing = [s for s in results if s["alpha_ci_low"] > 0]
    print(f"\n기각 기준(6셀 전부 alpha CI 하한 ≤ 0) 대비: CI 하한 > 0 인 셀 {len(passing)}개"
          + (" → 기준 충족(닫힘)" if not passing else " → " + ", ".join(f"{s['cell']}/{s['hold']}분" for s in passing)))
    print("\n월별 평균(bp):")
    for s in results:
        print(f"  {s['cell']}/{s['hold']}분: " + " ".join(f"{k[2:]}:{v:+.0f}" for k, v in s["monthly"].items()))

    # ── 보고용 분해 ─────────────────────────────────────────────────────────────
    print("\n## 보고용 — 보유 1분 (셀별, 판정 아님)")
    print(breakdown(ev, "ret_cell", holds=(1,)).to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
    ev["gap_dec"] = ev.groupby("date")["gap"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    print("\n## 보고용 — 갭 10분위(09:00 체결 뒤 아는 값, 진입 조건 아님) × 보유")
    gd = breakdown(ev, "gap_dec")
    piv = gd.pivot(index="gap_dec", columns="hold", values="mean")
    piv.columns = [f"{h}분 평균" for h in piv.columns]
    piv["갭 중앙(bp)"] = ev.groupby("gap_dec")["gap"].median() * 1e4
    piv["n"] = ev.groupby("gap_dec").size()
    print(piv.to_string(float_format=lambda x: f"{x:+.1f}"))
    print("\n## 보고용 — 상한가성(전일 종가=고가 ∧ 전일 수익률 ≥28%) × 보유")
    print(breakdown(ev, "limit_up").to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
    print("\n## 보고용 — 시장 × 보유")
    print(breakdown(ev, "market").to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
    print("\n## 보고용 — 셀 × 시장, 보유 5·15분")
    print(ev.groupby(["ret_cell", "market"])[["r5", "r15"]].agg(["mean", "count"])
          .mul([1e4, 1, 1e4, 1]).to_string(float_format=lambda x: f"{x:+.1f}"))
    pd.concat([gd.assign(by="gap_dec"), breakdown(ev, "limit_up").assign(by="limit_up"),
               breakdown(ev, "market").assign(by="market")]).to_csv(out / "breakdown.csv", index=False)
    print(f"\n→ {out}/events.pkl · cells.json · breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
