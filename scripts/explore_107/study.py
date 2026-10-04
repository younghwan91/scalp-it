"""107번 마감 30분·단일가 탐색 — 시간대 분해(보고) + 사전 지정 6셀(판정). 문서 §2 고정.

    deploy/research_run.sh .venv/bin/python -W ignore scripts/explore_107/study.py

출력: 표준출력(마크다운 표) + ``~/of80/close107_study/{results.json, tables.md}``.
시행 원장 ``research/logs/107-close-window/TRIALS.jsonl`` 에 셀마다 config 를 적는다.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
from build import CACHE, bad_days  # noqa: E402
from krx_quant_core.stats import cluster_bootstrap_ci, matched_alpha, matched_control, record_trial  # noqa: E402

OUT = Path.home() / "of80" / "close107_study"
LABEL = "107-close-window"
N_PER, N_BOOT, SEED = 5, 2000, 107
STRATA = ["date", "vol_q", "tv_q"]
BUCKETS = ("top", "bottom", "mid")
LEGS = {"a": "14:50→15:20 (연속, 41.5bp)", "b": "15:20→15:30 종가 (단일가, 39.8/41.3bp)"}


def load() -> pd.DataFrame:
    bad = bad_days()
    parts = [pickle.loads(f.read_bytes()) for f in sorted(CACHE.glob("*.pkl")) if f.stem not in bad]
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["p_vol20", "p_tv", "market"])
    lab = c.labels(df[["09:00", *c.PRICE_TIMES, "15:30"]], df.market)
    df = df.join(lab)
    g = df.groupby("date", sort=False)
    df["cond_b"] = g.cond.transform(c.tail_bucket)
    df["pret_b"] = g.p_ret1.transform(c.tail_bucket)
    df["vol_q"] = g.p_vol20.transform(c.quartile)
    df["tv_q"] = g.p_tv.transform(c.quartile)
    df["month"] = df.date.str[:7]
    return df


def monthly_sign(y: pd.Series, month: pd.Series) -> tuple[int, int]:
    m = y.groupby(month).mean().dropna()
    return int((m > 0).sum()), int(len(m))


def breakdown(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """측정 1 — 30분 구간 평균(bp, 비용 전) × {전체, 전일 수익률 상위 10%, 하위 10%} + 월별 양수 비율."""
    r = c.interval_returns(df[["09:00", *c.PRICE_TIMES, "15:30"]])
    r["date"], r["month"], r["pret_b"] = df.date, df.month, df.pret_b
    rows = []
    for col in r.columns[: len(c.INTERVALS)]:
        row = {"구간": col}
        for name, mask in (("전체", slice(None)), ("전일상위10%", r.pret_b == "top"), ("전일하위10%", r.pret_b == "bottom")):
            s = r.loc[mask, [col, "month"]].dropna()
            pos, tot = monthly_sign(s[col], s.month)
            row[f"{name} bp"] = s[col].mean()
            row[f"{name} 월+"] = f"{pos}/{tot}"
            row[f"{name} n"] = len(s)
        rows.append(row)
    t = pd.DataFrame(rows)
    lines = ["| 구간 | 전체 bp | 월+ | 상위10% bp | 월+ | 하위10% bp | 월+ | n(전체) |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for _, x in t.iterrows():
        lines.append(f"| {x['구간']} | {x['전체 bp']:+.1f} | {x['전체 월+']} | {x['전일상위10% bp']:+.1f} | {x['전일상위10% 월+']} "
                     f"| {x['전일하위10% bp']:+.1f} | {x['전일하위10% 월+']} | {x['전체 n']:,} |")
    return t, lines


def cell(df: pd.DataFrame, leg: str, bucket: str) -> dict:
    y = f"y_{leg}"
    pool = df.dropna(subset=[y, *STRATA])
    trades = pool[pool.cond_b == bucket]
    ctrl = matched_control(trades, pool, strata=STRATA, n_per=N_PER, seed=SEED)
    ma = matched_alpha(trades, pool, ctrl, value=y, cluster="date", n_boot=N_BOOT, seed=SEED)
    lo, hi, _ = cluster_bootstrap_ci(trades[y].to_numpy(), trades.date.to_numpy(), n_boot=N_BOOT, seed=SEED)
    pos, tot = monthly_sign(trades[y], trades.month)
    cost = c.COST_A if leg == "a" else trades.cost_b
    raw = float((trades[y] + cost).mean())
    config = {"leg": leg, "entry": c.ENTRY_A if leg == "a" else c.ENTRY_B, "exit": c.EXIT_A if leg == "a" else "15:30",
              "cond": f"09:00→{c.COND_AT}", "bucket": bucket, "tail": c.TAIL, "top_n": c.TOP_N,
              "cost": c.COST_A if leg == "a" else c.COST_B, "strata": STRATA, "n_per": N_PER, "seed": SEED,
              "price_src": "minute_bars 09:00 open / 15:30 close"}   # 첫 실행(일봉 open/close)은 자릿수 사고로 폐기
    fp = record_trial(LABEL, config, logs_dir=ROOT / "research" / "logs", note="탐색 6셀")
    return {"leg": leg, "bucket": bucket, "n": int(len(trades)), "days": int(trades.date.nunique()),
            "mean": float(trades[y].mean()), "median": float(trades[y].median()), "pos": float((trades[y] > 0).mean()),
            "raw_mean": raw, "ci_lo": lo, "ci_hi": hi, "alpha": ma.alpha, "alpha_lo": ma.ci_low, "alpha_hi": ma.ci_high,
            "p_greater": ma.p_greater, "unmatched": ma.n_unmatched, "month_pos": pos, "months": tot, "fingerprint": fp}


def main() -> int:
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)
    days = df.date.nunique()
    head = [f"표본: {len(df):,}행 · {days}거래일 · {df.date.min()}~{df.date.max()} · 종목 {df.code.nunique():,}",
            f"결측(5분 안 체결 없음): 14:50 {df['14:50'].isna().mean():.1%} · 15:20 {df['15:20'].isna().mean():.1%}",
            f"일봉 open/close 가 분봉과 1% 이상 어긋나는 행: {df.daily_mismatch.mean():.2%} "
            f"(종목 {df.loc[df.daily_mismatch, 'code'].nunique()}개)"]
    print("\n".join(head))
    t1, lines1 = breakdown(df)
    print("\n".join(lines1))
    cells = [cell(df, leg, b) for leg in LEGS for b in BUCKETS]
    lines2 = ["| 셀 | 다리 | 조건(09:00→14:50) | n | 일수 | 평균 | 중앙 | 양(+) | 비용 전 | 날짜 CI | **널 alpha** | **alpha CI** | 월+ |",
              "|--:|---|---|--:|--:|--:|--:|--:|--:|---|--:|---|--:|"]
    for i, r in enumerate(cells, 1):
        lines2.append(f"| {i} | {LEGS[r['leg']]} | {r['bucket']} | {r['n']:,} | {r['days']} | {r['mean']:+.1f} | {r['median']:+.1f} "
                      f"| {r['pos']:.1%} | {r['raw_mean']:+.1f} | [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {r['alpha']:+.1f} "
                      f"| [{r['alpha_lo']:+.1f}, {r['alpha_hi']:+.1f}] | {r['month_pos']}/{r['months']} |")
    print("\n".join(lines2))
    closed = all(r["alpha_lo"] <= 0 for r in cells)
    verdict = f"기각 기준(6셀 전부 널 alpha CI 하한 ≤ 0): {'충족 — 107 닫힘' if closed else '미충족 — 하한 > 0 셀 있음'}"
    print(verdict)
    (OUT / "tables.md").write_text("\n".join(head + ["", *lines1, "", *lines2, "", verdict]) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"head": head, "breakdown": t1.to_dict("records"), "cells": cells,
                                                    "closed": closed}, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
