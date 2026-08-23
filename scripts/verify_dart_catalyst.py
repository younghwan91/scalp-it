"""DART 재료 필터 검증 — 사전등록 `docs/research/manju/21-dart-catalyst-preregistration.md` 그대로 1회.

돌파(09)·상따(06 5차)의 **기존 진입 이벤트를 그대로 import**(재선별 금지)하고, 각 이벤트에
직전 K거래일 재료성 공시 유무(`cat_prior`)를 붙여 **익일 시가 net34** 를 재료有 vs 無 로 가른다.

지표 = 익일 시가 net34.
  상따: 등록된 ret(=익일 시가) 그대로. net34 = ret − 0.0034.
  돌파: 같은 진입 이벤트에 daily_bars(원주가) 익일 시가를 붙여 ret_nextopen 을 만든다(청산 지표만
        문서 질문에 맞춤 — 이벤트 재선별 아님). 등록 주 규칙 ret_stop2(당일청산)도 병기.

판정(각 기법): ①부트스트랩 mean(有)−mean(無) 95%CI 하한>0  ②재료有 월별 음수달 ≤ ⌊n/3⌋(≥20건 달만)
  ③재료有 단독 net34>0 & 상한가 오염 제거 후에도 net34>0.  셋 다 → 채택, 아니면 기각.

DB 는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dart_catalyst_events import (
    K_TRADING_DAYS,
    _load_env,
    load_disclosures,
    tag_events,
    trading_days,
)

COST, COST_HI = 0.0034, 0.0055
N_BOOT = 10000
SEED = 0


def _next_open(conn, start: str, end: str) -> dict[tuple[str, pd.Timestamp], float]:
    """daily_bars 원주가 익일 시가: (code, 진입일) -> 다음 거래일 open."""
    cur = conn.cursor()
    cur.execute("SELECT code, date, open FROM daily_bars WHERE date >= %s AND date <= %s", (start, end))
    d = pd.DataFrame(cur.fetchall(), columns=["code", "date", "open"])
    cur.close()
    d["code"] = d["code"].astype(str).str.strip().str.zfill(6)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["open"] = pd.to_numeric(d["open"], errors="coerce")
    d = d.sort_values(["code", "date"])
    d["nopen"] = d.groupby("code", sort=False)["open"].shift(-1)
    d["ndate"] = d.groupby("code", sort=False)["date"].shift(-1)
    return {(c, dt): no for c, dt, no in zip(d["code"], d["date"], d["nopen"], strict=True)}


def describe(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"n": 0}
    w, ls = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / ls.mean()) if len(w) and len(ls) and ls.mean() != 0 else np.nan
    return {"n": len(r), "mean": r.mean(), "net34": r.mean() - COST, "net55": r.mean() - COST_HI,
            "median": r.median(), "win": (r > 0).mean(), "pl": pl,
            "p5": r.quantile(.05), "p95": r.quantile(.95)}


def _row(tag: str, d: dict) -> str:
    if d.get("n", 0) == 0:
        return f"| {tag} | 0 | | | | | | |"
    return (f"| {tag} | {d['n']:,} | {d['mean']:+.2%} | {d['net34']:+.2%} | {d['net55']:+.2%} | "
            f"{d['median']:+.2%} | {d['win']:.1%} | {d['pl']:.2f} |")


HDR = ("| 구분 | n | 평균 | net34 | net55 | 중앙 | 승률 | 손익비 |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|")


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """mean(a)-mean(b) 점추정과 부트스트랩 95% CI."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    da = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    db = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    diffs = a[da].mean(axis=1) - b[db].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def monthly_neg(ev: pd.DataFrame, col: str, min_n: int = 20) -> tuple[list[str], int, int]:
    lines = ["| 월 | n | net34 |", "|---|---:|---:|"]
    neg = counted = 0
    for m, g in ev.groupby("month"):
        net = g[col].mean() - COST
        used = len(g) >= min_n
        is_neg = used and net < 0
        counted += int(used)
        neg += int(is_neg)
        lines.append(f"| {m} | {len(g):,} | {net:+.2%}{' 음수' if is_neg else ''}"
                     f"{'' if used else ' (미달)'} |")
    return lines, neg, counted


def analyze(name: str, ev: pd.DataFrame, col: str, excl_masks: dict[str, pd.Series],
            out: list[str]) -> bool:
    P = out.append
    ev = ev.copy()
    ev["month"] = pd.to_datetime(ev["date"]).dt.strftime("%Y-%m")
    yes = ev[ev["cat_prior"] == 1]
    no = ev[ev["cat_prior"] == 0]
    P(f"\n# {name} — 재료 필터 (지표: {col})\n")
    P(f"이벤트 {len(ev):,} · 재료有 {len(yes):,} ({len(yes)/len(ev):.1%}) · 재료無 {len(no):,} "
      f"· 당일공시 {int(ev['cat_same'].sum()):,}\n")
    P(HDR)
    P(_row("재료有 (cat_prior)", describe(yes[col])))
    P(_row("재료無", describe(no[col])))
    for lbl, mask in excl_masks.items():
        P(_row(f"재료有 · {lbl}", describe(ev.loc[(ev['cat_prior'] == 1) & mask, col])))
    P(_row("전체(참고)", describe(ev[col])))
    P("")

    # 조건 1 — 부트스트랩 차이
    diff, lo, hi = boot_diff(yes[col].to_numpy(dtype=float), no[col].to_numpy(dtype=float))
    c1 = (diff > 0) and (lo > 0)
    P(f"**조건1** mean(有)−mean(無) = {diff:+.2%}  95%CI [{lo:+.2%}, {hi:+.2%}]  → "
      f"하한>0 {'O' if c1 else 'X'}\n")

    # 조건 2 — 월별
    lines, neg, counted = monthly_neg(yes, col)
    max_neg = counted // 3
    c2 = counted > 0 and neg <= max_neg
    P("재료有 월별 (≥20건 달만 판정):\n")
    out.extend(lines)
    P(f"\n**조건2** 음수 달 {neg}/{counted} ≤ ⌊{counted}/3⌋={max_neg}  → {'O' if c2 else 'X'}\n")

    # 조건 3 — 재료有 단독 양(+), 오염 제거 후에도
    base = describe(yes[col])["net34"]
    excl_ok = True
    excl_report = []
    for lbl, mask in excl_masks.items():
        d = describe(ev.loc[(ev["cat_prior"] == 1) & mask, col])
        v = d.get("net34", float("nan"))
        excl_report.append(f"{lbl} {v:+.2%}")
        if not (d.get("n", 0) > 0 and v > 0):
            excl_ok = False
    c3 = (base > 0) and excl_ok
    P(f"**조건3** 재료有 net34 {base:+.2%}>0 & 오염제거 후 ({'; '.join(excl_report)}) 모두>0 "
      f"→ {'O' if c3 else 'X'}\n")

    verdict = c1 and c2 and c3
    P(f"## 판정 — {name}: **{'채택(전진 기록 착수)' if verdict else '기각'}**  "
      f"[조건1 {'O' if c1 else 'X'}][조건2 {'O' if c2 else 'X'}][조건3 {'O' if c3 else 'X'}]\n")

    # 서브유형 분해 (진단, 판정 아님)
    from collections import defaultdict
    agg = defaultdict(list)
    for subs, r in zip(yes["prior_subtypes"], yes[col], strict=True):
        if not np.isfinite(r):
            continue
        for t in set(str(subs).split("|")):
            if t:
                agg[t].append(r)
    P("서브유형 분해(중복 계수, **진단**):\n")
    P("| 서브유형 | n | net34 | 중앙 | 승률 |\n|---|---:|---:|---:|---:|")
    for t, arr in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        s = pd.Series(arr)
        P(f"| {t} | {len(s)} | {s.mean()-COST:+.2%} | {s.median():+.2%} | {(s>0).mean():.0%} |")
    P("")
    # 상위 20건 눈으로
    top = yes.sort_values(col, ascending=False).head(20)
    P("재료有 상위 20건 (code · date · net34 · subtypes):\n")
    for _, rr in top.iterrows():
        P(f"- {rr['code']} {pd.Timestamp(rr['date']).date()} {rr[col]-COST:+.2%} · {rr['prior_subtypes']}")
    P("")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakout-cache", required=True)
    ap.add_argument("--sangtta-cache", required=True)
    ap.add_argument("--dart-db", action="append", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _load_env(args.env)
    from scalp_it.adapters.db import connect

    disc = load_disclosures(args.dart_db, "20250701", "20260831")
    conn = connect()
    tdays = trading_days(conn, "2025-06-01", "2026-08-31")
    nopen = _next_open(conn, "2025-07-01", "2026-08-31")
    conn.close()

    out: list[str] = ["# DART 재료 필터 검증 — 사전등록 21번 그대로 1회\n"]
    out.append(f"재료성 공시 {len(disc):,}건 · DART 범위 {disc['rcept'].min().date()}"
               f"~{disc['rcept'].max().date()} · K={K_TRADING_DAYS}거래일 · 비용 34/55bp · "
               f"부트스트랩 {N_BOOT}회\n")

    # ---- 상따 (등록 ret = 익일 시가) --------------------------------------------------
    st = pd.read_pickle(args.sangtta_cache)
    st = st["ev"] if isinstance(st, dict) else st
    st = tag_events(st, disc, tdays)
    st_excl = {"상한가의심 제외": st["limit_at_entry"] == 0}
    v_st = analyze("상따(sangtta) 익일시가", st, "ret", st_excl, out)

    # ---- 돌파 (익일 시가 새로 산출 + 등록 당일청산 병기) --------------------------------
    bo = pd.read_pickle(args.breakout_cache)
    bo = bo["ev"] if isinstance(bo, dict) else bo
    bo = tag_events(bo, disc, tdays)
    bo["code"] = bo["code"].astype(str).str.strip().str.zfill(6)
    bo["date"] = pd.to_datetime(bo["date"]).dt.normalize()
    bo["nopen"] = [nopen.get((c, d), np.nan)
                   for c, d in zip(bo["code"], bo["date"], strict=True)]
    bo["ret_nextopen"] = bo["nopen"] / bo["entry"] - 1
    bo_excl = {"상한가마감 제외": bo["is_limit_close"] == 0,
               "ret_at_trigger<20%": bo["ret_at_trigger"] < 0.20}
    v_bo = analyze("돌파(breakout) 익일시가", bo, "ret_nextopen", bo_excl, out)
    # 병기: 등록 주 규칙 당일청산
    out.append("### 돌파 참고 — 등록 주 규칙 ret_stop2(당일청산) 로 본 재료有/無\n")
    out.append(HDR)
    out.append(_row("재료有", describe(bo.loc[bo["cat_prior"] == 1, "ret_stop2"])))
    out.append(_row("재료無", describe(bo.loc[bo["cat_prior"] == 0, "ret_stop2"])))
    out.append("")

    out.append(f"\n## 종합 판정: 상따 **{'채택' if v_st else '기각'}** · "
               f"돌파 **{'채택' if v_bo else '기각'}**\n")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
