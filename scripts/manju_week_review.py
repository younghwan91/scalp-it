#!/usr/bin/env python3
"""이번 주 복기 — 만쥬 규칙대로 했으면 얼마 벌었나.

시스템 점검이 아니라 **돈** 을 묻는 복기다. 그 주 분봉으로 만쥬 짝꿍매매를 그대로
재현하고, 규칙을 하나씩 바꿔가며 무엇이 돈을 만들었는지 본다.

만쥬 규칙(문서 45 대조표 그대로):

    대장    당일 상한가에 **닿은** 종목. 방아쇠는 상한가까지 3호가 남은 분.
    짝꿍    같은 테마(없으면 27종 섹터) 안에서 그 시점 등락률 2·3위. 이미 잠긴 건 뺀다.
    청산    대장이 상한가를 놓치면 손익 무관 전량 · +3% 뒤 트레일링 1.5%
            · 손절 −0.5%(대장이 굳히는 동안은 잠금) · 재난선 −2% · 3분 타임스탑

진입가는 **다음 분 시가**다. 방아쇠 분 종가에 산다고 하면 그 분 안에서 이미 오른 걸
공짜로 먹는 낙관이 된다(문서 42 §5).

    export $(grep -v '^#' ../quant-airflow/.env | xargs)
    uv run --extra pg python scripts/manju_week_review.py --start 2026-08-24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from limitup_timing import limit_up_price

from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect
from scalp_it.ticksize import tick_size

EPS = 1e-6
OPEN_M, CLOSE_M = 9 * 60, 15 * 60 + 30
LAST_TRIG = 15 * 60 + 10
NEAR_TICKS = 3
COSTS = {"net34": 0.0034, "net64": 0.0064}


def q(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    out = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    return out


def groups_of(conn, codes: list[str]) -> dict[str, str]:
    """종목 → 무리키. 실전 테마가 있으면 테마, 없으면 27종 섹터(감지기와 같은 규칙)."""
    sec = dict(q(conn, "select code, sector from stocks").itertuples(index=False))
    try:
        tm = q(conn, "select code, theme_code from theme_members where snapshot_date="
                     "(select max(snapshot_date) from theme_members)")
        theme: dict[str, str] = {}
        for code, t in tm.itertuples(index=False):
            theme.setdefault(str(code), str(t))
            theme[str(code)] = min(theme[str(code)], str(t))
    except Exception:  # noqa: BLE001 — 테마가 없어도 섹터로 돈다
        theme = {}
    return {c: (f"T:{theme[c]}" if c in theme else f"S:{sec.get(c, '')}") for c in codes}


def simulate_day(bars: pd.DataFrame, groups: dict[str, str], rule: dict) -> list[dict]:
    """하루치 분봉 → 체결 결과들."""
    bars = bars[bars["minute"] <= CLOSE_M]
    cw = bars.pivot(index="minute", columns="code", values="close")
    ow = bars.pivot(index="minute", columns="code", values="open")
    hw = bars.pivot(index="minute", columns="code", values="high")
    lw = bars.pivot(index="minute", columns="code", values="low")
    prev = bars.groupby("code")["prev_close"].first()
    cwf = cw.ffill()
    limit = pd.Series({c: limit_up_price(float(prev[c])) for c in cw.columns
                       if np.isfinite(prev.get(c, np.nan)) and prev[c] > 0})
    cw = cw[[c for c in cw.columns if c in limit.index]]
    if cw.empty:
        return []
    ret = cwf[cw.columns].div(prev[cw.columns], axis=1) - 1
    near = pd.Series({c: limit[c] - NEAR_TICKS * tick_size(float(prev[c]))
                      for c in cw.columns})

    out: list[dict] = []
    taken: set[tuple[str, int]] = set()
    for lead in cw.columns:
        hi = hw[lead]
        # 방아쇠: 상한가 3호가 이내에 **닿은 첫 분**. 그 분에 이미 잠겼어도 만쥬는
        # 그 직전 몇 초에 사므로 다음 분 시가 진입으로 근사한다.
        trig = hi[(hi >= near[lead] - EPS)]
        trig = trig[trig.index <= LAST_TRIG]
        if trig.empty:
            continue
        T = int(trig.index[0])
        if (T + 1) not in ow.index:
            continue
        # 짝꿍 — 같은 무리에서 그 시점 등락률 2·3위, 이미 잠긴 건 제외
        key = groups.get(lead)
        if not key:
            continue
        peers = [c for c in cw.columns if groups.get(c) == key]
        r_T = ret.loc[T, peers].dropna().sort_values(ascending=False)
        picked = None
        for rank, code in enumerate(r_T.index, start=1):
            if code == lead or rank < 2 or rank > rule["rank_max"]:
                continue
            if float(cwf.at[T, code]) >= limit[code] - EPS:
                continue                          # 이미 잠김 — 못 산다
            picked = code
            break
        if picked is None or (picked, T) in taken:
            continue
        taken.add((picked, T))
        entry = float(ow.at[T + 1, picked])
        if not np.isfinite(entry) or entry <= 0:
            continue
        # 대장이 상한가 근처를 벗어나는 분 = 대장이탈
        after = cwf[lead].loc[T + 1:]
        gone = after[after < near[lead] - EPS]
        brk = int(gone.index[0]) if len(gone) else None
        slip = tick_size(entry)
        peak, exit_px, reason = entry, None, "시간청산"
        held_lock = False
        last = min(T + rule["hold"], int(cw.index.max()))
        for m in range(T + 1, last + 1):
            if m not in cw.index:
                continue
            h, low, cl = hw.at[m, picked], lw.at[m, picked], cw.at[m, picked]
            if np.isfinite(h):
                peak = max(peak, float(h))
            lead_at = float(cwf.at[m, lead]) if m in cwf.index else np.nan
            held_lock = np.isfinite(lead_at) and lead_at >= limit[lead] - EPS
            if brk is not None and m >= brk:
                exit_px, reason = (float(cl) if np.isfinite(cl) else entry) - slip, "대장이탈"
                break
            if (rule["trail"] and peak >= entry * (1 + rule["trail_arm"])
                    and np.isfinite(low) and low <= peak * (1 - rule["trail"])):
                exit_px, reason = peak * (1 - rule["trail"]) - slip, "추세청산"
                break
            if (rule["stop"] and not (rule["stop_lock"] and held_lock)
                    and np.isfinite(low) and low <= entry * (1 + rule["stop"])):
                exit_px, reason = entry * (1 + rule["stop"]) - slip, "손절"
                break
            if (rule["disaster"] and np.isfinite(low)
                    and low <= entry * (1 + rule["disaster"])):
                exit_px, reason = entry * (1 + rule["disaster"]) - slip, "재난선"
                break
        if exit_px is None:
            cl = cw.at[last, picked] if last in cw.index else np.nan
            exit_px = float(cl) if np.isfinite(cl) else entry
        # 대장이 그날 상한가를 **굳혔는가**(잠긴 뒤 종가까지 유지). 진입 시점엔 모르는
        # 값이라 필터로 쓰면 미래를 보는 것이지만, "굳힐 대장만 골랐다면 얼마였나"의
        # 상한을 재는 데는 쓸 수 있다.
        tail_lead = cwf[lead].loc[T + 1:]
        locked_mins = int((tail_lead >= limit[lead] - EPS).sum())
        held_close = bool(len(tail_lead) and tail_lead.iloc[-1] >= limit[lead] - EPS)
        tv_lead = float(bars.loc[(bars["code"] == lead) & (bars["minute"] <= T),
                                 "trade_value"].sum()) if "trade_value" in bars else 0.0
        out.append({"lead": lead, "fol": picked, "T": T, "reason": reason,
                    "gross": exit_px / entry - 1.0, "lead_locked_mins": locked_mins,
                    "lead_held": held_close, "lead_tv": tv_lead})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="이번 주 만쥬 복기")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    a = ap.parse_args(argv)
    conn = connect(os.environ.get("KR_QUANT_DB"))
    end = a.end or a.start
    days = [d.strftime("%Y-%m-%d") for d in
            pd.to_datetime(q(conn, "select distinct date from daily_bars where date>=%s "
                                   "and date<=%s order by date", (a.start, end))["date"])]
    print(f"# 거래일 {len(days)}: {', '.join(days)}", file=sys.stderr)

    base = {"rank_max": 3, "hold": 3, "stop": -0.005, "stop_lock": True,
            "disaster": -0.02, "trail": 0.015, "trail_arm": 0.03}
    variants = {
        "만쥬 그대로": base,
        "  손절 잠금 끔": {**base, "stop_lock": False},
        "  트레일링 끔": {**base, "trail": 0.0},
        "  보유 10분": {**base, "hold": 10},
        "  보유 30분": {**base, "hold": 30},
        "  손절 없음": {**base, "stop": 0.0},
        "  2등주만(rank2)": {**base, "rank_max": 2},
    }
    per_day: dict[str, pd.DataFrame] = {}
    for ds in days:
        bars = load_minute_bars(conn, ds, ds)
        if bars.empty:
            continue
        bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
        prev = q(conn, "select code, close from daily_bars where date=(select max(date) "
                       "from daily_bars where date < %s)", (ds,))
        bars = bars.merge(prev.rename(columns={"close": "prev_close"}), on="code", how="left")
        bars["prev_close"] = pd.to_numeric(bars["prev_close"], errors="coerce")
        per_day[ds] = bars
    groups = groups_of(conn, sorted({c for b in per_day.values() for c in b["code"]}))

    print(f"\n{'설정':18s} {'건수':>4s} {'승률':>6s} {'평균':>8s} {'중앙':>8s} "
          f"{'net34':>8s} {'합계(1천만×1주기준)':>10s}")
    for name, rule in variants.items():
        rows: list[dict] = []
        for ds, bars in per_day.items():
            for r in simulate_day(bars, groups, rule):
                r["date"] = ds
                rows.append(r)
        if not rows:
            print(f"{name:18s}    0")
            continue
        g = pd.DataFrame(rows)["gross"]
        net = g - COSTS["net34"]
        print(f"{name:18s} {len(g):4d} {(g > 0).mean():5.0%} {g.mean() * 100:+7.3f}% "
              f"{g.median() * 100:+7.3f}% {net.mean() * 100:+7.3f}% "
              f"{net.sum() * 10_000_000:>+12,.0f}원")
    # 사유 분포는 기준 설정으로만
    rows = []
    for ds, b in per_day.items():
        for r in simulate_day(b, groups, base):
            r["date"] = ds
            rows.append(r)
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n[만쥬 그대로] 청산 사유: {df['reason'].value_counts().to_dict()}")
        print(f"  일자별 건수: {df.groupby('date').size().to_dict()}")
        best = df.nlargest(5, "gross")[["date", "lead", "fol", "gross", "reason"]]
        worst = df.nsmallest(5, "gross")[["date", "lead", "fol", "gross", "reason"]]
        print("\n  가장 잘된 5건:")
        for r in best.itertuples(index=False):
            print(f"    {r.date} {r.lead}→{r.fol} {r.gross * 100:+.2f}% ({r.reason})")
        print("  가장 나쁜 5건:")
        for r in worst.itertuples(index=False):
            print(f"    {r.date} {r.lead}→{r.fol} {r.gross * 100:+.2f}% ({r.reason})")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
