#!/usr/bin/env python3
"""짝꿍 dry-run 일일 성과 집계 — data/pair_positions.log 를 읽어
그날(또는 --date) 청산된 가상 포지션의 승률·손익·MFE/MAE·역선택(through vs touch)을 요약하고
data/pair_daily_summary.log 에 한 줄씩 누적한다. 며칠 쌓이면 엣지가 양(+)인지 한눈에 보인다."""
from __future__ import annotations
import json, sys, statistics as st
from collections import Counter
from datetime import date

def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else str(date.today())  # noqa: DTZ011
    closes = []
    try:
        for line in open("data/pair_positions.log", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "close" and str(r.get("ts", "")).startswith(day):
                closes.append(r)
    except FileNotFoundError:
        print(f"[{day}] 포지션 로그 없음")
        return 0
    if not closes:
        print(f"[{day}] 청산 0건")
        return 0

    def stats(rows, key="pnl"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        krw = [r.get("pnl_krw", 0) or 0 for r in rows]
        if not vals:
            return None
        wins = sum(1 for v in vals if v > 0)
        return dict(n=len(vals), win=wins, wr=wins/len(vals)*100,
                    mean=st.mean(vals)*100, total_krw=sum(krw))

    # 체결 가정별(보수 through / 낙관 touch)
    filled_through = [r for r in closes if r.get("filled_through")]
    filled_touch = [r for r in closes if r.get("filled_touch")]
    reasons = Counter(r.get("exit_reason") for r in closes)
    mae = [r["mae"]*100 for r in closes if r.get("mae") is not None]
    mfe = [r["mfe"]*100 for r in closes if r.get("mfe") is not None]

    parts = [f"[{day}] 청산 {len(closes)}건"]
    for label, rows in (("through(보수)", filled_through), ("touch(낙관)", filled_touch)):
        s = stats(rows)
        if s:
            parts.append(f"{label}: n{s['n']} 승률{s['wr']:.0f}% 평균{s['mean']:+.3f}% 누적{s['total_krw']:+.0f}원")
    parts.append(f"청산사유 {dict(reasons)}")
    if mae:
        parts.append(f"MAE중앙{st.median(mae):+.2f}% MFE중앙{st.median(mfe):+.2f}%")
    summary = " | ".join(parts)
    print(summary)
    with open("data/pair_daily_summary.log", "a", encoding="utf-8") as f:
        f.write(summary + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
