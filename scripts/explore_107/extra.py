"""107번 보조 기술 — 판정과 무관한 보고용 분해. 6셀·기각 기준은 study.py 가 고정했고 여기서 건드리지 않는다.

    deploy/research_run.sh .venv/bin/python -W ignore scripts/explore_107/extra.py

1. 다리 B(15:20→종가) 원수익이 정확히 0 인 비율 — 마감 단일가가 마지막 체결가와 같은 날이 얼마나 되나.
2. 셀 4(상위 10% × 단일가) 꼬리 — 1%/5% 절사 평균, 분위수. 평균 −110bp 가 꼬리인지 몸통인지.
3. 숏 방향(실행 불가, 보고만): 셀별 **비용 전** 원수익의 월별 부호 — 부호를 뒤집으면 숏의 월별 부호다.
4. 전일 거래대금 상위 300 만으로 시간대 분해 — 97(상위 300, −28bp/일)과 이어 읽기 위해.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
from study import OUT, breakdown, load, monthly_sign  # noqa: E402


def main() -> int:
    df = load()
    lines = []
    raw_b = df.y_b + df.cost_b
    lines.append(f"다리 B 원수익 == 0 비율: 전체 {(raw_b == 0).mean():.1%} · 상위10% {(raw_b[df.cond_b == 'top'] == 0).mean():.1%} "
                 f"· 하위10% {(raw_b[df.cond_b == 'bottom'] == 0).mean():.1%}")
    t4 = raw_b[df.cond_b == "top"].dropna()
    q = t4.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    trim = {p: float(t4[(t4 > t4.quantile(p)) & (t4 < t4.quantile(1 - p))].mean()) for p in (0.01, 0.05)}
    lines.append(f"셀 4 원수익(bp): 평균 {t4.mean():+.1f} · 절사1% {trim[0.01]:+.1f} · 절사5% {trim[0.05]:+.1f} · 분위 "
                 + " ".join(f"p{int(k * 100)}={v:+.0f}" for k, v in q.items()))
    lines.append("")
    lines.append("| 셀 | 다리 | 조건 | 비용 전 평균 | 비용 전 월+ (롱) | → 숏이면 월+ |")
    lines.append("|--:|---|---|--:|--:|--:|")
    i = 0
    for leg in ("a", "b"):
        for b in ("top", "bottom", "mid"):
            i += 1
            m = df.cond_b == b
            raw = (df.y_a + c.COST_A) if leg == "a" else raw_b
            s = raw[m].dropna()
            pos, tot = monthly_sign(s, df.month[s.index])
            lines.append(f"| {i} | {leg} | {b} | {s.mean():+.1f} | {pos}/{tot} | {tot - pos}/{tot} |")
    lines.append("")
    top300 = df[df.groupby("date").p_tv.rank(ascending=False, method="first") <= 300]
    lines.append(f"전일 거래대금 상위 300 만 ({len(top300):,}행):")
    _, l300 = breakdown(top300)
    lines += l300
    oc =(top300["15:30"] / top300["09:00"] - 1) * 1e4
    lines.append(f"상위 300 시가→종가 평균 {oc.mean():+.1f}bp · 월+ {'/'.join(map(str, monthly_sign(oc, top300.month)))} "
                 f"(97 의 10년 평균은 −28.0bp)")
    text = "\n".join(lines)
    print(text)
    (OUT / "extra.md").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
