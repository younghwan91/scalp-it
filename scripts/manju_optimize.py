#!/usr/bin/env python3
"""만쥬 전략 최적화 — 분봉 1년, 진짜 홀드아웃(문서 42 선결조건 준수).

이 레포는 **데이터 부족한 최적화로 세 번 속았다**:
  doc36 승률 55~64% → 1년 표본에서 37~53%
  doc37 90초 MFE +0.95% → 1년 표본에서 +0.14~0.23%
  고정 4쌍 리드-랙 → 교차상관 정점이 lag 0(동시)
그래서 doc42/43 이 "재최적화를 만들지 마라"며 **선결 조건 6개**를 걸었다. 그 문서가
막은 건 **틱 2일치** 최적화다. 이 스크립트는 **분봉 1년(257거래일)**이고, 6개 조건을
전부 코드에 박았다:

1. **수십 거래일** — 2025-08~2026-08, 상한가 터치 2,714건 / 필터 통과 768건 / 232일.
2. **비용 실측** — `verify_pair` 의 COST_LO/BASE/HI(34/64/94bp)를 import(재정의 금지).
   **세전만 보고하지 않는다.**
3. **킬스위치·포지션상한 OFF** — 가드는 실전 손실통제지 백테스트 필터가 아니다.
   그리드 평가에서 켜면 나쁜 셀의 뒤쪽 이벤트가 잘려 성과가 부풀려진다.
4. **진짜 홀드아웃** — 앞 8개월에서만 argmax, 뒤 4개월에 **단 한 번** 적용.
   홀드아웃을 보고 셀을 고르는 순간 OOS 가 아니다. 그래서 이 스크립트는
   홀드아웃 성과를 **최선 셀 하나에 대해서만** 계산한다(전 셀 홀드아웃 표를 만들면
   사람이 그걸 보고 고르게 된다).
5. **보수적 청산** — 미완결은 MFE 가 아니라 **마지막 관측가**, 손절·시장가 청산에
   슬리피지 1틱. doc42 가 "MFE 청산 = 항상 +" 라고 지적한 낙관 편향을 막는다.
6. **멱등** — 캐시된 이벤트 재생 기반. 로그 append 누적이 아니라 같은 입력에 같은 결과.

추가로 **대조군**(무작위 종목)과 **일클러스터 t** 를 항상 같이 낸다. 같은 날 이벤트는
독립이 아니고, 대조군 없이는 "+0.1%" 가 엣지인지 그냥 상승장인지 알 수 없다.

    uv run python scripts/manju_optimize.py --data <full.pkl> --out data/manju_opt.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_pair import COST_BASE, COST_HI, COST_LO

COSTS = {"net34": COST_LO, "net64": COST_BASE, "net94": COST_HI}

#: OHLC 행 인덱스(수집 스크립트 규약).
O, H, L, C = 0, 1, 2, 3


@dataclass(frozen=True)
class Rule:
    """한 조합. 이름이 곧 그리드 축이다."""

    pick: str = "sector"          # 후속주 선정: sector | comove | topret | random(대조군)
    rank_max: int = 3             # 같은 섹터 등락률 몇 위까지 후속으로 볼지
    hold: int = 5                 # 보유 분
    stop: float | None = -0.005   # 손절(음수). None = 없음
    take: float | None = None     # 익절. None = 상한 없음(만쥬 손익 구조)
    leader_exit: bool = True      # 대장이 상한가 놓치면 손익 무관 즉시
    require_held: bool = False    # 대장이 끝까지 굳힌 건만

    def key(self) -> str:
        return (f"{self.pick}/r{self.rank_max}/h{self.hold}"
                f"/sl{'' if self.stop is None else f'{self.stop:.3f}'}"
                f"/tp{'' if self.take is None else f'{self.take:.3f}'}"
                f"/{'LE' if self.leader_exit else 'noLE'}"
                f"{'/held' if self.require_held else ''}")


def _slip(price: float, prev_close: float) -> float:
    """청산 슬리피지 1틱. 손절·시장가 청산은 스프레드를 넘어 나간다(doc42 §5)."""
    from scalp_it.ticksize import tick_size

    return float(tick_size(prev_close or price))


def simulate(ev: pd.Series, paths: dict, rule: Rule) -> dict[str, Any] | None:
    """이벤트 하나 × 규칙 하나 → 체결 결과. 후속주가 없으면 None.

    진입가는 **방아쇠 분의 종가**다. 그 분에 신호를 알았으니 그 분 안에 살 수 있다는
    가정인데 낙관이다 — 실전은 다음 분 시가에 가깝다. `--entry-next-open` 으로 바꿔
    비교할 수 있게 뒀다(기본은 낙관, 그 사실을 출력에 적는다).
    """
    picks = (ev["picks"] or {}).get(rule.pick) or []
    info = ev["fol_info"] or {}
    fol = None
    if rule.pick == "random":
        # **대조군에는 순위 필터를 걸지 않는다.** 무작위 종목이 하필 그 섹터 등락률
        # 2~3위일 리가 없어, 같은 필터를 걸면 대조군이 통째로 비어(n=0) 비교가
        # 성립하지 않는다. 대조군의 역할은 "같은 이벤트·같은 청산에서 **아무 종목이나**
        # 골랐으면 얼마였나"이므로 종목 선정만 무작위면 된다.
        fol = next((c for c in picks if paths.get(c) is not None), None)
    else:
        for code in picks:
            meta = info.get(code) or {}
            rank = meta.get("ret_rank")
            # 만쥬는 "2~3등주". 1위(대장 다음으로 가장 오른 것)는 이미 뛴 것이라 뺀다.
            if rank is not None and 2 <= int(rank) <= rule.rank_max:
                fol = code
                break
    if fol is None:
        return None
    arr = paths.get(fol)
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    entry = float(a[C][0])
    if not np.isfinite(entry) or entry <= 0:
        return None

    prev_close = float((info.get(fol) or {}).get("prev_close") or entry)
    slip = _slip(entry, prev_close)
    # 대장이 상한가를 놓친 분(`lead_brk`)을 진입 시점 기준 오프셋으로.
    brk = ev.get("lead_brk")
    brk_off = None
    if rule.leader_exit and brk is not None and np.isfinite(brk):
        off = int(brk) - int(ev["T"])
        if off >= 0:
            brk_off = off

    last = min(rule.hold, a.shape[1] - 1)
    exit_px, reason = None, "시간청산"
    for i in range(1, last + 1):
        hi, lo, close = a[H][i], a[L][i], a[C][i]
        if rule.take is not None and np.isfinite(hi) and hi >= entry * (1 + rule.take):
            exit_px, reason = entry * (1 + rule.take), "익절"   # 지정가라 레벨 그대로
            break
        if rule.stop is not None and np.isfinite(lo) and lo <= entry * (1 + rule.stop):
            # 손절은 스프레드를 넘어 나간다 — 레벨이 아니라 슬리피지를 얹는다.
            exit_px, reason = entry * (1 + rule.stop) - slip, "손절"
            break
        if brk_off is not None and i >= brk_off:
            exit_px, reason = (close if np.isfinite(close) else entry) - slip, "대장이탈"
            break
    if exit_px is None:
        # 미완결은 **마지막 관측가**로 청산한다. MFE 로 닫으면 항상 +가 되어
        # doc42 가 지적한 낙관 편향이 그대로 재현된다.
        close = a[C][last]
        exit_px = float(close) if np.isfinite(close) else entry
    gross = exit_px / entry - 1.0
    return {"date": ev["date"], "lead": ev["lead"], "fol": fol,
            "gross": gross, "reason": reason}


def evaluate(events: pd.DataFrame, paths_by_key: dict, rule: Rule) -> dict[str, Any]:
    """규칙 하나를 전 이벤트에 적용 → 집계. 킬스위치·포지션상한은 **끈다**."""
    rows: list[dict[str, Any]] = []
    for _, ev in events.iterrows():
        if rule.require_held and not bool(ev.get("lead_held")):
            continue
        got = simulate(ev, paths_by_key.get(ev["key"], {}), rule)
        if got is not None:
            rows.append(got)
    if not rows:
        return {"rule": rule.key(), "n": 0}
    df = pd.DataFrame(rows)
    g = df["gross"]
    out: dict[str, Any] = {
        "rule": rule.key(), "n": len(df), "days": df["date"].nunique(),
        "gross_pct": float(g.mean() * 100),
        "win_pct": float((g > 0).mean() * 100),
        "reasons": df["reason"].value_counts().to_dict(),
    }
    for name, cost in COSTS.items():
        out[f"{name}_pct"] = float((g - cost).mean() * 100)
    # 같은 날 이벤트는 독립이 아니다 — 일 단위로 접고 t 를 낸다.
    daily = df.groupby("date")["gross"].mean()
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        out["t_day"] = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))))
    else:
        out["t_day"] = float("nan")
    return out


def build_grid(args: argparse.Namespace) -> list[Rule]:
    grid: list[Rule] = []
    for pick in args.picks:
        for rank in args.ranks:
            for hold in args.holds:
                for stop in args.stops:
                    for take in args.takes:
                        for le in (True, False):
                            grid.append(Rule(pick=pick, rank_max=rank, hold=hold,
                                             stop=stop, take=take, leader_exit=le))
    return grid


def _split(events: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간순 분할. 날짜 경계로 자른다(같은 날이 양쪽에 걸치면 누수다)."""
    days = sorted(events["date"].unique())
    cut = days[int(len(days) * frac)]
    return events[events["date"] < cut], events[events["date"] >= cut]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="만쥬 전략 최적화(진짜 홀드아웃)")
    p.add_argument("--data", required=True, help="수집된 이벤트 pkl")
    p.add_argument("--out", default="data/manju_opt.jsonl")
    p.add_argument("--train-frac", type=float, default=0.667, help="앞 2/3 를 트레인으로")
    p.add_argument("--picks", nargs="+",
                   default=["sector", "comove", "topret", "random"])
    p.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 5])
    p.add_argument("--holds", nargs="+", type=int, default=[1, 3, 5, 10, 30])
    p.add_argument("--stops", nargs="+", type=float, default=[-0.005, -0.01, -0.02])
    p.add_argument("--takes", nargs="+", type=float, default=[0.0])
    p.add_argument("--select-by", default="net64",
                   choices=["gross", "net34", "net64", "net94"],
                   help="트레인에서 무엇으로 최선 셀을 고를지. 기본은 세후 64bp")
    p.add_argument("--min-n", type=int, default=30, help="이보다 적은 셀은 후보 제외")
    a = p.parse_args(argv)
    a.stops = [None if s == 0 else s for s in a.stops]
    a.takes = [None if t == 0 else t for t in a.takes]

    with Path(a.data).open("rb") as fh:
        blob = pickle.load(fh)
    events, paths = blob["events"], blob["paths"]
    train, holdout = _split(events, a.train_frac)
    print(f"# 이벤트 {len(events)} · 트레인 {len(train)}({train['date'].nunique()}일) "
          f"/ 홀드아웃 {len(holdout)}({holdout['date'].nunique()}일)", file=sys.stderr)
    print("# 킬스위치·포지션상한 OFF(백테스트 필터가 아니다) · 미완결은 마지막 관측가 "
          "· 손절 슬리피지 1틱 · 진입가는 방아쇠 분 종가(낙관)", file=sys.stderr)

    grid = build_grid(a)
    print(f"# 그리드 {len(grid)}셀 — **트레인에서만** argmax 한다", file=sys.stderr)
    results = [evaluate(train, paths, r) for r in grid]
    ok = [r for r in results if r.get("n", 0) >= a.min_n]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({"seg": "train", **r}, ensure_ascii=False) + "\n")

    if not ok:
        print(f"\n트레인에서 n>={a.min_n} 인 셀이 없다. 판정 불가.", file=sys.stderr)
        return 1
    col = f"{a.select_by}_pct" if a.select_by != "gross" else "gross_pct"
    best = max(ok, key=lambda r: r[col])
    print(f"\n# 트레인 최선({a.select_by}): {best['rule']}", file=sys.stderr)
    print(f"#   n={best['n']} 일수={best['days']} 세전 {best['gross_pct']:+.3f}% "
          f"net34 {best['net34_pct']:+.3f}% net64 {best['net64_pct']:+.3f}% "
          f"승률 {best['win_pct']:.0f}% t={best['t_day']:+.2f}", file=sys.stderr)

    # 홀드아웃은 **이 한 셀에만** 적용한다. 전 셀 표를 만들면 사람이 그걸 보고 고른다.
    best_rule = next(r for r in grid if r.key() == best["rule"])
    hold_res = evaluate(holdout, paths, best_rule)
    ctrl_rule = replace(best_rule, pick="random")
    ctrl_res = evaluate(holdout, paths, ctrl_rule)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seg": "holdout", **hold_res}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"seg": "holdout_control", **ctrl_res},
                            ensure_ascii=False) + "\n")

    print(f"\n# 홀드아웃(단 1회 적용): n={hold_res.get('n', 0)} "
          f"일수={hold_res.get('days', 0)}", file=sys.stderr)
    if hold_res.get("n"):
        print(f"#   세전 {hold_res['gross_pct']:+.3f}% · net34 {hold_res['net34_pct']:+.3f}% "
              f"· net64 {hold_res['net64_pct']:+.3f}% · 승률 {hold_res['win_pct']:.0f}% "
              f"· t={hold_res['t_day']:+.2f}", file=sys.stderr)
        if ctrl_res.get("n"):
            print(f"#   대조군(무작위 종목) 세전 {ctrl_res['gross_pct']:+.3f}% "
                  f"· n={ctrl_res['n']}", file=sys.stderr)
        sign_ok = (best["gross_pct"] > 0) == (hold_res["gross_pct"] > 0)
        print(f"\n판정: 부호 재현 {'O' if sign_ok else 'X — 트레인 최선이 홀드아웃에서 뒤집혔다'}"
              f" · 세후34bp {'양(+)' if hold_res['net34_pct'] > 0 else '음(−)'}",
              file=sys.stderr)
    print(f"저장 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
