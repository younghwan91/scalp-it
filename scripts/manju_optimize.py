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
    trail: float | None = None    # 고점 대비 이만큼 밀리면 청산(추세 추종). None = 끔
    # 트레일링을 **언제 켜느냐**. 진입 즉시 켜면 진 적이 있다(승률 60%→40%) —
    # 2등주 자체 가격 노이즈가 되돌림 문턱을 그냥 스치기 때문이다. 만쥬가 말한 건
    # "2등주 +3% 도달 시 트레일링"이라 **수익이 난 뒤에야** 켜지는 물건이다.
    trail_arm: float = 0.0        # 고점이 진입가 대비 이만큼 오른 뒤에만 트레일링
    # 진입 슬리피지(틱). 만쥬는 "현재가 위 2~3호가 지정가 원클릭"으로 사고 호가가
    # 튈 땐 +1~2틱 불리하게 체결된다고 했다. 분봉 종가 진입 가정은 그만큼 낙관이다.
    entry_slip_ticks: float = 0.0
    # 대장이탈을 알아챈 뒤 **얼마나 빨리 나가느냐**. 분봉으론 초 단위를 못 본다.
    #   close = 그 분 종가 = 최대 60초 지연(비관, 기본)
    #   open  = 그 분 시가 = 즉시 반응(낙관 상한 — 실전 하한이 아니라 상한이다)
    #   mid   = 둘의 중간(중립 추정)
    # 만쥬는 호가 훼손을 보고 초 안에 나간다. 실제값은 open 과 close 사이 어딘가다.
    lead_exit_at: str = "close"
    require_held: bool = False    # 대장이 끝까지 굳힌 건만
    # 만쥬의 "2·3등주"는 **그 테마(섹터) 안에서의 순위**다. 수집기가 넣어준
    # ``ret_rank`` 는 시장 전체 순위라서 그대로 쓰면 "시장 전체 2위이면서 마침 같은
    # 섹터인 종목"이 되어 만쥬 규칙이 아니게 되고 표본도 통째로 날아간다.
    #   group    = 후보군 안에서 다시 매긴 순위(만쥬)
    #   universe = 시장 전체 순위(수집기 원값)
    rank_mode: str = "group"

    def key(self) -> str:
        return (f"{self.pick}/r{self.rank_max}/h{self.hold}"
                f"/sl{'' if self.stop is None else f'{self.stop:.3f}'}"
                f"/tp{'' if self.take is None else f'{self.take:.3f}'}"
                f"/{'LE' if self.leader_exit else 'noLE'}"
                f"{'' if self.trail is None else f'/tr{self.trail:.3f}'}"
                f"{'' if not self.trail_arm else f'@{self.trail_arm:.3f}'}"
                f"{'' if not self.entry_slip_ticks else f'/es{self.entry_slip_ticks:g}'}"
                f"{'/held' if self.require_held else ''}"
                f"{'' if self.rank_mode == 'group' else '/uni'}")


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
    elif rule.rank_mode == "group":
        # 후보군을 등락률로 다시 세운다 — 만쥬가 보는 건 테마 안의 서열이다.
        ranked = sorted(
            (c for c in picks if c in info),
            key=lambda c: -float((info.get(c) or {}).get("ret") or float("-inf")),
        )
        for i, code in enumerate(ranked, start=1):
            # 1위(대장 다음으로 가장 오른 것)는 이미 뛴 것이라 뺀다.
            if 2 <= i <= rule.rank_max:
                fol = code
                break
    else:
        for code in picks:
            meta = info.get(code) or {}
            rank = meta.get("ret_rank")
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
    # 살 때도 스프레드를 넘어 산다 — 위 호가를 긁는 주문이라 불리하게 체결된다.
    entry += rule.entry_slip_ticks * slip
    # 대장이 상한가를 놓친 분(`lead_brk`)을 진입 시점 기준 오프셋으로.
    brk = ev.get("lead_brk")
    brk_off = None
    if rule.leader_exit and brk is not None and np.isfinite(brk):
        off = int(brk) - int(ev["T"])
        if off >= 0:
            brk_off = off

    self_at = rule.lead_exit_at
    last = min(rule.hold, a.shape[1] - 1)
    exit_px, reason = None, "시간청산"
    peak = entry
    for i in range(1, last + 1):
        hi, lo, close = a[H][i], a[L][i], a[C][i]
        if np.isfinite(hi):
            peak = max(peak, float(hi))
        if rule.take is not None and np.isfinite(hi) and hi >= entry * (1 + rule.take):
            exit_px, reason = entry * (1 + rule.take), "익절"   # 지정가라 레벨 그대로
            break
        if rule.stop is not None and np.isfinite(lo) and lo <= entry * (1 + rule.stop):
            # 손절은 스프레드를 넘어 나간다 — 레벨이 아니라 슬리피지를 얹는다.
            exit_px, reason = entry * (1 + rule.stop) - slip, "손절"
            break
        if brk_off is not None and i >= brk_off:
            op = a[O][i]
            if self_at == "open" and np.isfinite(op):
                px = float(op)
            elif self_at == "mid" and np.isfinite(op) and np.isfinite(close):
                px = (float(op) + float(close)) / 2
            else:
                px = close if np.isfinite(close) else entry
            exit_px, reason = px - slip, "대장이탈"
            break
        # 추세 추종 — "상승 탄력이 충분히 나왔다고 판단되는 시점에 매도"의 기계화.
        # 시계를 보고 파는 게 아니라 **오버슈팅이 꺾일 때** 판다. 고점 대비 되돌림이
        # 문턱을 넘으면 나간다. 진입가 위에 있을 때만 발동한다(손절은 별개 축).
        if (rule.trail is not None and peak >= entry * (1 + rule.trail_arm)
                and peak > entry
                and np.isfinite(lo) and lo <= peak * (1 - rule.trail)):
            exit_px, reason = peak * (1 - rule.trail) - slip, "추세청산"
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
        # 만쥬가 준 실전 지표는 승률 45~55% · **손익비 2.5~4:1** 이다. 승률은 우리와
        # 맞는데 손익비가 1.3~1.4 로 갈린다 — 오버슈팅을 못 먹고 있다는 신호다.
        "avg_win_pct": float(g[g > 0].mean() * 100) if (g > 0).any() else 0.0,
        "avg_loss_pct": float(g[g <= 0].mean() * 100) if (g <= 0).any() else 0.0,
        "reasons": df["reason"].value_counts().to_dict(),
    }
    for name, cost in COSTS.items():
        out[f"{name}_pct"] = float((g - cost).mean() * 100)
    aw, al = out["avg_win_pct"], abs(out["avg_loss_pct"])
    out["pl_ratio"] = float(aw / al) if al > 0 else float("nan")
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
                            for tr in args.trails:
                                for rm in args.rank_modes:
                                    for arm in (args.trail_arms if tr else [0.0]):
                                        grid.append(Rule(
                                            pick=pick, rank_max=rank, hold=hold,
                                            stop=stop, take=take, leader_exit=le,
                                            trail=tr, trail_arm=arm, rank_mode=rm,
                                            entry_slip_ticks=args.entry_slip_ticks))
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
    p.add_argument("--trails", nargs="+", type=float, default=[0.0],
                   help="고점 대비 되돌림 청산(추세 추종). 0 = 끔")
    p.add_argument("--trail-arms", nargs="+", type=float, default=[0.0],
                   help="이만큼 오른 뒤에만 트레일링을 켠다(만쥬: +3%). 0 = 즉시")
    p.add_argument("--entry-slip-ticks", type=float, default=0.0,
                   help="진입 슬리피지(틱). 만쥬 실전은 위 호가를 긁어 1~2틱 불리")
    p.add_argument("--select-by", default="net64",
                   choices=["gross", "net34", "net64", "net94"],
                   help="트레인에서 무엇으로 최선 셀을 고를지. 기본은 세후 64bp")
    p.add_argument("--rank-modes", nargs="+", default=["group"],
                   choices=["group", "universe"],
                   help="2·3등주 순위를 테마 안(group)에서 볼지 시장 전체(universe)에서 볼지")
    p.add_argument("--min-n", type=int, default=30, help="이보다 적은 셀은 후보 제외")
    a = p.parse_args(argv)
    a.stops = [None if s == 0 else s for s in a.stops]
    a.takes = [None if t == 0 else t for t in a.takes]
    a.trails = [None if t == 0 else t for t in a.trails]

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
