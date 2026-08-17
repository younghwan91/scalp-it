"""`swing-closebet` — 종가베팅 규칙 하나를 10년 일봉에 돌려본다.

    swing-closebet --start 2016-09-09 --end 2026-08-14 \\
        --rule tv_ratio:3.0: --rule close_pos:0.8: --exit open

규칙은 ``컬럼:최소:최대`` 이고 비우면 무제한이다 (``tv_ratio:3.0:`` = 3 이상).
규칙 탐색은 이 도구의 일이 아니다 — 사전등록한 규칙 하나를 정직하게 재는 게
전부다.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .adapters.db import connect, load_ohlcv
from .closebet.backtest import DEFAULT_COST_GRID, run_close_bet
from .closebet.features import close_bet_features
from .closebet.labels import close_bet_labels

_EXIT_MAP = {"open": "ret_open", "high": "ret_high", "close": "ret_close"}


def parse_rule(spec: str) -> tuple[str, tuple[float | None, float | None]]:
    """``"tv_ratio:3.0:"`` → ``("tv_ratio", (3.0, None))``."""
    parts = spec.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise ValueError(f"규칙 형식은 '컬럼:최소:최대' 다: {spec!r}")
    col = parts[0].strip()

    def _bound(text: str) -> float | None:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"규칙 경계가 숫자가 아니다: {spec!r}") from exc

    lo, hi = _bound(parts[1]), _bound(parts[2])
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"최소가 최대보다 크다: {spec!r}")
    return col, (lo, hi)


def parse_rules(specs: list[str]) -> dict[str, tuple[float | None, float | None]]:
    return dict(parse_rule(s) for s in specs or [])


def format_summary(result: dict[str, Any], rules: dict, args: argparse.Namespace) -> str:
    """사람이 읽는 요약. 숫자를 숨기지 않는다."""
    rule_txt = ", ".join(
        f"{c}[{'' if lo is None else lo}~{'' if hi is None else hi}]"
        for c, (lo, hi) in rules.items()
    ) or "(무조건)"
    lines = [
        "=" * 62,
        f"종가베팅  {args.start} ~ {args.end}   청산={result['exit_col']}",
        f"규칙: {rule_txt}" + (f"   top_n={args.top_n}" if args.top_n else ""),
        "=" * 62,
        (f"거래 {result['n_trades']:,}건 / {result['n_days']:,}일"
         f"  (하루 평균 {result['trades_per_day']:.2f}건)"),
    ]
    if result["n_trades"] == 0:
        lines.append("조건을 만족하는 거래가 없다.")
        lines.append(f"다음날 봉 없어 제외: {result['n_excluded_no_next_day']:,}건")
        return "\n".join(lines)

    lines += [
        (f"평균 {result['mean']:+.4%}   중앙 {result['median']:+.4%}"
         f"   승률 {result['win_rate']:.2%}   손익비 {result['payoff']:.3f}"),
        (f"분포  p5 {result['p5']:+.2%}  p25 {result['p25']:+.2%}"
         f"  p75 {result['p75']:+.2%}  p95 {result['p95']:+.2%}"),
        "",
        "비용 민감도 (왕복 차감 후 평균):",
    ]
    for cost, net in result["cost_grid"].items():
        mark = "  <-- 기본" if abs(float(cost) - result["cost_round_trip"]) < 1e-12 else ""
        lines.append(f"  비용 {float(cost):.2%}  ->  {net:+.4%}{mark}")

    lines += [
        "",
        (f"상한가 의심 (수익률 > 29%): {result['n_limit_up']:,}건"
         f" ({result['limit_up_share']:.3%})"
         f" · 제외 시 평균 {result['mean_ex_limit_up']:+.4%}"),
        f"다음날 봉 없어 제외(거래정지·상폐): {result['n_excluded_no_next_day']:,}건",
        "",
        "연도별:",
    ]
    for year in sorted(result["by_year"]):
        stat = result["by_year"][year]
        lines.append(f"  {year}  평균 {stat['mean']:+.4%}  {stat['n']:,}건")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="종가베팅 백테스트 (D 종가 매수 → D+1 청산)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--exit", dest="exit_kind", default="open",
                        choices=sorted(_EXIT_MAP), help="청산 방식 (기본 다음날 시가)")
    parser.add_argument("--rule", action="append", default=[],
                        help="'컬럼:최소:최대'. 반복 지정. 비우면 무제한")
    parser.add_argument("--top-n", type=int, default=None,
                        help="날짜별 tv_ratio 상위 N 만")
    parser.add_argument("--cost", type=float, default=0.0023, help="왕복 비용")
    parser.add_argument("--adv-window", type=int, default=20, help="평균 거래대금 창")
    parser.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rules = parse_rules(args.rule)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001 — DSN 미설정·드라이버별 접속 오류를 한데 잡는다
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1

    try:
        print(f"일봉 로딩 {args.start} ~ {args.end} ...", file=sys.stderr)
        ohlcv = load_ohlcv(conn, args.start, args.end)
    finally:
        conn.close()

    print(f"  {len(ohlcv):,}행 · {ohlcv['code'].nunique():,}종목", file=sys.stderr)
    features = close_bet_features(ohlcv, adv_window=args.adv_window)
    labels = close_bet_labels(ohlcv)

    result = run_close_bet(
        features, labels,
        rules=rules, top_n=args.top_n,
        cost_round_trip=args.cost, exit_col=_EXIT_MAP[args.exit_kind],
        cost_grid=DEFAULT_COST_GRID,
    )
    result["start"] = args.start
    result["end"] = args.end
    result["rules"] = {c: [lo, hi] for c, (lo, hi) in rules.items()}
    result["top_n"] = args.top_n
    result["n_rows_loaded"] = len(ohlcv)

    print(format_summary(result, rules, args))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\nJSON 저장: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
