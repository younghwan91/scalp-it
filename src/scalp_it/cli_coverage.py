"""``scalp-coverage`` — 그날 급등 이벤트 중 우리가 본 비율을 남긴다.

**왜 매일 재나.** 만쥬 방식은 대장이 튀는 순간을 잡는 것이고, 그러려면 그 대장이
튀기 전에 구독돼 있어야 한다. 못 붙으면 신호도 주문도 체결도 없다 — 2026-08-26~28
사흘 내리 신호 0건이 그거였다(그 사흘 시장 상한가는 4·8·7건).

이 숫자가 없으면 슬롯 선점(08-31)이 효과가 있었는지도 감으로 판단하게 된다.
다음 레버(0D 를 REST 폴링으로 빼서 구독 50→100)를 고를 근거도 이 숫자다.

문턱을 여러 개 주는 게 기본이다. 상한가(+29.5%)는 다 놓쳤어도 만쥬 낙점 문턱
(+12%)은 절반 봤을 수 있고, **그 차이가 다음에 뭘 고칠지를 말해준다.**
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any

from .realtime.coverage import (
    Coverage,
    fetch_first_ticks,
    fetch_names,
    fetch_surge_rows,
    format_report,
    measure_coverage,
    surge_events,
)

#: 만쥬 낙점 문턱 — 대장스캔이 후보로 올리는 등락률(`--leader-scan-min-rt 12`).
MANJU_ADOPTION_RATE = 0.12
#: 상한가 — 전일종가 대비 +29.5%.
LIMIT_UP_RATE = 0.295


def run_day(conn: Any, *, day: str, market: str, min_rates: list[float],
            before: str | None) -> list[dict]:
    """하루치를 문턱마다 재서 **기록용 dict 목록**으로 돌려준다.

    DB 를 한 번만 읽고 문턱만 바꿔 다시 센다 — 같은 하루를 문턱마다 다시 조회하면
    느리기만 하고 답은 같다.
    """
    rows = fetch_surge_rows(conn, day=day, market=market)
    first = fetch_first_ticks(conn, day=day)
    out: list[dict] = []
    for rate in min_rates:
        events = surge_events(rows, min_rate=rate)
        cov = measure_coverage(events, first, before=before)
        out.append({
            "day": day, "market": market, "min_rate": rate, "before": before,
            "n_bars": len(rows),
            "n_events": cov.n_events, "n_covered": cov.n_covered,
            "rate": cov.rate, "n_subscribed": len(first),
            "covered": cov.covered, "missed": cov.missed,
        })
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scalp-coverage",
        description="그날 급등 이벤트 중 우리가 구독하고 있던 비율")
    p.add_argument("--day", default=None,
                   help="측정일 YYYY-MM-DD (기본: 오늘)")
    p.add_argument("--market", default="코스닥")
    # append 가 아니라 nargs 인 이유: append 는 기본값에 **덧붙어서** 기본 문턱을
    # 끌 수 없다. 기본값이 파서에 박혀 있어야 --help 로도 보인다.
    p.add_argument("--min-rate", type=float, nargs="+", default=[LIMIT_UP_RATE],
                   help="이벤트 문턱. 여럿 주면 문턱마다 잰다 "
                        f"(기본 {LIMIT_UP_RATE} 상한가, 흔히 {MANJU_ADOPTION_RATE} 낙점)")
    p.add_argument("--before", default=None,
                   help="이 시각(HH:MM:SS)까지 붙어 있던 구독만 인정 — "
                        "만쥬는 오전에 대장을 낙점한다")
    p.add_argument("--dsn", default=None, help="DB DSN (기본: 환경변수)")
    p.add_argument("--jsonl", default=None, help="이 파일에 한 줄씩 append")
    return p


def main(argv: list[str] | None = None) -> int:
    from .adapters.db import connect

    args = build_parser().parse_args(argv)
    day = args.day or date.today().isoformat()  # noqa: DTZ011 — KST naive
    conn = connect(args.dsn)
    rows = run_day(conn, day=day, market=args.market,
                   min_rates=args.min_rate, before=args.before)
    names = fetch_names(conn, sorted({c for r in rows for c in r["missed"]}))
    for row in rows:
        cov = Coverage(n_events=row["n_events"], n_covered=row["n_covered"],
                       covered=row["covered"], missed=row["missed"])
        print(format_report(cov, day=day, min_rate=row["min_rate"], names=names,
                            n_bars=row["n_bars"]))
        print()
    if args.jsonl:
        with open(args.jsonl, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
