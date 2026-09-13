"""틱 스캘핑 확정 판정 — 사전등록 `docs/research/manju/28` §0 "며칠~몇 주 더 쌓은 뒤 별도 문서".

`tick_sanity.py` 는 하루치 sanity(참고용)만 낸다. 이 스크립트는 여러 거래일을 하나의
`Accum` 에 누적해 **풀링된** 트리거 vs 비트리거·분위·판정을 한 번만 낸다 — 신호·라벨·
판정선은 28번에서 고정된 그대로, 날짜만 여러 개 돈다. 파라미터를 바꾸지 않는다.

사용:
  export $(grep -v '^#' ../quant-airflow/.env | xargs)
  uv run --extra pg python scripts/tick_sanity_confirm.py --dates data/tick_sanity_2026*.md
  uv run --extra pg python scripts/tick_sanity_confirm.py --from 2026-08-24 --to 2026-09-04

DB 는 읽기만 한다. 산출물은 `data/tick_sanity_confirm_<from>_<to>.md` + stdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tick_sanity import (
    Accum,
    _load_env,
    build_report,
    load_date_ticks,
    process_code,
)


def _dates_from_reports(patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        for p in sorted(pathlib.Path().glob(pat)):
            m = re.search(r"(\d{8})", p.name)
            if m:
                d = m.group(1)
                out.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    return sorted(set(out))


def _dates_from_range(start: str, end: str) -> list[str]:
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    out = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:  # 평일만 — 공휴일은 codes 조회가 빈 딕셔너리로 스킵됨
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="틱 스캘핑 여러 날 풀링 확정 판정 (읽기 전용)")
    ap.add_argument(
        "--dates", nargs="*", default=None,
        help="tick_sanity_YYYYMMDD.md 글롭 패턴(들). 지정 시 --from/--to 무시.",
    )
    ap.add_argument("--from", dest="date_from", default=None, help="시작일 YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="종료일 YYYY-MM-DD")
    ap.add_argument("--env", default=None, help=".env 경로(선택)")
    ap.add_argument("--out-dir", default="data", help="리포트 출력 폴더")
    args = ap.parse_args(argv)

    if args.dates:
        dates = _dates_from_reports(args.dates)
    elif args.date_from and args.date_to:
        dates = _dates_from_range(args.date_from, args.date_to)
    else:
        ap.error("--dates 또는 --from/--to 중 하나는 필요")

    if not dates:
        print("[tick_sanity_confirm] 대상 날짜 없음. 종료.")
        return 0

    _load_env(args.env)
    from scalp_it.intraday.store import connect

    conn = connect()
    acc = Accum()
    used_dates: list[str] = []
    try:
        for date in dates:
            day_data = load_date_ticks(conn, date)
            if not day_data:
                print(f"[tick_sanity_confirm] {date}: ticks 없음. 스킵.")
                continue
            used_dates.append(date)
            for code, data in day_data.items():
                process_code(acc, data, code)
    finally:
        conn.close()

    if not used_dates:
        print("[tick_sanity_confirm] 유효한 날짜 없음. 종료.")
        return 0

    label = f"{used_dates[0]} ~ {used_dates[-1]} ({len(used_dates)}일 풀링)"
    report = build_report(acc, label, pooled_days=len(used_dates))
    print(report)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{used_dates[0].replace('-', '')}_{used_dates[-1].replace('-', '')}"
    out = out_dir / f"tick_sanity_confirm_{tag}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[tick_sanity_confirm] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
