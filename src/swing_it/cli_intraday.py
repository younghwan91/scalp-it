"""`swing-intraday` — 키움 분봉을 TimescaleDB ``minute_bars`` 에 수집한다.

한 콜에 900행(≈2.5 거래일)이고 기본 40 페이지면 종목당 약 100 거래일이다.
종목당 40콜이라 100종목이면 4,000콜 — sleep 0.12s 기준 8분쯤 걸린다.

같은 명령을 다시 돌리면 이미 있는 최신 봉 이후만 받는다. 중단됐던 수집을
그냥 다시 실행하면 이어붙는다.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
from .intraday.store import (
    connect,
    count_bars,
    init_minute_bars,
    latest_ts,
    top_universe,
    upsert_minute_bars,
)


def resolve_codes(args: argparse.Namespace, conn: Any) -> list[str]:
    """``--codes`` 또는 ``--universe-top`` 으로 대상 종목을 정한다."""
    if args.codes:
        return [c.strip() for c in args.codes.split(",") if c.strip()]
    if args.universe_top:
        return top_universe(conn, args.universe_top, lookback_days=args.lookback_days)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="키움 분봉 수집 (ka10080)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--codes", default=None, help="쉼표 구분 종목코드")
    target.add_argument("--universe-top", type=int, default=None,
                        help="거래대금 상위 N 종목 (daily_bars_adjusted 최근 20일 평균)")
    parser.add_argument("--lookback-days", type=int, default=20,
                        help="상위 종목 선정에 쓸 거래일 수")
    parser.add_argument("--pages", type=int, default=40, help="종목당 최대 페이지")
    parser.add_argument("--tic-scope", default="1", help="분 단위. 1=1분봉")
    parser.add_argument("--since", default=None,
                        help="YYYYMMDD. 이 날짜 이전 봉이 나오면 그 종목을 멈춘다")
    parser.add_argument("--sleep", type=float, default=0.12, help="콜 사이 대기(초)")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument("--no-resume", action="store_true",
                        help="이미 있는 최신 ts 를 무시하고 처음부터 받는다")
    parser.add_argument("--dry-run", action="store_true",
                        help="대상 종목만 출력하고 호출하지 않는다")
    return parser


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.codes and not args.universe_top:
        print("--codes 또는 --universe-top 중 하나가 필요하다.", file=sys.stderr)
        return 2

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001 — DSN 미설정·드라이버별 접속 오류를 한데 잡는다
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1

    try:
        init_minute_bars(conn)
        codes = resolve_codes(args, conn)
        if not codes:
            print("대상 종목이 없다.", file=sys.stderr)
            return 1
        print(f"대상 {len(codes)}종목 · 최대 {args.pages}페이지 · sleep {args.sleep}s",
              file=sys.stderr)

        if args.dry_run:
            for code in codes:
                print(f"  {code} (기존 최신 {latest_ts(conn, code)})", file=sys.stderr)
            return 0

        try:
            appkey, secret = api_credentials()
        except KiwoomError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 1

        kiwoom = KiwoomClient(appkey, secret, client=client, sleep=args.sleep)
        total = 0
        failed: list[str] = []
        for i, code in enumerate(codes, 1):
            stop_at = None if args.no_resume else latest_ts(conn, code)
            try:
                rows = kiwoom.minute_chart(
                    code, tic_scope=args.tic_scope, pages=args.pages,
                    since=args.since, stop_at=stop_at,
                )
            except KiwoomError as exc:
                failed.append(code)
                print(f"  [{i}/{len(codes)}] {code}: 실패 — {exc}", file=sys.stderr)
                continue
            n = upsert_minute_bars(conn, rows)
            total += n
            span = f"{rows[-1]['ts']:%Y-%m-%d} ~ {rows[0]['ts']:%Y-%m-%d}" if rows else "없음"
            print(f"  [{i}/{len(codes)}] {code}: 신규 {n:,}행 · {span}", file=sys.stderr)

        print(f"\n적재 {total:,}행 · 테이블 총 {count_bars(conn):,}행", file=sys.stderr)
        if failed:
            print(f"실패 {len(failed)}종목: {','.join(failed)}", file=sys.stderr)
        print(total)
        return 1 if failed and total == 0 else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
