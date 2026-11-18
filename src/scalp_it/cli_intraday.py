"""`scalp-intraday` — 키움 분봉을 TimescaleDB ``minute_bars`` 에 수집한다.

한 콜에 900행(≈2.5 거래일)이고 기본 40 페이지면 종목당 약 100 거래일이다.
종목당 40콜이라 100종목이면 4,000콜 — sleep 0.12s 기준 8분쯤 걸린다.

같은 명령을 다시 돌리면 이미 있는 최신 봉 이후만 받는다. 중단됐던 수집을
그냥 다시 실행하면 이어붙는다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

import httpx

from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
from .intraday.store import (
    collection_universe,
    connect,
    count_bars,
    init_minute_bars,
    latest_ts,
    upsert_minute_bars,
)


def resolve_codes(args: argparse.Namespace, conn: Any) -> list[str]:
    """``--codes`` 또는 ``--universe-top`` 으로 대상 종목을 정한다."""
    if args.codes:
        return [c.strip() for c in args.codes.split(",") if c.strip()]
    if args.universe_top:
        # 상위 N 만 받으면 순위 밖으로 밀린 종목이 영영 멈춘다 — 최근 수집분을 합친다.
        # 그날 감지기가 구독한 종목(신규상장·대장 스캔)도 받는다 — 다음 날 08:55 기준가가 이 분봉이다.
        tick_day = None if args.no_tick_codes else date.today()  # noqa: DTZ011 — KST naive
        return collection_universe(conn, args.universe_top,
                                   lookback_days=args.lookback_days,
                                   keep_days=args.keep_days, tick_day=tick_day)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="키움 분봉 수집 (ka10080)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--codes", default=None, help="쉼표 구분 종목코드")
    target.add_argument("--universe-top", type=int, default=None,
                        help="거래대금 상위 N 종목 (daily_bars_adjusted 최근 20일 평균) "
                             "+ 최근 --keep-days 거래일 안에 분봉을 받은 종목")
    parser.add_argument("--lookback-days", type=int, default=20,
                        help="상위 종목 선정에 쓸 거래일 수")
    parser.add_argument("--keep-days", type=int, default=20,
                        help="--universe-top 에서 순위 밖이어도 계속 받는 '최근 수집' 거래일 수")
    parser.add_argument("--no-tick-codes", action="store_true",
                        help="--universe-top 에 오늘 ticks 구독 종목(신규상장·장중 편입)을 더하지 않는다")
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
    parser.add_argument("--max-consecutive-failures", type=int, default=20,
                        help="연속 실패가 이만큼 쌓이면 키움 장애로 보고 멈춘다(rc=1)")
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
        streak = 0
        aborted = None
        for i, code in enumerate(codes, 1):
            stop_at = None if args.no_resume else latest_ts(conn, code)
            try:
                rows = kiwoom.minute_chart(
                    code, tic_scope=args.tic_scope, pages=args.pages,
                    since=args.since, stop_at=stop_at,
                )
            except (KiwoomError, httpx.HTTPError) as exc:
                # 키움 게이트웨이 502·연결 끊김은 httpx 예외로 올라온다(KiwoomError 아님).
                # 예전엔 그게 루프 밖으로 새서 2026-11-06 1,921/2,557 에서 수집 전체가
                # 멈췄다. 한 종목 실패로 남은 종목을 버리지 않는다 — 실패 종목은 다음
                # 실행이 이어받기(stop_at)로 채운다. 단 장애가 계속되면 멈춘다 — 키움은
                # 5xx 를 재시도하지 않고 콜마다 30초 타임아웃이라 2,500종목을 다 돌면
                # 다음 날 08:55 감지기(같은 앱키)와 겹칠 수 있다. 인증 오류는 곧장 멈춘다.
                failed.append(code)
                streak += 1
                print(f"  [{i}/{len(codes)}] {code}: 실패 — {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403):
                    aborted = f"인증 오류 HTTP {status}"
                    break
                if streak >= args.max_consecutive_failures:
                    aborted = f"연속 {streak}종목 실패"
                    break
                continue
            streak = 0
            n = upsert_minute_bars(conn, rows)
            total += n
            span = f"{rows[-1]['ts']:%Y-%m-%d} ~ {rows[0]['ts']:%Y-%m-%d}" if rows else "없음"
            print(f"  [{i}/{len(codes)}] {code}: 신규 {n:,}행 · {span}", file=sys.stderr)

        print(f"\n적재 {total:,}행 · 테이블 총 {count_bars(conn):,}행", file=sys.stderr)
        if failed:
            print(f"실패 {len(failed)}종목: {','.join(failed)}", file=sys.stderr)
        if aborted:
            print(f"수집 중단 — {aborted}. 남은 종목은 다음 실행이 이어받는다.", file=sys.stderr)
        print(total)
        return 1 if aborted or (failed and total == 0) else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
