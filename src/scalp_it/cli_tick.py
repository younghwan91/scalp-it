"""``scalp-tick-collect`` — 키움 WebSocket 틱·호가 실시간 수집 데몬.

전일 이벤트 후보(거래대금 상위 + 전일 급등주)를 장전에 뽑아 0B(체결)·0D(호가)로
구독하고, 배치로 ``ticks``·``quotes`` 에 쌓는다. 히스토리 소급이 안 되므로(20번
문서) 지금부터 실시간으로 쌓는 것이 유일한 경로다.

예시::

    # 장전 08:50 기동, 15:40 종료. 전일 이벤트 후보 80종목 자동 선정.
    scalp-tick-collect --auto 80 --duration 08:50-15:40

    # 특정 종목만
    scalp-tick-collect --codes 005930,000660 --duration 09:00-15:30

    # 장 마감 후라도 연결·인증·구독·파싱 경로 검증(저장 안 함)
    scalp-tick-collect --codes 005930 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import date, datetime, time
from typing import Any

from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
from .realtime.collector import (
    WS_URL,
    WS_URL_MOCK,
    KiwoomRealtimeCollector,
    dry_run,
)
from .realtime.store import (
    connect,
    count_quotes,
    count_ticks,
    event_candidates,
    init_realtime_tables,
)


def parse_duration(spec: str | None) -> float | None:
    """``HH:MM-HH:MM`` 를 '지금부터 종료시각까지 초'로 바꾼다.

    시작 시각이 미래여도 지금 붙는다(장전 구독은 무해). 종료시각이 이미 지났으면
    0 을 준다. ``None`` 이면 무기한(Ctrl+C 까지).
    """
    if not spec:
        return None
    try:
        _start, end = spec.split("-", 1)
        hh, mm = (int(x) for x in end.strip().split(":", 1))
    except ValueError as exc:
        raise SystemExit(f"--duration 은 HH:MM-HH:MM 형식이어야 한다: {spec!r}") from exc
    now = datetime.now()  # noqa: DTZ005 — KST 로컬시각, 종료시각 계산용
    deadline = datetime.combine(now.date(), time(hh, mm))
    return max(0.0, (deadline - now).total_seconds())


def resolve_codes(args: argparse.Namespace, conn: Any) -> list[str]:
    if args.codes:
        return [c.strip() for c in args.codes.split(",") if c.strip()]
    if args.auto:
        return event_candidates(conn, args.auto)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="키움 WebSocket 틱·호가 수집 (0B/0D)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--codes", default=None, help="쉼표 구분 종목코드")
    target.add_argument("--auto", type=int, default=None, metavar="N",
                        help="전일 이벤트 후보 N종목 자동 선정(거래대금 상위 + 전일 급등주)")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument("--duration", default=None,
                        help="HH:MM-HH:MM. 종료시각까지 돈다. 생략 시 Ctrl+C 까지")
    parser.add_argument("--types", default="0B,0D",
                        help="구독 실시간 type (기본 0B,0D)")
    parser.add_argument("--flush-interval", type=float, default=1.5,
                        help="배치 저장 간격(초)")
    parser.add_argument("--mock", action="store_true",
                        help="모의투자 WebSocket 엔드포인트 사용")
    parser.add_argument("--dry-run", action="store_true",
                        help="구독만 하고 저장 안 함. REAL 프레임 몇 개 찍고 종료")
    parser.add_argument("--dry-frames", type=int, default=20,
                        help="--dry-run 에서 찍을 최대 REAL 프레임 수")
    parser.add_argument("--dry-timeout", type=float, default=30.0,
                        help="--dry-run 대기 한도(초)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.codes and not args.auto:
        print("--codes 또는 --auto 중 하나가 필요하다.", file=sys.stderr)
        return 2

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    url = WS_URL_MOCK if args.mock else WS_URL

    # --dry-run 은 DB 가 필요없다(저장 안 함). --codes 만 있으면 DB 접속을 건너뛴다.
    conn = None
    need_db = not (args.dry_run and args.codes)
    if need_db:
        try:
            conn = connect(args.dsn)
        except Exception as exc:  # noqa: BLE001
            print(f"DB 접속 실패: {exc}", file=sys.stderr)
            return 1

    try:
        if conn is not None and not args.dry_run:
            init_realtime_tables(conn)

        codes = resolve_codes(args, conn)
        if not codes:
            print("대상 종목이 없다.", file=sys.stderr)
            return 1
        print(f"대상 {len(codes)}종목 · type {types} · {url}", file=sys.stderr)

        try:
            appkey, secret = api_credentials()
        except KiwoomError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 1
        token = KiwoomClient(appkey, secret).token

        if args.dry_run:
            seen = asyncio.run(dry_run(
                token, codes, types=types, url=url,
                max_frames=args.dry_frames, timeout=args.dry_timeout,
            ))
            return 0 if seen >= 0 else 1

        duration = parse_duration(args.duration)
        collector = KiwoomRealtimeCollector(
            token, codes, conn=conn, types=types, url=url,
            flush_interval=args.flush_interval, trade_date=date.today(),  # noqa: DTZ011
        )
        print(f"커넥션 {collector.n_connections}개 · 종료까지 "
              f"{'무기한' if duration is None else f'{duration/60:.0f}분'}", file=sys.stderr)

        _run_with_signals(collector, duration)
        print(f"틱 {collector.ticks_stored:,} · 호가 {collector.quotes_stored:,} · "
              f"테이블 총 틱 {count_ticks(conn):,} · 호가 {count_quotes(conn):,}",
              file=sys.stderr)
        return 0
    finally:
        if conn is not None:
            conn.close()


def _run_with_signals(collector: KiwoomRealtimeCollector, duration: float | None) -> None:
    """SIGINT/SIGTERM 에 graceful flush. cron 종료(15:40)도 SIGTERM 이다."""
    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, collector.stop)
            except (NotImplementedError, ValueError):
                pass  # 일부 플랫폼·비메인스레드
        await collector.run(duration=duration)

    asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
