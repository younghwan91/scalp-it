"""`scalp-theme-snapshot` — 키움 테마 목록·구성종목을 오늘 날짜 스냅샷으로 저장한다.

키움 테마 API(``ka90001`` / ``ka90002``)는 현재 소속만 준다. 짝꿍매매 검증에
필요한 "그날 같은 테마였나" 는 **매일 저장해야만** 룩어헤드 없이 복원된다. 그래서
이 명령은 매일 장 마감 후 한 번 돌리는 용도다.

테마 목록은 100행 페이지 두 번이면 끝나고, 구성종목은 테마당 한 콜이다.
2026-08-23 실측은 테마 142개 · 테마-종목 903행 · 약 145콜 · sleep 0.3s 로 54초였다
(``qry_tp`` · ``date_tp`` · ``flu_pl_amt_tp`` · ``stex_tp`` 를 바꿔도 142개로 같았다).
분봉 수집기와 같은 API 키를 쓰므로 sleep 을 0.3 아래로 내리지 않는다.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from .adapters.kiwoom import KiwoomClient, KiwoomError, api_credentials
from .theme.store import (
    connect,
    count_groups,
    count_members,
    init_theme_tables,
    normalize_snapshot_date,
    upsert_theme_groups,
    upsert_theme_members,
)

#: 스냅샷 날짜는 한국 거래일 기준이다. 서버 tz 가 무엇이든 KST 로 찍는다.
_KST = ZoneInfo("Asia/Seoul")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="키움 테마 그룹·구성종목 일일 스냅샷 (ka90001/ka90002)")
    parser.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    parser.add_argument("--date", default=None,
                        help="스냅샷 날짜 YYYYMMDD 또는 YYYY-MM-DD. 기본 오늘")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="콜 사이 대기(초). 분봉 수집기와 키를 공유하므로 0.3 이상")
    parser.add_argument("--stex-tp", default="1", help="거래소 구분. 1=KRX")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서부터 N개 테마만 (점검용)")
    parser.add_argument("--dry-run", action="store_true",
                        help="테마 목록만 받아 출력하고 구성종목 호출·DB 쓰기를 하지 않는다")
    return parser


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        snapshot_date = normalize_snapshot_date(args.date or datetime.now(tz=_KST).date())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        appkey, secret = api_credentials()
    except KiwoomError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    kiwoom = KiwoomClient(appkey, secret, client=client, sleep=args.sleep)

    started = time.monotonic()
    try:
        groups = kiwoom.theme_groups(stex_tp=args.stex_tp)
    except KiwoomError as exc:
        print(f"테마 목록 실패: {exc}", file=sys.stderr)
        return 1
    if args.limit:
        groups = groups[: max(0, int(args.limit))]
    print(f"스냅샷 {snapshot_date} · 테마 {len(groups)}개 · sleep {args.sleep}s", file=sys.stderr)

    if args.dry_run:
        for g in groups:
            print(f"  {g['theme_code']} {g['theme_name']} ({g['stk_num']}종목)", file=sys.stderr)
        print(len(groups))
        return 0

    if not groups:
        print("테마가 하나도 없다. 저장하지 않는다.", file=sys.stderr)
        return 1

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001 — DSN 미설정·드라이버별 접속 오류를 한데 잡는다
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1

    try:
        init_theme_tables(conn)
        upsert_theme_groups(conn, snapshot_date, groups)

        total_members = 0
        failed: list[str] = []
        for i, g in enumerate(groups, 1):
            code = g["theme_code"]
            try:
                members = kiwoom.theme_members(code, stex_tp=args.stex_tp)
            except KiwoomError as exc:
                failed.append(code)
                print(f"  [{i}/{len(groups)}] {code} {g['theme_name']}: 실패 — {exc}",
                      file=sys.stderr)
                continue
            n = upsert_theme_members(conn, snapshot_date, members)
            total_members += n
            if i % 25 == 0 or i == len(groups):
                print(f"  [{i}/{len(groups)}] 누적 종목 {total_members:,}행", file=sys.stderr)

        elapsed = time.monotonic() - started
        print(
            f"\n{snapshot_date} 테마 {count_groups(conn, snapshot_date):,}개 · "
            f"테마-종목 {count_members(conn, snapshot_date):,}행 적재 "
            f"(전체 테마 {count_groups(conn):,} · 종목 {count_members(conn):,}) · "
            f"{elapsed:.0f}s",
            file=sys.stderr,
        )
        if failed:
            print(f"실패 {len(failed)}테마: {','.join(failed)}", file=sys.stderr)
        print(total_members)
        return 1 if failed and total_members == 0 else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
