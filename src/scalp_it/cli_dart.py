"""`scalp-dart` — OpenDART 공시를 로컬 sqlite 에 수집한다.

``corp_code`` 없는 공시검색은 기간이 3개월로 제한되므로 구간을 쪼개 돈다.
하루 1,000~1,500건, ``page_count=100`` 이므로 10년치가 약 30,000 콜이다.
인증키 3개면 하루 안에 끝난다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from krx_news_client import quarter_ranges

from .adapters.dart import DartClient, DartError, api_keys
from .dart.store import collected_range, init_dart_db, rows_to_frame, upsert_disclosures

# DART 는 ``corp_code`` 없는 검색을 **달력 3개월**로 센다. 90일 고정으로 쪼개면 짧은
# 달(2월)을 낀 구간에서 `100 corp_code가 없는 경우 검색기간은 3개월만 가능합니다` 로
# 죽는다(10년 수집이 21/80 구간에서 중단된 적이 있다). 쪼개기는
# ``krx_news_client.quarter_ranges`` 가 한다(말일 넘침 클램프 포함, 이 파일의 pandas
# 구현을 옮긴 것). ``scalp_it.cli_dart.quarter_ranges`` 로도 계속 import 할 수 있다.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenDART 공시 수집")
    parser.add_argument("--start", default="20160901")
    parser.add_argument("--end", default=None, help="생략 시 오늘")
    parser.add_argument("--db", default="data/dart.db")
    parser.add_argument("--corp-cls", default="Y,K", help="쉼표 구분. Y=유가 K=코스닥")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true",
                        help="구간만 출력하고 호출하지 않는다")
    args = parser.parse_args(argv)

    end = args.end or pd.Timestamp.today().strftime("%Y%m%d")
    ranges = quarter_ranges(args.start, end)
    classes = [c.strip() for c in args.corp_cls.split(",") if c.strip()]
    print(f"구간 {len(ranges)}개 × 법인구분 {len(classes)}개 = {len(ranges)*len(classes)} 회 조회",
          file=sys.stderr)

    if args.dry_run:
        for bgn, stop in ranges:
            print(f"  {bgn} ~ {stop}", file=sys.stderr)
        return 0

    try:
        keys = api_keys()
    except DartError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    dart = DartClient(keys, sleep=args.sleep)
    conn = init_dart_db(args.db)
    total = 0
    try:
        for bgn, stop in ranges:
            for cls in classes:
                rows = dart.search_all(bgn_de=bgn, end_de=stop, corp_cls=cls)
                n = upsert_disclosures(conn, rows_to_frame(rows))
                total += n
                print(f"  {bgn}~{stop} {cls}: 수신 {len(rows):,} · 적재 {n:,}", file=sys.stderr)
    except DartError as exc:
        print(f"\n중단: {exc}", file=sys.stderr)
        print(f"여기까지 {total:,}건 적재됨. 같은 명령을 다시 돌리면 이어받는다.",
              file=sys.stderr)
        return 1
    finally:
        span = collected_range(conn)
        conn.close()

    print(f"\n적재 {total:,}건 · 범위 {span}", file=sys.stderr)
    print(f"저장: {Path(args.db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
