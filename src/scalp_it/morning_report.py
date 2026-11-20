"""아침 리포트 — 만쥬 체인이 실제로 도는지 한 장으로 본다.

로그가 여러 개로 흩어져 있어 아침마다 다섯 군데를 열어야 했다. 이 리포트가 답하는
질문은 하나다: **어제(또는 오늘) 만쥬의 방아쇠가 몇 번 보였나.**

    방아쇠 구간 = 대장이 상한가 호가를 3개 남긴 구간, 아직 안 잠김

2026-08-24~28 에 이 구간에 틱이 들어온 종목이 **2개**였다. 10~20 이 돼야 만쥬 규칙이
처음으로 제대로 시험대에 오른다. 구독 종목 수·수집 채움·신호·주문은 그 지표가 왜
그 값인지 설명하는 보조 지표다.

    uv run --extra pg python -m scalp_it.morning_report --date today
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Any

from .realtime.pair_detector import limit_up_price
from .regular_session import base_prices
from .ticksize import tick_size


def in_trigger_zone(*, price: float, prev_close: float, ticks: int = 3) -> bool:
    """만쥬의 방아쇠 구간인가 — 상한가 ``ticks`` 호가 이내이되 **아직 안 닿음**.

    상한가에 닿은 뒤는 방아쇠가 지나간 것이다. 만쥬는 잠긴 대장을 사지 않고,
    우리 선정도 미잠김 우선이다(`1fefb38`).
    """
    if prev_close <= 0 or price <= 0:
        return False
    limit = limit_up_price(prev_close)
    if price >= limit:
        return False
    return price >= limit - ticks * tick_size(limit)


def _rows(conn: Any, sql: str, params: tuple = ()) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def collect(conn: Any, day: date, *, ticks: int = 3) -> dict[str, Any]:
    """그날의 체인 지표를 모은다."""
    prev = _rows(conn, "select max(date) from daily_bars where date < %s", (day,))
    prev_day = prev[0][0] if prev and prev[0][0] else None
    out: dict[str, Any] = {"date": day, "prev_day": prev_day}

    out["ticks"], out["codes"] = (_rows(
        conn, "select count(*), count(distinct code) from ticks where ts::date = %s",
        (day,)) or [(0, 0)])[0]
    out["max_concurrent"] = (_rows(
        conn,
        "select coalesce(max(n), 0) from (select date_trunc('minute', ts) m,"
        " count(distinct code) n from ticks where ts::date = %s group by 1) t",
        (day,)) or [(0,)])[0][0]

    if prev_day is None:
        out["zone_codes"] = out["zone_hits"] = 0
        return out

    # 방아쇠 구간 — 그날 틱의 최고가가 상한가 3호가 이내이면서 상한가 미만인 종목.
    # 상한가의 기준은 그날 가격제한폭 기준가(전일 정규장 종가)다. 전일 일봉 종가를 직접 쓰면
    # 2026-09-14 이후 애프터마켓 종가라 상한가가 틀린다 — 정문 함수로만 구한다.
    highs = _rows(
        conn,
        "select code, max(price::numeric) from ticks where ts::date = %s group by code",
        (day,))
    bases = base_prices(conn, day, [str(c) for c, _ in highs])
    rows = [(str(c), float(hi), float(bases[str(c)])) for c, hi in highs
            if hi is not None and str(c) in bases]
    zone = [(c, hi, pc) for c, hi, pc in rows
            if in_trigger_zone(price=hi, prev_close=pc, ticks=ticks)]
    out["zone_codes"] = len(zone)
    out["zone_list"] = sorted(c for c, _, _ in zone)
    out["subscribed_touched_limit"] = sum(
        1 for _, hi, pc in rows if float(hi) >= limit_up_price(float(pc)))
    return out


def render(m: dict[str, Any], logs: dict[str, str]) -> str:
    day = m["date"]
    zone = m.get("zone_codes", 0)
    verdict = ("체인이 돈다" if zone >= 10 else
               "아직 얇다" if zone >= 3 else "방아쇠를 못 봤다")
    lines = [
        f"# 아침 리포트 — {day}",
        "",
        f"**방아쇠 구간 종목 {zone}개** — {verdict}",
        "",
        "만쥬의 방아쇠는 대장이 상한가 3호가를 남긴 구간이다. 그 구간에 틱이 들어온",
        "종목 수가 이 리포트의 전부다. 2026-08-24~28 에 2개였고 10~20 이 목표다.",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 방아쇠 구간 종목 | **{zone}** |",
        f"| 상한가 도달(구독분) | {m.get('subscribed_touched_limit', 0)} |",
        f"| 구독 종목(하루 누적) | {m.get('codes', 0)} |",
        f"| 동시 구독 최대 | {m.get('max_concurrent', 0)} / 정원 50 |",
        f"| 저장 틱 | {m.get('ticks', 0):,} |",
    ]
    if m.get("zone_list"):
        lines += ["", "방아쇠 구간 종목: " + ", ".join(m["zone_list"][:20])]
    for name, text in logs.items():
        if text.strip():
            lines += ["", f"## {name}", "", "```", text.strip()[-1200:], "```"]
    return "\n".join(lines) + "\n"


def _tail(path: str, n: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def resolve_day(arg: str, today: date, conn: Any, *,
                previous_session: Any = None) -> date | None:
    """``--date`` 를 리포트할 날로 바꾼다.

    ``yesterday`` 는 달력 어제가 아니라 **직전 거래일**이다. 달력 어제로 두면
    월요일 08:30 크론이 일요일을 집계해 "방아쇠 0개" 를 latest.md 에 쓴다
    (2026-11-16 실제로 그랬다). 거래일은 관측된 일봉에서 구한다(휴장일 표 없음).
    """
    if arg == "today":
        return today
    if arg == "yesterday":
        if previous_session is None:
            from .market_regime import previous_session
        day = previous_session(conn, today)
        if day is None:
            return None
        return day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    return date.fromisoformat(arg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="아침 리포트 — 만쥬 체인이 도는지 한 장으로")
    p.add_argument("--date", default="today",
                   help="today | yesterday(직전 거래일) | YYYY-MM-DD")
    p.add_argument("--ticks", type=int, default=3, help="방아쇠 구간 호가 수(만쥬 §2 = 3)")
    p.add_argument("--out", default=None, help="파일로도 남긴다")
    args = p.parse_args(argv)

    today = datetime.now().date()  # noqa: DTZ005 — KST naive(프로젝트 표준)

    from .intraday.store import connect

    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        print("KR_QUANT_DB 없음", file=sys.stderr)
        return 2
    conn = connect(dsn)
    try:
        day = resolve_day(args.date, today, conn)
        if day is None:
            print("일봉에서 직전 거래일을 못 찾았다", file=sys.stderr)
            return 1
        print(f"리포트 대상일 {day} (--date {args.date})", file=sys.stderr)
        metrics = collect(conn, day, ticks=args.ticks)
    finally:
        conn.close()

    logs = {
        "헬스체크": _tail("data/tick_health.log", 10),
        "수집 채움 · 대장 스캔": "\n".join(
            ln for ln in _tail("data/pair_alerts.log", 400).splitlines()
            if "수집채움" in ln or "대장스캔" in ln)[-1200:],
        "감지기": _tail("data/pair_detect.log", 12),
    }
    text = render(metrics, logs)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
