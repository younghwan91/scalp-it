"""``scalp-tick-health`` — 실시간 틱·호가 수집 헬스체크.

내일(수집 첫날) 장중 REAL 프레임의 FID 매핑이 실제로 맞는지 자동 검증한다.
``scalp-tick-collect`` 가 쌓은 ``ticks``·``quotes`` 를 **읽기만** 하고, 최근 N분
수집 상태를 6개 항목으로 검사해 사람이 읽는 리포트 + 이상 플래그를 찍고
``data/tick_health.log`` 에 남긴다. 하나라도 실패면 종료코드 비0.

가장 중요한 검사는 **체결 방향 부호**(항목 2)다 — :func:`parse_tick_values` 는
체결량 FID(15)에 부호가 없으면 조용히 +1 로 채운다. 그러면 매수/매도 구분이
통째로 날아가 만쥬 신호의 핵심이 결손된다. side 에 +1·-1 이 둘 다 보이는지로
그 오매핑을 잡아낸다.

검사 항목(각각 OK/경고/실패):

1. **수집 여부** — 최근 N분 ticks·quotes 행 수 > 0, 구독 종목 수 vs 기대치(~80).
   0 이면 실패(연결·인증·구독). 단 장중이 아니면 "수집 없음(장중 아님)" 으로
   정상 처리한다.
2. **체결 방향 부호** — side 에 +1 과 -1 이 둘 다 나오는가. 한쪽뿐이면 실패.
3. **체결강도 분포** — strength 가 상수·전부 NULL 이 아니고 그럴듯한 범위인가.
4. **가격 정합** — tick price 가 그날(없으면 최근) daily_bars 고저 범위/스케일 안,
   best_bid ≤ best_ask.
5. **호가** — bid1<ask1, 총잔량>0, 10호가 채움, ts 갱신(1초 스냅샷), vi_flag 수신.
6. **raw 샘플** — 이상 유형별 raw 원문 1건씩 로그에 덤프(사후 FID 보정용).

운영은 postgres, 테스트는 sqlite fixture 로 판정 로직을 검증한다. 판정 함수는
DB 조회에서 뽑은 집계 dict 만 받는 순수 함수라 fixture 로 그대로 시험할 수 있다.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from .intraday.store import _bind_ts, _placeholder
from .realtime.store import connect

__all__ = ["Check", "Status", "build_parser", "main"]


# --------------------------------------------------------------------- 판정 모델


class Status(enum.Enum):
    """검사 결과 등급. ``rank`` 로 최악 등급·종료코드를 정한다."""

    OK = ("OK", 0)
    WARN = ("경고", 1)
    FAIL = ("실패", 2)

    def __init__(self, label: str, rank: int) -> None:
        self.label = label
        self.rank = rank


@dataclass
class Check:
    """검사 한 건의 결과. ``raw`` 는 이상 시 덤프할 원문 샘플."""

    name: str
    status: Status
    detail: str
    raw: str | None = None


@dataclass
class TickStats:
    """최근 N분 ticks 집계(순수 판정 입력)."""

    n_rows: int = 0
    n_codes: int = 0
    side_counts: dict[int | None, int] = field(default_factory=dict)
    strength_n: int = 0
    strength_null: int = 0
    strength_min: float | None = None
    strength_max: float | None = None
    strength_distinct: int = 0
    bid_gt_ask: int = 0  # best_bid > best_ask 위반 행 수
    price_min: dict[str, int] = field(default_factory=dict)  # code -> min price
    price_max: dict[str, int] = field(default_factory=dict)  # code -> max price
    latest_ts: str | None = None


@dataclass
class QuoteStats:
    """최근 N분 quotes 집계(순수 판정 입력)."""

    n_rows: int = 0
    n_codes: int = 0
    bid_ge_ask: int = 0  # bid1 >= ask1 위반 행 수
    total_qty_zero: int = 0  # 총잔량 <= 0 또는 NULL 행 수
    l10_filled: int = 0  # bid10·ask10 둘 다 채워진 행 수
    vi_seen: int = 0  # vi_flag 비NULL 행 수
    latest_ts: str | None = None


# ---------------------------------------------------------- 장중 여부 · 시각 헬퍼

_MARKET_OPEN = time(9, 0)
_MARKET_CLOSE = time(15, 30)


def is_market_hours(now: datetime) -> bool:
    """평일 09:00~15:30 이면 '데이터가 있어야 하는 장중'. 밖이면 0 건도 정상."""
    if now.weekday() >= 5:  # 토·일
        return False
    return _MARKET_OPEN <= now.time() <= _MARKET_CLOSE


def _fmt_ts(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(text[: len(fmt) + 6], fmt)  # noqa: DTZ007
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(text)
    return None


# --------------------------------------------------------------------- DB 조회


def fetch_tick_stats(conn: Any, cutoff: datetime) -> TickStats:
    """최근 N분 ticks 집계를 SQL 로 뽑는다(행을 통째로 안 끌어온다)."""
    ph = _placeholder(conn)
    cut = _bind_ts(conn, cutoff)
    cur = conn.cursor()
    st = TickStats()

    cur.execute(
        f"select count(*), count(distinct code), max(ts) from ticks where ts >= {ph}",
        (cut,),
    )
    row = cur.fetchone()
    st.n_rows = int(row[0] or 0)
    st.n_codes = int(row[1] or 0)
    st.latest_ts = _fmt_ts(row[2])
    if st.n_rows == 0:
        cur.close()
        return st

    cur.execute(
        f"select side, count(*) from ticks where ts >= {ph} group by side", (cut,)
    )
    st.side_counts = {(None if r[0] is None else int(r[0])): int(r[1]) for r in cur.fetchall()}

    cur.execute(
        "select count(*), count(strength), min(strength), max(strength), "
        f"count(distinct strength) from ticks where ts >= {ph}",
        (cut,),
    )
    r = cur.fetchone()
    st.strength_n = int(r[1] or 0)
    st.strength_null = int((r[0] or 0) - (r[1] or 0))
    st.strength_min = None if r[2] is None else float(r[2])
    st.strength_max = None if r[3] is None else float(r[3])
    st.strength_distinct = int(r[4] or 0)

    cur.execute(
        f"select count(*) from ticks where ts >= {ph} "
        "and best_bid is not null and best_ask is not null "
        "and best_bid > best_ask",
        (cut,),
    )
    st.bid_gt_ask = int((cur.fetchone() or [0])[0] or 0)

    cur.execute(
        f"select code, min(price), max(price) from ticks where ts >= {ph} "
        "and price is not null group by code",
        (cut,),
    )
    for code, pmin, pmax in cur.fetchall():
        st.price_min[str(code)] = int(pmin)
        st.price_max[str(code)] = int(pmax)
    cur.close()
    return st


def fetch_quote_stats(conn: Any, cutoff: datetime) -> QuoteStats:
    """최근 N분 quotes 집계."""
    ph = _placeholder(conn)
    cut = _bind_ts(conn, cutoff)
    cur = conn.cursor()
    qs = QuoteStats()

    cur.execute(
        f"select count(*), count(distinct code), max(ts) from quotes where ts >= {ph}",
        (cut,),
    )
    row = cur.fetchone()
    qs.n_rows = int(row[0] or 0)
    qs.n_codes = int(row[1] or 0)
    qs.latest_ts = _fmt_ts(row[2])
    if qs.n_rows == 0:
        cur.close()
        return qs

    cur.execute(
        f"select count(*) from quotes where ts >= {ph} "
        "and bid1 is not null and ask1 is not null and bid1 >= ask1",
        (cut,),
    )
    qs.bid_ge_ask = int((cur.fetchone() or [0])[0] or 0)

    cur.execute(
        f"select count(*) from quotes where ts >= {ph} "
        "and (total_bid_qty is null or total_ask_qty is null "
        "or total_bid_qty <= 0 or total_ask_qty <= 0)",
        (cut,),
    )
    qs.total_qty_zero = int((cur.fetchone() or [0])[0] or 0)

    cur.execute(
        f"select count(*) from quotes where ts >= {ph} "
        "and bid10 is not null and ask10 is not null",
        (cut,),
    )
    qs.l10_filled = int((cur.fetchone() or [0])[0] or 0)

    cur.execute(
        f"select count(*) from quotes where ts >= {ph} and vi_flag is not null",
        (cut,),
    )
    qs.vi_seen = int((cur.fetchone() or [0])[0] or 0)
    cur.close()
    return qs


def fetch_daily_ranges(
    conn: Any, trade_date: str | None
) -> tuple[dict[str, tuple[float, float]], str | None, bool]:
    """당일 daily_bars 고저 범위. 없으면 최근 거래일로 폴백(스케일 확인용).

    반환: (code -> (low, high), 참조한 날짜, 당일 여부). daily_bars 자체가 없으면
    ({}, None, False) — 가격 검사는 '참조 불가'로 경고 처리한다.
    """
    ph = _placeholder(conn)
    cur = conn.cursor()
    try:
        ref_date = trade_date
        same_day = True
        if ref_date is not None:
            cur.execute(
                f"select count(*) from daily_bars where date = {ph}", (ref_date,)
            )
            if int((cur.fetchone() or [0])[0] or 0) == 0:
                ref_date = None
        if ref_date is None:
            same_day = False
            cur.execute("select max(date) from daily_bars")
            got = cur.fetchone()
            ref_date = None if not got else got[0]
        if ref_date is None:
            cur.close()
            return {}, None, False
        cur.execute(
            f"select code, low, high from daily_bars where date = {ph} "
            "and low is not null and high is not null",
            (ref_date,),
        )
        ranges = {str(c): (float(lo), float(hi)) for c, lo, hi in cur.fetchall()}
        cur.close()
        return ranges, str(ref_date), same_day
    except Exception:  # noqa: BLE001 — daily_bars 미존재(테스트 sqlite 등)면 참조 불가
        with contextlib.suppress(Exception):
            cur.close()
        return {}, None, False


def fetch_raw_sample(conn: Any, table: str, cutoff: datetime, where: str = "") -> str | None:
    """이상 유형별 raw 원문 1건. 사후 FID 보정용."""
    ph = _placeholder(conn)
    cut = _bind_ts(conn, cutoff)
    clause = f" and {where}" if where else ""
    cur = conn.cursor()
    try:
        cur.execute(
            f"select raw from {table} where ts >= {ph}{clause} "
            "and raw is not null limit 1",
            (cut,),
        )
        row = cur.fetchone()
    except Exception:  # noqa: BLE001
        row = None
    finally:
        cur.close()
    if not row or row[0] is None:
        return None
    return str(row[0])[:500]


# --------------------------------------------------------------------- 순수 판정


def judge_collection(
    tick: TickStats, quote: QuoteStats, *, expected: int, minutes: int, market: bool
) -> Check:
    """항목 1 — 수집 여부·구독 종목 수."""
    name = "1. 수집 여부"
    if tick.n_rows == 0 and quote.n_rows == 0:
        if not market:
            return Check(name, Status.OK, f"최근 {minutes}분 수집 0건 — 장중 아님(정상)")
        return Check(
            name,
            Status.FAIL,
            f"최근 {minutes}분 ticks·quotes 0건 (장중) — 연결·인증·구독 확인",
        )
    n_codes = max(tick.n_codes, quote.n_codes)
    detail = (
        f"틱 {tick.n_rows:,}행/{tick.n_codes}종목 · "
        f"호가 {quote.n_rows:,}행/{quote.n_codes}종목 (기대 ~{expected})"
    )
    if tick.n_rows == 0 or quote.n_rows == 0:
        which = "ticks" if tick.n_rows == 0 else "quotes"
        return Check(name, Status.FAIL, f"{which} 만 0건 — 한쪽 구독 결손. {detail}")
    if market and n_codes < max(1, expected // 2):
        return Check(
            name, Status.WARN, f"구독 종목 {n_codes} < 기대치 절반 — 일부 구독 실패? {detail}"
        )
    return Check(name, Status.OK, detail)


def judge_side_signs(tick: TickStats) -> Check:
    """항목 2 — 체결 방향 부호(가장 중요). +1·-1 둘 다 보여야 한다."""
    name = "2. 체결 방향 부호"
    if tick.n_rows == 0:
        return Check(name, Status.OK, "판정 불가 (틱 없음)")
    counts = tick.side_counts
    buys = counts.get(1, 0)
    sells = counts.get(-1, 0)
    nulls = counts.get(None, 0)
    total = sum(counts.values()) or 1
    if buys > 0 and sells > 0:
        ratio = buys / (buys + sells)
        detail = (
            f"매수 +1 {buys:,} · 매도 -1 {sells:,} · 매수비율 {ratio:.0%} "
            f"(NULL {nulls:,})"
        )
        # 한쪽이 5% 미만이면 부호는 오지만 편향 — 경고로 눈에 띄게.
        if min(buys, sells) / total < 0.05:
            return Check(name, Status.WARN, f"한쪽 부호 5% 미만 — 편향 의심. {detail}")
        return Check(name, Status.OK, detail)
    only = "매수 +1" if buys > 0 else ("매도 -1" if sells > 0 else "부호 없음(NULL)")
    return Check(
        name,
        Status.FAIL,
        f"side 가 {only} 뿐 — 매수/매도 부호 FID(15) 미매핑. "
        f"+1 {buys:,} / -1 {sells:,} / NULL {nulls:,}",
    )


def judge_strength(tick: TickStats) -> Check:
    """항목 3 — 체결강도 분포. 상수·전부 NULL 아니고 그럴듯한 범위(0~수백)."""
    name = "3. 체결강도 분포"
    if tick.n_rows == 0:
        return Check(name, Status.OK, "판정 불가 (틱 없음)")
    if tick.strength_n == 0:
        return Check(name, Status.FAIL, "strength 전부 NULL — FID(228) 미매핑")
    detail = (
        f"n={tick.strength_n:,} · 범위 {tick.strength_min:g}~{tick.strength_max:g} · "
        f"distinct {tick.strength_distinct} · NULL {tick.strength_null:,}"
    )
    if tick.strength_distinct <= 1:
        return Check(name, Status.FAIL, f"strength 상수 — 미갱신 의심. {detail}")
    lo, hi = tick.strength_min, tick.strength_max
    if lo is not None and lo < 0:
        return Check(name, Status.WARN, f"음수 강도 — 스케일 확인. {detail}")
    if hi is not None and hi > 1000:
        return Check(name, Status.WARN, f"강도 상한 1000 초과 — 스케일 확인. {detail}")
    return Check(name, Status.OK, detail)


def judge_price(
    tick: TickStats,
    ranges: dict[str, tuple[float, float]],
    ref_date: str | None,
    same_day: bool,
) -> Check:
    """항목 4 — 가격 정합. best_bid≤best_ask, price 가 daily 고저/스케일 안."""
    name = "4. 가격 정합"
    if tick.n_rows == 0:
        return Check(name, Status.OK, "판정 불가 (틱 없음)")
    if tick.bid_gt_ask > 0:
        return Check(
            name, Status.FAIL, f"best_bid > best_ask 위반 {tick.bid_gt_ask:,}행 — 호가 부호 확인"
        )
    if not tick.price_min:
        return Check(name, Status.WARN, "price 전부 NULL — FID(10) 미매핑")
    if not ranges:
        return Check(
            name, Status.WARN, "daily_bars 참조 불가 — 가격 스케일 미확인 (best_bid≤best_ask 는 OK)"
        )
    # 당일이면 고저 범위 그대로, 폴백이면 스케일만(±30% 상한가 + 여유) 넓게 본다.
    lo_factor, hi_factor = (1.0, 1.0) if same_day else (0.5, 2.0)
    checked = 0
    offenders: list[str] = []
    for code, pmin in tick.price_min.items():
        rng = ranges.get(code)
        if rng is None:
            continue
        low, high = rng
        pmax = tick.price_max.get(code, pmin)
        checked += 1
        if pmin < low * lo_factor or pmax > high * hi_factor:
            offenders.append(f"{code}({pmin}~{pmax} vs {low:g}~{high:g})")
    mode = "당일 고저" if same_day else f"최근({ref_date}) 스케일"
    if checked == 0:
        return Check(name, Status.WARN, f"대조 가능한 종목 없음 (daily {ref_date})")
    if offenders:
        status = Status.FAIL if same_day else Status.WARN
        sample = ", ".join(offenders[:3])
        return Check(
            name,
            status,
            f"{mode} 범위 이탈 {len(offenders)}/{checked}종목 — 스케일 의심: {sample}",
        )
    return Check(name, Status.OK, f"{mode} 범위 내 {checked}종목 · best_bid≤best_ask OK")


def judge_quotes(quote: QuoteStats, *, market: bool) -> Check:
    """항목 5 — 호가. bid1<ask1, 총잔량>0, 10호가, ts 갱신, vi_flag."""
    name = "5. 호가"
    if quote.n_rows == 0:
        return Check(name, Status.OK, "판정 불가 (호가 없음)")
    parts: list[str] = []
    status = Status.OK
    if quote.bid_ge_ask > 0:
        status = Status.FAIL
        parts.append(f"bid1≥ask1 위반 {quote.bid_ge_ask:,}행")
    else:
        parts.append("bid1<ask1 OK")
    if quote.total_qty_zero > 0:
        frac = quote.total_qty_zero / quote.n_rows
        if frac > 0.5:
            status = max(status, Status.WARN, key=lambda s: s.rank)
        parts.append(f"총잔량≤0/NULL {quote.total_qty_zero:,}행({frac:.0%})")
    else:
        parts.append("총잔량>0 OK")
    l10_frac = quote.l10_filled / quote.n_rows
    if l10_frac < 0.5:
        status = max(status, Status.WARN, key=lambda s: s.rank)
        parts.append(f"10호가 채움 {l10_frac:.0%}")
    else:
        parts.append(f"10호가 {l10_frac:.0%}")
    # ts 갱신: 장중이면 최신 스냅샷이 최근이어야 한다.
    latest = _parse_ts(quote.latest_ts)
    if market and latest is not None:
        age = (datetime.now() - latest).total_seconds()  # noqa: DTZ005
        if age > 180:
            status = max(status, Status.WARN, key=lambda s: s.rank)
            parts.append(f"최신 호가 {age:.0f}s 전 — 갱신 지연?")
        else:
            parts.append(f"ts 갱신 {age:.0f}s 전")
    if quote.vi_seen == 0:
        parts.append("vi_flag 미수신(0D 미탑재일 수 있음)")
    else:
        parts.append(f"vi_flag {quote.vi_seen:,}건")
    return Check(name, status, " · ".join(parts))


# --------------------------------------------------------------------- 리포트


def run_checks(conn: Any, *, minutes: int, expected: int, now: datetime) -> list[Check]:
    """전 항목 검사를 돌려 Check 리스트로. 이상 시 raw 샘플을 붙인다."""
    cutoff = now - timedelta(minutes=minutes)
    market = is_market_hours(now)
    tick = fetch_tick_stats(conn, cutoff)
    quote = fetch_quote_stats(conn, cutoff)

    trade_date = None
    latest_tick = _parse_ts(tick.latest_ts)
    if latest_tick is not None:
        trade_date = latest_tick.strftime("%Y-%m-%d")
    ranges, ref_date, same_day = fetch_daily_ranges(conn, trade_date)

    checks = [
        judge_collection(tick, quote, expected=expected, minutes=minutes, market=market),
        judge_side_signs(tick),
        judge_strength(tick),
        judge_price(tick, ranges, ref_date, same_day),
        judge_quotes(quote, market=market),
    ]

    # 항목 6 — 이상(경고/실패) 유형별 raw 원문 1건씩 붙인다.
    _attach_raw_samples(conn, cutoff, checks)
    return checks


def _attach_raw_samples(conn: Any, cutoff: datetime, checks: list[Check]) -> None:
    """경고·실패 항목에 사후 FID 보정용 raw 샘플을 채운다."""
    by_name = {c.name: c for c in checks}
    side = by_name.get("2. 체결 방향 부호")
    if side and side.status is not Status.OK:
        side.raw = fetch_raw_sample(conn, "ticks", cutoff)
    strength = by_name.get("3. 체결강도 분포")
    if strength and strength.status is not Status.OK:
        strength.raw = fetch_raw_sample(conn, "ticks", cutoff)
    price = by_name.get("4. 가격 정합")
    if price and price.status is not Status.OK:
        price.raw = fetch_raw_sample(conn, "ticks", cutoff)
    quotes = by_name.get("5. 호가")
    if quotes and quotes.status is not Status.OK:
        quotes.raw = fetch_raw_sample(conn, "quotes", cutoff)


_MARK = {Status.OK: "[OK]", Status.WARN: "[경고]", Status.FAIL: "[실패]"}


def format_report(checks: list[Check], *, minutes: int, now: datetime) -> str:
    """사람이 읽는 리포트. 실패/경고엔 raw 샘플을 붙여 사후 보정에 쓴다."""
    worst = worst_status(checks)
    lines = [
        f"# 틱 수집 헬스체크 — {now:%Y-%m-%d %H:%M:%S} (최근 {minutes}분)",
        (
            f"종합: {_MARK[worst]}  "
            f"실패 {sum(c.status is Status.FAIL for c in checks)} · "
            f"경고 {sum(c.status is Status.WARN for c in checks)} · "
            f"OK {sum(c.status is Status.OK for c in checks)}"
        ),
        "",
    ]
    for c in checks:
        lines.append(f"{_MARK[c.status]} {c.name}: {c.detail}")
        if c.raw:
            lines.append(f"       raw: {c.raw}")
    return "\n".join(lines)


def worst_status(checks: list[Check]) -> Status:
    return max((c.status for c in checks), key=lambda s: s.rank, default=Status.OK)


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="틱·호가 실시간 수집 헬스체크 (읽기 전용)")
    p.add_argument("--minutes", type=int, default=15, help="검사 창(분, 기본 15)")
    p.add_argument("--expected", type=int, default=80, help="기대 구독 종목 수(기본 80)")
    p.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    p.add_argument(
        "--log-path", default="data/tick_health.log", help="리포트 append 로그 경로"
    )
    p.add_argument("--no-log", action="store_true", help="로그 파일에 남기지 않는다")
    return p


def _append_log(path: str, report: str) -> None:
    with contextlib.suppress(Exception):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(report + "\n\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now()  # noqa: DTZ005 — KST 로컬시각(DB ts 도 tz 없는 KST)

    try:
        conn = connect(args.dsn)
    except Exception as exc:  # noqa: BLE001
        print(f"DB 접속 실패: {exc}", file=sys.stderr)
        return 1
    try:
        checks = run_checks(
            conn, minutes=args.minutes, expected=args.expected, now=now
        )
    except Exception as exc:  # noqa: BLE001
        print(f"검사 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    report = format_report(checks, minutes=args.minutes, now=now)
    print(report)
    if not args.no_log:
        _append_log(args.log_path, report)

    return 1 if worst_status(checks) is Status.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
