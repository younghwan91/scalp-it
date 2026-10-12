"""정규장 가격의 정문 — 가격제한폭 기준가·정규장 종가·정규장 일봉은 이 모듈로만 구한다.

**애프터마켓 함정(2026-09-14~).** KRX 애프터마켓(16:00~20:00 연속매매, ETF·ETN 제외)이 생긴 뒤로
키움이 주는 일봉(``daily_bars``)의 고가·저가·종가·거래량·거래대금, ``supply_demand.close``,
분봉 16:00~19:59 봉에 정규장 밖 체결이 섞였다. 시가만 정상이다. 그런데 다음 날 가격제한폭의
**기준가는 전일 정규장 종가**다. 일봉 종가 × 1.3 으로 상한가를 구하면 09-14 이후 종목-일의
약 79% 에서 상한가가 틀렸다(2026-10 실측, 아래 :data:`AFTERMARKET_START` 참고).

출처와 우선순위(실측 근거, 2026-10-12 DB 대조):

* **기준가** :func:`base_prices` (그 세션 ``day`` 의 가격제한폭 기준가)

  1. ``supply_demand(day)`` 의 ``close / (1 + flu_rt/1e4)`` 를 호가단위로 반올림.
     ``flu_rt`` 는 등락률 ×100 이고 ``close`` 와 같은 시점 값이라 오염과 무관하게 항등식이 선다.
     분봉 15:30 과 76,883건 중 76,682건 일치했고, 어긋난 201건은 분봉이 뒤의 기업행동으로
     수정주가 재수집된 경우(비정수 분봉)와 그날 기준가가 재산정된 경우(042940 09-22, 207940 10-02)
     — **둘 다 이쪽이 맞다.** 다만 그날 행은 그날 16:00 이후에야 생기므로 장 전·장중에는 없다.
  2. 직전 거래일 분봉의 정규장 마지막 봉(15:30~15:32, 동시호가 체결이 없으면 15:19~15:29) 종가.
     장 전(08:55) 감지기가 실제로 쓰는 경로다. 분봉은 전일 15:45 크론(``cli_intraday``,
     거래대금 상위 ~1,400종목)이 적재한다. 09-14 이후 기준가와 100% 일치(5개 날짜, 각 ~1,400종목).
  3. 직전 거래일 ``daily_bars.close`` — **09-14 이전 날짜만.** 그 뒤로는 쓰지 않는다
     (틀린 기준가를 주느니 모른다고 한다. 감지기는 기준가가 없으면 상한가 판정을 건너뛴다).

* **정규장 종가** :func:`regular_closes` (그날 ``day`` 자체의 종가)

  1. 그날 분봉 정규장 마지막 봉. 2. 다음 거래일 ``supply_demand`` 의 기준가(위 1번 식).
  3. ``daily_bars.close`` — 09-14 이전만.

* **정규장 일봉** :func:`regular_session_bars` — 분봉 09:00~15:32 집계(고가·저가·거래량·거래대금).
  분봉이 없는 종목은 정규장 값을 알 수 없다(``minute_bars`` 는 ~1,400종목만 덮는다).

키움 기준가 TR 은 쓰지 않는다 — 기존 코드에 장 전 기준가를 주는 호출이 없고, 새 호출은 App key
1세션 제약으로 라이브 소켓과 토큰이 충돌한다.

``minute_bars.ts`` 는 시간대 없는 KST 다. 문자열 경계로 비교하고 ``at time zone`` 을 걸지 않는다.
SQL 은 sqlite 에서도 돌아야 한다(테스트). 표가 없으면(테스트 sqlite·권한) 그 출처를 건너뛴다.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .adapters.db import _placeholder as _ph
from .ticksize import tick_size

__all__ = [
    "AFTERMARKET_START",
    "SOURCE_DAILY",
    "SOURCE_MINUTE",
    "SOURCE_SUPPLY",
    "RegularBar",
    "base_price_from_supply",
    "base_price_sources",
    "base_prices",
    "limitup_codes",
    "next_session",
    "previous_session",
    "regular_close_sources",
    "regular_closes",
    "regular_session_bars",
]

#: KRX 애프터마켓 시작일. 이날부터 일봉 고가·저가·종가·거래량에 16:00~20:00 체결이 섞인다.
AFTERMARKET_START = date(2026, 9, 14)

SOURCE_SUPPLY = "supply_demand"
SOURCE_MINUTE = "minute"
SOURCE_DAILY = "daily_bars"

#: 정규장 분봉 창. 09:00 개장 ~ 15:30 종가 동시호가 봉(일부 종목은 15:31·15:32 에 찍힌다).
REGULAR_OPEN = "09:00:00"
REGULAR_END = "15:33:00"
#: 종가 봉 창. 동시호가 체결이 없는 종목은 15:20 직전 연속매매 봉이 종가다 — 그보다 이른 봉만
#: 있으면 수집이 덜 된 것일 수 있어 종가로 인정하지 않는다.
CLOSE_FROM = "15:19:00"


def _day(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _codes(codes: Iterable[str] | None) -> list[str] | None:
    if codes is None:
        return None
    return list(dict.fromkeys(str(c) for c in codes if c))


def _fetch(conn: Any, sql: str, params: Sequence[Any]) -> list[tuple] | None:
    """SELECT 하나. 표가 없거나 실패하면 None(그 출처는 건너뛴다). postgres 는 트랜잭션을 되살린다."""
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
        return rows
    except Exception:  # noqa: BLE001 — 출처 하나가 없다고 기준가 전체를 포기하지 않는다
        with contextlib.suppress(Exception):
            conn.rollback()
        return None


def _code_clause(conn: Any, codes: list[str] | None, column: str = "code") -> tuple[str, list[Any]]:
    if codes is None:
        return "", []
    ph = _ph(conn)
    return f" and {column} in ({', '.join([ph] * len(codes))})", list(codes)


def previous_session(conn: Any, day: date | str) -> date | None:
    """``day`` **미만**의 가장 최근 거래일(``daily_bars`` 기준). 장 전에는 그날 행이 없어도 된다."""
    ph = _ph(conn)
    rows = _fetch(conn, f"select max(date) from daily_bars where date < {ph}", [str(_day(day))])
    return _day(rows[0][0]) if rows and rows[0][0] else None


def next_session(conn: Any, day: date | str) -> date | None:
    """``day`` **초과**의 첫 거래일(``daily_bars`` 기준). 장 전·장중에는 아직 없어 None."""
    ph = _ph(conn)
    rows = _fetch(conn, f"select min(date) from daily_bars where date > {ph}", [str(_day(day))])
    return _day(rows[0][0]) if rows and rows[0][0] else None


def base_price_from_supply(close: float, flu_rt: float) -> int | None:
    """``supply_demand`` 한 행 → 그 세션 기준가(= 전일 정규장 종가).

    ``flu_rt`` 는 소수 둘째 자리 % 를 ×100 한 값이라 역산값이 몇 원 흔들린다. 그 가격대 호가단위로
    반올림한다. 반올림 폭이 정밀도(1e-4)보다 크면(ETF 처럼 다른 호가표를 쓰는 종목) 원 단위로만 맞춘다.
    """
    if not close or close <= 0 or flu_rt is None:
        return None
    denom = 1.0 + float(flu_rt) / 1e4
    if denom <= 0:
        return None
    raw = float(close) / denom
    tick = tick_size(raw)
    snapped = round(raw / tick) * tick
    if abs(snapped - raw) <= raw * 1e-4 + 0.5:
        return int(snapped)
    return round(raw)


def _supply_bases(conn: Any, day: date, codes: list[str] | None) -> dict[str, int]:
    ph = _ph(conn)
    clause, extra = _code_clause(conn, codes)
    rows = _fetch(conn, f"select code, close, flu_rt from supply_demand where date = {ph}{clause}",
                  [str(day), *extra]) or []
    out: dict[str, int] = {}
    for code, close, flu in rows:
        base = base_price_from_supply(close, flu)
        if base:
            out[str(code)] = base
    return out


def _minute_closes(conn: Any, day: date, codes: list[str] | None) -> dict[str, int]:
    """그날 정규장 마지막 분봉 종가(15:19~15:32 창 안의 가장 늦은 봉)."""
    ph = _ph(conn)
    clause, extra = _code_clause(conn, codes)
    d = str(day)
    rows = _fetch(
        conn,
        f"select code, ts, close from minute_bars where ts >= {ph} and ts < {ph}{clause}",
        [f"{d} {CLOSE_FROM}", f"{d} {REGULAR_END}", *extra],
    ) or []
    last: dict[str, tuple[str, float]] = {}
    for code, ts, close in rows:
        if close is None or float(close) <= 0:
            continue
        key, t = str(code), str(ts)
        if key not in last or t > last[key][0]:
            last[key] = (t, float(close))
    return {c: round(v) for c, (_, v) in last.items()}


def _daily_closes(conn: Any, day: date, codes: list[str] | None) -> dict[str, int]:
    if day >= AFTERMARKET_START:
        return {}  # 애프터마켓 오염 — 정규장 종가가 아니다
    ph = _ph(conn)
    clause, extra = _code_clause(conn, codes)
    rows = _fetch(conn, f"select code, close from daily_bars where date = {ph}{clause}",
                  [str(day), *extra]) or []
    return {str(c): int(v) for c, v in rows if v and int(v) > 0}


def _merge(sources: Sequence[tuple[str, dict[str, int]]], codes: list[str] | None) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for name, values in sources:
        for code, price in values.items():
            if code not in out and (codes is None or code in codes):
                out[code] = (price, name)
    return out


def base_price_sources(conn: Any, day: date | str,
                       codes: Iterable[str] | None = None) -> dict[str, tuple[int, str]]:
    """세션 ``day`` 의 가격제한폭 기준가와 출처. ``{code: (기준가, 출처)}`` — 모르는 종목은 빠진다."""
    d = _day(day)
    wanted = _codes(codes)
    if wanted == []:
        return {}
    srcs: list[tuple[str, dict[str, int]]] = [(SOURCE_SUPPLY, _supply_bases(conn, d, wanted))]
    prev = previous_session(conn, d)
    if prev is not None:
        missing = None if wanted is None else [c for c in wanted if c not in srcs[0][1]]
        if missing != []:
            srcs.append((SOURCE_MINUTE, _minute_closes(conn, prev, missing)))
            srcs.append((SOURCE_DAILY, _daily_closes(conn, prev, missing)))
    return _merge(srcs, wanted)


def base_prices(conn: Any, day: date | str, codes: Iterable[str] | None = None) -> dict[str, int]:
    """세션 ``day`` 의 가격제한폭 기준가(= 전일 **정규장** 종가). 상한가는 ``limit_up_price(이 값)``.

    장 전에 오늘 날짜를 넣으면 전일 분봉 15:30 으로 답한다. 모르는 종목은 dict 에 없다 —
    일봉 종가로 메우지 않는다(09-14 이후 79% 가 틀린 값이다).
    """
    return {c: p for c, (p, _) in base_price_sources(conn, day, codes).items()}


def regular_close_sources(conn: Any, day: date | str,
                          codes: Iterable[str] | None = None) -> dict[str, tuple[int, str]]:
    """``day`` 의 정규장 종가와 출처. 분봉 → 다음 거래일 ``supply_demand`` 기준가 → (09-14 전) 일봉."""
    d = _day(day)
    wanted = _codes(codes)
    if wanted == []:
        return {}
    srcs: list[tuple[str, dict[str, int]]] = [(SOURCE_MINUTE, _minute_closes(conn, d, wanted))]
    nxt = next_session(conn, d)
    if nxt is not None:
        srcs.append((SOURCE_SUPPLY, _supply_bases(conn, nxt, wanted)))
    srcs.append((SOURCE_DAILY, _daily_closes(conn, d, wanted)))
    return _merge(srcs, wanted)


def regular_closes(conn: Any, day: date | str, codes: Iterable[str] | None = None) -> dict[str, int]:
    """``day`` 의 정규장 종가. 모르는 종목은 dict 에 없다."""
    return {c: p for c, (p, _) in regular_close_sources(conn, day, codes).items()}


@dataclass(frozen=True)
class RegularBar:
    """분봉 09:00~15:32 집계. 거래대금은 ``daily_bars`` 와 같은 **백만원** 단위로 맞춘다."""

    code: str
    day: date
    high: float
    low: float
    volume: int
    trade_value: float  # 백만원


def regular_session_bars(conn: Any, day: date | str,
                         codes: Iterable[str] | None = None) -> dict[str, RegularBar]:
    """그날 정규장 고가·저가·거래량·거래대금(분봉 집계). 분봉이 없는 종목은 빠진다.

    시가는 일봉이 정상이고 종가는 :func:`regular_closes` 가 정문이라 여기서는 내지 않는다.
    """
    d = _day(day)
    wanted = _codes(codes)
    if wanted == []:
        return {}
    ph = _ph(conn)
    clause, extra = _code_clause(conn, wanted)
    ds = str(d)
    rows = _fetch(
        conn,
        "select code, max(high), min(low), sum(volume), sum(trade_value) from minute_bars "
        f"where ts >= {ph} and ts < {ph}{clause} group by code",
        [f"{ds} {REGULAR_OPEN}", f"{ds} {REGULAR_END}", *extra],
    ) or []
    out: dict[str, RegularBar] = {}
    for code, hi, lo, vol, tv in rows:
        if hi is None or lo is None or float(hi) <= 0:
            continue
        out[str(code)] = RegularBar(code=str(code), day=d, high=float(hi), low=float(lo),
                                    volume=int(vol or 0), trade_value=float(tv or 0.0) / 1e6)
    return out


#: 상한가 근사 문턱(정규장 종가 / 기준가). 호가단위 내림 때문에 30% 가 29.x% 로 찍혀 29.5% 로 본다.
LIMITUP_RATIO = 1.295


def limitup_codes(conn: Any, day: date | str, codes: Iterable[str] | None = None,
                  *, ratio: float = LIMITUP_RATIO) -> set[str]:
    """``day`` 에 정규장 종가가 기준가 × ``ratio`` 이상인 종목 = 상한가 마감 근사.

    일봉 ``close / lag(close)`` 로 세면 2026-09-14 이후 분자·분모가 둘 다 애프터마켓 값이라 틀린다.
    종가나 기준가를 모르는 종목은 세지 않는다(분봉이 없는 종목의 그날 종가는 다음 거래일 수급표가
    들어와야 알 수 있다 — 장 전에 전일을 세면 분봉 밖 종목은 빠진다).
    """
    wanted = _codes(codes)
    closes = regular_closes(conn, day, wanted)
    if not closes:
        return set()
    bases = base_prices(conn, day, list(closes))
    return {c for c, px in closes.items() if bases.get(c, 0) > 0 and px / float(bases[c]) >= ratio}
