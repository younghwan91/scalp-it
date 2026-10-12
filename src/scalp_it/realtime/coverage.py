"""커버리지 계기판 — 그날 급등 이벤트 중 우리가 **보고 있던** 비율.

만쥬 방식은 대장이 튀는 순간 같은 무리의 2·3등주를 잡는 것이라, 그 대장이 튀기
전에 이미 구독돼 있어야 한다. 구독 안 된 종목은 틱이 안 오고, 틱이 없으면 신호도
주문도 체결도 없다.

    구독 못 함 → 신호 0 → 주문 0 → 체결 0 → 수익 0

`leader_scan.py` 가 적어둔 실측이 **커버리지 2%** 다(최근 7거래일 상한가 52건 중
우리 유니버스 안에 있던 것 1건). 2026-08-26~28 에 시장 상한가는 4·8·7건이었는데
우리 신호는 사흘 내리 0건이었고, 매매일지엔 "희소 이벤트라 정상"이라고 적혔다.
정상이 아니었다 — 이벤트는 있었고 우리가 못 봤다.

**재지 않으면 고쳤는지 모른다.** 슬롯 선점(2026-08-31)도, 0D 를 REST 폴링으로
빼서 구독을 50→100 으로 늘리는 것도, 효과가 이 숫자로만 드러난다.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..adapters.db import _placeholder as _ph
from ..regular_session import base_prices, regular_closes

#: 등락률 경계의 부동소수점 허용오차. 호가 단위가 원 단위라 1e-9 면 충분하다.
_EPS = 1e-9


@dataclass(frozen=True)
class SurgeEvent:
    """그날 문턱 이상 튄 종목 하나."""

    code: str
    rate: float


@dataclass
class Coverage:
    """이벤트 대비 우리가 본 비율.

    ``rate`` 가 ``None`` 이면 **판정 불가**다(그날 이벤트가 없었다). 0% 와 구별해야
    한다 — 조용한 날과 못 본 날은 완전히 다른 얘기다.
    """

    n_events: int = 0
    n_covered: int = 0
    covered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        if self.n_events == 0:
            return None
        return self.n_covered / self.n_events


def surge_events(rows: Iterable[Sequence[Any]], *, min_rate: float) -> list[SurgeEvent]:
    """``(code, close, prev_close)`` 행에서 문턱 이상 튄 종목 → **큰 것부터**.

    큰 것부터인 이유는 놓친 목록을 위에서부터 읽기 위해서다. 30% 를 놓친 것과
    12% 를 놓친 것은 무게가 다르다.

    ``float`` 캐스팅이 핵심이다. 종가가 정수형이라 그냥 나누면 정수 나눗셈이 되어
    등락률이 0 이 된다 — ``market_regime.count_limitups`` 가 그 함정에 빠져 1년
    2,313건을 5건으로 셌던 적이 있다. 전일종가가 없거나 0 이면(신규 상장·데이터
    구멍) 등락률을 만들 수 없으므로 조용히 건너뛴다.
    """
    out: list[SurgeEvent] = []
    for row in rows:
        code, close, prev = row[0], row[1], row[2]
        if close is None or prev is None:
            continue
        prev_f = float(prev)
        if prev_f <= 0:
            continue
        rate = float(close) / prev_f - 1.0
        # 경계는 **허용오차와 함께** 본다. 1295/1000 - 1 은 0.29499999999999993 이라
        # `>= 0.295` 로 자르면 진짜 상한가가 이벤트에서 빠진다(그날 못 본 종목이
        # 아예 안 세어지므로 커버리지가 실제보다 좋아 보인다 — 최악의 방향이다).
        if rate >= min_rate - _EPS:
            out.append(SurgeEvent(str(code), rate))
    out.sort(key=lambda e: -e.rate)
    return out


def measure_coverage(events: Sequence[SurgeEvent],
                     first_seen: Mapping[str, Any],
                     *, before: Any = None) -> Coverage:
    """이벤트 중 우리가 본 것을 센다 → :class:`Coverage`.

    ``first_seen`` 은 그날 **처음 틱이 들어온 시각**을 코드별로 담는다(없으면 그날
    한 번도 안 붙은 것). ``before`` 를 주면 그 시각까지 붙어 있던 것만 인정한다 —
    만쥬는 오전에 대장을 낙점하므로 14시에 붙은 구독은 그날 아무 쓸모가 없었고,
    그걸 커버리지로 세면 숫자가 거짓말을 한다.

    구독했지만 이벤트가 아닌 종목은 세지 않는다. 삼성전자를 종일 붙여둔 것은
    그날 상한가를 본 게 아니다.
    """
    covered: list[str] = []
    missed: list[str] = []
    for ev in events:
        seen = first_seen.get(ev.code)
        ok = seen is not None and (before is None or seen <= before)
        (covered if ok else missed).append(ev.code)
    return Coverage(n_events=len(events), n_covered=len(covered),
                    covered=covered, missed=missed)


def count_session_codes(conn: Any, *, day: str, market: str) -> int:
    """그날 일봉이 있는 ``market`` 종목 수. :func:`fetch_surge_rows` 와의 차이 = 정규장 값을 모르는 종목."""
    ph = _ph(conn)
    cur = conn.cursor()
    cur.execute(
        f"select count(*) from daily_bars b join stocks s on s.code = b.code "
        f"where s.market = {ph} and b.date = {ph}",
        [market, day],
    )
    row = cur.fetchone()
    cur.close()
    return int(row[0]) if row and row[0] else 0


def fetch_surge_rows(conn: Any, *, day: str, market: str) -> list[tuple]:
    """그날 ``(code, close, prev_close)`` — 둘 다 **정규장** 값이다.

    ``close`` 는 그날 정규장 종가, ``prev_close`` 는 그날 가격제한폭 기준가(직전 거래일 정규장 종가)다.
    정문 :mod:`scalp_it.regular_session` 으로 구한다. ``daily_bars`` 종가를 ``lag`` 로 쓰면
    2026-09-14 이후 둘 다 애프터마켓 종가라 등락률이 틀리고, 달력 하루 전을 빼면 월요일마다 구멍이
    난다(주말·휴장). 정규장 종가나 기준가를 모르는 종목은 행이 없다.
    """
    ph = _ph(conn)
    cur = conn.cursor()
    cur.execute(
        f"select b.code from daily_bars b join stocks s on s.code = b.code "
        f"where s.market = {ph} and b.date = {ph}",
        [market, day],
    )
    codes = [str(r[0]) for r in cur.fetchall()]
    cur.close()
    if not codes:
        return []
    closes = regular_closes(conn, day, codes)
    bases = base_prices(conn, day, list(closes))
    return [(c, closes[c], bases[c]) for c in codes if c in closes and c in bases]


def fetch_first_ticks(conn: Any, *, day: str) -> dict[str, str]:
    """그날 종목별 **첫 틱 시각**(``HH:MM:SS``) — 안 들어온 종목은 아예 없다.

    "구독했나"의 증거로 REG 로그가 아니라 **실제로 들어온 틱**을 쓴다. 등록만 되고
    푸시가 안 오는 경우가 실제로 있었다(2026-08-28, 장중 REG 추가가 기존 구독을
    통째로 날린 사고). 우리가 알고 싶은 건 등록이 아니라 본 것이다.
    """
    ph = _ph(conn)
    cur = conn.cursor()
    cur.execute(
        f"select code, min(ts) from ticks where ts >= {ph} and ts < {ph} group by code",
        [f"{day} 00:00:00", f"{day} 23:59:59"],
    )
    out: dict[str, str] = {}
    for code, first in cur.fetchall():
        text = first if isinstance(first, str) else str(first)
        out[str(code)] = text[11:19] if len(text) >= 19 else text
    return out


def format_report(cov: Coverage, *, day: str, min_rate: float,
                  names: Mapping[str, str] | None = None,
                  n_bars: int | None = None, top: int = 10,
                  n_unknown: int = 0) -> str:
    """사람이 읽는 한 화면. 놓친 종목을 **이름으로** 적는다.

    비율만 적으면 다음 행동이 안 나온다. 어떤 종목을 놓쳤는지 보여야 "얘는 왜
    못 붙었나"를 그날 로그에서 바로 찾는다.
    """
    names = names or {}
    if cov.rate is not None:
        pct = f"{cov.rate:.1%}"
    elif n_bars == 0:
        # 일봉이 없으면 그날을 **모르는** 것이다. 조용한 날로 적으면 계기판이
        # 거짓말한다 — 장중이나 적재 전에 돌리면 늘 이 상태가 된다.
        pct = "판정 불가 (일봉 미적재 — 장 마감·적재 후 다시 재라)"
    else:
        pct = "판정 불가 (그날 이벤트 0건)"
    lines = [
        f"# 커버리지 {day} — 문턱 +{min_rate:.1%}",
        "",
        (f"이벤트 {cov.n_events}건 중 본 것 {cov.n_covered}건 "
         f"({cov.n_covered}/{cov.n_events}) — **{pct}**"),
    ]
    if n_unknown > 0:
        # 2026-09-14 이후 일봉 종가는 애프터마켓 값이라 못 쓴다. 분봉 수집(~1,400종목) 밖 종목의
        # 그날 정규장 종가는 다음 거래일 수급표(16:00 적재)가 들어와야 안다 — 그 전에는 이벤트가 덜 잡힌다.
        lines += ["", (f"(정규장 종가 미상 {n_unknown}종목 — 이벤트가 덜 잡혔을 수 있다. "
                       "다음 거래일 16:00 뒤 다시 재면 채워진다)")]
    if cov.missed:
        lines += ["", f"## 놓친 {len(cov.missed)}건"]
        lines += [f"- {c}{f' {names[c]}' if c in names else ''}"
                  for c in cov.missed[:top]]
        if len(cov.missed) > top:
            lines.append(f"- … 외 {len(cov.missed) - top}건")
    return "\n".join(lines)


def fetch_names(conn: Any, codes: Sequence[str]) -> dict[str, str]:
    """종목명 — 놓친 목록에 코드만 있으면 사람이 못 읽는다."""
    if not codes:
        return {}
    ph = _ph(conn)
    marks = ", ".join([ph] * len(codes))
    cur = conn.cursor()
    cur.execute(f"select code, name from stocks where code in ({marks})", list(codes))
    return {str(c): n for c, n in cur.fetchall() if n}
