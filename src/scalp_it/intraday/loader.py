"""분봉 읽기 어댑터 — ``minute_bars`` 조회 SQL 은 이 파일에만 존재한다.

쓰기는 :mod:`scalp_it.intraday.store`, 읽기는 여기다. 접속 대상은 ``KR_QUANT_DB``
(TimescaleDB hypertable) 이지만 SQL 은 sqlite 에서도 그대로 돌아가야 한다 —
테스트는 sqlite fixture 로 돌리고 운영만 postgres 이므로, 둘이 갈라지면
테스트가 아무것도 보증하지 못한다. 그래서 ``_placeholder`` 로 바인딩 기호만 바꾼다.

수집이 진행 중이라 종목 수는 계속 는다. :func:`load_minute_bars` 는 없는 종목을
조용히 건너뛴다 — 요청한 종목이 아직 안 들어왔다는 건 오류가 아니다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from scalp_it.intraday.store import MINUTE_COLUMNS

_NUMERIC = ("open", "high", "low", "close", "volume", "trade_value")


def _placeholder(conn: Any) -> str:
    return "%s" if conn.__class__.__module__.startswith("psycopg2") else "?"


def _day_bounds(start: str, end: str) -> tuple[str, str]:
    """``end`` 를 그날 끝까지 포함시킨다.

    ``ts <= '2026-08-17'`` 로 비교하면 그날 00:00 하나만 걸린다. 하루를 통째로
    담으려면 다음 날 00:00 미만이어야 한다.
    """
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    fmt = "%Y-%m-%d %H:%M:%S"
    return lo.strftime(fmt), hi.strftime(fmt)


#: 정규장 분봉 창 — 09:00 개장 ~ 15:30 종가 동시호가 봉(일부는 15:31·15:32 에 찍힌다). 끝은 미포함.
REGULAR_START = "09:00:00"
REGULAR_END = "15:33:00"


def load_minute_bars(
    conn: Any, start: str, end: str, codes: list[str] | None = None,
    *, regular_only: bool = False,
) -> pd.DataFrame:
    """``[start, end]`` 구간(양끝 포함, 일자 단위)의 분봉을 읽는다.

    **기본값은 하루 전체 봉이다.** 2026-09-14 KRX 애프터마켓 시작 뒤로는 16:00~19:59 봉(정규장 밖
    체결)과 15:35 전후 봉도 들어 있다. 정규장만 필요하면 ``regular_only=True`` 를 준다 — 기본값을
    바꾸면 기존 연구 재현이 달라질 수 있어 옵션으로 뒀다. 그날 종가·기준가는 이 함수가 아니라
    :mod:`scalp_it.regular_session` 이 정문이다.

    Args:
        conn: psycopg2 또는 sqlite3 연결.
        start, end: ``YYYY-MM-DD``. ``end`` 는 그날 마지막 봉까지 포함한다(애프터마켓 19:59 포함).
        codes: 종목 목록. ``None`` 이면 전 종목. 아직 수집 안 된 종목은
            그냥 빠진 채로 돌아온다 (오류 아님).
        regular_only: True 면 각 날짜의 09:00 ≤ ts < 15:33 봉만 남긴다.

    Returns:
        ``code, ts, open, high, low, close, volume, trade_value`` — ``code, ts`` 정렬.
    """
    ph = _placeholder(conn)
    lo, hi = _day_bounds(start, end)
    sql = (
        "select code, ts, open, high, low, close, volume, trade_value "
        f"from minute_bars where ts >= {ph} and ts < {ph}"
    )
    params: list[Any] = [lo, hi]
    if codes is not None:
        wanted = [str(c).strip() for c in codes if str(c).strip()]
        if not wanted:
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        sql += f" and code in ({','.join([ph] * len(wanted))})"
        params.extend(wanted)
    sql += " order by code, ts"

    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    df = pd.DataFrame(cur.fetchall(), columns=MINUTE_COLUMNS)
    cur.close()
    if df.empty:
        return pd.DataFrame(columns=MINUTE_COLUMNS)

    df["code"] = df["code"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if regular_only:
        tod = df["ts"].dt.strftime("%H:%M:%S")
        df = df[(tod >= REGULAR_START) & (tod < REGULAR_END)]
    return df.sort_values(["code", "ts"]).reset_index(drop=True)


def available_codes(conn: Any) -> list[str]:
    """``minute_bars`` 에 실제로 봉이 있는 종목 코드."""
    cur = conn.cursor()
    cur.execute("select distinct code from minute_bars order by code")
    codes = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None]
    cur.close()
    return [c for c in codes if c]


def coverage(conn: Any) -> dict:
    """수집 현황 한 눈에 — 종목 수·행 수·기간.

    Returns:
        ``{"codes": int, "rows": int, "start": Timestamp|None, "end": Timestamp|None}``.
        비어 있으면 0 과 ``None``.
    """
    cur = conn.cursor()
    cur.execute("select count(distinct code), count(*), min(ts), max(ts) from minute_bars")
    row = cur.fetchone()
    cur.close()
    if row is None:
        return {"codes": 0, "rows": 0, "start": None, "end": None}
    n_codes, n_rows, lo, hi = row
    return {
        "codes": int(n_codes or 0),
        "rows": int(n_rows or 0),
        "start": None if lo is None else pd.Timestamp(lo),
        "end": None if hi is None else pd.Timestamp(hi),
    }
