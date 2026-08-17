"""분봉 저장소 — ``minute_bars`` 쓰기 SQL 은 이 파일에만 존재한다.

공시(``dart/store.py``)와 달리 분봉은 sqlite 가 아니라 **TimescaleDB** 에 쓴다.
분봉은 종목당 하루 380행이라 규모가 다르고, 후속 백테스트가 일봉과 조인해야
하므로 같은 DB 에 있어야 한다. 접속 대상은 ``KR_QUANT_DB`` 다.

스키마 문자열은 postgres 와 sqlite 양쪽에서 그대로 실행된다. 테스트는 sqlite
fixture 로 돌리고 운영은 postgres 로 돌리므로 둘이 갈라지면 테스트가 무의미해진다.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

_PG_PREFIXES = ("postgresql://", "postgres://")

MINUTE_COLUMNS: list[str] = [
    "code", "ts", "open", "high", "low", "close", "volume", "trade_value",
]

#: ``timestamp`` · ``double precision`` · ``bigint`` 는 postgres 정식 타입이고,
#: sqlite 는 타입 이름을 친화성 규칙으로 받아들이므로 그대로 통한다.
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS minute_bars (
    code        text NOT NULL,
    ts          timestamp NOT NULL,
    open        double precision,
    high        double precision,
    low         double precision,
    close       double precision,
    volume      bigint,
    trade_value double precision,
    PRIMARY KEY (code, ts)
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_minute_bars_ts ON minute_bars(ts)"

_UPSERT = (
    "INSERT INTO minute_bars "
    "(code, ts, open, high, low, close, volume, trade_value) "
    "VALUES ({ph}) "
    "ON CONFLICT (code, ts) DO UPDATE SET "
    "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, "
    "volume=excluded.volume, trade_value=excluded.trade_value"
)


def is_postgres(conn: Any) -> bool:
    return conn.__class__.__module__.startswith("psycopg2")


def _placeholder(conn: Any) -> str:
    return "%s" if is_postgres(conn) else "?"


def connect(dsn: str | None = None) -> Any:
    """DSN 또는 ``KR_QUANT_DB`` 로 접속한다. postgres 면 psycopg2, 아니면 sqlite."""
    target = dsn or os.environ.get("KR_QUANT_DB")
    if not target:
        raise ValueError("DSN 이 없다. 인자로 넘기거나 KR_QUANT_DB 를 설정하라.")
    if target.startswith(_PG_PREFIXES):
        import psycopg2  # 지연 임포트 — sqlite 전용 사용자는 설치할 필요가 없다

        return psycopg2.connect(target)
    return sqlite3.connect(target)


def init_minute_bars(conn: Any) -> Any:
    """``minute_bars`` 와 ``ts`` 인덱스를 만든다. 이미 있으면 아무 일도 없다."""
    cur = conn.cursor()
    cur.execute(_CREATE_TABLE)
    cur.execute(_CREATE_INDEX)
    cur.close()
    conn.commit()
    return conn


def _bind_ts(conn: Any, value: Any) -> Any:
    """``ts`` 를 드라이버가 받는 형태로 바꾼다.

    psycopg2 는 ``datetime`` 을 그대로 받지만, sqlite3 는 3.12 부터 datetime
    어댑터가 폐기 예고 상태라 문자열로 넘긴다.
    """
    if isinstance(value, str) or is_postgres(conn):
        return value
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime()


def upsert_minute_bars(conn: Any, rows: Iterable[dict]) -> int:
    """``(code, ts)`` 기준 멱등 upsert. 적재 행 수를 돌려준다."""
    payload: list[Sequence[Any]] = []
    for row in rows:
        payload.append(
            (
                str(row["code"]),
                _bind_ts(conn, row["ts"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
                float(row["trade_value"]),
            )
        )
    if not payload:
        return 0
    ph = _placeholder(conn)
    sql = _UPSERT.format(ph=",".join([ph] * len(MINUTE_COLUMNS)))
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def latest_ts(conn: Any, code: str) -> datetime | None:
    """이미 저장된 그 종목의 최신 ``ts``. 없으면 ``None``.

    이어받기의 기준점이다 — 이 값보다 최신인 봉만 요청한다.
    """
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(f"select max(ts) from minute_bars where code = {ph}", (str(code),))
    row = cur.fetchone()
    cur.close()
    return _to_datetime(row[0]) if row else None


def count_bars(conn: Any, code: str | None = None) -> int:
    """적재된 분봉 수. ``code`` 를 주면 그 종목만."""
    cur = conn.cursor()
    if code is None:
        cur.execute("select count(*) from minute_bars")
    else:
        ph = _placeholder(conn)
        cur.execute(f"select count(*) from minute_bars where code = {ph}", (str(code),))
    row = cur.fetchone()
    cur.close()
    return int(row[0]) if row and row[0] is not None else 0


def top_universe(conn: Any, limit: int, *, lookback_days: int = 20) -> list[str]:
    """``daily_bars_adjusted`` 최근 ``lookback_days`` 일 평균 거래대금 상위 종목.

    '최근 20일'은 달력이 아니라 **거래일** 기준이다. 휴장·연휴가 끼면 달력
    20일은 거래일 13일밖에 안 되는 구간이 실제로 있다.
    """
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        "select distinct date from daily_bars_adjusted order by date desc limit " + ph,
        (int(lookback_days),),
    )
    dates = [r[0] for r in cur.fetchall()]
    if not dates:
        cur.close()
        return []
    cutoff = min(dates)
    cur.execute(
        "select code, avg(trade_value) as adv from daily_bars_adjusted "
        f"where date >= {ph} and trade_value is not null "
        f"group by code order by adv desc limit {ph}",
        (cutoff, int(limit)),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes
