"""분봉 저장소 — ``minute_bars`` 쓰기 SQL 은 이 파일에만 존재한다.

공시(``dart/store.py``)와 달리 분봉은 sqlite 가 아니라 **TimescaleDB** 에 쓴다.
분봉은 종목당 하루 380행이라 규모가 다르고, 후속 백테스트가 일봉과 조인해야
하므로 같은 DB 에 있어야 한다. 접속 대상은 ``KR_QUANT_DB`` 다.

스키마 문자열은 postgres 와 sqlite 양쪽에서 그대로 실행된다. 테스트는 sqlite
fixture 로 돌리고 운영은 postgres 로 돌리므로 둘이 갈라지면 테스트가 무의미해진다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from ..adapters.db import connect, is_postgres, _placeholder  # noqa: F401 — re-exported

#: 압축 대상: (code, ts) 로만 필터하는 조회가 대부분이라 segmentby='code',
#: 최신 우선 조회가 많아 orderby='ts DESC'. compress_after=14일은 실측
#: 근거가 있다 — upsert 는 그날 안에서만 일어나고(장중 재수집이 최신 값으로
#: 덮어씀), 다음 거래일부터는 그 청크에 쓰기가 없다. 14일이면 재수집 창을
#: 넉넉히 지나 압축해도 decompress→update→recompress churn 이 없다
#: (quant-airflow CLAUDE.md §3 이 경고하는 그 패턴).
_COMPRESS_AFTER = "14 days"

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


def _try_hypertable_and_compression(conn: Any) -> None:
    """postgres·TimescaleDB 면 하이퍼테이블 + 압축 정책을 건다.

    운영 DB 는 2026-09 에 psql 로 수동 승격됐고 이 함수가 그 상태를 코드에
    반영한다 — 반영 안 하면 새 DB 를 세울 때 minute_bars 만 일반 테이블로
    남는 드리프트가 난다(quant-airflow 가 겪은 것과 같은 부류의 사고).
    확장 미설치·권한 없음·이미 설정됨은 전부 무해하게 넘어간다.
    """
    if not is_postgres(conn):
        return
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT create_hypertable(%s, 'ts', "
            "if_not_exists => TRUE, migrate_data => TRUE)",
            ("minute_bars",),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
    try:
        cur.execute(
            "ALTER TABLE minute_bars SET ("
            "timescaledb.compress, "
            "timescaledb.compress_segmentby = 'code', "
            "timescaledb.compress_orderby = 'ts DESC')"
        )
        cur.execute(
            "SELECT add_compression_policy('minute_bars', INTERVAL %s)",
            (_COMPRESS_AFTER,),
        )
        conn.commit()
    except Exception:  # noqa: BLE001 — 이미 설정됨 등
        conn.rollback()
    finally:
        cur.close()


def init_minute_bars(conn: Any) -> Any:
    """``minute_bars`` 와 ``ts`` 인덱스를 만든다. 이미 있으면 아무 일도 없다."""
    cur = conn.cursor()
    cur.execute(_CREATE_TABLE)
    cur.execute(_CREATE_INDEX)
    cur.close()
    conn.commit()
    _try_hypertable_and_compression(conn)
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
