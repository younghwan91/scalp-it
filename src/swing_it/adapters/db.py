"""DB 읽기 어댑터 — SQL 은 이 파일에만 존재한다.

읽기 전용이다. swing-it 은 어떤 테이블에도 쓰지 않는다.
가격은 반드시 ``daily_bars_adjusted`` 에서 읽는다. ``daily_bars`` 는 분할 미반영이다.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pandas as pd

_PG_PREFIXES = ("postgresql://", "postgres://")

_PRICE_COLUMNS = ["code", "date", "open", "close", "trade_value"]
_SUPPLY_COLUMNS = [
    "code", "date", "individual", "foreign_", "institution",
    "penfnd_etc", "acc_trde_qty",
]


def connect(dsn: str | None = None) -> Any:
    """DSN 또는 ``KR_QUANT_DB`` 로 접속한다. postgres 면 psycopg2, 아니면 sqlite."""
    target = dsn or os.environ.get("KR_QUANT_DB")
    if not target:
        raise ValueError("DSN 이 없다. 인자로 넘기거나 KR_QUANT_DB 를 설정하라.")
    if target.startswith(_PG_PREFIXES):
        import psycopg2  # 지연 임포트 — sqlite 전용 사용자는 설치할 필요가 없다

        return psycopg2.connect(target)
    return sqlite3.connect(target)


def _placeholder(conn: Any) -> str:
    return "%s" if conn.__class__.__module__.startswith("psycopg2") else "?"


def _query(conn: Any, sql: str, params: tuple, columns: list[str]) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=columns)
    cur.close()
    df["date"] = pd.to_datetime(df["date"])
    for col in columns:
        if col not in ("code", "date"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_prices(conn: Any, start: str, end: str) -> pd.DataFrame:
    """``daily_bars_adjusted`` 에서 수정주가를 읽는다 (분할 반영본)."""
    ph = _placeholder(conn)
    sql = (
        "select code, date, open, close, trade_value from daily_bars_adjusted "
        f"where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), _PRICE_COLUMNS)


def load_supply_demand(conn: Any, start: str, end: str) -> pd.DataFrame:
    """투자자별 순매수를 읽는다. 개인·외국인·기관계·연기금만 쓴다."""
    ph = _placeholder(conn)
    sql = (
        "select code, date, individual, foreign_, institution, penfnd_etc, acc_trde_qty "
        f"from supply_demand where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), _SUPPLY_COLUMNS)
