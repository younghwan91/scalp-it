"""DB 읽기 어댑터 — SQL 은 이 파일에만 존재한다.

읽기 전용이다. scalp-it 은 어떤 테이블에도 쓰지 않는다.
가격은 반드시 ``daily_bars_adjusted`` 에서 읽는다. ``daily_bars`` 는 분할 미반영이다.

kr-quant(``storage.py``)·krx-signal-engine(``quant_airflow_db_client.py``)도
같은 TimescaleDB 를 읽는 자기 클라이언트를 따로 갖고 있다 — 이건 사고가
아니라 kr-quant 쪽이 이미 문서화해둔 의도적 결정이다: "수집·적재(쓰기)
쪽만 quant-airflow 로 합치고, 읽기 클라이언트는 독립된 사본으로 둔다"
(kr-quant/src/kr_quant/storage.py 상단 참고, 2026-09-12 조사로 확인). 그래서
여기를 공유 패키지로 뽑는 리팩터는 하지 않는다.
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


def is_postgres(conn: Any) -> bool:
    return conn.__class__.__module__.startswith("psycopg2")


def _placeholder(conn: Any) -> str:
    return "%s" if is_postgres(conn) else "?"


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


def load_daily_bars_raw(conn: Any, start: str, end: str) -> pd.DataFrame:
    """``daily_bars`` 에서 원주가(분할 미반영)를 읽는다.

    PEAD 진입일 상한가 판정(익일 시가 / 전일 종가 >= 1.29)은 원주가로만 옳다.
    수정주가는 분할이 반영돼 상한가 비율이 왜곡된다.
    """
    ph = _placeholder(conn)
    sql = (
        "select code, date, open, close from daily_bars "
        f"where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), ["code", "date", "open", "close"])


_EARNINGS_COLUMNS = ["code", "period", "avail_date", "op_income", "op_income_prior"]


def load_earnings(conn: Any) -> pd.DataFrame:
    """``earnings`` 에서 PEAD 시그널 원자료를 읽는다.

    ``avail_date`` 는 정보 도착 시점(PIT). ``op_income_prior`` 는 직전연도
    동일분기(YoY) 영업이익이다 (사전등록 23번에서 확인).
    """
    cur = conn.cursor()
    cur.execute(
        "select code, period, avail_date, op_income, op_income_prior from earnings"
    )
    df = pd.DataFrame(cur.fetchall(), columns=_EARNINGS_COLUMNS)
    cur.close()
    df["code"] = df["code"].astype(str).str.strip()
    df["period"] = df["period"].astype(str)
    df["avail_date"] = pd.to_datetime(df["avail_date"], errors="coerce")
    for col in ("op_income", "op_income_prior"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_supply_demand(conn: Any, start: str, end: str) -> pd.DataFrame:
    """투자자별 순매수를 읽는다. 개인·외국인·기관계·연기금만 쓴다."""
    ph = _placeholder(conn)
    sql = (
        "select code, date, individual, foreign_, institution, penfnd_etc, acc_trde_qty "
        f"from supply_demand where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), _SUPPLY_COLUMNS)


_MASTER_COLUMNS = ["code", "name", "market", "sector", "last_trade_date", "is_delisted"]


def load_stock_master(conn: Any) -> pd.DataFrame:
    """상장·폐지 종목을 합친 종목 마스터.

    폐지 종목을 포함하는 이유는 생존편향이다 (스펙 §11). 과거 기사에 나오는
    회사 중 상당수는 지금 상장돼 있지 않고, 그들을 빼면 표본이
    '살아남은 회사' 쪽으로 기운다.

    **``last_trade_date`` 로 상장 여부를 판정하지 마라.** 실 DB 에서
    ``delisted_stocks`` 4,162 건 중 3,693 건(89%)이 이 값이 NULL 이다.
    폐지 여부는 출처 테이블로 정하며 ``is_delisted`` 로 노출한다.
    ``last_trade_date`` 는 있으면 쓰는 참고값이다.

    같은 코드가 양쪽에 있으면(실 DB 기준 17 건) 상장 쪽을 남긴다 — 재상장·코드
    재사용 사례이고, 현재 거래 가능한 것이 우선이다.
    """
    sql = (
        "select code, name, market, sector, null as last_trade_date, 0 as is_delisted "
        "from stocks "
        "union all "
        "select code, name, market, null as sector, last_trade_date, 1 as is_delisted "
        "from delisted_stocks"
    )
    cur = conn.cursor()
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall(), columns=_MASTER_COLUMNS)
    cur.close()
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce")
    df["is_delisted"] = df["is_delisted"].astype(bool)
    # 상장 행이 먼저 오므로 keep="first" 가 곧 '상장 우선'이다.
    df = df[df["code"] != ""].drop_duplicates(subset="code", keep="first")
    return df.sort_values("code").reset_index(drop=True)


_OHLCV_COLUMNS = [
    "code", "date", "open", "high", "low", "close", "volume", "trade_value",
]


def load_ohlcv(conn: Any, start: str, end: str) -> pd.DataFrame:
    """``daily_bars_adjusted`` 에서 OHLCV 전체를 읽는다 (종가베팅용).

    :func:`load_prices` 는 ``open``·``close``·``trade_value`` 만 준다. 종가베팅
    피처는 종가가 고가 근처인지(``close_pos``)를 봐야 해서 ``high``·``low`` 가,
    거래량 급증을 보려면 ``volume`` 이 필요하다.
    """
    ph = _placeholder(conn)
    sql = (
        "select code, date, open, high, low, close, volume, trade_value "
        f"from daily_bars_adjusted where date >= {ph} and date <= {ph} "
        "order by code, date"
    )
    return _query(conn, sql, (start, end), _OHLCV_COLUMNS)
