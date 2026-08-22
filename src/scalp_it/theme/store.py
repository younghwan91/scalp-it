"""테마 스냅샷 저장소 — ``theme_groups`` · ``theme_members`` 쓰기 SQL 은 이 파일에만 있다.

키움 테마 API 는 현재 소속만 준다. 짝꿍매매 검증에는 "그날 기준으로 같은 테마였나"
가 필요한데, 그건 **매일 스냅샷을 쌓아야만** 룩어헤드 없이 복원된다. 그래서 두
테이블 모두 ``snapshot_date`` 가 PK 의 첫 컬럼이다.

정규화한 컬럼 외에 응답 원문을 ``raw`` (json) 에 통째로 남긴다. 키움이 필드를
바꾸거나 우리가 지금 안 뽑은 필드가 나중에 필요해져도 원문은 살아 있다.

스키마 문자열은 postgres 와 sqlite 양쪽에서 그대로 실행된다 (``intraday/store.py``
와 같은 규칙). ``jsonb`` 는 postgres 정식 타입이고 sqlite 는 모르는 타입 이름을
NUMERIC 친화성으로 받아들여 JSON 문자열을 그대로 텍스트로 보관한다. 접속은
``intraday.store.connect`` 를 그대로 쓴다 — 같은 ``KR_QUANT_DB`` 다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from ..intraday.store import _placeholder, connect

__all__ = [
    "connect",
    "count_groups",
    "count_members",
    "init_theme_tables",
    "normalize_snapshot_date",
    "upsert_theme_groups",
    "upsert_theme_members",
]

GROUP_COLUMNS: list[str] = [
    "snapshot_date", "theme_code", "theme_name", "stk_num", "flu_rt", "dt_prft_rt",
    "rising_stk_num", "fall_stk_num", "main_stk", "raw",
]
MEMBER_COLUMNS: list[str] = [
    "snapshot_date", "theme_code", "code", "name", "cur_prc", "flu_rt",
    "acc_trde_qty", "dt_prft_rt_n", "raw",
]

_CREATE_GROUPS = """
CREATE TABLE IF NOT EXISTS theme_groups (
    snapshot_date  date NOT NULL,
    theme_code     text NOT NULL,
    theme_name     text,
    stk_num        integer,
    flu_rt         double precision,
    dt_prft_rt     double precision,
    rising_stk_num integer,
    fall_stk_num   integer,
    main_stk       text,
    raw            jsonb,
    PRIMARY KEY (snapshot_date, theme_code)
)
"""
_CREATE_MEMBERS = """
CREATE TABLE IF NOT EXISTS theme_members (
    snapshot_date  date NOT NULL,
    theme_code     text NOT NULL,
    code           text NOT NULL,
    name           text,
    cur_prc        bigint,
    flu_rt         double precision,
    acc_trde_qty   bigint,
    dt_prft_rt_n   double precision,
    raw            jsonb,
    PRIMARY KEY (snapshot_date, theme_code, code)
)
"""
#: "이 종목이 그날 어느 테마에 있었나" 가 짝꿍 검증의 주 조회다.
_CREATE_MEMBER_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_theme_members_code ON theme_members(code, snapshot_date)"
)

_UPSERT_GROUPS = (
    "INSERT INTO theme_groups "
    "(snapshot_date, theme_code, theme_name, stk_num, flu_rt, dt_prft_rt, "
    "rising_stk_num, fall_stk_num, main_stk, raw) "
    "VALUES ({ph}) "
    "ON CONFLICT (snapshot_date, theme_code) DO UPDATE SET "
    "theme_name=excluded.theme_name, stk_num=excluded.stk_num, flu_rt=excluded.flu_rt, "
    "dt_prft_rt=excluded.dt_prft_rt, rising_stk_num=excluded.rising_stk_num, "
    "fall_stk_num=excluded.fall_stk_num, main_stk=excluded.main_stk, raw=excluded.raw"
)
_UPSERT_MEMBERS = (
    "INSERT INTO theme_members "
    "(snapshot_date, theme_code, code, name, cur_prc, flu_rt, acc_trde_qty, dt_prft_rt_n, raw) "
    "VALUES ({ph}) "
    "ON CONFLICT (snapshot_date, theme_code, code) DO UPDATE SET "
    "name=excluded.name, cur_prc=excluded.cur_prc, flu_rt=excluded.flu_rt, "
    "acc_trde_qty=excluded.acc_trde_qty, dt_prft_rt_n=excluded.dt_prft_rt_n, raw=excluded.raw"
)


def normalize_snapshot_date(value: Any) -> str:
    """``--date`` 인자·date·datetime 을 ``YYYY-MM-DD`` 문자열로 통일한다.

    postgres 의 ``date`` 컬럼과 sqlite 의 텍스트 비교 양쪽에서 같은 형태여야
    ``(snapshot_date, …)`` PK 가 제 구실을 한다.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value if value is not None else "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"날짜는 YYYYMMDD 또는 YYYY-MM-DD 여야 한다: {value!r}")
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def init_theme_tables(conn: Any) -> Any:
    """두 테이블과 종목 인덱스를 만든다. 이미 있으면 아무 일도 없다."""
    cur = conn.cursor()
    cur.execute(_CREATE_GROUPS)
    cur.execute(_CREATE_MEMBERS)
    cur.execute(_CREATE_MEMBER_IDX)
    cur.close()
    conn.commit()
    return conn


def _bind_raw(value: Any) -> Any:
    """원문 dict 를 JSON 문자열로 넘긴다. postgres 는 문자열→jsonb 암시 캐스트가 된다."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def upsert_theme_groups(conn: Any, snapshot_date: Any, rows: Iterable[dict]) -> int:
    """``(snapshot_date, theme_code)`` 기준 멱등 upsert. 적재 행 수를 돌려준다."""
    day = normalize_snapshot_date(snapshot_date)
    payload: list[Sequence[Any]] = []
    for row in rows:
        payload.append(
            (
                day,
                str(row["theme_code"]),
                row.get("theme_name"),
                _opt_int(row.get("stk_num")),
                _opt_float(row.get("flu_rt")),
                _opt_float(row.get("dt_prft_rt")),
                _opt_int(row.get("rising_stk_num")),
                _opt_int(row.get("fall_stk_num")),
                row.get("main_stk"),
                _bind_raw(row.get("raw")),
            )
        )
    if not payload:
        return 0
    ph = _placeholder(conn)
    sql = _UPSERT_GROUPS.format(ph=",".join([ph] * len(GROUP_COLUMNS)))
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def upsert_theme_members(conn: Any, snapshot_date: Any, rows: Iterable[dict]) -> int:
    """``(snapshot_date, theme_code, code)`` 기준 멱등 upsert. 적재 행 수를 돌려준다."""
    day = normalize_snapshot_date(snapshot_date)
    payload: list[Sequence[Any]] = []
    for row in rows:
        payload.append(
            (
                day,
                str(row["theme_code"]),
                str(row["code"]),
                row.get("name"),
                _opt_int(row.get("cur_prc")),
                _opt_float(row.get("flu_rt")),
                _opt_int(row.get("acc_trde_qty")),
                _opt_float(row.get("dt_prft_rt_n")),
                _bind_raw(row.get("raw")),
            )
        )
    if not payload:
        return 0
    ph = _placeholder(conn)
    sql = _UPSERT_MEMBERS.format(ph=",".join([ph] * len(MEMBER_COLUMNS)))
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def _count(conn: Any, table: str, snapshot_date: Any | None) -> int:
    cur = conn.cursor()
    if snapshot_date is None:
        cur.execute(f"select count(*) from {table}")
    else:
        ph = _placeholder(conn)
        cur.execute(
            f"select count(*) from {table} where snapshot_date = {ph}",
            (normalize_snapshot_date(snapshot_date),),
        )
    row = cur.fetchone()
    cur.close()
    return int(row[0]) if row and row[0] is not None else 0


def count_groups(conn: Any, snapshot_date: Any | None = None) -> int:
    """적재된 테마 수. 날짜를 주면 그 스냅샷만."""
    return _count(conn, "theme_groups", snapshot_date)


def count_members(conn: Any, snapshot_date: Any | None = None) -> int:
    """적재된 테마-종목 쌍 수. 날짜를 주면 그 스냅샷만."""
    return _count(conn, "theme_members", snapshot_date)
