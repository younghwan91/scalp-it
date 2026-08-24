"""실시간 틱·호가 저장소 — ``ticks`` · ``quotes`` 쓰기 SQL 은 이 파일에만 있다.

분봉(``minute_bars``)과 같은 ``KR_QUANT_DB`` 에 쓴다. 후속 백테스트가 분봉·일봉과
조인해야 하고, 틱은 분봉을 만든 원자료이므로 같은 DB 에 있어야 한다.

설계 원칙은 :mod:`scalp_it.intraday.store` 와 같다 — 스키마 문자열은 postgres 와
sqlite 양쪽에서 그대로 실행되고, 접속·placeholder·ts 바인딩 헬퍼를 그 모듈에서
재사용한다. 테스트는 sqlite fixture 로, 운영은 postgres(가능하면 TimescaleDB
하이퍼테이블)로 돈다.

키움 REAL 프레임의 FID 매핑은 장중 실측으로 확정해야 하므로(20번 문서), 정규화한
컬럼과 **별도로 원문 ``values`` 딕셔너리를 ``raw`` 에 JSON 으로 통째 보존**한다.
FID 를 잘못 매핑했더라도 raw 에서 되살릴 수 있다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from scalp_it.intraday.store import _bind_ts, _placeholder, connect, is_postgres

__all__ = [
    "QUOTE_COLUMNS",
    "TICK_COLUMNS",
    "VI_COLUMNS",
    "connect",
    "count_quotes",
    "count_ticks",
    "count_vi_events",
    "event_candidates",
    "init_realtime_tables",
    "insert_quotes",
    "insert_ticks",
    "insert_vi_events",
    "is_postgres",
]

# --------------------------------------------------------------------- 스키마

TICK_COLUMNS: list[str] = [
    "code", "ts", "seq", "price", "volume", "side", "strength",
    "cum_volume", "best_bid", "best_ask", "raw",
]

#: 10호가는 컬럼을 넓게 편다(배열 대신). sqlite 는 배열 타입이 없고, 넓은 컬럼이
#: 백테스트에서 인덱싱·필터가 쉽다.
_QUOTE_LEVELS = 10
_QUOTE_LEVEL_COLUMNS: list[str] = (
    [f"bid{i}" for i in range(1, _QUOTE_LEVELS + 1)]
    + [f"bidqty{i}" for i in range(1, _QUOTE_LEVELS + 1)]
    + [f"ask{i}" for i in range(1, _QUOTE_LEVELS + 1)]
    + [f"askqty{i}" for i in range(1, _QUOTE_LEVELS + 1)]
)
QUOTE_COLUMNS: list[str] = (
    ["code", "ts"]
    + _QUOTE_LEVEL_COLUMNS
    + ["total_bid_qty", "total_ask_qty", "vi_flag", "raw"]
)

_CREATE_TICKS = """
CREATE TABLE IF NOT EXISTS ticks (
    code       text NOT NULL,
    ts         timestamp NOT NULL,
    seq        integer NOT NULL,
    price      integer,
    volume     bigint,
    side       integer,
    strength   double precision,
    cum_volume bigint,
    best_bid   integer,
    best_ask   integer,
    raw        text,
    PRIMARY KEY (code, ts, seq)
)
"""

_CREATE_QUOTES = (
    "CREATE TABLE IF NOT EXISTS quotes (\n"
    "    code text NOT NULL,\n"
    "    ts   timestamp NOT NULL,\n"
    + "".join(f"    {c} integer,\n" for c in _QUOTE_LEVEL_COLUMNS)
    + "    total_bid_qty bigint,\n"
    "    total_ask_qty bigint,\n"
    "    vi_flag integer,\n"
    "    raw text,\n"
    "    PRIMARY KEY (code, ts)\n"
    ")"
)

#: VI(변동성완화장치) 발동·해제는 초당 흐르는 체결·호가와 성격이 다른 **이벤트**라
#: quotes 에 붙이지 않고 별도 테이블로 둔다. 종목당 하루 몇 건뿐이라 저장 비용도
#: 무시할 만하다. FID 매핑은 키움 문서(koapy realtype ``1h``) 기준이고 장중 REAL
#: 프레임으로 아직 최종 대조하지 못했으므로(VI 는 드물다) 원문을 ``raw`` 에 통째
#: 보존한다 — 오매핑이어도 raw 에서 되살린다.
VI_COLUMNS: list[str] = [
    "code", "ts", "vi_type", "vi_kind", "base_price",
    "trigger_price", "gap_rate", "release_time", "raw",
]

_CREATE_VI_EVENTS = """
CREATE TABLE IF NOT EXISTS vi_events (
    code          text NOT NULL,
    ts            timestamp NOT NULL,
    vi_type       text NOT NULL,
    vi_kind       text,
    base_price    integer,
    trigger_price integer,
    gap_rate      double precision,
    release_time  text,
    raw           text,
    PRIMARY KEY (code, ts, vi_type)
)
"""

_CREATE_TICKS_TS_IDX = "CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts)"
_CREATE_QUOTES_TS_IDX = "CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes(ts)"
_CREATE_VI_TS_IDX = "CREATE INDEX IF NOT EXISTS idx_vi_events_ts ON vi_events(ts)"


def _try_hypertable(conn: Any, table: str) -> None:
    """postgres·TimescaleDB 면 하이퍼테이블로 승격한다. 안 되면 일반 테이블로 둔다.

    TimescaleDB 확장이 없거나 권한이 없으면 조용히 넘어간다 — 일반 테이블로도
    수집·조회는 그대로 된다. 확장 미설치 환경(테스트 sqlite 포함)을 막지 않는다.
    """
    if not is_postgres(conn):
        return
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT create_hypertable(%s, 'ts', "
            "if_not_exists => TRUE, migrate_data => TRUE)",
            (table,),
        )
        conn.commit()
    except Exception:  # noqa: BLE001 — 확장 없음·이미 하이퍼테이블 등 전부 무해
        conn.rollback()
    finally:
        cur.close()


def init_realtime_tables(conn: Any) -> Any:
    """``ticks`` · ``quotes`` · ``vi_events`` 와 ``ts`` 인덱스를 만든다.

    이미 있으면 무해하다.
    """
    cur = conn.cursor()
    cur.execute(_CREATE_TICKS)
    cur.execute(_CREATE_QUOTES)
    cur.execute(_CREATE_VI_EVENTS)
    cur.execute(_CREATE_TICKS_TS_IDX)
    cur.execute(_CREATE_QUOTES_TS_IDX)
    cur.execute(_CREATE_VI_TS_IDX)
    cur.close()
    conn.commit()
    _try_hypertable(conn, "ticks")
    _try_hypertable(conn, "quotes")
    _try_hypertable(conn, "vi_events")
    return conn


# --------------------------------------------------------------------- 쓰기


def _raw_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def insert_ticks(conn: Any, rows: Iterable[dict]) -> int:
    """체결(0B) 행을 배치 insert 한다. ``(code, ts, seq)`` 중복은 건너뛴다.

    틱은 갱신 대상이 아니라 한 번 들어오면 불변이므로 ON CONFLICT DO NOTHING 이다.
    재연결 뒤 같은 프레임이 다시 오거나 재시작해도 멱등이다.
    """
    payload: list[Sequence[Any]] = []
    for row in rows:
        payload.append(
            (
                str(row["code"]),
                _bind_ts(conn, row["ts"]),
                int(row["seq"]),
                _opt_int(row.get("price")),
                _opt_int(row.get("volume")),
                _opt_int(row.get("side")),
                _opt_float(row.get("strength")),
                _opt_int(row.get("cum_volume")),
                _opt_int(row.get("best_bid")),
                _opt_int(row.get("best_ask")),
                _raw_text(row.get("raw")),
            )
        )
    if not payload:
        return 0
    ph = _placeholder(conn)
    sql = (
        f"INSERT INTO ticks ({', '.join(TICK_COLUMNS)}) "
        f"VALUES ({','.join([ph] * len(TICK_COLUMNS))}) "
        "ON CONFLICT (code, ts, seq) DO NOTHING"
    )
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def insert_quotes(conn: Any, rows: Iterable[dict]) -> int:
    """호가(0D) 행을 배치 insert 한다. ``(code, ts)`` 중복은 최신 값으로 덮는다.

    같은 초에 호가가 여러 번 갱신되면 마지막 스냅샷만 남긴다(1초 다운샘플).
    """
    payload: list[Sequence[Any]] = []
    for row in rows:
        values: list[Any] = [str(row["code"]), _bind_ts(conn, row["ts"])]
        values += [_opt_int(row.get(c)) for c in _QUOTE_LEVEL_COLUMNS]
        values += [
            _opt_int(row.get("total_bid_qty")),
            _opt_int(row.get("total_ask_qty")),
            _opt_int(row.get("vi_flag")),
            _raw_text(row.get("raw")),
        ]
        payload.append(tuple(values))
    if not payload:
        return 0
    ph = _placeholder(conn)
    updatable = [c for c in QUOTE_COLUMNS if c not in ("code", "ts")]
    sql = (
        f"INSERT INTO quotes ({', '.join(QUOTE_COLUMNS)}) "
        f"VALUES ({','.join([ph] * len(QUOTE_COLUMNS))}) "
        "ON CONFLICT (code, ts) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in updatable)
    )
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def insert_vi_events(conn: Any, rows: Iterable[dict]) -> int:
    """VI 발동·해제(1h) 행을 배치 insert 한다. ``(code, ts, vi_type)`` 중복은 덮는다.

    같은 종목·같은 시각·같은 발동구분이 재연결로 다시 오면 최신 값으로 갱신한다
    (멱등). VI 는 드문 이벤트라 배치는 대개 비어 있거나 한두 건이다.
    """
    payload: list[Sequence[Any]] = []
    for row in rows:
        payload.append(
            (
                str(row["code"]),
                _bind_ts(conn, row["ts"]),
                str(row["vi_type"]),
                _opt_str(row.get("vi_kind")),
                _opt_int(row.get("base_price")),
                _opt_int(row.get("trigger_price")),
                _opt_float(row.get("gap_rate")),
                _opt_str(row.get("release_time")),
                _raw_text(row.get("raw")),
            )
        )
    if not payload:
        return 0
    ph = _placeholder(conn)
    updatable = [c for c in VI_COLUMNS if c not in ("code", "ts", "vi_type")]
    sql = (
        f"INSERT INTO vi_events ({', '.join(VI_COLUMNS)}) "
        f"VALUES ({','.join([ph] * len(VI_COLUMNS))}) "
        "ON CONFLICT (code, ts, vi_type) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in updatable)
    )
    cur = conn.cursor()
    cur.executemany(sql, payload)
    cur.close()
    conn.commit()
    return len(payload)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


# --------------------------------------------------------------------- 조회


def count_ticks(conn: Any, code: str | None = None) -> int:
    return _count(conn, "ticks", code)


def count_quotes(conn: Any, code: str | None = None) -> int:
    return _count(conn, "quotes", code)


def count_vi_events(conn: Any, code: str | None = None) -> int:
    return _count(conn, "vi_events", code)


def _count(conn: Any, table: str, code: str | None) -> int:
    cur = conn.cursor()
    if code is None:
        cur.execute(f"select count(*) from {table}")
    else:
        ph = _placeholder(conn)
        cur.execute(f"select count(*) from {table} where code = {ph}", (str(code),))
    row = cur.fetchone()
    cur.close()
    return int(row[0]) if row and row[0] is not None else 0


def event_candidates(conn: Any, limit: int, *, tv_floor: float = 5_000.0) -> list[str]:
    """내일 구독할 '전일 이벤트 후보' 종목.

    ``--auto`` 의 근간이다. 두 갈래를 합친다:

    1. **전일 거래대금 상위** — :func:`scalp_it.intraday.store.top_universe` 와 같은
       ADV 기준(최근 20 거래일 평균)으로 유동성 있는 종목.
    2. **전일 급등주(상따/돌파 프록시)** — 가장 최근 거래일 종가가 그 전 거래일
       종가 대비 크게 오른 종목. 상따·돌파 이벤트는 대개 전일 강세의 연장이라,
       스캐너 스크립트를 통째로 돌리지 않고도 후보를 싸게 잡는다.

    ``daily_bars_adjusted`` 만 읽고, ``trade_value`` 하한(백만원 단위, 기본 50억)을
    둔다. 두 리스트를 순서대로 합쳐 중복을 제거하고 ``limit`` 개로 자른다.

    거래대금 상위와 급등주를 절반씩 섞는다 — 어느 한쪽이 후보를 독식하지 않게.
    """
    from scalp_it.intraday.store import top_universe

    limit = max(0, int(limit))
    if limit == 0:
        return []

    half = max(1, limit // 2)
    liquid = top_universe(conn, half)
    movers = _prev_day_movers(conn, limit, tv_floor=tv_floor)

    ordered: list[str] = []
    seen: set[str] = set()
    # 급등주와 유동주를 번갈아 넣어 한쪽 편향을 막는다.
    for a, b in zip(movers, liquid):
        for code in (a, b):
            if code and code not in seen:
                seen.add(code)
                ordered.append(code)
    for code in movers + liquid:  # 길이가 다른 쪽 잔여분
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered[:limit]


def _prev_day_movers(conn: Any, limit: int, *, tv_floor: float) -> list[str]:
    """가장 최근 거래일 종가 상승률 상위. 거래대금 하한 적용."""
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute("select distinct date from daily_bars_adjusted order by date desc limit 2")
    dates = [r[0] for r in cur.fetchall()]
    if len(dates) < 2:
        cur.close()
        return []
    last, prev = dates[0], dates[1]
    cur.execute(
        "select t.code from daily_bars_adjusted t "
        "join daily_bars_adjusted y on y.code = t.code "
        f"where t.date = {ph} and y.date = {ph} "
        "and t.close is not null and y.close is not null and y.close > 0 "
        f"and t.trade_value is not null and t.trade_value >= {ph} "
        f"order by (t.close / y.close) desc limit {ph}",
        (last, prev, float(tv_floor), int(limit)),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes
