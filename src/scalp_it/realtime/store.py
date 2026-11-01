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

from scalp_it.intraday.store import _bind_ts, _placeholder, connect, is_postgres, pg_upsert, ts_key

__all__ = [
    "QUOTE_COLUMNS",
    "QUOTE_EVENT_COLUMNS",
    "TICK_COLUMNS",
    "VI_COLUMNS",
    "connect",
    "count_quote_events",
    "count_quotes",
    "count_ticks",
    "count_vi_events",
    "event_candidates",
    "init_quote_events_table",
    "init_realtime_tables",
    "insert_quote_events",
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

#: 0D 수신 **전부**를 남기는 이벤트 테이블. ``quotes`` 는 초당 마지막 스냅샷만 남기는데
#: (DO UPDATE), 2026-09-10 실측 수신 1,673,301건 중 615,226행(37%)만 남았다 — 대기열이
#: 빨리 줄어드는 바쁜 초일수록 더 많이 버린다. hftbacktest 식 이벤트 단위 대기열 모델에는
#: 초 안 순서가 필요해서 따로 쌓는다(``quotes`` 를 읽는 기존 연구 코드는 그대로 둔다).
#:
#: - ``qseq`` — 같은 ``(code, ts)`` 안 0D 수신 순서.
#: - ``ticks_before`` — 이 호가를 받기 전에 받은, 같은 ``(code, ts)`` 틱 수. 틱 PK 의
#:   ``seq`` 와 맞물려 **초 안 틱·호가 선후**를 복원한다(``ticks.seq < ticks_before`` 가 앞).
#:   0B·0D 는 같은 커넥션·같은 프레임 순서로 오므로 수신 순서가 곧 그 선후다.
#: - ``recv_ts`` — 로컬 수신 시각(마이크로초, KST naive). 거래소 ``ts`` 는 초 절삭이다.
#:
#: raw 는 두지 않는다(``quotes.raw`` 가 초 끝 원문을 이미 보존하고, 행 수가 3배라 무겁다).
QUOTE_EVENT_COLUMNS: list[str] = (
    ["code", "ts", "qseq", "ticks_before", "recv_ts"]
    + _QUOTE_LEVEL_COLUMNS
    + ["total_bid_qty", "total_ask_qty"]
)

_CREATE_QUOTE_EVENTS = (
    "CREATE TABLE IF NOT EXISTS quote_events (\n"
    "    code text NOT NULL,\n"
    "    ts   timestamp NOT NULL,\n"
    "    qseq integer NOT NULL,\n"
    "    ticks_before integer,\n"
    "    recv_ts timestamp,\n"
    + "".join(f"    {c} integer,\n" for c in _QUOTE_LEVEL_COLUMNS)
    + "    total_bid_qty bigint,\n"
    "    total_ask_qty bigint,\n"
    "    PRIMARY KEY (code, ts, qseq)\n"
    ")"
)
_CREATE_QUOTE_EVENTS_TS_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_quote_events_ts ON quote_events(ts)"
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


#: compress_after=14일 — ticks 는 ON CONFLICT DO NOTHING(불변, 즉시 안전)이고
#: quotes 는 같은 초 스냅샷을 그날 안에서만 DO UPDATE 로 덮어쓴다. 다음
#: 거래일부터 그 청크에 쓰기가 없으므로 14일 뒤 압축해도 재압축 churn 이
#: 없다(quant-airflow CLAUDE.md §3 이 경고하는 패턴). vi_events 는 하루
#: 몇 건뿐이라 압축 대상에서 뺀다 — 이득보다 정책 관리 비용이 크다.
#:
#: ⚠️ 압축된 청크는 timescaledb_information.chunks 조회에서 원래 청크명이
#: 빈 채로 잡히고 실데이터는 compress_hyper_*_chunk 로 옮겨간다 — 형제 레포
#: quant-airflow의 scripts/backup_to_gdrive.sh(heavy_chunk_excludes)가 이걸
#: 놓쳐 압축 데이터를 "core"(가벼운 일일 백업)에 그대로 실었다. 이 정책이
#: 들어온 날(2026-09-11, 이 파일의 압축 도입 커밋)에 08-24 이후 오래된 청크가
#: 한꺼번에 압축되면서 core 가 420MB → 1.84GB 로 뛰었고, 그게 09-13 에
#: 발견될 때까지 아무 신호도 없었다(에러가 아니라 파일이 커질 뿐이다).
#: 여기 압축 대상 테이블이나 compress_after 값을 바꾸면 그쪽 exclude 목록도
#: 같이 볼 것.
_COMPRESS_AFTER = "14 days"


def _try_compression(conn: Any, table: str, segmentby: str) -> None:
    """압축 설정 + 정책을 건다. 이미 걸려 있거나 확장이 없으면 조용히 넘어간다."""
    if not is_postgres(conn):
        return
    cur = conn.cursor()
    try:
        cur.execute(
            f"ALTER TABLE {table} SET ("
            "timescaledb.compress, "
            f"timescaledb.compress_segmentby = '{segmentby}', "
            "timescaledb.compress_orderby = 'ts DESC')"
        )
        cur.execute(
            "SELECT add_compression_policy(%s, INTERVAL %s)",
            (table, _COMPRESS_AFTER),
        )
        conn.commit()
    except Exception:  # noqa: BLE001 — 이미 설정됨 등 전부 무해
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
    _try_compression(conn, "ticks", "code")
    _try_compression(conn, "quotes", "code")
    init_quote_events_table(conn)
    return conn


def init_quote_events_table(conn: Any) -> bool:
    """``quote_events`` 를 만든다. 실패해도 예외를 삼키고 False — 틱·호가 수집을 막지 않는다.

    ⚠️ 압축 대상 테이블이 하나 늘었다 — quant-airflow ``scripts/backup_to_gdrive.sh`` 의
    ``HEAVY_TABLES`` 에도 넣어야 core 백업이 부풀지 않는다(위 ``_COMPRESS_AFTER`` 주석).
    """
    cur = conn.cursor()
    try:
        cur.execute(_CREATE_QUOTE_EVENTS)
        cur.execute(_CREATE_QUOTE_EVENTS_TS_IDX)
        conn.commit()
    except Exception:  # noqa: BLE001 — 부가 테이블: 없어도 기존 수집은 돈다
        conn.rollback()
        return False
    finally:
        cur.close()
    _try_hypertable(conn, "quote_events")
    _try_compression(conn, "quote_events", "code")
    return True


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
    head = f"INSERT INTO ticks ({', '.join(TICK_COLUMNS)}) VALUES "
    tail = " ON CONFLICT (code, ts, seq) DO NOTHING"
    cur = conn.cursor()
    if is_postgres(conn):
        # 행마다 왕복하지 않게 한 문장으로 보낸다. DO NOTHING 은 한 문장 안 중복 키도 받는다.
        from psycopg2.extras import execute_values

        execute_values(cur, head + "%s" + tail, payload, page_size=2000)
    else:
        ph = _placeholder(conn)
        cur.executemany(head + f"({','.join([ph] * len(TICK_COLUMNS))})" + tail, payload)
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
    updatable = [c for c in QUOTE_COLUMNS if c not in ("code", "ts")]
    head = f"INSERT INTO quotes ({', '.join(QUOTE_COLUMNS)}) VALUES "
    tail = " ON CONFLICT (code, ts) DO UPDATE SET " + ", ".join(
        f"{c}=excluded.{c}" for c in updatable
    )
    if is_postgres(conn):
        pg_upsert(conn, head, tail, payload, key=lambda r: (r[0], ts_key(r[1])))
    else:
        ph = _placeholder(conn)
        cur = conn.cursor()
        cur.executemany(head + f"({','.join([ph] * len(QUOTE_COLUMNS))})" + tail, payload)
        cur.close()
    conn.commit()
    return len(payload)


def _bind_recv_ts(conn: Any, value: Any) -> Any:
    """수신 시각은 마이크로초까지 보존한다(``_bind_ts`` 는 sqlite 에서 초로 자른다)."""
    if value is None or isinstance(value, str) or is_postgres(conn):
        return value
    return value.isoformat(sep=" ", timespec="microseconds")


def insert_quote_events(conn: Any, rows: Iterable[dict]) -> int:
    """0D 수신 전부를 insert 한다. ``(code, ts, qseq)`` 중복은 건너뛴다(불변 이벤트)."""
    payload: list[Sequence[Any]] = []
    for row in rows:
        recv = row.get("recv_ts")
        values: list[Any] = [
            str(row["code"]), _bind_ts(conn, row["ts"]), int(row["qseq"]),
            _opt_int(row.get("ticks_before")),
            _bind_recv_ts(conn, recv),
        ]
        values += [_opt_int(row.get(c)) for c in _QUOTE_LEVEL_COLUMNS]
        values += [_opt_int(row.get("total_bid_qty")), _opt_int(row.get("total_ask_qty"))]
        payload.append(tuple(values))
    if not payload:
        return 0
    head = f"INSERT INTO quote_events ({', '.join(QUOTE_EVENT_COLUMNS)}) VALUES "
    tail = " ON CONFLICT (code, ts, qseq) DO NOTHING"
    cur = conn.cursor()
    if is_postgres(conn):
        # executemany 는 행마다 왕복한다. flush 는 이벤트 루프 안 동기 호출이라, 초당 수십~
        # 수백 행을 LAN 너머로 한 줄씩 보내면 장 시작 폭주 때 감지기 프레임 처리가 밀린다.
        # 부가 수집이 라이브 경로를 늦추지 않게 한 문장으로 보낸다.
        from psycopg2.extras import execute_values

        execute_values(cur, head + "%s" + tail, payload, page_size=2000)
    else:
        ph = _placeholder(conn)
        cur.executemany(head + f"({','.join([ph] * len(QUOTE_EVENT_COLUMNS))})" + tail, payload)
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
    updatable = [c for c in VI_COLUMNS if c not in ("code", "ts", "vi_type")]
    head = f"INSERT INTO vi_events ({', '.join(VI_COLUMNS)}) VALUES "
    tail = " ON CONFLICT (code, ts, vi_type) DO UPDATE SET " + ", ".join(
        f"{c}=excluded.{c}" for c in updatable
    )
    if is_postgres(conn):
        pg_upsert(conn, head, tail, payload, key=lambda r: (r[0], ts_key(r[1]), r[2]))
    else:
        ph = _placeholder(conn)
        cur = conn.cursor()
        cur.executemany(head + f"({','.join([ph] * len(VI_COLUMNS))})" + tail, payload)
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


def count_quote_events(conn: Any, code: str | None = None) -> int:
    return _count(conn, "quote_events", code)


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


def _liquid_above_floor(conn: Any, tv_floor: float) -> list[str]:
    """최신 거래일 거래대금이 하한 이상인 종목코드."""
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute("select max(date) from daily_bars_adjusted")
    row = cur.fetchone()
    if not row or row[0] is None:
        cur.close()
        return []
    cur.execute(
        "select code from daily_bars_adjusted "
        f"where date = {ph} and trade_value is not null and trade_value >= {ph}",
        (row[0], float(tv_floor)),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes


def event_candidates(conn: Any, limit: int, *, tv_floor: float = 5_000.0,
                     movers_frac: float = 0.8, min_gain: float = 0.10) -> list[str]:
    """내일 구독할 '전일 이벤트 후보' 종목.

    ``--auto`` 의 근간이다. 두 갈래를 합친다:

    1. **전일 거래대금 상위** — :func:`scalp_it.intraday.store.top_universe` 와 같은
       ADV 기준(최근 20 거래일 평균)으로 유동성 있는 종목.
    2. **전일 급등주(상따/돌파 프록시)** — 가장 최근 거래일 **정규장** 종가가 그날
       기준가(전 거래일 정규장 종가) 대비 크게 오른 종목(:func:`_prev_day_movers`, 정문
       ``regular_session``). 상따·돌파 이벤트는 대개 전일 강세의 연장이라,
       스캐너 스크립트를 통째로 돌리지 않고도 후보를 싸게 잡는다.

    순위·거래대금은 ``daily_bars_adjusted`` 를 읽고, ``trade_value`` 하한(백만원 단위, 기본 50억)을
    둔다. 두 리스트를 순서대로 합쳐 중복을 제거하고 ``limit`` 개로 자른다.

    **급등주 위주(기본 80%)** 로 섞는다 — 만쥬 상따/돌파/짝꿍 대상은 대형 유동주가
    아니라 급등 소형주다(2026-08-24 발견: 거래대금 상위 절반 편성은 삼성전자·NAVER
    같은 대형주가 슬롯을 먹어 정작 상한가 갈 종목이 안 들어왔다 — `research/manju/34`).
    나머지(20%)만 유동성 상위를 넣어 짝꿍 대장주·시장 맥락을 확인한다.
    ``min_gain`` (기본 +10%) 이상 오른 전일 급등주만 movers 로 본다.
    """
    from scalp_it.intraday.store import top_universe

    limit = max(0, int(limit))
    if limit == 0:
        return []

    # 급등주 우선. 유동주는 (a) 최소 보장분(1-movers_frac) + (b) 급등주가 적은 날
    # 남는 슬롯을 채우는 예비 풀 — 항상 limit 개를 채워 구독 슬롯을 낭비하지 않는다.
    movers = _prev_day_movers(conn, limit, tv_floor=tv_floor, min_gain=min_gain)
    n_liquid_min = max(0, round(limit * (1.0 - movers_frac)))
    # 예비 풀도 거래대금 하한을 존중 — 저유동 종목엔 구독 슬롯을 쓰지 않는다.
    floor_ok = set(_liquid_above_floor(conn, tv_floor))
    liquid = [c for c in top_universe(conn, limit * 2) if c in floor_ok][:limit]
    # 급등주가 많으면 유동주는 최소 보장분만 앞에 끼우고, 나머지는 급등주로.
    movers = movers[: max(0, limit - n_liquid_min)] or movers

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


def _prev_day_movers(conn: Any, limit: int, *, tv_floor: float,
                     min_gain: float = 0.10) -> list[str]:
    """가장 최근 거래일 **정규장** 종가 상승률 상위. 거래대금 하한 + 최소 상승률 적용.

    상승률 = 그날 정규장 종가 / 그날 가격제한폭 기준가 − 1 — 둘 다 정문
    :mod:`scalp_it.regular_session` 으로 구한다. 예전에는 ``daily_bars_adjusted`` 종가 두 날의 비율이었는데,
    2026-09-14 이후 그 종가는 분자·분모 모두 애프터마켓(16:00~20:00) 종가라 급등주 판정이 틀렸다.
    정규장 종가를 모르는 종목(분봉 수집 밖 + 다음 거래일 수급표 전)은 빠진다. 거래대금 하한은 일봉
    값(애프터마켓 포함)을 그대로 쓴다 — 문턱 용도라 조금 부풀어도 판정이 뒤집히지 않는다.
    """
    from ..regular_session import base_prices, regular_closes

    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute("select max(date) from daily_bars_adjusted")
    row = cur.fetchone()
    if not row or row[0] is None:
        cur.close()
        return []
    last = row[0]
    cur.execute(
        "select code from daily_bars_adjusted "
        f"where date = {ph} and trade_value is not null and trade_value >= {ph}",
        (last, float(tv_floor)),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    if not codes:
        return []
    closes = regular_closes(conn, last, codes)
    bases = base_prices(conn, last, list(closes))
    gains = [(c, px / float(bases[c]) - 1.0) for c, px in closes.items() if bases.get(c, 0) > 0]
    picked = sorted((g for g in gains if g[1] >= float(min_gain)), key=lambda g: (-g[1], g[0]))
    return [c for c, _ in picked[: int(limit)]]
