"""분봉 저장소 — ``minute_bars`` 쓰기 SQL 은 이 파일에만 존재한다.

공시(``dart/store.py``)와 달리 분봉은 sqlite 가 아니라 **TimescaleDB** 에 쓴다.
분봉은 종목당 하루 380행이라 규모가 다르고, 후속 백테스트가 일봉과 조인해야
하므로 같은 DB 에 있어야 한다. 접속 대상은 ``KR_QUANT_DB`` 다.

스키마 문자열은 postgres 와 sqlite 양쪽에서 그대로 실행된다. 테스트는 sqlite
fixture 로 돌리고 운영은 postgres 로 돌리므로 둘이 갈라지면 테스트가 무의미해진다.
"""

from __future__ import annotations

import contextlib
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ..adapters.db import _placeholder, connect, is_postgres  # noqa: F401 — re-exported

#: 압축 대상: (code, ts) 로만 필터하는 조회가 대부분이라 segmentby='code',
#: 최신 우선 조회가 많아 orderby='ts DESC'. compress_after=14일은 실측
#: 근거가 있다 — upsert 는 그날 안에서만 일어나고(장중 재수집이 최신 값으로
#: 덮어씀), 다음 거래일부터는 그 청크에 쓰기가 없다. 14일이면 재수집 창을
#: 넉넉히 지나 압축해도 decompress→update→recompress churn 이 없다
#: (quant-airflow CLAUDE.md §3 이 경고하는 그 패턴).
#:
#: ⚠️ 압축된 청크는 형제 레포 quant-airflow의 scripts/backup_to_gdrive.sh 가
#: "core"(가벼운 일일 백업)에서 제외하는 목록을 빠져나간다 — 왜 그런지와 실제로
#: 무슨 일이 있었는지는 realtime/store.py 의 같은 주석에 있다. 여기 압축
#: 대상·주기를 바꾸면 그쪽 exclude 목록도 같이 볼 것.
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


#: SQLSTATE 21000 — ``ON CONFLICT DO UPDATE`` 가 한 문장 안에서 같은 행을 두 번 건드렸다.
_CARDINALITY_VIOLATION = "21000"

#: TimescaleDB 2.17 은 **압축 청크**와 충돌한 다중 행 ``ON CONFLICT DO NOTHING`` 에서 그 뒤 행을 오류 없이
#: 버린다(오늘 청크로 가는 행까지). 그래서 압축 청크에 닿을 수 있는 행은 행마다 보낸다. 경계는 배치마다
#: 조회한 압축 청크의 끝(:func:`_compressed_until`, 손으로 압축한 청크까지 잡는다)과 이 나이 중 늦은 쪽이다.
#: 이 나이는 조회와 insert 사이에 압축 정책(``compress_after`` 14일)이 돌아도 안전하게 하는 하한이다.
FAST_PATH_MAX_AGE = timedelta(days=7)

#: ``str(datetime)``·``isoformat()`` 이 내는 모양만 정규화한다. 소수 7자리 이상은 파이썬이 자르고 PG 는
#: 반올림해서, 정규화하면 PG 가 다르게 보는 두 행을 합쳐 행을 잃는다.
_NAIVE_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d{1,6})?")


def ts_key(value: Any) -> Any:
    """배치 안 중복을 거를 때 ``timestamp``(tz 없음) 열 값을 PG 가 보는 키에 맞춘다.

    스풀에서 되살린 행의 ts 는 ``str(datetime)`` 문자열이고 라이브 행은 naive ``datetime`` 이다.
    같은 시각이라도 파이썬 키로는 달라서 중복 제거를 빠져나간다. 그래서 그 모양의 문자열만
    naive ``datetime`` 으로 읽는다. 그 밖의 문자열(오프셋, 소수 7자리 이상 등)은 그대로 둔다 —
    덜 합친 것은 :func:`pg_upsert` 의 21000 폴백이 받고, 더 합치면 행을 잃는다.
    """
    if isinstance(value, str) and _NAIVE_TS_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value)
        except ValueError:  # 24:00:00·23:59:60 처럼 PG 만 받는 값 — 그대로 둬 원래 경로로 보낸다
            return value
    return value


def _naive_ts(value: Any) -> datetime | None:
    ts = ts_key(value)
    return ts if isinstance(ts, datetime) and ts.tzinfo is None else None


def _compressed_until(conn: Any, table: str) -> datetime | None:
    """``table`` 의 압축 청크가 덮는 마지막 시각(naive). 압축 청크가 없거나 TimescaleDB 가 아니면 ``None``."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT to_regclass('timescaledb_information.chunks') IS NOT NULL")
        if not cur.fetchone()[0]:
            return None
        # tz 없는 timestamp 차원의 경계는 UTC 로 해석된 timestamptz 로 나온다 → UTC 로 되돌리면 naive 경계.
        cur.execute(
            "SELECT max(range_end) AT TIME ZONE 'UTC' FROM timescaledb_information.chunks "
            "WHERE hypertable_name = %s AND hypertable_schema = ANY(current_schemas(false)) "
            "AND is_compressed",
            (table,),
        )
        return cur.fetchone()[0]
    finally:
        cur.close()


def pg_write(
    conn: Any,
    head: str,
    tail: str,
    payload: Sequence[Sequence[Any]],
    *,
    ts_index: int | None,
    table: str | None = None,
    key: Any = None,
    page_size: int = 2000,
    now: datetime | None = None,
) -> None:
    """``INSERT ... VALUES <행들> ON CONFLICT ...`` 를 PG 에 보낸다. 결과는 행마다 ``executemany`` 와 같다.

    - 최근 행(압축 청크 끝 + 1일, :data:`FAST_PATH_MAX_AGE` 둘 다보다 새 naive ts)은 ``execute_values``
      한 문장으로 보낸다. ``executemany`` 는 행마다 왕복한다.
    - 옛 행(압축 청크에 닿을 수 있다)은 원래대로 행마다 보낸다. 옛 행과 최근 행은 naive ts 가 달라
      충돌 키가 겹치지 않으므로 나눠 보내도 결과가 같다.
    - naive ``datetime`` 으로 읽을 수 없는 ts(오프셋 문자열 등)가 하나라도 있으면 PG 에서 어느 행과
      같은 키인지 알 수 없어 순서를 지켜야 하므로, 배치 전체를 원래 경로로 보낸다.
      ``ts_index=None`` 은 압축하지 않는 일반 테이블(테마 스냅샷) — 전부 한 문장으로 보낸다.
    - ``key`` 가 있으면(``DO UPDATE``) 한 문장에 같은 키가 두 번 나오면 안 되므로(21000) 같은 키는
      **마지막 행만, 마지막 자리에** 남긴다. 파이썬 키로는 달랐는데 PG 에선 같은 키라 21000 이 나면
      롤백하고 그 배치를 원래 경로로 다시 넣는다.

    커밋은 호출자가 한다.
    """
    from psycopg2.extras import execute_values

    if ts_index is None:
        recent, old = list(payload), []
    else:
        stamps = [_naive_ts(r[ts_index]) for r in payload]
        if any(t is None for t in stamps):
            recent, old = [], list(payload)
        else:
            cutoff = (now or datetime.now()) - FAST_PATH_MAX_AGE  # noqa: DTZ005 — KST naive, 프로젝트 표준
            until = _compressed_until(conn, table) if table else None
            if until is not None:
                cutoff = max(cutoff, until + timedelta(days=1))
            is_recent = [t >= cutoff for t in stamps]  # NaT 등 비교가 안 되는 값은 False → 옛 행 쪽
            recent = [r for r, ok in zip(payload, is_recent, strict=True) if ok]
            old = [r for r, ok in zip(payload, is_recent, strict=True) if not ok]
    row_sql = head + "(" + ",".join(["%s"] * len(payload[0])) + ")" + tail if payload else ""
    if key is not None:
        last: dict[Any, Sequence[Any]] = {}
        for r in recent:
            k = key(r)
            last.pop(k, None)
            last[k] = r
        recent = list(last.values())
    cur = conn.cursor()
    try:
        if recent:
            execute_values(cur, head + "%s" + tail, recent, page_size=page_size)
        if old:
            cur.executemany(row_sql, old)
    except Exception as exc:
        if key is None or getattr(exc, "pgcode", None) != _CARDINALITY_VIOLATION:
            raise
        cur.close()
        conn.rollback()
        cur = conn.cursor()
        cur.executemany(row_sql, payload)
    cur.close()


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
    dates = _recent_trading_dates(conn, lookback_days)
    if not dates:
        return []
    cutoff = min(dates)
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        "select code, avg(trade_value) as adv from daily_bars_adjusted "
        f"where date >= {ph} and trade_value is not null "
        f"group by code order by adv desc limit {ph}",
        (cutoff, int(limit)),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes


def _recent_trading_dates(conn: Any, days: int) -> list[Any]:
    """``daily_bars_adjusted`` 기준 최근 ``days`` 거래일(최신 우선)."""
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        "select distinct date from daily_bars_adjusted order by date desc limit " + ph,
        (int(days),),
    )
    dates = [r[0] for r in cur.fetchall()]
    cur.close()
    return dates


def recently_collected(conn: Any, *, keep_days: int = 20) -> list[str]:
    """최근 ``keep_days`` 거래일 안에 ``minute_bars`` 행이 하나라도 있는 종목.

    거래일 경계는 :func:`top_universe` 와 같이 ``daily_bars_adjusted`` 의 날짜로
    정한다. ``ts`` 하한만 거는 조회라 하이퍼테이블 청크 제외가 그대로 먹는다.
    """
    dates = _recent_trading_dates(conn, keep_days)
    if not dates:
        return []
    cutoff = min(dates)
    ph = _placeholder(conn)
    cur = conn.cursor()
    cur.execute(
        f"select distinct code from minute_bars where ts >= {ph} order by code",
        (cutoff,),
    )
    codes = [str(r[0]).strip() for r in cur.fetchall()]
    cur.close()
    return codes


def ticked_codes(conn: Any, day: date) -> list[str]:
    """``day`` 에 실시간 ``ticks`` 가 한 건이라도 있는 종목(감지기·수집기가 그날 구독한 종목).

    표가 없거나 읽기 실패면 빈 리스트 — 분봉 수집을 막지 않는다. ``ts`` 범위만 거는 조회라
    ``idx_ticks_ts`` 를 탄다(실측 하루 ~1초).
    """
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(f"select distinct code from ticks where ts >= {ph} and ts < {ph}",
                    (f"{day.isoformat()} 00:00:00", f"{(day + timedelta(days=1)).isoformat()} 00:00:00"))
        codes = sorted(str(r[0]).strip() for r in cur.fetchall() if r[0])
        cur.close()
        return codes
    except Exception as exc:  # noqa: BLE001 — 틱 표 문제로 분봉 수집을 멈추지 않는다
        print(f"[분봉] 그날 구독 종목 조회 실패(건너뜀): {type(exc).__name__}", file=sys.stderr)
        with contextlib.suppress(Exception):
            conn.rollback()
        return []


def collection_universe(
    conn: Any, limit: int, *, lookback_days: int = 20, keep_days: int = 20,
    tick_day: date | None = None,
) -> list[str]:
    """분봉 수집 대상 = ADV 상위 ``limit`` **∪** 그날 구독 종목 **∪** 최근 ``keep_days`` 거래일 수집 종목.

    ADV 상위만 쓰면 순위 밖으로 밀린 종목은 그날로 갱신이 영영 끊긴다 — 실제로
    상위 1400 하드캡(2026-09-22~) 아래에서 333종목이 09-21 에, 217종목이 08-21 에
    멈춘 채 남았다(2026-10-12 데이터 감사). 최근에 받던 종목은 순위에서 빠져도
    ``keep_days`` 동안은 계속 받고, 그 뒤에도 순위 밖이면 그때 떨어진다.

    ``tick_day`` 를 주면 그날 ``ticks`` 에 있는 종목(:func:`ticked_codes`)을 더한다. 신규상장주는
    상장 당일 일봉이 없어 ADV 순위에 못 들고, 그날 분봉이 다음 날 15:45 에야 들어와 상장 다음 날
    08:55 감지기가 기준가(전일 정규장 종가)를 몰랐다(486510 09-29, 10-12 리뷰). 감지기가 그날
    구독한 종목(대장 스캔·신규상장·돌발 재료)은 다음 날 기준가가 필요한 종목이라 같은 날 받는다.
    실측 하루 10~17종목 추가(09-29·10-07·10-12).

    순서는 ADV 상위, 그날 구독분, 유지분(코드순). 그래서 대상 수는 ``limit`` 보다 클 수 있다.

    ⚠️ 이건 **멈춤**을 막을 뿐 **구멍**은 못 막는다. 크론은 ``--pages 6``
    (≈13 거래일)이라, ``keep_days`` 밖으로 떨어졌다가 한참 뒤 순위에 돌아온
    종목은 최신 13거래일만 받고 그 사이가 빈다. 메우려면 그 종목만 골라
    ``--codes ... --pages 40`` 으로 따로 돌려야 한다.
    """
    top = top_universe(conn, limit, lookback_days=lookback_days)
    seen = set(top)
    ticked: list[str] = []
    if tick_day is not None:
        for c in ticked_codes(conn, tick_day):
            if c not in seen:
                seen.add(c)
                ticked.append(c)
    kept = [c for c in recently_collected(conn, keep_days=keep_days) if c not in seen]
    return top + ticked + kept
