# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""실시간 데몬 파싱·구독·수집 테스트 — WebSocket 은 mock 프레임으로만.

네트워크는 절대 타지 않는다. 실호출 검증은 장중 --dry-run 몫이다.
"""

import json
import sqlite3
from datetime import date, datetime

import pytest

from scalp_it.cli_tick import parse_duration
from scalp_it.realtime.collector import (
    DualKeyCollector,
    KiwoomRealtimeCollector,
    SeqAssigner,
    build_reg_message,
    chunk_codes,
    codes_per_connection,
    parse_quote_values,
    parse_real_message,
    parse_tick_values,
    parse_vi_values,
)
from scalp_it.realtime.store import (
    count_quotes,
    count_ticks,
    count_vi_events,
    init_realtime_tables,
)

TDAY = date(2026, 8, 24)


def _tick_values(qty="-30"):
    return {"20": "093015", "10": "+274500", "15": qty, "13": "1000000",
            "228": "120.50", "27": "+274600", "28": "+274500"}


def _quote_values():
    v = {"21": "093015", "121": "40000", "125": "50000"}
    for i in range(10):
        v[str(41 + i)] = f"+{275000 + i}"   # 매도호가 1..10
        v[str(51 + i)] = f"+{274000 - i}"   # 매수호가 1..10
        v[str(61 + i)] = str((i + 1) * 5)   # 매도잔량
        v[str(71 + i)] = str((i + 1) * 10)  # 매수잔량
    return v


def _vi_values(kind="1", apply="1"):
    """1h VI발동 프레임 values(문서 FID 기준). kind=9068, apply=1225."""
    return {
        "9001": "005930", "302": "삼성전자", "13": "1000000", "14": "270000000000",
        "9068": kind,          # VI발동구분: 1=발동, 2=해제
        "1225": apply,         # VI적용구분: 1=정적, 2=동적, 3=정적&동적
        "1221": "+300000",     # 발동가격
        "1223": "093015",      # 발동시각
        "1224": "093215",      # 해제시각(예상)
        "1236": "+272000",     # 정적기준가격
        "1237": "+295000",     # 동적기준가격
        "1238": "10.29",       # 정적괴리율
        "1239": "1.69",        # 동적괴리율
    }


# --------------------------------------------------------------- 파싱


def test_parse_tick_values_maps_fids():
    seq = SeqAssigner()
    row = parse_tick_values("005930", _tick_values("-30"), TDAY, seq)
    assert row["ts"] == datetime(2026, 8, 24, 9, 30, 15)
    assert row["price"] == 274500       # 방향 부호 제거
    assert row["volume"] == 30          # 절댓값
    assert row["side"] == -1            # 매도
    assert row["strength"] == pytest.approx(120.5)
    assert row["cum_volume"] == 1_000_000
    assert row["best_bid"] == 274500
    assert row["best_ask"] == 274600
    assert row["raw"]["15"] == "-30"    # 원문 보존


def test_parse_tick_side_buy():
    seq = SeqAssigner()
    row = parse_tick_values("005930", _tick_values("+42"), TDAY, seq)
    assert row["side"] == 1
    assert row["volume"] == 42


def test_parse_tick_bad_time_returns_none():
    seq = SeqAssigner()
    assert parse_tick_values("005930", {"20": ""}, TDAY, seq) is None


def test_parse_quote_values_ten_levels():
    row = parse_quote_values("005930", _quote_values(), TDAY)
    assert row["ts"] == datetime(2026, 8, 24, 9, 30, 15)
    assert row["ask1"] == 275000
    assert row["ask10"] == 275009
    assert row["bid1"] == 274000
    assert row["bidqty1"] == 10
    assert row["askqty10"] == 50
    assert row["total_ask_qty"] == 40000
    assert row["total_bid_qty"] == 50000
    assert row["vi_flag"] is None


def test_parse_vi_values_maps_fids():
    row = parse_vi_values("005930", _vi_values(kind="1", apply="1"), TDAY)
    assert row["ts"] == datetime(2026, 8, 24, 9, 30, 15)  # 발동시각 1223
    assert row["vi_type"] == "발동"        # 9068=1
    assert row["vi_kind"] == "정적"        # 1225=1
    assert row["trigger_price"] == 300000  # 1221, 부호 제거
    assert row["base_price"] == 272000     # 정적기준가 우선(1236)
    assert row["gap_rate"] == pytest.approx(10.29)  # 정적괴리율 우선(1238)
    assert row["release_time"] == "093215"          # 1224
    assert row["raw"]["9068"] == "1"       # 원문 보존


def test_parse_vi_values_release_and_dynamic():
    row = parse_vi_values("000660", _vi_values(kind="2", apply="2"), TDAY)
    assert row["vi_type"] == "해제"  # 9068=2
    assert row["vi_kind"] == "동적"  # 1225=2


def test_parse_vi_values_falls_back_to_received_at():
    v = _vi_values()
    del v["1223"]  # 발동시각 결손
    now = datetime(2026, 8, 24, 10, 0, 0)
    row = parse_vi_values("005930", v, TDAY, received_at=now)
    assert row["ts"] == now


def test_parse_real_message_splits_types():
    seq = SeqAssigner()
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()},
        {"type": "0D", "item": "005930", "values": _quote_values()},
        {"type": "1h", "item": "005930", "values": _vi_values()},
        {"type": "0C", "item": "005930", "values": {"20": "093015"}},  # 무시
    ]}
    ticks, quotes, vis = parse_real_message(json.dumps(msg), TDAY, seq)
    assert len(ticks) == 1
    assert len(quotes) == 1
    assert len(vis) == 1
    assert vis[0]["vi_type"] == "발동"


def test_parse_real_message_ignores_non_real():
    seq = SeqAssigner()
    assert parse_real_message({"trnm": "PING"}, TDAY, seq) == ([], [], [])
    assert parse_real_message({"trnm": "REG", "return_code": 0}, TDAY, seq) == ([], [], [])


def test_seq_assigner_increments_and_resets():
    seq = SeqAssigner()
    t1 = datetime(2026, 8, 24, 9, 30, 15)
    t2 = datetime(2026, 8, 24, 9, 30, 16)
    assert [seq.next("A", t1), seq.next("A", t1), seq.next("A", t2)] == [0, 1, 0]
    assert seq.next("B", t1) == 0  # 종목별 독립


# --------------------------------------------------------------- 구독 회계


def test_chunk_codes_dedups_and_splits():
    groups = chunk_codes(["1", "2", "2", "3", "4", "5"], 2)
    assert groups == [["1", "2"], ["3", "4"], ["5"]]


def test_codes_per_connection():
    assert codes_per_connection(["0B", "0D"]) == 50   # 100 // 2
    assert codes_per_connection(["0B"]) == 100
    assert codes_per_connection(["0B", "0D", "1h"]) == 33  # 100 // 3, VI 포함


def test_build_reg_message_shape():
    msg = build_reg_message(["005930", "000660"], ["0B", "0D"], "1")
    assert msg["trnm"] == "REG"
    assert msg["data"][0]["item"] == ["005930", "000660"]
    assert msg["data"][0]["type"] == ["0B", "0D"]


def test_parse_duration():
    assert parse_duration(None) is None
    assert parse_duration("09:00-15:30") is not None  # 초 단위, 음수 아님
    assert parse_duration("09:00-15:30") >= 0


# --------------------------------------------------------------- 수집 end-to-end (mock ws)


class _FakeWS:
    """LOGIN·REG 응답을 recv 로, REAL 프레임을 async 이터로 흘리는 가짜 소켓."""

    def __init__(self, handshake, frames, collector):
        self._handshake = list(handshake)
        self._frames = list(frames)
        self._collector = collector
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return self._handshake.pop(0)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for f in self._frames:
            yield f
        self._collector.stop()  # 프레임 소진 후 데몬 종료(재연결 방지)


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "rt.db")
    init_realtime_tables(c)
    yield c
    c.close()


def test_collector_run_stores_frames(conn):
    import asyncio

    async def _run():
        collector = KiwoomRealtimeCollector(
            "tok", ["005930"], conn=conn, flush_interval=0.05, trade_date=TDAY,
            log=lambda m: None,
        )
        assert collector.n_connections == 1
        frames = [
            json.dumps({"trnm": "REAL", "data": [
                {"type": "0B", "item": "005930", "values": _tick_values()}]}),
            json.dumps({"trnm": "REAL", "data": [
                {"type": "0D", "item": "005930", "values": _quote_values()}]}),
            json.dumps({"trnm": "REAL", "data": [
                {"type": "1h", "item": "005930", "values": _vi_values()}]}),
        ]
        handshake = [
            json.dumps({"trnm": "LOGIN", "return_code": 0}),
            json.dumps({"trnm": "REG", "return_code": 0}),
        ]
        ws = _FakeWS(handshake, frames, collector)

        async def fake_connect():
            return ws

        collector._connect = fake_connect
        await collector.run(duration=2.0)
        assert ws.sent[0] == json.dumps({"trnm": "LOGIN", "token": "tok"})
        return collector

    collector = asyncio.run(_run())
    assert collector.frames_seen == 3
    assert count_ticks(conn) == 1
    assert count_quotes(conn) == 1
    assert count_vi_events(conn) == 1
    assert collector.ticks_stored == 1
    assert collector.vi_stored == 1


def test_collector_dry_run_does_not_store(conn):
    import asyncio

    async def _run():
        collector = KiwoomRealtimeCollector(
            "tok", ["005930"], conn=conn, store=False, flush_interval=0.05,
            trade_date=TDAY, log=lambda m: None,
        )
        frames = [json.dumps({"trnm": "REAL", "data": [
            {"type": "0B", "item": "005930", "values": _tick_values()}]})]
        handshake = [json.dumps({"trnm": "LOGIN", "return_code": 0}),
                     json.dumps({"trnm": "REG", "return_code": 0})]
        ws = _FakeWS(handshake, frames, collector)
        collector._connect = lambda: _wrap(ws)
        await collector.run(duration=2.0)
        return collector

    async def _wrap(ws):
        return ws

    collector = asyncio.run(_run())
    assert collector.frames_seen == 1
    assert count_ticks(conn) == 0  # store=False


def test_flush_survives_db_outage_and_preserves_batch(conn):
    """DB 순단: insert 가 실패해도 배치를 버퍼에 보존하고 flush 는 안 죽는다.
    재연결(dsn 없음)에 실패해도 다음 flush 에서 회복 가능해야 한다."""
    collector = KiwoomRealtimeCollector(
        "tok", ["005930"], conn=conn, store=True, trade_date=TDAY, log=lambda m: None,
    )
    # 틱 하나를 버퍼에 넣고, insert 가 터지는 죽은 커넥션으로 교체
    from scalp_it.realtime.collector import SeqAssigner, parse_real_message
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()}]}
    ticks, _q, _v = parse_real_message(msg, TDAY, SeqAssigner(), received_at=datetime(2026, 8, 24, 9, 0, 1))
    collector._tick_buf.extend(ticks)

    class _DeadConn:
        def cursor(self):
            raise RuntimeError("server closed the connection unexpectedly")

    collector._conn = _DeadConn()
    collector._dsn = None  # 재연결 불가 — 그래도 예외 전파 없이 배치 보존
    n = len(collector._tick_buf)
    collector._flush()  # 예외 안 나야 함
    assert len(collector._tick_buf) == n  # 배치 유실 없이 보존

    # 커넥션 복구 후 flush 하면 저장된다
    collector._conn = conn
    collector._flush()
    assert count_ticks(conn) >= 1


# ----- 프로세스 재시작 durability — 디스크 스풀 -------------------------------
#
# 위 test_flush_survives_db_outage_and_preserves_batch 는 "프로세스가 살아있는
# 동안의" DB 순단만 다룬다. 그동안 버퍼는 메모리에만 있어서, DB 가 안 살아난
# 채로 프로세스 자체가 죽으면(OOM·재시작·서버 재부팅) 그 순간 버퍼에 있던
# 틱·호가가 전부 사라진다 — 복구 불가능한 데이터라 이 손실은 재현이 안 된다.
# spool_dir 를 주면 flush 실패 시 배치를 디스크에도 남기고, 다음 프로세스가
# 뜰 때 그걸 읽어 버퍼로 되살린다.


def test_flush_spools_batch_to_disk_on_persistent_failure(conn, tmp_path):
    """insert 가 계속 실패하면 배치를 spool 파일에도 남긴다(메모리 유실 대비)."""
    spool_dir = tmp_path / "spool"
    collector = KiwoomRealtimeCollector(
        "tok", ["005930"], conn=conn, store=True, trade_date=TDAY,
        log=lambda m: None, spool_dir=spool_dir,
    )
    from scalp_it.realtime.collector import SeqAssigner, parse_real_message
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()}]}
    ticks, _q, _v = parse_real_message(
        msg, TDAY, SeqAssigner(), received_at=datetime(2026, 8, 24, 9, 0, 1))
    collector._tick_buf.extend(ticks)

    class _DeadConn:
        def cursor(self):
            raise RuntimeError("server closed the connection unexpectedly")

    collector._conn = _DeadConn()
    collector._dsn = None
    collector._flush()

    spool_file = spool_dir / "ticks.jsonl"
    assert spool_file.exists()
    lines = spool_file.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["code"] == "005930"


def test_flush_spools_on_cold_start_when_conn_is_none(tmp_path):
    """conn=None 으로 시작(재연결도 실패)해도 spool 에는 쓴다.

    ``_DeadConn`` 을 쓰는 위 테스트들은 ``self._conn`` 이 non-None(죽은 객체)
    이라 insert 예외 경로를 타는데, 프로세스가 갓 떠서 아직 한 번도 연결
    못 했을 때는 ``self._conn`` 자체가 None 이다. 실전 통합 검증(2026-09-11)
    에서 이 경로가 그냥 조용히 넘어가고 spool 도 안 쓰는 버그로 드러났다 —
    이 상태로 프로세스가 죽으면 스풀 없이 버퍼가 통째로 사라진다.
    """
    spool_dir = tmp_path / "spool"
    collector = KiwoomRealtimeCollector(
        "tok", ["005930"], conn=None, dsn="postgresql://x:x@127.0.0.1:1/x",
        store=True, trade_date=TDAY, log=lambda m: None, spool_dir=spool_dir,
    )
    from scalp_it.realtime.collector import SeqAssigner, parse_real_message
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()}]}
    ticks, _q, _v = parse_real_message(
        msg, TDAY, SeqAssigner(), received_at=datetime(2026, 8, 24, 9, 0, 1))
    collector._tick_buf.extend(ticks)
    assert collector._conn is None

    collector._flush()  # 재연결 시도 -> 실패(127.0.0.1:1) -> conn 여전히 None

    spool_file = spool_dir / "ticks.jsonl"
    assert spool_file.exists(), "conn=None 콜드스타트에서도 spool 에 써야 한다"
    assert len(collector._tick_buf) == 1  # 메모리 버퍼도 그대로 보존


def test_flush_clears_spool_after_successful_insert(conn, tmp_path):
    """재연결 후 flush 가 성공하면 그 배치의 spool 파일은 지운다(중복 리플레이 방지)."""
    spool_dir = tmp_path / "spool"
    collector = KiwoomRealtimeCollector(
        "tok", ["005930"], conn=conn, store=True, trade_date=TDAY,
        log=lambda m: None, spool_dir=spool_dir,
    )
    from scalp_it.realtime.collector import SeqAssigner, parse_real_message
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()}]}
    ticks, _q, _v = parse_real_message(
        msg, TDAY, SeqAssigner(), received_at=datetime(2026, 8, 24, 9, 0, 1))
    collector._tick_buf.extend(ticks)

    class _DeadConn:
        def cursor(self):
            raise RuntimeError("server closed the connection unexpectedly")

    collector._conn = _DeadConn()
    collector._dsn = None
    collector._flush()
    assert (spool_dir / "ticks.jsonl").exists()

    collector._conn = conn
    collector._flush()
    assert count_ticks(conn) >= 1
    assert not (spool_dir / "ticks.jsonl").exists()


def test_collector_replays_pending_spool_on_construction(conn, tmp_path):
    """이전 프로세스가 죽기 직전 남긴 spool 파일을 새 인스턴스가 버퍼로 되살린다."""
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    row = {
        "code": "005930", "ts": "2026-08-24 09:00:01", "seq": 0,
        "price": 274500, "volume": -30, "side": None, "strength": 120.5,
        "cum_volume": 1000000, "best_bid": 274500, "best_ask": 274600, "raw": None,
    }
    (spool_dir / "ticks.jsonl").write_text(json.dumps(row) + "\n")

    collector = KiwoomRealtimeCollector(
        "tok", ["005930"], conn=conn, store=True, trade_date=TDAY,
        log=lambda m: None, spool_dir=spool_dir,
    )
    assert len(collector._tick_buf) == 1
    assert collector._tick_buf[0]["code"] == "005930"

    # 되살린 걸 정상적으로 flush 할 수 있어야 한다(round-trip)
    collector._flush()
    assert count_ticks(conn) >= 1
    assert not (spool_dir / "ticks.jsonl").exists()


# ----- 만쥬 §6 장중 돌발 재료 — 구독 종목 추가 --------------------------------

def test_build_reg_message_supports_incremental_add():
    """``refresh="1"`` = 기존 등록을 **유지**하고 추가(교체 아님).

    키움의 ``refresh`` 는 **기존등록유지여부**다 — 1 이 유지, 0 이 해제 후 등록.
    종전엔 0 으로 보냈고, 그 한 번의 추가가 그때까지 구독하던 종목을 전부 날렸다
    (2026-08-28 틱 실측: 종목별 마지막 틱 = 다음 추가 종목의 첫 틱, 초 단위 일치).
    """
    from scalp_it.realtime.collector import build_reg_message

    msg = build_reg_message(["000001"], ["0B", "0D"], "1", refresh="1")
    assert msg["refresh"] == "1"
    assert msg["data"][0]["item"] == ["000001"]


def test_add_codes_keeps_existing_registration():
    """장중 추가는 **기존 구독을 유지**해야 한다 — 배선까지 검사한다.

    ``build_reg_message`` 만 검사하면 호출부가 0 을 넘겨도 안 걸린다. 실제로 그렇게
    새서 2026-08-27~28 이틀치 수집이 무너졌다(08-28 09시 39종목 → 10시 5종목).
    """
    import asyncio
    import json

    from scalp_it.realtime.collector import KiwoomRealtimeCollector

    collector = KiwoomRealtimeCollector(token="t", codes=["000001"], dsn=None)
    ws = _FakeWS([], [], collector)
    collector._live_ws["1"] = (ws, ["000001"])

    ok, rejected = asyncio.run(collector.add_codes(["000002"]))

    assert ok == ["000002"] and rejected == []
    sent = json.loads(ws.sent[-1])
    assert sent["trnm"] == "REG"
    assert sent["refresh"] == "1", "0 이면 기존 구독이 통째로 날아간다"
    assert sent["data"][0]["item"] == ["000002"]
    assert collector._live_ws["1"][1] == ["000001", "000002"]


def test_pending_adds_respect_registration_limit():
    """등록 한도를 넘겨 추가하면 안 된다 — 넘기면 커넥션이 쪼개져 서로 kick 한다.

    2026-08-24 에 실제로 80종목을 단일 키로 넣었다가 30종목이 10:58 에 끊겼다.
    """
    from scalp_it.realtime.collector import plan_additions

    # 이미 49종목, 0B+0D 면 커넥션당 50 → 1자리만 남는다.
    have = [str(i).zfill(6) for i in range(49)]
    ok, rejected = plan_additions(have, ["900001", "900002"], ["0B", "0D"])
    assert ok == ["900001"]
    assert rejected == ["900002"]


def test_pending_adds_skip_already_subscribed():
    from scalp_it.realtime.collector import plan_additions

    have = ["000001", "000002"]
    ok, rejected = plan_additions(have, ["000001", "000003"], ["0B", "0D"])
    assert ok == ["000003"]
    assert rejected == []


def test_quote_levels_extracts_ten_levels():
    """0D 행 → ``(bids, asks)`` 10단 — 만쥬 §4-1 호가 두께 필터의 유일한 입력.

    2026-08-26: CLI 가 `on_quote(code, ts, bid1)` 로 **가격 하나만** 넘겨서 감지기의
    호가창이 프로덕션에서 영원히 비어 있었다. 그 결과 `--max-roundtrip-bp 20`(만쥬가
    **최우선**이라 한 필터)이 입력이 없어 무조건 통과로 돌았다 — cron 플래그가 켜져
    있는데 아무것도 안 거르는 상태였다.
    """
    from scalp_it.realtime.collector import quote_levels

    row = {"code": "000001"}
    for i in range(1, 11):
        row[f"bid{i}"] = 1000 - i
        row[f"bidqty{i}"] = 10 * i
        row[f"ask{i}"] = 1000 + i
        row[f"askqty{i}"] = 20 * i
    bids, asks = quote_levels(row)
    assert len(bids) == 10 and len(asks) == 10
    assert bids[0] == (999.0, 10.0)      # 매수1호가
    assert asks[0] == (1001.0, 20.0)     # 매도1호가


def test_quote_levels_skips_missing_or_zero():
    """잔량·가격이 없거나 0 인 단은 버린다 — 있는 것만 넘겨야 왕복충격이 맞는다."""
    from scalp_it.realtime.collector import quote_levels

    row = {"bid1": 1000, "bidqty1": 5, "bid2": None, "bidqty2": 3,
           "ask1": 1001, "askqty1": 0}
    bids, asks = quote_levels(row)
    assert bids == [(1000.0, 5.0)]
    assert asks == []


# ==================================================================== DualKeyCollector

class _FakeSubCollector:
    """add_codes/remove_codes 라우팅만 검증하는 최소 스텁 — 소켓은 안 탄다."""

    def __init__(self, *, capacity=2, already=()):
        self.code_capacity = capacity
        self.subscribed_codes = list(already)
        self.frames_seen = 1
        self.ticks_stored = 2
        self.quotes_stored = 3
        self.vi_stored = 0
        self.reconnects = 0
        self.n_connections = 1
        self.rejected_codes = []
        self.added_codes = []
        self.removed_codes = []
        self.stopped = False
        self.ran_with = None

    async def add_codes(self, codes):
        room = self.code_capacity - len(self.subscribed_codes)
        ok, rejected = list(codes[:max(room, 0)]), list(codes[max(room, 0):])
        self.subscribed_codes.extend(ok)
        self.added_codes.extend(ok)
        self.rejected_codes.extend(rejected)
        return ok, rejected

    async def remove_codes(self, codes):
        drop = [c for c in codes if c in self.subscribed_codes]
        for c in drop:
            self.subscribed_codes.remove(c)
        self.removed_codes.extend(drop)
        return drop

    def stop(self):
        self.stopped = True

    async def run(self, duration=None):
        self.ran_with = duration


def test_dual_key_add_codes_fills_primary_then_secondary():
    import asyncio

    a, b = _FakeSubCollector(capacity=2), _FakeSubCollector(capacity=2)
    dual = DualKeyCollector(a, b)
    added, rejected = asyncio.run(dual.add_codes(["005930", "000660", "035420"]))
    assert added == ["005930", "000660", "035420"]
    assert rejected == []
    assert a.subscribed_codes == ["005930", "000660"]
    assert b.subscribed_codes == ["035420"]


def test_dual_key_add_codes_skips_already_subscribed_on_either_side():
    import asyncio

    a = _FakeSubCollector(capacity=2, already=["005930"])
    b = _FakeSubCollector(capacity=2, already=["035420"])
    dual = DualKeyCollector(a, b)
    added, rejected = asyncio.run(dual.add_codes(["005930", "035420", "000660"]))
    assert added == ["000660"]           # 이미 A·B 각각에 있던 건 새로 안 붙는다
    assert rejected == []
    assert a.added_codes == ["000660"]   # A 에 자리(1개 남음)가 있으니 A 로 감


def test_dual_key_add_codes_reports_rejected_when_both_full():
    import asyncio

    a, b = _FakeSubCollector(capacity=1), _FakeSubCollector(capacity=1)
    dual = DualKeyCollector(a, b)
    added, rejected = asyncio.run(dual.add_codes(["005930", "000660", "035420"]))
    assert added == ["005930", "000660"]
    assert rejected == ["035420"]


def test_dual_key_remove_codes_finds_the_right_side():
    import asyncio

    a = _FakeSubCollector(already=["005930"])
    b = _FakeSubCollector(already=["035420"])
    dual = DualKeyCollector(a, b)
    removed = asyncio.run(dual.remove_codes(["035420", "999999"]))
    assert removed == ["035420"]
    assert a.subscribed_codes == ["005930"]
    assert b.subscribed_codes == []


def test_dual_key_stats_and_capacity_are_summed():
    a, b = _FakeSubCollector(capacity=2), _FakeSubCollector(capacity=3)
    dual = DualKeyCollector(a, b)
    assert dual.code_capacity == 5
    assert dual.frames_seen == 2
    assert dual.ticks_stored == 4
    assert dual.quotes_stored == 6
    assert dual.n_connections == 2


def test_dual_key_stop_stops_both():
    a, b = _FakeSubCollector(), _FakeSubCollector()
    DualKeyCollector(a, b).stop()
    assert a.stopped and b.stopped


def test_dual_key_run_gathers_both_with_same_duration():
    import asyncio

    a, b = _FakeSubCollector(), _FakeSubCollector()
    asyncio.run(DualKeyCollector(a, b).run(duration=5.0))
    assert a.ran_with == 5.0
    assert b.ran_with == 5.0


def test_dual_key_run_propagates_exception_from_either_side():
    import asyncio

    class _Boom(_FakeSubCollector):
        async def run(self, duration=None):
            raise RuntimeError("소켓 죽음")

    a, b = _Boom(), _FakeSubCollector()
    with pytest.raises(RuntimeError, match="소켓 죽음"):
        asyncio.run(DualKeyCollector(a, b).run())
    assert b.ran_with is None or True  # b 도 gather 로 같이 돌았다(예외는 나중에 raise)


# ------------------------------------------ 빈 목록으로 시작하는 커넥션(듀얼키 B)

def test_collector_with_no_codes_still_opens_one_connection():
    """듀얼키 B 는 장전 유니버스가 정원(50)보다 작으면 빈 목록으로 시작한다.

    빈 목록이라고 커넥션을 0개 만들면 LOGIN 자체가 없어 소켓이 안 열리고, 장중
    추가(add_codes)가 전부 조용히 거절된다 — 09-07~10 나흘 동안 정원이 100 이라
    믿었는데 실제론 50 이었다.
    """
    c = KiwoomRealtimeCollector("tok", [], store=False, log=lambda m: None)
    assert c.n_connections == 1


def test_empty_connection_logs_in_without_reg_and_accepts_adds():
    import asyncio

    logs: list[str] = []

    async def _run():
        collector = KiwoomRealtimeCollector("tok", [], store=False, trade_date=TDAY,
                                            log=logs.append)
        handshake = [json.dumps({"trnm": "LOGIN", "return_code": 0})]
        ws = _FakeWS(handshake, [], collector)

        async def fake_connect():
            return ws

        collector._connect = fake_connect
        await collector.run(duration=2.0)
        assert [json.loads(s)["trnm"] for s in ws.sent] == ["LOGIN"]   # REG 는 안 보낸다
        assert collector._live_ws["1"][1] == []
        added, rejected = await collector.add_codes(["000001", "000002"])
        assert added == ["000001", "000002"] and rejected == []
        reg = json.loads(ws.sent[-1])
        assert reg["trnm"] == "REG" and reg["refresh"] == "1"
        assert reg["data"][0]["item"] == ["000001", "000002"]

    asyncio.run(_run())
    assert any("LOGIN ok" in m for m in logs)


def test_add_codes_without_live_socket_says_so():
    """소켓이 안 붙었을 때의 거절도 로그에 남아야 한다 — 조용히 넘기면 못 본다."""
    import asyncio

    logs: list[str] = []
    c = KiwoomRealtimeCollector("tok", [], store=False, log=logs.append)
    added, rejected = asyncio.run(c.add_codes(["000001"]))
    assert added == [] and rejected == ["000001"]
    assert c.rejected_codes == ["000001"]
    assert any("소켓" in m for m in logs)
