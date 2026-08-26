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


# ----- 만쥬 §6 장중 돌발 재료 — 구독 종목 추가 --------------------------------

def test_build_reg_message_supports_incremental_add():
    """``refresh="0"`` = 기존 등록에 **추가**(교체 아님).

    만쥬 §6 은 "예상치 못한 돌발 뉴스가 나오면 즉각 관련 종목 매수"다. 장전에 고른
    유니버스에 그 종목이 없으면 구독부터 없어서 감시조차 못 한다. 그래서 장중 추가가
    필요하고, refresh 를 1(교체)로 보내면 **기존 종목이 통째로 날아간다**.
    """
    from scalp_it.realtime.collector import build_reg_message

    msg = build_reg_message(["000001"], ["0B", "0D"], "1", refresh="0")
    assert msg["refresh"] == "0"
    assert msg["data"][0]["item"] == ["000001"]


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
