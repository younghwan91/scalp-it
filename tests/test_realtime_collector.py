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
)
from scalp_it.realtime.store import count_quotes, count_ticks, init_realtime_tables

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


def test_parse_real_message_splits_types():
    seq = SeqAssigner()
    msg = {"trnm": "REAL", "data": [
        {"type": "0B", "item": "005930", "values": _tick_values()},
        {"type": "0D", "item": "005930", "values": _quote_values()},
        {"type": "0C", "item": "005930", "values": {"20": "093015"}},  # 무시
    ]}
    ticks, quotes = parse_real_message(json.dumps(msg), TDAY, seq)
    assert len(ticks) == 1
    assert len(quotes) == 1


def test_parse_real_message_ignores_non_real():
    seq = SeqAssigner()
    assert parse_real_message({"trnm": "PING"}, TDAY, seq) == ([], [])
    assert parse_real_message({"trnm": "REG", "return_code": 0}, TDAY, seq) == ([], [])


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
    assert collector.frames_seen == 2
    assert count_ticks(conn) == 1
    assert count_quotes(conn) == 1
    assert collector.ticks_stored == 1


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
