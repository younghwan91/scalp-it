"""86번 호랑이 규칙 실시간 그림자 — 주문 없음. 규칙 정본은 docs/research/manju/86 §2~§4.

리플레이(scripts/tiger_86/engine.py)와 같은 판정을 소켓 프레임에서 한다:
타기 = 누적 체결대금 상위 30 · 체결강도 30초 변화 ≥ 문턱 · 10초 매수 같은 수량 ≥3 · 3초 매수 > 매도,
진입 = 다음 초 매수1호가 지정가(대기열: 우리 가격 매도 체결 누적 ≥ 앞 잔량+1, 또는 가격 뚫림),
10초 미체결·불꺼짐이면 취소, 청산 = 체결강도 최고치 −1 또는 −2호가.
"""
from datetime import datetime, timedelta

from scalp_it.realtime.tiger_shadow import TigerShadow


def _t(sec_after_0930: int) -> datetime:
    return datetime(2026, 9, 16, 9, 30, 0) + timedelta(seconds=sec_after_0930)


def _shadow(q=5.0):
    logged: list[dict] = []
    s = TigerShadow(prev_quantile=q, top_n=30, sink=logged.append, min_quantile_rows=10**9)
    return s, logged


def _warm(s, code="100000", until=40, price=10000, strength=100.0):
    for k in range(until):
        s.on_quote(code, _t(k), bid1=price - 10, ask1=price, bidqty1=500)
        # 수량을 매초 다르게 — 같은 수량 반복(알고리즘 체결)으로 잡히지 않게
        s.on_tick(code, _t(k), price, volume=100 + k, side=1, strength=strength, best_bid=price - 10, best_ask=price)


def _algo_burst(s, code, k, price=10000, strength=110.0):
    for _ in range(3):                      # 같은 수량 7주 매수 3건 = 알고리즘 체결
        s.on_tick(code, _t(k), price, volume=7, side=1, strength=strength, best_bid=price - 10, best_ask=price)


def test_no_signal_without_algo_repeats():
    s, logged = _shadow()
    _warm(s)
    s.on_tick("100000", _t(40), 10000, volume=50, side=1, strength=120.0, best_bid=9990, best_ask=10000)
    s.on_tick("100000", _t(41), 10000, volume=1, side=1, strength=120.0, best_bid=9990, best_ask=10000)
    assert s.pending_count() == 0


def test_limit_fills_through_queue_then_stops_out():
    s, logged = _shadow()
    _warm(s)
    _algo_burst(s, "100000", 40)                                          # 신호 초 40
    s.on_quote("100000", _t(41), bid1=9990, ask1=10000, bidqty1=5)        # 다음 초 매수1 9990, 앞 잔량 5
    s.on_tick("100000", _t(41), 10000, volume=1, side=1, strength=110.0, best_bid=9990, best_ask=10000)
    assert s.pending_count() == 1
    s.on_tick("100000", _t(42), 9990, volume=6, side=-1, strength=110.0, best_bid=9990, best_ask=10000)  # 6 ≥ 5+1 → 체결
    s.on_tick("100000", _t(43), 9990, volume=1, side=1, strength=110.5, best_bid=9990, best_ask=10000)   # 42초 마감 → 체결
    assert s.open_count() == 1
    s.on_tick("100000", _t(50), 9970, volume=1, side=-1, strength=110.4, best_bid=9970, best_ask=9980)   # −2호가
    s.on_tick("100000", _t(51), 9970, volume=1, side=1, strength=110.4, best_bid=9970, best_ask=9980)    # 50초 마감
    assert s.open_count() == 0
    (r,) = [x for x in logged if x["event"] == "close"]
    assert r["entry"] == 9990 and r["exit"] == 9970 and r["reason"] == "손절"


def test_unfilled_limit_cancels_after_10s():
    s, logged = _shadow()
    _warm(s)
    _algo_burst(s, "100000", 40)
    s.on_quote("100000", _t(41), bid1=9990, ask1=10000, bidqty1=10_000)
    for k in range(41, 53):
        s.on_tick("100000", _t(k), 10000, volume=1, side=1, strength=110.0, best_bid=9990, best_ask=10000)
    assert s.pending_count() == 0 and s.open_count() == 0
    assert any(x["event"] == "cancel" for x in logged)


def test_strength_break_exits():
    s, logged = _shadow()
    _warm(s)
    _algo_burst(s, "100000", 40)
    s.on_quote("100000", _t(41), bid1=9990, ask1=10000, bidqty1=0)
    s.on_tick("100000", _t(41), 10000, volume=1, side=1, strength=110.0, best_bid=9990, best_ask=10000)
    s.on_tick("100000", _t(42), 9980, volume=1, side=-1, strength=112.0, best_bid=9990, best_ask=10000)  # 뚫림 체결
    s.on_tick("100000", _t(44), 10010, volume=1, side=1, strength=113.0, best_bid=10000, best_ask=10010)
    s.on_tick("100000", _t(45), 10000, volume=1, side=-1, strength=111.9, best_bid=10000, best_ask=10010)  # 113−1 이하
    s.on_tick("100000", _t(46), 10000, volume=1, side=1, strength=111.9, best_bid=10000, best_ask=10010)   # 45초 마감
    (r,) = [x for x in logged if x["event"] == "close"]
    assert r["reason"] == "불꺼짐" and r["exit"] == 10000


def test_only_top_turnover_codes_trade():
    logged: list[dict] = []
    s = TigerShadow(prev_quantile=5.0, top_n=1, sink=logged.append, min_quantile_rows=10**9)
    _warm(s, code="BIG", price=50000)          # 체결대금이 훨씬 크다
    _warm(s, code="SMALL", price=1000)
    _algo_burst(s, "SMALL", 40, price=1000)
    s.on_tick("SMALL", _t(41), 1000, volume=1, side=1, strength=110.0, best_bid=999, best_ask=1000)
    s.on_tick("SMALL", _t(42), 1000, volume=1, side=1, strength=110.0, best_bid=999, best_ask=1000)
    assert s.pending_count() == 0, "누적 체결대금 상위 밖이면 타지 않는다"


def test_cron_turns_on_tiger_shadow(cron_args):
    assert cron_args.tiger_shadow is True


def test_tiger_shadow_failures_never_reach_trading_loop(capsys):
    import argparse
    from datetime import date

    from scalp_it import cli_pair_detect as C

    args = argparse.Namespace(tiger_shadow=True, tiger_shadow_log="/nonexistent/x.jsonl",
                              tiger_prevq="/nonexistent.json", dsn="postgresql://nobody@127.0.0.1:1/none")
    assert C._build_tiger_shadow(args, date(2026, 9, 16)) == (None, set())

    class Boom:
        def on_quote(self, *a, **k):
            raise RuntimeError("boom")

    C._feed_tiger_shadow(Boom(), {"005930"}, [], [{"code": "005930", "ts": None}])
    assert "호랑이그림자" in capsys.readouterr().err
