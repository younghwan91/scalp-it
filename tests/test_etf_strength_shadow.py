"""81번 ETF 체결강도 규칙의 실시간 그림자 — 주문 없이 가상 체결만 기록한다.

판정은 DB 리플레이(`scripts/etf_strength_81.py`)가 정본이다. 이 그림자는 채택 당일
실주문으로 뒤집을 수 있게 **같은 규칙을 소켓에서 돌려두는 것**이고, 리플레이와의
괴리(패리티)를 매일 보는 용도다.
"""
from datetime import datetime

from scalp_it.realtime.etf_strength import Cell, StrengthShadow


def _ts(hms: str) -> datetime:
    return datetime.fromisoformat(f"2026-09-16 {hms}")


def _shadow(thr=1.0, cells=(Cell(q=0.99, sdrop=3.0, T=300),)):
    logged: list[dict] = []
    s = StrengthShadow({("069500", c.q): thr for c in cells}, cells=cells, sink=logged.append)
    return s, logged


def _warm(s, strength=100.0, until="09:30:00", px=(10000, 10005)):
    """09:29:00 부터 until 전까지 매초 체결강도를 고정해 30초 전 값을 만든다."""
    t0, end = _ts("09:29:00"), _ts(until)
    for sec in range(int((end - t0).total_seconds())):
        cur = datetime.fromtimestamp(t0.timestamp() + sec)
        s.on_tick("069500", cur, px[1], strength=strength, best_bid=px[0], best_ask=px[1])


def test_no_threshold_means_no_trade():
    s = StrengthShadow({}, cells=(Cell(0.99, 3.0, 300),), sink=lambda r: None)
    s.on_tick("069500", _ts("09:30:00"), 10005, strength=150.0, best_bid=10000, best_ask=10005)
    assert s.open_count() == 0


def test_enters_on_acceleration_and_exits_when_strength_breaks():
    s, logged = _shadow(thr=5.0)
    _warm(s, strength=100.0)
    # 09:30:00 체결강도 +10 (30초 전 100) → 신호. 진입은 1초 뒤 매도1호가.
    s.on_tick("069500", _ts("09:30:00"), 10005, strength=110.0, best_bid=10000, best_ask=10005)
    assert s.open_count() == 0, "신호 초에는 아직 안 샀다(1초 지연)"
    s.on_tick("069500", _ts("09:30:01"), 10010, strength=112.0, best_bid=10005, best_ask=10010)
    assert s.open_count() == 1
    s.on_tick("069500", _ts("09:30:05"), 10020, strength=115.0, best_bid=10020, best_ask=10025)
    # 최고 115 − 3 = 112 이하로 꺾임 → 그 초 매수1호가에 청산
    s.on_tick("069500", _ts("09:30:09"), 10030, strength=111.5, best_bid=10030, best_ask=10035)
    assert s.open_count() == 0
    (r,) = logged
    assert r["entry"] == 10010 and r["exit"] == 10030 and r["reason"] == "강도꺾임"
    assert abs(r["ret"] - (10030 / 10010 - 1 - 0.0003)) < 1e-9


def test_time_exit():
    s, logged = _shadow(thr=5.0, cells=(Cell(0.99, 10.0, 60),))
    _warm(s)
    s.on_tick("069500", _ts("09:30:00"), 10005, strength=110.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:30:01"), 10005, strength=110.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:31:01"), 10000, strength=110.0, best_bid=9995, best_ask=10000)
    (r,) = logged
    assert r["reason"] == "시간" and r["exit"] == 9995


def test_wide_spread_blocks_entry():
    s, _ = _shadow(thr=5.0)
    _warm(s)
    s.on_tick("069500", _ts("09:30:00"), 10010, strength=110.0, best_bid=10000, best_ask=10010)
    s.on_tick("069500", _ts("09:30:01"), 10010, strength=110.0, best_bid=10000, best_ask=10010)
    assert s.open_count() == 0, "스프레드 2틱(5원 단위)이면 진입 안 함"


def test_outside_window_no_entry():
    s, _ = _shadow(thr=5.0)
    s.on_tick("069500", _ts("09:00:30"), 10005, strength=100.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:01:00"), 10005, strength=150.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:01:01"), 10005, strength=150.0, best_bid=10000, best_ask=10005)
    assert s.open_count() == 0, "09:05 이전은 진입 안 함(§2)"


def test_cooldown_60s_from_signal():
    s, logged = _shadow(thr=5.0, cells=(Cell(0.99, 3.0, 300),))
    _warm(s)
    s.on_tick("069500", _ts("09:30:00"), 10005, strength=110.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:30:01"), 10005, strength=110.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:30:02"), 10005, strength=106.0, best_bid=10000, best_ask=10005)  # 꺾임 청산
    assert len(logged) == 1
    # 09:30:40 다시 가속이어도 신호(09:30:00)로부터 60초 안 → 무시
    s.on_tick("069500", _ts("09:30:40"), 10005, strength=130.0, best_bid=10000, best_ask=10005)
    s.on_tick("069500", _ts("09:30:41"), 10005, strength=130.0, best_bid=10000, best_ask=10005)
    assert s.open_count() == 0


def test_cron_runs_the_shadow_on_the_pinned_etfs(cron_args):
    """그림자 대상 = 수집 고정 대상. 어긋나면 틱이 안 들어오는 종목을 그림자가 기다린다."""
    assert cron_args.etf_shadow == cron_args.collect_pin


def test_shadow_failures_never_reach_the_trading_loop(capsys):
    import argparse
    from datetime import date

    from scalp_it import cli_pair_detect as C

    args = argparse.Namespace(etf_shadow="069500", etf_shadow_log="/nonexistent/x.jsonl",
                              dsn="postgresql://nobody@127.0.0.1:1/none")
    assert C._build_etf_shadow(args, date(2026, 9, 16)) is None

    class Boom:
        def on_quote(self, *a, **k):
            raise RuntimeError("boom")

    C._feed_etf_shadow(Boom(), [], [{"code": "069500", "ts": None}])   # 예외를 삼켜야 한다
    assert "ETF그림자" in capsys.readouterr().err
