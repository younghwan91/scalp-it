"""수집 슬롯 채우기 — 매매가 안 쓰는 자리로 틱을 쌓는다.

만쥬 규칙 검증에 필요한 건 장중 미시구조인데(사전등록 47번 결론), 감지기는 후보
10~20종목만 구독해 정원 50 중 30자리 이상이 하루 종일 빈다. 2026-08-28 실측에서
구독은 12종목, 저장 틱은 6만행이었다(8/24 는 80종목·180만행).

**수집 슬롯은 감지기에 등록하지 않는다** — 신호도 주문도 만들지 않고 저장만 한다.
"""
from scalp_it.realtime.collect_fill import plan_fill


def test_fills_the_room_left_by_trading():
    """매매가 안 쓰는 자리만 채운다 — 예비분은 남긴다."""
    got = plan_fill(candidates=["A", "B", "C", "D", "E"],
                    subscribed=["X", "Y"], cap=6, reserve=2)
    assert got == ["A", "B"]          # 6 − 2(구독중) − 2(예비) = 2자리


def test_never_takes_a_code_already_subscribed():
    """이미 구독 중인 종목을 다시 붙이면 등록만 낭비한다."""
    got = plan_fill(candidates=["A", "B", "C"],
                    subscribed=["A"], cap=10, reserve=0)
    assert got == ["B", "C"]


def test_yields_everything_when_trading_is_busy():
    """감지기가 자리를 다 쓰면 수집은 한 자리도 안 가져간다."""
    assert plan_fill(candidates=["A"], subscribed=list("ABCDEFGH"),
                     cap=10, reserve=2) == []


def test_reserve_larger_than_cap_never_goes_negative():
    assert plan_fill(candidates=["A"], subscribed=[], cap=3, reserve=99) == []


def test_keeps_candidate_order_and_drops_duplicates():
    """후보 순서(거래대금 상위)가 우선순위다."""
    got = plan_fill(candidates=["A", "A", "B"], subscribed=[], cap=10, reserve=0)
    assert got == ["A", "B"]


def test_collector_reports_live_subscription_and_capacity():
    """수집 슬롯은 **라이브 등록 목록**을 봐야 한다 — 장전 목록은 이미 낡았다."""
    from scalp_it.realtime.collector import KiwoomRealtimeCollector

    c = KiwoomRealtimeCollector.__new__(KiwoomRealtimeCollector)
    c._types = ["0B", "0D"]
    c._reg_limit = 100
    c._live_ws = {}
    assert c.subscribed_codes == []
    assert c.code_capacity == 50                      # 100 ÷ 2

    c._live_ws = {1: (object(), ["000001", "000002"])}
    assert c.subscribed_codes == ["000001", "000002"]


def test_fill_never_reaches_the_detector_or_the_order_whitelist():
    """**이 테스트가 이 기능의 안전장치다.**

    수집 슬롯이 감지기에 등록되면 그 종목에서 신호가 나고, 주문 화이트리스트까지
    넓어지면 실주문이 나간다. 수집은 저장만 해야 한다 — `_watch_collect_fill` 은
    `collector.add_codes` 만 부르고 `detector.add_codes`·`_allow_orders` 는
    부르지 않는다. 소스를 직접 검사한다(호출이 하나라도 늘면 여기서 걸린다).
    """
    import inspect

    from scalp_it import cli_pair_detect

    src = inspect.getsource(cli_pair_detect._watch_collect_fill)
    assert "collector.add_codes" in src
    assert "detector.add_codes" not in src, "수집 종목이 감지기에 등록되면 안 된다"
    assert "_allow_orders" not in src, "수집 종목이 주문 허용에 들어가면 안 된다"


def test_cron_turns_collect_fill_on_with_room_left_for_trading(cron_args):
    """cron 이 수집 채움을 켜고 있고, 매매 예비 자리를 남기는지.

    정원 50(등록 100 ÷ 0B·0D) 중 예비 12 를 빼면 수집은 최대 38자리까지 쓴다.
    """
    args = cron_args
    assert args.collect_fill_interval == 60
    assert args.collect_reserve == 12


def test_watcher_actually_subscribes_from_the_db(tmp_path):
    """**장중에 처음 도는 코드다** — 한 바퀴를 실제로 돌려 본다.

    순수 함수와 소스 검사만으로는 SQL 오타·플레이스홀더 불일치·속성 이름 실수를
    못 잡는다. sqlite 로 daily_bars 를 만들어 한 사이클을 태운다.
    """
    import argparse
    import asyncio
    import sqlite3

    from scalp_it import cli_pair_detect as C

    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute("create table daily_bars (code text, date text, trade_value real)")
    conn.executemany("insert into daily_bars values (?,?,?)",
                     [("A", "2026-08-28", 900.0), ("B", "2026-08-28", 800.0),
                      ("C", "2026-08-28", 700.0)])
    conn.commit()

    class FakeCollector:
        code_capacity = 5
        subscribed_codes = ["Z"]

        def __init__(self) -> None:
            self.added: list[str] = []

        async def add_codes(self, codes):
            self.added.extend(codes)
            return list(codes), []

    class FakeDetector:
        codes = {"B"}          # 감지기가 아는 종목은 수집이 안 건드린다

    class FakeBus:
        def __init__(self) -> None:
            self.msgs: list[str] = []

        def notice(self, msg: str) -> None:
            self.msgs.append(msg)

    collector, bus = FakeCollector(), FakeBus()
    args = argparse.Namespace(collect_fill_interval=0.01, collect_reserve=2,
                              collect_fill_pool=300)

    async def run_one_cycle():
        task = asyncio.create_task(
            C._watch_collect_fill(collector, FakeDetector(), bus, args, conn))
        await asyncio.sleep(0.2)
        task.cancel()

    asyncio.run(run_one_cycle())

    # 정원 5 − 구독중 1 − 예비 2 = 2자리. 후보는 거래대금 순 A,B,C 인데 B 는 감지기 몫.
    assert collector.added == ["A", "C"]
    assert any("수집채움" in m for m in bus.msgs)
