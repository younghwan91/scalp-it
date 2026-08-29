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

        async def remove_codes(self, codes):
            return list(codes)

    class FakeDetector:
        codes = {"B"}          # 감지기가 아는 종목은 수집이 안 건드린다

    class FakeBus:
        def __init__(self) -> None:
            self.msgs: list[str] = []

        def notice(self, msg: str) -> None:
            self.msgs.append(msg)

    collector, bus = FakeCollector(), FakeBus()
    args = argparse.Namespace(collect_fill_interval=0.01, collect_reserve=2,
                              collect_fill_pool=300, collect_fill_ttl=600.0)

    async def run_one_cycle():
        task = asyncio.create_task(
            C._watch_collect_fill(collector, FakeDetector(), bus, args, conn))
        await asyncio.sleep(0.2)
        task.cancel()

    asyncio.run(run_one_cycle())

    # 정원 5 − 구독중 1 − 예비 2 = 2자리. 후보는 거래대금 순 A,B,C 인데 B 는 감지기 몫.
    assert collector.added == ["A", "C"]
    assert any("수집채움" in m for m in bus.msgs)


def test_pool_prefers_todays_movers_over_yesterdays_turnover():
    """수집 후보는 **오늘 올라가는 종목**이어야 한다 — 만쥬의 순간을 보려면.

    전일 거래대금 상위로 채우면 틱은 쌓이지만 방아쇠는 못 본다. 실측 커버리지가
    2% 였던 바로 그 기준이다(그 주 상한가 52건 중 유니버스 안에 1건). 스캐너가 이미
    10초마다 등락률 상위를 받아오니, 그 순위를 문턱 아래까지 후보로 쓴다 — 12% 를
    못 넘은 종목이 이따가 상한가로 가고, 그때 이미 붙어 있어야 한다.
    """
    from scalp_it.realtime.collect_fill import fill_pool

    movers = ["M1", "M2", "M3"]          # 오늘 등락률 상위(문턱 아래 포함)
    turnover = ["T1", "T2"]              # 전일 거래대금 상위

    assert fill_pool(movers, turnover)[:3] == movers, "오늘 움직이는 종목이 먼저다"
    assert "T1" in fill_pool(movers, turnover), "그 뒤를 전일 거래대금이 받친다"
    assert fill_pool([], turnover) == turnover, "스캔이 아직 없으면 폴백"
    assert fill_pool(["M1", "T1"], turnover) == ["M1", "T1", "T2"], "중복은 한 번만"


def test_stale_fill_codes_yield_their_slot():
    """순위에서 사라진 지 오래된 수집 종목은 자리를 내준다.

    자리를 한 번 잡고 안 놓으면 오전에 붙은 38종목이 하루를 다 먹는다. 만쥬의 대장은
    오후에도 나오고, 그때 자리가 없으면 그날 그 종목은 못 본다. 반대로 순위를 잠깐
    들락거린다고 바로 떼면 등록만 낭비하므로 TTL 을 둔다.
    """
    from scalp_it.realtime.collect_fill import plan_evictions

    seen = {"A": 1000.0, "B": 700.0, "C": 100.0}
    # now=1000, ttl=600 → C(900초 전)만 만료. B(300초 전)는 유예 안에 있다.
    assert plan_evictions(seen, now=1000.0, ttl=600.0) == ["C"]


def test_nothing_evicted_when_all_fresh():
    from scalp_it.realtime.collect_fill import plan_evictions

    assert plan_evictions({"A": 990.0}, now=1000.0, ttl=600.0) == []


def test_eviction_is_deterministic_oldest_first():
    """오래 안 보인 것부터 내린다 — 자리가 하나뿐일 때 무엇을 버릴지 정해져야 한다."""
    from scalp_it.realtime.collect_fill import plan_evictions

    seen = {"A": 100.0, "B": 50.0, "C": 200.0}
    assert plan_evictions(seen, now=1000.0, ttl=600.0) == ["B", "A", "C"]


def test_afternoon_mover_gets_a_slot_from_a_stale_one(tmp_path):
    """오후에 올라온 종목이 **아침에 붙은 종목의 자리를 받는다.**

    만쥬의 대장은 오후에도 나온다. 자리를 한 번 잡고 안 놓으면 그날 그 종목은 못
    보고, 그러면 만쥬 방식이 성립하지 않는다. 순위에서 사라진 지 TTL 이 지난
    종목을 내리고 새 종목을 붙이는지 한 바퀴 돌려 본다.
    """
    import argparse
    import asyncio
    import sqlite3

    from scalp_it import cli_pair_detect as C

    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute("create table daily_bars (code text, date text, trade_value real)")
    conn.execute("insert into daily_bars values ('T1','2026-08-28',100.0)")
    conn.commit()

    class FakeCollector:
        code_capacity = 1          # 자리 하나뿐 — 회전을 해야만 새 종목이 들어온다
        def __init__(self):
            self.subscribed_codes: list[str] = []
            self.added: list[str] = []
            self.removed: list[str] = []

        async def add_codes(self, codes):
            self.added.extend(codes)
            self.subscribed_codes = [*self.subscribed_codes, *codes]
            return list(codes), []

        async def remove_codes(self, codes):
            self.removed.extend(codes)
            self.subscribed_codes = [c for c in self.subscribed_codes
                                     if c not in codes]
            return list(codes)

    class FakeDetector:
        codes: set[str] = set()

    class FakeBus:
        def __init__(self): self.msgs: list[str] = []
        def notice(self, m): self.msgs.append(m)

    collector, bus = FakeCollector(), FakeBus()
    movers = ["OLD"]
    args = argparse.Namespace(collect_fill_interval=0.01, collect_reserve=0,
                              collect_fill_pool=300, collect_fill_ttl=0.05)

    async def scenario():
        task = asyncio.create_task(
            C._watch_collect_fill(collector, FakeDetector(), bus, args, conn, movers))
        await asyncio.sleep(0.2)           # OLD 가 순위에 있는 동안
        movers[:] = ["NEW"]                # 오후 — 순위가 갈렸다
        await asyncio.sleep(1.5)           # TTL(0.05s) 경과 + 다음 사이클
        task.cancel()

    asyncio.run(scenario())

    assert collector.added[0] == "OLD", "먼저 아침 종목을 붙인다"
    assert "OLD" in collector.removed, "순위에서 사라진 종목이 자리를 내줘야 한다"
    assert "NEW" in collector.added, "오후 종목이 그 자리를 받아야 한다"
