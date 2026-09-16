"""RL 정책 → 실주문 다리 — **승격 토큰이 있을 때만** 주문이 나간다.

승격 체크리스트(docs/PROMOTION_CHECKLIST.md) 11개를 통과하면 `research/runs/<study>/PROMOTED` 가 생기고,
그때부터 이 다리가 열린다. 토큰이 없으면 그림자 기록만 남는다 — 실수로 켜질 수 없게 하는 게 목적이다.
"""
from scalp_it.realtime.rl_order_bridge import make_sink


class _Exec:
    def __init__(self):
        self.calls = []

    def submit_buy(self, code, price, qty=1):
        self.calls.append(("buy", code, price, qty))

    def submit_sell(self, code, price, qty=1):
        self.calls.append(("sell", code, price, qty))


def test_no_token_means_no_order(tmp_path):
    logged, ex = [], _Exec()
    sink = make_sink(logged.append, executor=ex, study="rl93", runs_dir=tmp_path, live=True)
    sink({"event": "buy", "code": "005930", "sec": 100, "price": 10000, "kind": "시장가"})
    assert ex.calls == [], "승격 토큰이 없으면 주문이 나가면 안 된다"
    assert logged and logged[0]["order"] == "차단(미승격)"


def test_token_but_live_flag_off_means_no_order(tmp_path):
    (tmp_path / "rl93").mkdir()
    (tmp_path / "rl93" / "PROMOTED").write_text("ok")
    logged, ex = [], _Exec()
    sink = make_sink(logged.append, executor=ex, study="rl93", runs_dir=tmp_path, live=False)
    sink({"event": "buy", "code": "005930", "sec": 100, "price": 10000, "kind": "시장가"})
    assert ex.calls == []
    assert logged[0]["order"] == "그림자"


def test_token_and_live_flag_places_order(tmp_path):
    (tmp_path / "rl93").mkdir()
    (tmp_path / "rl93" / "PROMOTED").write_text("ok")
    logged, ex = [], _Exec()
    sink = make_sink(logged.append, executor=ex, study="rl93", runs_dir=tmp_path, live=True)
    sink({"event": "buy", "code": "005930", "sec": 100, "price": 10000, "kind": "시장가"})
    sink({"event": "sell", "code": "005930", "sec": 140, "price": 10050, "ret": 0.002})
    assert ex.calls == [("buy", "005930", 10000, 1), ("sell", "005930", 10050, 1)]
    assert [r["order"] for r in logged] == ["실주문", "실주문"]


def test_executor_failure_never_breaks_logging(tmp_path):
    (tmp_path / "rl93").mkdir()
    (tmp_path / "rl93" / "PROMOTED").write_text("ok")

    class Boom:
        def submit_buy(self, *a, **k):
            raise RuntimeError("거부")

    logged = []
    sink = make_sink(logged.append, executor=Boom(), study="rl93", runs_dir=tmp_path, live=True)
    sink({"event": "buy", "code": "005930", "sec": 100, "price": 10000, "kind": "시장가"})
    assert logged[0]["order"].startswith("실패")


def test_non_trade_events_pass_through(tmp_path):
    logged, ex = [], _Exec()
    sink = make_sink(logged.append, executor=ex, study="rl93", runs_dir=tmp_path, live=True)
    sink({"event": "limit_intent", "code": "005930", "sec": 100, "price": 9990})
    assert ex.calls == [] and logged[0]["event"] == "limit_intent"
