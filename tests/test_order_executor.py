# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""주문 실행기 테스트. mock 주문 클라이언트로 체결·청산·dedup·킬을 검증한다.

**실 클라이언트·실 네트워크 없음.** 합성 알림·틱만.
"""

from datetime import datetime, timedelta

from scalp_it.adapters.kiwoom_order import OrderIntent, OrderResult
from scalp_it.realtime.order_executor import OrderExecutor
from scalp_it.realtime.pair_detector import Alert


class FakeClient:
    """KiwoomOrderClient 대역 — 실제 발주 대신 호출을 기록한다."""

    def __init__(self, filled=True):
        self.buys = []
        self.sells = []
        self.cancels = []
        self._filled = filled

    def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.buys.append((code, qty, price, live))
        intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
        return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="B1")

    def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.sells.append((code, qty, price, live))
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="S1")

    def cancel(self, ord_no, code, qty, *, exchange="KRX", live=False):
        self.cancels.append((ord_no, code, qty, live))
        intent = OrderIntent(side="cancel", code=code, qty=qty, price=0)
        return OrderResult(intent=intent, dry_run=not live, submitted=live)

    def is_filled(self, ord_no, code=None):
        return self._filled


def _alert(code="294570", ts=None, entry=10000, tp=10050, sl=9960,
           disaster=None, exit_after=90):
    t = ts or datetime(2026, 8, 25, 9, 30, 0)
    return Alert(
        ts=t, leader="124500", follower=code, leader_surge=0.02, cosurge=0.006,
        premove=0.001, price=float(entry), bid1=float(entry), entry=float(entry),
        take_profit=float(tp), stop_loss=(None if sl is None else float(sl)),
        disaster_line=(None if disaster is None else float(disaster)),
        exit_ts=t + timedelta(seconds=exit_after), turnover=1e9,
        leader_name="리더", follower_name="쿠콘",
    )


def test_buy_on_signal_dry_run():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, live=False)
    ex.on_signal(_alert())
    assert c.buys == [("294570", 1, 10000, False)]     # live=False
    assert len(ex.positions) == 1 and ex.positions[0].status == "pending"


def test_dedup_same_code_second_signal_skipped():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    ex.on_signal(_alert(entry=10000))
    ex.on_signal(_alert(entry=10000))                  # 같은 종목 재신호
    assert len(c.buys) == 1                             # 두 번째 매수 생략
    assert len(ex.positions) == 1


def test_fill_then_take_profit_sells():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    # 진입가로 내려오면 체결(dry touch)
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)
    assert ex.positions[0].status == "filled"
    # 익절가 도달 → 매도
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)
    assert c.sells and c.sells[0][0] == "294570"
    assert ex.history[-1].exit_reason == "익절"


def test_stop_loss_sells():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, sl=9960))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 10000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=3), 9960)    # 손절
    assert ex.history[-1].exit_reason == "손절" and c.sells


def test_time_exit_sells():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1, exit_after=90))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 10000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=95), 10010)  # 청산시각 지남
    assert ex.history[-1].exit_reason == "시간청산"


def test_unfilled_cancel_after_timeout():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000))
    # 진입가 위에서만 거래 → 미체결. 15초 지나면 취소.
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10100)
    assert c.cancels and ex.history[-1].exit_reason == "미체결취소"


def test_on_kill_cancels_pending():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    ex.on_signal(_alert())
    ex.on_kill()
    assert c.cancels and ex.positions[0].status == "cancelled"


def test_blocked_result_not_tracked_as_open():
    class BlockingClient(FakeClient):
        def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
            return OrderResult(intent=intent, dry_run=not live, submitted=False,
                               blocked=True, blocked_reason="블록리스트")

    c = BlockingClient()
    ex = OrderExecutor(c, qty=1)
    ex.on_signal(_alert())
    assert ex.positions == []                          # 차단분은 활성 아님
    assert ex.history[-1].status == "blocked"


def test_live_flag_flows_to_client():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, live=True)
    ex.on_signal(_alert())
    assert c.buys[0][3] is True                        # live=True 전달
