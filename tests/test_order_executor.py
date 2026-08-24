# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""주문 실행기 테스트. mock 주문 클라이언트로 체결·청산·dedup·킬을 검증한다.

**실 클라이언트·실 네트워크 없음.** 합성 알림·틱만.
"""

from datetime import datetime, timedelta

from scalp_it.adapters.kiwoom import KiwoomError
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


# ------------------------------------------------------------------ N-1: 매도 미성사 → open 유지

class _BlockingSellClient(FakeClient):
    def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.sells.append((code, qty, price, live))
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        return OrderResult(intent=intent, dry_run=not live, submitted=False,
                           blocked=True, blocked_reason="블록리스트")


def test_sell_blocked_keeps_position_open_not_exited():
    """N-1(b): 청산 매도가 차단되면 '청산완료' 표기 금지 → 포지션 open 유지."""
    c = _BlockingSellClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 익절 매도 시도 → 차단
    assert c.sells                                            # 매도 시도는 했다
    assert len(ex.positions) == 1                            # 무감시 방치 금지
    assert ex.positions[0].status == "filled"               # exited 로 착각 안 함
    assert all(p.status != "exited" for p in ex.history)


def test_sell_retries_next_tick_after_transient_failure():
    """N-1/N-2: 1차 매도 실패 후 다음 틱에 재시도해 성사되면 exited."""

    class _FlakySell(FakeClient):
        def __init__(self):
            super().__init__()
            self._first = True

        def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            self.sells.append((code, qty, price, live))
            intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
            if self._first:
                self._first = False
                return OrderResult(intent=intent, dry_run=not live, submitted=False,
                                   blocked=True, blocked_reason="일시차단")
            return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="S1")

    c = _FlakySell()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 1차 매도 차단 → open 유지
    assert ex.positions[0].status == "filled"
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10050)   # 2차 매도 성사
    assert ex.history[-1].status == "exited" and ex.history[-1].exit_reason == "익절"
    assert len(c.sells) == 2


# ------------------------------------------------------------------ N-2: 주문 거부(예외) → 루프 지속

def test_buy_exception_does_not_crash_and_skips_position():
    """N-2: 매수 거부(KiwoomError)로 크래시 금지 → 포지션 안 열고 스킵."""

    class _RaisingBuy(FakeClient):
        def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            raise KiwoomError("가격밴드 이탈 거부")

    c = _RaisingBuy()
    ex = OrderExecutor(c, qty=1)
    res = ex.on_signal(_alert())                            # 예외 전파 안 됨
    assert res is None and ex.positions == []


def test_sell_exception_keeps_position_open_and_continues():
    """N-2: 매도 거부(KiwoomError)로 크래시 금지 → open 유지, 루프 계속."""

    class _RaisingSell(FakeClient):
        def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            raise KiwoomError("장마감 거부")

    c = _RaisingSell()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 매도 예외 → 삼켜짐
    assert len(ex.positions) == 1 and ex.positions[0].status == "filled"


def test_cancel_exception_keeps_pending():
    """N-2: 미체결 취소 거부(KiwoomError)로 크래시 금지 → pending 유지."""

    class _RaisingCancel(FakeClient):
        def cancel(self, ord_no, code, qty, *, exchange="KRX", live=False):
            raise KiwoomError("취소 거부")

    c = _RaisingCancel()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000))
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10100)  # 미체결 취소 시도 → 예외
    assert len(ex.positions) == 1 and ex.positions[0].status == "pending"


def test_other_positions_managed_after_one_rejection():
    """N-2: 한 종목 매도 거부가 다른 종목의 정상 청산을 막지 않는다."""

    class _RejectOne(FakeClient):
        def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            self.sells.append((code, qty, price, live))
            intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
            if code == "294570":
                raise KiwoomError("거부")
            return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="S1")

    c = _RejectOne()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    a1 = _alert(code="294570", ts=t0, entry=10000, tp=10050)
    a2 = _alert(code="033340", ts=t0, entry=20000, tp=20100)
    ex.on_signal(a1)
    ex.on_signal(a2)
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)   # 체결
    ex.on_tick("033340", t0 + timedelta(seconds=2), 20000)   # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 거부 → open 유지
    ex.on_tick("033340", t0 + timedelta(seconds=5), 20100)   # 정상 청산
    codes_open = {p.code for p in ex.positions}
    assert "294570" in codes_open                            # 거부분은 open 유지
    assert any(p.code == "033340" and p.status == "exited" for p in ex.history)
