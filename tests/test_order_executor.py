# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""주문 실행기 테스트. mock 주문 클라이언트로 체결·청산·dedup·킬을 검증한다.

**실 클라이언트·실 네트워크 없음.** 합성 알림·틱만.
"""

from datetime import datetime, timedelta

from scalp_it.adapters.kiwoom import KiwoomError
from scalp_it.adapters.kiwoom_order import OrderIntent, OrderResult
from scalp_it.realtime.order_executor import (
    LEADER_HOLDING,
    LEADER_LOST,
    OrderExecutor,
)
from scalp_it.realtime.pair_detector import Alert


class FakeClient:
    """KiwoomOrderClient 대역 — 실제 발주 대신 호출을 기록한다."""

    def __init__(self, filled=True, cancel_rc=0, holdings=None):
        self.buys = []
        self.sells = []
        self.cancels = []
        self._filled = filled
        self._cancel_rc = cancel_rc          # 0 이면 취소 성공, 그 외는 브로커 거부
        self._holdings = holdings            # live 매도 체결확인용 잔고 대역

    def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.buys.append((code, qty, price, live))
        intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
        # live 발주 성공은 rc=0 — OrderResult.ok 가 실주문에서 rc 를 본다.
        return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="B1",
                           return_code=(0 if live else None))

    def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.sells.append((code, qty, price, live))
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        # live 발주 성공은 rc=0 — OrderResult.ok 가 실주문에서 rc 를 본다.
        return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="S1",
                           return_code=(0 if live else None))

    def cancel(self, ord_no, code, qty, *, exchange="KRX", live=False):
        self.cancels.append((ord_no, code, qty, live))
        intent = OrderIntent(side="cancel", code=code, qty=qty, price=0)
        if self._cancel_rc == 0:
            return OrderResult(intent=intent, dry_run=not live, submitted=live,
                               return_code=(0 if live else None))
        # 거부 — 어댑터 cancel() 은 rc!=0 에 예외를 **안 던진다**(HTTP 4xx 만 던짐).
        # 거부는 실주문에서만 나므로 dry_run=False 로 둔다(dry_run=True 면 ok 가 True).
        return OrderResult(intent=intent, dry_run=False, submitted=True,
                           return_code=self._cancel_rc, return_msg="이미 체결된 주문")

    def is_filled(self, ord_no, code=None):
        return self._filled

    def holdings(self, *, force=False):
        return list(self._holdings or [])


def _alert(code="294570", ts=None, entry=10000, tp=10050, sl=9960,
           disaster=None, exit_after=90, leader="124500"):
    t = ts or datetime(2026, 8, 25, 9, 30, 0)
    return Alert(
        ts=t, leader=leader, follower=code, leader_surge=0.02, cosurge=0.006,
        premove=0.001, price=float(entry), bid1=float(entry), entry=float(entry),
        take_profit=float(tp), stop_loss=(None if sl is None else float(sl)),
        disaster_line=(None if disaster is None else float(disaster)),
        exit_ts=t + timedelta(seconds=exit_after), turnover=1e9,
        leader_name="리더", follower_name="쿠콘",
    )


def _last(ex):
    """마지막으로 다룬 포지션. 매도 접수 후 체결확인 전이면 positions 에 남아 있다."""
    return (ex.history + ex.positions)[-1] if (ex.history or ex.positions) else None


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
    # 진입가를 관통해야 체결(dry through — 가드 회계와 동일 가정)
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    assert ex.positions[0].status == "filled"
    # 익절가 도달 → 매도 **접수**(아직 청산 아님, C-1)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)
    assert c.sells and c.sells[0][0] == "294570"
    assert ex.positions[0].status == "selling" and ex.positions[0].exit_reason == "익절"
    assert ex.history == []
    # 매도 지정가를 관통해야 체결(dry through 대칭)
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10051)
    assert ex.history[-1].status == "exited" and ex.history[-1].exit_reason == "익절"


def test_stop_loss_sells():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, sl=9960))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)    # 체결(관통)
    ex.on_tick("294570", t0 + timedelta(seconds=3), 9960)    # 손절 매도 접수
    assert ex.positions[0].status == "selling" and c.sells
    ex.on_tick("294570", t0 + timedelta(seconds=4), 9961)    # 매도 체결
    assert ex.history[-1].exit_reason == "손절" and ex.history[-1].status == "exited"


def test_time_exit_sells():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1, exit_after=90))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)    # 체결(관통)
    ex.on_tick("294570", t0 + timedelta(seconds=95), 10010)  # 청산시각 지남 → 매도 접수
    assert ex.positions[0].status == "selling"
    ex.on_tick("294570", t0 + timedelta(seconds=96), 10011)  # 매도 체결
    assert ex.history[-1].exit_reason == "시간청산" and ex.history[-1].status == "exited"


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
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)    # 체결(관통)
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
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)    # 체결(관통)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 1차 매도 차단 → open 유지
    assert ex.positions[0].status == "filled"
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10050)   # 2차 매도 접수 성공
    assert ex.positions[0].status == "selling"
    ex.on_tick("294570", t0 + timedelta(seconds=7), 10051)   # 매도 체결
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
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)    # 체결(관통)
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
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)    # 체결(관통)
    ex.on_tick("033340", t0 + timedelta(seconds=2), 19999)   # 체결(관통)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)   # 거부 → open 유지
    ex.on_tick("033340", t0 + timedelta(seconds=5), 20100)   # 정상 매도 접수
    ex.on_tick("033340", t0 + timedelta(seconds=6), 20101)   # 매도 체결
    codes_open = {p.code for p in ex.positions}
    assert "294570" in codes_open                            # 거부분은 open 유지
    assert any(p.code == "033340" and p.status == "exited" for p in ex.history)


def test_dry_fill_basis_matches_risk_guard_through():
    """실행기와 리스크 가드가 같은 틱열에서 같은 청산 결론을 내야 한다.

    2026-08-25 033340 회귀: 지정가 539 알림 뒤 가격이 542(익절선 위)까지 올랐다가
    536(손절선 아래)으로 빠졌다. 실행기는 touch(≤entry) 로 즉시 체결로 보고 익절,
    가드는 through(<entry) 기준이라 상승분을 못 먹고 손절 — 같은 알림에 상반된 기록.
    dry-run 성과 집계는 가드 로그를 쓰므로, 실행기가 회계 기준보다 낙관적이면
    측정한 엣지와 실제 주문기 거동이 갈린다.
    """
    from scalp_it.realtime.risk_guard import RiskGuard, RiskGuardConfig

    t = datetime(2026, 8, 25, 9, 2, 0)
    alert = _alert(code="033340", ts=t, entry=539, tp=541.695, sl=536.844)
    ticks = [(t, 539.0), (t + timedelta(seconds=10), 542.0),
             (t + timedelta(seconds=24), 536.0)]

    c = FakeClient()
    ex = OrderExecutor(c, qty=1, live=False)
    ex.on_signal(alert)
    for ts_, px in ticks:
        ex.on_tick("033340", ts_, px)

    guard = RiskGuard(RiskGuardConfig(daily_loss_limit=0.0, max_consecutive_losses=0,
                                      max_positions=99, stop_file=None))
    guard.on_alert(alert)
    for ts_, px in ticks:
        guard.on_tick("033340", ts_, px)

    ex_reason = _last(ex).exit_reason
    guard_reason = (guard.closed_positions[-1].exit_reason if guard.closed_positions else "미청산")
    assert ex_reason == guard_reason, (
        f"실행기={ex_reason} vs 가드={guard_reason} — 체결 가정 불일치"
    )


# =================================================================== C-1: 매도 접수 ≠ 청산 완료

def test_sell_accepted_is_not_exited_until_fill():
    """C-1: 매도 접수(rc=0)만으로 exited 로 찍으면 실계좌에 1주가 남는다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)     # 매수 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)    # 익절 매도 접수
    assert len(ex.positions) == 1                             # positions 에 남아 계속 관리
    assert ex.positions[0].status == "selling"
    assert ex.history == []                                   # 아직 청산 아님
    # 매도가 안 붙는 동안 틱을 계속 먹어도 exited 로 넘어가지 않는다.
    ex.on_tick("294570", t0 + timedelta(seconds=8), 10049)
    assert ex.positions[0].status == "selling" and ex.history == []


def test_sell_fill_dry_touch_basis():
    """touch 기준이면 매도 지정가를 찍기만 해도 체결로 본다(매수와 대칭)."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, fill_basis="touch")
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)    # touch 매수 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)    # 익절 매도 접수
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10050)    # touch → 매도 체결
    assert ex.history[-1].status == "exited"


def test_live_sell_fill_uses_holdings_and_query_failure_is_not_a_fill():
    """live 매도 체결확인: 잔고에서 사라져야 체결. 조회 예외를 체결로 단정 금지."""

    class _LiveClient(FakeClient):
        def __init__(self):
            super().__init__(filled=True)
            self.rows = [{"code": "294570", "qty": 1}]
            self.raise_next = False

        def holdings(self, *, force=False):
            if self.raise_next:
                raise KiwoomError("잔고 조회 타임아웃")
            return list(self.rows)

    c = _LiveClient()
    ex = OrderExecutor(c, qty=1, live=True)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 10000)    # live 매수 체결(is_filled=True)
    assert ex.positions[0].status == "filled"
    ex.on_tick("294570", t0 + timedelta(seconds=3), 10050)    # 익절 매도 접수
    assert ex.positions[0].status == "selling"
    ex.on_tick("294570", t0 + timedelta(seconds=4), 10060)    # 아직 보유 → 미체결
    assert ex.positions[0].status == "selling"
    c.raise_next = True
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10060)    # 조회 예외 → 체결 단정 금지
    assert ex.positions[0].status == "selling"
    c.raise_next = False
    c.rows = []                                               # 잔고에서 사라짐 → 체결
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10060)
    assert ex.history[-1].status == "exited"


def test_unfilled_sell_chases_with_requote_and_alerts():
    """C-1: 미체결 매도는 타임아웃 뒤 취소 → 현재가 재호가(추격)하고 경보를 남긴다."""
    notes = []
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=10.0, notice=notes.append)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, sl=9960))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)     # 매수 체결
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9960)     # 손절 매도 @9960 접수
    assert len(c.sells) == 1
    ex.on_tick("294570", t0 + timedelta(seconds=13), 9940)    # 10초 경과·미체결 → 추격
    assert c.cancels and c.cancels[-1][0] == "S1"             # 기존 매도 취소
    assert len(c.sells) == 2 and c.sells[-1][2] == 9940       # 현재가로 재호가
    assert any("재호가" in n and "[주문][경보]" in n for n in notes)
    assert ex.positions[0].status == "selling"


def test_sell_chase_limit_keeps_position_monitored_with_big_alert():
    """추격 상한 도달 시 조용히 버리지 않고 큰 경보 + 계속 감시."""
    notes = []
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=10.0, max_sell_chases=2, notice=notes.append)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, sl=9960))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9960)     # 최초 매도
    for i in range(1, 6):                                     # 충분히 오래 미체결
        ex.on_tick("294570", t0 + timedelta(seconds=2 + 11 * i), 9950 - i)
    assert len(c.sells) == 1 + 2                              # 추격은 상한(2)까지만
    assert any("상한" in n for n in notes)
    assert len(ex.positions) == 1 and ex.positions[0].status == "selling"


def test_summary_counts_selling_as_active():
    """C-1: 매도 대기분은 '청산'이 아니라 '활성'으로 집계돼야 한다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)    # 매도 접수
    s = ex.summary()
    assert "활성 1" in s and "청산 0" in s
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10051)    # 체결
    s2 = ex.summary()
    assert "활성 0" in s2 and "청산 1" in s2


def test_dedup_holds_while_selling():
    """매도 대기 중인 종목에 재신호가 와도 새 매수를 내지 않는다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)    # selling
    ex.on_signal(_alert(ts=t0 + timedelta(seconds=6), entry=10000))
    assert len(c.buys) == 1


# =================================================================== C-2: 취소 거부 오판

def test_cancel_rejection_is_not_treated_as_success():
    """C-2: 브로커가 취소를 거부(rc!=0)하면 cancelled 로 확정하면 안 된다."""
    notes = []
    c = FakeClient(cancel_rc=3, filled=False)
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0, notice=notes.append)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000))
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10100)   # 미체결 취소 시도 → 거부
    assert c.cancels                                          # 취소는 시도했다
    assert len(ex.positions) == 1 and ex.positions[0].status == "pending"
    assert ex.history == []
    assert any("[주문][경보]" in n and "취소" in n for n in notes)


def test_cancel_rejection_then_fill_detected_next_tick():
    """C-2 복구 경로: 취소 거부(사실은 이미 체결) → 다음 틱에 체결로 넘어간다."""
    c = FakeClient(cancel_rc=3)
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10100)   # 취소 거부 → pending 유지
    assert ex.positions[0].status == "pending"
    ex.on_tick("294570", t0 + timedelta(seconds=25), 9999)    # 관통 → 체결 감지
    assert ex.positions[0].status == "filled"


def test_on_kill_cancel_rejection_keeps_pending():
    """킬스위치 취소도 거부되면 pending 유지(버리지 않는다)."""
    c = FakeClient(cancel_rc=3)
    ex = OrderExecutor(c, qty=1)
    ex.on_signal(_alert())
    ex.on_kill()
    assert ex.positions[0].status == "pending"


# =================================================================== C-3: 마감 flush

def test_flush_cancels_pending():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000))
    ex.flush(t0 + timedelta(seconds=60))
    assert c.cancels
    assert ex.positions == []
    assert ex.history[-1].status == "cancelled" and ex.history[-1].exit_reason == "마감미체결취소"


def test_flush_sells_filled_position_at_last_seen_price():
    """C-3: 마감 flush 가 체결분을 청산하지 않으면 오버나이트로 넘어간다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 15, 19, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1, exit_after=3600))
    ex.on_tick("294570", t0 + timedelta(seconds=10), 9999)    # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=30), 9990)    # 마지막으로 본 체결가
    ex.flush(t0 + timedelta(seconds=60))
    assert c.sells and c.sells[-1][2] == 9990                 # 마지막 가격으로 매도
    assert ex.positions[0].status == "selling"
    assert ex.positions[0].exit_reason == "마감청산"


def test_flush_filled_without_any_tick_alerts_loudly():
    """가격을 한 번도 못 봤으면 조용히 넘기지 말고 경보."""
    notes = []
    c = FakeClient(filled=True)
    ex = OrderExecutor(c, qty=1, live=True, notice=notes.append)
    t0 = datetime(2026, 8, 25, 15, 19, 0)
    ex.on_signal(_alert(ts=t0, entry=10000))
    ex.positions[0].status = "filled"                          # 틱 없이 체결만 된 상태
    ex.flush(t0 + timedelta(seconds=60))
    assert not c.sells
    assert any("[주문][경보]" in n for n in notes)
    assert len(ex.positions) == 1                              # 버리지 않는다


def test_flush_requotes_outstanding_sell():
    """C-3: 마감 시 미체결 매도는 마지막 가격으로 재호가한다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 15, 19, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=5), 9999)      # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=10), 10050)    # 익절 매도 @10050
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10030)    # 안 붙은 채 가격 하락
    ex.flush(t0 + timedelta(seconds=60))
    assert c.cancels and c.cancels[-1][0] == "S1"
    assert len(c.sells) == 2 and c.sells[-1][2] == 10030
    assert ex.positions[0].status == "selling"


def test_flush_leaves_selling_alone_when_price_unchanged():
    """이미 현재가에 걸려 있으면 재호가 없이 경보만 남긴다."""
    notes = []
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, notice=notes.append)
    t0 = datetime(2026, 8, 25, 15, 19, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=5), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=10), 10050)    # 매도 @10050, 마지막가 10050
    ex.flush(t0 + timedelta(seconds=60))
    assert len(c.sells) == 1 and not c.cancels
    assert any("[주문][경보]" in n for n in notes)


def test_flush_cancels_pending_even_before_deadline():
    """마감 flush 는 취소 시각 전이라도 대기 매수를 남기지 않는다(장 밖 주문 방지)."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 15, 19, 55)
    ex.on_signal(_alert(ts=t0, entry=10000))
    ex.flush(t0 + timedelta(seconds=5))                        # deadline(+15s) 전
    assert c.cancels and ex.positions == []
    assert ex.history[-1].status == "cancelled"


# ----------------------------------------- 대장 조건부 청산(만쥬 §5-3) — 실행기

class _BlockOneSell(FakeClient):
    """첫 매도만 거부하고 그 다음부터는 정상 접수하는 대역."""

    def __init__(self):
        super().__init__()
        self._blocked_once = False

    def sell_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
        self.sells.append((code, qty, price, live))
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        if not self._blocked_once:
            self._blocked_once = True
            return OrderResult(intent=intent, dry_run=not live, submitted=False,
                               blocked=True, blocked_reason="일시거부")
        return OrderResult(intent=intent, dry_run=not live, submitted=live, ord_no="S1",
                           return_code=(0 if live else None))


def test_leader_lost_sells_filled_position_regardless_of_pnl():
    """대장 이탈 → 손익 무관 즉시 매도 접수. 접수는 청산이 아니므로 selling."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)     # 매수 체결
    ex.on_tick("294570", t0 + timedelta(seconds=4), 10020)    # 익절·손절 아님
    ex.on_leader_state("124500", t0 + timedelta(seconds=5), LEADER_LOST)
    assert c.sells and c.sells[-1][:3] == ("294570", 1, 10020)
    pos = ex.positions[0]
    assert pos.status == "selling" and pos.exit_reason == "대장이탈"
    # 체결 확인 후에만 exited (C-1 회귀 방지)
    ex.on_tick("294570", t0 + timedelta(seconds=7), 10021)
    assert ex.history[-1].status == "exited"


def test_leader_lost_cancels_pending_buy():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0))
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    assert c.cancels and c.cancels[-1][0] == "B1"
    assert ex.positions == []
    assert ex.history[-1].status == "cancelled"
    assert ex.history[-1].exit_reason == "대장이탈취소"


def test_leader_lost_cancel_rejection_keeps_pending_and_retries():
    """C-2: 취소 거부를 성공으로 오판하지 않는다. 다음 틱에 재시도."""
    c = FakeClient(cancel_rc=-1)
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0))
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    assert ex.positions and ex.positions[0].status == "pending"
    ex.on_tick("294570", t0 + timedelta(seconds=4), 10001)    # 체결 안 되는 가격
    assert len(c.cancels) == 2                                 # 재시도했다


def test_leader_lost_only_affects_positions_of_that_leader():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(code="294570", ts=t0, leader="124500"))
    ex.on_signal(_alert(code="333333", ts=t0, leader="999999"))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("333333", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    assert [p.code for p in ex.positions if p.status == "selling"] == ["294570"]
    assert [p.code for p in ex.positions if p.status == "filled"] == ["333333"]


def test_leader_holding_state_is_noop():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_HOLDING)
    assert c.sells == [] and ex.positions[0].status == "filled"


def test_leader_lost_sell_failure_retries_on_next_tick():
    """매도 미성사면 포지션을 버리지 않고 다음 틱에 다시 청산 시도(N-1)."""
    c = _BlockOneSell()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    assert ex.positions[0].status == "filled"                  # 접수 실패 → 유지
    ex.on_tick("294570", t0 + timedelta(seconds=4), 10020)     # 어느 선도 아님
    assert ex.positions[0].status == "selling"
    assert ex.positions[0].exit_reason == "대장이탈"


def test_take_profit_wins_over_leader_lost_on_same_tick():
    """이미 익절선을 넘긴 틱이면 익절이 유리하다 — 익절 > 대장이탈."""
    c = _BlockOneSell()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)  # 접수 거부
    ex.on_tick("294570", t0 + timedelta(seconds=4), 10050)
    assert ex.positions[0].exit_reason == "익절"


def test_leader_lost_beats_stop_loss_and_time_exit():
    """손절·시간청산보다는 대장이탈이 앞선다(손익 무관 즉시)."""
    c = _BlockOneSell()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050, sl=9960, exit_after=10))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    ex.on_tick("294570", t0 + timedelta(seconds=20), 9950)     # 손절선·청산시각 동시 충족
    assert ex.positions[0].exit_reason == "대장이탈"


def test_leader_state_never_fed_keeps_legacy_time_exit():
    """하위호환 — 대장 상태를 안 먹이면 기존 시간청산 그대로."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, exit_after=10))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=20), 10000)
    assert _last(ex).exit_reason == "시간청산"


# ------------------------------------------------- 국면별 비중 조절(만쥬 §8-1)

def test_size_multiplier_scales_order_qty():
    c = FakeClient()
    ex = OrderExecutor(c, qty=10, size_multiplier=0.3)
    ex.on_signal(_alert())
    assert c.buys == [("294570", 3, 10000, False)]       # 10주 × 0.3 = 3주


def test_size_multiplier_keeps_at_least_one_share():
    """1주 × 0.3 = 0주가 되면 매매가 사라진다 — 진입하는 이상 최소 1주는 낸다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, size_multiplier=0.3)
    ex.on_signal(_alert())
    assert c.buys == [("294570", 1, 10000, False)]


def test_size_multiplier_zero_blocks_entry():
    c = FakeClient()
    ex = OrderExecutor(c, qty=5, size_multiplier=0.0)
    assert ex.on_signal(_alert()) is None
    assert c.buys == [] and ex.positions == []


def test_size_multiplier_zero_still_exits_open_position():
    """극침체로 배수가 0이 되어도 **들고 있는 포지션의 청산은 정상 동작**해야 한다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)          # 체결
    ex.set_size_multiplier(0.0)
    assert ex.on_signal(_alert(code="123456", ts=t0)) is None      # 신규만 차단
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10050)         # 익절 접수
    assert c.sells and c.sells[0][1] == 1
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10060)         # 체결확인
    assert _last(ex).status == "exited"


def test_size_multiplier_default_unchanged():
    c = FakeClient()
    ex = OrderExecutor(c, qty=2)
    ex.on_signal(_alert())
    assert c.buys == [("294570", 2, 10000, False)]


# ---------------------------------------------------- 분할 매도(만쥬 §5-2)

def test_partial_exit_sells_portion_then_remainder():
    c = FakeClient()
    ex = OrderExecutor(c, qty=4, partial_exit_ratio=0.5, partial_exit_gain=0.002)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)          # 체결 4주
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10025)         # +0.25% → 2주 분할 접수
    pos = ex.positions[0]
    assert pos.status == "selling" and c.sells[-1][1] == 2
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10030)         # 분할 체결
    assert pos.status == "filled" and pos.qty == 2 and pos.partial_done is True
    assert pos.partial_qty == 2 and ex.history == []
    ex.on_tick("294570", t0 + timedelta(seconds=8), 10050)         # 잔량 익절 접수
    assert c.sells[-1][1] == 2 and pos.exit_reason == "익절"
    ex.on_tick("294570", t0 + timedelta(seconds=9), 10060)
    assert _last(ex).status == "exited" and _last(ex).qty == 2


def test_partial_skipped_when_qty_too_small():
    """1주는 쪼갤 수 없다 — 분할 설정이 있어도 기존(전량) 경로 그대로."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, partial_exit_ratio=0.5, partial_exit_gain=0.002)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10025)
    assert c.sells == []                                            # 분할 없음
    assert ex.positions[0].status == "filled"


def test_partial_disabled_by_default():
    c = FakeClient()
    ex = OrderExecutor(c, qty=4)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10025)
    assert c.sells == []


def test_leader_lost_sells_full_remainder_not_partial():
    c = FakeClient()
    ex = OrderExecutor(c, qty=4, partial_exit_ratio=0.5, partial_exit_gain=0.002)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050, leader="124500"))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_leader_state("124500", t0 + timedelta(seconds=3), LEADER_LOST)
    assert c.sells[-1][1] == 4 and ex.positions[0].exit_reason == "대장이탈"


def test_partial_sell_chase_uses_partial_qty():
    """분할 매도가 안 붙어 추격 재호가할 때도 수량은 분할분이어야 한다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=4, unfilled_timeout=5.0,
                       partial_exit_ratio=0.5, partial_exit_gain=0.002)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10025)          # 분할 2주 접수 @10025
    ex.on_tick("294570", t0 + timedelta(seconds=12), 10020)         # 미체결·타임아웃 → 추격
    assert c.cancels[-1][2] == 2                                     # 취소 수량 = 분할분
    assert c.sells[-1][1] == 2 and c.sells[-1][2] == 10020


def test_live_partial_fill_uses_holdings_delta():
    """live 매도 체결확인은 잔고 '소멸'이 아니라 **잔여 수량 도달**로 판정해야 한다."""
    hold = [{"code": "294570", "qty": 4}]
    c = FakeClient(holdings=hold)
    ex = OrderExecutor(c, qty=4, live=True, partial_exit_ratio=0.5,
                       partial_exit_gain=0.002)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=10050))
    ex.on_tick("294570", t0 + timedelta(seconds=2), 9999)           # is_filled True → 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10025)          # 분할 2주 접수
    assert ex.positions[0].status == "selling"
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10030)          # 잔고 4 그대로 → 미체결
    assert ex.positions[0].status == "selling"
    hold[0]["qty"] = 2                                               # 2주만 체결됨
    ex.on_tick("294570", t0 + timedelta(seconds=7), 10030)
    assert ex.positions[0].status == "filled" and ex.positions[0].qty == 2


# ------------------------------------------------------------------ 불타기(만쥬 A-4)


def test_pyramid_disabled_by_default():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)      # 체결
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10500)     # +5% — arm 없으면 무시
    assert len(c.buys) == 1 and ex.positions[0].qty == 1


def test_pyramid_add_buy_then_fill_increases_qty():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)       # 체결(entry)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)      # +3% 도달 → 추가매수 접수
    pos = ex.positions[0]
    assert len(c.buys) == 2 and c.buys[1] == ("294570", 1, 10300, False)
    assert pos.add_pending_qty == 1 and pos.qty == 1            # 아직 체결 전
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10299)      # 관통 → 추가매수 체결
    assert pos.qty == 2 and pos.add_pending_qty == 0 and pos.pyramid_next == 1


def test_pyramid_two_stages():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03, 0.06), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)      # 1단계 발동
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10299)      # 1단계 체결 → qty 2
    pos = ex.positions[0]
    assert pos.qty == 2 and pos.pyramid_next == 1
    ex.on_tick("294570", t0 + timedelta(seconds=7), 10600)      # 2단계(+6%) 발동
    assert pos.add_pending_qty == 1
    ex.on_tick("294570", t0 + timedelta(seconds=8), 10599)      # 2단계 체결 → qty 3
    assert pos.qty == 3 and pos.pyramid_next == 2
    ex.on_tick("294570", t0 + timedelta(seconds=9), 20000)      # 더 올라도 단계 끝 — 추가 없음
    assert len(c.buys) == 3


def test_pyramid_no_double_buy_while_add_pending():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)      # 발동, 접수
    ex.on_tick("294570", t0 + timedelta(seconds=6), 10400)      # 아직 미체결(관통 안 됨) — 재발주 금지
    ex.on_tick("294570", t0 + timedelta(seconds=7), 10500)
    assert len(c.buys) == 2                                     # entry 1 + add 1, 재시도 없음


def test_pyramid_never_fires_on_loss():
    """물타기(손실 중 추가매수)는 arm 이 항상 양수라 구조적으로 안 걸린다."""
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=None))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 9500)       # −5% 손실
    assert len(c.buys) == 1                                      # 추가매수 없음


def test_pyramid_unfilled_cancel_advances_stage_no_retry_storm():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1,
                       unfilled_timeout=15.0)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)       # 발동, 접수 @10300
    # 15초 동안 관통(through) 없이 계속 그 이상에서만 거래 → 미체결
    ex.on_tick("294570", t0 + timedelta(seconds=21), 10310)      # 데드라인 경과 → 취소
    pos = ex.positions[0]
    assert pos.add_pending_qty == 0 and pos.pyramid_next == 1 and c.cancels
    ex.on_tick("294570", t0 + timedelta(seconds=22), 20000)      # 더 올라도 재시도 없음(단계 소진)
    assert len(c.buys) == 2


def test_pyramid_blocked_result_skips_stage():
    class BlockOnSecondBuy(FakeClient):
        def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            if len(self.buys) == 1:  # 두 번째 호출(추가매수) 이 함수 진입 전 buys 길이가 1
                self.buys.append((code, qty, price, live))
                intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
                return OrderResult(intent=intent, dry_run=not live, submitted=False,
                                   blocked=True, blocked_reason="가격밴드")
            return super().buy_limit(code, qty, price, live=live, ref_price=ref_price)

    c = BlockOnSecondBuy()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)       # 차단됨 — 단계 건너뜀
    pos = ex.positions[0]
    assert pos.add_pending_qty == 0 and pos.pyramid_next == 1 and pos.qty == 1
    ex.on_tick("294570", t0 + timedelta(seconds=6), 20000)       # 재시도 없음
    assert len(c.buys) == 2


def test_pyramid_exception_skips_stage():
    class RaiseOnSecondBuy(FakeClient):
        def buy_limit(self, code, qty, price, *, live=False, ref_price=None, exchange="KRX"):
            if len(self.buys) == 1:
                raise KiwoomError("일시 오류")
            return super().buy_limit(code, qty, price, live=live, ref_price=ref_price)

    c = RaiseOnSecondBuy()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)       # 예외 — 단계 건너뜀, 크래시 없음
    pos = ex.positions[0]
    assert pos.add_pending_qty == 0 and pos.pyramid_next == 1


def test_flush_cancels_pending_pyramid_add_before_close_sell():
    c = FakeClient()
    ex = OrderExecutor(c, qty=1, pyramid_arms=(0.03,), pyramid_add_qty=1)
    t0 = datetime(2026, 8, 25, 9, 30, 0)
    ex.on_signal(_alert(ts=t0, entry=10000, tp=99999, sl=1))
    ex.on_tick("294570", t0 + timedelta(seconds=1), 9999)
    ex.on_tick("294570", t0 + timedelta(seconds=5), 10300)       # 추가매수 접수, 미체결
    pos = ex.positions[0]
    assert pos.add_pending_qty == 1
    ex.flush(t0 + timedelta(seconds=6))
    assert pos.add_pending_qty == 0                              # 마감 전 취소됨
    assert any(cancel[0] == "B1" for cancel in c.cancels)        # 추가매수 주문 취소 접수
    assert _last(ex).status in ("selling", "exited")             # 원 포지션은 정상 마감청산
