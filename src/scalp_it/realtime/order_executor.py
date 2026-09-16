"""짝꿍 신호 → 실계좌 주문 실행기 (dry-run 기본, 이중 게이트).

감지기 알림이 risk_guard 를 통과한 **뒤에만** 이리로 온다(동일 게이트). 여기서
:class:`~scalp_it.adapters.kiwoom_order.KiwoomOrderClient` 로 1주 지정가 매수를
내고(기본 dry-run), 틱으로 체결·청산(익절/손절/재난선/시간청산)을 관리한다.

안전 원칙
========

* **매수 전용 진입.** 매도는 오직 **이 실행기가 낸 매수 포지션의 청산**으로만 나간다.
  무보유 종목 매도 경로는 없다.
* **종목 dedup.** 같은 후속주가 여러 대장 쌍에 걸려 한 틱에 두 번 알림와도, 이미
  활성(대기/체결) 포지션이 있으면 **두 번째 매수를 건너뛴다** — 중복발주 방지.
* **미체결 취소.** 매수가 ``unfilled_timeout`` 초 안에 안 붙으면 취소한다.
* **접수 ≠ 체결.** 매도 접수(rc=0)는 청산이 아니다. 체결이 확인돼야 exited 로 찍는다
  (C-1). 미체결 매도는 타임아웃마다 취소 후 현재가로 재호가(추격)하고, 추격 상한에
  닿으면 큰 경보를 남기되 포지션은 계속 감시한다 — 무감시가 최악이다.
* **취소는 반환값을 본다.** 브로커가 취소를 거부(rc≠0)해도 어댑터는 예외를 안 던지므로
  ``OrderResult`` 를 직접 검사한다(C-2). 거부면 포지션을 버리지 않고 유지·재시도한다.
* **마감 청산.** :meth:`flush` 는 대기 매수뿐 아니라 **체결·매도대기분도 정리**한다
  (C-3). 오버나이트로 넘어가면 다음 날 인수할 코드가 없다.
* **국면별 비중(만쥬 §8-1).** 주문 수량에 국면 배수를 곱한다. 배수는 **주입**받는다
  (이 모듈은 DB·시계를 모른다). 진입하는 이상 최소 1주는 유지하고, 배수 0(극침체)이면
  **신규 진입만** 막는다 — 보유분 청산은 그대로 돈다.
* **분할 매도(만쥬 §5-2).** 익절·시간 계열에 한해 1차로 일부만 판다. 대장이탈·손절·
  재난선은 §5-3 "손익 무관 전량"이라 쪼개지 않는다. 기본값은 꺼짐(기존 전량 동작).
* **대장 조건부 청산.** 포지션마다 그 포지션을 만든 대장 코드를 달고, 감지기가
  :meth:`on_leader_state` 로 대장의 상한가 이탈을 통보하면 그 대장으로 연 포지션을
  손익 무관 즉시 매도 접수한다(만쥬 §5-3). 대기 매수는 취소한다.
* **킬스위치 연동.** 가드 킬 발동 시 :meth:`on_kill` 로 대기 매수를 전부 취소.
* **live 는 클라이언트가 판단.** 실행기는 ``live`` 플래그를 클라이언트 ``submit`` 에
  넘길 뿐, 실제 POST 여부는 어댑터의 dry-run/live 이중 안전에 달렸다.

이 모듈은 시계를 모르고 틱을 주입받아 순수하게 상태를 옮기므로 mock 클라이언트·
합성 틱으로 단위 테스트한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from krx_quant_core.backtest.fills import limit_buy_filled, limit_sell_filled

from .risk_guard import LEADER_HOLDING, LEADER_LOST

if TYPE_CHECKING:
    from krx_quant_core.execution.book import PositionBook

    from ..adapters.kiwoom_order import KiwoomOrderClient, OrderResult
    from .pair_detector import Alert

__all__ = ["LEADER_HOLDING", "LEADER_LOST", "LivePosition", "OrderExecutor"]

_EPS = 1e-9


@dataclass
class LivePosition:
    """실행기가 관리하는 실주문 포지션 하나(1주 지정가 매수 → 청산)."""

    code: str
    name: str
    qty: int                            # **현재 보유 수량**(분할 매도가 붙으면 줄어든다)
    entry: int                          # 매수 지정가(bid1)
    take_profit: float | None           # 익절가(원). None = 상한 없음
    stop_loss: float | None
    disaster_line: float | None
    alert_ts: datetime
    exit_ts: datetime                   # 시간청산 시각
    cancel_deadline: datetime           # 미체결 취소 시각
    leader: str = ""                    # 이 포지션을 만든 대장 코드(대장 조건부 청산용)
    leader_lost: bool = False           # 대장이 상한가를 놓쳤다고 통보받았는가
    buy_ord_no: str | None = None
    # selling = 청산 매도 접수됨·체결확인 대기(C-1). exited 는 체결 확인 후에만.
    status: str = "pending"             # pending | filled | selling | exited | cancelled | blocked
    fill_ts: datetime | None = None
    fill_price: int | None = None
    exit_reason: str = ""
    sell_ord_no: str | None = None
    sell_price: int | None = None       # 접수한 청산 지정가
    sell_qty: int = 0                   # 접수한 청산 수량(분할이면 일부, 아니면 전량)
    sell_deadline: datetime | None = None   # 이 시각까지 안 붙으면 취소 후 재호가
    sell_chases: int = 0                # 재호가(추격) 횟수
    chase_alerted: bool = False         # 추격 상한 경보를 이미 냈는가(중복 경보 방지)
    last_price: float | None = None     # 마지막으로 본 체결가 — flush 의 청산가로 쓴다
    qty_initial: int = 0                # 진입 수량(분할 회계용). 0 이면 qty 와 같다
    partial_done: bool = False          # 1차 분할이 이미 체결됐는가
    partial_qty: int = 0                # 분할로 이미 판 누적 수량
    partial_price: int | None = None    # 분할 체결가(접수가)
    partial_reason: str = ""            # 분할익절 | 분할시간청산
    # 불타기(만쥬 A-4) — entry 대비 확정 수익 arm 마다 같은 크기로 추가매수.
    # 물타기(손실 중 추가매수)는 arm 이 항상 양수라 구조적으로 안 걸린다.
    pyramid_next: int = 0               # 다음에 시도할 단계 인덱스(arms 리스트 기준)
    add_pending_qty: int = 0            # 접수한 추가매수 수량(0 = 대기 없음)
    add_ord_no: str | None = None
    add_price: int | None = None        # 접수한 추가매수 지정가
    add_deadline: datetime | None = None    # 이 시각까지 안 붙으면 취소


class OrderExecutor:
    """알림 → 매수, 틱 → 체결·청산. dry-run 기본."""

    def __init__(
        self,
        client: KiwoomOrderClient,
        *,
        qty: int = 1,
        live: bool = False,
        unfilled_timeout: float = 15.0,
        max_sell_chases: int = 3,
        fill_basis: str = "through",
        leader_exit: bool = True,
        size_multiplier: float = 1.0,
        partial_exit_ratio: float = 0.0,
        partial_exit_gain: float | None = None,
        partial_exit_after_s: float | None = None,
        pyramid_arms: tuple[float, ...] = (),
        pyramid_add_qty: int = 1,
        notice: Callable[[str], None] | None = None,
        book: PositionBook | None = None,
    ) -> None:
        self._client = client
        self._qty = max(1, int(qty))
        self._live = bool(live)
        self._unfilled_timeout = float(unfilled_timeout)
        # 청산 매도 재호가 상한 — 무한 추격(가격을 쫓아 계속 내리는 것)을 막는다.
        self._max_sell_chases = max(0, int(max_sell_chases))
        self._fill_basis = fill_basis if fill_basis in ("through", "touch") else "through"
        # 대장 조건부 청산(만쥬 §5-3). 켜 두어도 대장 상태를 안 먹이면 아무 일도 없다.
        self._leader_exit = bool(leader_exit)
        # 국면별 비중(만쥬 §8-1). config 가 아니라 속성으로 두는 건 장중에 국면이
        # 바뀌면 CLI 가 :meth:`set_size_multiplier` 로 갈아끼우기 때문이다.
        self.size_multiplier = max(0.0, float(size_multiplier))
        # 분할 매도(만쥬 §5-2). ratio 가 0 이면 완전히 꺼진 상태 = 기존(전량) 동작.
        self._partial_ratio = float(partial_exit_ratio)
        self._partial_gain = partial_exit_gain
        self._partial_after_s = partial_exit_after_s
        # 불타기(만쥬 A-4). arms 는 entry 대비 누적 수익률 오름차순(예: (0.03, 0.06)).
        # 빈 튜플 = 끔(기존 동작). 각 단계마다 pyramid_add_qty 주씩 추가매수.
        self._pyramid_arms = tuple(float(a) for a in pyramid_arms)
        self._pyramid_add_qty = max(1, int(pyramid_add_qty))
        self._notice = notice or (lambda _s: None)
        # 체결 장부(krx-quant-core PositionBook) — **기록 전용**. 판단에 쓰지 않고, 실패해도
        # 경고만 남긴다(:meth:`_journal`). None 이면 아무것도 안 한다(기존 동작 그대로).
        self._book = book
        self.positions: list[LivePosition] = []
        self.history: list[LivePosition] = []

    # ----- 국면별 비중(만쥬 §8-1) -------------------------------------------

    def set_size_multiplier(self, mult: float) -> None:
        """국면 배수를 갱신한다. 다음 진입부터 적용되고, 보유분은 건드리지 않는다."""
        self.size_multiplier = max(0.0, float(mult))

    def effective_qty(self) -> int:
        """이번 진입에 낼 수량. 0 이면 **진입 금지**(만쥬 §8-1 극침체 = 매매를 쉰다).

        1주 × 0.3 = 0주가 되면 배수가 곧 '매매 중단'이 되어 버린다. 만쥬는 침체장에
        "몸을 사리고 관망"(비중 축소)이라 했지 "매매 금지"라 하지 않았고, 금지는
        극침체(배수 0)에만 붙였다. 그래서 **진입하는 이상 최소 1주**를 유지하고,
        배수가 정확히 0일 때만 진입 자체를 막는다. 청산 경로는 이 함수를 안 지나므로
        배수가 0이어도 들고 있는 건 정상적으로 팔린다.
        """
        if self.size_multiplier <= _EPS:
            return 0
        return max(1, round(self._qty * self.size_multiplier))

    # ----- 조회 --------------------------------------------------------------

    def _active_for(self, code: str) -> LivePosition | None:
        for p in self.positions:
            if p.code == code and p.status in ("pending", "filled", "selling"):
                return p
        return None

    # ----- 진입 --------------------------------------------------------------

    def on_signal(self, alert: Alert) -> OrderResult | None:
        """가드 통과 알림 하나 → 1주 지정가 매수. dedup·차단 시 None 반환 가능."""
        code = alert.follower
        # dedup — 이미 활성 포지션이 있으면 두 번째 매수를 건너뛴다.
        if self._active_for(code) is not None:
            self._notice(f"[주문] dedup — {alert.follower_name or code} 이미 활성 포지션, 매수 생략")
            return None
        qty = self.effective_qty()
        if qty <= 0:
            self._notice(
                f"[주문] 국면차단(배수 0) — {alert.follower_name or code} 신규 진입 없음"
            )
            return None
        entry = round(alert.entry)
        try:
            result = self._client.buy_limit(
                code, qty, entry, live=self._live, ref_price=alert.price
            )
        except Exception as exc:  # noqa: BLE001 — N-2: 매수 거부로 프로세스 크래시 금지
            self._notice(
                f"[주문][경보] 매수 실패(예외) — {alert.follower_name or code} "
                f"@ {entry:,}: {exc} · 포지션 열지 않고 스킵"
            )
            return None
        pos = LivePosition(
            code=code,
            name=alert.follower_name or code,
            qty=qty,
            qty_initial=qty,
            entry=entry,
            take_profit=alert.take_profit,
            stop_loss=alert.stop_loss,
            disaster_line=alert.disaster_line,
            alert_ts=alert.ts,
            exit_ts=alert.exit_ts,
            cancel_deadline=alert.ts + timedelta(seconds=self._unfilled_timeout),
            leader=alert.leader,
            buy_ord_no=result.ord_no,
        )
        if result.blocked:
            pos.status = "blocked"
            pos.exit_reason = result.blocked_reason
            self.history.append(pos)
            return result
        self.positions.append(pos)
        return result

    # ----- 틱: 체결·청산 -----------------------------------------------------

    def on_tick(self, code: str, ts: datetime, price: float) -> None:
        """후속주 틱으로 대기 매수의 체결·미체결취소·청산을 진행한다."""
        still: list[LivePosition] = []
        for pos in self.positions:
            if pos.code != code:
                still.append(pos)
                continue
            # C-3: flush 에는 가격 인자가 없다. 마지막으로 본 체결가를 여기서 남겨
            # 마감 청산가로 쓴다.
            pos.last_price = float(price)
            done = self._advance(pos, ts, float(price))
            if done:
                self.history.append(pos)
            else:
                still.append(pos)
        self.positions = still

    def _advance(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        if ts < pos.alert_ts:
            return False
        if pos.status == "pending":
            filled = self._check_fill(pos, ts, price)
            if not filled:
                # 미체결 취소 시각 도달 또는 대장 이탈(살 이유 상실) → 취소.
                if ts >= pos.cancel_deadline or pos.leader_lost:
                    why = "대장이탈취소" if pos.leader_lost else "미체결취소"
                    if not self._try_cancel(pos, why):
                        return False   # 취소 실패 → pending 유지, 다음 틱 재시도
                    pos.status = "cancelled"
                    pos.exit_reason = why
                    self._notice(f"[주문] {why} — {pos.name}({pos.code})")
                    return True
                return False
            # 방금 체결됨 → 아래 청산 판정으로 진행
        if pos.status == "selling":
            return self._advance_selling(pos, ts, price)
        if pos.status != "filled":
            return False
        # 청산: 익절 → 대장이탈 → 손절 → 재난선 → 시간청산
        # 대장이탈이 손절·재난선·시간청산보다 앞인 이유는 만쥬 §5-3 "손익 무관 즉시"다.
        # 익절만 그 앞에 두는데, 익절은 미리 걸어둔 지정가라 그 레벨을 찍은 틱이면
        # 대장 통보보다 먼저 붙었다고 보는 게 실제에 가깝기 때문이다.
        # 여기 대장이탈 분기는 :meth:`on_leader_state` 의 즉시 매도가 **미성사**했을 때
        # 다음 틱에 다시 타는 경로다(무감시 방지).
        reason: str | None = None
        if pos.take_profit is not None and price >= pos.take_profit - _EPS:
            reason = "익절"
        elif pos.leader_lost:
            reason = "대장이탈"
        elif pos.stop_loss is not None and price <= pos.stop_loss + _EPS:
            reason = "손절"
        elif pos.disaster_line is not None and price <= pos.disaster_line + _EPS:
            reason = "재난선"
        elif ts >= pos.exit_ts:
            reason = "시간청산"
        if reason is None:
            # 전량 청산 사유가 하나도 안 걸린 뒤에만 불타기·1차 분할을 본다.
            self._maybe_pyramid(pos, ts, price)
            self._maybe_partial(pos, ts, price)
            return False
        # N-1/N-2/C-1: 매도는 **접수**까지만. 접수 성공이면 selling 으로 두고 계속 틱을
        # 먹여 체결을 확인한다. 차단·거부·예외면 filled 로 남아 다음 틱에 재시도.
        # 어느 쪽이든 done 이 아니다 — history 이동은 체결 확인 뒤에만.
        self._sell(pos, ts, round(price), reason)
        return False

    def _advance_selling(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        """매도 접수 후: 체결 확인 → exited. 안 붙으면 타임아웃마다 추격 재호가.

        C-1: 접수만 확인하고 exited 로 찍으면 실계좌엔 1주가 남는데 시스템은 청산으로
        집계한다. 하락 중 손절 지정가는 정확히 안 붙는 구간이라 실제로 자주 일어난다.
        """
        if self._check_sell_fill(pos, price):
            if 0 < pos.sell_qty < pos.qty:
                self._journal("sell", pos.code, pos.sell_qty, pos.sell_price, ts, pos.sell_ord_no)
                # 분할 체결 — 포지션은 **끝나지 않았다**. 잔량을 그대로 감시로 되돌린다.
                # 잔량의 청산규칙은 **다음 틱**부터 다시 돈다(이 틱은 여기서 끝난다).
                # 한 틱 지연을 감수하는 이유: 같은 틱에서 곧바로 전량 매도를 이어 내면
                # 방금 취소·재호가 경로와 뒤엉켜 이중 매도 위험이 생긴다.
                self._settle_partial(pos)
                return False
            pos.status = "exited"
            self._journal("sell", pos.code, pos.qty, pos.sell_price, ts, pos.sell_ord_no)
            self._notice(
                f"[주문] 청산체결({pos.exit_reason}) — {pos.name}({pos.code}) "
                f"{pos.qty}주 @ {(pos.sell_price or 0):,}"
            )
            return True
        if pos.sell_deadline is None or ts < pos.sell_deadline:
            return False
        if pos.sell_chases >= self._max_sell_chases:
            # 상한 도달 — 조용히 버리는 게 최악이다. 큰 경보 후에도 감시는 계속한다.
            if not pos.chase_alerted:
                pos.chase_alerted = True
                self._notice(
                    f"[주문][경보][중대] 청산 재호가 상한({self._max_sell_chases}) 도달 — "
                    f"{pos.name}({pos.code}) {pos.qty}주 @ {(pos.sell_price or 0):,} 미체결 "
                    f"({pos.exit_reason}) · 수동 확인 필요. 포지션은 계속 감시한다."
                )
            return False
        # 추격: 기존 매도 취소 → 현재가로 재호가. 수량은 **접수했던 그 수량** 그대로다
        # (분할이면 분할분). 여기서 전량으로 바꾸면 분할 의도가 조용히 뒤집힌다.
        chase_qty = pos.sell_qty or pos.qty
        if not self._try_cancel(pos, "청산재호가", ord_no=pos.sell_ord_no, qty=chase_qty):
            # 취소 실패 — 이중 매도가 나가면 무보유 매도가 되므로 재발주하지 않는다.
            # 다음 타임아웃에 재시도.
            pos.sell_deadline = ts + timedelta(seconds=self._unfilled_timeout)
            return False
        pos.sell_chases += 1
        new_price = round(price)
        self._notice(
            f"[주문][경보] 청산 재호가 {pos.sell_chases}/{self._max_sell_chases} — "
            f"{pos.name}({pos.code}) {(pos.sell_price or 0):,} → {new_price:,} ({pos.exit_reason})"
        )
        # 취소가 성사된 순간 시장에 우리 매도는 없다 → filled 로 되돌려 놓고 재발주.
        # 재발주가 실패해도 filled 로 남아 다음 틱에 청산규칙이 다시 돈다(무감시 방지).
        pos.status = "filled"
        pos.sell_ord_no = None
        self._sell(pos, ts, new_price, pos.exit_reason, qty=chase_qty)
        return False

    # ----- 분할 매도(만쥬 §5-2) ---------------------------------------------

    def _settle_partial(self, pos: LivePosition) -> None:
        """분할 매도 체결 정산 — 보유 수량을 줄이고 감시 상태(filled)로 되돌린다."""
        sold = pos.sell_qty
        pos.qty -= sold
        pos.partial_done = True
        pos.partial_qty += sold
        pos.partial_price = pos.sell_price
        pos.partial_reason = pos.exit_reason
        self._notice(
            f"[주문] 분할체결({pos.exit_reason}) — {pos.name}({pos.code}) "
            f"{sold}주 @ {(pos.sell_price or 0):,} · 잔량 {pos.qty}주 계속 감시"
        )
        # 매도 상태를 완전히 비운다. 안 비우면 잔량 청산 때 옛 주문번호로 취소를 걸거나
        # 옛 접수 수량으로 체결확인을 해 수량이 어긋난다.
        pos.status = "filled"
        pos.exit_reason = ""
        pos.sell_ord_no = None
        pos.sell_price = None
        pos.sell_qty = 0
        pos.sell_deadline = None
        pos.sell_chases = 0
        pos.chase_alerted = False

    def _maybe_pyramid(self, pos: LivePosition, ts: datetime, price: float) -> None:
        """불타기(만쥬 A-4) — entry 대비 확정 수익 arm 마다 같은 크기로 추가매수.

        > "1차 진입 후 방향성이 맞고 3~4% 이상 확정 수익이 붙었을 때만 비중을 2배
        >  확대." (49번 §3-2) / "물타기(평단 하향 추가매수)는 절대 금지."

        **물타기가 구조적으로 안 걸리는 이유**: ``_pyramid_arms`` 는 전부 entry 대비
        수익률이라 항상 양수고, 이 메서드는 :meth:`_advance`가 익절·대장이탈·손절·
        재난선·시간청산 중 **어느 것도 안 걸린** 틱에서만 부른다 — 손실 구간에선
        에초에 호출되지 않거나, 호출돼도 ``ret`` 이 양수 arm 을 못 넘어 아무 일도
        없다.

        한 번에 한 단계씩만 접수한다(``add_pending_qty`` 게이트) — 매수 접수 자체는
        빠르지만 체결 확인 전에 다음 단계를 또 접수하면 이중 발주가 될 수 있다.
        체결 확인 전 미체결취소 시각이 지나면 취소를 시도하고, 취소가 **성사되면**
        그 단계는 건너뛴다(``pyramid_next`` 전진) — 실패한 단계를 매 틱 재시도하면
        가격이 계속 유리한 동안 취소·재시도가 무한 반복될 수 있다. 취소 자체가
        실패(브로커 거부)하면 포지션을 그대로 두고 다음 틱 재시도한다(N-2 원칙).
        """
        if not self._pyramid_arms:
            return
        if pos.add_pending_qty > 0:
            if self._is_buy_filled(pos.add_ord_no, float(pos.add_price or 0), pos.code, price):
                self._journal("buy", pos.code, pos.add_pending_qty, pos.add_price, ts, pos.add_ord_no)
                pos.qty += pos.add_pending_qty
                pos.pyramid_next += 1
                self._notice(
                    f"[주문] 불타기 체결 — {pos.name}({pos.code}) +{pos.add_pending_qty}주 "
                    f"@ {pos.add_price:,} (누적 {pos.qty}주)"
                )
                pos.add_pending_qty = 0
                pos.add_ord_no = None
                pos.add_price = None
                pos.add_deadline = None
                return
            if pos.add_deadline is not None and ts >= pos.add_deadline:
                if self._try_cancel(
                    pos, "불타기미체결취소", ord_no=pos.add_ord_no, qty=pos.add_pending_qty
                ):
                    self._notice(
                        f"[주문] 불타기미체결취소 — {pos.name}({pos.code}) {pos.add_pending_qty}주"
                    )
                    pos.add_pending_qty = 0
                    pos.add_ord_no = None
                    pos.add_price = None
                    pos.add_deadline = None
                    pos.pyramid_next += 1  # 이 단계는 건너뛴다 — 재시도 스톰 방지
                # 취소 실패면 다음 틱 재시도(포지션 유지, add_pending_qty 그대로)
            return
        if pos.pyramid_next >= len(self._pyramid_arms) or pos.entry <= 0:
            return
        ret = price / pos.entry - 1.0
        if ret < self._pyramid_arms[pos.pyramid_next]:
            return
        add_qty = self._pyramid_add_qty
        add_price = round(price)
        try:
            result = self._client.buy_limit(pos.code, add_qty, add_price, live=self._live, ref_price=price)
        except Exception as exc:  # noqa: BLE001 — N-2 와 동일 원칙: 매수 실패로 크래시 금지
            self._notice(
                f"[주문][경보] 불타기 실패(예외) — {pos.name}({pos.code}) @ {add_price:,}: "
                f"{exc} · 이 단계 건너뜀"
            )
            pos.pyramid_next += 1
            return
        if result.blocked:
            self._notice(
                f"[주문] 불타기 차단 — {pos.name}({pos.code}): {result.blocked_reason} · 이 단계 건너뜀"
            )
            pos.pyramid_next += 1
            return
        pos.add_pending_qty = add_qty
        pos.add_ord_no = result.ord_no
        pos.add_price = add_price
        pos.add_deadline = ts + timedelta(seconds=self._unfilled_timeout)
        self._notice(
            f"[주문] 불타기 발동 — {pos.name}({pos.code}) +{add_qty}주 @ {add_price:,} "
            f"(수익률 {ret*100:.1f}%, {pos.pyramid_next + 1}/{len(self._pyramid_arms)}단계)"
        )

    def _maybe_partial(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        """1차 분할 매도를 접수한다(만쥬 §5-2 "분할 또는 전량 매도"). 접수했으면 True.

        **분할 대상은 익절·시간청산 계열뿐이다.** 대장이탈·손절·재난선은 §5-3 이
        "손익 무관 **전량**"이라 쪼개지 않는다 — 매수 근거가 사라졌는데 절반을 들고
        있는 건 그 규칙의 정반대다. 그래서 방아쇠는 (a) 수익률 도달, (b) 체결 후
        경과시간 둘뿐이고, 전량 사유가 걸린 틱에서는 호출되지 않는다.

        1주는 쪼갤 수 없다. ``int(qty*ratio)`` 가 1 미만이거나 전량이면 분할을 **하지
        않고** 기존 전량 경로에 맡긴다 — 억지로 반올림해 전량을 내보내면 설정이
        의도치 않게 조기 전량청산으로 둔갑한다.
        """
        if pos.partial_done or not (_EPS < self._partial_ratio < 1.0 - _EPS):
            return False
        pqty = int(pos.qty * self._partial_ratio)
        if pqty < 1 or pqty >= pos.qty:
            return False
        reason = ""
        if (self._partial_gain is not None
                and price >= pos.entry * (1.0 + self._partial_gain) - _EPS):
            reason = "분할익절"
        elif (self._partial_after_s is not None and pos.fill_ts is not None
                and (ts - pos.fill_ts).total_seconds() >= self._partial_after_s):
            reason = "분할시간청산"
        if not reason:
            return False
        return self._sell(pos, ts, round(price), reason, qty=pqty)

    def _is_buy_filled(self, ord_no: str | None, ref_price: float, code: str, price: float) -> bool:
        """매수 지정가(``ref_price``)가 체결됐는지. live 는 미체결조회, dry 는 틱 관통 근사.

        :meth:`_check_fill`(진입)과 불타기 추가매수 양쪽이 같은 판정을 쓰도록 뽑아냈다
        — 판정 기준이 갈리면 같은 틱열에서 회계와 실행기가 서로 다른 체결을 기록한다
        (2026-08-25 033340 사고와 같은 종류).
        """
        if self._live and ord_no:
            try:
                return self._client.is_filled(ord_no, code)
            except Exception:  # noqa: BLE001 — 조회 실패 시 체결로 단정하지 않는다
                return False
        # dry-run 근사 — 회계 기준(RiskGuard.fill_basis)과 **같은 가정**을 써야 한다.
        # through(보수·기본): 지정가를 관통(미만)해야 큐가 비워져 내 주문이 채워진다.
        # touch(낙관): 지정가를 찍기만 하면 체결. 큐 맨 앞이라는 가정.
        # 판정식은 krx-quant-core backtest.fills 에 있다(회계 RiskGuard 와 같은 함수).
        # _fill_basis 는 __init__ 에서 through/touch 로 정규화돼 코어의 ValueError 에 닿지 않는다.
        return limit_buy_filled(ref_price, price, self._fill_basis)

    def _check_fill(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        """대기 매수가 체결됐는지."""
        filled = self._is_buy_filled(pos.buy_ord_no, pos.entry, pos.code, price)
        if filled:
            pos.status = "filled"
            pos.fill_ts = ts
            pos.fill_price = pos.entry
            self._journal("buy", pos.code, pos.qty, pos.entry, ts, pos.buy_ord_no)
            self._notice(f"[주문] 체결 — {pos.name}({pos.code}) {pos.qty}주 @ {pos.entry:,}")
        return filled

    def _journal(self, side: str, code: str, qty: int, price: float | None, ts: datetime,
                 ord_no: str | None) -> None:
        """체결 1건을 장부에 남긴다. **어떤 실패도 실행기로 새지 않는다**(경고만).

        장부는 사후 대사·손익 기록용이다. 여기서 예외가 나가면 체결 확인 뒤 상태 전이가
        끊겨 포지션이 무감시로 남을 수 있으므로, 잡아서 알림만 남기고 계속 간다.

        ``price`` 는 체결 확인 시점에 실행기가 아는 값 — 매수·불타기는 **지정가**, 매도는
        접수가다. 실제 체결가(부분 체결·가격 개선)와 다를 수 있다.
        """
        if self._book is None:
            return
        try:
            # 지연 import — 코어 0.5.0 전용 모듈이라, 최상위에 두면 핀이 늦은 날 실시간
            # 프로세스 import 자체가 죽는다. 장부를 켠 경우에만 필요하다.
            from krx_quant_core.execution.events import Fill

            self._book.apply(Fill(ord_no=str(ord_no or ""), code=code, side=side,  # type: ignore[arg-type]
                                  qty=int(qty), price=int(price or 0), ts=ts))
        except Exception as exc:  # noqa: BLE001 — 기록 실패가 매매를 멈추면 안 된다
            self._notice(f"[주문][경고] 장부 기록 실패 — {code} {side} {qty}주 @ {price}: {exc}")

    def _check_sell_fill(self, pos: LivePosition, price: float) -> bool:
        """청산 매도가 체결됐는지. live 는 잔고 소멸, dry 는 매수와 대칭인 근사.

        live 에서 :meth:`KiwoomOrderClient.is_filled` 를 그대로 쓰면 **뜻이 뒤집힌다** —
        그 메서드는 '잔고에 있으면 True(매수 체결)' 라, 매도에 쓰면 아직 보유 중인
        포지션을 청산 완료로 찍는다(= C-1 그 자체). 그래서 매도는 잔고 수량이
        **접수 후 잔여 수량까지 줄었는지**로 판정한다. 전량 매도면 잔여가 0이라
        기존의 '잔고에서 사라짐'과 완전히 같고, 분할 매도면 남아야 할 만큼만 남았는지를
        본다 — 여기서 '0인가'만 보면 분할 체결이 영영 미체결로 읽혀 무한 추격이 된다.
        조회 예외는 매수 경로와 같은 원칙으로 '체결 아님'(fail-safe).
        """
        if pos.sell_price is None:
            return False
        if self._live and pos.sell_ord_no:
            holdings = getattr(self._client, "holdings", None)
            if holdings is None:
                self._notice(
                    f"[주문][경보] 청산 체결확인 불가 — {pos.name}({pos.code}) "
                    f"잔고 조회 경로가 없다 · 수동 확인 필요"
                )
                return False
            try:
                rows = holdings()
            except Exception:  # noqa: BLE001 — 조회 실패를 체결로 단정하지 않는다
                return False
            expect_left = max(0, pos.qty - (pos.sell_qty or pos.qty))
            for h in rows:
                try:
                    if h["code"] == pos.code and int(h.get("qty") or 0) > expect_left:
                        return False      # 아직 접수분이 안 빠짐 → 미체결
                except (KeyError, TypeError, ValueError):
                    return False          # 응답이 이상하면 체결 단정 안 함
            return True                   # 잔여 수량까지 줄었다 → 매도 체결
        # dry-run 근사 — 매수와 대칭. 매도 지정가는 가격이 그 **위로** 가야 붙는다.
        return limit_sell_filled(pos.sell_price, price, self._fill_basis)

    def _sell(self, pos: LivePosition, ts: datetime, price: int, reason: str,
              *, qty: int | None = None) -> bool:
        """청산 매도를 **접수**한다. 접수 성공 시 True(selling), 미성사 시 False(filled 유지).

        N-1: ``result.blocked``/실패를 확인하지 않고 무조건 '청산완료(exited)' 로
        표기하면, 실계좌엔 매도가 안 나간 채 시스템만 청산으로 착각해 실포지션이
        무감시로 남는다.
        C-1: 접수 성공도 청산이 아니다 — 체결 확인 전까지 selling 으로 두고 계속 감시한다.
        N-2: 발주 거부는 어댑터에서 예외(KiwoomError)로 튄다 → 여기서 잡아 루프를
        살리고 포지션을 open 으로 유지, 다음 틱에 재시도한다.
        """
        sell_qty = pos.qty if qty is None else max(1, min(int(qty), pos.qty))
        try:
            result = self._client.sell_limit(
                pos.code, sell_qty, price, live=self._live, ref_price=price
            )
        except Exception as exc:  # noqa: BLE001 — 거부·네트워크 어떤 예외든 크래시 금지
            self._notice(
                f"[주문][경보] 청산 매도 실패(예외) — {pos.name}({pos.code}) {reason} "
                f"@ {price:,}: {exc} · 실포지션 무감시 방지 위해 open 유지, 다음 틱 재시도"
            )
            return False
        if result.blocked or not result.ok:
            why = result.blocked_reason or f"rc={result.return_code}"
            self._notice(
                f"[주문][경보] 청산 매도 미성사 — {pos.name}({pos.code}) {reason} "
                f"@ {price:,}: {why} · open 유지, 다음 틱 재시도"
            )
            return False
        pos.status = "selling"
        pos.exit_reason = reason
        pos.sell_ord_no = result.ord_no
        pos.sell_price = price
        pos.sell_qty = sell_qty
        pos.sell_deadline = ts + timedelta(seconds=self._unfilled_timeout)
        tag = f"{sell_qty}/{pos.qty}주" if sell_qty < pos.qty else f"{pos.qty}주"
        self._notice(
            f"[주문] 청산주문({reason}) — {pos.name}({pos.code}) {tag} @ {price:,} "
            f"· 체결확인 대기"
        )
        return True

    def _try_cancel(self, pos: LivePosition, reason: str, *, ord_no: str | None = None,
                    qty: int | None = None) -> bool:
        """주문 취소를 시도한다. 성공 시 True, 실패 시 False(호출부가 유지·재시도 판단).

        N-2: 거부·예외를 잡아 크래시를 막는다.
        C-2: 어댑터 ``cancel()`` 은 매수 ``submit()`` 과 달리 ``return_code != 0`` 에
        예외를 **안 던진다**(HTTP 4xx 만 던짐). 반환값을 안 보면 브로커가 취소를
        거부해도 성공으로 오판해 포지션을 버린다 — 방아쇠는 ``holdings()`` 5초 TTL
        캐시다. 방금 체결된 주문을 미체결로 보고 취소를 걸면 브로커가 "이미 체결됨"
        으로 거부하는데, 그걸 성공으로 읽으면 체결된 1주가 시스템 밖으로 영구 이탈한다.
        (dry-run 결과는 ``dry_run=True`` 라 ``ok`` 가 True — 정상 취소가 실패로
        오판되지 않는다.)
        """
        target = ord_no if ord_no is not None else pos.buy_ord_no
        # 분할 매도를 취소할 때는 보유 수량이 아니라 **접수한 수량**을 넘겨야 한다.
        cancel_qty = pos.qty if qty is None else int(qty)
        try:
            result = self._client.cancel(target or "", pos.code, cancel_qty, live=self._live)
        except Exception as exc:  # noqa: BLE001 — 취소 거부로 프로세스 크래시 금지
            self._notice(
                f"[주문][경보] 취소 실패(예외·{reason}) — {pos.name}({pos.code}): {exc} · "
                f"포지션 유지, 재시도"
            )
            return False
        if result is None or result.blocked or not result.ok:
            why = (result.blocked_reason or f"rc={result.return_code} {result.return_msg}".strip()
                   if result is not None else "응답 없음")
            self._notice(
                f"[주문][경보] 취소 미성사({reason}) — {pos.name}({pos.code}) 주문#{target}: "
                f"{why} · 포지션 유지, 다음 틱 재시도(이미 체결됐다면 체결로 넘어간다)"
            )
            return False
        return True

    # ----- 대장 상태(만쥬 §5-3) ----------------------------------------------

    def on_leader_state(self, leader_code: str, ts: datetime, state: str) -> None:
        """감지기가 판정한 대장 상태를 먹인다. 이탈이면 그 대장 포지션을 즉시 청산.

        만쥬 §5-3: **"대장주가 상한가를 말지 못하거나 풀리고 꺾이는 순간, 손익과 무관하게
        즉시 전량 매도."** 실행기는 상한가를 스스로 계산하지 않는다 — 판정은 감지기 몫이고
        여기 오는 건 결론뿐이다.

        안전 원칙은 그대로다: 매도는 **접수까지만**(``selling``), 체결이 확인돼야
        ``exited``(C-1). 접수 실패면 포지션을 버리지 않고 ``filled`` 로 유지해 다음 틱에
        재시도한다(N-1). 이미 매도가 나가 있는 ``selling`` 은 손대지 않는다 — 취소 없이
        재발주하면 이중 매도가 되어 무보유 매도로 이어진다.
        """
        if state != LEADER_LOST or not self._leader_exit:
            return
        for pos in self.positions:
            if pos.leader != leader_code or pos.status not in ("pending", "filled"):
                continue
            pos.leader_lost = True
        # pending 취소·filled 매도는 상태를 옮기므로 목록을 따로 훑는다.
        still: list[LivePosition] = []
        for pos in self.positions:
            if not (pos.leader_lost and pos.leader == leader_code):
                still.append(pos)
                continue
            if pos.status == "pending":
                if not self._try_cancel(pos, "대장이탈취소"):
                    still.append(pos)      # 취소 거부 → pending 유지, 다음 틱 재시도
                    continue
                pos.status = "cancelled"
                pos.exit_reason = "대장이탈취소"
                self._notice(f"[주문] 대장이탈취소 — {pos.name}({pos.code}) 대기 매수 취소")
                self.history.append(pos)
                continue
            if pos.status == "filled":
                if pos.last_price is None:
                    # 체결 직후 틱을 못 받아 호가를 모른다. 플래그는 남았으니 다음 틱에
                    # 청산규칙이 대장이탈로 탄다.
                    self._notice(
                        f"[주문][경보] 대장이탈 즉시청산 보류 — {pos.name}({pos.code}): "
                        f"청산가를 모른다(틱 없음) · 다음 틱에 즉시 청산"
                    )
                else:
                    self._sell(pos, ts, round(pos.last_price), "대장이탈")
            still.append(pos)
        self.positions = still

    # ----- 킬/마감 -----------------------------------------------------------

    def on_kill(self) -> None:
        """가드 킬 발동 → 대기 매수 전부 취소(체결분은 청산규칙에 맡긴다)."""
        for pos in self.positions:
            if pos.status == "pending":
                if not self._try_cancel(pos, "킬스위치취소"):
                    continue   # 실패분은 pending 유지 — 다음 on_kill/flush 에서 재시도
                pos.status = "cancelled"
                pos.exit_reason = "킬스위치취소"
                self._notice(f"[주문] 킬스위치 — 대기 매수 취소 {pos.name}({pos.code})")

    def flush(self, now: datetime) -> None:
        """장 마감·틱 공백 정리: 대기분 취소 + **체결·매도대기분 마감청산**(C-3).

        C-3: 예전엔 ``pending`` 만 처리해서 15:19 진입 → 15:20 소켓 종료 → flush →
        filled 그대로 → 프로세스 종료 → **오버나이트**였다. 포지션은 메모리에만 있어
        다음 날 인수 코드가 없고, 재진입 차단 목록에 올라 팔아줄 주체도 없다.
        Ctrl-C/SIGTERM 도 같은 경로다.

        flush 에는 가격 인자가 없으므로 :meth:`on_tick` 이 기록해 둔 ``last_price``
        를 청산가로 쓴다. 그마저 없으면(틱을 한 번도 못 받음) 조용히 넘기지 않고
        사람이 볼 경보를 남긴다.
        """
        for pos in list(self.positions):
            # 마감 flush 는 '오늘이 끝났다' 는 뜻이라 취소 시각 도달 여부를 안 따진다.
            # 15:19:55 에 낸 대기 매수는 아직 deadline 전이지만, 그대로 두면 주문이
            # 장 밖으로 넘어간다.
            if pos.status == "pending":
                if not self._try_cancel(pos, "마감미체결취소"):
                    continue
                pos.status = "cancelled"
                pos.exit_reason = "마감미체결취소"
                self._notice(f"[주문] 마감 미체결 취소 — {pos.name}({pos.code})")
                self.positions.remove(pos)
                self.history.append(pos)
            elif pos.status == "filled":
                if pos.add_pending_qty > 0:
                    # 불타기 추가매수가 미체결로 남아 있다 — 마감청산 전에 반드시
                    # 취소해야 한다. 안 그러면 진입 매수와 같은 C-3(오버나이트 잔존
                    # 주문) 이 추가매수 쪽에서 그대로 재발한다.
                    if not self._try_cancel(
                        pos, "불타기마감취소", ord_no=pos.add_ord_no, qty=pos.add_pending_qty
                    ):
                        self._notice(
                            f"[주문][경보] 불타기 미체결 마감취소 실패 — {pos.name}({pos.code}) "
                            f"{pos.add_pending_qty}주 @ {(pos.add_price or 0):,} · 수동 확인 필요, "
                            f"이번 flush 는 이 포지션을 건너뛴다"
                        )
                        continue
                    self._notice(
                        f"[주문] 불타기마감취소 — {pos.name}({pos.code}) {pos.add_pending_qty}주"
                    )
                    pos.add_pending_qty = 0
                    pos.add_ord_no = None
                    pos.add_price = None
                    pos.add_deadline = None
                if pos.last_price is None:
                    self._notice(
                        f"[주문][경보] 마감청산 불가 — {pos.name}({pos.code}) {pos.qty}주: "
                        f"체결 후 틱을 한 번도 못 받아 청산가를 모른다 · 수동 확인 필요"
                    )
                    continue
                self._sell(pos, now, round(pos.last_price), "마감청산")
            elif pos.status == "selling":
                # 매도가 이미 나가 있다. 마지막 가격이 접수가와 다르면 그쪽으로 재호가,
                # 같거나 모르면 이중 매도 위험을 피해 그대로 두고 경보만 남긴다.
                if pos.last_price is None or round(pos.last_price) == pos.sell_price:
                    self._notice(
                        f"[주문][경보] 마감 시점 청산 미체결 — {pos.name}({pos.code}) "
                        f"{pos.qty}주 @ {(pos.sell_price or 0):,} ({pos.exit_reason}) "
                        f"주문 잔존 · 수동 확인 필요"
                    )
                    continue
                if not self._try_cancel(pos, "마감재호가", ord_no=pos.sell_ord_no,
                                        qty=pos.sell_qty or pos.qty):
                    continue
                pos.status = "filled"
                pos.sell_ord_no = None
                pos.sell_qty = 0
                # 마감은 전량이다 — 분할 접수분만 재호가하면 잔량이 오버나이트로 넘어간다.
                self._sell(pos, now, round(pos.last_price), "마감청산")

    def summary(self) -> str:
        # 상태로만 센다 — 취소분이 positions 에 남아 있든 history 에 있든 결과가 같아야
        # 하고, selling(매도 접수·체결확인 대기)은 **청산이 아니라 활성**이다(C-1).
        allpos = self.positions + self.history
        active = [p for p in allpos if p.status in ("pending", "filled", "selling")]
        selling = [p for p in allpos if p.status == "selling"]
        exited = [p for p in allpos if p.status == "exited"]
        blocked = [p for p in allpos if p.status == "blocked"]
        cancelled = [p for p in allpos if p.status == "cancelled"]
        mode = "LIVE" if self._live else "DRY-RUN"
        return (f"[실행기 {mode}] 활성 {len(active)}(매도대기 {len(selling)}) · "
                f"청산 {len(exited)} · 취소 {len(cancelled)} · 차단 {len(blocked)}")
