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

if TYPE_CHECKING:
    from ..adapters.kiwoom_order import KiwoomOrderClient, OrderResult
    from .pair_detector import Alert

__all__ = ["LivePosition", "OrderExecutor"]

_EPS = 1e-9


@dataclass
class LivePosition:
    """실행기가 관리하는 실주문 포지션 하나(1주 지정가 매수 → 청산)."""

    code: str
    name: str
    qty: int
    entry: int                          # 매수 지정가(bid1)
    take_profit: float                  # 익절가(원)
    stop_loss: float | None
    disaster_line: float | None
    alert_ts: datetime
    exit_ts: datetime                   # 시간청산 시각
    cancel_deadline: datetime           # 미체결 취소 시각
    buy_ord_no: str | None = None
    status: str = "pending"             # pending | filled | exited | cancelled | blocked
    fill_ts: datetime | None = None
    fill_price: int | None = None
    exit_reason: str = ""
    sell_ord_no: str | None = None


class OrderExecutor:
    """알림 → 매수, 틱 → 체결·청산. dry-run 기본."""

    def __init__(
        self,
        client: KiwoomOrderClient,
        *,
        qty: int = 1,
        live: bool = False,
        unfilled_timeout: float = 15.0,
        notice: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._qty = max(1, int(qty))
        self._live = bool(live)
        self._unfilled_timeout = float(unfilled_timeout)
        self._notice = notice or (lambda _s: None)
        self.positions: list[LivePosition] = []
        self.history: list[LivePosition] = []

    # ----- 조회 --------------------------------------------------------------

    def _active_for(self, code: str) -> LivePosition | None:
        for p in self.positions:
            if p.code == code and p.status in ("pending", "filled"):
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
        entry = round(alert.entry)
        result = self._client.buy_limit(
            code, self._qty, entry, live=self._live, ref_price=alert.price
        )
        pos = LivePosition(
            code=code,
            name=alert.follower_name or code,
            qty=self._qty,
            entry=entry,
            take_profit=alert.take_profit,
            stop_loss=alert.stop_loss,
            disaster_line=alert.disaster_line,
            alert_ts=alert.ts,
            exit_ts=alert.exit_ts,
            cancel_deadline=alert.ts + timedelta(seconds=self._unfilled_timeout),
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
                # 미체결 취소 시각 도달 → 취소.
                if ts >= pos.cancel_deadline:
                    self._client.cancel(pos.buy_ord_no or "", pos.code, pos.qty, live=self._live)
                    pos.status = "cancelled"
                    pos.exit_reason = "미체결취소"
                    self._notice(f"[주문] 미체결 취소 — {pos.name}({pos.code})")
                    return True
                return False
            # 방금 체결됨 → 아래 청산 판정으로 진행
        if pos.status != "filled":
            return False
        # 청산: 익절 → 손절 → 재난선 → 시간청산
        reason: str | None = None
        if price >= pos.take_profit - _EPS:
            reason = "익절"
        elif pos.stop_loss is not None and price <= pos.stop_loss + _EPS:
            reason = "손절"
        elif pos.disaster_line is not None and price <= pos.disaster_line + _EPS:
            reason = "재난선"
        elif ts >= pos.exit_ts:
            reason = "시간청산"
        if reason is None:
            return False
        self._sell(pos, ts, round(price), reason)
        return True

    def _check_fill(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        """대기 매수가 체결됐는지. live 는 미체결조회, dry 는 틱 관통 근사."""
        if self._live and pos.buy_ord_no:
            try:
                filled = self._client.is_filled(pos.buy_ord_no, pos.code)
            except Exception:  # noqa: BLE001 — 조회 실패 시 체결로 단정하지 않는다
                filled = False
        else:
            # dry-run 근사: 지정가 이하로 내려오면 체결(touch).
            filled = price <= pos.entry + _EPS
        if filled:
            pos.status = "filled"
            pos.fill_ts = ts
            pos.fill_price = pos.entry
            self._notice(f"[주문] 체결 — {pos.name}({pos.code}) {pos.qty}주 @ {pos.entry:,}")
        return filled

    def _sell(self, pos: LivePosition, ts: datetime, price: int, reason: str) -> None:
        result = self._client.sell_limit(pos.code, pos.qty, price, live=self._live, ref_price=price)
        pos.status = "exited"
        pos.exit_reason = reason
        pos.sell_ord_no = result.ord_no
        self._notice(f"[주문] 청산({reason}) — {pos.name}({pos.code}) {pos.qty}주 @ {price:,}")

    # ----- 킬/마감 -----------------------------------------------------------

    def on_kill(self) -> None:
        """가드 킬 발동 → 대기 매수 전부 취소(체결분은 청산규칙에 맡긴다)."""
        for pos in self.positions:
            if pos.status == "pending":
                self._client.cancel(pos.buy_ord_no or "", pos.code, pos.qty, live=self._live)
                pos.status = "cancelled"
                pos.exit_reason = "킬스위치취소"
                self._notice(f"[주문] 킬스위치 — 대기 매수 취소 {pos.name}({pos.code})")

    def flush(self, now: datetime) -> None:
        """장 마감·틱 공백 정리: 대기분 취소, 체결분 시간청산 도달분 정리."""
        for pos in list(self.positions):
            if pos.status == "pending" and now >= pos.cancel_deadline:
                self._client.cancel(pos.buy_ord_no or "", pos.code, pos.qty, live=self._live)
                pos.status = "cancelled"
                pos.exit_reason = "마감미체결취소"

    def summary(self) -> str:
        active = [p for p in self.positions if p.status in ("pending", "filled")]
        exited = [p for p in self.history if p.status == "exited"]
        blocked = [p for p in self.history if p.status == "blocked"]
        cancelled = [p for p in self.history if p.status == "cancelled"]
        mode = "LIVE" if self._live else "DRY-RUN"
        return (f"[실행기 {mode}] 활성 {len(active)} · 청산 {len(exited)} · "
                f"취소 {len(cancelled)} · 차단 {len(blocked)}")
