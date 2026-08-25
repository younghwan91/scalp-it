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
    # selling = 청산 매도 접수됨·체결확인 대기(C-1). exited 는 체결 확인 후에만.
    status: str = "pending"             # pending | filled | selling | exited | cancelled | blocked
    fill_ts: datetime | None = None
    fill_price: int | None = None
    exit_reason: str = ""
    sell_ord_no: str | None = None
    sell_price: int | None = None       # 접수한 청산 지정가
    sell_deadline: datetime | None = None   # 이 시각까지 안 붙으면 취소 후 재호가
    sell_chases: int = 0                # 재호가(추격) 횟수
    chase_alerted: bool = False         # 추격 상한 경보를 이미 냈는가(중복 경보 방지)
    last_price: float | None = None     # 마지막으로 본 체결가 — flush 의 청산가로 쓴다


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
        notice: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._qty = max(1, int(qty))
        self._live = bool(live)
        self._unfilled_timeout = float(unfilled_timeout)
        # 청산 매도 재호가 상한 — 무한 추격(가격을 쫓아 계속 내리는 것)을 막는다.
        self._max_sell_chases = max(0, int(max_sell_chases))
        self._fill_basis = fill_basis if fill_basis in ("through", "touch") else "through"
        self._notice = notice or (lambda _s: None)
        self.positions: list[LivePosition] = []
        self.history: list[LivePosition] = []

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
        entry = round(alert.entry)
        try:
            result = self._client.buy_limit(
                code, self._qty, entry, live=self._live, ref_price=alert.price
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
                # 미체결 취소 시각 도달 → 취소.
                if ts >= pos.cancel_deadline:
                    if not self._try_cancel(pos, "미체결취소"):
                        return False   # 취소 실패 → pending 유지, 다음 틱 재시도
                    pos.status = "cancelled"
                    pos.exit_reason = "미체결취소"
                    self._notice(f"[주문] 미체결 취소 — {pos.name}({pos.code})")
                    return True
                return False
            # 방금 체결됨 → 아래 청산 판정으로 진행
        if pos.status == "selling":
            return self._advance_selling(pos, ts, price)
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
            pos.status = "exited"
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
        # 추격: 기존 매도 취소 → 현재가로 재호가.
        if not self._try_cancel(pos, "청산재호가", ord_no=pos.sell_ord_no):
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
        self._sell(pos, ts, new_price, pos.exit_reason)
        return False

    def _check_fill(self, pos: LivePosition, ts: datetime, price: float) -> bool:
        """대기 매수가 체결됐는지. live 는 미체결조회, dry 는 틱 관통 근사."""
        if self._live and pos.buy_ord_no:
            try:
                filled = self._client.is_filled(pos.buy_ord_no, pos.code)
            except Exception:  # noqa: BLE001 — 조회 실패 시 체결로 단정하지 않는다
                filled = False
        else:
            # dry-run 근사 — 회계 기준(RiskGuard.fill_basis)과 **같은 가정**을 써야 한다.
            # through(보수·기본): 지정가를 관통(미만)해야 큐가 비워져 내 주문이 채워진다.
            # touch(낙관): 지정가를 찍기만 하면 체결. 큐 맨 앞이라는 가정.
            # 두 가정이 갈리면 같은 틱열에서 실행기는 익절·가드는 손절을 기록해
            # (2026-08-25 033340) dry-run 으로 측정한 엣지가 주문기 거동과 달라진다.
            filled = (price < pos.entry - _EPS if self._fill_basis == "through"
                      else price <= pos.entry + _EPS)
        if filled:
            pos.status = "filled"
            pos.fill_ts = ts
            pos.fill_price = pos.entry
            self._notice(f"[주문] 체결 — {pos.name}({pos.code}) {pos.qty}주 @ {pos.entry:,}")
        return filled

    def _check_sell_fill(self, pos: LivePosition, price: float) -> bool:
        """청산 매도가 체결됐는지. live 는 잔고 소멸, dry 는 매수와 대칭인 근사.

        live 에서 :meth:`KiwoomOrderClient.is_filled` 를 그대로 쓰면 **뜻이 뒤집힌다** —
        그 메서드는 '잔고에 있으면 True(매수 체결)' 라, 매도에 쓰면 아직 보유 중인
        포지션을 청산 완료로 찍는다(= C-1 그 자체). 그래서 매도는 잔고에서 **사라졌는지**
        로 판정한다. 조회 예외는 매수 경로와 같은 원칙으로 '체결 아님'(fail-safe).
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
            for h in rows:
                try:
                    if h["code"] == pos.code and int(h.get("qty") or 0) > 0:
                        return False      # 아직 보유 → 미체결
                except (KeyError, TypeError, ValueError):
                    return False          # 응답이 이상하면 체결 단정 안 함
            return True                   # 잔고에서 사라짐 → 매도 체결
        # dry-run 근사 — 매수와 대칭. 매도 지정가는 가격이 그 **위로** 가야 붙는다.
        return (price > pos.sell_price + _EPS if self._fill_basis == "through"
                else price >= pos.sell_price - _EPS)

    def _sell(self, pos: LivePosition, ts: datetime, price: int, reason: str) -> bool:
        """청산 매도를 **접수**한다. 접수 성공 시 True(selling), 미성사 시 False(filled 유지).

        N-1: ``result.blocked``/실패를 확인하지 않고 무조건 '청산완료(exited)' 로
        표기하면, 실계좌엔 매도가 안 나간 채 시스템만 청산으로 착각해 실포지션이
        무감시로 남는다.
        C-1: 접수 성공도 청산이 아니다 — 체결 확인 전까지 selling 으로 두고 계속 감시한다.
        N-2: 발주 거부는 어댑터에서 예외(KiwoomError)로 튄다 → 여기서 잡아 루프를
        살리고 포지션을 open 으로 유지, 다음 틱에 재시도한다.
        """
        try:
            result = self._client.sell_limit(
                pos.code, pos.qty, price, live=self._live, ref_price=price
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
        pos.sell_deadline = ts + timedelta(seconds=self._unfilled_timeout)
        self._notice(
            f"[주문] 청산주문({reason}) — {pos.name}({pos.code}) {pos.qty}주 @ {price:,} "
            f"· 체결확인 대기"
        )
        return True

    def _try_cancel(self, pos: LivePosition, reason: str, *, ord_no: str | None = None) -> bool:
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
        try:
            result = self._client.cancel(target or "", pos.code, pos.qty, live=self._live)
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
                if not self._try_cancel(pos, "마감재호가", ord_no=pos.sell_ord_no):
                    continue
                pos.status = "filled"
                pos.sell_ord_no = None
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
