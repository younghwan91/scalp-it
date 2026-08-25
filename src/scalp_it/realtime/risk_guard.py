"""짝꿍 리스크 가드 — 일일 손실한도·킬스위치·가상 포지션 추적.

냉정 감사(`docs/research/manju/38`)가 **소액 실전 착수의 선결 조건**으로 요구한 것.
짝꿍 엣지는 미확립(사후선택 착시 60~70%)이라 실전 목적은 "돈 버는" 게 아니라
"데이터 수집·리허설"이다 — 그러려면 손실 통제가 필수다.

이 모듈은 순수 로직(DB·시계 무지)이라 mock 틱·알림으로 단위 테스트한다. CLI 가
알림을 낼 때 :meth:`RiskGuard.on_alert` 를 거치게 하고, 틱마다
:meth:`RiskGuard.on_tick` 을 먹여 열린 가상 포지션을 규칙대로 청산(리허설)한다.

핵심 개념
=========

* **가상 포지션**: 감지기가 낸 알림 = 가상 포지션. 진입가(bid1)·익절·손절·재난선·
  청산시각을 기록한다. bid1 지정가 매수라 체결 여부가 불확실 — 두 체결 가정을 병기한다:

  - **touch(낙관)**: 이후 체결가가 bid1 **이하**로 내려오면 체결. 큐 앞이라고 가정.
  - **through(보수, 기본)**: 이후 체결가가 bid1 **미만**(관통)이어야 체결. 큐가 비워져야
    내 지정가가 채워진다는 가정. 감사가 "through 를 실전 기대로" 요구 → 회계 기본값.

  체결 뒤 익절/손절/재난선/시간청산 규칙으로 청산가를 정해 손익을 계산한다(리허설).
  사용자가 실제 체결했다고 표시하면(:meth:`mark_executed`) 그 포지션은 수동 체결로 본다.

* **킬스위치 3종 트리거**:

  (a) **일일 손실한도** — 누적 실현손실이 한도(원 또는 자본 %) 도달.
  (b) **연속 손절 N회** — 연속으로 손실 청산이 N회.
  (c) **수동 정지** — 정지 파일(``stop_file``)이 존재하면 즉시.

  발동되면 그날 신규 알림을 억제하고, 알림 대신 "🛑 킬스위치 발동" 배너를 한 번 낸다.

* **동시 보유 상한**: 열린 포지션이 ``max_positions`` 이상이면 신규 알림 스킵(킬 아님).

* **일일 리셋**: 날짜가 바뀌면 손실·연속손절 카운터와 킬 상태를 리셋한다.

안전 제약(불변)
==============

* **감지기 알림만 추적한다.** 손실한도·손익은 오직 감지기가 낸 짝꿍 신호 → 그 리허설/
  실체결 가상 포지션에서만 계산한다. 사용자 계좌의 다른 보유 종목 손익은 **절대 포함하지
  않는다** — 이 모듈은 계좌 잔고·보유를 조회하지도 건드리지도 않는다.
* **자동 주문 없음.** 알림·손익 기록만 한다. 매수·매도는 사람이 수동으로. 킬스위치도
  "신규 알림 억제"일 뿐 어떤 청산 주문도 자동으로 내지 않는다(주문 API 코드 없음).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pair_detector import Alert

__all__ = [
    "KILL_CONSECUTIVE",
    "KILL_DAILY_LOSS",
    "KILL_MANUAL",
    "Decision",
    "Position",
    "PositionLogSink",
    "RiskGuard",
    "RiskGuardConfig",
]

# 킬스위치 트리거 사유(문자열 상수)
KILL_DAILY_LOSS = "daily_loss"
KILL_CONSECUTIVE = "consecutive_losses"
KILL_MANUAL = "manual_stop"

_EPS = 1e-9


def _opposite_basis(basis: str) -> str:
    return "touch" if basis == "through" else "through"


# --------------------------------------------------------------------- config

@dataclass(frozen=True)
class RiskGuardConfig:
    """가드 파라미터. 보수적으로 기본값을 잡는다(감사 요구: 손실=학습비)."""

    daily_loss_limit: float = 50_000.0        # 일일 손실한도(원). 0/None = 끔
    daily_loss_limit_pct: float | None = None  # 있으면 자본 대비 %로 한도 산출(우선)
    daily_capital: float = 1_000_000.0        # % 한도·기록 기준 자본(원)
    position_notional: float = 100_000.0      # 가상 포지션 1건 명목(원) — 수익률→원 환산
    max_consecutive_losses: int = 3           # 연속 손절 상한(킬 트리거 b)
    max_positions: int = 2                    # 동시 보유 가상 포지션 상한
    fill_basis: str = "through"               # 'through'(보수·기본) | 'touch'(낙관)
    track_shadow_fill: bool = True            # 반대 가정을 그림자로 병행 추적(로그 전용)
    cost_rate: float = 0.0                    # 왕복 비용률(수익률에서 차감). 0 = 끔
    stop_file: str | None = "data/pair_STOP"  # 존재 시 수동 킬(트리거 c). None = 끔

    def loss_limit_krw(self) -> float:
        """실효 일일 손실한도(원). pct 가 있으면 자본×pct, 없으면 절대액."""
        if self.daily_loss_limit_pct:
            return abs(self.daily_capital * self.daily_loss_limit_pct)
        return abs(self.daily_loss_limit or 0.0)


# --------------------------------------------------------------------- 포지션

@dataclass
class Position:
    """알림 하나에서 생긴 가상 포지션. 체결·청산·손익·MAE 를 추적한다."""

    follower: str
    leader: str
    alert_ts: datetime
    entry: float
    take_profit: float
    stop_loss: float | None
    disaster_line: float | None
    exit_ts: datetime
    notional: float
    follower_name: str = ""
    basis: str = "through"                    # 이 포지션이 굴러가는 체결 가정
    shadow: bool = False                      # True 면 회계 밖(로그 전용) 그림자
    # 체결(두 가정 병기)
    touch_fill_ts: datetime | None = None
    through_fill_ts: datetime | None = None
    last_price: float | None = None           # 마지막으로 관측한 체결가(flush 계상용)
    manual: bool = False                      # 사용자가 실제 체결로 표시
    manual_entry: float | None = None
    manual_ts: datetime | None = None
    # 익스커전(체결 후 진입가 대비)
    mae: float = 0.0                          # 최대 역행(≤0)
    mfe: float = 0.0                          # 최대 순행(≥0)
    # 결과
    status: str = "open"                      # open | closed | unfilled
    exit_reason: str = ""                     # 익절 | 손절 | 재난선 | 시간청산 | 미체결
    exit_ts_actual: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None                  # 실현 수익률(비용 차감 후)
    pnl_krw: float | None = None              # 실현 손익(원)

    # ----- 체결 판정(회계 기준 = fill_basis) --------------------------------

    def _fill_ts(self, basis: str) -> datetime | None:
        return self.manual_ts if self.manual else (
            self.through_fill_ts if basis == "through" else self.touch_fill_ts)

    def is_filled(self, basis: str) -> bool:
        return self._fill_ts(basis) is not None

    def fill_latency(self, basis: str) -> float | None:
        ft = self._fill_ts(basis)
        return None if ft is None else max(0.0, (ft - self.alert_ts).total_seconds())

    def eff_entry(self) -> float:
        return self.manual_entry if (self.manual and self.manual_entry) else self.entry

    def to_record(self, basis: str | None = None, event: str = "close") -> dict[str, Any]:
        """포지션 로그 한 줄(감사 요구: MAE·지연·체결여부·손익, 가정별 한 줄씩)."""
        basis = basis or self.basis
        rec: dict[str, Any] = {
            "ts": (self.exit_ts_actual or self.alert_ts).isoformat(sep=" ", timespec="seconds"),
            "event": event,
            "leader": self.leader,
            "follower": self.follower,
            "follower_name": self.follower_name,
            "alert_ts": self.alert_ts.isoformat(sep=" ", timespec="seconds"),
            "entry": round(self.eff_entry(), 4),
            "status": self.status,
            "exit_reason": self.exit_reason,
            "manual": self.manual,
            # 체결 여부·지연 — 두 가정 병기
            "fill_basis": basis,
            "filled": self.is_filled(basis),
            "filled_touch": self.touch_fill_ts is not None,
            "filled_through": self.through_fill_ts is not None,
            "latency_s": self.fill_latency(basis),
            "latency_touch_s": self.fill_latency("touch"),
            "latency_through_s": self.fill_latency("through"),
            # 익스커전·손익
            "mae": round(self.mae, 6),
            "mfe": round(self.mfe, 6),
            "exit_price": self.exit_price,
            "pnl": None if self.pnl is None else round(self.pnl, 6),
            "pnl_krw": None if self.pnl_krw is None else round(self.pnl_krw, 2),
        }
        return rec


# --------------------------------------------------------------------- 결정

@dataclass(frozen=True)
class Decision:
    """알림 통과 여부. ``allowed=False`` 면 신규 알림을 억제한다."""

    allowed: bool
    reason: str = ""                          # 억제 사유(kill 사유 또는 'max_positions')
    killed: bool = False                      # 킬스위치 발동 상태
    banner: str | None = None                 # 처음 발동 시 한 번 낼 배너 텍스트


# --------------------------------------------------------------------- 가드

class RiskGuard:
    """가상 포지션 추적 + 킬스위치. 알림은 :meth:`on_alert`, 틱은 :meth:`on_tick`."""

    def __init__(self, config: RiskGuardConfig | None = None) -> None:
        self.config = config or RiskGuardConfig()
        self.day: date | None = None
        self.realized_krw: float = 0.0        # 당일 누적 실현손익(원)
        self.consecutive_losses: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.killed: bool = False
        self.kill_reason: str = ""
        self._banner_shown: bool = False
        self.open_positions: list[Position] = []
        self.closed_positions: list[Position] = []
        # 반대 체결 가정의 그림자 — 로그·집계 전용. 회계(realized_krw·연속손절·킬·상한)에
        # 절대 끼어들지 않는다. 실전 회계는 보수 기준 하나여야 하기 때문.
        self.shadow_positions: list[Position] = []
        self.closed_shadows: list[Position] = []
        self._on_close = None                 # 콜백(Position) — 로깅용

    def set_close_callback(self, fn: Any) -> None:
        """포지션 청산 때 호출할 콜백(로깅 sink 연결용)."""
        self._on_close = fn

    # ----- 일일 리셋 ----------------------------------------------------------

    def _observe_date(self, when: datetime) -> None:
        d = when.date()
        if self.day is None:
            self.day = d
            return
        if d != self.day:
            # 날짜 변경 → 카운터·킬 리셋. 전날 미청산 포지션은 버린다.
            self.day = d
            self.realized_krw = 0.0
            self.consecutive_losses = 0
            self.wins = 0
            self.losses = 0
            self.killed = False
            self.kill_reason = ""
            self._banner_shown = False
            self.open_positions.clear()
            self.shadow_positions.clear()

    # ----- 킬스위치 판정 ------------------------------------------------------

    def _stop_file_present(self) -> bool:
        sf = self.config.stop_file
        return bool(sf) and Path(sf).exists()

    def _refresh_kill(self) -> None:
        """현재 상태로 킬 여부 갱신(수동 정지·손실한도·연속손절)."""
        if self.killed:
            return
        cfg = self.config
        if self._stop_file_present():
            self._trigger_kill(KILL_MANUAL)
        elif (cfg.max_consecutive_losses > 0
              and self.consecutive_losses >= cfg.max_consecutive_losses):
            self._trigger_kill(KILL_CONSECUTIVE)
        elif (cfg.loss_limit_krw() > 0
              and self.realized_krw <= -cfg.loss_limit_krw() + _EPS):
            self._trigger_kill(KILL_DAILY_LOSS)

    def _trigger_kill(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason

    def _banner(self) -> str | None:
        if self._banner_shown:
            return None
        self._banner_shown = True
        label = {
            KILL_DAILY_LOSS: (f"일일 손실한도 도달"
                              f"(실현 {self.realized_krw:,.0f}원 / 한도 "
                              f"-{self.config.loss_limit_krw():,.0f}원)"),
            KILL_CONSECUTIVE: f"연속 손절 {self.consecutive_losses}회",
            KILL_MANUAL: f"수동 정지 파일({self.config.stop_file})",
        }.get(self.kill_reason, self.kill_reason)
        return f"🛑 킬스위치 발동 — {label}. 오늘 신규 알림 중단."

    # ----- 알림 진입점 --------------------------------------------------------

    def on_alert(self, alert: Alert) -> Decision:
        """알림 하나를 가드에 태운다. 통과 시 가상 포지션 등록, 막히면 억제 결정."""
        self._observe_date(alert.ts)
        self._refresh_kill()
        if self.killed:
            return Decision(allowed=False, reason=self.kill_reason,
                            killed=True, banner=self._banner())
        if len(self.open_positions) >= self.config.max_positions:
            return Decision(allowed=False, reason="max_positions")
        # 통과 — 가상 포지션 등록
        basis = self.config.fill_basis
        self.open_positions.append(self._new_position(alert, basis, shadow=False))
        # 반대 가정은 **독립 수명**으로 굴린다: 자기 체결 시점부터 자기 MAE/MFE 를 재고
        # 자기 청산 판정을 한다. 하나의 수명으로 두 가정을 병기하면 through ⊆ touch 라
        # 낙관 시나리오가 영영 측정되지 않는다(실물 033340 사례).
        if self.config.track_shadow_fill:
            self.shadow_positions.append(
                self._new_position(alert, _opposite_basis(basis), shadow=True))
        return Decision(allowed=True)

    def _new_position(self, alert: Alert, basis: str, *, shadow: bool) -> Position:
        return Position(
            follower=alert.follower,
            leader=alert.leader,
            alert_ts=alert.ts,
            entry=alert.entry,
            take_profit=alert.take_profit,
            stop_loss=alert.stop_loss,
            disaster_line=alert.disaster_line,
            exit_ts=alert.exit_ts,
            notional=self.config.position_notional,
            follower_name=alert.follower_name,
            basis=basis,
            shadow=shadow,
        )

    # ----- 틱 진입점(리허설 체결·청산) ---------------------------------------

    def on_tick(self, code: str, ts: datetime, price: float) -> None:
        """후속주 틱으로 열린 포지션의 체결·MAE·청산을 리허설 규칙대로 진행한다."""
        self._observe_date(ts)
        if not self.open_positions and not self.shadow_positions:
            return
        self.open_positions = self._advance_all(self.open_positions, code, ts, price)
        self.shadow_positions = self._advance_all(self.shadow_positions, code, ts, price)

    def _advance_all(self, positions: list[Position], code: str, ts: datetime,
                     price: float) -> list[Position]:
        still_open: list[Position] = []
        for pos in positions:
            if pos.follower != code or not self._advance(pos, ts, price, pos.basis):
                still_open.append(pos)
        return still_open

    def _advance(self, pos: Position, ts: datetime, price: float, basis: str) -> bool:
        """포지션 하나에 틱 하나 적용. 청산되면 True."""
        if ts < pos.alert_ts:
            return False
        pos.last_price = price
        # 체결 판정(두 가정 병기) — 알림 시각 이후 첫 조건 충족
        if pos.touch_fill_ts is None and price <= pos.entry + _EPS:
            pos.touch_fill_ts = ts
        if pos.through_fill_ts is None and price < pos.entry - _EPS:
            pos.through_fill_ts = ts

        filled = pos.is_filled(basis)
        # 회계 기준으로 미체결인 채 청산시각 도달 → 미체결 만료
        if not filled:
            if ts >= pos.exit_ts:
                self._close(pos, ts, price=None, reason="미체결", filled=False)
                return True
            return False

        # 체결 후 — MAE/MFE 갱신
        entry = pos.eff_entry()
        ret = price / entry - 1.0
        pos.mae = min(pos.mae, ret)
        pos.mfe = max(pos.mfe, ret)

        # 청산 규칙: 익절 → 손절 → 재난선 → 시간청산
        # 청산가는 **주문 유형**에 맞춘다. 익절은 익절가에 미리 걸어둔 지정가가 채워지는
        # 것이라 레벨 그대로가 맞다. 반면 손절·재난선은 하락을 확인한 뒤 스프레드를 넘어
        # 파는 시장가라 실제 트리거 틱이 실현가다 — 레벨로 계상하면 손실이 과소계상된다
        # (실측: 계상 -0.400% vs 실제 -0.4941%, 19% 과소).
        if price >= pos.take_profit - _EPS:
            self._close(pos, ts, pos.take_profit, "익절", filled=True)
            return True
        if pos.stop_loss is not None and price <= pos.stop_loss + _EPS:
            self._close(pos, ts, price, "손절", filled=True)
            return True
        if pos.disaster_line is not None and price <= pos.disaster_line + _EPS:
            self._close(pos, ts, price, "재난선", filled=True)
            return True
        if ts >= pos.exit_ts:
            self._close(pos, ts, price, "시간청산", filled=True)
            return True
        return False

    def _close(self, pos: Position, ts: datetime, price: float | None,
               reason: str, *, filled: bool) -> None:
        pos.status = "closed" if filled else "unfilled"
        pos.exit_reason = reason
        pos.exit_ts_actual = ts
        pos.exit_price = price
        if filled and price is not None:
            entry = pos.eff_entry()
            gross = price / entry - 1.0
            pos.pnl = gross - self.config.cost_rate
            pos.pnl_krw = pos.pnl * pos.notional
            # 그림자 손익은 기록만 한다 — 실현손익·승패·연속손절·킬스위치는 회계 기준
            # 포지션 하나로만 굴러야 두 가정이 리스크 통제를 오염시키지 않는다.
            if not pos.shadow:
                self.realized_krw += pos.pnl_krw
                if pos.pnl < -_EPS:
                    self.losses += 1
                    self.consecutive_losses += 1
                else:
                    self.wins += 1
                    self.consecutive_losses = 0
                self._refresh_kill()
        (self.closed_shadows if pos.shadow else self.closed_positions).append(pos)
        if self._on_close is not None:
            self._on_close(pos)

    # ----- 수동 마킹·정지 -----------------------------------------------------

    def mark_executed(self, follower: str, entry: float | None = None,
                      ts: datetime | None = None) -> bool:
        """열린 가상 포지션을 사용자가 실제 체결했다고 표시. 성공 시 True."""
        for pos in self.open_positions:
            if pos.follower == follower and not pos.manual:
                pos.manual = True
                pos.manual_ts = ts or pos.alert_ts
                pos.manual_entry = entry
                # 실제 체결이 확인되면 체결 가정 자체가 무의미하다 → 그림자 폐기.
                self.shadow_positions = [
                    p for p in self.shadow_positions
                    if not (p.follower == follower and p.alert_ts == pos.alert_ts)]
                return True
        return False

    def force_stop(self) -> None:
        """수동 킬스위치(파일 없이 즉시)."""
        self._trigger_kill(KILL_MANUAL)

    def flush_expired(self, now: datetime) -> None:
        """청산시각이 지난 열린 포지션을 마지막 상태로 정리(장 마감·틱 공백 대비)."""
        self._observe_date(now)
        self.open_positions = self._flush(self.open_positions, now)
        self.shadow_positions = self._flush(self.shadow_positions, now)

    def _flush(self, positions: list[Position], now: datetime) -> list[Position]:
        still: list[Position] = []
        for pos in positions:
            if now >= pos.exit_ts:
                if pos.is_filled(pos.basis):
                    entry = pos.eff_entry()
                    # 마지막으로 **본** 가격으로 계상한다. 예전엔 entry*(1+mfe) 를 썼는데
                    # MFE 는 그 포지션이 본 최고가라, flush 를 타는 모든 건(틱 공백·장
                    # 마감)의 손익이 체계적으로 부풀려졌다.
                    px = round(pos.last_price if pos.last_price is not None else entry, 4)
                    self._close(pos, now, px, "시간청산", filled=True)
                else:
                    self._close(pos, now, None, "미체결", filled=False)
            else:
                still.append(pos)
        return still

    # ----- 상태 요약 ----------------------------------------------------------

    def status_line(self) -> str:
        state = f"🛑킬({self.kill_reason})" if self.killed else "정상"
        return (f"[가드] {state} 실현 {self.realized_krw:,.0f}원 "
                f"승{self.wins}/패{self.losses} 연속손절{self.consecutive_losses} "
                f"보유{len(self.open_positions)}/{self.config.max_positions}")


# --------------------------------------------------------------------- 로그 sink

class PositionLogSink:
    """가상 포지션 이벤트를 ``data/pair_positions.log`` 에 JSON 한 줄씩 append.

    감사 요구: 매 알림·가상체결의 MAE·지연·체결여부·손익을 남겨 나중에 실측 분석.
    ``event`` = ``open``(등록) | ``close``(청산) | ``killswitch``(발동).
    """

    def __init__(self, path: str | Path = "data/pair_positions.log", *,
                 basis: str = "through", echo: bool = False) -> None:
        self.path = Path(path)
        self.basis = basis
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print("[pos] " + line, flush=True)

    def log_open(self, alert: Alert) -> None:
        self._write({
            "ts": alert.ts.isoformat(sep=" ", timespec="seconds"),
            "event": "open",
            "leader": alert.leader,
            "follower": alert.follower,
            "follower_name": alert.follower_name,
            "entry": alert.entry,
            "take_profit": alert.take_profit,
            "stop_loss": alert.stop_loss,
            "disaster_line": alert.disaster_line,
            "exit_ts": alert.exit_ts.isoformat(sep=" ", timespec="seconds"),
        })

    def log_close(self, pos: Position) -> None:
        # 미체결은 청산이 아니다 — 별도 이벤트로 갈라야 집계의 "청산 N건"·MAE/MFE 중앙값에
        # 초기값 0.0 이 섞이지 않는다(체결률은 unfilled 건수로 따로 본다).
        event = "unfilled" if pos.status == "unfilled" else "close"
        self._write(pos.to_record(pos.basis or self.basis, event))

    def log_kill(self, when: datetime, reason: str, realized_krw: float) -> None:
        self._write({
            "ts": when.isoformat(sep=" ", timespec="seconds"),
            "event": "killswitch",
            "reason": reason,
            "realized_krw": round(realized_krw, 2),
        })
