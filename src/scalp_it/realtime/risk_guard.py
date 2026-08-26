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

* **대장 조건부 청산**(만쥬 §5-3): 포지션마다 그 포지션을 만든 **대장 코드**가 붙어 있고,
  감지기가 :meth:`RiskGuard.on_leader_state` 로 대장의 상한가 유지/이탈을 통보한다.
  이탈이면 그 대장으로 연 포지션을 **손익 무관 즉시** 청산한다(사유 ``대장이탈``).
  이 모듈은 상한가를 스스로 계산하지 않는다 — 판정은 감지기 몫이다.

* **국면별 비중**(만쥬 §8-1): 포지션 명목 = ``position_notional × size_multiplier``.
  배수는 :meth:`RiskGuard.set_size_multiplier` 로 **주입**받는다 — 이 모듈은 DB 를
  모르므로 ``market_regime`` 을 import 하지 않는다. 배수 0(극침체)은 **신규 알림만**
  막고(킬 아님) 보유분 청산은 정상 진행한다.

* **분할 매도**(만쥬 §5-2): ``partial_exit_gain`` 도달 또는 ``partial_exit_after_s``
  경과 시 ``partial_exit_ratio`` 만큼 먼저 실현하고 잔량을 기존 규칙으로 굴린다.
  대장이탈·손절·재난선은 §5-3 "손익 무관 전량"이라 분할 대상이 아니다. 기본값 꺼짐.

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
    "BLOCK_REGIME_ZERO",
    "KILL_CONSECUTIVE",
    "KILL_DAILY_LOSS",
    "KILL_MANUAL",
    "LEADER_HOLDING",
    "LEADER_LOST",
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

# 대장 상태(만쥬 §5-3) — **감지기가 판정해 넘겨준다**. 이 모듈은 DB·시계를 모르므로
# 상한가 여부를 스스로 계산하지 않는다. 여기 오는 건 이미 내려진 판정뿐이다.
LEADER_HOLDING = "holding"   # 상한가를 말고 있음(유지 중)
LEADER_LOST = "lost"         # 상한가를 못 말거나 풀리고 꺾임 = 매수 이유 상실

# 신규 알림 억제 사유 — 킬스위치가 **아니다**. 국면 배수가 0(극침체)이면 그날 신규
# 진입만 막고, 배수를 되돌리면 즉시 재개된다. 킬은 하루를 접는 것이라 성격이 다르다.
BLOCK_REGIME_ZERO = "regime_zero"

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
    # 대장 조건부 청산(만쥬 §5-3). 켜 두어도 대장 상태를 안 먹이면 아무 일도 없다.
    leader_exit: bool = True
    # 국면별 비중(만쥬 §8-1) — 배수는 **주입**받는다. 여기서 ``market_regime`` 을
    # import 하지 않는 건 그 모듈이 DB 를 알기 때문이다(이 모듈은 DB·시계 무지가 규약).
    # 배선은 CLI 몫이고, 장중 국면이 바뀌면 :meth:`RiskGuard.set_size_multiplier` 로 갱신.
    size_multiplier: float = 1.0
    # 만쥬 §9-6 — 손실 킬스위치의 거울상. 오전에 목표 이상 벌면 오후엔 비중을 줄인다.
    #   원문: "오전에 큰 수익을 낸 뒤 오후에 흥분해 무리하게 베팅하다 수익을 전부
    #   날리는 패턴을 **가장 경계**" / "목표치를 대폭 낮추고 소액 비중으로만 접근".
    #   멈추는 게 아니라 **줄이는** 것이다 — 그래서 킬스위치가 아니라 배수로 만든다.
    daily_profit_target: float = 0.0           # 일일 이익 목표(원). 0 = 끔
    profit_size_multiplier: float = 0.3        # 목표 도달 후 곱할 배수
    # 분할 매도(만쥬 §5-2 "탄력이 충분히 나온 시점에 **분할 또는 전량** 매도").
    # 1차로 ``partial_exit_ratio`` 만큼 덜어내고 잔량은 기존 규칙(익절/손절/재난선/
    # 대장이탈/시간청산) 그대로 굴린다. ratio 가 0 이면 완전히 꺼진 상태 = 기존 동작.
    partial_exit_ratio: float = 0.0            # 1차 분할 비율(0<r<1). 0 = 끔
    partial_exit_gain: float | None = None     # 이 수익률 도달 시 1차 분할(예: 0.002)
    partial_exit_after_s: float | None = None  # 체결 후 이 초 경과 시 1차 분할
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
    take_profit: float | None
    stop_loss: float | None
    disaster_line: float | None
    exit_ts: datetime
    notional: float
    follower_name: str = ""
    basis: str = "through"                    # 이 포지션이 굴러가는 체결 가정
    shadow: bool = False                      # True 면 회계 밖(로그 전용) 그림자
    leader_lost: bool = False                 # 대장이 상한가를 놓쳤다고 통보받았는가
    # 분할 매도(만쥬 §5-2). remaining 은 아직 안 판 비율 — 잔량 손익은 여기에 곱한다.
    remaining: float = 1.0
    partial_done: bool = False
    partial_ratio: float = 0.0
    partial_price: float | None = None
    partial_ts: datetime | None = None
    partial_reason: str = ""
    partial_pnl: float | None = None
    partial_pnl_krw: float = 0.0
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
            "leader_lost": self.leader_lost,
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
            # 분할 매도 — 잔량 비율과 1차분 실현손익을 같이 남겨야 회계가 맞춰진다.
            "remaining": round(self.remaining, 6),
            "partial_ratio": round(self.partial_ratio, 6),
            "partial_reason": self.partial_reason,
            "partial_price": self.partial_price,
            "partial_pnl": None if self.partial_pnl is None else round(self.partial_pnl, 6),
            "partial_pnl_krw": round(self.partial_pnl_krw, 2),
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
        # 국면 배수는 장중에 바뀔 수 있어 config(frozen) 밖에 둔다. 초기값은 설정에서.
        self.size_multiplier: float = float(self.config.size_multiplier)
        self.profit_capped: bool = False      # 일일 이익 목표 도달(만쥬 §9-6)
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

    def effective_notional(self) -> float:
        """지금 열 포지션의 명목(원) = 기본 × 국면배수 × (이익목표 도달 시 절제배수).

        만쥬 §8-1(국면별 비중)과 §9-6(오전 수익 뒤 오후 축소)이 곱으로 겹친다.
        둘 다 "덜 걸어라"라 방향이 같으므로 곱하는 게 자연스럽다.
        """
        mult = self.size_multiplier
        if self.profit_capped:
            mult *= max(0.0, float(self.config.profit_size_multiplier))
        return self.config.position_notional * mult

    def set_size_multiplier(self, mult: float) -> None:
        """국면 배수를 갱신한다(만쥬 §8-1). CLI 가 ``market_regime`` 판정을 주입한다.

        **이미 열린 포지션은 건드리지 않는다** — 명목이 소급 변경되면 그 포지션의
        회계가 진입 시점과 달라진다. 배수는 다음 진입부터 적용된다.
        """
        self.size_multiplier = max(0.0, float(mult))

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
            self.profit_capped = False
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
        """현재 상태로 킬 여부·이익절제 갱신(수동 정지·손실한도·연속손절·이익목표)."""
        cfg = self.config
        # 이익 목표는 **킬이 아니다** — 매매를 멈추지 않고 비중만 줄인다(§9-6).
        if cfg.daily_profit_target > 0 and self.realized_krw >= cfg.daily_profit_target:
            self.profit_capped = True
        if self.killed:
            return
        # 이익 목표는 **킬이 아니다** — 매매를 멈추지 않고 비중만 줄인다(§9-6).
        # 그래서 killed 판정보다 먼저, 그리고 독립적으로 갱신한다.
        if cfg.daily_profit_target > 0 and self.realized_krw >= cfg.daily_profit_target:
            self.profit_capped = True
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
        # 국면 배수 0(만쥬 §8-1 극침체) = 신규 진입 금지. **킬이 아니다** — 배너도
        # 카운터도 건드리지 않고, 배수가 돌아오면 그 자리에서 다시 통과한다.
        # 기존 포지션의 청산 경로(on_tick/on_leader_state/flush)는 여기를 안 지나므로
        # 배수가 0이어도 들고 있는 건 규칙대로 정상 청산된다.
        if self.size_multiplier <= _EPS:
            return Decision(allowed=False, reason=BLOCK_REGIME_ZERO)
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
            notional=self.effective_notional(),
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

        # 청산 규칙: 익절 → (대장이탈) → 손절 → 재난선 → 시간청산
        # 대장이탈은 틱이 아니라 :meth:`on_leader_state` 로 들어와 그 자리에서 즉시
        # 청산되므로 여기 분기가 없다. 우선순위 근거는 그 메서드 주석에 적었다.
        # 청산가는 **주문 유형**에 맞춘다. 익절은 익절가에 미리 걸어둔 지정가가 채워지는
        # 것이라 레벨 그대로가 맞다. 반면 손절·재난선은 하락을 확인한 뒤 스프레드를 넘어
        # 파는 시장가라 실제 트리거 틱이 실현가다 — 레벨로 계상하면 손실이 과소계상된다
        # (실측: 계상 -0.400% vs 실제 -0.4941%, 19% 과소).
        if pos.take_profit is not None and price >= pos.take_profit - _EPS:
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
        # 전량 청산 사유가 하나도 안 걸린 뒤에만 1차 분할을 본다(만쥬 §5-2). 순서를
        # 뒤집으면 익절가를 찍은 틱에서 절반만 팔고 나머지가 뒤로 밀린다.
        self._maybe_partial(pos, ts, price)
        return False

    # ----- 분할 매도(만쥬 §5-2) ---------------------------------------------

    def _maybe_partial(self, pos: Position, ts: datetime, price: float) -> None:
        """1차 분할 청산. 조건 충족 시 ``ratio`` 만큼 실현하고 잔량을 계속 굴린다.

        만쥬 §5-2: "탄력이 충분히 나온 시점에서 **분할 또는 전량** 매도로 수익 실현."
        **분할 대상은 익절·시간청산 계열뿐이다.** 대장이탈·손절·재난선은 §5-3 원문이
        "손익 무관 **전량**"이라 쪼개지 않는다 — 매수 근거가 사라졌는데 절반을 들고
        있는 건 그 규칙의 정반대다. 그래서 분할 방아쇠는 (a) 수익률 도달, (b) 체결 후
        경과시간 둘뿐이고, 전량 사유가 걸리는 틱에서는 아예 호출되지 않는다.

        승패(연속손절·킬스위치)는 여기서 세지 않는다. 포지션이 아직 안 끝났고, 1차
        익절만으로 연속손절 카운터가 리셋되면 킬스위치가 무력화되기 때문이다. 실현
        손익(원)에는 그 즉시 반영한다 — 실제로 현금이 들어온 몫이라 손실한도에는
        정직하게 반영되어야 한다.
        """
        cfg = self.config
        ratio = cfg.partial_exit_ratio
        if pos.partial_done or not (_EPS < ratio < 1.0 - _EPS):
            return
        entry = pos.eff_entry()
        reason = ""
        if (cfg.partial_exit_gain is not None
                and price >= entry * (1.0 + cfg.partial_exit_gain) - _EPS):
            reason = "분할익절"
        elif cfg.partial_exit_after_s is not None:
            lat = pos.fill_latency(pos.basis)   # 알림→체결 지연(초)
            if lat is not None:
                held = (ts - pos.alert_ts).total_seconds() - lat
                if held >= cfg.partial_exit_after_s:
                    reason = "분할시간청산"
        if not reason:
            return
        pos.partial_done = True
        pos.partial_ratio = ratio
        pos.partial_reason = reason
        pos.partial_ts = ts
        pos.partial_price = round(price, 4)
        # 분할은 "지금 시장가로 판다"에 가깝다(탄력을 보고 내리는 판단) → 트리거 틱의
        # 가격이 실현가다. 익절처럼 미리 걸어둔 지정가가 아니므로 레벨을 쓰지 않는다.
        pos.partial_pnl = (price / entry - 1.0) - cfg.cost_rate
        pos.partial_pnl_krw = pos.partial_pnl * pos.notional * ratio
        pos.remaining = 1.0 - ratio
        if not pos.shadow:
            self.realized_krw += pos.partial_pnl_krw
            self._refresh_kill()
        if self._on_close is not None:
            self._on_close(pos)   # status 는 아직 open — sink 가 event='partial' 로 쓴다

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
            # 분할이 있었으면 잔량 몫만 계상한다(1차분은 이미 realized 에 들어갔다).
            pos.pnl_krw = pos.pnl * pos.notional * pos.remaining
            # 그림자 손익은 기록만 한다 — 실현손익·승패·연속손절·킬스위치는 회계 기준
            # 포지션 하나로만 굴러야 두 가정이 리스크 통제를 오염시키지 않는다.
            if not pos.shadow:
                self.realized_krw += pos.pnl_krw
                # 승패는 그 포지션이 최종적으로 남긴 **합계**로 가른다 — 1차 익절 후
                # 잔량이 살짝 밀린 건을 '손절'로 세면 연속손절 킬이 과민해진다.
                outcome = (pos.partial_pnl_krw + pos.pnl_krw) if pos.partial_done else pos.pnl
                if outcome < -_EPS:
                    self.losses += 1
                    self.consecutive_losses += 1
                else:
                    self.wins += 1
                    self.consecutive_losses = 0
                self._refresh_kill()
        (self.closed_shadows if pos.shadow else self.closed_positions).append(pos)
        if self._on_close is not None:
            self._on_close(pos)

    # ----- 대장 상태(만쥬 §5-3) ----------------------------------------------

    def on_leader_state(self, leader_code: str, ts: datetime, state: str) -> None:
        """감지기가 판정한 대장 상태를 먹인다. 이탈이면 그 대장으로 연 포지션 전량 청산.

        만쥬 §5-3 원문: **"대장주가 상한가를 말지 못하거나 상한가가 풀리고 꺾이는 순간,
        현재 수익률에 상관없이 즉시 탈출."** 우리 청산은 지금까지 후속주의 ±% 만 봤다 —
        정보가 있는 쪽(대장)을 안 보고 있었다. 1년 분봉 예비검증(753건)에서 이 축
        하나가 세전 −0.024% → +0.192% 로 21.6bp 를 움직였다.

        **우선순위 — 익절 > 대장이탈 > 손절 > 재난선 > 시간청산.**
        원문은 "손익 무관 즉시"라 손절·시간청산보다 무조건 앞이다. 익절만 예외로 앞에
        두는데, 익절은 미리 걸어둔 지정가라 그 레벨을 찍은 틱이면 대장 통보가 오기 전에
        이미 체결됐다고 보는 게 실제에 가깝기 때문이다. 여기서는 청산선을 이미 넘긴
        포지션이 열려 있을 수 없으므로(틱 경로가 먼저 닫는다) 사유는 항상 대장이탈이다.
        시간청산과 같은 순간에 겹쳐도 대장이탈로 기록한다 — 나간 이유가 그쪽이다.

        ``state`` 는 :data:`LEADER_HOLDING` / :data:`LEADER_LOST`. 유지 중은 무동작이고
        모르는 값도 무시한다(하위호환: 한 번도 안 먹이면 기존 동작 그대로).
        """
        if state != LEADER_LOST or not self.config.leader_exit:
            return
        self._observe_date(ts)
        if not self.open_positions and not self.shadow_positions:
            return
        # 그림자(반대 체결가정)에도 똑같이 적용한다 — 한쪽만 대장 규칙으로 나가면
        # touch/through 이중회계가 서로 다른 전략을 재는 꼴이 된다.
        self.open_positions = self._leader_exit(self.open_positions, leader_code, ts)
        self.shadow_positions = self._leader_exit(self.shadow_positions, leader_code, ts)

    def _leader_exit(self, positions: list[Position], leader_code: str,
                     ts: datetime) -> list[Position]:
        still: list[Position] = []
        for pos in positions:
            if pos.leader != leader_code or ts < pos.alert_ts:
                still.append(pos)
                continue
            pos.leader_lost = True
            if not pos.is_filled(pos.basis):
                # 아직 안 붙은 지정가 — 살 이유가 사라졌으니 그대로 소멸시킨다.
                self._close(pos, ts, None, "미체결(대장이탈)", filled=False)
                continue
            # 마지막으로 **본** 가격으로 계상한다(flush 와 같은 원칙). 대장 통보는
            # 후속주 틱과 별개로 오므로 그 순간의 후속주 가격을 우리는 모른다.
            entry = pos.eff_entry()
            px = round(pos.last_price if pos.last_price is not None else entry, 4)
            self._close(pos, ts, px, "대장이탈", filled=True)
        return still

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
        if pos.status == "unfilled":
            event = "unfilled"
        elif pos.status == "open" and pos.partial_done:
            # 아직 열린 채로 콜백이 온 유일한 경우 = 1차 분할. 청산으로 세면 안 된다.
            event = "partial"
        else:
            event = "close"
        self._write(pos.to_record(pos.basis or self.basis, event))

    def log_kill(self, when: datetime, reason: str, realized_krw: float) -> None:
        self._write({
            "ts": when.isoformat(sep=" ", timespec="seconds"),
            "event": "killswitch",
            "reason": reason,
            "realized_krw": round(realized_krw, 2),
        })
