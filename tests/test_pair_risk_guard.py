# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""짝꿍 리스크 가드 단위테스트 — 손실한도·연속손절·일일리셋·포지션상한·체결가정.

가드는 DB·시계를 모르므로 mock 알림·틱만 먹여 순수 판정을 고정한다.
"""

import json
from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import (
    KILL_CONSECUTIVE,
    KILL_DAILY_LOSS,
    KILL_MANUAL,
    PositionLogSink,
    RiskGuard,
    RiskGuardConfig,
)

BASE = datetime(2026, 8, 24, 9, 10, 0)
FOL = "000002"


def _alert(secs: float = 0.0, *, entry: float = 100.0, tp: float = 100.5,
           sl: float | None = 99.6, disaster: float | None = None,
           hold: float = 90.0, follower: str = FOL) -> Alert:
    ts = BASE + timedelta(seconds=secs)
    return Alert(
        ts=ts, leader="000001", follower=follower,
        leader_surge=0.02, cosurge=0.006, premove=0.0,
        price=entry, bid1=entry, entry=entry,
        take_profit=tp, stop_loss=sl, disaster_line=disaster,
        exit_ts=ts + timedelta(seconds=hold), turnover=1e9,
        leader_name="대장", follower_name="후속",
    )


def _guard(**kw) -> RiskGuard:
    base = {"daily_loss_limit": 0.0, "max_consecutive_losses": 0,
            "max_positions": 99, "stop_file": None, "position_notional": 100_000.0}
    base.update(kw)
    return RiskGuard(RiskGuardConfig(**base))


def _settle_loss(g: RiskGuard, secs: float, follower: str = FOL) -> None:
    """알림→진입(관통 체결)→손절가 관통으로 손실 청산 한 사이클."""
    g.on_alert(_alert(secs, follower=follower))
    g.on_tick(follower, BASE + timedelta(seconds=secs + 1), 99.9)   # < entry → through 체결
    g.on_tick(follower, BASE + timedelta(seconds=secs + 2), 99.5)   # <= stop → 손절


def _settle_win(g: RiskGuard, secs: float, follower: str = FOL) -> None:
    g.on_alert(_alert(secs, follower=follower))
    g.on_tick(follower, BASE + timedelta(seconds=secs + 1), 99.9)   # through 체결
    g.on_tick(follower, BASE + timedelta(seconds=secs + 2), 100.6)  # >= tp → 익절


# --------------------------------------------------------------- 체결·청산 기본

def test_through_fill_and_take_profit():
    g = _guard()
    _settle_win(g, 0)
    assert len(g.closed_positions) == 1
    pos = g.closed_positions[0]
    assert pos.status == "closed"
    assert pos.exit_reason == "익절"
    assert pos.through_fill_ts is not None
    assert pos.pnl > 0
    assert g.wins == 1 and g.losses == 0


def test_stop_loss_counts_as_loss():
    g = _guard()
    _settle_loss(g, 0)
    pos = g.closed_positions[0]
    assert pos.exit_reason == "손절"
    assert pos.pnl < 0
    assert g.losses == 1
    assert g.consecutive_losses == 1
    assert g.realized_krw < 0


# --------------------------------------------------------------- 일일 손실한도

def test_daily_loss_limit_triggers_killswitch():
    # 손절 1회 = -0.4% * 10만 = -400원. 한도 1000원 → 3회째에 도달.
    g = _guard(daily_loss_limit=1000.0)
    for i in range(3):
        _settle_loss(g, i * 100, follower=FOL)
    assert g.killed is True
    assert g.kill_reason == KILL_DAILY_LOSS
    # 발동 후 신규 알림 억제 + 배너 1회
    d = g.on_alert(_alert(500))
    assert d.allowed is False
    assert d.killed is True
    assert d.banner is not None and "🛑" in d.banner
    # 배너는 한 번만
    assert g.on_alert(_alert(510)).banner is None


# --------------------------------------------------------------- 연속 손절

def test_consecutive_losses_trigger_killswitch():
    g = _guard(max_consecutive_losses=2)
    _settle_loss(g, 0)
    assert g.killed is False
    _settle_loss(g, 100)
    assert g.killed is True
    assert g.kill_reason == KILL_CONSECUTIVE


def test_win_resets_consecutive_counter():
    g = _guard(max_consecutive_losses=3)
    _settle_loss(g, 0)
    _settle_loss(g, 100)
    assert g.consecutive_losses == 2
    _settle_win(g, 200)
    assert g.consecutive_losses == 0
    assert g.killed is False


# --------------------------------------------------------------- 포지션 상한

def test_max_positions_suppresses_new_alert():
    g = _guard(max_positions=1)
    d1 = g.on_alert(_alert(0, follower="000002"))
    assert d1.allowed is True
    # 첫 포지션 미청산 상태에서 둘째 알림 → 상한 초과로 억제(킬 아님)
    d2 = g.on_alert(_alert(1, follower="000003"))
    assert d2.allowed is False
    assert d2.reason == "max_positions"
    assert d2.killed is False


def test_position_freed_after_close_allows_new():
    g = _guard(max_positions=1)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.6)  # 익절 청산 → 슬롯 반환
    assert g.open_positions == []
    assert g.on_alert(_alert(200)).allowed is True


# --------------------------------------------------------------- 일일 리셋

def test_daily_reset_clears_counters_and_kill():
    g = _guard(max_consecutive_losses=2)
    _settle_loss(g, 0)
    _settle_loss(g, 100)
    assert g.killed is True
    # 다음 날 알림 → 날짜 변경 감지 → 리셋 후 통과
    nextday = _alert(0)
    object.__setattr__(nextday, "ts", BASE + timedelta(days=1))
    object.__setattr__(nextday, "exit_ts", BASE + timedelta(days=1, seconds=90))
    d = g.on_alert(nextday)
    assert g.killed is False
    assert g.consecutive_losses == 0
    assert g.realized_krw == 0.0
    assert d.allowed is True


# --------------------------------------------------------------- 수동 정지 파일

def test_manual_stop_file_triggers_kill(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("stop", encoding="utf-8")
    g = _guard(stop_file=str(stop))
    d = g.on_alert(_alert(0))
    assert d.allowed is False
    assert d.killed is True
    assert g.kill_reason == KILL_MANUAL


def test_no_stop_file_no_kill(tmp_path):
    g = _guard(stop_file=str(tmp_path / "absent"))
    assert g.on_alert(_alert(0)).allowed is True


# --------------------------------------------------------------- 체결 가정

def test_touch_vs_through_fill():
    # 진입가 딱 터치(100)만 오는 경우: touch 는 체결, through 는 미체결.
    touch = _guard(fill_basis="touch")
    touch.on_alert(_alert(0))
    touch.on_tick(FOL, BASE + timedelta(seconds=1), 100.0)   # == entry
    touch.on_tick(FOL, BASE + timedelta(seconds=2), 100.6)   # 익절
    assert touch.closed_positions[0].exit_reason == "익절"

    through = _guard(fill_basis="through")
    through.on_alert(_alert(0))
    through.on_tick(FOL, BASE + timedelta(seconds=1), 100.0)  # 관통 아님
    through.on_tick(FOL, BASE + timedelta(seconds=95), 100.6)  # 청산시각 지남
    pos = through.closed_positions[0]
    assert pos.status == "unfilled"
    assert pos.exit_reason == "미체결"
    assert pos.pnl is None


def test_mae_tracked_after_fill():
    g = _guard()
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)   # 체결
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.7)   # 역행
    g.on_tick(FOL, BASE + timedelta(seconds=3), 100.6)  # 익절
    pos = g.closed_positions[0]
    # 99.7/100 - 1 = -0.003 이 최대 역행
    assert pos.mae < -0.0029
    assert pos.mfe > 0.0


# --------------------------------------------------------------- 수동 체결 표시

def test_mark_executed_forces_fill():
    g = _guard(fill_basis="through")
    g.on_alert(_alert(0))
    assert g.mark_executed(FOL, entry=100.0) is True
    # 관통 없이도 수동 체결이라 익절가 도달 시 청산
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.6)
    pos = g.closed_positions[0]
    assert pos.manual is True
    assert pos.exit_reason == "익절"


# --------------------------------------------------------------- 포지션 로그

def test_position_log_records_metrics(tmp_path):
    log = tmp_path / "positions.log"
    sink = PositionLogSink(log, basis="through")
    g = _guard()
    g.set_close_callback(sink.log_close)
    a = _alert(0)
    sink.log_open(a)
    g.on_alert(a)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.5)   # 손절
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "open"
    close = lines[1]
    assert close["event"] == "close"
    assert close["exit_reason"] == "손절"
    assert "mae" in close and "latency_through_s" in close
    assert close["filled"] is True
    assert close["pnl"] < 0
