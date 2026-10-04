# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""RiskGuard 킬 판정 ≡ 코어 ``KillSwitch`` 판정 — 위임 경로 시나리오 회귀.

1단계(그림자)에서 RiskGuard 자체 판정과 코어 ``krx_quant_core.risk.KillSwitch`` 를 나란히
굴려 매 단계 일치를 확인했고, 2단계에서 RiskGuard 의 킬 상태가 KillSwitch 를 비추는
프로퍼티가 됐다. 이 파일은 그 시나리오(알림·틱·청산·분할·정지파일·날짜변경·이익반납 —
``test_pair_risk_guard.py``·``test_profit_giveback.py`` 의 킬 케이스 재사용)를 그대로 돌려
**매 단계** 가드가 외부에 보이는 상태와 KillSwitch 내부 상태가 같고, 킬 시점·사유가
원본 RiskGuard 규칙 그대로인지 고정한다. 코어 KillSwitch 의 의미가 바뀌면 여기서 먼저 깨진다.
"""

import json
from datetime import datetime, timedelta

from krx_quant_core.risk import KillSwitch

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import (
    KILL_CONSECUTIVE,
    KILL_DAILY_LOSS,
    KILL_GIVEBACK,
    KILL_MANUAL,
    LEADER_LOST,
    PositionLogSink,
    RiskGuard,
    RiskGuardConfig,
)

BASE = datetime(2026, 8, 24, 9, 10, 0)
FOL = "000002"


def _alert(secs: float = 0.0, *, entry: float = 100.0, tp: float | None = 100.5,
           sl: float | None = 99.6, disaster: float | None = None,
           hold: float = 90.0, follower: str = FOL, days: int = 0) -> Alert:
    ts = BASE + timedelta(days=days, seconds=secs)
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


def _same(g: RiskGuard) -> None:
    """가드가 보이는 킬 상태와 KillSwitch 내부 상태가 같은지 — 한 단계 뒤마다 부른다."""
    ks: KillSwitch = g.killswitch
    assert (g.killed, g.kill_reason) == (ks.killed, ks.kill_reason)
    assert abs(g.realized_krw - ks.realized_krw) < 1e-6
    assert abs(g.peak_realized_krw - ks.peak_realized_krw) < 1e-6
    assert g.consecutive_losses == ks.consecutive_losses
    assert g.day == ks.day
    # 승패 집계는 가드가 따로 세지만 같은 is_loss 규칙이라 코어 집계와도 같아야 한다.
    assert (g.wins, g.losses) == (ks.wins, ks.losses)


def _loss(g: RiskGuard, secs: float, follower: str = FOL, days: int = 0) -> None:
    """알림→관통 체결→손절 관통(99.5 = −0.5%)."""
    t0 = BASE + timedelta(days=days, seconds=secs)
    g.on_alert(_alert(secs, follower=follower, days=days)); _same(g)
    g.on_tick(follower, t0 + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(follower, t0 + timedelta(seconds=2), 99.5); _same(g)


def _win(g: RiskGuard, secs: float, follower: str = FOL, days: int = 0) -> None:
    t0 = BASE + timedelta(days=days, seconds=secs)
    g.on_alert(_alert(secs, follower=follower, days=days)); _same(g)
    g.on_tick(follower, t0 + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(follower, t0 + timedelta(seconds=2), 100.6); _same(g)


# --------------------------------------------------------------- 구성 매핑

def test_config_mapping_is_explicit():
    g = RiskGuard(RiskGuardConfig(daily_loss_limit=1400.0, daily_loss_limit_pct=0.002,
                                  daily_capital=2_000_000.0, max_consecutive_losses=2,
                                  profit_giveback_ratio=0.5, profit_giveback_min_krw=10_000,
                                  stop_file="data/x_STOP"))
    c = g.killswitch.config
    assert c.loss_limit_krw() == g.config.loss_limit_krw() == 4_000.0
    assert c.max_consecutive_losses == 2
    assert (c.profit_giveback_ratio, c.profit_giveback_min_krw) == (0.5, 10_000)
    assert c.stop_file == "data/x_STOP" and c.sticky_stop_file is True


def test_default_config_maps_defaults_not_core_defaults():
    g = RiskGuard(RiskGuardConfig())
    c = g.killswitch.config
    assert c.loss_limit_krw() == 50_000.0
    assert c.max_consecutive_losses == 3
    assert c.stop_file == "data/pair_STOP"


# --------------------------------------------------------------- 일일 손실한도

def test_daily_loss_limit_parity():
    g = _guard(daily_loss_limit=1400.0)
    for i in range(3):
        _loss(g, i * 100)
    assert g.killed and g.kill_reason == KILL_DAILY_LOSS
    d = g.on_alert(_alert(500)); _same(g)
    assert d.allowed is False and d.banner is not None
    assert g.on_alert(_alert(510)).banner is None; _same(g)


def test_pct_limit_parity():
    g = _guard(daily_loss_limit=50_000.0, daily_loss_limit_pct=0.001, daily_capital=1_000_000)
    _loss(g, 0)
    assert not g.killed
    _loss(g, 100)
    assert g.killed and g.kill_reason == KILL_DAILY_LOSS   # 한도 1000원, 2회째


def test_kill_is_sticky_after_recovery_parity():
    g = _guard(daily_loss_limit=1000.0)
    _loss(g, 0)
    _loss(g, 100)
    assert g.killed
    # 킬 뒤 들고 있던 포지션은 없다 — 다음 알림은 막힌다. 그림자도 같이 막혀야 한다.
    for i in range(3):
        g.on_alert(_alert(300 + i)); _same(g)
    assert g.killed and g.killswitch.killed


# --------------------------------------------------------------- 연속 손절

def test_consecutive_losses_parity():
    g = _guard(max_consecutive_losses=2)
    _loss(g, 0)
    assert not g.killed
    _loss(g, 100)
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE


def test_win_resets_counter_parity():
    g = _guard(max_consecutive_losses=3)
    _loss(g, 0)
    _loss(g, 100)
    assert g.consecutive_losses == 2
    _win(g, 200)
    assert g.consecutive_losses == 0 and not g.killed


def test_consecutive_before_daily_loss_parity():
    g = _guard(max_consecutive_losses=2, daily_loss_limit=1000.0)
    _loss(g, 0)
    _loss(g, 100)
    assert g.kill_reason == KILL_CONSECUTIVE


def test_leader_lost_loss_parity():
    g = _guard(max_consecutive_losses=1)
    a = _alert(0, sl=None)
    g.on_alert(a); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.7); _same(g)
    g.on_leader_state("000001", BASE + timedelta(seconds=3), LEADER_LOST); _same(g)
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE


def test_unfilled_and_time_exit_parity():
    """미체결은 기록이 아니고, 시간청산은 마지막 가격으로 기록 — 둘 다 그림자와 같이."""
    g = _guard(max_consecutive_losses=1, cost_rate=0.001)
    g.on_alert(_alert(0, tp=None, sl=None, hold=10.0)); _same(g)
    g.flush_expired(BASE + timedelta(seconds=11)); _same(g)          # 미체결 → 승패 없음
    assert g.consecutive_losses == 0 and g.closed_positions[-1].status == "unfilled"
    g.on_alert(_alert(20, tp=None, sl=None, hold=10.0)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=21), 99.9); _same(g)      # 체결
    g.on_tick(FOL, BASE + timedelta(seconds=22), 100.0); _same(g)
    g.flush_expired(BASE + timedelta(seconds=31)); _same(g)          # 시간청산: 비용 탓에 손실
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE


# --------------------------------------------------------------- 분할 매도

def test_partial_then_stop_counts_total_outcome_parity():
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002, max_consecutive_losses=1)
    g.on_alert(_alert(0)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25); _same(g)   # 1차 분할 +125
    assert g.open_positions[0].partial_done
    g.on_tick(FOL, BASE + timedelta(seconds=3), 99.5); _same(g)     # 잔량 손절 −250
    assert abs(g.realized_krw - (125.0 - 250.0)) < 1e-6
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE


def test_partial_then_small_remainder_loss_is_a_win_parity():
    """1차 익절 뒤 잔량이 살짝 밀린 건은 **합계** 기준으로 승 — 그림자도 is_loss 로 따라야 한다."""
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002, max_consecutive_losses=1,
               cost_rate=0.001)
    g.on_alert(_alert(0, tp=None, sl=None, hold=10.0)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25); _same(g)   # 분할 +75(비용 차감)
    g.on_tick(FOL, BASE + timedelta(seconds=3), 100.05); _same(g)
    g.flush_expired(BASE + timedelta(seconds=11)); _same(g)         # 잔량 시간청산 −25
    done = g.closed_positions[-1]
    assert done.pnl_krw < 0 < done.partial_pnl_krw + done.pnl_krw
    assert not g.killed and g.consecutive_losses == 0 and g.wins == 1


def test_partial_realized_does_not_touch_streak_parity():
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002, max_consecutive_losses=2,
               daily_loss_limit=600.0)
    _loss(g, 0)                                                      # −500, 연속 1
    g.on_alert(_alert(100, tp=None)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=101), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=102), 100.25); _same(g)  # 분할 +125 → −375
    assert g.consecutive_losses == 1 and not g.killed
    g.on_tick(FOL, BASE + timedelta(seconds=103), 99.5); _same(g)    # 잔량 −250 → −625 ≤ −600
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE            # 연속이 손실한도보다 먼저


# --------------------------------------------------------------- 일일 리셋

def test_day_change_resets_both_parity():
    g = _guard(max_consecutive_losses=2)
    _loss(g, 0)
    _loss(g, 100)
    assert g.killed
    d = g.on_alert(_alert(0, days=1, follower="000009")); _same(g)
    assert d.allowed and not g.killed and g.realized_krw == 0.0
    _loss(g, 100, days=1)
    assert g.consecutive_losses == 1 and not g.killed


def test_day_change_seen_on_tick_only_parity():
    """날짜 변경을 **틱**이 먼저 본다 — 알림 없이도 그림자가 같이 리셋돼야 status 가 안 썩는다."""
    g = _guard(max_consecutive_losses=1)
    _loss(g, 0)
    assert g.killed
    g.on_tick("999999", BASE + timedelta(days=1), 100.0); _same(g)
    assert not g.killed and g.killswitch.day == (BASE + timedelta(days=1)).date()
    g.flush_expired(BASE + timedelta(days=2)); _same(g)


def test_first_observation_is_not_a_reset_parity():
    g = _guard()
    g.force_stop(); _same(g)
    d = g.on_alert(_alert(0)); _same(g)
    assert d.allowed is False and g.kill_reason == KILL_MANUAL


# --------------------------------------------------------------- 수동 정지

def test_stop_file_parity(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("stop", encoding="utf-8")
    g = _guard(stop_file=str(stop))
    d = g.on_alert(_alert(0)); _same(g)
    assert d.allowed is False and g.kill_reason == KILL_MANUAL


def test_absent_stop_file_parity(tmp_path):
    g = _guard(stop_file=str(tmp_path / "absent"))
    assert g.on_alert(_alert(0)).allowed; _same(g)


def test_stop_file_appearing_mid_day_is_sticky_parity(tmp_path):
    stop = tmp_path / "pair_STOP"
    g = _guard(stop_file=str(stop), max_consecutive_losses=1)
    assert g.on_alert(_alert(0)).allowed; _same(g)
    stop.touch()
    g.on_alert(_alert(10, follower="000003")); _same(g)
    assert g.kill_reason == KILL_MANUAL
    stop.unlink()
    g.on_alert(_alert(20, follower="000004")); _same(g)
    assert g.killed and g.kill_reason == KILL_MANUAL                # 지워도 그날 고정
    # 들고 있던 포지션의 손절은 그대로 — 사유는 바뀌지 않는다.
    g.on_tick(FOL, BASE + timedelta(seconds=30), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=31), 99.5); _same(g)
    assert g.kill_reason == KILL_MANUAL
    assert g.on_alert(_alert(0, days=1)).allowed; _same(g)          # 다음 날 해제


def test_stop_file_beats_consecutive_parity(tmp_path):
    stop = tmp_path / "STOP"
    g = _guard(stop_file=str(stop), max_consecutive_losses=1)
    g.on_alert(_alert(0)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9); _same(g)
    stop.touch()
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.5); _same(g)
    assert g.kill_reason == KILL_MANUAL


def test_force_stop_parity():
    g = _guard(max_consecutive_losses=1)
    _loss(g, 0)
    assert g.kill_reason == KILL_CONSECUTIVE
    g.force_stop(); _same(g)
    assert g.kill_reason == KILL_MANUAL                             # 이미 킬이어도 사유 덮음
    g.on_alert(_alert(50)); _same(g)
    g2 = _guard()
    g2.on_alert(_alert(0)); _same(g2)
    g2.force_stop(); _same(g2)
    assert g2.on_alert(_alert(5)).killed; _same(g2)


# --------------------------------------------------------------- 이익 반납(타 트레이더 B B-3)

def _gb(**kw) -> RiskGuard:
    # 익절 +0.5% × 1천만 = +50,000 → 정점. 배수로 손실 크기를 조절한다.
    return _guard(profit_giveback_ratio=0.5, profit_giveback_min_krw=10_000,
                  position_notional=10_000_000.0, **kw)


def test_half_giveback_parity():
    g = _gb()
    _win(g, 0)
    assert abs(g.peak_realized_krw - 50_000) < 1e-6
    g.set_size_multiplier(0.5)
    _loss(g, 100)                                                   # −25,000 → 정확히 절반
    assert g.killed and g.kill_reason == KILL_GIVEBACK


def test_shallow_giveback_parity():
    g = _gb()
    _win(g, 0)
    g.set_size_multiplier(0.4)
    _loss(g, 100)                                                   # −20,000 → 60% 남음
    assert not g.killed


def test_small_peak_ignored_parity():
    g = _gb()
    g.set_size_multiplier(0.1)                                      # 정점 5,000 < 문턱
    _win(g, 0)
    _loss(g, 100)
    assert not g.killed and abs(g.realized_krw) < 1e-6


def test_giveback_before_stop_file_and_consecutive_parity(tmp_path):
    stop = tmp_path / "STOP"
    g = _gb(stop_file=str(stop), max_consecutive_losses=1)
    _win(g, 0)
    g.set_size_multiplier(0.6)
    g.on_alert(_alert(100)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=101), 99.9); _same(g)
    stop.touch()                                                    # 포지션 든 채로 정지 파일
    g.on_tick(FOL, BASE + timedelta(seconds=102), 99.5); _same(g)   # −30,000 → 40% 남음
    assert g.kill_reason == KILL_GIVEBACK


def test_giveback_via_partial_parity():
    """분할 익절로 정점을 만들고 잔량 손절로 반납 — record_realized 경로도 정점을 갱신한다."""
    g = _gb(partial_exit_ratio=0.5, partial_exit_gain=0.002, max_consecutive_losses=5)
    g.on_alert(_alert(0, tp=None, sl=None, hold=10.0)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25); _same(g)   # +12,500 정점
    assert abs(g.peak_realized_krw - 12_500) < 1e-6
    g.on_tick(FOL, BASE + timedelta(seconds=3), 99.8); _same(g)
    g.flush_expired(BASE + timedelta(seconds=11)); _same(g)         # 잔량 −10,000 → 2,500 ≤ 6,250
    assert g.killed and g.kill_reason == KILL_GIVEBACK


# --------------------------------------------------------------- 긴 하루 + 로그

def test_long_session_stays_in_sync(tmp_path):
    """sink 를 붙이고 여러 날을 굴려도 매 단계 일치 + 로그 이벤트 종류가 그대로."""
    log = tmp_path / "positions.log"
    stop = tmp_path / "STOP"
    sink = PositionLogSink(log, basis="through")
    # 손실한도 900원 = 손절 2회(−1000)에 닿아 연속손절 3회보다 먼저 온다.
    g = _guard(daily_loss_limit=900.0, max_consecutive_losses=3, stop_file=str(stop),
               partial_exit_ratio=0.5, partial_exit_gain=0.002,
               profit_giveback_ratio=0.5, profit_giveback_min_krw=200.0)
    g.attach_sink(sink)
    # 1일차: 익절·분할·손절 뒤 이익반납 킬
    _win(g, 0)
    g.on_alert(_alert(100, tp=None)); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=101), 99.9); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=102), 100.25); _same(g)
    g.on_tick(FOL, BASE + timedelta(seconds=103), 99.5); _same(g)
    _loss(g, 200)
    assert g.killed and g.kill_reason == KILL_GIVEBACK
    g.on_alert(_alert(300)); _same(g)
    # 2일차: 작은 비중(손절 −100씩)으로 연속손절 킬 → 수동 정지 덮어쓰기
    g.set_size_multiplier(0.2)
    for i in range(3):
        _loss(g, i * 100, days=1)
    assert g.kill_reason == KILL_CONSECUTIVE
    g.force_stop(); _same(g)
    # 3일차: 정지 파일
    stop.touch()
    assert g.on_alert(_alert(0, days=2)).killed; _same(g)
    stop.unlink()
    # 4일차: 손실한도
    g.set_size_multiplier(1.0)
    for i in range(2):
        _loss(g, i * 100, days=3)
    assert g.kill_reason == KILL_DAILY_LOSS
    g.flush_expired(BASE + timedelta(days=4)); _same(g)
    events = [json.loads(x)["event"] for x in log.read_text(encoding="utf-8").splitlines()]
    # open/killswitch 는 CLI 가 쓴다 — 가드가 직접 남기는 건 청산·분할·미체결뿐.
    assert {"close", "partial"} <= set(events) <= {"close", "partial", "unfilled"}


def test_direct_state_assignment_drives_core():
    """``g.realized_krw = …`` + ``_refresh_kill()`` 관례(test_profit_giveback)가 코어까지 간다."""
    g = _guard(profit_giveback_ratio=0.5, profit_giveback_min_krw=10_000)
    g.realized_krw = 100_000
    g._refresh_kill(); _same(g)
    assert g.killswitch.peak_realized_krw == 100_000 and g.day is None
    g.realized_krw = 50_000
    g._refresh_kill(); _same(g)
    assert g.kill_reason == KILL_GIVEBACK
    # 그 뒤 첫 날짜 관측은 리셋이 아니다 — 킬이 살아 있어야 한다.
    assert g.on_alert(_alert(0)).killed; _same(g)
