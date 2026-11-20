"""진입 후 후속주가 안 뜨면 본절 탈출 — **타 트레이더 B B-2**.

> **Trigger 2 (Execution Failure)**: Stock B fails to move upward within 1-2 ticks
> post-entry. -> Action: EXECUTE SCRATCH STOP (매수가 즉시 손절/본절 탈출).

만쥬는 짝꿍이 "키 맞추기 슈팅"을 쏠 때 먹는다. 사고 나서 안 뜨면 그건 슈팅이 아니라
그냥 물린 것이고, −0.5% 손절까지 기다릴 이유가 없다.

⚠️ **시간 기준은 원문에 없다.** "within 1-2 ticks"는 가격 폭이지 시간이 아니다.
몇 초를 기다릴지는 우리가 정한다(`scratch_secs`).
"""
from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import RiskGuard, RiskGuardConfig

BASE = datetime(2026, 8, 31, 9, 30, 0)
FOL = "000002"


def _alert(entry: float = 10_000.0) -> Alert:
    return Alert(
        ts=BASE, leader="000001", follower=FOL,
        leader_surge=0.02, cosurge=0.006, premove=0.0,
        price=entry, bid1=entry, entry=entry,
        take_profit=None, stop_loss=None, disaster_line=None,
        exit_ts=BASE + timedelta(seconds=180), turnover=1e9,
        leader_name="대장", follower_name="후속",
    )


def _guard(**kw) -> RiskGuard:
    base = {"daily_loss_limit": 0.0, "max_consecutive_losses": 0,
            "max_positions": 99, "stop_file": None, "position_notional": 1_000_000.0}
    base.update(kw)
    g = RiskGuard(RiskGuardConfig(**base))
    g.on_alert(_alert())
    g.on_tick(FOL, BASE + timedelta(seconds=1), 9_999)   # 체결
    return g


def _tick(g: RiskGuard, secs: float, px: float) -> None:
    g.on_tick(FOL, BASE + timedelta(seconds=secs), px)


def test_no_lift_within_the_window_is_a_scratch():
    """진입가 위로 1호가도 못 가면 본절 탈출."""
    g = _guard(scratch_ticks=1, scratch_secs=30.0)
    _tick(g, 35, 9_999)                     # 제자리
    assert not g.open_positions
    assert g.closed_positions[-1].exit_reason == "본절탈출"


def test_lifting_one_tick_keeps_the_position():
    """1호가라도 위로 갔으면 살려둔다 — 슈팅이 시작된 것이다."""
    g = _guard(scratch_ticks=1, scratch_secs=30.0)
    _tick(g, 5, 10_020)                     # 진입가 위로 떴다
    _tick(g, 35, 9_999)
    assert g.open_positions, "한 번 떴으면 본절탈출 대상이 아니다"


def test_inside_the_window_waits():
    g = _guard(scratch_ticks=1, scratch_secs=30.0)
    _tick(g, 10, 9_999)
    assert g.open_positions


def test_off_by_default():
    g = _guard()
    _tick(g, 120, 9_999)
    assert g.open_positions


def test_cron_arms_the_scratch_stop(cron_args):
    from scalp_it.cli_pair_detect import build_guard

    g = build_guard(cron_args)
    assert g is not None
    assert g.config.scratch_ticks == 1
    assert g.config.scratch_secs == 30.0
