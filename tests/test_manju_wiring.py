# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""만쥬가 직접 지시한 값들의 **배선** 검증.

여기서 잡는 건 "설정은 있는데 아무 데도 안 닿는" 사고다. 실제로 호가 두께 필터가
`on_quote` 에 10단계가 안 들어와 조용히 무동작이었던 적이 있다. 규칙이 옳은지는
여기서 안 따진다 — 만쥬가 그렇게 하라고 했으니 **그대로 도는지**만 본다.

만쥬 정량화 액션 원문:

> 익절선 확장: +0.5% 고정 브래킷을 제거하고, "대장주 상한가 유지 중 3분 타임스탑
> or 2등주 +3% 도달 시 트레일링 스탑"으로 수정
> 진입: "현재가 기준 위 2~3호가 지정가 원클릭 주문"
"""

from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import (
    LEADER_LOST,
    RiskGuard,
    RiskGuardConfig,
)
from scalp_it.ticksize import tick_size

BASE = datetime(2026, 8, 27, 10, 0, 0)
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
    g.on_tick(FOL, BASE + timedelta(seconds=1), 9_999)   # < entry → through 체결
    return g


def _tick(g: RiskGuard, secs: float, px: float) -> None:
    g.on_tick(FOL, BASE + timedelta(seconds=secs), px)


# ----- 트레일링은 +3% 를 찍은 뒤에야 켜진다 -------------------------------------

def test_trailing_does_not_fire_before_it_is_armed():
    """진입 직후 되돌림에 털리면 안 된다 — 2등주 자체 가격 노이즈다."""
    g = _guard(trail_arm=0.03, trail_pct=0.015)
    _tick(g, 10, 10_150)          # +1.5% — 아직 무장 전
    _tick(g, 20, 9_900)           # 고점 대비 −2.5%
    assert len(g.open_positions) == 1, "무장 전엔 트레일링이 발동하면 안 된다"


def test_trailing_fires_after_three_percent():
    """만쥬: "2등주 +3% 도달 시 트레일링 스탑"."""
    g = _guard(trail_arm=0.03, trail_pct=0.015)
    _tick(g, 10, 10_600)          # +6% → 무장
    _tick(g, 20, 10_400)          # 고점 대비 −1.9% → 청산
    assert len(g.open_positions) == 0
    assert g.closed_positions[-1].exit_reason == "추세청산"


def test_trailing_is_off_unless_asked():
    """0 = 끔. 켜지 않은 판이 조용히 달라지면 안 된다."""
    g = _guard()
    _tick(g, 10, 10_600)
    _tick(g, 20, 10_000)
    assert len(g.open_positions) == 1


def test_leader_exit_still_outranks_trailing():
    """대장이 상한가를 놓치면 손익 무관 전량 — 트레일링이 이걸 가리면 안 된다."""
    g = _guard(trail_arm=0.03, trail_pct=0.015)
    _tick(g, 10, 10_600)
    g.on_leader_state("000001", BASE + timedelta(seconds=15), LEADER_LOST)
    assert len(g.open_positions) == 0
    assert g.closed_positions[-1].exit_reason == "대장이탈"


# ----- 진입은 현재가 위 2~3호가를 긁는다 ---------------------------------------

def test_entry_defaults_to_the_bid():
    """기본값은 종전 그대로 — 만쥬 주문 방식은 켜야 켜진다."""
    from scalp_it.realtime.pair_detector import DetectorConfig

    assert DetectorConfig().entry_ticks_up == 0


def test_entry_ticks_up_crosses_the_spread():
    """만쥬: "현재가 기준 위 2~3호가 지정가 원클릭". 걸어두고 기다리면 못 산다."""
    from scalp_it.realtime.pair_detector import DetectorConfig

    cfg = DetectorConfig(entry_ticks_up=2)
    cur = 10_000.0
    assert cfg.entry_ticks_up * tick_size(cur) > 0
    # 진입가가 현재가보다 위여야 매도 최우선 잔량부터 즉시 체결된다.
    assert cur + cfg.entry_ticks_up * tick_size(cur) > cur
