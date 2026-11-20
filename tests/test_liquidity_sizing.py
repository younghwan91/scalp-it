"""호가가 얇으면 비중을 줄인다 — 만쥬 A-2 Thin Orderbook Restriction.

> - Thick Orderbook (e.g., Large Group Stocks): Normal allocation.
> - Thin Orderbook (e.g., Small Theme Followers): Cap size to prevent slippage
>   (**Max 10-20% of normal size**).

통합 매뉴얼 §1-3 도 같은 말이다 — "대형 그룹주는 비중 투입 가능, 얇은 개별 테마주는
철저히 비중을 축소".

종전 우리 구현은 `max_roundtrip_bp` 로 **배제**만 했다(20bp 넘으면 알림 자체를 버림).
만쥬는 배제가 아니라 **축소**하라고 한다 — 얇아도 기회면 소액으로 잡는다.
"""
import pytest

from scalp_it.realtime.pair_detector import liquidity_size_multiplier


@pytest.mark.parametrize("bp,expected", [
    (0.0, 1.0),      # 충격 0 = 아주 두껍다
    (5.0, 1.0),      # 얇음 문턱 아래 → 평소 비중
    (10.0, 1.0),     # 경계값은 아직 두꺼운 쪽
    (10.1, 0.2),     # 얇음 → 20%
    (19.9, 0.2),
])
def test_thin_book_gets_a_fraction(bp, expected):
    assert liquidity_size_multiplier(bp, thin_bp=10.0, thin_multiplier=0.2) == expected


def test_unknown_depth_is_treated_as_thin():
    """호가를 못 봤으면 얇다고 본다 — 모르면 작게 가는 쪽이 안전하다."""
    assert liquidity_size_multiplier(None, thin_bp=10.0, thin_multiplier=0.2) == 0.2


def test_off_when_threshold_is_zero():
    assert liquidity_size_multiplier(50.0, thin_bp=0.0, thin_multiplier=0.2) == 1.0


def test_guard_shrinks_notional_for_a_thin_follower():
    """가드가 얇은 후속주의 **명목을 줄여** 포지션을 연다 — 배제가 아니라 축소."""
    from datetime import datetime, timedelta

    from scalp_it.realtime.pair_detector import Alert
    from scalp_it.realtime.risk_guard import RiskGuard, RiskGuardConfig

    base = datetime(2026, 8, 31, 9, 30, 0)

    def _alert(rt_bp: float) -> Alert:
        return Alert(
            ts=base, leader="000001", follower="000002",
            leader_surge=0.02, cosurge=0.006, premove=0.0,
            price=10_000.0, bid1=10_000.0, entry=10_000.0,
            take_profit=None, stop_loss=None, disaster_line=None,
            exit_ts=base + timedelta(seconds=180), turnover=1e9,
            leader_name="대장", follower_name="후속", rt_bp=rt_bp,
        )

    cfg = dict(daily_loss_limit=0.0, max_consecutive_losses=0, max_positions=99,
               stop_file=None, position_notional=10_000_000.0,
               thin_book_bp=10.0, thin_book_multiplier=0.2)
    thick = RiskGuard(RiskGuardConfig(**cfg))
    thick.on_alert(_alert(rt_bp=5.0))
    assert thick.open_positions[0].notional == 10_000_000.0

    thin = RiskGuard(RiskGuardConfig(**cfg))
    thin.on_alert(_alert(rt_bp=15.0))
    assert thin.open_positions[0].notional == 2_000_000.0, "얇으면 20%"


def test_cron_arms_liquidity_sizing(cron_args):
    from scalp_it.cli_pair_detect import build_guard

    g = build_guard(cron_args)
    assert g is not None
    assert g.config.thin_book_bp == 10.0
    assert g.config.thin_book_multiplier == 0.2
