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


# ----- 진입 방아쇠는 "상한가 3호가 남은 시점" ---------------------------------

def test_near_ticks_fires_only_next_to_the_limit():
    """만쥬: "대장이 상한가 호가를 3개 남겨둔 3~5초 사이에 매수"."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector, limit_up_price

    prev = 10_000.0
    limit = limit_up_price(prev)
    d = PairDetector([], DetectorConfig(limitup_near_ticks=3), prev_close={"000001": prev})
    kind = d._limitup_kind
    t = tick_size(limit)
    assert kind("000001", BASE, limit - 2 * t) == "near", "3호가 안이면 발동"
    assert kind("000001", BASE, limit - 6 * t) is None, "멀면 발동하면 안 된다"


def test_near_pct_would_have_fired_far_too_early():
    """종전 비율 방아쇠(5%p)는 등락률 +23%대에서 터진다 — 만쥬 시점이 아니다."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector, limit_up_price

    prev = 10_000.0
    limit = limit_up_price(prev)
    d = PairDetector([], DetectorConfig(limitup_near_pct=0.05), prev_close={"000001": prev})
    far = limit * 0.96                      # 상한가에서 4% 아래 = 등락률 +25% 근처
    assert d._limitup_kind("000001", BASE, far) == "near"
    assert far < limit - 6 * tick_size(limit), "만쥬 기준으론 아직 살 자리가 아니다"


# ----- 2등주 서열은 호가두께 → 틱강도 → 가벼운 소형주 --------------------------

def test_follower_priority_defaults_to_rank():
    """켜지 않은 판이 조용히 달라지면 안 된다."""
    from scalp_it.realtime.pair_detector import DetectorConfig

    assert DetectorConfig().follower_priority == "rank"


def test_manju_priority_prefers_the_thicker_book():
    """만쥬 1순위: "1회 주문을 1~2호가 안에서 받아줄 잔량이 깔려 있는가"."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    d = PairDetector([], DetectorConfig(follower_priority="manju", order_krw=10_000_000))
    thin = [(10_000.0, 1)] * 10
    thick = [(10_000.0, 10_000)] * 10
    d._book["THIN"] = (thin, [(10_050.0, 1)] * 10)
    d._book["THICK"] = (thick, [(10_010.0, 10_000)] * 10)
    order = sorted(["THIN", "THICK"], key=lambda c: d._manju_priority(c, BASE))
    assert order[0] == "THICK"


def test_unmeasurable_book_goes_last():
    """두께를 모르는 종목을 두껍다고 볼 근거가 없다."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    d = PairDetector([], DetectorConfig(follower_priority="manju", order_krw=10_000_000))
    d._book["KNOWN"] = ([(10_000.0, 10_000)] * 10, [(10_010.0, 10_000)] * 10)
    order = sorted(["UNKNOWN", "KNOWN"], key=lambda c: d._manju_priority(c, BASE))
    assert order[0] == "KNOWN"


# ----- 낙폭 방어: 정점 대비 꺾이면 덜 건다 -------------------------------------

def _cycle(g: RiskGuard, secs: float, follower: str, exit_px: float) -> None:
    """알림 → 관통 체결 → 지정 가격에서 시간청산 한 사이클."""
    a = _alert()
    a = type(a)(**{**a.__dict__, "ts": BASE + timedelta(seconds=secs),
                   "follower": follower,
                   "exit_ts": BASE + timedelta(seconds=secs + 2)})
    g.on_alert(a)
    g.on_tick(follower, BASE + timedelta(seconds=secs + 1), 9_999)
    g.on_tick(follower, BASE + timedelta(seconds=secs + 3), exit_px)


def test_drawdown_cuts_size_after_giving_back_profit():
    """만쥬: "과거 정점 대비 계좌가 꺾일 때 예수금을 1/3~1/5로 축소"."""
    g = RiskGuard(RiskGuardConfig(
        position_notional=1_000_000, daily_loss_limit=0.0, max_consecutive_losses=0,
        max_positions=99, stop_file=None, drawdown_krw=30_000))
    full = g.effective_notional()
    _cycle(g, 0, "000002", 10_500)      # +5% → 실현 +5만, 정점 갱신
    assert g.effective_notional() == full, "벌고 있는 동안은 그대로"
    _cycle(g, 10, "000003", 9_600)      # −4% → 정점 대비 9만 반납
    assert g.effective_notional() < full
    assert g.effective_notional() == full / 3


def test_drawdown_does_not_unwind_on_recovery():
    """줄인 비중으로 조금 회복했다고 다시 키우면 손실 복구 심리 그대로다."""
    g = RiskGuard(RiskGuardConfig(
        position_notional=1_000_000, daily_loss_limit=0.0, max_consecutive_losses=0,
        max_positions=99, stop_file=None, drawdown_krw=30_000))
    _cycle(g, 0, "000002", 10_500)
    _cycle(g, 10, "000003", 9_600)
    cut = g.effective_notional()
    _cycle(g, 20, "000004", 10_500)     # 회복
    assert g.effective_notional() == cut, "한 번 걸리면 그날은 안 풀린다"


def test_drawdown_is_off_by_default():
    g = RiskGuard(RiskGuardConfig(
        position_notional=1_000_000, daily_loss_limit=0.0, max_consecutive_losses=0,
        max_positions=99, stop_file=None))
    _cycle(g, 0, "000002", 10_500)
    _cycle(g, 10, "000003", 9_000)
    assert g.effective_notional() == 1_000_000


# ----- 쉬는 국면에도 "샀으면 어땠나"는 남긴다 ----------------------------------

def _blocked_guard(**kw) -> RiskGuard:
    base = {"daily_loss_limit": 0.0, "max_consecutive_losses": 0, "max_positions": 99,
            "stop_file": None, "position_notional": 10_000_000.0}
    base.update(kw)
    g = RiskGuard(RiskGuardConfig(**base))
    g.set_size_multiplier(0.0)          # 거래대금 미달 → 만쥬 기준 진입 금지
    return g


def test_blocked_signal_opens_nothing_real():
    """만쥬 규칙대로 **안 산다** — 이건 그림자를 켜도 그대로여야 한다."""
    from scalp_it.realtime.risk_guard import BLOCK_REGIME_ZERO

    for shadow in (False, True):
        g = _blocked_guard(shadow_when_blocked=shadow)
        d = g.on_alert(_alert())
        assert not d.allowed and d.reason == BLOCK_REGIME_ZERO
        assert g.open_positions == []
        assert not d.killed, "국면 차단은 킬이 아니다"


def test_blocked_signal_is_recorded_as_shadow():
    """쉬는 국면이 몇 달이면 표본이 0 이라 규칙 검증조차 안 된다."""
    g = _blocked_guard(shadow_when_blocked=True)
    g.on_alert(_alert())
    assert len(g.shadow_positions) == 1
    assert g.shadow_positions[0].shadow is True
    assert g.shadow_positions[0].notional == 10_000_000.0, \
        "배수 0 을 그대로 곱하면 명목 0 이라 손익이 전부 0 이 된다"


def test_shadow_never_touches_real_accounting():
    """실현손익·승패·연속손절 어디에도 안 닿아야 한다."""
    g = _blocked_guard(shadow_when_blocked=True)
    g.on_alert(_alert())
    _tick(g, 1, 9_999)                  # 그림자 체결
    _tick(g, 200, 9_000)                # 크게 밀림 → 그림자 청산
    assert g.realized_krw == 0.0
    assert g.consecutive_losses == 0
    assert g.losses == 0
    assert not g.killed


def test_shadow_off_by_default():
    g = _blocked_guard()
    g.on_alert(_alert())
    assert g.shadow_positions == []


# ----- 대장이 굳히고 있으면 후속주 가격으로 손절하지 않는다 ---------------------

def _stop_guard(**kw) -> RiskGuard:
    base = {"daily_loss_limit": 0.0, "max_consecutive_losses": 0, "max_positions": 99,
            "stop_file": None, "position_notional": 10_000_000.0}
    base.update(kw)
    g = RiskGuard(RiskGuardConfig(**base))
    a = _alert()
    g.on_alert(type(a)(**{**a.__dict__, "stop_loss": 9_950.0, "disaster_line": 9_700.0}))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 9_999)      # through 체결
    return g


def test_stop_fires_when_leader_state_is_unknown():
    """대장 통보를 한 번도 못 받았으면 종전대로 손절한다 — 하위호환."""
    g = _stop_guard(stop_needs_leader_weak=True)
    _tick(g, 10, 9_940)
    assert len(g.open_positions) == 0
    assert g.closed_positions[-1].exit_reason == "손절"


def test_stop_is_locked_while_the_leader_holds_the_limit():
    """만쥬: "2등주의 가격 지표로 손절하지 말고 대장주 이탈을 트리거로"."""
    from scalp_it.realtime.risk_guard import LEADER_HOLDING

    g = _stop_guard(stop_needs_leader_weak=True)
    g.on_leader_state("000001", BASE + timedelta(seconds=5), LEADER_HOLDING)
    _tick(g, 10, 9_940)                                    # 손절선 아래 — 그래도 버틴다
    assert len(g.open_positions) == 1, "대장이 굳히는 동안은 스프레드 노이즈로 안 나간다"


def test_disaster_line_still_fires_while_the_leader_holds():
    """재난선은 대장 상태와 무관한 진짜 사고 대비다 — 같이 잠기면 안 된다."""
    from scalp_it.realtime.risk_guard import LEADER_HOLDING

    g = _stop_guard(stop_needs_leader_weak=True)
    g.on_leader_state("000001", BASE + timedelta(seconds=5), LEADER_HOLDING)
    _tick(g, 10, 9_600)
    assert len(g.open_positions) == 0
    assert g.closed_positions[-1].exit_reason == "재난선"


def test_leader_losing_the_limit_reopens_the_stop_and_exits():
    """대장이 흔들리면 손익 무관 전량 — 잠금은 그때 풀린다."""
    from scalp_it.realtime.risk_guard import LEADER_HOLDING, LEADER_LOST

    g = _stop_guard(stop_needs_leader_weak=True)
    g.on_leader_state("000001", BASE + timedelta(seconds=5), LEADER_HOLDING)
    _tick(g, 10, 9_940)
    g.on_leader_state("000001", BASE + timedelta(seconds=15), LEADER_LOST)
    assert len(g.open_positions) == 0
    assert g.closed_positions[-1].exit_reason == "대장이탈"


def test_lock_is_off_by_default():
    g = _stop_guard()
    g.on_leader_state("000001", BASE + timedelta(seconds=5), "holding")
    _tick(g, 10, 9_940)
    assert g.closed_positions[-1].exit_reason == "손절"


# ----- 전일종가가 오늘 것이 아니면 상한가를 계산하지 않는다 --------------------

def test_limit_up_is_none_when_prev_close_is_stale():
    """2026-08-28 체리부로 재현.

    일봉 전일종가가 670 으로 6일째 고정인데 그날 체결은 1,574~1,639 였다(거래정지 후
    재개로 보인다). 그 값으로 상한가를 구하면 871 이 나와 **모든 틱이 "상한가 위"로
    읽히고** 방아쇠가 통째로 엉뚱한 자리에서 터진다. 하루 변동폭은 ±30% 로 막혀
    있으므로 앞뒤가 안 맞으면 판단을 안 하는 게 맞다.
    """
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    d = PairDetector([], DetectorConfig(), prev_close={"066360": 670.0},
                     sectors={"066360": "음식료"}, universe={"066360"})
    assert d.limit_up("066360") == 871, "틱을 보기 전엔 그대로 계산한다"
    d.on_tick("066360", BASE, 1_600.0)
    assert d.limit_up("066360") is None, "오늘 가격과 앞뒤가 안 맞으면 판단 보류"


def test_limit_up_survives_a_normal_limit_up_move():
    """상한가(+30%)까지 올라도 계산은 살아 있어야 한다 — 그게 방아쇠의 자리다."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    d = PairDetector([], DetectorConfig(), prev_close={"000002": 10_000.0},
                     sectors={"000002": "전기/전자"}, universe={"000002"})
    lim = d.limit_up("000002")
    d.on_tick("000002", BASE, lim)
    assert d.limit_up("000002") == lim, "정상 상한가는 막으면 안 된다"


def test_limit_up_survives_a_lower_limit_move():
    """하한가(−30%)도 정상 범위다."""
    from scalp_it.realtime.pair_detector import DetectorConfig, PairDetector

    d = PairDetector([], DetectorConfig(), prev_close={"000002": 10_000.0},
                     sectors={"000002": "전기/전자"}, universe={"000002"})
    d.on_tick("000002", BASE, 7_000.0)
    assert d.limit_up("000002") is not None
