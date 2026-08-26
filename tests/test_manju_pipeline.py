# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""만쥬 경로 통합 테스트 — 감지기 → 가드 → 실행기가 끝에서 끝까지 도는가.

단위 테스트는 조각마다 있지만, **만쥬 방식 전체가 한 줄로 이어지는지**는 아무도
확인하지 않았다. 내일 장에서 그게 안 돌면 하루를 통째로 날린다. 그래서 합성 틱으로
다음 사슬을 통째로 태운다:

    유니버스·섹터·전일종가 주입
      → 대장이 상한가 근처로 치고 올라감(방아쇠)
      → 같은 섹터 등락률 2~3위를 후속으로 선정
      → 가드·실행기가 포지션을 연다
      → 대장이 상한가를 놓침
      → drain_leader_events() → on_leader_state() → **손익 무관 즉시 청산**

마지막 두 줄이 만쥬 §5-3 의 핵심이고 우리에게 없던 축이다. 이 파일이 깨지면
'만쥬 방식으로 돌고 있다'는 말을 할 수 없다.

DB·소켓 없음. 합성 틱만.
"""

from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import (
    DetectorConfig,
    PairDetector,
    limit_up_price,
)
from scalp_it.realtime.risk_guard import (
    LEADER_LOST,
    RiskGuard,
    RiskGuardConfig,
)

BASE = datetime(2026, 8, 26, 9, 30, 0)

LEAD = "100000"       # 대장 — 전일종가 10,000 → 상한가 13,000
FOL_2ND = "200000"    # 같은 섹터 2위
FOL_3RD = "300000"    # 같은 섹터 3위
OTHER = "900000"      # 다른 섹터 (후속 후보가 되면 안 된다)

PREV_CLOSE = {LEAD: 10_000, FOL_2ND: 10_000, FOL_3RD: 10_000, OTHER: 10_000}
SECTORS = {LEAD: "전기/전자", FOL_2ND: "전기/전자", FOL_3RD: "전기/전자",
           OTHER: "금융"}


def _manju_config(**kw):
    """만쥬 설정 — 내일 cron 이 쓰는 값과 같은 뼈대.

    방아쇠 외 관문(동반급등·프리무브·매수비율·호가단위)은 이 테스트의 관심사가
    아니라 열어 둔다. 그것들은 각자 단위 테스트가 따로 있다.
    """
    base = {
        "limitup_near_pct": 0.05,      # 상한가까지 5%p 이내 = 등락률 +25%
        "limitup_require_surge": True,  # 정체 종목 배제(원본 presurge 의 모멘텀 조건)
        "surge_ret": 0.015,
        "surge_window": 30.0,
        "take_profit": None,            # 익절 상한 없음
        "stop_loss": -0.005,            # 약손절 −0.5%
        "time_exit": 240.0,             # 타임아웃 4분
        "cosurge_min": 0.0,
        "premove_max": 1.0,
        "min_buy_ratio": 0.0,
        "min_sl_ticks": 0.0,
        "max_roundtrip_bp": 0.0,        # 호가 두께는 10호가 주입이 필요 — 별도 테스트
        "cooldown": 0.0,
    }
    base.update(kw)
    return DetectorConfig.stable(**base)


def _detector(**kw):
    return PairDetector(
        [], _manju_config(**kw),
        universe=list(PREV_CLOSE), sectors=SECTORS, prev_close=PREV_CLOSE,
    )


def _feed(det, code, series, *, vol=1000.0, side=1):
    """``[(초, 가격)]`` 을 흘리고 발생한 알림을 모은다."""
    out = []
    for sec, px in series:
        out += det.on_tick(code, BASE + timedelta(seconds=sec), px,
                           bid1=px, volume=vol, side=side)
    return out


def _flat(lo, hi, px):
    return [(s, px) for s in range(lo, hi + 1)]


def test_limit_up_price_matches_repo_convention():
    """상한가 = 전일종가×1.3 을 호가단위로 내림. 검증 스크립트와 같은 값이어야 한다."""
    assert limit_up_price(10_000) == 13_000
    assert limit_up_price(539) == 700          # 539×1.3=700.7 → 1원 단위 내림
    assert limit_up_price(1_000) == 1_300


def test_manju_chain_entry_to_leader_exit():
    """만쥬 사슬 전체 — 방아쇠 → 섹터 후속 선정 → 진입 → 대장이탈 → 즉시 청산."""
    det = _detector()
    guard = RiskGuard(RiskGuardConfig(
        daily_loss_limit=0.0, max_consecutive_losses=0, max_positions=99,
        stop_file=None, leader_exit=True))

    # 워밍업 — 후속 후보들은 60초치 히스토리가 있어야 판정이 돈다(프리무브 창).
    for code, px in ((FOL_2ND, 11_500), (FOL_3RD, 11_000), (OTHER, 12_900)):
        _feed(det, code, _flat(-90, 40, px))

    # 대장: 잠잠하다가 상한가(13,000) 근처로 치고 올라간다.
    #   12,350 = +23.5% → 상한가까지 5%p 밖 → 아직 발동 안 함
    #   12,500 = +25.0% → 5%p 이내 진입 + 그 구간 서지 → 발동
    alerts = _feed(det, LEAD, _flat(-90, 20, 12_000) + [(25, 12_350), (30, 12_500)])

    assert alerts, "상한가 근접 방아쇠가 안 터졌다"
    a = alerts[-1]
    assert a.leader == LEAD
    assert a.follower in (FOL_2ND, FOL_3RD), "후속주는 같은 섹터에서 골라야 한다"
    assert a.follower != OTHER, "다른 섹터가 후속으로 뽑히면 만쥬 방식이 아니다"
    assert a.take_profit is None, "만쥬 손익 구조 — 익절 상한이 없어야 한다"

    # 가드가 포지션을 연다.
    assert guard.on_alert(a).allowed
    assert len(guard.open_positions) == 1
    fol = a.follower

    # 후속주가 체결되게 진입가를 관통시킨다(through 회계 기준).
    guard.on_tick(fol, BASE + timedelta(seconds=35), a.entry - 10)
    pos = guard.open_positions[0]
    assert pos.is_filled(pos.basis), "후속주가 체결돼야 청산 규칙이 돈다"

    # 대장이 상한가를 놓친다 → 감지기가 이탈로 판정해야 한다.
    _feed(det, LEAD, [(40, 13_000), (45, 13_000), (50, 12_400)])
    events = det.drain_leader_events()
    lost = [e for e in events if e[2] == LEADER_LOST and e[0] == LEAD]
    assert lost, f"대장 이탈이 통보되지 않았다 (큐={events})"

    # 통보를 먹이면 손익 무관 즉시 청산 — 이것이 만쥬 §5-3 의 핵심이다.
    code, ts, state = lost[0]
    guard.on_leader_state(code, ts, state)
    assert not guard.open_positions, "대장이 이탈했는데 포지션이 남아 있다"
    closed = [p for p in guard.closed_positions if not p.shadow]
    assert closed and closed[-1].exit_reason == "대장이탈"


def test_leader_holding_does_not_close_position():
    """대장이 상한가를 유지하면 청산되지 않는다 — 이탈만 방아쇠다."""
    det = _detector()
    guard = RiskGuard(RiskGuardConfig(
        daily_loss_limit=0.0, max_consecutive_losses=0, max_positions=99,
        stop_file=None, leader_exit=True))
    for code, px in ((FOL_2ND, 11_500), (FOL_3RD, 11_000)):
        _feed(det, code, _flat(-90, 60, px))
    alerts = _feed(det, LEAD, _flat(-90, 20, 12_000) + [(25, 12_350), (30, 12_500)])
    assert alerts
    guard.on_alert(alerts[-1])
    guard.on_tick(alerts[-1].follower, BASE + timedelta(seconds=35),
                  alerts[-1].entry - 10)

    # 대장이 상한가에 붙어 유지 — 이탈 통보가 있으면 안 된다.
    _feed(det, LEAD, [(40, 13_000), (50, 13_000), (60, 13_000)])
    lost = [e for e in det.drain_leader_events() if e[2] == LEADER_LOST]
    assert not lost, "상한가를 유지 중인데 이탈로 통보됐다"
    assert len(guard.open_positions) == 1


def test_unknown_prev_close_never_triggers():
    """전일종가를 모르는 종목은 상한가를 못 구한다 → 방아쇠 없음(억지 추정 금지)."""
    det = PairDetector(
        [], _manju_config(),
        universe=[LEAD, FOL_2ND], sectors=SECTORS, prev_close={FOL_2ND: 10_000},
    )
    _feed(det, FOL_2ND, _flat(-90, 40, 11_500))
    alerts = _feed(det, LEAD, _flat(-90, 20, 12_000) + [(25, 12_350), (30, 12_500)])
    assert alerts == []
    assert det.limit_up(LEAD) is None
