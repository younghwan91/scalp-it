"""대장이 **더 못 올라가거나 하락하는 순간** 후속주를 판다(비공개 출처 — 공개본에서 뺐다).

종전 구현은 **A 가 상한가를 한 번 잠근 뒤 풀릴 때**만 청산을 냈다. 우리가 사는 시점은
A 가 상한가 3호가를 남긴 때라, A 가 상한가에 못 닿고 주저앉는 경우가 그 규칙에
아예 없었다. 그때 B 는 자기 가격(−0.5% 손절)이나 180초 타임스탑으로만 나왔다 —
후속주 가격 지표로 손절하는 방식이다.
"""
from datetime import datetime, timedelta

import pytest

from scalp_it.realtime.pair_detector import (
    LEADER_LOST,
    LOST_FADE,
    LOST_STALL,
    DetectorConfig,
    PairDetector,
)

BASE = datetime(2026, 8, 31, 9, 30, 0)
PREV = 10_000          # 전일종가 → 상한가 13,000, 호가단위 10원


def _det(**kw) -> PairDetector:
    """운영과 같은 **동적 모드**로 만든다 — 유니버스와 섹터가 둘 다 있어야 성립한다."""
    cfg = DetectorConfig(leader_stall_secs=10.0, leader_drop_ticks=3, **kw)
    return _build(cfg)


def _build(cfg: DetectorConfig) -> PairDetector:
    d = PairDetector(
        pairs=[], config=cfg,
        universe=["LEADER", "FOLLOWER"],
        sectors={"LEADER": "전기전자", "FOLLOWER": "전기전자"},
        prev_close={"LEADER": PREV, "FOLLOWER": PREV},
    )
    return d


def _tick(d: PairDetector, secs: float, price: float) -> None:
    d.on_tick("LEADER", BASE + timedelta(seconds=secs), price)


def _reasons(d: PairDetector) -> list[str]:
    return [ev.reason for ev in d.drain_leader_events() if ev[2] == LEADER_LOST]


def test_leader_that_stops_climbing_triggers_exit_without_ever_locking():
    """상한가를 한 번도 못 찍어도, 오름이 멈추면 이탈이다."""
    d = _det()
    _tick(d, 0, 12_900)      # 방아쇠 구간 진입(상한가 13,000 의 10호가 아래)
    _tick(d, 3, 12_950)      # 아직 오르는 중
    _tick(d, 20, 12_950)     # 17초째 신고가 없음 → 정체
    assert LOST_STALL in _reasons(d)


def test_leader_that_slips_from_its_high_triggers_exit():
    """고점 대비 3호가 밀리면 하락이다 — 상한가와 무관하다."""
    d = _det()
    _tick(d, 0, 12_900)
    _tick(d, 2, 12_950)      # 고점
    _tick(d, 4, 12_920)      # 3호가(30원) 아래 → 하락
    assert LOST_FADE in _reasons(d)


def test_still_climbing_is_not_an_exit():
    """계속 오르면 안 판다 — 만쥬가 먹는 오버슈팅을 잘라먹지 않는다."""
    d = _det()
    for i, p in enumerate([12_900, 12_930, 12_960, 12_990]):
        _tick(d, i * 3, p)
    assert _reasons(d) == []


def test_exit_fires_only_once():
    """같은 이탈을 두 번 통보하면 가드·실행기가 중복 청산을 시도한다."""
    d = _det()
    _tick(d, 0, 12_900)
    _tick(d, 2, 12_950)
    _tick(d, 4, 12_920)
    _tick(d, 6, 12_900)
    assert len(_reasons(d)) == 1


@pytest.mark.parametrize("stall,drop", [(0.0, 0), (0.0, 3), (10.0, 0)])
def test_both_knobs_off_keeps_old_behaviour(stall, drop):
    """수치가 0 이면 그 조건은 끈다 — 만쥬가 정량 기준을 안 줘서 우리 임의값이다."""
    d = _build(DetectorConfig(leader_stall_secs=stall, leader_drop_ticks=drop))
    _tick(d, 0, 12_900)
    _tick(d, 2, 12_950)
    _tick(d, 60, 12_800)
    got = _reasons(d)
    assert LOST_STALL not in got or stall > 0
    assert LOST_FADE not in got or drop > 0


def test_cron_turns_the_fade_exit_on(cron_args):
    """cron 이 실제로 켜 두는지 — 플래그가 있는데 안 닿은 사고가 여럿 있었다."""
    from scalp_it.cli_pair_detect import build_config

    cfg = build_config(cron_args)
    assert cfg.leader_stall_secs == 15.0
    assert cfg.leader_drop_ticks == 1, "지시서 §3 'ticks down by 1 tick'"


def test_wall_window_matches_manju_three_seconds(cron_args):
    """벽 감소 관측 창은 **3초**다(비공개 출처).

    종전 기본값은 30초였다 — 10배 긴 창이라 3초 만에 확 깎여도 30초 전 피크와
    비교하니 훨씬 둔감하다. 즉시 반응해야 할 신호가 늦게 돈다. 코드에도
    문서에도 왜 30초인지 이유가 없었고 CLI 플래그조차 없어 바꿀 수도 없었다
    (2026-08-30 파라미터 전수 감사에서 발견).
    """
    from scalp_it.cli_pair_detect import build_config

    assert build_config(cron_args).lock_wall_window == 3.0


def test_locked_leader_is_never_stalled_out():
    """**Phase 2(상한가 안착) 에서는 정체 판정을 하지 않는다.**

    만쥬 A2(2026-08-30) — 청산은 상태 머신이다.
      Phase 1(안착 전): "매수 체결이 멈추거나 1~2호가라도 밀리는 순간" 즉시 청산
      Phase 2(안착 후): "잔량 급감·풀림, VI 갭하락, 매도 폭탄" 즉시 청산

    상한가에 잠기면 가격이 상한가에 붙어 움직이지 않는다 — 신고가가 안 나온다.
    Phase 1 규칙을 그대로 돌리면 **굳힌 대장을 15초 만에 이탈로 찍어** 이길 거래를
    전부 잘라먹는다. Phase 2 의 방어는 벽 감소(`lock_wall_decay`)와 매도 폭탄
    (`dump_krw`)이 맡는다.
    """
    d = _det()
    _tick(d, 0, 12_900)
    _tick(d, 2, 13_000)      # 상한가 안착 → Phase 2
    for i in range(3, 60, 3):
        _tick(d, i, 13_000)  # 굳히는 중 — 신고가는 안 나온다
    assert LOST_STALL not in _reasons(d)
    assert LOST_FADE not in _reasons(d)


def test_phase1_uses_manju_one_to_two_ticks():
    """Phase 1 하락 문턱은 만쥬의 **1~2호가**다 — 3호가는 우리가 늦게 잡은 값이었다."""
    d = PairDetector(pairs=[], config=DetectorConfig(
        leader_stall_secs=15.0, leader_drop_ticks=2),
        universe=["LEADER", "FOLLOWER"],
        sectors={"LEADER": "전기전자", "FOLLOWER": "전기전자"},
        prev_close={"LEADER": PREV, "FOLLOWER": PREV})
    _tick(d, 0, 12_900)
    _tick(d, 2, 12_950)      # 고점
    _tick(d, 4, 12_930)      # 2호가(20원) 아래 → 하락
    assert LOST_FADE in _reasons(d)
