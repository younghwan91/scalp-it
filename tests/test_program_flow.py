"""장중 프로그램 수급 필터(만쥬 "쏟아내는 장세면 진입을 거른다") 단위테스트.

여기서 잡는 건 **조용한 무동작**이다. 키움 응답의 음수가 ``"--4756"`` 로 오는데
그냥 ``float()`` 하면 예외가 나고 0 이 된다 — 쏟아내는 장세가 "수급 중립"으로 읽혀
필터가 통째로 안 돈다. 켜져 있는데 아무것도 안 거르는 상태가 제일 위험하다.
"""

from scalp_it.realtime.program_flow import ProgramFlow, latest_flow, program_body

# 2026-08-27 코스닥 ka90005 실측 형태.
ROW = {
    "cntr_tm": "200000",
    "dfrt_trde_netprps_qty": "--26", "ndiffpro_trde_netprps_qty": "--4756",
    "all_netprps": "+157832", "kospi200": "+145997",
}


def test_double_minus_is_a_negative_number():
    """``--4756`` 을 못 읽으면 대량 순매도가 0 으로 읽힌다 — 필터가 무동작이 된다."""
    f = latest_flow([ROW])
    assert f is not None
    assert f.nondiff_net_qty == -4756
    assert f.net_qty == -4782, "차익 −26 + 비차익 −4756"
    assert f.net_amount == 157832


def test_plus_sign_is_stripped():
    f = latest_flow([{**ROW, "ndiffpro_trde_netprps_qty": "+1200",
                      "dfrt_trde_netprps_qty": "+300"}])
    assert f.net_qty == 1500


def test_blank_and_garbage_read_as_zero_not_crash():
    f = latest_flow([{"cntr_tm": "090100", "ndiffpro_trde_netprps_qty": "",
                      "dfrt_trde_netprps_qty": "N/A", "all_netprps": None}])
    assert f.net_qty == 0 and f.net_amount == 0


def test_latest_row_is_the_first_one():
    """장중엔 첫 행이 직전 분이다 — 뒤 행을 읽으면 몇 시간 전 수급으로 판단하게 된다."""
    rows = [{**ROW, "cntr_tm": "1030"}, {**ROW, "cntr_tm": "0900"}]
    assert latest_flow(rows).time == "1030"


def test_empty_rows_yield_none():
    assert latest_flow([]) is None


# ----- 만쥬의 문턱 -------------------------------------------------------------

def _flow(net, nondiff):
    return ProgramFlow(time="1030", net_qty=net, net_amount=0.0, nondiff_net_qty=nondiff)


def test_dumping_at_manju_threshold():
    """만쥬: (비공개 회신 — 공개본에서 뺐다) — 천주 단위로 700~800."""
    assert _flow(-900, -900).dumping(min_net_qty=700) is True
    assert _flow(-500, -500).dumping(min_net_qty=700) is False


def test_nondiff_alone_is_enough_to_block():
    """차익거래는 지수 헤지라 방향이 다를 수 있다 — 실질 수급이 쏟아내면 안 산다."""
    assert _flow(-100, -900).dumping(min_net_qty=700) is True


def test_net_buying_is_never_dumping():
    assert _flow(+2000, +2000).dumping(min_net_qty=700) is False


def test_threshold_zero_turns_the_filter_off():
    assert _flow(-99999, -99999).dumping(min_net_qty=0) is False


def test_body_targets_kosdaq_by_default():
    assert program_body("20260827")["mrkt_tp"] == "P101"
    assert program_body("20260827", "P001")["mrkt_tp"] == "P001"
    assert program_body("20260827")["date"] == "20260827"


# ----- 가드 배선: 차단은 킬이 아니다 -------------------------------------------

from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import (
    BLOCK_PROGRAM_DUMP,
    RiskGuard,
    RiskGuardConfig,
)

BASE = datetime(2026, 8, 27, 10, 0, 0)  # noqa: DTZ001 — KST naive, 프로젝트 표준


def _alert():
    return Alert(
        ts=BASE, leader="000001", follower="000002", leader_surge=0.02, cosurge=0.006,
        premove=0.0, price=10_000, bid1=10_000, entry=10_000, take_profit=None,
        stop_loss=9_950.0, disaster_line=None,
        exit_ts=BASE + timedelta(seconds=180), turnover=1e9,
        leader_name="대장", follower_name="후속",
    )


def _guard(**kw):
    base = {"daily_loss_limit": 0.0, "max_consecutive_losses": 0, "max_positions": 99,
            "stop_file": None, "position_notional": 10_000_000.0}
    base.update(kw)
    return RiskGuard(RiskGuardConfig(**base))


def test_dumping_blocks_new_entries():
    g = _guard()
    g.set_program_dumping(True)
    d = g.on_alert(_alert())
    assert not d.allowed and d.reason == BLOCK_PROGRAM_DUMP
    assert g.open_positions == []


def test_dumping_is_not_a_killswitch():
    """수급이 돌아오면 그 자리에서 다시 사야 한다 — 하루를 접는 게 아니다."""
    g = _guard()
    g.set_program_dumping(True)
    assert not g.on_alert(_alert()).allowed
    assert not g.killed, "국면·수급 차단은 킬이 아니다"
    g.set_program_dumping(False)
    assert g.on_alert(_alert()).allowed


def test_open_positions_still_exit_while_blocked():
    """열린 포지션의 청산은 차단과 무관하게 돌아야 한다 — 안 그러면 물린다."""
    g = _guard()
    g.on_alert(_alert())
    g.on_tick("000002", BASE + timedelta(seconds=1), 9_999)
    g.set_program_dumping(True)
    g.on_tick("000002", BASE + timedelta(seconds=10), 9_940)
    assert g.open_positions == []
    assert g.closed_positions[-1].exit_reason == "손절"


def test_blocked_signal_can_still_be_shadowed():
    """쏟아내는 장세에 안 산 것도 "샀으면 어땠나"는 남는다."""
    g = _guard(shadow_when_blocked=True)
    g.set_program_dumping(True)
    g.on_alert(_alert())
    assert len(g.shadow_positions) == 1
    assert g.shadow_positions[0].blocked is True
