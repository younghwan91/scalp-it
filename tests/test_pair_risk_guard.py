# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""짝꿍 리스크 가드 단위테스트 — 손실한도·연속손절·일일리셋·포지션상한·체결가정.

가드는 DB·시계를 모르므로 mock 알림·틱만 먹여 순수 판정을 고정한다.
"""

import json
from datetime import datetime, timedelta

from scalp_it.realtime.pair_detector import Alert
from scalp_it.realtime.risk_guard import (
    BLOCK_REGIME_ZERO,
    KILL_CONSECUTIVE,
    KILL_DAILY_LOSS,
    KILL_MANUAL,
    LEADER_HOLDING,
    LEADER_LOST,
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
    # 손절 1회 = 실제 트리거 틱 99.5 → -0.5% * 10만 = -500원. 한도 1400원 → 3회째에 도달.
    g = _guard(daily_loss_limit=1400.0)
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


# ------------------------------------------------- D-1 그림자 포지션(두 가정 독립)

def test_shadow_position_runs_opposite_assumption_independently():
    """회계는 through 인데 touch 가정도 **독립 수명**으로 굴려야 한다.

    실물 재현(2026-08-25 033340): 지정가 539 를 09:02:00 에 터치 → 09:02:02 에 542 로
    익절선(541.695) 돌파 → touch 가정이면 익절. 그러나 through 체결은 09:02:22(538 관통)
    이라 그 상승을 못 보고 09:02:24 에 536 으로 손절. 두 결과가 **달라야** 한다.
    """
    g = _guard(fill_basis="through")
    a = _alert(0, entry=539.0, tp=541.695, sl=536.844, hold=90)
    g.on_alert(a)
    g.on_tick(FOL, BASE + timedelta(seconds=0), 539.0)   # touch 체결(관통 아님)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 542.0)   # touch 가정 → 익절
    g.on_tick(FOL, BASE + timedelta(seconds=22), 538.0)  # through 체결
    g.on_tick(FOL, BASE + timedelta(seconds=24), 536.0)  # through 가정 → 손절

    real = g.closed_positions[0]
    assert real.basis == "through"
    assert real.exit_reason == "손절"
    assert real.pnl < 0

    shadow = g.closed_shadows[0]
    assert shadow.basis == "touch"
    assert shadow.exit_reason == "익절"
    assert shadow.pnl > 0
    # 그림자는 자기 체결 시점부터 자기 MFE 를 잰다(회계 포지션과 다른 값)
    assert shadow.mfe > 0.0
    assert real.mfe == 0.0


def test_shadow_never_touches_accounting():
    """그림자 손익은 realized_krw·승패·연속손절·킬스위치에 절대 반영되지 않는다."""
    g = _guard(max_consecutive_losses=1, fill_basis="through")
    a = _alert(0, entry=539.0, tp=541.695, sl=536.844, hold=90)
    g.on_alert(a)
    g.on_tick(FOL, BASE + timedelta(seconds=0), 539.0)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 542.0)   # 그림자 익절(+)
    assert g.realized_krw == 0.0
    assert g.wins == 0 and g.losses == 0
    g.on_tick(FOL, BASE + timedelta(seconds=22), 538.0)
    g.on_tick(FOL, BASE + timedelta(seconds=24), 536.0)  # 회계 손절(-)
    assert g.realized_krw < 0
    assert g.losses == 1 and g.wins == 0
    assert g.killed is True                              # 회계 손절만 킬을 만든다


def test_shadow_not_counted_in_max_positions():
    g = _guard(max_positions=1)
    assert g.on_alert(_alert(0, follower="000002")).allowed is True
    assert len(g.open_positions) == 1                    # 그림자는 상한 계산 밖


def test_shadow_close_logged_with_own_basis(tmp_path):
    log = tmp_path / "positions.log"
    sink = PositionLogSink(log, basis="through")
    g = _guard(fill_basis="through")
    g.set_close_callback(sink.log_close)
    g.on_alert(_alert(0, entry=539.0, tp=541.695, sl=536.844, hold=90))
    g.on_tick(FOL, BASE + timedelta(seconds=0), 539.0)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 542.0)
    g.on_tick(FOL, BASE + timedelta(seconds=22), 538.0)
    g.on_tick(FOL, BASE + timedelta(seconds=24), 536.0)
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    by_basis = {r["fill_basis"]: r for r in rows if r["event"] == "close"}
    assert set(by_basis) == {"through", "touch"}
    assert by_basis["touch"]["exit_reason"] == "익절"
    assert by_basis["through"]["exit_reason"] == "손절"


# ------------------------------------------------- D-3 청산가 계상(주문 유형별)

def test_stop_loss_uses_actual_trigger_tick():
    """손절은 하락을 확인하고 스프레드를 넘어 파는 것 — 브래킷 레벨이 아니라 실제 틱."""
    g = _guard()
    g.on_alert(_alert(0))                                  # entry 100, sl 99.6
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.4)      # 손절선 아래로 관통
    pos = g.closed_positions[0]
    assert pos.exit_price == 99.4
    assert abs(pos.pnl - (99.4 / 100.0 - 1.0)) < 1e-9


def test_disaster_line_uses_actual_trigger_tick():
    g = _guard()
    g.on_alert(_alert(0, sl=None, disaster=99.0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 98.5)
    pos = g.closed_positions[0]
    assert pos.exit_reason == "재난선"
    assert pos.exit_price == 98.5


def test_take_profit_uses_bracket_level():
    """익절은 익절가에 미리 걸어둔 지정가가 체결되는 것 — 레벨 그대로."""
    g = _guard()
    g.on_alert(_alert(0))                                  # tp 100.5
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 101.2)     # 훌쩍 넘어가도
    pos = g.closed_positions[0]
    assert pos.exit_price == 100.5


# ------------------------------------------------- D-4 미체결은 청산이 아니다

def test_unfilled_logged_as_separate_event(tmp_path):
    log = tmp_path / "positions.log"
    sink = PositionLogSink(log, basis="through")
    g = _guard(fill_basis="through")
    g.set_close_callback(sink.log_close)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 100.0)     # 관통 아님 → through 미체결
    g.on_tick(FOL, BASE + timedelta(seconds=95), 100.6)    # 청산시각 경과
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    through = next(r for r in rows if r.get("fill_basis") == "through")
    assert through["event"] == "unfilled"
    assert through["status"] == "unfilled"
    assert all(r["event"] != "close" for r in rows if r.get("fill_basis") == "through")


# ------------------------------------------------- D-5 일일 집계(가정별·중복·비용)

def _summary_mod(tmp_path):
    """scripts/pair_daily_summary.py 를 로드하고 입출력 경로를 tmp 로 돌린다."""
    import importlib.util
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts" / "pair_daily_summary.py"
    spec = importlib.util.spec_from_file_location("pair_daily_summary", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.POS_LOG = tmp_path / "positions.log"
    mod.TXT_LOG = tmp_path / "summary.log"
    mod.JSONL = tmp_path / "summary.jsonl"
    return mod


def _run_summary(tmp_path, guard_setup) -> tuple:
    mod = _summary_mod(tmp_path)
    sink = PositionLogSink(mod.POS_LOG, basis="through")
    g = _guard(fill_basis="through")
    g.set_close_callback(sink.log_close)
    guard_setup(g)
    import sys
    argv = sys.argv
    sys.argv = ["pair_daily_summary.py", "2026-08-24"]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    recs = [json.loads(x) for x in mod.JSONL.read_text(encoding="utf-8").splitlines()]
    return mod, recs[-1]


def test_summary_groups_by_fill_basis_not_boolean(tmp_path):
    """033340 재현: 같은 알림이 touch 는 익절, through 는 손절로 **다르게** 잡혀야 한다."""
    def setup(g):
        g.on_alert(_alert(0, entry=539.0, tp=541.695, sl=536.844, hold=90))
        g.on_tick(FOL, BASE + timedelta(seconds=0), 539.0)
        g.on_tick(FOL, BASE + timedelta(seconds=2), 542.0)
        g.on_tick(FOL, BASE + timedelta(seconds=22), 538.0)
        g.on_tick(FOL, BASE + timedelta(seconds=24), 536.0)

    _, rec = _run_summary(tmp_path, setup)
    thr, tch = rec["by_basis"]["through"], rec["by_basis"]["touch"]
    assert thr["gross"]["total_krw"] < 0
    assert tch["gross"]["total_krw"] > 0            # 낙관 시나리오가 드디어 측정된다
    assert thr["exit_reasons"] == {"손절": 1}
    assert tch["exit_reasons"] == {"익절": 1}


def test_summary_excludes_unfilled_from_close_stats(tmp_path):
    """미체결은 청산 건수·MAE/MFE 중앙값에서 빠지고 별도 항목으로만 보고된다."""
    def setup(g):
        _settle_loss(g, 0)                                   # 체결 후 손절
        g.on_alert(_alert(200, follower="000003"))
        g.on_tick("000003", BASE + timedelta(seconds=201), 100.0)   # 관통 없음
        g.on_tick("000003", BASE + timedelta(seconds=295), 100.0)   # 미체결 만료

    _, rec = _run_summary(tmp_path, setup)
    thr = rec["by_basis"]["through"]
    assert thr["n_closed"] == 1
    assert thr["n_unfilled"] == 1
    assert thr["fill_rate_pct"] == 50.0
    assert thr["median_mae_pct"] < -0.1                      # 0.0 이 섞이면 희석된다
    assert thr["gross"]["n"] == 1


def test_summary_upserts_instead_of_appending(tmp_path):
    def setup(g):
        _settle_loss(g, 0)

    mod, _ = _run_summary(tmp_path, setup)
    import sys
    argv = sys.argv
    sys.argv = ["pair_daily_summary.py", "2026-08-24"]
    try:
        mod.main()
        mod.main()
    finally:
        sys.argv = argv
    assert len(mod.JSONL.read_text(encoding="utf-8").splitlines()) == 1
    assert len(mod.TXT_LOG.read_text(encoding="utf-8").splitlines()) == 1


def test_summary_reports_three_cost_scenarios(tmp_path):
    def setup(g):
        _settle_loss(g, 0)

    _, rec = _run_summary(tmp_path, setup)
    thr = rec["by_basis"]["through"]
    assert set(thr["net"]) == {"net34", "net64", "net94"}
    for key, cost in rec["cost_scenarios"].items():
        expected = thr["gross"]["mean_pct"] - cost * 100
        assert abs(thr["net"][key]["mean_pct"] - expected) < 1e-6


def test_flush_time_exit_uses_last_price_not_best_price():
    """flush 시간청산은 **마지막 관측가**로 계상해야 한다 — 최고가가 아니라.

    기존 구현은 entry*(1+mfe) 를 썼는데 MFE 는 그 포지션이 본 최고가다. 틱 공백·장
    마감으로 flush 를 타는 모든 건의 손익이 체계적으로 부풀려진다(낙관 편향).
    """
    g = _guard()
    a = _alert(entry=100.0, tp=100.5, sl=None, hold=90.0)
    g.on_alert(a)
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.0)    # 체결(관통)
    g.on_tick(FOL, BASE + timedelta(seconds=10), 100.4)  # 고점(MFE) — 익절선 미달
    g.on_tick(FOL, BASE + timedelta(seconds=20), 99.2)   # 마지막 관측가
    g.flush_expired(BASE + timedelta(seconds=200))       # 청산시각 지남

    closed = [p for p in g.closed_positions if not p.shadow]
    assert closed and closed[-1].exit_reason == "시간청산"
    assert closed[-1].exit_price == 99.2, (
        f"마지막 관측가 99.2 로 계상돼야 하는데 {closed[-1].exit_price}"
    )


# ------------------------------------------------- 대장 조건부 청산(만쥬 §5-3)

def _leader_alert(secs: float = 0.0, *, leader: str = "000001", follower: str = FOL,
                  entry: float = 100.0, tp: float | None = 100.5,
                  sl: float | None = 99.6, hold: float = 90.0) -> Alert:
    ts = BASE + timedelta(seconds=secs)
    return Alert(
        ts=ts, leader=leader, follower=follower,
        leader_surge=0.02, cosurge=0.006, premove=0.0,
        price=entry, bid1=entry, entry=entry,
        take_profit=tp, stop_loss=sl, disaster_line=None,
        exit_ts=ts + timedelta(seconds=hold), turnover=1e9,
        leader_name="대장", follower_name="후속",
    )


def test_leader_lost_exits_filled_position_regardless_of_pnl():
    """대장이 상한가를 놓치면 손익 무관 즉시 청산(익절선·손절선 사이여도)."""
    g = _guard()
    g.on_alert(_leader_alert())
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)     # through 체결
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.2)    # 익절·손절 어느 선도 아님
    assert len(g.open_positions) == 1
    g.on_leader_state("000001", BASE + timedelta(seconds=3), LEADER_LOST)
    assert g.open_positions == []
    pos = g.closed_positions[-1]
    assert pos.exit_reason == "대장이탈"
    assert pos.status == "closed"
    assert pos.exit_price == 100.2                        # 마지막으로 본 가격에 청산
    assert pos.pnl is not None and pos.pnl > 0            # 수익 중이어도 나간다


def test_leader_holding_is_noop():
    """'유지 중' 상태는 아무것도 하지 않는다."""
    g = _guard()
    g.on_alert(_leader_alert())
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_leader_state("000001", BASE + timedelta(seconds=2), LEADER_HOLDING)
    assert len(g.open_positions) == 1


def test_leader_lost_only_touches_that_leader():
    g = _guard()
    g.on_alert(_leader_alert(follower="000002", leader="000001"))
    g.on_alert(_leader_alert(follower="000003", leader="000009"))
    g.on_tick("000002", BASE + timedelta(seconds=1), 99.9)
    g.on_tick("000003", BASE + timedelta(seconds=1), 99.9)
    g.on_leader_state("000001", BASE + timedelta(seconds=2), LEADER_LOST)
    assert [p.follower for p in g.open_positions] == ["000003"]


def test_leader_lost_applies_to_shadow_positions():
    """그림자(반대 체결가정)도 같은 대장 사유로 나가야 이중회계가 어긋나지 않는다."""
    g = _guard(track_shadow_fill=True, fill_basis="through")
    g.on_alert(_leader_alert())
    g.on_tick(FOL, BASE + timedelta(seconds=1), 100.0)    # touch 만 체결
    g.on_leader_state("000001", BASE + timedelta(seconds=2), LEADER_LOST)
    assert g.open_positions == [] and g.shadow_positions == []
    shadow = g.closed_shadows[-1]
    assert shadow.exit_reason == "대장이탈" and shadow.status == "closed"


def test_leader_lost_unfilled_position_expires_as_unfilled():
    """미체결 상태에서 대장이 이탈하면 살 이유가 없다 → 미체결로 정리(손익 없음)."""
    g = _guard()
    g.on_alert(_leader_alert())
    g.on_leader_state("000001", BASE + timedelta(seconds=2), LEADER_LOST)
    assert g.open_positions == []
    pos = g.closed_positions[-1]
    assert pos.status == "unfilled"
    assert pos.pnl is None
    assert "대장이탈" in pos.exit_reason


def test_leader_lost_beats_time_exit_on_same_instant():
    """같은 순간에 시간청산과 겹치면 대장이탈로 기록한다(만쥬: 근거 상실이 절대 기준)."""
    g = _guard()
    g.on_alert(_leader_alert(hold=10.0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_leader_state("000001", BASE + timedelta(seconds=10), LEADER_LOST)
    assert g.closed_positions[-1].exit_reason == "대장이탈"


def test_leader_state_never_fed_keeps_legacy_behaviour():
    """하위호환 — 대장 상태를 한 번도 안 먹이면 시간청산 그대로."""
    g = _guard()
    g.on_alert(_leader_alert(hold=10.0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=11), 100.1)
    assert g.closed_positions[-1].exit_reason == "시간청산"


def test_leader_lost_counts_into_kill_switch_accounting():
    """대장이탈 손실도 연속손절·실현손익에 정상 반영된다."""
    g = _guard(max_consecutive_losses=1)
    g.on_alert(_leader_alert(sl=None))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 99.7)
    g.on_leader_state("000001", BASE + timedelta(seconds=3), LEADER_LOST)
    assert g.realized_krw < 0
    assert g.killed and g.kill_reason == KILL_CONSECUTIVE


# ------------------------------------------------- 국면별 비중 조절(만쥬 §8-1)

def test_size_multiplier_scales_position_notional():
    """배수 0.3 이면 명목이 30%로 줄고 손익(원)도 그만큼만 잡힌다."""
    g = _guard(size_multiplier=0.3)
    _settle_win(g, 0)
    pos = g.closed_positions[0]
    assert pos.notional == 30_000.0
    assert abs(pos.pnl_krw - pos.pnl * 30_000.0) < 1e-6
    assert abs(g.realized_krw - pos.pnl_krw) < 1e-6


def test_size_multiplier_zero_blocks_new_alert_but_is_not_kill():
    """극침체(배수 0)는 신규 진입 금지. 킬스위치와는 다르다 — 배수를 되돌리면 재개."""
    g = _guard(size_multiplier=0.0)
    d = g.on_alert(_alert(0))
    assert d.allowed is False
    assert d.reason == BLOCK_REGIME_ZERO
    assert d.killed is False and g.killed is False
    assert g.open_positions == [] and g.shadow_positions == []
    g.set_size_multiplier(1.0)
    assert g.on_alert(_alert(10)).allowed is True


def test_size_multiplier_zero_still_closes_open_positions():
    """배수 0으로 바뀌어도 **들고 있는 포지션의 청산은 정상 동작**해야 한다."""
    g = _guard()
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)     # 체결
    g.set_size_multiplier(0.0)
    assert g.on_alert(_alert(5, follower="000003")).allowed is False   # 신규만 차단
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.6)    # 익절은 그대로
    assert g.closed_positions[0].exit_reason == "익절"
    assert g.realized_krw > 0


def test_size_multiplier_default_is_unchanged():
    g = _guard()
    _settle_win(g, 0)
    assert g.closed_positions[0].notional == 100_000.0


# ---------------------------------------------------- 분할 매도(만쥬 §5-2)

def test_partial_exit_at_gain_then_remainder_takes_profit():
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)     # 체결
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25)   # +0.25% → 1차 분할
    pos = g.open_positions[0]
    assert pos.partial_done is True and pos.remaining == 0.5
    assert abs(pos.partial_pnl_krw - 0.0025 * 100_000 * 0.5) < 1e-6
    assert abs(g.realized_krw - pos.partial_pnl_krw) < 1e-6
    g.on_tick(FOL, BASE + timedelta(seconds=3), 100.6)    # 잔량은 기존 익절 규칙
    done = g.closed_positions[0]
    assert done.exit_reason == "익절"
    assert abs(done.pnl_krw - 0.005 * 100_000 * 0.5) < 1e-6
    assert abs(g.realized_krw - (125.0 + 250.0)) < 1e-6


def test_partial_exit_by_elapsed_time():
    g = _guard(partial_exit_ratio=0.4, partial_exit_after_s=30.0)
    g.on_alert(_alert(0, tp=None))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)     # 체결
    g.on_tick(FOL, BASE + timedelta(seconds=10), 100.1)   # 아직 30초 전
    assert g.open_positions[0].partial_done is False
    g.on_tick(FOL, BASE + timedelta(seconds=40), 100.1)   # 체결 후 39초 → 분할
    assert g.open_positions[0].partial_done is True
    assert g.open_positions[0].remaining == 0.6


def test_partial_fires_only_once():
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002)
    g.on_alert(_alert(0, tp=None))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25)
    g.on_tick(FOL, BASE + timedelta(seconds=3), 100.3)
    assert g.open_positions[0].remaining == 0.5


def test_leader_exit_is_full_never_partial():
    """만쥬 §5-3 은 '손익 무관 **전량**' — 대장이탈은 분할 대상이 아니다."""
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_leader_state("000001", BASE + timedelta(seconds=2), LEADER_LOST)
    pos = g.closed_positions[0]
    assert pos.exit_reason == "대장이탈"
    assert pos.partial_done is False and pos.remaining == 1.0
    assert abs(pos.pnl_krw - pos.pnl * 100_000) < 1e-6


def test_stop_loss_after_partial_uses_remainder_only():
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25)   # 1차 분할(+125원)
    g.on_tick(FOL, BASE + timedelta(seconds=3), 99.5)     # 잔량 손절
    done = g.closed_positions[0]
    assert done.exit_reason == "손절"
    assert abs(done.pnl_krw - done.pnl * 100_000 * 0.5) < 1e-6
    # 승패는 그 포지션의 **합계**로 가른다(분할 익절 + 잔량 손절).
    total = done.partial_pnl_krw + done.pnl_krw
    assert abs(g.realized_krw - total) < 1e-6
    assert (g.losses == 1) is (total < 0)


def test_partial_disabled_by_default():
    g = _guard()
    _settle_win(g, 0)
    pos = g.closed_positions[0]
    assert pos.partial_done is False and pos.remaining == 1.0
    assert abs(pos.pnl_krw - pos.pnl * 100_000) < 1e-6


def test_partial_logged_as_its_own_event(tmp_path):
    log = tmp_path / "positions.log"
    sink = PositionLogSink(log, basis="through")
    g = _guard(partial_exit_ratio=0.5, partial_exit_gain=0.002)
    g.set_close_callback(sink.log_close)
    g.on_alert(_alert(0))
    g.on_tick(FOL, BASE + timedelta(seconds=1), 99.9)
    g.on_tick(FOL, BASE + timedelta(seconds=2), 100.25)
    g.on_tick(FOL, BASE + timedelta(seconds=3), 100.6)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    events = [x["event"] for x in lines]
    assert events[0] == "partial" and events[-1] == "close"
    assert lines[0]["partial_ratio"] == 0.5
    assert lines[-1]["remaining"] == 0.5
