"""최적화 하네스가 만쥬 규칙을 실제로 표현하는지 — 순위·테마·청산 축 검증.

여기서 잡는 건 **의미가 미끄러지는 버그**다. 숫자는 그럴듯한데 재고 있는 대상이
만쥬가 말한 것과 다른 경우가 제일 위험하다. 실제로 그런 게 있었다:

수집기가 넣어주는 ``ret_rank`` 는 **시장 전체** 등락률 순위다. 그걸 그대로
"2·3등주" 필터에 쓰면 "같은 섹터이면서 **시장 전체 2위**인 종목"을 사게 된다.
만쥬 규칙("그 테마 안의 2·3등주")이 아니고, 표본도 464 → 118 로 통째로 날아간다.
그래도 테스트는 초록이었다 — 순위가 무엇의 순위인지 아무도 안 물었기 때문이다.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from manju_optimize import Rule, evaluate, simulate


def _path(closes, opens=None):
    """(O,H,L,C) × 분 배열. 고·저는 종가로 채운다(청산 문턱만 보는 테스트)."""
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens if opens is not None else closes, dtype=float)
    return np.stack([o, c, c, c])


def _event(picks, info, *, lead_brk=None, T=600):
    return pd.Series({
        "date": pd.Timestamp("2026-06-01"), "lead": "000001", "T": T,
        "lead_brk": lead_brk, "lead_held": True, "picks": picks, "fol_info": info,
        "key": "k",
    })


# ----- 순위는 **테마 안의** 순위여야 한다 -------------------------------------

def _rank_fixture():
    """섹터 후보 3개. 시장 전체 순위는 5·9·14 위라 universe 기준으론 아무도 안 걸린다."""
    picks = {"sector": ["A", "B", "C"]}
    info = {
        "A": {"ret": 0.20, "ret_rank": 5, "prev_close": 1000.0},
        "B": {"ret": 0.12, "ret_rank": 9, "prev_close": 1000.0},
        "C": {"ret": 0.07, "ret_rank": 14, "prev_close": 1000.0},
    }
    paths = {c: _path([1000, 1100, 1100, 1100, 1100, 1100, 1100]) for c in "ABC"}
    return _event(picks, info), paths


def test_group_rank_buys_the_second_in_the_theme():
    """만쥬의 2등주는 **테마 안에서** 2등이다 — 시장 전체 2위가 아니다."""
    ev, paths = _rank_fixture()
    got = simulate(ev, paths, Rule(pick="sector", rank_max=2, rank_mode="group"))
    assert got is not None, "테마 내 2등주가 있는데 못 골랐다"
    assert got["fol"] == "B", "등락률 1위 A 를 건너뛰고 2위 B 를 사야 한다"


def test_universe_rank_is_a_different_and_much_stricter_thing():
    """옛 동작을 그대로 남겨 둔다 — 무엇이 달랐는지 재현할 수 있어야 한다."""
    ev, paths = _rank_fixture()
    assert simulate(ev, paths, Rule(pick="sector", rank_max=2,
                                    rank_mode="universe")) is None


def test_group_rank_skips_the_leaders_closest_follower():
    """1등주는 **이미 뛴 것**이라 사지 않는다(만쥬 §3)."""
    ev, paths = _rank_fixture()
    got = simulate(ev, paths, Rule(pick="sector", rank_max=3, rank_mode="group"))
    assert got["fol"] != "A"


def test_rank_mode_shows_up_in_the_cell_name():
    """그리드 결과를 나중에 읽을 때 어느 기준이었는지 이름만 보고 알아야 한다."""
    assert "/uni" in Rule(rank_mode="universe").key()
    assert "/uni" not in Rule(rank_mode="group").key()


# ----- 테마 선정 방식은 이름만 바꾸면 되도록 -----------------------------------

@pytest.mark.parametrize("pick", ["sector", "theme", "cosurge", "kiwoom", "topret"])
def test_every_pick_kind_flows_through(pick):
    """새 테마 정의를 넣을 때 하네스를 고칠 일이 없어야 한다."""
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    ev = _event({pick: ["A", "B"]}, info)
    paths = {c: _path([1000, 1050, 1050]) for c in "AB"}
    got = simulate(ev, paths, Rule(pick=pick, rank_max=2, rank_mode="group"))
    assert got is not None and got["fol"] == "B"


def test_missing_pick_kind_is_skipped_not_crashed():
    """어떤 이벤트엔 그 테마 후보가 없다 — 조용히 건너뛰어야 한다."""
    ev = _event({"sector": ["A"]}, {"A": {"ret": 0.2, "prev_close": 1000.0}})
    assert simulate(ev, {"A": _path([1000, 1050])}, Rule(pick="theme")) is None


# ----- 대장발 청산 -------------------------------------------------------------

def test_leader_exit_fires_at_the_break_minute():
    """대장이 상한가를 놓치면 손익 무관 그 분에 나간다 — 만쥬의 핵심 규칙."""
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    # 2분째에 대장이 이탈. 그 뒤로 오르지만 우리는 못 먹어야 한다.
    paths = {c: _path([1000, 1010, 1020, 2000, 2000, 2000]) for c in "AB"}
    ev = _event({"sector": ["A", "B"]}, info, lead_brk=602, T=600)
    got = simulate(ev, paths, Rule(pick="sector", rank_max=2, hold=5,
                                   stop=None, leader_exit=True, rank_mode="group"))
    assert got["reason"] == "대장이탈"
    assert got["gross"] < 0.05, "이탈 후 상승을 먹으면 안 된다"


def test_leader_exit_off_rides_to_the_horizon():
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    paths = {c: _path([1000, 1010, 1020, 2000, 2000, 2000]) for c in "AB"}
    ev = _event({"sector": ["A", "B"]}, info, lead_brk=602, T=600)
    got = simulate(ev, paths, Rule(pick="sector", rank_max=2, hold=5,
                                   stop=None, leader_exit=False, rank_mode="group"))
    assert got["gross"] > 0.5


def test_exit_latency_bounds_are_ordered():
    """청산 반응 속도는 open ≤ mid ≤ close 순으로 나빠져야 한다(하락 중 청산)."""
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    # 이탈 분에 시가 990 → 종가 900. 늦게 나갈수록 손해다.
    paths = {c: _path([1000, 900], opens=[1000, 990]) for c in "AB"}
    ev = _event({"sector": ["A", "B"]}, info, lead_brk=601, T=600)
    out = {at: simulate(ev, paths, Rule(pick="sector", rank_max=2, hold=5, stop=None,
                                        leader_exit=True, lead_exit_at=at,
                                        rank_mode="group"))["gross"]
           for at in ("open", "mid", "close")}
    assert out["open"] > out["mid"] > out["close"]


# ----- 집계 -------------------------------------------------------------------

def test_pl_ratio_is_avg_win_over_avg_loss():
    """만쥬가 준 실전 지표가 손익비라서 그 정의가 정확해야 한다."""
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    events = pd.DataFrame([
        _event({"sector": ["A", "B"]}, info, T=600),
        _event({"sector": ["A", "B"]}, info, T=601),
    ])
    events["key"] = ["k1", "k2"]
    paths = {"k1": {c: _path([1000, 1100]) for c in "AB"},     # +10%
             "k2": {c: _path([1000, 950]) for c in "AB"}}      # −5%
    r = evaluate(events, paths, Rule(pick="sector", rank_max=2, hold=1, stop=None,
                                     rank_mode="group"))
    assert r["n"] == 2
    assert r["pl_ratio"] == pytest.approx(2.0, rel=0.05)


# ----- 트레일링은 **수익이 난 뒤에야** 켜져야 한다 ------------------------------

def _trail_fixture():
    """+1.5% 갔다 본전까지 밀린 뒤 +6% 로 분출하고 +3% 로 되돌아오는 경로."""
    info = {"A": {"ret": 0.2, "prev_close": 1000.0}, "B": {"ret": 0.1, "prev_close": 1000.0}}
    closes = [1000, 1015, 1000, 1060, 1030, 1030]
    paths = {c: _path(closes) for c in "AB"}
    return _event({"sector": ["A", "B"]}, info), paths


def test_trailing_armed_from_entry_gets_shaken_out():
    """진입부터 켜면 초반 노이즈에 털린다 — 실제로 승률 60%→40% 로 무너졌던 자리."""
    ev, paths = _trail_fixture()
    got = simulate(ev, paths, Rule(pick="sector", rank_max=2, hold=5, stop=None,
                                   leader_exit=False, trail=0.01, trail_arm=0.0,
                                   rank_mode="group"))
    assert got["reason"] == "추세청산"
    assert got["gross"] < 0.02, "초반 되돌림에 털려 뒤에 온 분출을 못 먹는다"


def test_trailing_armed_at_three_percent_rides_the_overshoot():
    """만쥬가 말한 건 "+3% 도달 시 트레일링" — 그때까지는 안 켜져야 한다."""
    ev, paths = _trail_fixture()
    got = simulate(ev, paths, Rule(pick="sector", rank_max=2, hold=5, stop=None,
                                   leader_exit=False, trail=0.01, trail_arm=0.03,
                                   rank_mode="group"))
    assert got["gross"] > 0.02, "+6% 까지 태운 뒤에 나와야 한다"


def test_entry_slippage_makes_every_trade_worse():
    """위 호가를 긁어 사면 진입가가 불리해진다 — 분봉 종가 진입은 낙관이다."""
    ev, paths = _trail_fixture()
    base = Rule(pick="sector", rank_max=2, hold=5, stop=None, leader_exit=False,
                rank_mode="group")
    clean = simulate(ev, paths, base)["gross"]
    slipped = simulate(ev, paths, replace(base, entry_slip_ticks=2))["gross"]
    assert slipped < clean
