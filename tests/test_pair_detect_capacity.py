"""장전 구독 정원 분할 — 1h(VI) 를 켜면 앱키당 정원이 50→33 으로 준다(2026-10-12).

정원을 넘치면 어떤 종목·쌍이 빠졌는지 보여야 하고, 대장·후속 중 하나만 구독된
반쪽 쌍을 만들어선 안 된다(감지가 안 되는데 자리만 먹는다).
"""

from scalp_it.cli_pair_detect import _report_dropped, build_parser, plan_subscription
from scalp_it.realtime.collector import (
    TYPE_QUOTE,
    TYPE_TRADE,
    TYPE_VI,
    codes_per_connection,
)
from scalp_it.realtime.pair_detector import Pair


def test_vi_cuts_capacity_to_33_per_key():
    assert codes_per_connection([TYPE_TRADE, TYPE_QUOTE]) == 50
    assert codes_per_connection([TYPE_TRADE, TYPE_QUOTE, TYPE_VI]) == 33


def test_vi_flag_defaults_on_and_can_be_turned_off():
    assert build_parser().parse_args([]).vi is True
    assert build_parser().parse_args(["--no-vi"]).vi is False


def test_dual_key_splits_into_two_full_keys_and_reports_overflow():
    codes = [f"{i:06d}" for i in range(80)]
    groups, dropped, broken = plan_subscription(codes, [], cap=33, slots=2)
    assert [len(g) for g in groups] == [33, 33]
    assert not set(groups[0]) & set(groups[1])
    assert len(dropped) == 14
    assert set(dropped) | set(groups[0]) | set(groups[1]) == set(codes)
    assert broken == []


def test_pairs_go_in_whole_before_other_codes():
    # 코드순이면 맨 뒤라 잘렸을 쌍이 먼저 들어간다.
    pairs = [Pair("900001", "900002")]
    codes = [f"{i:06d}" for i in range(10)] + ["900001", "900002"]
    groups, dropped, broken = plan_subscription(codes, pairs, cap=3, slots=1)
    assert groups == [["900001", "900002", "000000"]]
    assert broken == []
    assert "900001" not in dropped


def test_pair_that_does_not_fit_is_dropped_whole_not_half():
    pairs = [Pair("000001", "000002"), Pair("000003", "000004")]
    codes = ["000001", "000002", "000003", "000004", "000005"]
    groups, dropped, broken = plan_subscription(codes, pairs, cap=3, slots=1)
    sub = set(groups[0])
    assert {"000001", "000002"} <= sub
    assert broken == [pairs[1]]
    # 둘째 쌍은 반쪽만 들어가지 않는다 — 남는 한 자리는 쌍이 아닌 종목이 쓴다.
    assert not ({"000003", "000004"} <= sub)
    assert "000005" in sub
    assert sorted(dropped) == ["000003", "000004"]


def test_pairs_sharing_a_leader_only_count_new_codes():
    pairs = [Pair("000001", "000002"), Pair("000001", "000003")]
    groups, _dropped, broken = plan_subscription(
        ["000001", "000002", "000003"], pairs, cap=3, slots=1)
    assert broken == [] and sorted(groups[0]) == ["000001", "000002", "000003"]


def test_dedicated_key_is_not_capped():
    codes = [f"{i:06d}" for i in range(120)]
    groups, dropped, broken = plan_subscription(codes, [], cap=33, slots=None)
    assert len(groups) == 1 and len(groups[0]) == 120
    assert dropped == [] and broken == []


def test_overflow_is_logged_not_silent(capsys):
    pairs = [Pair("000003", "000004")]
    _report_dropped(["000003", "000004"], pairs, cap=33, slots=2,
                    types=[TYPE_TRADE, TYPE_QUOTE, TYPE_VI])
    err = capsys.readouterr().err
    assert "33×2=66" in err
    assert "종목 2개·쌍 1개" in err
    assert "000003→000004" in err


def test_nothing_logged_when_everything_fits(capsys):
    _report_dropped([], [], cap=33, slots=2, types=[TYPE_TRADE])
    assert capsys.readouterr().err == ""
