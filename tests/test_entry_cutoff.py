"""오후 신규 진입 차단 — 오전 핵심 구간 매매 후 오후 추격 매매는 배제한다
(비공개 출처 — 공개본에서 뺐다).

**청산은 계속 돈다.** 신규 진입만 막는다 — 오후에 손절도 못 하면 그대로 물린다.
"""
import pytest

from scalp_it.realtime.leader_scan import past_cutoff


@pytest.mark.parametrize("now,cut,expected", [
    ("0959", "1000", False),
    ("1000", "1000", False),   # 정각은 아직 창 안
    ("1001", "1000", True),
    ("1430", "1000", True),
    ("09:59", "10:00", False), # 콜론 표기도 받는다
])
def test_cutoff_blocks_only_after_the_hour(now, cut, expected):
    assert past_cutoff(now, cut) is expected


def test_empty_cutoff_never_blocks():
    """빈 값 = 끔. 오타로 하루를 통째로 날리지 않는다."""
    assert past_cutoff("1430", "") is False
    assert past_cutoff("1430", "   ") is False


def test_malformed_cutoff_never_blocks():
    assert past_cutoff("1430", "저녁") is False


def test_cron_limits_trading_to_the_morning(cron_args):
    """cron 이 오전 창과 오후 차단을 둘 다 켜는지."""
    assert cron_args.leader_scan_window == "0900-1000"
    assert cron_args.no_entry_after == "1000"
