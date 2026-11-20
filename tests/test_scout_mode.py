"""Gate 0 아래에서는 **정찰병 비중**으로만 매매한다(비공개 출처 — 공개본에서 뺐다).

Gate 0 = 코스닥 7~8조 이하의 거래 침체기. 2등주 호가가 비어 슬리피지와 윗꼬리
리스크가 크므로 평소 비중의 1/10 로만 대응한다.

우리는 게이트를 8조 → 5조로 낮춰 놓았다(2026-08-27 사용자 결정, 실측 데이터 확보용).
그 결정과 모순되지 않는다 — 다만 그 구간에서는 비중이 1/10 이어야
한다. 실주문은 이미 1주지만 **가상 회계 명목이 1천만원 그대로**라, 회계가 '평소
비중'으로 기록돼 성과 해석이 틀어진다.
"""
from scalp_it.market_regime import scout_multiplier


def test_below_gate_zero_is_scout_size():
    """5~8조 = Gate 0 아래 → 1/10 정찰병."""
    assert scout_multiplier(turnover_jo=5.5, gate_jo=8.0) == 0.1
    assert scout_multiplier(turnover_jo=7.9, gate_jo=8.0) == 0.1


def test_at_or_above_gate_zero_is_full_size():
    """8조 이상이면 평소 비중 — 여기부터가 만쥬가 말한 유효 구간이다."""
    assert scout_multiplier(turnover_jo=8.0, gate_jo=8.0) == 1.0
    assert scout_multiplier(turnover_jo=12.0, gate_jo=8.0) == 1.0


def test_gate_off_means_full_size():
    assert scout_multiplier(turnover_jo=3.0, gate_jo=0.0) == 1.0


def test_cron_sets_gate_zero_at_manju_eight_jo(cron_args):
    """cron 이 Gate 0 를 8조로 두는지 — 진입 차단선(5조)과는 다른 층이다."""
    assert cron_args.scout_gate_jo == 8.0
    assert cron_args.halt_below_jo == 5.0
