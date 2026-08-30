"""Gate 0 아래에서는 **정찰병 비중**으로만 매매한다 — 만쥬 A1(2026-08-30).

> **Gate 0 (최소 유동성 필터)**: 코스닥 7~8조 이하는 시장 전체에 돈이 마른 극단적
> 거래 침체기입니다. 이때는 2등주의 호가가 비어있어 슬리피지와 윗꼬리 리스크가
> 폭증하므로 시스템 전체 매매를 쉬거나 극소액으로 통제해야 합니다.
>
> **현재 판단(코스닥 5~6조)**: 만쥬 기준으로는 '비중을 대폭 줄이고 방어하는 장세'가
> 맞습니다. 짝꿍주가 보이더라도 **평소 비중의 1/10 또는 소액 정찰병**으로만 대응하며
> 시드를 온전히 보존해야 하는 구간입니다.

우리는 게이트를 8조 → 5조로 낮춰 놓았다(2026-08-27 사용자 결정, 실측 데이터 확보용).
만쥬 답변은 그 결정을 **부정하지 않는다** — 다만 그 구간에서는 비중이 1/10 이어야
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
    """cron 이 Gate 0 를 만쥬 원문의 8조로 두는지 — 진입 차단선(5조)과는 다른 층이다."""
    assert cron_args.scout_gate_jo == 8.0
    assert cron_args.halt_below_jo == 5.0
