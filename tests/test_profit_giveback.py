"""정점 이익 50% 반납 = 그날 하드스톱 — **타 트레이더 B B-3**(51번 명세).

> Track intraday peak net profit (P_peak).
> If (P_current <= P_peak * 0.50) after achieving significant P_peak:
>   -> **HARD STOP. Terminate all open positions and shutdown trading for the day.**

종전 `drawdown_krw`(정점 대비 10만원 반납)는 **킬이 아니라 비중 축소**였다(46 §10
"예수금 1/3~1/5로 축소"). 지시서는 그것과 별개로 **비율 기준 하드스톱**을 요구한다 —
번 걸 절반 뱉으면 그날 끝이다.
"""
from datetime import datetime  # noqa: F401

from scalp_it.realtime.risk_guard import KILL_GIVEBACK, RiskGuardConfig, RiskGuard

TS = datetime(2026, 8, 31, 10, 0, 0)


def _guard(**kw) -> RiskGuard:
    return RiskGuard(RiskGuardConfig(profit_giveback_ratio=0.5,
                                 profit_giveback_min_krw=10_000, **kw))


def test_half_giveback_kills_the_day():
    g = _guard()
    g.realized_krw = 100_000
    g._refresh_kill()                 # 정점 기록
    g.realized_krw = 50_000           # 정확히 절반 반납
    g._refresh_kill()
    assert g.killed and g.kill_reason == KILL_GIVEBACK


def test_shallow_giveback_does_not_kill():
    g = _guard()
    g.realized_krw = 100_000
    g._refresh_kill()
    g.realized_krw = 60_000           # 40% 반납 — 아직 아니다
    g._refresh_kill()
    assert not g.killed


def test_small_peak_is_ignored():
    """정점이 미미하면 안 죽인다 — 'significant P_peak' 단서.

    이게 없으면 +100원 벌고 +40원으로 준 순간 그날이 끝난다.
    """
    g = _guard()
    g.realized_krw = 5_000            # 문턱 10,000원 미만
    g._refresh_kill()
    g.realized_krw = 0
    g._refresh_kill()
    assert not g.killed


def test_rule_off_by_default():
    """기본값은 꺼짐 — 켜는 건 cron 의 몫이다."""
    g = RiskGuard(RiskGuardConfig())
    g.realized_krw = 100_000
    g._refresh_kill()
    g.realized_krw = 0
    g._refresh_kill()
    assert not g.killed


def test_cron_arms_the_circuit_breakers(cron_args):
    """cron 이 지시서 §5 회로차단기를 켜는지."""
    from scalp_it.cli_pair_detect import build_guard

    g = build_guard(cron_args)
    assert g is not None
    assert g.config.profit_giveback_ratio == 0.5
    assert g.config.max_consecutive_losses == 2, "지시서 §5 'On 2 consecutive losses'"
