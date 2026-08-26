"""CLI 배선 스모크 — 감지기·가드·실행기가 **실제로 생성되는가**.

2026-08-26: `build_order_executor` 가 `OrderExecutor` 가 받지 않는 인자를 넘겨
`--live-order` 에서 `TypeError` 로 죽었다. 936개 테스트가 통과하는데도 못 잡았다 —
**아무 테스트도 이 함수를 부르지 않았기 때문이다**(`grep build_order_executor tests/` → 0건).

내일 08:55 cron 이 그대로 죽을 뻔했다. 배선은 조립일 뿐이라 테스트를 안 붙이기 쉬운데,
조립이 깨지면 나머지가 다 멀쩡해도 하루를 날린다.
"""

import pytest

from scalp_it.cli_pair_detect import (
    build_config,
    build_guard,
    build_order_executor,
    build_parser,
    resolve_pairs,
)
from scalp_it.realtime.pair_detector import PairDetector

#: cron(`scripts/pair_detect_daily.sh`)이 실제로 쓰는 플래그. 여기서 갈리면
#: 테스트는 통과하는데 장에서 죽는다.
CRON_FLAGS = [
    # cron 은 `--live --live-order` 를 쓴다. `--live` 는 소켓 직결이라 단위테스트에서
    # 켤 수 없지만, `--live-order`(실행기 생성)는 반드시 태워야 한다 — 오배선이 거기서 죽는다.
    "--live-order",
    "--take-profit", "0", "--stop-loss", "-0.005", "--hold", "240",
    "--limitup-near", "0.05", "--limitup-require-surge", "--vi-near", "0.01",
    "--universe-size", "45", "--max-roundtrip-bp", "20", "--order-krw", "10000000",
    "--lock-min-krw", "300000000", "--lock-wall-decay", "0.5",
    "--dump-krw", "300000000", "--big-trades", "1", "--new-listing-days", "5",
    "--breaking-interval", "90", "--breaking-min-weight", "0.7", "--breaking-max", "3",
]


class _FakeOrderClient:
    """실행기 생성만 확인하면 되므로 계좌·발주는 흉내만 낸다."""

    def masked_account(self) -> str:
        return "0000******"

    def holdings(self):
        return []


@pytest.fixture
def cron_args():
    return build_parser().parse_args(CRON_FLAGS)


def test_cron_flags_parse(cron_args):
    """cron 플래그가 파서를 통과해야 한다 — 오타 하나면 08:55 에 죽는다."""
    assert cron_args.take_profit == 0
    assert cron_args.limitup_near == 0.05
    assert cron_args.vi_near == 0.01


def test_detector_builds_from_cron_flags(cron_args):
    cfg = build_config(cron_args)
    det = PairDetector(resolve_pairs(cron_args), cfg)
    assert det.config.take_profit is None      # --take-profit 0 = 상한 없음
    assert det.config.limitup_near_pct == 0.05
    assert det.config.vi_near_pct == 0.01


def test_guard_builds_from_cron_flags(cron_args):
    g = build_guard(cron_args)
    assert g is not None
    assert g.config.leader_exit is True


def test_order_executor_builds_from_cron_flags(cron_args, monkeypatch):
    """**이 테스트가 없어서 TypeError 를 놓쳤다.**

    `build_order_executor` 는 인증·화이트리스트를 거쳐 `OrderExecutor(...)` 를 부른다.
    인증 단계에서 조기 반환되면 그 호출을 안 지나므로, 클라이언트를 주입해 끝까지 태운다.
    """
    import scalp_it.cli_pair_detect as cli

    monkeypatch.setattr(cli, "_build_order_client", lambda *a, **k: _FakeOrderClient(),
                        raising=False)
    ex = build_order_executor(cron_args, resolve_pairs(cron_args),
                              client=_FakeOrderClient())
    assert ex is not None
    assert ex._fill_basis == "through"


def test_guard_only_settings_are_not_passed_to_executor():
    """§9-6 이익 절제는 **가드 전용**이다 — 실행기에 넘기면 TypeError.

    두 모듈이 '비중'이라는 같은 개념을 각자 구현해서 생긴 사고였다. 시그니처를
    직접 확인해 재발을 막는다.
    """
    import inspect

    from scalp_it.realtime.order_executor import OrderExecutor

    params = set(inspect.signature(OrderExecutor.__init__).parameters)
    assert "daily_profit_target" not in params
    assert "profit_size_multiplier" not in params
    # 가드에는 있어야 한다
    from scalp_it.realtime.risk_guard import RiskGuardConfig

    assert hasattr(RiskGuardConfig(), "daily_profit_target")
