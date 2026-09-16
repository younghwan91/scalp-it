"""RL 정책 실시간 추론 그림자 — 주문 없음, 백테스트와 같은 판단을 하는지 확인하는 경로.

승격 체크리스트 9번(의사결정 패리티)의 전제다. 정책은 학습 산출물(가중치·표준화·피처 목록)만 받고,
피처는 **백테스트와 같은 커널**(core `backtest.lob` 스트림)에서 나온 값만 쓴다 — 여기서 자체 계산하지 않는다.
"""
import numpy as np
import pytest

from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow


def _policy(feature_names, *, always=0):
    """행동 `always` 를 항상 고르는 가짜 정책(4행동: 유지·시장가매수·지정가매수·매도)."""
    d = len(feature_names) + RLShadow.EXTRA
    w1 = np.zeros((16, d), np.float32)
    b1 = np.zeros(16, np.float32)
    w2 = np.zeros((4, 16), np.float32)
    b2 = np.zeros(4, np.float32)
    b2[always] = 1.0
    return PolicyWeights(features=list(feature_names), layers=[(w1, b1), (w2, b2)],
                         mu=np.zeros(len(feature_names)), sd=np.ones(len(feature_names)),
                         lo=np.full(len(feature_names), -1e9), hi=np.full(len(feature_names), 1e9))


class _Stream:
    """피처 커널 대역 — 초마다 고정 값을 돌려준다(진짜 커널은 core 가 준다)."""

    def __init__(self, values):
        self.values = values
        self.calls = 0

    def update(self, **kw):
        self.calls += 1
        return dict(self.values)


def _feed(sh, code, secs, price=10000, bid=9990, ask=10000):
    from datetime import datetime, timedelta
    t0 = datetime(2026, 9, 17, 9, 30, 0)
    for k in range(secs):
        ts = t0 + timedelta(seconds=k)
        sh.on_tick(code, ts, price, volume=1, side=1, strength=100.0, best_bid=bid, best_ask=ask)


def test_missing_feature_disables_policy_instead_of_guessing():
    """커널이 정책이 요구하는 피처를 못 주면 **추론을 끈다** — 0으로 채워 넣고 돌리면 조용한 오답이 된다."""
    pol = _policy(["acc", "없는피처"], always=1)
    logged = []
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append)
    _feed(sh, "005930", 3)
    assert sh.disabled_reason and "없는피처" in sh.disabled_reason
    assert not any(r["event"] == "buy" for r in logged)


def test_market_buy_then_sell_logs_round_trip():
    pol = _policy(["acc"], always=1)                  # 항상 시장가 매수
    logged = []
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append)
    _feed(sh, "005930", 4)
    assert sh.open_count() == 1
    buys = [r for r in logged if r["event"] == "buy"]
    assert buys and buys[0]["price"] == 10000 and buys[0]["kind"] == "시장가"

    pol.layers[1][1][:] = 0
    pol.layers[1][1][3] = 1.0                          # 이제 항상 매도
    _feed(sh, "005930", 3)
    sells = [r for r in logged if r["event"] == "sell"]
    assert sells and sells[0]["price"] == 9990
    assert abs(sells[0]["ret"] - (9990 / 10000 - 1 - 0.0023)) < 1e-9


def test_hold_action_does_nothing():
    pol = _policy(["acc"], always=0)
    logged = []
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append)
    _feed(sh, "005930", 5)
    assert sh.open_count() == 0 and not logged


def test_sell_without_position_is_masked():
    pol = _policy(["acc"], always=3)                  # 무포지션인데 매도
    logged = []
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append)
    _feed(sh, "005930", 4)
    assert not logged, "무포지션 매도는 마스킹돼야 한다"


def test_two_codes_keep_separate_streams_and_positions():
    pol = _policy(["acc"], always=1)
    logged = []
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append)
    _feed(sh, "005930", 3)
    _feed(sh, "000660", 3, price=20000, bid=19950, ask=20000)
    assert sh.open_count() == 2
    assert {r["code"] for r in logged if r["event"] == "buy"} == {"005930", "000660"}


def test_weights_load_from_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    sd = {"net.0.weight": torch.zeros(8, 5), "net.0.bias": torch.zeros(8),
          "net.2.weight": torch.zeros(4, 8), "net.2.bias": torch.tensor([0.0, 1.0, 0.0, 0.0])}
    p = tmp_path / "ckpt.pt"
    torch.save(sd, p)
    pol = PolicyWeights.from_checkpoint(p, features=["acc"], mu=np.zeros(1), sd_=np.ones(1),
                                        lo=np.zeros(1), hi=np.ones(1))
    assert len(pol.layers) == 2
    x = np.zeros(1 + RLShadow.EXTRA, np.float32)
    assert int(np.argmax(pol.forward(x))) == 1
