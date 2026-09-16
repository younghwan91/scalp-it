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


def test_cross_code_features_come_from_provider():
    """종목 간 순위(cum_rank·dayret_rank)는 커널이 못 준다 — 공급자가 채우고, 없으면 추론을 끈다."""
    pol = _policy(["acc", "cum_rank"], always=1)
    logged = []
    ranks = {}

    def provider(code, sec):
        return {"cum_rank": ranks.get(code, 99.0)}

    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=logged.append,
                  extra_features=provider)
    ranks["005930"] = 3.0
    _feed(sh, "005930", 4)
    assert sh.disabled_reason is None, "공급자가 채우면 꺼지면 안 된다"
    assert any(r["event"] == "buy" for r in logged)


def test_cumval_rank_helper_ranks_by_turnover():
    """그림자가 스스로 당일 누적 체결대금 순위를 만든다 — 백테스트의 cum_rank 와 같은 정의."""
    pol = _policy(["acc", "cum_rank"], always=0)
    sh = RLShadow(pol, stream_factory=lambda: _Stream({"acc": 1.0}), sink=lambda r: None,
                  extra_features="turnover_rank")
    _feed(sh, "AAA", 3, price=10000)          # 10,000 × 1주 × 3초
    _feed(sh, "BBB", 3, price=50000)          # 더 큼
    assert sh.extra_for("BBB", 0)["cum_rank"] == 1
    assert sh.extra_for("AAA", 0)["cum_rank"] == 2


def test_raw_events_reach_window_style_kernel():
    """창 재계산 커널은 집계가 아니라 원시 틱·호가가 필요하다 — 그림자가 그걸 넘겨야 한다."""
    seen = {"ticks": 0, "quotes": 0, "sec": -1}

    class _WindowLike:
        sec = -1

        def feed_tick(self, *a):
            seen["ticks"] += 1

        def feed_quote(self, *a):
            seen["quotes"] += 1

        def update(self, **kw):
            seen["sec"] = self.sec
            return {"acc": 1.0}

    pol = _policy(["acc"], always=0)
    sh = RLShadow(pol, stream_factory=lambda code=None: _WindowLike(), sink=lambda r: None)
    from datetime import datetime, timedelta
    t0 = datetime(2026, 9, 17, 9, 30, 0)
    for k in range(3):
        sh.on_quote("005930", t0 + timedelta(seconds=k), bid1=9990, ask1=10000, bidqty1=10, askqty1=10)
        sh.on_tick("005930", t0 + timedelta(seconds=k), 10000, volume=1, side=1, strength=100.0,
                   best_bid=9990, best_ask=10000)
    assert seen["ticks"] == 3 and seen["quotes"] == 3
    assert seen["sec"] >= 0, "커널에 초가 전달돼야 한다"


def test_weights_load_from_npz(tmp_path):
    """운용 환경(torch 없음)에서도 정책을 읽어야 한다."""
    p = tmp_path / "weights.npz"
    np.savez(p, w0=np.zeros((8, 5), np.float32), b0=np.zeros(8, np.float32),
             w1=np.zeros((4, 8), np.float32), b1=np.array([0, 0, 1, 0], np.float32))
    pol = PolicyWeights.from_npz(p, features=["acc"], mu=np.zeros(1), sd_=np.ones(1),
                                 lo=np.zeros(1), hi=np.ones(1))
    assert int(np.argmax(pol.forward(np.zeros(1 + RLShadow.EXTRA, np.float32)))) == 2


# ── 지정가 대기열 체결(86 §3) 을 실시간에서도 흉내낸다 ────────────────────────
# 96번은 지정가 진입만 한다. 그림자가 체결을 안 만들면 의사결정 패리티(승격 9번)를
# 잴 수 없고, 정책이 라이브에서 무엇을 했는지 기록이 남지 않는다.
def _shadow_with_limit(sink_rows, *, action):
    """행동을 고정한 정책으로 그림자를 만든다 — 체결 로직만 시험하기 위해서다."""
    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    n_act = 4
    w = np.zeros((n_act, 5), np.float32)
    w[action] = 1.0                                  # 그 행동의 Q 만 크게
    pol = PolicyWeights(features=["f"], layers=[(w, np.array([0, 0, 0, 0], np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            return {"f": 1.0}

    return RLShadow(pol, stream_factory=Stream, sink=sink_rows.append, codes={"A"},
                    extra_features=lambda c, s: {})


def test_limit_buy_fills_when_queue_ahead_is_consumed():
    """앞 잔량 + 1 만큼 매도 체결이 지나가면 우리 지정가가 채워진다(86 §3)."""
    from datetime import datetime

    rows = []
    sh = _shadow_with_limit(rows, action=2)          # 2 = 지정가 매수
    t = datetime(2026, 9, 17, 9, 10, 0)
    sh.on_quote("A", t, bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
    sh.on_tick("A", t, 1005, volume=1, side=1)
    sh.on_quote("A", t.replace(second=1), bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
    # 다음 초부터 우리 가격에 매도 체결이 지나간다 — 앞 5 + 우리 1 = 6
    for s in (2, 3):
        ts = t.replace(second=s)
        sh.on_quote("A", ts, bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
        sh.on_tick("A", ts, 1000, volume=4, side=-1)
    sh.on_quote("A", t.replace(second=4), bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
    buys = [r for r in rows if r["event"] == "buy"]
    assert buys, f"체결이 안 만들어졌다: {rows}"
    assert buys[0]["kind"] == "지정가" and buys[0]["price"] == 1000


def test_limit_buy_cancels_after_timeout_without_fill():
    """10초 안에 안 채워지면 취소한다 — 86 §3 의 전제다."""
    from datetime import datetime

    rows = []
    sh = _shadow_with_limit(rows, action=2)
    t = datetime(2026, 9, 17, 9, 10, 0)
    for s in range(0, 14):
        ts = t.replace(second=s)
        sh.on_quote("A", ts, bid1=1000, ask1=1010, bidqty1=500, askqty1=50)
        sh.on_tick("A", ts, 1005, volume=1, side=1)
    sh.on_quote("A", t.replace(second=14), bid1=1000, ask1=1010, bidqty1=500, askqty1=50)
    assert not [r for r in rows if r["event"] == "buy"]
    assert [r for r in rows if r["event"] == "limit_cancel"], f"취소 기록이 없다: {rows}"


def test_limit_buy_fills_when_price_trades_through():
    """가격이 우리 지정가 아래로 뚫고 내려가면 대기열과 무관하게 체결이다."""
    from datetime import datetime

    rows = []
    sh = _shadow_with_limit(rows, action=2)
    t = datetime(2026, 9, 17, 9, 10, 0)
    sh.on_quote("A", t, bid1=1000, ask1=1010, bidqty1=9999, askqty1=50)
    sh.on_tick("A", t, 1005, volume=1, side=1)
    sh.on_quote("A", t.replace(second=1), bid1=1000, ask1=1010, bidqty1=9999, askqty1=50)
    ts = t.replace(second=2)
    sh.on_quote("A", ts, bid1=990, ask1=1000, bidqty1=9999, askqty1=50)
    sh.on_tick("A", ts, 995, volume=1, side=-1)       # 우리 가격보다 아래에서 체결
    sh.on_quote("A", t.replace(second=3), bid1=990, ask1=1000, bidqty1=9999, askqty1=50)
    assert [r for r in rows if r["event"] == "buy"], f"뚫렸는데 체결이 없다: {rows}"
