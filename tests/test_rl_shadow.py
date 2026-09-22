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


# ── 행동 공간을 정책이 알려준다 ────────────────────────────────────────────────
# 96번은 행동이 3개(유지·지정가매수·매도)다. 그림자가 4개로 고정돼 있으면 인덱스가
# 한 칸씩 밀려 "매도를 눌렀는데 지정가를 건다" 같은 조용한 오류가 된다.
def test_three_action_policy_maps_by_name_not_index():
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    rows = []
    w = np.zeros((3, 5), np.float32)
    w[2] = 1.0                                   # 3행동 정책에서 2 = 매도
    pol = PolicyWeights(features=["f"], layers=[(w, np.zeros(3, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9),
                        actions=("hold", "limit_buy", "sell"))

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            return {"f": 1.0}

    sh = RLShadow(pol, stream_factory=Stream, sink=rows.append, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 17, 9, 10, 0)
    for s in (0, 1, 2):
        ts = t.replace(second=s)
        sh.on_quote("A", ts, bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
        sh.on_tick("A", ts, 1005, volume=1, side=1)
    # 보유가 없으니 매도는 가려져야 하고, 지정가도 시장가도 안 걸려야 한다
    assert not [r for r in rows if r["event"] in ("buy", "limit_place")], rows


def test_default_action_space_is_the_four_action_one():
    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights

    pol = PolicyWeights(features=["f"], layers=[(np.zeros((4, 1), np.float32), np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.zeros(1), hi=np.ones(1))
    assert pol.actions == ("hold", "market_buy", "limit_buy", "sell")


def test_three_action_policy_places_limit_when_it_chooses_action_one():
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    rows = []
    w = np.zeros((3, 5), np.float32)
    w[1] = 1.0                                   # 1 = 지정가 매수
    pol = PolicyWeights(features=["f"], layers=[(w, np.zeros(3, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9),
                        actions=("hold", "limit_buy", "sell"))

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            return {"f": 1.0}

    sh = RLShadow(pol, stream_factory=Stream, sink=rows.append, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 17, 9, 10, 0)
    for s in (0, 1, 2):
        ts = t.replace(second=s)
        sh.on_quote("A", ts, bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
        sh.on_tick("A", ts, 1005, volume=1, side=1)
    assert [r for r in rows if r["event"] == "limit_place"], rows


# ── 커널이 아직 안 데워진 것과 구조적으로 못 주는 것을 구분한다 ──────────────
# 2026-09-17 실측: 장 초반 창 커널이 빈 dict 를 돌려주자 그림자가 "피처를 못 준다"로 보고
# **하루 종일 추론을 껐다**(기록 0건). 준비 안 됨은 그 초를 건너뛰면 되고, 구조적 결손만 꺼야 한다.
def _shadow_with_kernel(rows, kernel_out_seq):
    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    w = np.zeros((4, 5), np.float32)
    w[1] = 1.0                                   # 항상 시장가 매수
    pol = PolicyWeights(features=["f"], layers=[(w, np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))

    class Stream:
        def __init__(self, code=None):
            self.sec = 0
            self.i = 0

        def update(self, **kw):
            out = kernel_out_seq[min(self.i, len(kernel_out_seq) - 1)]
            self.i += 1
            return dict(out)

    return RLShadow(pol, stream_factory=Stream, sink=rows.append, codes={"A"},
                    extra_features=lambda c, s: {})


def _feed_secs(sh, secs):
    from datetime import datetime

    t = datetime(2026, 9, 17, 9, 0, 0)
    for s in secs:
        ts = t.replace(minute=s // 60, second=s % 60)
        sh.on_quote("A", ts, bid1=1000, ask1=1010, bidqty1=5, askqty1=50)
        sh.on_tick("A", ts, 1005, volume=1, side=1)


def test_empty_kernel_output_does_not_disable_for_the_day():
    """창이 안 데워진 초는 건너뛰기만 한다 — 나중에 값이 오면 다시 돈다."""
    rows = []
    sh = _shadow_with_kernel(rows, [{}, {}, {"f": 1.0}])
    _feed_secs(sh, [0, 1, 2, 3, 4])
    assert sh.disabled_reason is None, f"준비 안 됨을 결손으로 오해했다: {sh.disabled_reason}"
    assert [r for r in rows if r["event"] == "buy"], f"창이 데워진 뒤에도 안 돌았다: {rows}"


def test_structural_missing_feature_still_disables():
    """값은 주는데 정책이 요구하는 열이 없으면 그건 진짜 결손이다 — 꺼야 한다."""
    rows = []
    sh = _shadow_with_kernel(rows, [{"other": 1.0}])
    _feed_secs(sh, [0, 1, 2, 3])
    assert sh.disabled_reason and "f" in sh.disabled_reason
    assert not [r for r in rows if r["event"] == "buy"]


def test_disabling_is_announced_immediately_not_at_shutdown():
    """꺼지면 그 자리에서 기록한다 — 15:20 종료 때 알면 하루를 버린다."""
    rows = []
    sh = _shadow_with_kernel(rows, [{"other": 1.0}])
    _feed_secs(sh, [0, 1, 2, 3])
    off = [r for r in rows if r["event"] == "disabled"]
    assert off and "f" in off[0]["reason"]
    assert len(off) == 1, "매 초 반복해서 남기면 로그가 덮인다"


# ── v2 증분 커널은 그 초의 체결 배열을 요구한다 ───────────────────────────────
# 창 재계산 커널은 초당 55~129ms 라 매매 프로세스 안에서 못 돈다(09-16~18 에 수집까지 갉아먹었다).
# core 0.5.1 의 SecondFeatureStreamV2 는 증분이라 훨씬 싸지만, 집계값이 아니라 **그 초의 체결**을 받는다.
def test_shadow_passes_this_seconds_trades_to_the_stream():
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    seen = []

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            seen.append({k: list(v) if k in ("price", "volume", "side") else v
                         for k, v in kw.items() if k in ("price", "volume", "side")})
            return {"f": 1.0}

    pol = PolicyWeights(features=["f"], layers=[(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))
    sh = RLShadow(pol, stream_factory=Stream, sink=lambda r: None, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 23, 9, 10, 0)
    sh.on_quote("A", t, bid1=1000, ask1=1010, bidqty1=5, askqty1=5)
    sh.on_tick("A", t, 1005, volume=3, side=1)
    sh.on_tick("A", t, 1006, volume=4, side=-1)
    sh.on_tick("A", t.replace(second=1), 1007, volume=5, side=1)   # 초가 넘어가며 0초 마감
    assert seen, "스트림이 안 불렸다"
    first = seen[0]
    assert first["price"] == [1005.0, 1006.0], first
    assert first["volume"] == [3.0, 4.0], first
    assert first["side"] == [1, -1], first


def test_this_seconds_trades_reset_each_second():
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    seen = []

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            seen.append(list(kw.get("price", ())))
            return {"f": 1.0}

    pol = PolicyWeights(features=["f"], layers=[(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))
    sh = RLShadow(pol, stream_factory=Stream, sink=lambda r: None, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 23, 9, 10, 0)
    for s, px in ((0, 1000.0), (1, 1001.0), (2, 1002.0)):
        sh.on_quote("A", t.replace(second=s), bid1=1000, ask1=1010, bidqty1=5, askqty1=5)
        sh.on_tick("A", t.replace(second=s), px, volume=1, side=1)
    sh.on_quote("A", t.replace(second=3), bid1=1000, ask1=1010, bidqty1=5, askqty1=5)
    assert seen[:3] == [[1000.0], [1001.0], [1002.0]], f"초마다 비워지지 않았다: {seen[:3]}"


# ── 커널 입력은 "그 초에 실제로 온 값"이어야 한다 ─────────────────────────────
# 배치(aggregate_seconds_v2)는 호가가 없는 초에 NaN 을 넣는다(069500/09-22 에서 22,800초 중 3,881초).
# 그림자가 직전 값을 물고 넘기면 값이 달라져 패리티가 무의미해진다. 반면 체결 판정·평가손익은
# 직전 호가를 써야 한다 — 두 용도를 분리한다.
def test_kernel_gets_nan_for_fields_that_did_not_arrive_this_second():
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    seen = []

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            seen.append({k: kw.get(k) for k in ("bid", "ask", "bidqty1", "strength")})
            return {"f": 1.0}

    pol = PolicyWeights(features=["f"], layers=[(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))
    sh = RLShadow(pol, stream_factory=Stream, sink=lambda r: None, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 23, 9, 10, 0)
    sh.on_quote("A", t, bid1=1000, ask1=1010, bidqty1=7, askqty1=8)
    sh.on_tick("A", t, 1005, volume=1, side=1, strength=120.0)
    # 1초: 체결만 있고 호가는 없다
    sh.on_tick("A", t.replace(second=1), 1006, volume=1, side=1)
    # 2초: 아무것도 없다 → 1초 마감이 일어나게 호가 하나
    sh.on_quote("A", t.replace(second=2), bid1=1000, ask1=1010, bidqty1=9, askqty1=9)

    assert seen[0]["bid"] == 1000.0 and seen[0]["bidqty1"] == 7.0
    assert seen[0]["strength"] == 120.0
    s1 = seen[1]
    assert np.isnan(s1["bid"]) and np.isnan(s1["ask"]), f"호가 없는 초에 직전 값을 넘겼다: {s1}"
    assert np.isnan(s1["bidqty1"]), s1
    assert np.isnan(s1["strength"]), "체결강도도 그 초에 온 것만 넘겨야 한다"


def test_fill_logic_still_uses_the_last_known_quote():
    """커널 입력은 NaN 이어도 체결·평가손익은 직전 호가로 계속 돌아야 한다."""
    from datetime import datetime

    import numpy as np

    from scalp_it.realtime.rl_shadow import PolicyWeights, RLShadow

    rows = []
    w = np.zeros((4, 5), np.float32)
    w[1] = 1.0                                     # 시장가 매수

    class Stream:
        def __init__(self, code=None):
            self.sec = 0

        def update(self, **kw):
            return {"f": 1.0}

    pol = PolicyWeights(features=["f"], layers=[(w, np.zeros(4, np.float32))],
                        mu=np.zeros(1), sd=np.ones(1), lo=np.full(1, -1e9), hi=np.full(1, 1e9))
    sh = RLShadow(pol, stream_factory=Stream, sink=rows.append, codes={"A"},
                  extra_features=lambda c, s: {})
    t = datetime(2026, 9, 23, 9, 10, 0)
    sh.on_quote("A", t, bid1=1000, ask1=1010, bidqty1=5, askqty1=5)
    sh.on_tick("A", t, 1005, volume=1, side=1)
    sh.on_tick("A", t.replace(second=1), 1006, volume=1, side=1)   # 호가 없는 초
    sh.on_tick("A", t.replace(second=2), 1007, volume=1, side=1)
    buys = [r for r in rows if r["event"] == "buy"]
    assert buys and buys[0]["price"] == 1010.0, f"직전 매도1호가로 체결돼야 한다: {rows}"
