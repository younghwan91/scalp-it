"""v2 증분 커널 어댑터 — 그림자 규약(update(**많은 인자))을 v2 시그니처로 좁힌다.

창 재계산 커널은 초당 55~129ms 라 매매 프로세스 안에서 못 돈다(09-16~18 에 수집까지 갉아먹었다).
v2 는 증분이라 싸지만 인자가 다르다. 여기서 한 칸 어긋나면 값이 조용히 달라져 패리티가 무의미해진다.
"""
import inspect

from scalp_it.cli_pair_detect import _V2Adapter


class FakeV2:
    def __init__(self):
        self.calls = []

    def update(self, *, bid=float("nan"), ask=float("nan"), strength=float("nan"),
               bidqty1=float("nan"), askqty1=float("nan"), bid_depth3=float("nan"),
               ask_depth3=float("nan"), bidqty1_raw=float("nan"), askqty1_raw=float("nan"),
               price=(), volume=(), side=()):
        self.calls.append(dict(bid=bid, ask=ask, strength=strength, bidqty1=bidqty1,
                               askqty1=askqty1, bid_depth3=bid_depth3, ask_depth3=ask_depth3,
                               bidqty1_raw=bidqty1_raw, askqty1_raw=askqty1_raw,
                               price=tuple(price), volume=tuple(volume), side=tuple(side)))
        return {"spread_bp": 1.0}


def test_drops_arguments_v2_does_not_take():
    inner = FakeV2()
    a = _V2Adapter(inner)
    out = a.update(buy=1.0, sell=2.0, cnt=3.0, bid=100.0, ask=101.0, strength=110.0,
                   bidqty1=5.0, askqty1=6.0, bid_depth3=7.0, ask_depth3=8.0,
                   price=(100.5,), volume=(2.0,), side=(1,))
    assert out == {"spread_bp": 1.0}
    c = inner.calls[0]
    assert c["bid"] == 100.0 and c["ask"] == 101.0
    assert c["price"] == (100.5,) and c["volume"] == (2.0,) and c["side"] == (1,)


def test_raw_quantities_default_to_the_snapshot_values():
    """core 규약: *_raw 는 그 초의 마지막 스냅샷. 우리 그림자가 들고 있는 값이 곧 그것이다."""
    inner = FakeV2()
    _V2Adapter(inner).update(bidqty1=5.0, askqty1=6.0)
    c = inner.calls[0]
    assert c["bidqty1_raw"] == 5.0 and c["askqty1_raw"] == 6.0


def test_adapter_only_forwards_names_the_inner_update_accepts():
    """v2 가 인자를 바꾸면 어댑터가 조용히 깨지지 않고 여기서 걸린다."""
    inner = FakeV2()
    allowed = set(inspect.signature(inner.update).parameters)
    a = _V2Adapter(inner)
    a.update(buy=1.0, 엉뚱한인자=2.0, bid=1.0)
    assert set(inner.calls[0]) <= allowed
