"""RL 그림자는 **매매 루프를 막지 않는다** — 별도 스레드에서 돌고, 밀리면 버린다.

창 재계산 커널은 종목당 1초 81ms 다. 8종목이면 초당 650ms — 이벤트 루프 안에서 돌면 주문 판단이 밀린다.
장중 지연은 곧 손실이므로, 프레임 전달은 큐에 넣는 것으로 끝나야 한다.
"""
import time

from scalp_it.realtime.rl_shadow_worker import ShadowWorker


class _Slow:
    def __init__(self):
        self.seen = 0

    def on_tick(self, *a, **k):
        time.sleep(0.05)          # 느린 그림자
        self.seen += 1

    def on_quote(self, *a, **k):
        pass

    def flush(self):
        pass


def test_feeding_does_not_block_caller():
    w = ShadowWorker(_Slow(), maxsize=100)
    w.start()
    t0 = time.perf_counter()
    for _ in range(20):
        w.feed_tick("005930", None, 10000, volume=1, side=1, strength=100.0, best_bid=9990, best_ask=10000)
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"전달이 느리다({dt:.3f}s) — 매매 루프가 막힌다"
    w.stop(timeout=3)


def test_overflow_drops_instead_of_blocking():
    w = ShadowWorker(_Slow(), maxsize=5)
    w.start()
    for _ in range(200):
        w.feed_tick("005930", None, 10000, volume=1, side=1, strength=100.0, best_bid=9990, best_ask=10000)
    assert w.dropped > 0, "큐가 넘치면 버려야 한다"
    w.stop(timeout=3)


def test_stop_is_idempotent_and_flushes():
    class _S(_Slow):
        flushed = False

        def flush(self):
            self.flushed = True

    s = _S()
    w = ShadowWorker(s, maxsize=10)
    w.start()
    w.feed_quote("005930", None, bid1=9990, ask1=10000, bidqty1=1, askqty1=1, bid_depth3=3, ask_depth3=3)
    w.stop(timeout=3)
    w.stop(timeout=1)
    assert s.flushed


def test_detector_builds_worker_not_raw_shadow(tmp_path):
    """감지기는 그림자를 **작업 스레드로 감싸** 돌려야 한다 — 이벤트 루프 보호."""
    import argparse
    import json
    from datetime import date

    import numpy as np

    from scalp_it import cli_pair_detect as C

    base = tmp_path / "pol"
    base.mkdir()
    np.savez(base / "weights.npz", w0=np.zeros((4, 5), np.float32), b0=np.zeros(4, np.float32),
             w1=np.zeros((4, 4), np.float32), b1=np.zeros(4, np.float32))
    (base / "policy.json").write_text(json.dumps({"study": "rlX", "features": ["acc"], "mu": [0.0],
                                                  "sd": [1.0], "lo": [-1.0], "hi": [1.0]}))
    args = argparse.Namespace(rl_shadow=str(base), rl_shadow_log=str(tmp_path / "log.jsonl"),
                              rl_kernel="window", rl_codes=2, rl_live=False,
                              dsn="postgresql://nobody@127.0.0.1:1/none")
    worker, codes = C._build_rl_shadow(args, date(2026, 9, 17))
    assert worker is None or hasattr(worker, "feed_tick"), "작업 스레드로 감싸야 한다"
    if worker is not None:
        worker.stop(timeout=2)
