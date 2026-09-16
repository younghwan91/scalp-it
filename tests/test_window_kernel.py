"""창 재계산 커널 — 실시간 값이 배치와 **같은 함수**에서 나오는지 확인한다.

core v2 증분 커널이 나오기 전의 다리. 값이 다르면 의사결정 패리티(승격 9번)가 무의미해지므로,
여기서는 "배치 전체를 한 번 돌린 결과"와 "창으로 초마다 돌린 결과"가 같은지 본다.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="실시간 창 커널은 백테스트와 같은 함수를 쓰고, 그 함수가 numba 를 쓴다")

from scalp_it.realtime.window_kernel import WindowKernelStream, load_batch_modules  # noqa: E402


def _synth(n_sec=900, seed=7):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 9, 17, 9, 0, 0)
    ticks, quotes = [], []
    px = 10000
    for s in range(n_sec):
        px = max(1000, px + int(rng.integers(-2, 3)) * 10)
        for _ in range(int(rng.integers(0, 4))):
            ticks.append((t0 + timedelta(seconds=s), px, float(rng.integers(1, 30)),
                          int(rng.choice([1, -1])), 100 + s * 0.01, px - 10, px))
        quotes.append((t0 + timedelta(seconds=s), px - 10, px, *rng.integers(1, 500, 6).astype(float)))
    return ticks, quotes


def test_window_features_equal_full_batch():
    f83, f84 = load_batch_modules()
    ticks, quotes = _synth()
    tk = pd.DataFrame(ticks, columns=["ts", "price", "volume", "side", "strength", "best_bid", "best_ask"])
    tk.insert(0, "code", "TEST")
    tk["seq"] = range(len(tk))
    qt = pd.DataFrame(quotes, columns=["ts", "bid1", "ask1", "bidqty1", "bidqty2", "bidqty3",
                                       "askqty1", "askqty2", "askqty3"])
    qt.insert(0, "code", "TEST")
    f, p = f83.build_code(tk, qt)
    x = f84._extra(tk, qt, f, p)

    stream = WindowKernelStream("TEST", day="2026-09-17")
    for row in ticks:
        stream.add_tick(*row)
    for row in quotes:
        stream.add_quote(*row)

    checked = 0
    for sec in (700, 800, 890):                      # 창(700초)이 다 찬 뒤 구간만 비교
        live = stream.update(sec)
        for name, arr in list(x.items())[:12] + [(k, f[k]) for k in ("spread_bp", "imb3", "ret300", "dstr30")]:
            ref = float(arr[sec])
            got = live.get(name)
            if got is None or (np.isnan(ref) and np.isnan(got)):
                continue
            assert abs(ref - got) < 1e-6, f"{name}@{sec}: 배치 {ref} vs 창 {got}"
            checked += 1
    assert checked > 20, "비교한 항목이 너무 적다"


def test_empty_input_returns_empty_dict():
    stream = WindowKernelStream("TEST", day="2026-09-17")
    assert stream.update(100) == {}


@pytest.mark.parametrize("sec", [750, 860])
def test_trim_keeps_window_bounded(sec):
    ticks, quotes = _synth()
    stream = WindowKernelStream("TEST", day="2026-09-17", window_sec=300)
    for row in ticks:
        stream.add_tick(*row)
    for row in quotes:
        stream.add_quote(*row)
    stream.update(sec)
    kept = [t[0].minute * 60 + t[0].second for t in stream.ticks]
    assert not kept or min(kept) >= sec - 300, "창 밖 데이터가 남아 메모리가 샌다"
