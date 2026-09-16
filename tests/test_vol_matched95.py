"""95h번 변동성 매칭 — 대조가 성립하려면 **변동성이 결정 시점 정보**여야 한다.

미래 초가 들어가면 매칭 자체가 look-ahead 가 되고, 결론(딥 체결이 정보인가)이 통째로 거짓이 된다.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "ceiling_95" / "vol_matched.py"
spec = importlib.util.spec_from_file_location("vol_matched95", _p)
V = importlib.util.module_from_spec(spec)
sys.modules["vol_matched95"] = V
spec.loader.exec_module(V)


def test_realized_vol_does_not_see_the_future():
    rng = np.random.default_rng(1)
    mid = 10000 + np.cumsum(rng.normal(0, 3, 2000))
    a = V.pre_entry_rv(mid)
    m2 = mid.copy()
    m2[1500:] *= 5                                   # 미래만 크게 흔든다
    b = V.pre_entry_rv(m2)
    ok = np.isfinite(a[:1500]) & np.isfinite(b[:1500])
    assert np.allclose(a[:1500][ok], b[:1500][ok]), "변동성이 미래를 본다 — 매칭이 무의미해진다"


def test_needs_enough_history_before_reporting():
    mid = np.full(100, 10000.0)
    v = V.pre_entry_rv(mid)
    assert np.isnan(v[:29]).all(), "표본이 모자란 구간은 값을 내면 안 된다"


def test_higher_volatility_gives_higher_measure():
    rng = np.random.default_rng(2)
    calm = 10000 + np.cumsum(rng.normal(0, 1, 1000))
    wild = 10000 + np.cumsum(rng.normal(0, 10, 1000))
    assert np.nanmean(V.pre_entry_rv(wild)) > np.nanmean(V.pre_entry_rv(calm))
