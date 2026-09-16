"""96번 분 단위 피처 — **미래를 보지 않는지**가 전부다. 여기서 한 칸 새면 전진 판정이 통째로 거짓이 된다."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "minute96", Path(__file__).resolve().parents[1] / "scripts" / "minute_96" / "features.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _path(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    mid = 10000 + np.cumsum(rng.normal(0, 5, n))
    return {"bid": (mid - 5).astype(np.float32), "ask": (mid + 5).astype(np.float32),
            "last": mid.astype(np.float32), "hi": mid.astype(np.float32),
            "tick": np.full(n, 10, np.float32),
            "buyval": rng.uniform(0, 1e6, n).astype(np.float32),
            "sellval": rng.uniform(0, 1e6, n).astype(np.float32),
            "strength": (100 + rng.normal(0, 10, n)).astype(np.float32),
            "e_ofi": rng.normal(0, 100, n).astype(np.float32)}


def test_no_lookahead_future_change_does_not_move_past():
    p = _path()
    a = M.minute_features(p)
    q = {k: v.copy() for k, v in p.items()}
    for k in ("bid", "ask", "last", "buyval", "sellval", "strength", "e_ofi"):
        q[k][2000:] = q[k][2000:] * 3 + 1          # 미래만 크게 바꾼다
    b = M.minute_features(q)
    for name in M.FEATURES:
        assert np.allclose(a[name][:2000], b[name][:2000], atol=1e-6), f"{name} 이 미래를 본다"


def test_returns_are_basis_points_of_the_right_lag():
    n = 3000
    mid = np.linspace(10000, 11000, n)
    p = _path(n)
    p["bid"] = (mid - 5).astype(np.float32)
    p["ask"] = (mid + 5).astype(np.float32)
    x = M.minute_features(p)
    t = 2000
    exp = (mid[t] / mid[t - 60] - 1) * 1e4
    assert abs(x["ret60"][t] - exp) < 0.5


def test_range_position_bounded_and_drawdown_nonpositive():
    x = M.minute_features(_path())
    for w in M.RANGE_W:
        v = x[f"rangepos{w}"]
        assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9
    assert x["dd_hi"].max() <= 1e-6, "당일 고가 대비 낙폭은 양수가 될 수 없다"


def test_every_declared_feature_is_finite():
    x = M.minute_features(_path())
    assert set(x) == set(M.FEATURES)
    for k, v in x.items():
        assert np.isfinite(v).all(), f"{k} 에 NaN/inf 가 남았다 — 학습이 조용히 망가진다"
