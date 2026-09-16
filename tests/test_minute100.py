"""100번 분봉 피처 — **미래를 안 보는지**와 라벨 창이 정확히 30분인지.

83~96 이 17일로 돌아 검정력이 없었다. 표본을 20배로 키우는 만큼, 여기서 한 칸 새면
거짓 신호가 20배 더 그럴듯하게 나온다.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve().parents[1] / "scripts" / "minute_100" / "build.py"


def _mod():
    spec = importlib.util.spec_from_file_location("minute100", _p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["minute100"] = m
    spec.loader.exec_module(m)
    return m


def _bars(n=300, seed=5):
    rng = np.random.default_rng(seed)
    px = 10000 + np.cumsum(rng.normal(0, 8, n))
    t0 = pd.Timestamp("2026-05-04 09:00")
    return pd.DataFrame({"code": "A", "ts": [t0 + pd.Timedelta(minutes=i) for i in range(n)],
                         "open": px, "high": px + 5, "low": px - 5, "close": px,
                         "volume": 1000, "trade_value": px * 1000, "prev_close": 10000.0})


def test_label_is_exactly_thirty_minutes_ahead_minus_cost():
    m = _mod()
    b = _bars()
    f = m.features_for_day(b)
    row = f.iloc[0]
    i = int(row.minute - 9 * 60)                       # 09:00 시작이라 분 인덱스와 같다
    c = b.close.to_numpy()
    assert abs(row.y - ((c[i + m.HORIZON] / c[i] - 1 - m.COST) * 1e4)) < 1e-6


def test_features_do_not_see_the_future():
    m = _mod()
    b = _bars()
    a = m.features_for_day(b)
    b2 = _bars()
    cut = 200
    for col in ("open", "high", "low", "close", "trade_value"):
        b2.loc[cut:, col] = b2.loc[cut:, col] * 3 + 1
    c = m.features_for_day(b2)
    cols = [x for x in a.columns if x not in ("y", "code", "ts")]
    early = a.minute < 9 * 60 + cut
    assert early.sum() > 5
    for col in cols:
        x, y = a.loc[early, col].to_numpy(float), c.loc[early.values, col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        assert np.allclose(x[ok], y[ok], atol=1e-6), f"{col} 이 미래를 본다"


def test_decision_points_are_every_five_minutes_in_window():
    m = _mod()
    f = m.features_for_day(_bars(400))
    mins = f.minute.to_numpy()
    assert ((mins - m.LO_MIN) % m.STEP == 0).all()
    assert mins.min() >= m.LO_MIN and mins.max() <= m.HI_MIN


def test_cost_is_subtracted_once():
    m = _mod()
    b = _bars()
    b[["open", "high", "low", "close"]] = 10000.0
    f = m.features_for_day(b)
    assert abs(f.y.iloc[0] + m.COST * 1e4) < 1e-6
