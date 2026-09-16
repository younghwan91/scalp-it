"""98번 심화 — 귀무분포·분산분해·대안 규칙 산술을 고정한다.

여기가 어긋나면 "이 정도 프리미엄은 신호 없이도 나온다"는 문장이 근거를 잃는다. 그래서
거래 표를 손으로 만들어 답을 아는 경우로 박는다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "selection98_deep", Path(__file__).resolve().parents[1] / "scripts" / "selection_98" / "deep.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _t(days, rets):
    return pd.DataFrame({"day": days, "ret": np.asarray(rets, float) / 1e4})


# ── 네 가지 선택 기준 ───────────────────────────────────────────────────────
def test_daymean_averages_days_not_trades():
    t = _t(["a", "b", "b", "b"], [300, -10, -10, -10])   # 하루 한 건이 절반의 무게를 갖는다
    assert abs(D.score(t, "daymean") - (300 - 10) / 2) < 1e-9


def test_pooled_averages_trades():
    t = _t(["a", "b", "b", "b"], [300, -10, -10, -10])
    assert abs(D.score(t, "pooled") - (300 - 30) / 4) < 1e-9


def test_trade_weighted_daymean_equals_pooled():
    """거래 가중 일평균은 정의상 풀링 평균과 같다 — 같은 값이 나와야 잣대가 맞다."""
    t = _t(["a", "b", "b", "b"], [300, -10, -10, -10])
    assert abs(D.score(t, "wdaymean") - D.score(t, "pooled")) < 1e-9


def test_min_trades_per_day_drops_thin_days():
    t = _t(["a"] + ["b"] * 10, [300] + [-10] * 10)
    assert abs(D.score(t, "daymean_min10") - (-10)) < 1e-9, "하루 1건짜리 날이 살아남으면 안 된다"


def test_min_trades_per_day_is_nan_when_no_day_qualifies():
    t = _t(["a", "b"], [300, -10])
    assert np.isnan(D.score(t, "daymean_min10"))


# ── 하루 거래 k 건일 때 일평균의 표준오차 ──────────────────────────────────
def test_day_se_falls_as_one_over_sqrt_k():
    se4 = D.day_mean_se(sigma_bp=100.0, k=4)
    se16 = D.day_mean_se(sigma_bp=100.0, k=16)
    assert abs(se4 - 50.0) < 1e-9 and abs(se16 - 25.0) < 1e-9


def test_daymean_se_combines_days_independently():
    """3일 일평균의 표준오차 = (1/3)·sqrt(Σ σ²/k_d) — 얇은 하루가 전체를 지배한다."""
    got = D.daymean_se(100.0, [1, 100, 100])
    assert abs(got - (1 / 3) * np.sqrt(100.0**2 / 1 + 100.0**2 / 100 + 100.0**2 / 100)) < 1e-9
    assert got > D.daymean_se(100.0, [67, 67, 67]), "같은 총 거래라도 쏠리면 더 흔들린다"


# ── 귀무분포: 거래의 날짜 딱지만 섞는다 ────────────────────────────────────
def test_permute_days_keeps_counts_and_pooled_mean():
    rng = np.random.default_rng(0)
    t = _t(["a", "a", "b", "c", "c", "c"], [10, -20, 30, -40, 50, -60])
    p = D.permute_days(t, rng)
    assert p.day.value_counts().to_dict() == t.day.value_counts().to_dict()
    assert abs(D.score(p, "pooled") - D.score(t, "pooled")) < 1e-9, "풀링 평균은 섞어도 안 변한다"


def test_permute_days_changes_daymean():
    """날짜 딱지를 섞으면 일평균은 흔들린다 — 그 흔들림이 곧 '신호 없음'의 폭이다."""
    rng = np.random.default_rng(1)
    t = _t(["a"] + ["b"] * 20, [500] + [-20] * 20)
    vals = [D.score(D.permute_days(t, rng), "daymean") for _ in range(50)]
    assert np.std(vals) > 10.0


def test_null_max_premium_is_positive_by_construction():
    rng = np.random.default_rng(2)
    trades = {(1, i): _t(["a"] * 10 + ["b"] * 10 + ["c"] * 10, rng.normal(0, 200, 30)) for i in range(8)}
    null = D.null_max(trades, "daymean", n_rep=30, rng=rng)
    assert null["max"].mean() > null["pool_mean"].mean(), "최댓값이 평균보다 낮으면 계산이 틀렸다"
