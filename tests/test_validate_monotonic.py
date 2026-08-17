import numpy as np
import pandas as pd

from swing_it.validate.monotonic import decile_table, monotonicity


def _pred(n: int, signal: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prediction = rng.normal(size=n)
    excess = prediction * signal + rng.normal(scale=0.02, size=n)
    return pd.DataFrame({
        "code": [f"{i % 50:06d}" for i in range(n)],
        "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D"),
        "excess_return": excess,
        "prediction": prediction,
        "fold": 0,
    })


def test_decile_table_shape():
    table = decile_table(_pred(2000, 0.05))
    assert list(table.columns) == ["decile", "n", "mean_excess", "median_excess"]
    assert len(table) == 10
    assert table["decile"].tolist() == list(range(10))


def test_monotone_when_prediction_carries_signal():
    got = monotonicity(_pred(2000, 0.05))
    assert got["is_monotone"] is True
    assert got["spearman"] > 0.6
    assert got["top_minus_bottom"] > 0


def test_not_monotone_when_prediction_is_noise():
    got = monotonicity(_pred(2000, 0.0))
    assert got["is_monotone"] is False


def test_inverted_signal_is_not_monotone():
    """예측이 거꾸로면 단조가 아니다 — 상위를 골라 들어갈 수 없다."""
    pred = _pred(2000, 0.05)
    pred["prediction"] = -pred["prediction"]
    assert monotonicity(pred)["is_monotone"] is False


def test_report_has_all_keys():
    got = monotonicity(_pred(500, 0.05))
    for key in ("spearman", "top_minus_bottom", "top_vs_median", "is_monotone"):
        assert key in got


def _multi_fold(n_per_fold: int = 500, seed: int = 0) -> pd.DataFrame:
    """폴드마다 예측값 스케일이 다른 프레임.

    실제로 이렇게 된다 — 폴드마다 다른 학습 데이터로 적합하므로 예측값의
    절대 수준이 폴드마다 다르다. 전역 풀링으로 분위를 나누면 스케일이 섞인다.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for fold, (shift, scale) in enumerate([(0.0, 1.0), (2.0, 1.0), (-1.0, 0.5), (0.5, 2.0)]):
        pred = rng.normal(size=n_per_fold) * scale + shift
        # 폴드 안에서는 예측이 결과를 잘 설명한다
        excess = (pred - shift) / scale * 0.05 + rng.normal(scale=0.01, size=n_per_fold)
        blocks.append(pd.DataFrame({
            "code": [f"{i % 40:06d}" for i in range(n_per_fold)],
            "date": pd.to_datetime("2024-01-01")
                    + pd.to_timedelta(fold * 200 + np.arange(n_per_fold) // 5, unit="D"),
            "excess_return": excess,
            "prediction": pred,
            "fold": fold,
        }))
    return pd.concat(blocks, ignore_index=True)


def test_deciles_are_computed_within_fold():
    """폴드별 분위여야 한다. 전역 분위는 폴드 간 스케일 차이에 오염된다."""
    table = decile_table(_multi_fold())
    assert list(table.columns) == ["decile", "n", "mean_excess", "median_excess"]
    assert len(table) == 10


def test_fold_scale_shift_does_not_break_monotonicity():
    """폴드 안에서 신호가 있으면, 스케일이 달라도 단조로 판정돼야 한다.

    회귀 방지: 전역 풀링으로 분위를 나누면 최상위 분위가 '예측 스케일이 큰
    폴드의 행'으로 채워져 단조가 깨진다. 실측에서 그렇게 됐다.
    """
    got = monotonicity(_multi_fold())
    assert got["is_monotone"] is True
    assert got["top_decile_is_best"] is True


def test_top_decile_reversal_is_reported():
    """최상위 분위가 최고가 아니면 그 사실이 드러나야 한다.

    스피어만만 보면 0.8 을 넘으면서도 꼭대기가 꺾일 수 있고, 그러면
    '확실한 것만 골라 들어간다'가 성립하지 않는다.
    """
    rng = np.random.default_rng(1)
    n = 2000
    pred = rng.normal(size=n)
    # 상위 5% 만 뒤집는다
    excess = pred * 0.05 + rng.normal(scale=0.01, size=n)
    cut = np.quantile(pred, 0.95)
    excess = np.where(pred >= cut, -0.05, excess)
    frame = pd.DataFrame({
        "code": [f"{i % 40:06d}" for i in range(n)],
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D"),
        "excess_return": excess, "prediction": pred, "fold": 0,
    })
    got = monotonicity(frame)
    assert got["top_decile_is_best"] is False
    assert got["is_monotone"] is False


def test_is_monotone_requires_top_decile_best():
    got = monotonicity(_multi_fold())
    assert got["is_monotone"] == (got["spearman"] >= 0.6
                                  and got["top_minus_bottom"] > 0
                                  and got["top_decile_is_best"])
