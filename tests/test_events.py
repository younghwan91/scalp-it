import numpy as np
import pandas as pd

from scalp_it.events import add_liquidity_stats, extract_events


def _prices(code: str, trade_values: list[float]) -> pd.DataFrame:
    n = len(trade_values)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [100.0] * n,
            "close": [100.0] * n,
            "trade_value": trade_values,
        }
    )


def test_adv_excludes_current_day():
    df = _prices("A", [10.0, 20.0, 30.0, 999.0])
    out = add_liquidity_stats(df, window=3)
    # 4번째 행의 adv 는 앞의 3일 평균 (10+20+30)/3 = 20 이어야 한다. 999 가 섞이면 lookahead.
    assert out.iloc[3]["adv"] == 20.0
    assert out.iloc[3]["tv_ratio"] == 999.0 / 20.0


def test_adv_is_nan_before_window_fills():
    df = _prices("A", [10.0, 20.0, 30.0])
    out = add_liquidity_stats(df, window=3)
    assert np.isnan(out.iloc[0]["adv"])
    assert np.isnan(out.iloc[2]["adv"])  # 앞 2일뿐이라 window=3 미충족


def test_stats_computed_per_code_independently():
    df = pd.concat([_prices("A", [10.0] * 4), _prices("B", [1000.0] * 4)], ignore_index=True)
    out = add_liquidity_stats(df, window=3)
    a = out[out["code"] == "A"].iloc[3]
    b = out[out["code"] == "B"].iloc[3]
    assert a["adv"] == 10.0
    assert b["adv"] == 1000.0


def test_extract_events_applies_both_gates():
    # adv 6000 (>= 5000 통과), 마지막 날 거래대금 24000 → 배수 4.0 (>= 3.0 통과)
    df = _prices("A", [6000.0, 6000.0, 6000.0, 24000.0])
    events = extract_events(df, adv_floor=5000.0, tv_multiple=3.0, window=3)
    assert list(events.columns) == ["code", "date", "adv", "tv_ratio"]
    assert len(events) == 1
    assert events.iloc[0]["tv_ratio"] == 4.0


def test_extract_events_rejects_illiquid_even_when_spiking():
    # 배수는 10배지만 adv 가 100 뿐이라 유동성 하한에서 탈락해야 한다
    df = _prices("A", [100.0, 100.0, 100.0, 1000.0])
    events = extract_events(df, adv_floor=5000.0, tv_multiple=3.0, window=3)
    assert events.empty
