# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""`scripts/tick_sanity.py` 단위테스트 — 트리거 판정·라벨 계산·빈 데이터.

scripts 는 패키지가 아니라 importlib 로 파일에서 직접 로드한다. 운영은 postgres 지만
테스트는 sqlite fixture(ticks 스키마 동일)로 돈다.
"""
import importlib.util
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from scalp_it.realtime.store import init_realtime_tables, insert_ticks

_SPEC = importlib.util.spec_from_file_location(
    "tick_sanity", Path(__file__).resolve().parents[1] / "scripts" / "tick_sanity.py"
)
ts_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ts_mod)


# --------------------------------------------------------- 순수 함수: trailing


def test_trailing_features_window_and_buy_ratio():
    # ts 0,1,2,3,4 초. side 매수/매도 섞임. window=2 초.
    ts = np.array([0, 1, 2, 3, 4])
    side = np.array([1, -1, 1, 1, -1])
    vol = np.array([10.0, 10, 10, 10, 10])
    strength = np.array([100.0, 100, 100, 100, 100])
    ms, br, cnt = ts_mod.trailing_features(ts, side, vol, strength, window=2)
    # i=0: 윈도 {0}. count=1
    assert cnt[0] == 1
    # i=4: 윈도 = ts in (2,4] => idx 3,4. count=2
    assert cnt[4] == 2
    # 매수비율 i=4: buy_vol(idx3=+1)=10, 총20 => 0.5
    assert br[4] == pytest.approx(0.5)
    # 강도 평균은 전부 100
    assert ms[4] == pytest.approx(100.0)


def test_trailing_buy_ratio_all_buy():
    ts = np.array([0, 1, 2])
    side = np.array([1, 1, 1])
    vol = np.array([5.0, 5, 5])
    strength = np.array([50.0, 60, 70])
    _, br, _ = ts_mod.trailing_features(ts, side, vol, strength, window=10)
    assert br[2] == pytest.approx(1.0)


# --------------------------------------------------------- 트리거 판정


def test_find_triggers_conditions_and_cooldown():
    # 강도·매수비율이 임계 넘는 지점이 여러 개지만 쿨다운으로 솎인다.
    ts = np.array([0, 5, 100, 105])
    ms = np.array([90.0, 90, 90, 90])          # 전부 임계 이상
    br = np.array([0.9, 0.9, 0.9, 0.9])
    cnt = np.array([10, 10, 10, 10])
    idx = ts_mod.find_trigger_indices(
        ms, br, cnt, ts, strength_thr=80.0, buy_thr=0.6, min_count=5, cooldown=50
    )
    # 0 채택 → 5 는 쿨다운(50s) 안이라 제외 → 100 채택 → 105 제외
    assert idx == [0, 2]


def test_find_triggers_gate_and_thresholds():
    ts = np.array([0, 1, 2, 3])
    ms = np.array([90.0, 70, 90, 90])   # idx1 강도 미달
    br = np.array([0.9, 0.9, 0.4, 0.9])  # idx2 매수비율 미달
    cnt = np.array([10, 10, 10, 2])      # idx3 활동게이트 미달
    idx = ts_mod.find_trigger_indices(
        ms, br, cnt, ts, strength_thr=80.0, buy_thr=0.6, min_count=5, cooldown=0
    )
    assert idx == [0]


def test_percentile_threshold_respects_gate():
    ms = np.array([10.0, 20, 30, 40, 1000])
    cnt = np.array([10, 10, 10, 10, 1])   # 마지막은 게이트 탈락 → 분포서 제외
    thr = ts_mod.percentile_threshold(ms, cnt, min_count=5, pctl=100.0)
    assert thr == pytest.approx(40.0)


def test_percentile_threshold_empty_is_inf():
    ms = np.array([np.nan, np.nan])
    cnt = np.array([1, 1])
    assert ts_mod.percentile_threshold(ms, cnt, min_count=5, pctl=80.0) == float("inf")


# --------------------------------------------------------- 라벨(MFE·close)


def test_forward_label_mfe_and_close():
    ts = np.array([0, 10, 20, 30, 40])
    price = np.array([100.0, 110, 105, 108, 200])
    # i=0, horizon=30 => (0,30] => idx1,2,3 prices 110,105,108
    lab = ts_mod.forward_label(ts, price, 0, 30)
    assert lab is not None
    mfe, close = lab
    assert mfe == pytest.approx(110 / 100 - 1)   # 최고 110
    assert close == pytest.approx(108 / 100 - 1)  # 마지막 108


def test_forward_label_none_when_no_future():
    ts = np.array([0, 10])
    price = np.array([100.0, 110])
    # i=1 이후 체결 없음
    assert ts_mod.forward_label(ts, price, 1, 30) is None


def test_forward_label_none_when_horizon_too_short():
    ts = np.array([0, 100])
    price = np.array([100.0, 110])
    # horizon 30s 안에 다음 체결(100s) 없음
    assert ts_mod.forward_label(ts, price, 0, 30) is None


def test_forward_labels_all_matches_scalar():
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.integers(1, 5, size=60)).astype(np.int64)
    price = (100 + rng.normal(0, 3, size=60)).cumsum() / 10 + 100
    mfe, close = ts_mod.forward_labels_all(ts, price, 30)
    for i in range(60):
        lab = ts_mod.forward_label(ts, price, i, 30)
        if lab is None:
            assert np.isnan(close[i])
            assert np.isnan(mfe[i])
        else:
            assert mfe[i] == pytest.approx(lab[0])
            assert close[i] == pytest.approx(lab[1])


# --------------------------------------------------------- 분위·부트스트랩


def test_quintile_means_monotone():
    sig = np.arange(100.0)
    ret = np.arange(100.0) / 1000.0   # 신호와 완전 단조
    qm = ts_mod.quintile_means(sig, ret)
    assert len(qm) == 5
    means = [row[2] for row in qm]
    assert means[4] > means[0]


def test_bootstrap_diff_ci_positive():
    treat = np.full(200, 0.02)
    base = np.full(200, 0.0)
    d, lo, _hi = ts_mod.bootstrap_diff_ci(treat, base, n_boot=200)
    assert d == pytest.approx(0.02)
    assert lo > 0


def test_bootstrap_diff_ci_empty():
    d, _lo, _hi = ts_mod.bootstrap_diff_ci(np.array([]), np.array([1.0]))
    assert np.isnan(d)


# --------------------------------------------------------- 빈 데이터 · DB 경로


@pytest.fixture()
def tick_db(tmp_path):
    c = sqlite3.connect(tmp_path / "rt.db")
    init_realtime_tables(c)
    yield c
    c.close()


def test_codes_for_date_empty(tick_db):
    assert ts_mod._codes_for_date(tick_db, "2026-08-25") == []


def test_load_code_day_none_when_empty(tick_db):
    assert ts_mod._load_code_day(tick_db, "005930", "2026-08-25") is None


def test_end_to_end_sqlite(tick_db):
    """ticks fixture 로 트리거·라벨·리포트가 끝까지 도는지."""
    base = datetime(2026, 8, 25, 10, 0, 0)
    rows = []
    # 강한 매수 체결 20건(1초 간격) → 트리거, 이후 가격 상승
    for k in range(20):
        rows.append({
            "code": "005930", "ts": base.replace(second=k), "seq": k,
            "price": 1000 + k * 2, "volume": 100, "side": 1, "strength": 150.0,
            "cum_volume": 100 * (k + 1), "best_bid": 1000, "best_ask": 1001,
            "raw": None,
        })
    # 이후 60초간 상승 궤적(라벨용)
    for k in range(20, 90):
        rows.append({
            "code": "005930", "ts": base.replace(second=k % 60, minute=(k // 60)),
            "seq": k, "price": 1040 + (k - 20), "volume": 10, "side": 1,
            "strength": 90.0, "cum_volume": 3000 + k, "best_bid": 1040,
            "best_ask": 1041, "raw": None,
        })
    insert_ticks(tick_db, rows)

    codes = ts_mod._codes_for_date(tick_db, "2026-08-25")
    assert codes == ["005930"]
    acc = ts_mod.Accum()
    data = ts_mod._load_code_day(tick_db, "005930", "2026-08-25")
    assert data is not None
    ts_mod.process_code(acc, data, "005930")
    # 트리거가 최소 1건 잡혀야 한다
    assert acc.n_trig[30] >= 1
    report = ts_mod.build_report(acc, "2026-08-25")
    assert "틱 스캘핑 sanity" in report
    assert "판정" in report


def test_epoch_seconds_is_seconds_regardless_of_datetime_unit():
    """``epoch_seconds`` 는 datetime64 단위(ns·us·ms)와 무관하게 **초**를 준다.

    운영(postgres)은 psycopg2 가 ``datetime`` 을 주고 pandas 2 가 이를
    ``datetime64[us]`` 로 잡는다. 종전 ``astype("int64") // 1e9`` 는 그때
    1,000초 단위를 뱉어 3분치 틱이 전부 같은 값으로 뭉쳤다(2026-08-28).
    """
    import pandas as pd

    base = datetime(2026, 8, 25, 10, 0, 0)
    stamps = [base, base.replace(second=30), base.replace(minute=1)]
    for unit in ("ns", "us", "ms", "s"):
        ts = pd.to_datetime(stamps).as_unit(unit)
        sec = ts_mod.epoch_seconds(ts)
        assert list(np.diff(sec)) == [30, 30], unit
        assert sec[0] > 1_700_000_000, unit


def test_load_code_day_ts_is_seconds(tick_db):
    """DB 경로도 초 단위여야 한다 — 30초 간격 틱은 30 만큼 벌어진다."""
    base = datetime(2026, 8, 25, 10, 0, 0)
    rows = [{
        "code": "005930", "ts": base + timedelta(seconds=30 * k), "seq": k,
        "price": 1000, "volume": 10, "side": 1, "strength": 100.0,
        "cum_volume": 10 * (k + 1), "best_bid": 999, "best_ask": 1001, "raw": None,
    } for k in range(3)]
    insert_ticks(tick_db, rows)

    ts_sec, *_ = ts_mod._load_code_day(tick_db, "005930", "2026-08-25")
    assert list(np.diff(ts_sec)) == [30, 30]


# ---- 코어 커널 경유판 == 예전 파이썬 루프(_py) — 결측·동점·같은 초 다수·폴백 경로까지 ----------


def _random_day(seed, n=3000):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(1_787_000_000, 1_787_000_000 + 23_000, n)).astype(np.int64)
    price = (10000 + np.cumsum(rng.integers(-1, 2, n)) * 10).astype(float)
    side = rng.choice([-1, 0, 1], n)
    vol = rng.integers(0, 3000, n).astype(float)
    stg = rng.uniform(30, 250, n)
    if seed % 2:
        price[rng.random(n) < 0.02] = np.nan
        stg[rng.random(n) < 0.02] = np.nan
        price[rng.random(n) < 0.01] = 0.0
    return ts, price, side, vol, stg


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("w", [1, 10, 30, 60, 120, 600])
def test_kernel_trailing_matches_python_loop(seed, w):
    ts, _price, side, vol, stg = _random_day(seed)
    got = ts_mod.trailing_features(ts, side, vol, stg, w)
    want = ts_mod._trailing_features_py(ts, side, vol, stg, w)
    for g, x in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, x)
        assert g.dtype == x.dtype


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("h", [0, 1, 5, 30, 60, 120, 300])
def test_kernel_forward_matches_python_loop(seed, h):
    ts, price, *_ = _random_day(seed)
    got = ts_mod.forward_labels_all(ts, price, h)
    want = ts_mod._forward_labels_all_py(ts, price, h)
    for g, x in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, x)


def test_unsorted_or_bad_window_uses_python_loop(monkeypatch):
    ts, _price, side, vol, stg = _random_day(0, n=50)
    calls = []
    real = ts_mod._trailing_features_py
    monkeypatch.setattr(ts_mod, "_trailing_features_py", lambda *a: calls.append(1) or real(*a))
    ts_mod.trailing_features(ts[::-1].copy(), side, vol, stg, 10)  # 정렬 안 됨
    with pytest.raises(IndexError):  # 예전 루프의 창 0 동작(끝에서 인덱스 초과) 그대로
        ts_mod.trailing_features(ts, side, vol, stg, 0)
    assert len(calls) == 2
