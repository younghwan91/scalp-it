# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""`scripts/tick_sanity.py` 단위테스트 — 트리거 판정·라벨 계산·빈 데이터.

scripts 는 패키지가 아니라 importlib 로 파일에서 직접 로드한다. 운영은 postgres 지만
테스트는 sqlite fixture(ticks 스키마 동일)로 돈다.
"""
import importlib.util
import sqlite3
from datetime import datetime
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
