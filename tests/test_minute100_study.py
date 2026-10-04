"""102번 학습·대조·판정 — **분할이 안 겹치는지, 널이 자기 자신을 세지 않는지, 게이트가 산술대로인지.**

표본이 14배인 만큼 여기서 한 칸 새면 거짓 양성이 14배 그럴듯해진다. 그래서 손으로 아는 값으로 고정한다.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("minute100_study", ROOT / "scripts" / "minute_100" / "study.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["minute100_study"] = m
    spec.loader.exec_module(m)
    return m


def test_split_matches_prereg_and_never_overlaps():
    m = _mod()
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-07-01", "2026-09-30")]
    tr, va, fi = m.split(days)
    assert tr[0] == "2025-08-01" and tr[-1] == "2026-04-30"
    assert va[0] == "2026-05-01" and va[-1] == "2026-07-31"
    assert fi[0] == "2026-08-03" and fi[-1] == "2026-09-16"      # 08-01·02 는 주말
    assert not (set(tr) & set(va)) and not (set(va) & set(fi))
    assert "2026-09-17" not in fi, "최종 구간 끝(09-16)을 넘으면 사전등록 OOS 와 어긋난다"


def test_null_alpha_excludes_self_and_small_cells():
    m = _mod()
    df = pd.DataFrame({
        "date": ["d"] * 7, "minute": [600] * 7, "vol_q": [0] * 7, "tv_q": [0] * 7,
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    })
    a = m.null_alpha(df)
    # 첫 행 10: 나머지 6개 평균 = 45 → alpha = -35 (자기 자신을 셀 평균에 넣으면 -30 이 된다)
    assert abs(a.iloc[0] - (-35.0)) < 1e-9
    small = pd.DataFrame({"date": ["d"] * 4, "minute": [600] * 4, "vol_q": [0] * 4, "tv_q": [0] * 4,
                          "y": [1.0, 2.0, 3.0, 4.0]})
    assert m.null_alpha(small).isna().all(), "셀이 5행 미만이면 alpha 를 내면 안 된다"


def test_null_alpha_cells_are_separated_by_all_four_keys():
    m = _mod()
    rows = []
    for q in (0, 1):
        for i in range(6):
            rows.append({"date": "d", "minute": 600, "vol_q": q, "tv_q": 0, "y": float(100 * q + i)})
    df = pd.DataFrame(rows)
    a = m.null_alpha(df)
    assert abs(a.iloc[0] - (0 - np.mean([1, 2, 3, 4, 5]))) < 1e-9, "다른 변동성 분위 셀의 값이 섞였다"


def test_top_slice_is_ceil_of_fraction_and_sorted_by_prediction():
    m = _mod()
    df = pd.DataFrame({"y": np.arange(10.0)})
    pred = np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5], float)
    t = m.top_slice(df, pred, 0.25)
    assert len(t) == 3 and list(t.y) == [1.0, 3.0, 5.0]


def test_gates_reject_an_obviously_negative_sample():
    m = _mod()
    rng = np.random.default_rng(0)
    days = [f"2026-08-{d:02d}" for d in range(1, 26)]
    df = pd.DataFrame({"date": np.repeat(days, 40), "y": rng.normal(-30, 50, 1000),
                       "alpha": rng.normal(-5, 50, 1000)})
    g, verdict = m.gates(df, 2450)
    assert verdict == "기각" and not g["checks"]["1 평균>0"]


def test_gates_require_minimum_days_and_rows():
    m = _mod()
    df = pd.DataFrame({"date": ["2026-08-01"] * 60, "y": np.full(60, 50.0), "alpha": np.full(60, 50.0)})
    g, verdict = m.gates(df, 2450)
    assert not g["checks"]["7 일≥20·행≥500"] and verdict == "기각"


def test_feature_columns_exclude_label_and_keys():
    m = _mod()
    df = pd.DataFrame({"ret1": [0.0], "y": [0.0], "code": ["A"], "ts": [pd.Timestamp("2026-08-01")],
                       "date": ["2026-08-01"], "vol_q": [0], "tv_q": [0], "alpha": [0.0]})
    f = m.feats_of(df)
    assert "y" not in f and "code" not in f and "date" not in f
    assert "ret1" in f


@pytest.mark.parametrize("col", ["date", "code"])
def test_prior_day_cells_uses_only_prior_information(col):
    """셀 분위가 당일 값을 쓰면 널이 look-ahead 가 된다 — 코드 검사."""
    src = (ROOT / "scripts" / "minute_100" / "study.py").read_text(encoding="utf-8")
    assert "shift(1)" in src and "p_vol" in src and "p_tv" in src
