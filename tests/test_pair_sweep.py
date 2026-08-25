"""``scripts/pair_sweep.py`` — 그리드 파싱·미존재 필드 스킵·집계 산술·합성 틱 1회전.

DB 없이 돈다. 합성 틱은 리플레이와 같은 6튜플 ``(code, ts, price, volume, best_bid, side)``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pair_sweep.py"


def _load():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("pair_sweep", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ps = _load()


# --------------------------------------------------------------------- 그리드

def test_grid_parses_axes_and_cartesian_product():
    axes, warns = ps.parse_grid(["surge_window=10,15,30", "cosurge_min=0.005,0.002"])
    assert warns == []
    assert axes == [("surge_window", [10.0, 15.0, 30.0]),
                    ("cosurge_min", [0.005, 0.002])]
    combos = ps.build_combos(axes)
    assert len(combos) == 6
    assert {"surge_window": 15.0, "cosurge_min": 0.002} in combos


def test_grid_skips_unknown_field_with_warning():
    axes, warns = ps.parse_grid(["surge_ret=0.01", "no_such_knob=1,2"])
    assert [n for n, _ in axes] == ["surge_ret"]
    assert len(warns) == 1
    assert "no_such_knob" in warns[0]


def test_grid_only_uses_dataclass_fields_not_a_hardcoded_list():
    known = ps._config_fields()
    assert set(known) == {f.name for f in fields(ps.DetectorConfig)}
    # 문서화한 축 중 실제로 있는 것만 통과해야 한다(아직 없는 축은 경고로 빠진다).
    axes, warns = ps.parse_grid([f"{n}=1" for n in ps.SWEEP_HINT])
    assert len(axes) + len(warns) == len(ps.SWEEP_HINT)
    assert all(n in known for n, _ in axes)


def test_optional_fields_take_none_and_zero_means_off():
    axes, _ = ps.parse_grid(["stop_loss=-0.004,none,0"])
    assert axes[0][1] == [-0.004, None, None]


def test_none_rejected_for_non_optional_field():
    # take_profit 은 만쥬 손익 구조(상한 없음)를 위해 Optional 이 됐다 → 진짜 비-Optional
    # 필드로 규약을 고정한다.
    with pytest.raises(ValueError, match="None"):
        ps.parse_grid(["surge_window=none"])


def test_take_profit_accepts_none_meaning_no_cap():
    """익절 상한 없음도 스윕 축이어야 한다 — 만쥬 손익 구조."""
    axes, _ = ps.parse_grid(["take_profit=0.008,none"])
    assert axes[0][1] == [0.008, None]


def test_duplicate_axis_is_an_error():
    with pytest.raises(ValueError, match="중복"):
        ps.parse_grid(["surge_ret=0.01", "surge_ret=0.02"])


def test_bad_spec_is_an_error():
    with pytest.raises(ValueError, match="NAME=v1"):
        ps.parse_grid(["surge_ret"])


def test_no_axes_gives_single_base_combo():
    assert ps.build_combos([]) == [{}]


def test_history_warning_when_windows_exceed_history():
    cfg = ps.DetectorConfig.stable(surge_window=300.0)
    assert "history" in (ps.history_warning(cfg) or "")
    assert ps.history_warning(ps.DetectorConfig.stable()) is None


# --------------------------------------------------------------------- 집계

def _rec(entry, exit_px, status="closed", reason="익절", mae=-0.001, mfe=0.006):
    return {"entry": entry, "exit_price": exit_px, "status": status,
            "exit_reason": reason, "mae": mae, "mfe": mfe}


def test_summarize_arithmetic():
    recs = [
        _rec(10_000, 10_100),                        # 세전 +1.00%
        _rec(10_000, 9_960, reason="손절", mae=-0.004, mfe=0.001),   # 세전 −0.40%
        _rec(10_000, None, status="unfilled", reason="미체결", mae=0.0, mfe=0.0),
    ]
    m = ps.summarize_records(recs, notional=100_000.0)
    assert (m["positions"], m["filled"], m["unfilled"]) == (3, 2, 1)
    assert m["exit_reasons"] == {"익절": 1, "손절": 1, "미체결": 1}
    # 세전: 평균 (1.00 − 0.40)/2 = +0.30%, 합계 0.006×100,000 = 600원
    assert m["gross"]["mean_pct"] == pytest.approx(0.30)
    assert m["gross"]["total_krw"] == pytest.approx(600.0)
    assert m["gross"]["win_rate_pct"] == pytest.approx(50.0)
    # 세후 64bp: 평균 0.30 − 0.64 = −0.34%. 승률은 비용 차감 후로 다시 센다
    # (+1.00−0.64 = +0.36 만 살아남아 1/2).
    assert m["net64"]["mean_pct"] == pytest.approx(-0.34)
    assert m["net64"]["total_krw"] == pytest.approx(600.0 - 0.0064 * 2 * 100_000)
    assert m["net64"]["win_rate_pct"] == pytest.approx(50.0)
    # 94bp 면 +1.00−0.94 = +0.06 만 남아 여전히 1/2, 평균은 −0.64%.
    assert m["net94"]["win_rate_pct"] == pytest.approx(50.0)
    assert m["net34"]["mean_pct"] == pytest.approx(-0.04)
    assert m["net94"]["mean_pct"] == pytest.approx(-0.64)
    # 미체결은 MAE/MFE 중앙값에서 빠진다(0.0 이 섞이면 리스크가 작아 보인다).
    assert m["mae_med_pct"] == pytest.approx(-0.25)
    assert m["mfe_med_pct"] == pytest.approx(0.35)


def test_summarize_costs_come_from_verify_pair():
    from verify_pair import COST_BASE, COST_HI, COST_LO

    assert ps.COSTS == {"net34": COST_LO, "net64": COST_BASE, "net94": COST_HI}


def test_summarize_empty_is_safe():
    m = ps.summarize_records([], notional=100_000.0)
    assert m["filled"] == 0
    assert m["net64"]["mean_pct"] is None
    assert m["mae_med_pct"] is None


# --------------------------------------------------------------------- 1회전

def _synthetic_rows() -> list[tuple]:
    """대장 AAA 가 t+100~110 에 +2% 서지, 후속 BBB 는 잠잠하다 눌린 뒤 익절까지.

    후속 bid1 = 현재가 − 5 → 진입 9,995. 알림 뒤 9,990(관통 체결) → 10,100(익절).
    """
    t0 = datetime(2026, 8, 24, 9, 30, 0)  # noqa: DTZ001  DB 틱과 같은 naive 시각
    rows: list[tuple] = []
    for i in range(140):
        ts = t0 + timedelta(seconds=i)
        lead = 1000.0 if i < 100 else 1000.0 + 2.0 * (i - 100)     # 100→110 에서 +2%
        rows.append(("AAA", ts, lead, 100.0, lead - 1, 1))
        if i < 121:
            fol = 10_000.0
        elif i < 125:
            fol = 9_990.0                                          # through 체결
        else:
            fol = 10_100.0                                         # 익절 도달
        rows.append(("BBB", ts, fol, 100.0, fol - 5, 1))
    rows.sort(key=lambda r: r[1])
    return rows


def _sweep_cfg(**kw):
    base = {"surge_window": 10.0, "surge_ret": 0.01, "cosurge_min": 0.0,
            "premove_max": 1.0, "min_buy_ratio": 0.0, "min_sl_ticks": 0.0,
            "take_profit": 0.005, "stop_loss": -0.004, "time_exit": 30.0}
    base.update(kw)
    return ps.DetectorConfig.stable(**base)


def test_run_combo_on_synthetic_ticks():
    rows = _synthetic_rows()
    pairs = [ps.Pair(leader="AAA", follower="BBB")]
    gcfg = ps.RiskGuardConfig(daily_loss_limit=0.0, max_consecutive_losses=0,
                              stop_file=None, max_positions=1000,
                              position_notional=100_000.0, cost_rate=0.0)
    res = ps.run_combo(pairs, _sweep_cfg(), rows, gcfg, rows[-1][1])
    assert res["alerts"] == 1
    assert set(res["by_basis"]) == {"through", "touch"}
    through = res["by_basis"]["through"]
    assert through["filled"] == 1
    assert through["exit_reasons"] == {"익절": 1}
    assert through["gross"]["mean_pct"] == pytest.approx(0.5, abs=0.02)
    assert through["net64"]["mean_pct"] == pytest.approx(0.5 - 0.64, abs=0.02)


def test_run_combo_is_reusable_on_the_same_rows():
    """같은 틱 리스트를 두 조합에 다시 흘려도 결과가 오염되지 않는다(스윕의 존재 이유)."""
    rows = _synthetic_rows()
    pairs = [ps.Pair(leader="AAA", follower="BBB")]
    gcfg = ps.RiskGuardConfig(daily_loss_limit=0.0, max_consecutive_losses=0,
                              stop_file=None, max_positions=1000, cost_rate=0.0)
    first = ps.run_combo(pairs, _sweep_cfg(), rows, gcfg, rows[-1][1])
    again = ps.run_combo(pairs, _sweep_cfg(), rows, gcfg, rows[-1][1])
    assert first["by_basis"]["through"] == again["by_basis"]["through"]
    # 서지 임계를 올리면 신호가 사라진다 — 조합이 실제로 반영된다는 확인.
    tight = ps.run_combo(pairs, _sweep_cfg(surge_ret=0.5), rows, gcfg, rows[-1][1])
    assert tight["alerts"] == 0
    assert tight["by_basis"]["through"]["filled"] == 0
