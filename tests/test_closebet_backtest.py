"""종가베팅 백테스트 엔진 테스트 — 규칙 필터·top_n·비용·집계."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scalp_it.cli_closebet import build_parser, format_summary, parse_rule, parse_rules
from scalp_it.closebet.backtest import apply_rules, run_close_bet


def frames(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``code, date, tv_ratio, close_pos, ret_open`` dict 목록을 피처·라벨로 쪼갠다."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    feat = df[["code", "date", "tv_ratio", "close_pos"]].copy()
    lab = df[["code", "date", "ret_open"]].copy()
    lab["ret_high"] = lab["ret_open"] + 0.01
    lab["ret_close"] = lab["ret_open"] - 0.005
    lab["gap_next"] = lab["ret_open"]
    return feat, lab


BASIC = [
    {"code": "A", "date": "2024-01-02", "tv_ratio": 5.0, "close_pos": 0.9, "ret_open": 0.02},
    {"code": "B", "date": "2024-01-02", "tv_ratio": 4.0, "close_pos": 0.9, "ret_open": -0.01},
    {"code": "C", "date": "2024-01-02", "tv_ratio": 1.0, "close_pos": 0.9, "ret_open": 0.50},
    {"code": "A", "date": "2024-01-03", "tv_ratio": 6.0, "close_pos": 0.3, "ret_open": 0.30},
    {"code": "B", "date": "2024-01-03", "tv_ratio": 3.5, "close_pos": 0.95, "ret_open": 0.04},
]


def test_apply_rules_min_bound_is_inclusive():
    feat, _ = frames(BASIC)
    kept = apply_rules(feat, {"tv_ratio": (3.5, None)})
    assert sorted(kept["tv_ratio"]) == [3.5, 4.0, 5.0, 6.0]


def test_apply_rules_max_bound_is_inclusive():
    feat, _ = frames(BASIC)
    kept = apply_rules(feat, {"tv_ratio": (None, 4.0)})
    assert sorted(kept["tv_ratio"]) == [1.0, 3.5, 4.0]


def test_apply_rules_combines_conditions_with_and():
    feat, _ = frames(BASIC)
    kept = apply_rules(feat, {"tv_ratio": (3.0, None), "close_pos": (0.9, None)})
    # close_pos 0.3 인 A 1/3 이 떨어져 나간다
    assert set(zip(kept["code"], kept["date"].dt.day)) == {("A", 2), ("B", 2), ("B", 3)}


def test_apply_rules_drops_nan_rows():
    feat, _ = frames(BASIC)
    feat.loc[0, "tv_ratio"] = float("nan")
    kept = apply_rules(feat, {"tv_ratio": (0.0, None)})
    assert len(kept) == len(feat) - 1


def test_apply_rules_unknown_column_raises():
    feat, _ = frames(BASIC)
    with pytest.raises(KeyError, match="nope"):
        apply_rules(feat, {"nope": (1.0, None)})


def test_run_close_bet_basic_stats():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)}, cost_round_trip=0.0)
    # A/B 1/2, A/B 1/3 → 4건, 2일
    assert res["n_trades"] == 4
    assert res["n_days"] == 2
    assert res["trades_per_day"] == pytest.approx(2.0)
    assert res["mean"] == pytest.approx((0.02 - 0.01 + 0.30 + 0.04) / 4)
    assert res["median"] == pytest.approx((0.02 + 0.04) / 2)
    assert res["win_rate"] == pytest.approx(0.75)
    assert res["payoff"] == pytest.approx(((0.02 + 0.30 + 0.04) / 3) / 0.01)


def test_cost_is_subtracted_from_mean():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)}, cost_round_trip=0.0023)
    assert res["net_mean"] == pytest.approx(res["mean"] - 0.0023)
    assert res["cost_grid"]["0.0000"] == pytest.approx(res["mean"])
    assert res["cost_grid"]["0.0035"] == pytest.approx(res["mean"] - 0.0035)


def test_top_n_is_applied_per_date():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={}, top_n=1, cost_round_trip=0.0)
    # 1/2 은 A(tv 5.0), 1/3 은 A(tv 6.0) → 날짜당 1건씩 2건
    assert res["n_trades"] == 2
    assert res["n_days"] == 2
    assert res["mean"] == pytest.approx((0.02 + 0.30) / 2)


def test_exit_col_switches_the_return_used():
    feat, lab = frames(BASIC)
    base = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)}, cost_round_trip=0.0)
    high = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)},
                         cost_round_trip=0.0, exit_col="ret_high")
    assert high["mean"] == pytest.approx(base["mean"] + 0.01)
    assert high["exit_col"] == "ret_high"


def test_invalid_exit_col_raises():
    feat, lab = frames(BASIC)
    with pytest.raises(ValueError, match="exit_col"):
        run_close_bet(feat, lab, rules={}, exit_col="ret_tomorrow")


def test_rows_without_next_day_are_excluded_and_counted():
    rows = [dict(r) for r in BASIC]
    rows[0]["ret_open"] = float("nan")  # 다음날 봉 없음
    feat, lab = frames(rows)
    lab.loc[0, ["ret_high", "ret_close", "gap_next"]] = float("nan")
    res = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)}, cost_round_trip=0.0)
    assert res["n_trades"] == 3
    assert res["n_excluded_no_next_day"] == 1


def test_limit_up_suspects_are_counted_separately():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={}, cost_round_trip=0.0)
    # 0.50 과 0.30 이 29% 초과
    assert res["n_limit_up"] == 2
    assert res["limit_up_share"] == pytest.approx(2 / 5)
    assert res["mean_ex_limit_up"] == pytest.approx((0.02 - 0.01 + 0.04) / 3)


def test_by_year_aggregates_mean_and_count():
    rows = [dict(r) for r in BASIC]
    rows.append({"code": "A", "date": "2025-01-02", "tv_ratio": 9.0,
                 "close_pos": 0.9, "ret_open": 0.10})
    feat, lab = frames(rows)
    res = run_close_bet(feat, lab, rules={}, cost_round_trip=0.0)
    assert res["by_year"]["2024"]["n"] == 5
    assert res["by_year"]["2025"] == {"mean": pytest.approx(0.10), "n": 1}


def test_no_matching_trades_returns_zeroed_result():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={"tv_ratio": (1_000.0, None)})
    assert res["n_trades"] == 0
    assert res["n_days"] == 0
    assert math.isnan(res["mean"])
    assert res["by_year"] == {}


def test_merge_keeps_only_rows_present_in_both_frames():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab.iloc[:2], rules={}, cost_round_trip=0.0)
    assert res["n_trades"] == 2


def test_parse_rule_forms():
    assert parse_rule("tv_ratio:3.0:") == ("tv_ratio", (3.0, None))
    assert parse_rule("ret_day:0.05:0.29") == ("ret_day", (0.05, 0.29))
    assert parse_rule("gap_prev::0.1") == ("gap_prev", (None, 0.1))
    assert parse_rules(["a:1:", "b::2"]) == {"a": (1.0, None), "b": (None, 2.0)}


@pytest.mark.parametrize("bad", ["tv_ratio:3.0", ":1:2", "tv_ratio:x:", "a:5:1"])
def test_parse_rule_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        parse_rule(bad)


def test_format_summary_prints_costs_and_exclusions():
    feat, lab = frames(BASIC)
    res = run_close_bet(feat, lab, rules={"tv_ratio": (3.0, None)})
    args = build_parser().parse_args(
        ["--start", "2024-01-01", "--end", "2024-12-31", "--rule", "tv_ratio:3.0:"]
    )
    text = format_summary(res, {"tv_ratio": (3.0, None)}, args)
    assert "비용 0.23%" in text
    assert "상한가 의심" in text
    assert "2024" in text
