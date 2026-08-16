from swing_it.report.layers_md import render_layer_report

CHECKS_PASS = {
    "전체_널_대비": True, "모든_폴드_널_대비": True,
    "손_안_댄_최종창": True, "플라시보_FAIL": True,
}
CHECKS_FOLD_FAIL = {**CHECKS_PASS, "모든_폴드_널_대비": False}


def _row(layer, label, n_features, mean, verdict, overall, checks, ratio):
    return {
        "layer": layer, "label": label, "n_features": n_features,
        "strategy_mean": mean, "null_mean": 0.0002, "null_p95": 0.0044,
        "percentile": 98.5, "net_mean": mean - 0.0046, "verdict": verdict,
        "overall_verdict": overall,
        "mean_pool_per_date": 15 / ratio, "selection_ratio": ratio,
        "low_selectivity": ratio > 0.5,
        "p10": -0.081, "p10_gap": 0.019,
        "tail_capture_rate": 0.16, "tail_avoidance": 0.04,
        "detail": {"checks": checks, "downside": {"pool_p10": -0.100}},
    }


def _comparison(llm_verdict: str) -> dict:
    return {
        "rows": [
            _row("supply", "① 수급 단독", 7, 0.0031, "FAIL", "FAIL",
                 CHECKS_FOLD_FAIL, 0.62),
            _row("supply_news", "② + 규칙 뉴스", 12, 0.0088, "PASS", "PASS",
                 CHECKS_PASS, 0.31),
            _row("supply_news_llm", "③ + LLM", 17, 0.0091, "PASS", "PASS",
                 CHECKS_PASS, 0.31),
        ],
        "deltas": [
            {"layer": "supply_news", "baseline": "supply", "mean_diff": 0.0057,
             "ci_low": 0.0021, "ci_high": 0.0094, "p_greater": 0.998, "n_dates": 412},
            {"layer": "supply_news_llm", "baseline": "supply_news", "mean_diff": 0.0003,
             "ci_low": -0.0019, "ci_high": 0.0026, "p_greater": 0.61, "n_dates": 412},
        ],
        "llm": {
            "layer2_mean": 0.0088, "layer3_mean": 0.0091, "delta": 0.0003,
            "ci_low": -0.0019, "ci_high": 0.0026, "p_greater": 0.61,
            "layer3_overall": "PASS", "verdict": llm_verdict,
        },
        "pead": [
            {"layer": "supply", "sharpe_annual": 0.13, "t_stat": 0.16,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
            {"layer": "supply_news", "sharpe_annual": 0.38, "t_stat": 0.48,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
            {"layer": "supply_news_llm", "sharpe_annual": 0.39, "t_stat": 0.50,
             "n_effective": 10, "pead_sharpe": 0.80, "pead_t": 2.20,
             "verdict": "LOSES_TO_PEAD"},
        ],
        "sample": {
            "n_rows": 8421, "n_codes": 913, "date_range": ["2019-03-04", "2026-06-20"],
            "n_folds": 4, "top_n": 15, "n_trials": 200, "n_boot": 2000,
            "cost_one_way": 0.0023, "hold_days": 40, "seed": 0,
        },
    }


def test_report_states_the_llm_verdict_up_front():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "LLM_NOT_JUSTIFIED" in md[:600]


def test_report_lists_every_layer_with_both_verdicts():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for label in ["① 수급 단독", "② + 규칙 뉴스", "③ + LLM"]:
        assert label in md
    assert "FAIL" in md
    assert "PASS" in md


def test_report_breaks_out_the_four_checks():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for check in ["전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL"]:
        assert check in md


def test_report_shows_selection_ratio_so_low_selectivity_cannot_hide():
    """계획 1 폴드 3 은 풀의 62%를 담고도 '상위 15 선별'이라 불렸다."""
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "선별도" in md
    assert "selection_ratio" in md
    assert "0.62" in md
    assert "24.2건" in md  # 15 / 0.62


def test_report_warns_when_selection_ratio_exceeds_half():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "선택성이 낮다" in md
    assert "① 수급 단독" in md.split("선택성이 낮다")[1][:80]


def test_report_omits_the_warning_when_every_layer_is_selective():
    comparison = _comparison("LLM_NOT_JUSTIFIED")
    for row in comparison["rows"]:
        row["selection_ratio"] = 0.31
        row["mean_pool_per_date"] = 15 / 0.31
        row["low_selectivity"] = False
    assert "선택성이 낮다" not in render_layer_report(comparison)


def test_report_shows_downside_next_to_the_mean():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "하방 회피" in md
    assert "tail_capture_rate" in md
    assert "0.160" in md
    assert "p10" in md


def test_report_shows_the_delta_confidence_interval():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "-0.0019" in md


def test_report_includes_pead_comparison():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    assert "PEAD" in md
    assert "LOSES_TO_PEAD" in md
    assert "0.80" in md


def test_report_records_preregistered_parameters():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"))
    for token in ["top_n", "15", "n_trials", "200", "0.0023"]:
        assert token in md


def test_report_prints_the_plan1_baseline_with_a_warning():
    md = render_layer_report(
        _comparison("LLM_NOT_JUSTIFIED"),
        baseline_reference={"strategy_mean": 0.00594, "verdict": "FAIL",
                            "n_events": 25559},
    )
    assert "계획 1 베이스라인" in md
    assert "25,559" in md
    assert "표본이 다르므로" in md


def test_report_prints_the_preregistration_decision():
    md = render_layer_report(
        _comparison("LLM_NOT_JUSTIFIED"),
        preregistration={"tv_multiple": 1.0, "mean_pool": 27.4, "n_events": 8421,
                         "decided_at": "2026-08-17"},
    )
    assert "사전등록" in md
    assert "tv_multiple" in md
    assert "27.4" in md


def test_report_does_not_soften_a_justified_verdict():
    md = render_layer_report(_comparison("LLM_JUSTIFIED"))
    assert "LLM_JUSTIFIED" in md
    assert "LLM_NOT_JUSTIFIED" not in md
