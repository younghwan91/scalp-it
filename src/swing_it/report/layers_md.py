"""층 비교 결과 마크다운 리포트.

판정을 맨 위에 둔다. FAIL 을 뒤쪽 표 안에 묻어두면 사람은 앞의 숫자만 보고
좋은 결과라고 읽는다. 스펙 §10 — 부정 결과도 성공이고, 성공은 명확히 보여야 한다.

계획 1 의 판정 문서 형식을 그대로 따른다: 네 검사를 개별 컬럼으로 펼치고,
전체 평균 하나로 판정하지 않았음을 리포트 자체가 증명하게 한다.
"""

from __future__ import annotations

_LLM_MESSAGE = {
    "LLM_JUSTIFIED": (
        "③층이 ②층을 유의하게 이겼고 ③층 자체도 종합 판정을 통과했다. "
        "LLM 피처 추출 비용을 계속 지불할 근거가 있다."
    ),
    "LLM_NOT_JUSTIFIED": (
        "③층이 ②층을 유의하게 이기지 못했다(또는 ③층이 종합 판정을 못 넘었다). "
        "**LLM 을 쓸 이유가 없다.** 규칙 기반 뉴스 피처까지가 이 데이터에서 나오는 전부다."
    ),
}

_CHECK_ORDER = ("전체_널_대비", "모든_폴드_널_대비", "손_안_댄_최종창", "플라시보_FAIL")


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def render_layer_report(
    comparison: dict,
    *,
    baseline_reference: dict | None = None,
    preregistration: dict | None = None,
) -> str:
    rows = comparison["rows"]
    deltas = comparison["deltas"]
    llm = comparison["llm"]
    sample = comparison["sample"]
    labels = {row["layer"]: row["label"] for row in rows}

    out = [
        "# 층별 기여도 판정",
        "",
        f"## 판정: {llm['verdict']}",
        "",
        _LLM_MESSAGE[llm["verdict"]],
        "",
        (f"③ − ② = {llm['delta']:+.4f} "
        f"(95% CI {llm['ci_low']:+.4f} ~ {llm['ci_high']:+.4f}, "
        f"P(차이>0) = {llm['p_greater']:.3f}, ③층 종합 판정 {llm['layer3_overall']})"),
        "",
        "## 표본",
        "",
        (f"- 이벤트 {sample['n_rows']:,}건 · 종목 {sample['n_codes']:,}개 · "
        f"{sample['date_range'][0]} ~ {sample['date_range'][1]}"),
        "- 세 층 모두 **같은 행**을 본다. 한 층에서라도 피처가 결측인 이벤트는 전부 제외됐다.",
        "",
        "### 사전등록 파라미터",
        "",
        (f"`top_n={sample['top_n']}` · `n_folds={sample['n_folds']}` · "
        f"`n_trials={sample['n_trials']}` · `n_boot={sample['n_boot']}` · "
        f"`cost_one_way={sample['cost_one_way']}` · `hold_days={sample['hold_days']}` · "
        f"`seed={sample['seed']}`"),
        "",
    ]

    if preregistration:
        out += [
            (f"이벤트 정의는 성과를 보기 전에 확정했다 ({preregistration['decided_at']}): "
            f"`tv_multiple={preregistration['tv_multiple']}` — "
            f"dry-run 실측 하루 평균 풀 {preregistration['mean_pool']:.1f}건 · "
            f"총 {preregistration['n_events']:,}건 기준."),
            "",
        ]

    out += [
        "## 층별 종합 판정 (동일 폴드 · 동일 널)",
        "",
        "| 층 | 피처 | 전략 평균 | 비용차감 후 | "
        + " | ".join(_CHECK_ORDER)
        + " | 종합 |",
        "|---|---|---|---|" + "---|" * len(_CHECK_ORDER) + "---|",
    ]
    for row in rows:
        checks = row["detail"]["checks"]
        marks = " | ".join(_mark(checks[name]) for name in _CHECK_ORDER)
        out.append(
            f"| {row['label']} | {row['n_features']} | {_pct(row['strategy_mean'])} | "
            f"{_pct(row['net_mean'])} | {marks} | **{row['overall_verdict']}** |"
        )

    out += [
        "",
        ("> 네 검사를 **모두** 통과해야 종합 PASS 다. 계획 1 은 전체 널 대비만 보면 PASS 였고 "
        "폴드별로 뜯으니 3/4 폴드가 랜덤과 구별되지 않았다. 플라시보만 부호가 반대다 — "
        "라벨을 섞었는데 PASS 면 누수이므로 FAIL 이 나와야 통과다."),
        "",
        "## 음성대조 널",
        "",
        "| 층 | 널 평균 | 널 p95 | 백분위 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {_pct(row['null_mean'])} | {_pct(row['null_p95'])} | "
            f"{row['percentile']:.1f} |"
        )
    out += [
        "",
        ("> 널 평균·널 p95 가 세 층에서 같은 것이 정상이다. 같은 표본 위에서 같은 시드로 "
        "만든 같은 널이기 때문이다. 다르다면 표본이 어긋난 것이고 비교는 무효다."),
        "",
        "## 선별도",
        "",
        "| 층 | 하루 평균 풀 | selection_ratio |",
        "|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {row['mean_pool_per_date']:.1f}건 | "
            f"{row['selection_ratio']:.2f} |"
        )
    out += [
        "",
        ("> `selection_ratio` = `top_n / mean_pool_per_date`. 계획 1 폴드 3 은 하루 평균 "
        "24.1건 중 상위 15 = **0.62** 였다. 뉴스 축이 들어오면 이 값이 내려가야 정상이다."),
        "",
    ]
    if any(row["low_selectivity"] for row in rows):
        offenders = ", ".join(
            row["label"] for row in rows if row["low_selectivity"]
        )
        out += [
            (f"⚠️ **선택성이 낮다 — {offenders}.** `selection_ratio` 가 0.5 를 넘는다. "
            "후보 풀의 절반 이상을 담고 있으므로 이것을 '상위 선별'이라고 읽으면 안 된다. "
            "성과가 나오더라도 그것은 선별의 성과가 아니라 풀 자체의 성과에 가깝다."),
            "",
        ]

    out += [
        "## 하방 회피",
        "",
        "| 층 | 평균 | 선택분 p10 | 풀 p10 | p10 격차 | tail_capture_rate |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        detail = row["detail"]["downside"]
        out.append(
            f"| {row['label']} | {_pct(row['strategy_mean'])} | {_pct(row['p10'])} | "
            f"{_pct(detail['pool_p10'])} | {_pct(row['p10_gap'])} | "
            f"{row['tail_capture_rate']:.3f} |"
        )
    out += [
        "",
        ("> **평균과 나란히 읽어라.** 계획 1 에서 모델이 값을 한 유일한 구간(폴드 3, 나쁜 "
        "레짐)에서 한 일은 상위를 잘 고른 게 아니라 최악의 꼬리를 피한 것이었다 — "
        "전체 −2.99% vs 선택분 −0.93%. 같은 구간의 평균 gap 은 나머지 세 폴드에서 "
        "0.0001 수준이었다."),
        ("> **`tail_capture_rate`** = 날짜별 풀 하위 20%에 속한 이벤트를 선택분이 담은 비율. "
        "**낮을수록 좋다.** 무작위 선택이면 0.200 이다."),
        "> **p10 격차** = 선택분 하위 10% 분위 − 풀 하위 10% 분위. 양수면 하방이 얕다.",
        "",
        "## 층 간 증분 (날짜 페어드 부트스트랩)",
        "",
        "| 비교 | 평균 차이 | 95% CI | P(차이>0) | 날짜 수 |",
        "|---|---|---|---|---|",
    ]
    for delta in deltas:
        cur = labels.get(delta["layer"], delta["layer"])
        prev = labels.get(delta["baseline"], delta["baseline"])
        out.append(
            f"| {cur} − {prev} | {delta['mean_diff']:+.4f} | "
            f"{delta['ci_low']:+.4f} ~ {delta['ci_high']:+.4f} | "
            f"{delta['p_greater']:.3f} | {delta['n_dates']} |"
        )

    pead_rows = comparison["pead"]
    out += [
        "",
        "## PEAD 대비 (스펙 §7)",
        "",
        (f"기준: kr-quant 검증 PEAD — 연율 Sharpe {pead_rows[0]['pead_sharpe']:.2f}, "
        f"t {pead_rows[0]['pead_t']:.2f}"),
        "",
        "| 층 | 연율 Sharpe | t (겹침 보정) | 유효 표본 | 판정 |",
        "|---|---|---|---|---|",
    ]
    for row in pead_rows:
        out.append(
            f"| {labels.get(row['layer'], row['layer'])} | {row['sharpe_annual']:.2f} | "
            f"{row['t_stat']:.2f} | {row['n_effective']} | **{row['verdict']}** |"
        )
    out += [
        "",
        ("> t 는 겹침 보정본이다. 보유 40거래일이 겹치므로 날짜별 관측은 독립이 아니고, "
        "보정 없이는 t 가 크게 부풀려진다. 유효 표본 = 겹치지 않는 블록 수."),
        "",
    ]

    if baseline_reference:
        out += [
            "## 계획 1 베이스라인 참고값",
            "",
            (f"- 전략 평균 {_pct(baseline_reference['strategy_mean'])} · "
            f"종합 판정 {baseline_reference['verdict']} · "
            f"이벤트 {baseline_reference['n_events']:,}건"),
            "",
            ("> **표본이 다르므로 위 층별 표와 직접 비교하지 말 것.** 계획 1 은 기사 유무와 "
            "무관한 모든 거래대금 급증 이벤트를 쓰고, 여기 ①층은 세 층 피처가 모두 있는 "
            "이벤트만 쓴다."),
            "",
        ]

    return "\n".join(out)
