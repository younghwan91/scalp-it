"""층 비교 하네스 — 세 층을 동일 조건에서 재는 유일한 경로 (스펙 §3).

동일 조건이란 셋을 말한다:

1. **동일 표본** — ``layers.build_layered_dataset`` 이 보장한다. 여기서는
   층 피처가 서로 누적 관계인지만 다시 확인한다.
2. **동일 폴드** — ``fold_dates`` 를 한 번 계산해 세 층에 그대로 넘긴다.
3. **동일 널** — ``random_null`` 은 ``excess_return`` 과 날짜만 쓰므로,
   1·2 가 지켜지면 세 층의 널이 같아진다. 같지 않으면 어딘가 어긋난 것이다.

층 간 차이가 우연인지는 날짜 페어드 부트스트랩으로 잰다. 같은 날짜의
두 층 포트폴리오를 짝지어 차이를 보므로 시장 공통 변동이 상쇄된다.

층별 판정은 ``gate_report`` 가 아니라 ``layer_verdict`` 다. 계획 1 이
``gate_report`` 만 보고 PASS 를 받았다가 폴드별로 뜯려 FAIL 이 된 전례가 있다.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from ..layers import LAYER_LABELS
from ..model.baseline import portfolio_returns
from ..model.layered import fold_dates, walk_forward_predict_features
from .layer_verdict import layer_verdict
from .pead import benchmark_all_layers


def paired_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """날짜로 짝지은 ``a - b`` 차이의 부트스트랩 분포.

    Args:
        a, b: ``portfolio_returns`` 결과 (``[date, n_positions, mean_excess]``).

    Returns:
        ``mean_diff`` (관측 평균 차이), ``ci_low``/``ci_high`` (95% 신뢰구간),
        ``p_greater`` (부트스트랩 표본 중 차이가 0 초과인 비율), ``n_dates``.
    """
    merged = a.merge(b, on="date", suffixes=("_a", "_b"))
    diff = (merged["mean_excess_a"] - merged["mean_excess_b"]).to_numpy()
    n = len(diff)
    if n == 0:
        raise ValueError("두 층의 공통 날짜가 없다 — 폴드가 어긋났다")

    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_greater": float((means > 0).mean()),
        "n_dates": int(n),
    }


def _check_nested(spec: dict[str, list[str]]) -> None:
    names = list(spec)
    for prev, cur in pairwise(names):
        if spec[cur][: len(spec[prev])] != spec[prev]:
            raise ValueError(
                f"층 '{cur}' 이 '{prev}' 의 누적 확장이 아니다. "
                "층은 피처를 더하기만 해야 기여도 분리가 성립한다."
            )


def llm_verdict(layer_rows: list[dict], deltas: list[dict]) -> dict:
    """③층이 LLM 을 쓸 값을 하는지 판정한다 (스펙 §3).

    세 조건이 모두 참일 때만 ``LLM_JUSTIFIED``:
    ③−② 차이가 양수 / 그 차이의 95% CI 하한이 0 초과 / ③층의 **종합 판정**이 PASS.

    세 번째가 ``gate`` 가 아니라 ``overall_verdict`` 인 것이 요점이다. 계획 1 은
    ``gate`` 만 보면 PASS 였지만 폴드별로는 랜덤과 구별되지 않았다. ③이 그 함정에
    걸린 채로 ②를 이겨봤자 LLM 비용을 정당화하지 못한다.
    """
    by_layer = {row["layer"]: row for row in layer_rows}
    layer2 = by_layer["supply_news"]
    layer3 = by_layer["supply_news_llm"]
    delta = next(d for d in deltas if d["layer"] == "supply_news_llm")

    passed = (
        delta["mean_diff"] > 0
        and delta["ci_low"] > 0
        and layer3["overall_verdict"] == "PASS"
    )
    return {
        "layer2_mean": float(layer2["strategy_mean"]),
        "layer3_mean": float(layer3["strategy_mean"]),
        "delta": float(delta["mean_diff"]),
        "ci_low": float(delta["ci_low"]),
        "ci_high": float(delta["ci_high"]),
        "p_greater": float(delta["p_greater"]),
        "layer3_overall": layer3["overall_verdict"],
        "verdict": "LLM_JUSTIFIED" if passed else "LLM_NOT_JUSTIFIED",
    }


def compare_layers(
    ds: pd.DataFrame,
    spec: dict[str, list[str]],
    *,
    top_n: int = 15,
    n_folds: int = 4,
    min_train: int = 200,
    n_trials: int = 200,
    n_boot: int = 2000,
    cost_one_way: float = 0.0023,
    hold_days: int = 40,
    seed: int = 0,
) -> dict:
    """세 층을 동일 표본·동일 폴드·동일 널로 비교한다."""
    _check_nested(spec)
    folds = fold_dates(ds, n_folds=n_folds, min_train=min_train)

    rows: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    for layer, columns in spec.items():
        pred = walk_forward_predict_features(ds, columns, folds=folds, seed=seed)
        detail = layer_verdict(
            pred,
            top_n=top_n,
            n_trials=n_trials,
            cost_one_way=cost_one_way,
            seed=seed,
        )
        curves[layer] = portfolio_returns(pred, top_n=top_n)
        rows.append(
            {
                "layer": layer,
                "label": LAYER_LABELS.get(layer, layer),
                "n_features": len(columns),
                **detail["gate"],
                "overall_verdict": detail["overall_verdict"],
                # 선별도·하방은 평균과 나란히 읽혀야 하므로 최상위로 끌어올린다.
                "mean_pool_per_date": detail["selectivity"]["mean_pool_per_date"],
                "selection_ratio": detail["selectivity"]["selection_ratio"],
                "low_selectivity": detail["selectivity"]["low_selectivity"],
                "p10": detail["downside"]["p10"],
                "p10_gap": detail["downside"]["p10_gap"],
                "tail_capture_rate": detail["downside"]["tail_capture_rate"],
                "tail_avoidance": detail["downside"]["tail_avoidance"],
                "detail": detail,
            }
        )

    names = list(spec)
    deltas = [
        {
            "layer": cur,
            "baseline": prev,
            **paired_bootstrap(curves[cur], curves[prev], n_boot=n_boot, seed=seed),
        }
        for prev, cur in pairwise(names)
    ]
    has_llm_layer = {"supply_news", "supply_news_llm"} <= set(spec)

    return {
        "rows": rows,
        "deltas": deltas,
        "llm": llm_verdict(rows, deltas) if has_llm_layer else None,
        "pead": benchmark_all_layers(curves, hold_days=hold_days),
        "sample": {
            "n_rows": len(ds),
            "n_codes": int(ds["code"].nunique()),
            "date_range": [str(ds["date"].min().date()), str(ds["date"].max().date())],
            "n_folds": n_folds,
            "top_n": top_n,
            "n_trials": n_trials,
            "n_boot": n_boot,
            "cost_one_way": cost_one_way,
            "hold_days": hold_days,
            "seed": seed,
        },
    }
