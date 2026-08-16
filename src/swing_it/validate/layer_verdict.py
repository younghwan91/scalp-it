"""층 하나에 대한 종합 판정.

``gate_report`` 하나로 판정하지 않는다. 계획 1 이 정확히 그렇게 했다가 PASS 를
받았고, 폴드별로 뜯으니 3/4 폴드에서 전략이 랜덤 널과 소수점 넷째 자리까지
같았다. 전체 표본 평균은 한 구간의 방어 효과를 전 기간의 예측력으로 오인하게
만든다 (``docs/research/2026-08-17-baseline-verdict.md``).

그래서 네 검사를 모두 통과해야 ``PASS`` 다. 플라시보만 부호가 반대다 —
라벨을 섞었는데 PASS 가 나오면 누수이므로 FAIL 이 정상이다.

여기에 계획 1 이 사후에 발견한 두 축을 더한다:

- **하방 회피** — 폴드 3 이 보여준 것은 "상위를 잘 고르기"보다 "최악의 꼬리를
  피하기"가 이 데이터에서 더 실현 가능하다는 것이었다. 평균만 보면 놓친다.
- **선별도** — 폴드 3 은 하루 24건 중 상위 15를 골라 풀의 62%를 담고 있었다.
  그걸 "선별"이라 부르면 안 된다. 숫자로 보이게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics import fold_null_table, placebo_shuffle, untouched_window
from .gate import gate_report

CHECK_NAMES: tuple[str, ...] = (
    "전체_널_대비",
    "모든_폴드_널_대비",
    "손_안_댄_최종창",
    "플라시보_FAIL",
)


LOW_SELECTIVITY_RATIO = 0.5


def selectivity(pred: pd.DataFrame, *, top_n: int = 15) -> dict:
    """후보 풀 크기와 ``top_n`` 이 그 풀에서 차지하는 비율.

    ``selection_ratio`` 가 0.62 면 "상위 15 선별"이 아니라 "풀의 62%를 담은 것"이다
    (계획 1 폴드 3 의 실측값). 이 숫자를 리포트에 찍지 않으면 선별했다는 말이
    사실을 오도한다.
    """
    sizes = pred.groupby("date").size()
    mean_pool = float(sizes.mean())
    ratio = float(min(1.0, top_n / mean_pool)) if mean_pool > 0 else 1.0
    return {
        "mean_pool_per_date": mean_pool,
        "median_pool_per_date": float(sizes.median()),
        "min_pool_per_date": int(sizes.min()),
        "selection_ratio": ratio,
        "low_selectivity": bool(ratio > LOW_SELECTIVITY_RATIO),
    }


def downside_stats(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    tail_quantile: float = 0.2,
) -> dict:
    """선택분이 하방을 얼마나 피했는가 (verdict 문서 시사점 2).

    두 축을 잰다.

    - ``p10`` — 선택분 초과수익의 하위 10% 분위. 전체 선택 행을 모아 한 번에
      잰다. 풀의 같은 분위(``pool_p10``)와의 차이가 ``p10_gap`` 이고, 양수면
      선택분의 하방이 풀보다 얕다는 뜻이다.
    - ``tail_capture_rate`` — 날짜마다 풀의 ``tail_quantile`` 임계값을 구하고
      선택된 종목 중 몇 %가 그 아래로 떨어졌는지를 센다. **낮을수록 좋다.**
      무작위 선택이면 기대값이 정확히 ``tail_quantile`` 이다.

    계획 1 에서 모델이 값을 한 유일한 구간이 정확히 이 축이었다. 평균 gap 은
    네 폴드 중 셋에서 0.0001 수준이었지만, 나쁜 레짐에서 최악의 꼬리를 피한
    효과는 2%p 였다.
    """
    captures = []
    picked_all = []
    for _, group in pred.groupby("date"):
        pool = group["excess_return"].to_numpy()
        if len(pool) == 0:
            continue
        threshold = float(np.quantile(pool, tail_quantile))
        picked = group.nlargest(top_n, "prediction")["excess_return"].to_numpy()
        captures.append(float((picked <= threshold).mean()))
        picked_all.append(picked)

    picked_pool = np.concatenate(picked_all)
    p10 = float(np.quantile(picked_pool, 0.10))
    pool_p10 = float(np.quantile(pred["excess_return"].to_numpy(), 0.10))
    capture = float(np.mean(captures))
    return {
        "p10": p10,
        "pool_p10": pool_p10,
        "p10_gap": p10 - pool_p10,
        "tail_capture_rate": capture,
        "expected_tail_rate": float(tail_quantile),
        "tail_avoidance": float(tail_quantile - capture),
    }


def layer_verdict(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    cost_one_way: float = 0.0023,
    fraction: float = 0.25,
    seed: int = 0,
) -> dict:
    """네 검사를 모두 통과해야 PASS 인 종합 판정."""
    gate = gate_report(
        pred, top_n=top_n, n_trials=n_trials, cost_one_way=cost_one_way, seed=seed
    )
    folds = fold_null_table(pred, top_n=top_n, n_trials=n_trials, seed=seed)
    untouched = untouched_window(
        pred,
        fraction=fraction,
        top_n=top_n,
        n_trials=n_trials,
        cost_one_way=cost_one_way,
        seed=seed,
    )
    placebo = gate_report(
        placebo_shuffle(pred, seed=seed),
        top_n=top_n,
        n_trials=n_trials,
        cost_one_way=cost_one_way,
        seed=seed,
    )

    checks = {
        "전체_널_대비": gate["verdict"] == "PASS",
        "모든_폴드_널_대비": bool(folds["beats_null"].all()),
        "손_안_댄_최종창": untouched["net_mean"] > 0 and untouched["percentile"] > 95.0,
        # 라벨을 섞었는데 PASS 면 누수다. FAIL 이 나와야 통과.
        "플라시보_FAIL": placebo["verdict"] == "FAIL",
    }
    return {
        "gate": gate,
        "folds": folds.to_dict("records"),
        "untouched": untouched,
        "placebo": placebo,
        "downside": downside_stats(pred, top_n=top_n),
        "selectivity": selectivity(pred, top_n=top_n),
        "checks": checks,
        "overall_verdict": "PASS" if all(checks.values()) else "FAIL",
    }
