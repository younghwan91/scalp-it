"""'핫함' 3축 계산과 일별 랭킹.

백테스트 유효성의 핵심은 과거와 현재가 같은 방식으로 계산되는 것이다
(스펙 §4). 조회수·댓글수처럼 과거 데이터가 없는 지표는 쓰지 않는다.
여기 쓰는 세 축은 빅카인즈·krx-news-rest-api·가격DB 양쪽에서 모두 나온다.

가중치는 사전등록 값이다 (스펙 §6). 성과를 보고 조정하기 시작하면
그 순간 검증이 무의미해진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

W_ARTICLES = 0.4
W_OUTLETS = 0.3
W_TRADE = 0.3

_COUNT_COLUMNS = ["date", "code", "n_articles", "n_outlets", "primary_news_id"]
_RANK_COLUMNS = [
    "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
    "tv_ratio", "hotness", "rank", "in_train", "in_ops", "primary_news_id",
]


def daily_counts(clustered: pd.DataFrame) -> pd.DataFrame:
    """(날짜, 종목) 별 사건 수와 매체 수.

    ``n_articles`` 는 기사 행 수가 아니라 **클러스터 수**다. 통신사 기사를
    여러 매체가 받아쓴 것은 사건 하나로 센다.
    """
    if clustered.empty:
        return pd.DataFrame(columns=_COUNT_COLUMNS)

    counts = (
        clustered.groupby(["date", "code"])
        .agg(n_articles=("cluster_id", "nunique"), n_outlets=("outlet", "nunique"))
        .reset_index()
    )
    primary = (
        clustered[clustered["is_primary"]]
        .sort_values(["date", "code", "news_id"])
        .groupby(["date", "code"], as_index=False)["news_id"]
        .first()
        .rename(columns={"news_id": "primary_news_id"})
    )
    out = counts.merge(primary, on=["date", "code"], how="left")
    return out[_COUNT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _dense_grid(counts: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """(전 종목 × 전 거래일) 격자로 펼치고 기사 없는 날을 0 으로 채운다.

    기사가 난 날만 모아 롤링하면 '평소'가 과대평가된다. 평소에 조용한
    종목일수록 급증 배수가 낮게 나오는, 정확히 반대 방향의 편향이다.

    날짜 축은 가격 데이터의 거래일에서 가져온다. 기사가 난 날만으로
    축을 만들면 그것 자체가 같은 편향을 다시 만든다.
    """
    axis = pd.Index(sorted(set(calendar) | set(counts["date"])), name="date")
    dates = axis
    codes = pd.Index(sorted(counts["code"].unique()), name="code")
    grid = pd.MultiIndex.from_product([codes, dates]).to_frame(index=False)
    dense = grid.merge(counts, on=["code", "date"], how="left")
    dense["n_articles"] = dense["n_articles"].fillna(0.0)
    return dense.sort_values(["code", "date"])


def add_hotness(
    counts: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    baseline_window: int = 60,
    min_periods: int = 20,
    baseline_floor: float = 0.5,
) -> pd.DataFrame:
    """기사 급증 배수·매체 수·거래대금 급증 배수를 합쳐 핫함 점수를 만든다.

    Args:
        liquidity: 계획 1 ``events.add_liquidity_stats`` 의 ``[code, date, adv, tv_ratio]``.
            baseline 의 달력 축도 여기서 가져온다 — 거래일 목록의 유일한 출처다.
        baseline_window: 평소 기사 건수를 재는 달력일 구간.
        min_periods: 이 일수를 못 채우면 baseline 을 ``baseline_floor`` 로 둔다.
        baseline_floor: 평소 0건인 종목의 분모 하한. 0.5 이면 '평소 0건 → 오늘 1건'이
            배수 2.0 이 된다.
    """
    if counts.empty:
        return pd.DataFrame(
            columns=[*_COUNT_COLUMNS, "baseline_articles", "article_ratio",
                     "adv", "tv_ratio", "hotness"]
        )

    dense = _dense_grid(counts, liquidity["date"])
    baseline = dense.groupby("code", sort=False)["n_articles"].transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=min_periods).mean()
    )
    dense["baseline_articles"] = baseline.fillna(baseline_floor).clip(lower=baseline_floor)

    out = counts.merge(
        dense[["code", "date", "baseline_articles"]], on=["code", "date"], how="left"
    )
    out["article_ratio"] = out["n_articles"] / out["baseline_articles"]

    out = out.merge(
        liquidity[["code", "date", "adv", "tv_ratio"]], on=["code", "date"], how="left"
    )
    # 가격 행이 없으면 거래대금 급증도 없었던 것으로 본다. adv 는 NaN 으로 남겨
    # 랭킹 단계의 유동성 하한에서 탈락시킨다.
    out["tv_ratio"] = out["tv_ratio"].fillna(1.0)

    out["hotness"] = (
        W_ARTICLES * np.log1p(out["article_ratio"])
        + W_OUTLETS * np.log1p(out["n_outlets"])
        + W_TRADE * np.log1p(out["tv_ratio"])
    )
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def rank_daily(
    hot: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    train_top: int = 50,
    ops_top: int = 10,
) -> pd.DataFrame:
    """유동성 하한을 통과한 종목을 날짜별 핫함 순으로 세우고 학습셋을 자른다.

    학습은 일 50건, 운영은 상위 10건이다 (스펙 §5). 학습 표본을 넉넉히
    잡는 이유는 두 가지다 — 표본이 4배가 되고, '핫함 순위' 자체가 피처가 된다.

    ``adv`` 가 ``NaN`` 인 행(상장 전·거래정지)은 비교가 False 라 자동 탈락한다.
    """
    if hot.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)

    out = hot[hot["adv"] >= adv_floor].copy()
    out = out.sort_values(["date", "hotness", "code"], ascending=[True, False, True])
    out["rank"] = out.groupby("date").cumcount() + 1
    out["in_train"] = out["rank"] <= train_top
    out["in_ops"] = out["rank"] <= ops_top
    out = out[out["in_train"]]
    return out[_RANK_COLUMNS].reset_index(drop=True)
