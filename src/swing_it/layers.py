"""층 정의와 층별 단일 데이터셋 조립 (스펙 §3).

핵심은 하나다: **세 층이 같은 행 위에서 돌아야 한다.** 층마다 표본이 다르면
"③이 ②를 이겼다"는 문장이 아무것도 말해주지 않는다. 그래서 층을 서로 다른
데이터셋으로 만들지 않고, 세 층 피처를 전부 붙인 하나의 테이블을 만든 뒤
층은 그 위에서 쓰는 **컬럼의 부분집합**으로만 정의한다.

결측 채우기 규칙은 두지 않는다. 규칙은 하나다 — 어느 층 피처든 결측이면
그 행은 세 층 모두에서 빠진다. 채움 규칙을 두는 순간 "언제 채웠고 언제
떨궜나"를 추적해야 하고, 그 복잡도가 값을 하지 않는다.

계획 1 의 ``build_dataset`` 시그니처는 건드리지 않는다. 여기서 감싸 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .dataset import FEATURE_COLUMNS, build_dataset

LAYER_NAMES: list[str] = ["supply", "supply_news", "supply_news_llm"]

LAYER_LABELS: dict[str, str] = {
    "supply": "① 수급 단독",
    "supply_news": "② + 규칙 뉴스",
    "supply_news_llm": "③ + LLM",
}


def layer_features(
    news_columns: Sequence[str] = (),
    llm_columns: Sequence[str] = (),
) -> dict[str, list[str]]:
    """층 이름 → 그 층이 쓰는 피처 컬럼 목록(누적)."""
    supply = list(FEATURE_COLUMNS)
    news = [*supply, *news_columns]
    return {
        "supply": supply,
        "supply_news": news,
        "supply_news_llm": [*news, *llm_columns],
    }


def _merge_side(
    base: pd.DataFrame,
    side: pd.DataFrame | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """(code, date) 로 보조 피처 프레임을 왼쪽 조인한다."""
    if not columns:
        return base
    if side is None:
        side = pd.DataFrame(columns=["code", "date", *columns])
    missing = [c for c in ["code", "date", *columns] if c not in side.columns]
    if missing:
        raise ValueError(f"보조 프레임에 컬럼이 없다: {missing}")
    frame = side[["code", "date", *columns]].copy()
    frame["code"] = frame["code"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    out = base.merge(frame, on=["code", "date"], how="left")
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_layered_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    news: pd.DataFrame | None = None,
    news_columns: Sequence[str] = (),
    llm: pd.DataFrame | None = None,
    llm_columns: Sequence[str] = (),
    require_label: bool = True,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """세 층의 피처를 모두 붙인 단일 학습 테이블.

    피처가 하나라도 결측인 행은 층과 무관하게 전부 떨군다. ③층에서만 쓰는
    LLM 피처가 없다는 이유로 ①층이 그 행을 쓰면 표본이 어긋난다.

    Args:
        require_label: ``True`` 면 라벨 결측 행을 떨군다(학습·검증용).
            ``False`` 면 남긴다 — 오늘 후보 추론용으로, 아직 D+40 이 오지 않아
            라벨이 존재할 수 없는 최근 이벤트를 살린다.
    """
    common = {
        "adv_floor": adv_floor,
        "tv_multiple": tv_multiple,
        "entry_offset": entry_offset,
        "hold_start": hold_start,
        "hold_end": hold_end,
    }
    base = (
        build_dataset(prices, supply_demand, **common)
        if require_label
        else _unlabelled_base(prices, supply_demand, **common)
    )

    out = _merge_side(base, news, list(news_columns))
    out = _merge_side(out, llm, list(llm_columns))

    feature_columns = [*FEATURE_COLUMNS, *news_columns, *llm_columns]
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_columns)
    if require_label:
        out = out.dropna(subset=["excess_return"])

    columns = ["code", "date", *feature_columns, "fwd_return", "excess_return"]
    return out[columns].sort_values(["date", "code"]).reset_index(drop=True)


def _unlabelled_base(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    adv_floor: float,
    tv_multiple: float,
    entry_offset: int,
    hold_start: int,
    hold_end: int,
) -> pd.DataFrame:
    """라벨 없는 이벤트까지 포함한 베이스 테이블.

    ``build_dataset`` 과 같은 피처를 만들되 ``dropna`` 를 라벨에 걸지 않는다.
    계획 1 의 함수를 수정하지 않기 위해 조립 순서만 여기서 다시 쓴다.
    """
    from .events import extract_events
    from .features.supply import supply_features
    from .labels import excess_return, forward_return

    columns = ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]
    events = extract_events(prices, adv_floor=adv_floor, tv_multiple=tv_multiple)
    if events.empty:
        return pd.DataFrame(columns=columns)

    supply = supply_features(supply_demand)
    labels = excess_return(
        forward_return(
            prices,
            entry_offset=entry_offset,
            hold_start=hold_start,
            hold_end=hold_end,
        )
    )
    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    # 진입가가 아직 없는 행(달력 마지막 날 이벤트)은 이벤트일 종가로 대신한다.
    # 스케일 분모로만 쓰이므로 근사로 충분하고, 라벨에는 영향이 없다.
    df = df.merge(prices[["code", "date", "close"]], on=["code", "date"], how="left")
    shares_capacity = (df["adv"] * 1e6) / df["entry_price"].fillna(df["close"])
    for scaled, raw in (
        ("smart_money_net_5d_scaled", "smart_money_net_5d"),
        ("indiv_net_5d_scaled", "indiv_net_5d"),
        ("pension_net_5d_scaled", "pension_net_5d"),
    ):
        df[scaled] = df[raw] / shares_capacity
    df["log_adv"] = np.log10(df["adv"])
    return df[columns]
