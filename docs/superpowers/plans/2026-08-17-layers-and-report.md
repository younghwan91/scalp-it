# 계획 4: 층별 모델 비교 + 근거 카드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 §3 의 세 층(①수급 단독 / ②+규칙 뉴스 / ③+LLM)을 **동일 표본·동일 폴드·동일 음성대조 널** 위에서 비교해 각 층이 값을 하는지 판정하고, PEAD 대비 판정과 사람이 읽는 근거 카드, 6개월 뒤 채점 가능한 전진 기록까지 산출한다.

**Architecture:** 층은 "다른 데이터셋"이 아니라 **하나의 데이터셋 위에서 쓰는 피처 컬럼의 부분집합**이다. 세 층이 같은 행 위에서 돌게 만드는 것이 이 계획의 전부라 해도 된다 — 표본이 다르면 층 비교는 아무것도 말해주지 않는다. 그래서 데이터셋 조립 시점에 세 층 피처를 모두 붙이고 하나라도 결측인 행을 통째로 떨군다. 폴드 경계는 한 번 계산해 세 층에 그대로 넘기고, 널은 `excess_return` 만으로 만들어지므로 세 층에서 자동으로 동일해진다(테스트로 강제한다).

**Tech Stack:** Python ≥3.10, uv, pandas ≥2.0, scikit-learn, numpy, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-08-17-swing-it-design.md`

**선행 계획:** 계획 1 `2026-08-17-baseline-skeleton.md` (필수), 계획 2 뉴스 인제스트, 계획 3 LLM 피처

## Global Constraints

- **수정주가만 사용한다.** 가격은 `daily_bars_adjusted` 에서만 읽는다. `daily_bars` 는 분할 미반영이며 절대 쓰지 않는다.
- **point-in-time 엄수.** 시점 t 의 피처는 t 시점에 알 수 있는 정보만 쓴다. 롤링 통계는 반드시 `.shift(1)` 후 계산한다.
- **유동성 하한.** 이벤트 후보는 진입 직전 20거래일 평균 `trade_value` ≥ 5000 (백만원 = 50억원). 스펙 §5.
- **드리프트를 자르지 않는다.** 라벨은 D+21 ~ D+41 구간 보유. 중간 손절·이익실현 없음. 스펙 §2.
- **동시 보유 15종목 이상.** `top_n` 하한 15. 4~8 종목 집중은 이미 죽은 영역이다. 스펙 §2.
- **음성대조 없이 승격 없음.** 모든 성과 주장은 랜덤 진입 널 대비로 보고한다. 스펙 §2.
- **모델은 얕게 고정.** `GradientBoostingRegressor(max_depth=2, n_estimators=100)`. 층이 늘어도 여기서 올리지 않는다. 스펙 §6.
- **FAIL 을 숨기지 않는다.** 리포트는 FAIL 을 그대로 출력하고, 근거 카드에도 그 층의 게이트 판정을 함께 찍는다. 파라미터를 바꿔 재시도하는 경로는 만들지 않는다. 스펙 §10.
- **데이터 커밋 금지.** 결과물은 `data/`·`results/` 아래 두며 `.gitignore` 로 차단돼 있다.
- **DB 는 읽기 전용.**
- **계획 1·2·3 이 만든 파일은 수정하지 않는다.** 유일한 예외는 `pyproject.toml` 의 `[project.scripts]` 엔트리 추가(Task 7).

## 계획 1 에서 그대로 쓰는 것

이 계획은 계획 1 의 다음 이름들을 **깨지 않고 재사용**한다. 계획 1 테스트는 전부 계속 통과해야 한다.

| 이름 | 시그니처 |
|---|---|
| `swing_it.dataset.FEATURE_COLUMNS` | `list[str]` — 7개 수급·유동성 피처 |
| `swing_it.dataset.build_dataset` | `(prices, supply_demand, *, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` |
| `swing_it.model.baseline.walk_forward_predict` | `(ds, *, n_folds=4, min_train=200, seed=0) -> pd.DataFrame` |
| `swing_it.model.baseline.portfolio_returns` | `(pred, *, top_n=15) -> pd.DataFrame` |
| `swing_it.validate.gate.gate_report` | `(pred, *, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict` |
| `swing_it.validate.gate.random_null` | `(pred, *, top_n=15, n_trials=200, seed=0) -> np.ndarray` |

`build_dataset` 의 시그니처는 손대지 않는다. 층 확장은 **피처 목록을 인자로 받는 새 함수**(`build_layered_dataset`, `walk_forward_predict_features`)로 옆에 둔다. Task 2 는 `walk_forward_predict_features(ds, FEATURE_COLUMNS)` 가 `walk_forward_predict(ds)` 와 **행 단위로 동일한 결과**를 내는지 테스트로 고정한다 — 이게 계획 1 과의 호환 계약이다.

## 계획 2·3 에서 받는 것 — 프레임 계약

이 계획은 계획 2·3 의 **함수를 직접 import 하지 않는다.** 대신 조립된 피처 프레임과 컬럼 이름 목록을 **인자로** 받는다. 그래서 계획 2·3 의 내부 함수명이 바뀌어도 계획 4 는 깨지지 않는다.

| 인자 | 계약 |
|---|---|
| `news: pd.DataFrame` | 컬럼 `[code, date, *news_columns]`. `date` 는 `datetime64[ns]`, 이벤트일 기준. 계획 2 산출 |
| `news_columns: Sequence[str]` | 규칙 기반 뉴스 피처 이름 목록. 계획 2 의 `NEWS_FEATURE_COLUMNS` |
| `llm: pd.DataFrame` | 컬럼 `[code, date, *llm_columns]`. 계획 3 산출 |
| `llm_columns: Sequence[str]` | LLM 구조화 피처 이름 목록. 계획 3 의 `LLM_FEATURE_COLUMNS` |
| `articles: pd.DataFrame` | 근거 카드 표시용. 컬럼 `[code, date, title, outlet, published_at, url]`. 계획 2 산출 |

CLI(Task 7)에서만 계획 2·3 모듈을 import 해 이 프레임들을 얻는다. 어댑터 경계와 같은 발상이다(스펙 §8).

**뉴스 피처 결측 정책:** 건수 계열(`*_count`, `*_outlets`) 은 "기사 없음"이 곧 0 이므로 `fillna(0)`. 나머지 뉴스·LLM 피처의 결측은 채우지 않는다 → 그 행은 떨어진다. 결과적으로 **비교 표본은 LLM 피처가 존재하는 이벤트**로 좁혀진다. 스펙 §5 "기사가 유니버스를 정한다"와 일치하며, 세 층이 같은 행을 보게 하는 유일한 방법이다.

## 표본이 좁아지는 것에 대해

층 비교 표본은 계획 1 베이스라인 표본보다 작다(기사 없는 거래대금 급증 이벤트가 빠지므로). 그래서 **①층의 여기 성과는 계획 1 의 `results/baseline.json` 숫자와 다를 수 있다.** 이건 버그가 아니다. 계획 4 리포트는 두 숫자를 나란히 찍되, 층 간 비교에는 여기 숫자만 쓴다. 다른 표본 위의 두 숫자를 비교하는 것이 정확히 이 계획이 막으려는 오류다.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/swing_it/layers.py` | 층 정의 + 세 층 피처를 모두 붙인 단일 데이터셋 조립 |
| `src/swing_it/model/layered.py` | 피처 목록을 인자로 받는 walk-forward, 공유 폴드 경계, 최종 모델 적합 |
| `src/swing_it/validate/compare.py` | 동일조건 층 비교 · 페어드 부트스트랩 · LLM 정당화 판정 |
| `src/swing_it/validate/pead.py` | 겹침 보정 Sharpe·t 계산과 PEAD 대비 판정 |
| `src/swing_it/report/__init__.py` | 리포트 계층 선언 |
| `src/swing_it/report/cards.py` | 피처 기여도 · 유사 과거 사례 · 근거 카드 마크다운 |
| `src/swing_it/report/layers_md.py` | 층 비교 결과 마크다운 리포트 |
| `src/swing_it/report/forward_log.py` | 전진 기록 append · 사후 채점 |
| `src/swing_it/cli_layers.py` | `swing-layers` — 층별 비교 판정 |
| `src/swing_it/cli_today.py` | `swing-today` — 오늘 후보 카드 + 전진 기록 |
| `tests/` | 위 각 모듈 대응 테스트 |

## 사전등록 상수 (스펙 §6 — 그리드 돌리기 전에 확정)

```
top_n            = 15
n_folds          = 4
min_train        = 200
n_trials         = 200      (음성대조 널 시행 수)
n_boot           = 2000     (페어드 부트스트랩)
cost_one_way     = 0.0023   (왕복 46bp)
hold_end         = 40
seed             = 0
PEAD_SHARPE      = 0.80     (kr-quant 검증값, 스펙 §7)
PEAD_T           = 2.20
COMPARABLE_RATIO = 0.75     (PEAD Sharpe 의 75% 이상이면 '비등')
```

---

### Task 1: 층 정의와 층별 단일 데이터셋

**Files:**
- Create: `src/swing_it/layers.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Consumes:
  - `swing_it.dataset.FEATURE_COLUMNS: list[str]`
  - `swing_it.dataset.build_dataset(prices, supply_demand, *, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, fwd_return, excess_return]`
  - 계획 2 의 `news` 프레임 `[code, date, *news_columns]`, 계획 3 의 `llm` 프레임 `[code, date, *llm_columns]`
- Produces:
  - `LAYER_NAMES: list[str]` — `["supply", "supply_news", "supply_news_llm"]` (고정 순서 = 누적 순서)
  - `LAYER_LABELS: dict[str, str]` — 한국어 표시명. `{"supply": "① 수급 단독", "supply_news": "② + 규칙 뉴스", "supply_news_llm": "③ + LLM"}`
  - `COUNT_LIKE_SUFFIXES: tuple[str, ...]` — `("_count", "_outlets")`
  - `layer_features(news_columns: Sequence[str] = (), llm_columns: Sequence[str] = ()) -> dict[str, list[str]]` — 층 이름 → 그 층이 쓰는 피처 컬럼 목록(누적)
  - `build_layered_dataset(prices, supply_demand, *, news=None, news_columns=(), llm=None, llm_columns=(), require_label=True, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, *news_columns, *llm_columns, fwd_return, excess_return]`

`require_label=False` 는 오늘 후보 추론용이다. 라벨이 아직 존재할 수 없는 최근 행을 남긴다. 피처 결측은 그때도 허용하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_layers.py`:

```python
import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.layers import (
    LAYER_LABELS,
    LAYER_NAMES,
    build_layered_dataset,
    layer_features,
)

NEWS_COLS = ["news_count_ratio", "news_outlets"]
LLM_COLS = ["llm_specificity", "llm_persistence"]


def _prices(codes: list[str], n: int, spike_at: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "open": 100.0 + i,
                    "close": 100.0 + i,
                    "trade_value": 6000.0 * (5.0 if i == spike_at else 1.0),
                }
            )
    return pd.DataFrame(rows)


def _supply(codes: list[str], n: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "individual": -100,
                    "foreign_": 60,
                    "institution": 40,
                    "penfnd_etc": 10,
                    "acc_trde_qty": 1000,
                }
            )
    return pd.DataFrame(rows)


def _side(codes: list[str], n: int, columns: list[str], value: float) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            row = {
                "code": code,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
            row.update({col: value for col in columns})
            rows.append(row)
    return pd.DataFrame(rows)


def test_layers_are_cumulative():
    spec = layer_features(NEWS_COLS, LLM_COLS)
    assert list(spec) == LAYER_NAMES
    assert spec["supply"] == FEATURE_COLUMNS
    assert spec["supply_news"] == [*FEATURE_COLUMNS, *NEWS_COLS]
    assert spec["supply_news_llm"] == [*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]


def test_layer_features_defaults_to_supply_only():
    spec = layer_features()
    assert spec["supply"] == spec["supply_news"] == spec["supply_news_llm"]


def test_layer_labels_cover_every_layer():
    assert set(LAYER_LABELS) == set(LAYER_NAMES)


def test_layered_dataset_column_order():
    codes = ["A", "B"]
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25),
        _supply(codes, 80),
        news=_side(codes, 80, NEWS_COLS, 2.0),
        news_columns=NEWS_COLS,
        llm=_side(codes, 80, LLM_COLS, 0.5),
        llm_columns=LLM_COLS,
        hold_start=10,
        hold_end=20,
    )
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS,
        "fwd_return", "excess_return",
    ]


def test_every_layer_sees_identical_rows():
    """세 층이 같은 행 위에서 돌지 않으면 층 비교는 아무 의미가 없다."""
    codes = ["A", "B"]
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25),
        _supply(codes, 80),
        news=_side(codes, 80, NEWS_COLS, 2.0),
        news_columns=NEWS_COLS,
        llm=_side(codes, 80, LLM_COLS, 0.5),
        llm_columns=LLM_COLS,
        hold_start=10,
        hold_end=20,
    )
    spec = layer_features(NEWS_COLS, LLM_COLS)
    for columns in spec.values():
        assert ds[columns].notna().all().all()


def test_missing_llm_row_is_dropped_from_all_layers():
    codes = ["A", "B"]
    llm = _side(codes, 80, LLM_COLS, 0.5)
    # B 의 급증일 LLM 피처를 결측으로 만든다 → B 는 ①층에서도 빠져야 한다
    spike_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=25)
    mask = (llm["code"] == "B") & (llm["date"] == spike_date)
    llm.loc[mask, LLM_COLS] = np.nan
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25),
        _supply(codes, 80),
        news=_side(codes, 80, NEWS_COLS, 2.0),
        news_columns=NEWS_COLS,
        llm=llm,
        llm_columns=LLM_COLS,
        hold_start=10,
        hold_end=20,
    )
    assert "B" not in set(ds["code"])


def test_count_like_news_columns_fill_zero():
    """기사가 없는 이벤트의 '건수 0' 은 결측이 아니라 정보다."""
    codes = ["A", "B"]
    news = _side(["A"], 80, NEWS_COLS, 2.0)  # B 의 뉴스 행은 아예 없다
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25),
        _supply(codes, 80),
        news=news,
        news_columns=["news_count_ratio", "news_outlets"],
        hold_start=10,
        hold_end=20,
    )
    b = ds[ds["code"] == "B"]
    assert len(b) == 1
    assert b.iloc[0]["news_outlets"] == 0.0


def test_require_label_false_keeps_unlabelled_recent_rows():
    codes = ["A", "B"]
    prices = _prices(codes, 80, spike_at=78)  # 미래가 없어 라벨이 안 나오는 급증일
    ds = build_layered_dataset(
        prices,
        _supply(codes, 80),
        news=_side(codes, 80, NEWS_COLS, 2.0),
        news_columns=NEWS_COLS,
        llm=_side(codes, 80, LLM_COLS, 0.5),
        llm_columns=LLM_COLS,
        require_label=False,
        hold_start=10,
        hold_end=20,
    )
    assert len(ds) == 2
    assert ds["excess_return"].isna().all()
    assert ds[[*FEATURE_COLUMNS, *NEWS_COLS, *LLM_COLS]].notna().all().all()


def test_no_side_frames_reduces_to_baseline_columns():
    codes = ["A", "B"]
    ds = build_layered_dataset(
        _prices(codes, 80, spike_at=25), _supply(codes, 80), hold_start=10, hold_end=20
    )
    assert list(ds.columns) == [
        "code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"
    ]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_layers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.layers'`

- [ ] **Step 3: 구현**

`src/swing_it/layers.py`:

```python
"""층 정의와 층별 단일 데이터셋 조립 (스펙 §3).

핵심은 하나다: **세 층이 같은 행 위에서 돌아야 한다.** 층마다 표본이 다르면
"③이 ②를 이겼다"는 문장이 아무것도 말해주지 않는다. 그래서 층을 서로 다른
데이터셋으로 만들지 않고, 세 층 피처를 전부 붙인 하나의 테이블을 만든 뒤
층은 그 위에서 쓰는 **컬럼의 부분집합**으로만 정의한다.

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

# 이 접미사로 끝나는 뉴스 피처는 결측을 0 으로 채운다.
# '기사가 한 건도 없었다'는 모르는 값이 아니라 관측된 값이다.
COUNT_LIKE_SUFFIXES: tuple[str, ...] = ("_count", "_outlets")


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
    frame["date"] = pd.to_datetime(frame["date"])
    out = base.merge(frame, on=["code", "date"], how="left")
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col.endswith(COUNT_LIKE_SUFFIXES):
            out[col] = out[col].fillna(0.0)
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
    base = build_dataset(
        prices,
        supply_demand,
        adv_floor=adv_floor,
        tv_multiple=tv_multiple,
        entry_offset=entry_offset,
        hold_start=hold_start,
        hold_end=hold_end,
    )
    if not require_label:
        # build_dataset 은 라벨 결측 행을 이미 떨궜다. 라벨 없는 이벤트를 되살리려면
        # 이벤트·피처만 다시 만들고 라벨은 왼쪽 조인한다.
        base = _unlabelled_base(
            prices,
            supply_demand,
            adv_floor=adv_floor,
            tv_multiple=tv_multiple,
            entry_offset=entry_offset,
            hold_start=hold_start,
            hold_end=hold_end,
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

    events = extract_events(prices, adv_floor=adv_floor, tv_multiple=tv_multiple)
    empty = ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]
    if events.empty:
        return pd.DataFrame(columns=empty)

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
    close = prices[["code", "date", "close"]]
    df = df.merge(close, on=["code", "date"], how="left")
    entry = df["entry_price"].fillna(df["close"])
    shares_capacity = (df["adv"] * 1e6) / entry
    for scaled, raw in (
        ("smart_money_net_5d_scaled", "smart_money_net_5d"),
        ("indiv_net_5d_scaled", "indiv_net_5d"),
        ("pension_net_5d_scaled", "pension_net_5d"),
    ):
        df[scaled] = df[raw] / shares_capacity
    df["log_adv"] = np.log10(df["adv"])
    return df[empty]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_layers.py -v`
Expected: 9 passed

- [ ] **Step 5: 계획 1 회귀 확인**

Run: `uv run pytest tests/test_dataset.py tests/test_events.py tests/test_labels.py -v`
Expected: 전부 통과 (계획 1 의 시그니처를 건드리지 않았음을 확인한다)

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/layers.py tests/test_layers.py
git commit -m "feat(layers): 세 층이 동일 표본을 쓰는 단일 데이터셋 조립"
```

---

### Task 2: 피처 목록을 받는 walk-forward 와 공유 폴드

**Files:**
- Create: `src/swing_it/model/layered.py`
- Test: `tests/test_model_layered.py`

**Interfaces:**
- Consumes:
  - `swing_it.dataset.FEATURE_COLUMNS`
  - `swing_it.model.baseline.walk_forward_predict(ds, *, n_folds=4, min_train=200, seed=0) -> pd.DataFrame` — 동등성 검증용
- Produces:
  - `fold_dates(ds: pd.DataFrame, *, n_folds: int = 4, min_train: int = 200) -> list[np.ndarray]` — 폴드별 검증 날짜 배열. **한 번 계산해 모든 층에 넘긴다.**
  - `walk_forward_predict_features(ds: pd.DataFrame, feature_columns: Sequence[str], *, folds: list[np.ndarray] | None = None, n_folds: int = 4, min_train: int = 200, seed: int = 0) -> pd.DataFrame` — 컬럼 `[code, date, excess_return, prediction, fold]`. 계획 1 과 동일한 컬럼·동일한 폴드 규칙.
  - `fit_final_model(ds: pd.DataFrame, feature_columns: Sequence[str], *, seed: int = 0) -> tuple[GradientBoostingRegressor, pd.Series]` — 전체 데이터로 적합한 모델과 **학습셋 피처 중앙값**(기여도 계산의 기준점)

`fold_dates` 를 분리하는 이유가 이 태스크의 전부다. 층마다 폴드를 따로 계산하면 표본이 같아도 경계가 미세하게 달라질 수 있고, 그러면 "③이 ②를 이겼다"가 폴드 차이일 수 있다.

모델은 `GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=seed)` 로 고정한다. 층이 늘어도 여기서 올리지 않는다(스펙 §6).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_model_layered.py`:

```python
import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.model.baseline import walk_forward_predict
from swing_it.model.layered import (
    fit_final_model,
    fold_dates,
    walk_forward_predict_features,
)

EXTRA = ["news_count_ratio", "llm_specificity"]


def _dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in [*FEATURE_COLUMNS, *EXTRA]})
    df["code"] = [f"{i % 40:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    df["excess_return"] = df[FEATURE_COLUMNS[0]] * 0.05 + rng.normal(scale=0.01, size=n)
    return df[["code", "date", *FEATURE_COLUMNS, *EXTRA, "excess_return"]]


def test_matches_plan1_walk_forward_exactly():
    """계획 1 과의 호환 계약. 이게 깨지면 층 비교의 ①층이 베이스라인이 아니다."""
    ds = _dataset(1200)
    expected = walk_forward_predict(ds, n_folds=4, min_train=200, seed=0)
    got = walk_forward_predict_features(
        ds, FEATURE_COLUMNS, n_folds=4, min_train=200, seed=0
    )
    pd.testing.assert_frame_equal(expected, got)


def test_extra_features_are_actually_used():
    ds = _dataset(1200)
    ds["planted"] = ds["excess_return"] * 3.0  # 라벨을 거의 그대로 담은 피처
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, "planted"])
    assert b["prediction"].corr(b["excess_return"]) > a["prediction"].corr(a["excess_return"])


def test_fold_dates_are_contiguous_and_ordered():
    folds = fold_dates(_dataset(1200), n_folds=4, min_train=200)
    assert len(folds) == 4
    for i in range(1, len(folds)):
        assert folds[i][0] > folds[i - 1][-1]


def test_shared_folds_give_identical_fold_boundaries_across_feature_sets():
    ds = _dataset(1200)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    a = walk_forward_predict_features(ds, FEATURE_COLUMNS, folds=folds)
    b = walk_forward_predict_features(ds, [*FEATURE_COLUMNS, *EXTRA], folds=folds)
    pd.testing.assert_frame_equal(
        a[["code", "date", "excess_return", "fold"]],
        b[["code", "date", "excess_return", "fold"]],
    )


def test_never_trains_on_the_dates_it_predicts():
    ds = _dataset(1200)
    folds = fold_dates(ds, n_folds=4, min_train=200)
    pred = walk_forward_predict_features(ds, FEATURE_COLUMNS, folds=folds)
    bounds = pred.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    for i in range(1, len(bounds)):
        assert bounds.iloc[i]["min"] > bounds.iloc[i - 1]["max"]


def test_fit_final_model_returns_model_and_median_baseline():
    ds = _dataset(600)
    model, baseline = fit_final_model(ds, [*FEATURE_COLUMNS, *EXTRA])
    assert list(baseline.index) == [*FEATURE_COLUMNS, *EXTRA]
    assert np.isclose(baseline["news_count_ratio"], ds["news_count_ratio"].median())
    assert model.predict(ds[[*FEATURE_COLUMNS, *EXTRA]].head(3)).shape == (3,)


def test_model_stays_shallow():
    """스펙 §6 — 층이 늘어도 모델은 얕게 고정한다."""
    model, _ = fit_final_model(_dataset(600), FEATURE_COLUMNS)
    assert model.max_depth == 2
    assert model.n_estimators == 100
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_model_layered.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.model.layered'`

- [ ] **Step 3: 구현**

`src/swing_it/model/layered.py`:

```python
"""피처 목록을 인자로 받는 walk-forward — 층 비교의 실행부.

계획 1 의 ``walk_forward_predict`` 는 ``FEATURE_COLUMNS`` 를 고정으로 쓴다.
그 시그니처를 깨지 않기 위해 여기에 확장판을 따로 둔다. 두 함수가 같은 결과를
내는지는 ``tests/test_model_layered.py::test_matches_plan1_walk_forward_exactly``
가 고정한다.

``fold_dates`` 를 분리한 것이 이 모듈의 요점이다. 폴드 경계를 한 번 계산해
세 층에 그대로 넘겨야 "③이 ②를 이겼다"가 폴드 차이가 아님을 보증할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

MAX_DEPTH = 2
N_ESTIMATORS = 100


def fold_dates(
    ds: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 200,
) -> list[np.ndarray]:
    """검증 구간을 시간순 ``n_folds`` 등분한 날짜 배열 목록.

    계획 1 ``walk_forward_predict`` 와 동일한 규칙이다. 폴드 경계는 날짜 단위로
    끊는다 — 같은 날짜가 학습과 예측에 동시에 들어가면 같은 날 다른 종목의
    결과가 새어 들어온다.
    """
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    if len(df) <= min_train:
        raise ValueError(f"표본이 {len(df)} 행뿐이라 min_train={min_train} 을 못 채운다")

    first_test_date = df.loc[min_train, "date"]
    test_dates = dates[dates > first_test_date]
    if len(test_dates) < n_folds:
        raise ValueError(f"검증 가능한 날짜가 {len(test_dates)} 개뿐이라 {n_folds} 폴드를 못 만든다")
    return list(np.array_split(test_dates, n_folds))


def walk_forward_predict_features(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    folds: list[np.ndarray] | None = None,
    n_folds: int = 4,
    min_train: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """지정한 피처만 써서 시간순 폴드로 학습·예측한다.

    Args:
        folds: ``fold_dates`` 결과. 층 비교에서는 **반드시** 미리 계산해 넘긴다.
            생략하면 이 함수가 직접 계산한다(단일 층 실행용).
    """
    columns = list(feature_columns)
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    if folds is None:
        folds = fold_dates(df, n_folds=n_folds, min_train=min_train)

    results = []
    for fold, chunk in enumerate(folds):
        train = df[df["date"] < chunk[0]]
        test = df[df["date"].isin(chunk)]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(
            max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
        )
        model.fit(train[columns], train["excess_return"])
        block = test[["code", "date", "excess_return"]].copy()
        block["prediction"] = model.predict(test[columns])
        block["fold"] = fold
        results.append(block)

    return pd.concat(results, ignore_index=True)


def fit_final_model(
    ds: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[GradientBoostingRegressor, pd.Series]:
    """전 구간으로 적합한 모델과 피처 중앙값을 함께 돌려준다.

    중앙값은 근거 카드의 기여도 계산에서 '평범한 이벤트' 기준점으로 쓴다
    (Task 5). 모델과 함께 나와야 기준점이 학습 분포와 어긋나지 않는다.
    """
    columns = list(feature_columns)
    model = GradientBoostingRegressor(
        max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=seed
    )
    model.fit(ds[columns], ds["excess_return"])
    return model, ds[columns].median()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_model_layered.py -v`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/model/layered.py tests/test_model_layered.py
git commit -m "feat(model): 피처 지정 walk-forward + 층 간 공유 폴드 경계"
```

---

### Task 3: 동일조건 층 비교 하네스

**Files:**
- Create: `src/swing_it/validate/compare.py`
- Test: `tests/test_validate_compare.py`

**Interfaces:**
- Consumes:
  - `swing_it.layers.LAYER_NAMES`, `layer_features`
  - `swing_it.model.layered.fold_dates`, `walk_forward_predict_features`
  - `swing_it.model.baseline.portfolio_returns(pred, *, top_n=15) -> pd.DataFrame`
  - `swing_it.validate.gate.gate_report(pred, *, top_n=15, n_trials=200, cost_one_way=0.0023, seed=0) -> dict`
  - `swing_it.validate.gate.random_null(pred, *, top_n=15, n_trials=200, seed=0) -> np.ndarray`
- Produces:
  - `paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict` — `a`, `b` 는 `portfolio_returns` 프레임. 날짜로 짝지어 `a - b` 차이를 부트스트랩. 키 `mean_diff, ci_low, ci_high, p_greater, n_dates`
  - `compare_layers(ds: pd.DataFrame, spec: dict[str, list[str]], *, top_n: int = 15, n_folds: int = 4, min_train: int = 200, n_trials: int = 200, n_boot: int = 2000, cost_one_way: float = 0.0023, seed: int = 0) -> dict` — 키 `rows`(층별 `list[dict]`), `deltas`(인접 층 비교 `list[dict]`), `llm`(LLM 정당화 판정 `dict`), `sample`(표본 메타 `dict`)
  - `llm_verdict(layer_rows: list[dict], deltas: list[dict]) -> dict` — 키 `layer2_mean, layer3_mean, delta, ci_low, ci_high, p_greater, layer3_gate, verdict`. `verdict` 는 `"LLM_JUSTIFIED"` 또는 `"LLM_NOT_JUSTIFIED"`

`compare_layers` 의 `rows` 각 원소 키: `layer, label, n_features, strategy_mean, null_mean, null_p95, percentile, net_mean, verdict`.
`deltas` 각 원소 키: `layer, baseline, mean_diff, ci_low, ci_high, p_greater, n_dates`.

**LLM 정당화 판정 규칙 (사전등록):** `verdict == "LLM_JUSTIFIED"` 는 세 조건이 **모두** 참일 때만.
1. ③ − ② 평균 차이가 양수
2. 그 차이의 부트스트랩 95% 신뢰구간 하한이 0 초과
3. ③층 자체가 음성대조 게이트 `PASS`

③이 널을 못 이기면 ②보다 나아 봤자 둘 다 노이즈다. 그래서 3번이 들어간다.

**널 동일성:** `random_null` 은 `pred` 의 `excess_return` 과 날짜 그룹만 쓴다. 세 층의 `pred` 는 같은 행 집합이므로(Task 1) 널 배열이 **바이트 단위로 같아야 한다.** 이걸 테스트로 강제한다. 같지 않다면 표본이 어긋났다는 뜻이고, 그 순간 비교는 무효다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validate_compare.py`:

```python
import numpy as np
import pandas as pd
import pytest

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.validate.compare import compare_layers, llm_verdict, paired_bootstrap

NEWS = ["news_count_ratio"]
LLM = ["llm_specificity"]


def _ds(n: int, *, llm_is_signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [*FEATURE_COLUMNS, *NEWS, *LLM]
    df = pd.DataFrame({col: rng.normal(size=n) for col in cols})
    df["code"] = [f"{i % 30:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 30, unit="D")
    noise = rng.normal(scale=0.02, size=n)
    df["excess_return"] = noise + (df["llm_specificity"] * 0.08 if llm_is_signal else 0.0)
    return df[["code", "date", *cols, "excess_return"]]


def _spec() -> dict[str, list[str]]:
    return {
        "supply": list(FEATURE_COLUMNS),
        "supply_news": [*FEATURE_COLUMNS, *NEWS],
        "supply_news_llm": [*FEATURE_COLUMNS, *NEWS, *LLM],
    }


def _portfolio(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.01, size=200)
    out = paired_bootstrap(
        _portfolio(list(base + 0.02)), _portfolio(list(base)), n_boot=1000
    )
    assert np.isclose(out["mean_diff"], 0.02, atol=0.005)
    assert out["ci_low"] > 0
    assert out["p_greater"] > 0.95
    assert out["n_dates"] == 200


def test_paired_bootstrap_finds_nothing_when_series_are_identical():
    values = list(np.random.default_rng(1).normal(scale=0.01, size=200))
    out = paired_bootstrap(_portfolio(values), _portfolio(values), n_boot=1000)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_paired_bootstrap_pairs_on_date_not_position():
    a = _portfolio([0.1, 0.2, 0.3])
    b = _portfolio([0.1, 0.2, 0.3]).iloc[::-1].reset_index(drop=True)
    out = paired_bootstrap(a, b, n_boot=200)
    assert out["mean_diff"] == 0.0  # 날짜로 짝지으면 차이가 0 이다


def test_compare_layers_returns_one_row_per_layer():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    assert [r["layer"] for r in result["rows"]] == [
        "supply", "supply_news", "supply_news_llm"
    ]
    assert [r["n_features"] for r in result["rows"]] == [7, 8, 9]


def test_all_layers_share_an_identical_null():
    """널이 다르면 세 층이 다른 표본 위에 있다는 뜻이고, 비교는 그 순간 무효다."""
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    nulls = {r["null_mean"] for r in result["rows"]}
    p95s = {r["null_p95"] for r in result["rows"]}
    assert len(nulls) == 1
    assert len(p95s) == 1


def test_llm_justified_when_llm_feature_carries_the_signal():
    result = compare_layers(
        _ds(3000, llm_is_signal=True), _spec(), n_trials=200, n_boot=1000
    )
    assert result["llm"]["verdict"] == "LLM_JUSTIFIED"
    assert result["llm"]["delta"] > 0


def test_llm_not_justified_when_llm_feature_is_noise():
    result = compare_layers(
        _ds(3000, llm_is_signal=False), _spec(), n_trials=200, n_boot=1000
    )
    assert result["llm"]["verdict"] == "LLM_NOT_JUSTIFIED"


def test_llm_not_justified_even_if_it_beats_layer2_but_fails_the_gate():
    """②보다 낫지만 랜덤 널을 못 이기면 둘 다 노이즈다."""
    rows = [
        {"layer": "supply_news", "strategy_mean": 0.001, "verdict": "FAIL"},
        {"layer": "supply_news_llm", "strategy_mean": 0.004, "verdict": "FAIL"},
    ]
    deltas = [
        {
            "layer": "supply_news_llm",
            "baseline": "supply_news",
            "mean_diff": 0.003,
            "ci_low": 0.001,
            "ci_high": 0.005,
            "p_greater": 0.99,
            "n_dates": 100,
        }
    ]
    assert llm_verdict(rows, deltas)["verdict"] == "LLM_NOT_JUSTIFIED"


def test_deltas_compare_each_layer_to_the_previous_one():
    result = compare_layers(_ds(1500, llm_is_signal=True), _spec(), n_trials=50, n_boot=200)
    pairs = [(d["layer"], d["baseline"]) for d in result["deltas"]]
    assert pairs == [
        ("supply_news", "supply"),
        ("supply_news_llm", "supply_news"),
    ]


def test_sample_metadata_is_reported():
    ds = _ds(1500, llm_is_signal=True)
    result = compare_layers(ds, _spec(), n_trials=50, n_boot=200)
    assert result["sample"]["n_rows"] == len(ds)
    assert result["sample"]["n_codes"] == ds["code"].nunique()
    assert result["sample"]["top_n"] == 15


def test_compare_layers_rejects_a_spec_whose_layers_are_not_nested():
    ds = _ds(1500, llm_is_signal=True)
    bad = {"supply": list(FEATURE_COLUMNS), "supply_news": NEWS}
    with pytest.raises(ValueError, match="누적"):
        compare_layers(ds, bad, n_trials=50, n_boot=200)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_validate_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.validate.compare'`

- [ ] **Step 3: 구현**

`src/swing_it/validate/compare.py`:

```python
"""층 비교 하네스 — 세 층을 동일 조건에서 재는 유일한 경로 (스펙 §3).

동일 조건이란 셋을 말한다:

1. **동일 표본** — ``layers.build_layered_dataset`` 이 보장한다. 여기서는
   층 피처가 서로 누적 관계인지만 다시 확인한다.
2. **동일 폴드** — ``fold_dates`` 를 한 번 계산해 세 층에 그대로 넘긴다.
3. **동일 널** — ``random_null`` 은 ``excess_return`` 과 날짜만 쓰므로,
   1·2 가 지켜지면 세 층의 널이 같아진다. 같지 않으면 어딘가 어긋난 것이다.

층 간 차이가 우연인지는 날짜 페어드 부트스트랩으로 잰다. 같은 날짜의
두 층 포트폴리오를 짝지어 차이를 보므로 시장 공통 변동이 상쇄된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..layers import LAYER_LABELS
from ..model.baseline import portfolio_returns
from ..model.layered import fold_dates, walk_forward_predict_features
from .gate import gate_report


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
    draws = rng.integers(0, n, size=(n_boot, n))
    means = diff[draws].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_greater": float((means > 0).mean()),
        "n_dates": int(n),
    }


def _check_nested(spec: dict[str, list[str]]) -> None:
    names = list(spec)
    for prev, cur in zip(names, names[1:]):
        if spec[cur][: len(spec[prev])] != spec[prev]:
            raise ValueError(
                f"층 '{cur}' 이 '{prev}' 의 누적 확장이 아니다. "
                "층은 피처를 더하기만 해야 기여도 분리가 성립한다."
            )


def llm_verdict(layer_rows: list[dict], deltas: list[dict]) -> dict:
    """③층이 LLM 을 쓸 값을 하는지 판정한다 (스펙 §3).

    세 조건이 모두 참일 때만 ``LLM_JUSTIFIED``:
    ③−② 차이가 양수 / 그 차이의 95% CI 하한이 0 초과 / ③층이 게이트 PASS.

    세 번째 조건이 있는 이유: ③이 랜덤 널을 못 이기면 ②보다 나아 봤자
    둘 다 노이즈다. 노이즈끼리의 우열은 LLM 비용을 정당화하지 않는다.
    """
    by_layer = {row["layer"]: row for row in layer_rows}
    layer2 = by_layer["supply_news"]
    layer3 = by_layer["supply_news_llm"]
    delta = next(d for d in deltas if d["layer"] == "supply_news_llm")

    passed = (
        delta["mean_diff"] > 0
        and delta["ci_low"] > 0
        and layer3["verdict"] == "PASS"
    )
    return {
        "layer2_mean": float(layer2["strategy_mean"]),
        "layer3_mean": float(layer3["strategy_mean"]),
        "delta": float(delta["mean_diff"]),
        "ci_low": float(delta["ci_low"]),
        "ci_high": float(delta["ci_high"]),
        "p_greater": float(delta["p_greater"]),
        "layer3_gate": layer3["verdict"],
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
    seed: int = 0,
) -> dict:
    """세 층을 동일 표본·동일 폴드·동일 널로 비교한다."""
    _check_nested(spec)
    folds = fold_dates(ds, n_folds=n_folds, min_train=min_train)

    rows: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    for layer, columns in spec.items():
        pred = walk_forward_predict_features(ds, columns, folds=folds, seed=seed)
        report = gate_report(
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
                **report,
            }
        )

    names = list(spec)
    deltas = [
        {"layer": cur, "baseline": prev, **paired_bootstrap(
            curves[cur], curves[prev], n_boot=n_boot, seed=seed
        )}
        for prev, cur in zip(names, names[1:])
    ]

    has_llm_layer = "supply_news_llm" in spec and "supply_news" in spec
    return {
        "rows": rows,
        "deltas": deltas,
        "llm": llm_verdict(rows, deltas) if has_llm_layer else None,
        "sample": {
            "n_rows": int(len(ds)),
            "n_codes": int(ds["code"].nunique()),
            "date_range": [str(ds["date"].min().date()), str(ds["date"].max().date())],
            "n_folds": n_folds,
            "top_n": top_n,
            "n_trials": n_trials,
            "n_boot": n_boot,
            "cost_one_way": cost_one_way,
            "seed": seed,
        },
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_validate_compare.py -v`
Expected: 11 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/validate/compare.py tests/test_validate_compare.py
git commit -m "feat(validate): 동일 표본·폴드·널 층 비교 + LLM 정당화 판정"
```

---

### Task 4: PEAD 대비 벤치마크

**Files:**
- Create: `src/swing_it/validate/pead.py`
- Test: `tests/test_validate_pead.py`

**Interfaces:**
- Consumes: `swing_it.model.baseline.portfolio_returns(pred, *, top_n=15) -> pd.DataFrame`
- Produces:
  - `PEAD_SHARPE: float` = `0.80`, `PEAD_T: float` = `2.20`, `COMPARABLE_RATIO: float` = `0.75`
  - `overlap_stats(curve: pd.DataFrame, *, hold_days: int = 40) -> dict` — 키 `mean, sd, n_obs, n_effective, sharpe_annual, t_stat`
  - `pead_benchmark(curve: pd.DataFrame, *, hold_days: int = 40) -> dict` — `overlap_stats` 의 모든 키 + `pead_sharpe, pead_t, verdict`. `verdict` ∈ `{"BEATS_PEAD", "COMPARABLE_TO_PEAD", "LOSES_TO_PEAD"}`

**겹침 보정이 이 태스크의 핵심이다.** 이벤트가 매일 발생하고 보유가 40거래일이므로 관측이 독립이 아니다. 보정 없이 t 를 계산하면 40배 가까이 부풀려진다.

- 관측 단위: 날짜별 `mean_excess` (40거래일 보유 수익률)
- 연율 Sharpe: `mean / sd * sqrt(252 / hold_days)`
- 유효 표본: `n_effective = max(1, floor(n_obs / hold_days))` — 겹치지 않는 블록 수
- t: `mean / (sd / sqrt(n_effective))`

이 보정은 보수적이다. 진짜 유효 표본은 이보다 조금 클 수 있다. 여기서는 부풀림을 막는 쪽으로 틀리는 것이 옳다 — 스펙 §2 가 남긴 교훈이 정확히 그것이다.

**판정 (사전등록):**
- `BEATS_PEAD` — `sharpe_annual > 0.80` **그리고** `t_stat > 2.20`
- `COMPARABLE_TO_PEAD` — `sharpe_annual >= 0.60` (`0.80 × 0.75`)
- 그 외 `LOSES_TO_PEAD`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validate_pead.py`:

```python
import numpy as np
import pandas as pd

from swing_it.validate.pead import (
    COMPARABLE_RATIO,
    PEAD_SHARPE,
    PEAD_T,
    overlap_stats,
    pead_benchmark,
)


def _curve(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(values), freq="B"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_registered_constants_match_the_spec():
    """스펙 §7 — kr-quant PEAD 검증값 Sharpe ~0.8, t ~2.2."""
    assert PEAD_SHARPE == 0.80
    assert PEAD_T == 2.20
    assert COMPARABLE_RATIO == 0.75


def test_sharpe_is_annualised_from_the_holding_period():
    values = [0.02] * 400
    values[0] = 0.01  # sd 가 0 이 되지 않게
    stats = overlap_stats(_curve(values), hold_days=40)
    series = pd.Series(values)
    expected = series.mean() / series.std(ddof=1) * np.sqrt(252 / 40)
    assert np.isclose(stats["sharpe_annual"], expected)


def test_effective_sample_divides_by_the_holding_period():
    stats = overlap_stats(_curve(list(np.random.default_rng(0).normal(size=400))), hold_days=40)
    assert stats["n_obs"] == 400
    assert stats["n_effective"] == 10


def test_effective_sample_never_drops_below_one():
    stats = overlap_stats(_curve([0.01, 0.02, 0.03]), hold_days=40)
    assert stats["n_effective"] == 1


def test_overlap_correction_shrinks_the_t_stat():
    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=0.01, scale=0.05, size=400))
    stats = overlap_stats(_curve(values), hold_days=40)
    naive_t = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(400))
    assert stats["t_stat"] < naive_t / 5  # 겹침 보정이 실제로 걸려 있어야 한다


def test_beats_pead_needs_both_sharpe_and_t():
    rng = np.random.default_rng(1)
    strong = list(rng.normal(loc=0.06, scale=0.05, size=800))
    out = pead_benchmark(_curve(strong), hold_days=40)
    assert out["sharpe_annual"] > PEAD_SHARPE
    assert out["t_stat"] > PEAD_T
    assert out["verdict"] == "BEATS_PEAD"


def test_loses_to_pead_when_there_is_no_edge():
    rng = np.random.default_rng(2)
    out = pead_benchmark(_curve(list(rng.normal(loc=0.0, scale=0.05, size=800))), hold_days=40)
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_negative_mean_never_reads_as_comparable():
    rng = np.random.default_rng(3)
    out = pead_benchmark(_curve(list(rng.normal(loc=-0.03, scale=0.05, size=800))), hold_days=40)
    assert out["sharpe_annual"] < 0
    assert out["verdict"] == "LOSES_TO_PEAD"


def test_report_carries_the_pead_reference_values():
    out = pead_benchmark(_curve([0.01, 0.02, 0.03, 0.04] * 50), hold_days=40)
    assert out["pead_sharpe"] == PEAD_SHARPE
    assert out["pead_t"] == PEAD_T
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_validate_pead.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.validate.pead'`

- [ ] **Step 3: 구현**

`src/swing_it/validate/pead.py`:

```python
"""PEAD 대비 벤치마크 (스펙 §7).

PEAD 는 이 레포의 중심이 아니라 **넘어야 할 선**이다. 이미 검증된 알파가
있는데 뉴스 모델이 그보다 못하다면, 뉴스 모델을 운영에 올릴 이유가 없다.

기준값은 kr-quant PEAD 검증 결과다: 전 구간 Sharpe ~0.8, t ~2.2
(저회전, point-in-time 유동성 하한 적용본).

겹침 보정: 이벤트는 매일 나고 보유는 40거래일이라 날짜별 관측이 독립이 아니다.
보정 없는 t 는 크게 부풀려진다. 여기서는 겹치지 않는 블록 수
(``n_obs // hold_days``) 를 유효 표본으로 쓴다. 보수적인 쪽으로 틀리는 선택이다.
"""

from __future__ import annotations

import math

import pandas as pd

PEAD_SHARPE: float = 0.80
PEAD_T: float = 2.20
COMPARABLE_RATIO: float = 0.75

TRADING_DAYS = 252


def overlap_stats(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """날짜별 포트폴리오 초과수익 시계열의 겹침 보정 통계.

    Args:
        curve: ``portfolio_returns`` 결과. ``mean_excess`` 는 40거래일 보유 수익률.
        hold_days: 보유 거래일 수. 겹침 정도를 결정한다.
    """
    values = curve["mean_excess"].dropna().to_numpy(dtype=float)
    n_obs = int(len(values))
    if n_obs < 2:
        raise ValueError(f"관측이 {n_obs} 개뿐이라 분산을 못 잰다")

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    n_effective = max(1, n_obs // hold_days)
    sharpe = mean / sd * math.sqrt(TRADING_DAYS / hold_days) if sd > 0 else 0.0
    t_stat = mean / (sd / math.sqrt(n_effective)) if sd > 0 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "n_obs": n_obs,
        "n_effective": int(n_effective),
        "sharpe_annual": float(sharpe),
        "t_stat": float(t_stat),
    }


def pead_benchmark(curve: pd.DataFrame, *, hold_days: int = 40) -> dict:
    """검증된 PEAD 대비 판정. 이기든 지든 명시적으로 남긴다 (스펙 §10.5)."""
    stats = overlap_stats(curve, hold_days=hold_days)
    sharpe = stats["sharpe_annual"]
    if sharpe > PEAD_SHARPE and stats["t_stat"] > PEAD_T:
        verdict = "BEATS_PEAD"
    elif sharpe >= PEAD_SHARPE * COMPARABLE_RATIO:
        verdict = "COMPARABLE_TO_PEAD"
    else:
        verdict = "LOSES_TO_PEAD"
    return {**stats, "pead_sharpe": PEAD_SHARPE, "pead_t": PEAD_T, "verdict": verdict}


def benchmark_all_layers(
    curves: dict[str, pd.DataFrame], *, hold_days: int = 40
) -> list[dict]:
    """층별 PEAD 판정 목록. ``[{layer, ...pead_benchmark}]``."""
    return [
        {"layer": layer, **pead_benchmark(curve, hold_days=hold_days)}
        for layer, curve in curves.items()
    ]


__all__ = [
    "COMPARABLE_RATIO",
    "PEAD_SHARPE",
    "PEAD_T",
    "benchmark_all_layers",
    "overlap_stats",
    "pead_benchmark",
]
```

이 모듈은 `math` 만 쓴다. `numpy` 를 임포트하지 마라 — ruff 가 `F401` 로 잡는다.
(테스트 쪽은 `numpy` 를 쓰므로 거기서는 임포트한다.)

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_validate_pead.py -v`
Expected: 9 passed

- [ ] **Step 5: 린트**

Run: `uv run ruff check src/swing_it/validate/pead.py`
Expected: `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/validate/pead.py tests/test_validate_pead.py
git commit -m "feat(validate): 겹침 보정 Sharpe·t 와 PEAD 대비 판정"
```

---

### Task 5: 피처 기여도 · 유사 과거 사례 · 근거 카드

**Files:**
- Create: `src/swing_it/report/__init__.py`
- Create: `src/swing_it/report/cards.py`
- Test: `tests/test_report_cards.py`

**Interfaces:**
- Consumes:
  - `swing_it.model.layered.fit_final_model(ds, feature_columns, *, seed=0) -> tuple[GradientBoostingRegressor, pd.Series]`
  - `articles` 프레임 `[code, date, title, outlet, published_at, url]` (계획 2)
- Produces:
  - `feature_contributions(model, row: pd.Series, baseline: pd.Series, feature_columns: Sequence[str]) -> pd.Series` — 피처별 기여도, 내림차순 정렬
  - `find_analogs(train: pd.DataFrame, row: pd.Series, feature_columns: Sequence[str], *, k: int = 3) -> pd.DataFrame` — 컬럼 `[code, date, excess_return, distance]`
  - `render_card(row: pd.Series, *, rank: int, contributions: pd.Series, analogs: pd.DataFrame, articles: pd.DataFrame, gate_verdict: str, layer_label: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp, names: dict[str, str] | None = None, top_features: int = 5) -> str` — 마크다운 문자열
  - `render_candidates(candidates, *, model, baseline, train, feature_columns, articles, gate_verdict, layer_label, entry_dates, exit_dates, names=None) -> str` — 후보 전체 마크다운

**기여도 정의:** 피처 `j` 를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량, 즉 `f(x) - f(x | x_j := median_j)`. 상호작용 때문에 합이 총 예측과 정확히 일치하지 않는다. **이 사실을 카드에 각주로 찍는다.** 정확한 분해가 필요하면 SHAP 이 필요하지만, 하루 10건을 사람이 읽고 "왜 걸렸나"를 보는 용도에는 순위만 맞으면 충분하고, 의존성을 하나 덜 늘린다.

**유사 사례:** 학습셋 표준편차로 표준화한 피처 공간의 유클리드 거리 최근접 `k` 건. 그 사례들의 **실현** 초과수익을 보여준다 — 예측이 아니라.

**FAIL 표시 강제:** 카드 맨 아래에 그 층의 게이트 판정을 반드시 찍는다. `FAIL` 인 모델의 카드가 추천처럼 읽히면 안 된다. 이건 테스트로 강제한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_report_cards.py`:

```python
import numpy as np
import pandas as pd

from swing_it.model.layered import fit_final_model
from swing_it.report.cards import (
    feature_contributions,
    find_analogs,
    render_card,
)

COLS = ["tv_ratio", "foreign_net_ratio", "llm_specificity"]


def _train(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in COLS})
    df["code"] = [f"{i % 20:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    df["excess_return"] = df["llm_specificity"] * 0.08 + rng.normal(scale=0.01, size=n)
    return df


def _candidate() -> pd.Series:
    return pd.Series(
        {
            "code": "034020",
            "date": pd.Timestamp("2026-08-14"),
            "tv_ratio": 4.6,
            "foreign_net_ratio": 0.083,
            "llm_specificity": 2.5,
            "prediction": 0.0481,
        }
    )


def _articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["034020", "034020", "005930"],
            "date": pd.to_datetime(["2026-08-14", "2026-08-14", "2026-08-14"]),
            "title": ["체코 원전 2기 최종 계약", "원전 수주 잔고 사상 최대", "무관한 기사"],
            "outlet": ["한국경제", "더벨", "한국경제"],
            "published_at": pd.to_datetime(
                ["2026-08-14 09:12", "2026-08-14 11:40", "2026-08-14 10:00"]
            ),
            "url": ["https://x/1", "https://x/2", "https://x/3"],
        }
    )


def test_contribution_is_zero_for_a_feature_already_at_baseline():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate().copy()
    row["llm_specificity"] = baseline["llm_specificity"]
    contrib = feature_contributions(model, row, baseline, COLS)
    assert np.isclose(contrib["llm_specificity"], 0.0)


def test_the_driving_feature_ranks_first():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert contrib.index[0] == "llm_specificity"
    assert contrib.iloc[0] > 0


def test_contributions_are_sorted_descending():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    contrib = feature_contributions(model, _candidate(), baseline, COLS)
    assert list(contrib) == sorted(contrib, reverse=True)


def test_analogs_return_k_nearest_with_realised_returns():
    train = _train()
    analogs = find_analogs(train, _candidate(), COLS, k=3)
    assert list(analogs.columns) == ["code", "date", "excess_return", "distance"]
    assert len(analogs) == 3
    assert analogs["distance"].is_monotonic_increasing


def test_analogs_are_actually_close_in_feature_space():
    train = _train()
    row = train.iloc[7].copy()
    analogs = find_analogs(train, row, COLS, k=1)
    assert analogs.iloc[0]["distance"] < 1e-9  # 자기 자신이 최근접


def test_card_contains_the_essentials():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    card = render_card(
        row,
        rank=3,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="FAIL",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
        names={"034020": "두산에너빌리티"},
    )
    assert "034020 두산에너빌리티" in card
    assert "2026-08-17" in card  # 진입일
    assert "2026-10-13" in card  # 목표 청산일
    assert "llm_specificity" in card
    assert "체코 원전 2기 최종 계약" in card
    assert "유사 과거 사례" in card


def test_card_never_hides_a_failed_gate():
    """스펙 §10 — FAIL 을 숨기지 않는다. FAIL 모델의 카드는 추천이 아니다."""
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="FAIL",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "FAIL" in card
    assert "매매 신호가 아니다" in card


def test_card_only_shows_articles_for_this_code_and_date():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles(),
        gate_verdict="PASS",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "무관한 기사" not in card
    assert "2건 · 매체 2곳" in card


def test_card_works_without_any_articles():
    train = _train()
    model, baseline = fit_final_model(train, COLS)
    row = _candidate()
    card = render_card(
        row,
        rank=1,
        contributions=feature_contributions(model, row, baseline, COLS),
        analogs=find_analogs(train, row, COLS, k=3),
        articles=_articles().iloc[0:0],
        gate_verdict="PASS",
        layer_label="③ + LLM",
        entry_date=pd.Timestamp("2026-08-17"),
        exit_date=pd.Timestamp("2026-10-13"),
    )
    assert "근거 기사 없음" in card
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_report_cards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report'`

- [ ] **Step 3: 구현**

`src/swing_it/report/__init__.py`:

```python
"""리포트 계층 — 사람이 읽는 출력.

스펙 §1: 최종 매매 판단은 사람이 한다. 그래서 이 계층의 목적은 설득이 아니라
**근거 제시**다. 모델이 게이트를 통과하지 못했으면 그 사실도 같이 제시한다.

하루 10건 규모를 전제로 한다 (스펙 §5).
"""
```

`src/swing_it/report/cards.py`:

```python
"""근거 카드 — 왜 걸렸는지, 얼마나 들 건지, 과거에 비슷한 게 어땠는지.

기여도는 '피처를 학습셋 중앙값으로 되돌리면 예측이 얼마나 줄어드는가'다.
상호작용 때문에 합이 총 예측과 정확히 맞지는 않는다. 카드에 그렇게 적는다.
정확한 가법 분해가 필요하면 SHAP 이 필요하지만, 하루 10건을 사람이 훑고
'무엇이 이걸 밀어올렸나'를 보는 데는 순위면 충분하다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DISCLAIMER = (
    "이 카드는 근거 제시이지 매매 신호가 아니다. 진입·청산·비중은 사람이 정한다."
)


def feature_contributions(
    model,
    row: pd.Series,
    baseline: pd.Series,
    feature_columns: Sequence[str],
) -> pd.Series:
    """피처를 하나씩 중앙값으로 되돌렸을 때의 예측 감소량. 내림차순."""
    columns = list(feature_columns)
    x = pd.DataFrame([row[columns].astype(float).to_dict()], columns=columns)
    full = float(model.predict(x)[0])

    out = {}
    for col in columns:
        alt = x.copy()
        alt[col] = float(baseline[col])
        out[col] = full - float(model.predict(alt)[0])
    return pd.Series(out).sort_values(ascending=False)


def find_analogs(
    train: pd.DataFrame,
    row: pd.Series,
    feature_columns: Sequence[str],
    *,
    k: int = 3,
) -> pd.DataFrame:
    """표준화 피처 공간에서 가장 가까운 과거 이벤트 ``k`` 건과 그 **실현** 초과수익."""
    columns = list(feature_columns)
    scale = train[columns].std(ddof=0).replace(0.0, 1.0)
    target = (row[columns].astype(float) - train[columns].mean()) / scale
    scaled = (train[columns] - train[columns].mean()) / scale

    distance = np.sqrt(((scaled - target) ** 2).sum(axis=1))
    out = train[["code", "date", "excess_return"]].copy()
    out["distance"] = distance.to_numpy()
    return out.nsmallest(k, "distance").reset_index(drop=True)


def _format_articles(articles: pd.DataFrame, code: str, date: pd.Timestamp) -> str:
    if articles is None or articles.empty:
        return "**근거 기사 없음** — 수급·유동성 피처만으로 걸린 후보다.\n"
    hit = articles[(articles["code"] == code) & (articles["date"] == date)]
    if hit.empty:
        return "**근거 기사 없음** — 수급·유동성 피처만으로 걸린 후보다.\n"

    lines = [f"**근거 기사 {len(hit)}건 · 매체 {hit['outlet'].nunique()}곳**", ""]
    for _, art in hit.sort_values("published_at").iterrows():
        stamp = pd.Timestamp(art["published_at"]).strftime("%m-%d %H:%M")
        lines.append(f"- [{art['outlet']}] [{art['title']}]({art['url']}) — {stamp}")
    return "\n".join(lines) + "\n"


def render_card(
    row: pd.Series,
    *,
    rank: int,
    contributions: pd.Series,
    analogs: pd.DataFrame,
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    names: dict[str, str] | None = None,
    top_features: int = 5,
) -> str:
    """후보 한 건의 근거 카드 마크다운."""
    code = str(row["code"])
    name = (names or {}).get(code, "")
    heading = f"{code} {name}".strip()
    event_date = pd.Timestamp(row["date"])

    head = [
        f"### {rank}. {heading} — 예측 초과수익 {row['prediction']:+.2%}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 층 | {layer_label} |",
        f"| 이벤트일 | {event_date.date()} |",
        f"| 진입 | {pd.Timestamp(entry_date).date()} 시가 |",
        f"| 목표 청산 | {pd.Timestamp(exit_date).date()} 종가 (D+40, 중간 청산 없음) |",
        "",
    ]

    top = contributions.head(top_features)
    contrib = ["**왜 걸렸나 — 예측 기여도 상위 " + str(len(top)) + "**", "",
               "| 피처 | 값 | 기여 |", "|---|---|---|"]
    for feature, value in top.items():
        contrib.append(f"| `{feature}` | {float(row[feature]):.3f} | {value:+.4f} |")
    contrib += [
        "",
        "> 기여도 = 해당 피처를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량.",
        "> 피처 간 상호작용이 있어 합이 총 예측과 정확히 일치하지는 않는다. 순위로 읽어라.",
        "",
    ]

    body = [_format_articles(articles, code, event_date), ""]

    analog_lines = [f"**유사 과거 사례 (피처 공간 최근접 {len(analogs)}건, 실현 수익)**", "",
                    "| 종목 | 이벤트일 | 실현 초과수익 | 거리 |", "|---|---|---|---|"]
    for _, a in analogs.iterrows():
        analog_lines.append(
            f"| {a['code']} | {pd.Timestamp(a['date']).date()} | "
            f"{a['excess_return']:+.2%} | {a['distance']:.2f} |"
        )

    tail = [
        "",
        f"**게이트 판정: {gate_verdict}** ({layer_label} 층). {DISCLAIMER}",
        "",
    ]
    if gate_verdict != "PASS":
        tail.insert(
            1,
            "> 이 층은 음성대조 널을 이기지 못했다. 아래 후보는 모델이 뽑았다는 사실 외에 "
            "어떤 통계적 근거도 없다.",
        )

    return "\n".join([*head, *contrib, *body, *analog_lines, *tail])


def render_candidates(
    candidates: pd.DataFrame,
    *,
    model,
    baseline: pd.Series,
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    articles: pd.DataFrame,
    gate_verdict: str,
    layer_label: str,
    entry_dates: dict[str, pd.Timestamp],
    exit_dates: dict[str, pd.Timestamp],
    names: dict[str, str] | None = None,
) -> str:
    """후보 전체를 예측 내림차순으로 렌더한다."""
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    cards = []
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        cards.append(
            render_card(
                row,
                rank=rank,
                contributions=feature_contributions(model, row, baseline, feature_columns),
                analogs=find_analogs(train, row, feature_columns, k=3),
                articles=articles,
                gate_verdict=gate_verdict,
                layer_label=layer_label,
                entry_date=entry_dates[str(row["code"])],
                exit_date=exit_dates[str(row["code"])],
                names=names,
            )
        )
    return "\n---\n\n".join(cards)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_report_cards.py -v`
Expected: 9 passed

- [ ] **Step 5: 렌더 결과 눈으로 확인**

Run:
```bash
uv run python - <<'PY'
import pandas as pd
from tests.test_report_cards import _articles, _candidate, _train, COLS
from swing_it.model.layered import fit_final_model
from swing_it.report.cards import feature_contributions, find_analogs, render_card

train = _train()
model, baseline = fit_final_model(train, COLS)
row = _candidate()
print(render_card(
    row, rank=3,
    contributions=feature_contributions(model, row, baseline, COLS),
    analogs=find_analogs(train, row, COLS, k=3),
    articles=_articles(), gate_verdict="FAIL", layer_label="③ + LLM",
    entry_date=pd.Timestamp("2026-08-17"), exit_date=pd.Timestamp("2026-10-13"),
    names={"034020": "두산에너빌리티"},
))
PY
```

Expected: 아래와 같은 마크다운이 나온다 (숫자는 난수 시드에 따라 달라진다).

````markdown
### 3. 034020 두산에너빌리티 — 예측 초과수익 +4.81%

| 항목 | 값 |
|---|---|
| 층 | ③ + LLM |
| 이벤트일 | 2026-08-14 |
| 진입 | 2026-08-17 시가 |
| 목표 청산 | 2026-10-13 종가 (D+40, 중간 청산 없음) |

**왜 걸렸나 — 예측 기여도 상위 3**

| 피처 | 값 | 기여 |
|---|---|---|
| `llm_specificity` | 2.500 | +0.1893 |
| `tv_ratio` | 4.600 | +0.0041 |
| `foreign_net_ratio` | 0.083 | -0.0007 |

> 기여도 = 해당 피처를 학습셋 중앙값으로 되돌렸을 때의 예측 감소량.
> 피처 간 상호작용이 있어 합이 총 예측과 정확히 일치하지는 않는다. 순위로 읽어라.

**근거 기사 2건 · 매체 2곳**

- [한국경제] [체코 원전 2기 최종 계약](https://x/1) — 08-14 09:12
- [더벨] [원전 수주 잔고 사상 최대](https://x/2) — 08-14 11:40

**유사 과거 사례 (피처 공간 최근접 3건, 실현 수익)**

| 종목 | 이벤트일 | 실현 초과수익 | 거리 |
|---|---|---|---|
| 000013 | 2024-01-14 | +18.42% | 0.61 |
| 000007 | 2024-02-21 | +15.03% | 0.74 |
| 000002 | 2024-03-08 | +21.10% | 0.88 |

> 이 층은 음성대조 널을 이기지 못했다. 아래 후보는 모델이 뽑았다는 사실 외에 어떤 통계적 근거도 없다.

**게이트 판정: FAIL** (③ + LLM 층). 이 카드는 근거 제시이지 매매 신호가 아니다. 진입·청산·비중은 사람이 정한다.
````

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/report tests/test_report_cards.py
git commit -m "feat(report): 피처 기여도·유사 사례·근거 카드 마크다운"
```

---

### Task 6: 전진 기록 (forward log)

**Files:**
- Create: `src/swing_it/report/forward_log.py`
- Test: `tests/test_forward_log.py`

**Interfaces:**
- Consumes: `swing_it.layers.build_layered_dataset(..., require_label=False)` 결과 행, `swing_it.labels.excess_return` 산출 라벨 프레임
- Produces:
  - `LOG_KEY: tuple[str, ...]` — `("logged_at", "layer", "code", "event_date")`
  - `build_records(candidates: pd.DataFrame, *, feature_columns: Sequence[str], layer: str, gate_verdict: str, logged_at: str, entry_dates: dict[str, str], exit_dates: dict[str, str], seed: int = 0, version: str | None = None) -> list[dict]`
  - `append_forward_log(path: str | Path, records: Sequence[dict]) -> int` — JSONL append-only. 이미 있는 키는 **건너뛴다**(덮어쓰지 않는다). 새로 쓴 건수를 반환.
  - `read_forward_log(path: str | Path) -> pd.DataFrame` — 빈 파일/없는 파일이면 빈 프레임
  - `score_forward_log(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame` — `labels` 는 `[code, date, excess_return]`. 컬럼 `[*log, realized_excess]`. 아직 결과가 안 나온 행은 `NaN`

**append-only 인 이유:** 전진검증의 가치는 "그때 그렇게 예측했다"는 사실이 사후에 바뀌지 않는 데 있다. 같은 날 같은 종목을 다시 기록하려 하면 조용히 건너뛴다. 모델을 고쳐 다시 돌려도 과거 기록은 남는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_forward_log.py`:

```python
import json

import numpy as np
import pandas as pd

from swing_it.report.forward_log import (
    append_forward_log,
    build_records,
    read_forward_log,
    score_forward_log,
)

COLS = ["tv_ratio", "llm_specificity"]


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["034020", "005930"],
            "date": pd.to_datetime(["2026-08-14", "2026-08-14"]),
            "tv_ratio": [4.6, 3.2],
            "llm_specificity": [2.5, 0.4],
            "prediction": [0.0481, 0.0192],
        }
    )


def _records() -> list[dict]:
    return build_records(
        _candidates(),
        feature_columns=COLS,
        layer="supply_news_llm",
        gate_verdict="FAIL",
        logged_at="2026-08-17",
        entry_dates={"034020": "2026-08-17", "005930": "2026-08-17"},
        exit_dates={"034020": "2026-10-13", "005930": "2026-10-13"},
    )


def test_record_captures_features_and_context():
    rec = _records()[0]
    assert rec["code"] == "034020"
    assert rec["event_date"] == "2026-08-14"
    assert rec["logged_at"] == "2026-08-17"
    assert rec["layer"] == "supply_news_llm"
    assert rec["gate_verdict"] == "FAIL"
    assert rec["rank"] == 1
    assert rec["features"] == {"tv_ratio": 4.6, "llm_specificity": 2.5}
    assert rec["entry_date_planned"] == "2026-08-17"
    assert rec["exit_date_planned"] == "2026-10-13"


def test_records_are_ranked_by_prediction():
    ranks = {r["code"]: r["rank"] for r in _records()}
    assert ranks == {"034020": 1, "005930": 2}


def test_append_writes_jsonl(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    assert append_forward_log(path, _records()) == 2
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["code"] == "034020"


def test_rerunning_the_same_day_writes_nothing(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    assert append_forward_log(path, _records()) == 0
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_existing_record_is_never_overwritten(tmp_path):
    """전진검증의 가치는 '그때 그렇게 예측했다'가 사후에 안 바뀌는 데 있다."""
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    tampered = _records()
    tampered[0]["prediction"] = 0.9999
    append_forward_log(path, tampered)
    log = read_forward_log(path)
    assert log[log["code"] == "034020"]["prediction"].tolist() == [0.0481]


def test_a_later_day_appends_a_new_row(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    later = build_records(
        _candidates(),
        feature_columns=COLS,
        layer="supply_news_llm",
        gate_verdict="FAIL",
        logged_at="2026-08-18",
        entry_dates={"034020": "2026-08-18", "005930": "2026-08-18"},
        exit_dates={"034020": "2026-10-14", "005930": "2026-10-14"},
    )
    assert append_forward_log(path, later) == 2
    assert len(read_forward_log(path)) == 4


def test_read_missing_file_returns_empty_frame(tmp_path):
    assert read_forward_log(tmp_path / "nope.jsonl").empty


def test_score_fills_realised_returns_where_available(tmp_path):
    path = tmp_path / "forward_log.jsonl"
    append_forward_log(path, _records())
    labels = pd.DataFrame(
        {
            "code": ["034020"],
            "date": pd.to_datetime(["2026-08-14"]),
            "excess_return": [0.062],
        }
    )
    scored = score_forward_log(read_forward_log(path), labels)
    assert scored.set_index("code").loc["034020", "realized_excess"] == 0.062
    assert np.isnan(scored.set_index("code").loc["005930", "realized_excess"])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_forward_log.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report.forward_log'`

- [ ] **Step 3: 구현**

`src/swing_it/report/forward_log.py`:

```python
"""전진 기록 — 오늘의 후보와 그때의 피처를 영구 저장한다 (스펙 §9).

6개월 뒤 실전 검증을 가능하게 하는 유일한 장치다. walk-forward 는 과거
데이터를 시간순으로 잘라 쓴 것이라 '손 안 댄 창'이라고 부르기 어렵다.
오늘 기록해 6개월 뒤 채점하는 것만이 진짜 out-of-sample 이다.

**append-only.** 같은 (logged_at, layer, code, event_date) 는 조용히 건너뛴다.
모델을 고쳐 다시 돌려도 과거 예측은 남는다. 사후에 바뀔 수 있는 기록은
전진검증 자산이 아니다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

LOG_KEY: tuple[str, ...] = ("logged_at", "layer", "code", "event_date")


def build_records(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    layer: str,
    gate_verdict: str,
    logged_at: str,
    entry_dates: dict[str, str],
    exit_dates: dict[str, str],
    seed: int = 0,
    version: str | None = None,
) -> list[dict]:
    """후보 프레임을 전진 기록 레코드 목록으로 바꾼다."""
    from .. import __version__

    columns = list(feature_columns)
    ordered = candidates.sort_values("prediction", ascending=False).reset_index(drop=True)
    records = []
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        code = str(row["code"])
        records.append(
            {
                "logged_at": logged_at,
                "layer": layer,
                "code": code,
                "event_date": str(pd.Timestamp(row["date"]).date()),
                "rank": rank,
                "prediction": float(row["prediction"]),
                "gate_verdict": gate_verdict,
                "entry_date_planned": entry_dates[code],
                "exit_date_planned": exit_dates[code],
                "features": {col: float(row[col]) for col in columns},
                "model_seed": int(seed),
                "code_version": version or __version__,
            }
        )
    return records


def _key(record: dict) -> tuple:
    return tuple(record[field] for field in LOG_KEY)


def append_forward_log(path: str | Path, records: Sequence[dict]) -> int:
    """중복을 건너뛰고 JSONL 에 덧붙인다. 새로 쓴 건수를 돌려준다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple] = set()
    if target.exists():
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(_key(json.loads(line)))

    written = 0
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            key = _key(record)
            if key in seen:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1
    return written


def read_forward_log(path: str | Path) -> pd.DataFrame:
    """전진 기록을 프레임으로 읽는다. 없으면 빈 프레임."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def score_forward_log(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """기록된 후보에 실현 초과수익을 붙인다. 아직 안 나온 건 ``NaN``.

    Args:
        labels: ``[code, date, excess_return]``. ``date`` 는 이벤트일.
    """
    if log.empty:
        return log
    out = log.copy()
    out["_event_date"] = pd.to_datetime(out["event_date"])
    realized = labels[["code", "date", "excess_return"]].rename(
        columns={"date": "_event_date", "excess_return": "realized_excess"}
    )
    realized["_event_date"] = pd.to_datetime(realized["_event_date"])
    out = out.merge(realized, on=["code", "_event_date"], how="left")
    return out.drop(columns=["_event_date"])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_forward_log.py -v`
Expected: 8 passed

- [ ] **Step 5: `.gitignore` 에 전진 기록 경로가 이미 차단돼 있는지 확인**

Run: `grep -E '^(data/|results/)' .gitignore`
Expected: `data/` 와 `results/` 두 줄이 나온다. 안 나오면 그 줄을 추가하고 함께 커밋한다 — 전진 기록은 종목·예측이 담긴 데이터라 절대 커밋하지 않는다(스펙 §8).

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/report/forward_log.py tests/test_forward_log.py
git commit -m "feat(report): append-only 전진 기록과 사후 채점"
```

---

### Task 7: 층 비교 리포트와 CLI 2종

**Files:**
- Create: `src/swing_it/report/layers_md.py`
- Create: `src/swing_it/cli_layers.py`
- Create: `src/swing_it/cli_today.py`
- Modify: `pyproject.toml` — `[project.scripts]` 에 두 줄 추가
- Test: `tests/test_report_layers_md.py`

**Interfaces:**
- Consumes:
  - `swing_it.validate.compare.compare_layers(...) -> dict` (키 `rows, deltas, llm, sample`)
  - `swing_it.validate.pead.benchmark_all_layers(curves, *, hold_days=40) -> list[dict]`
  - `swing_it.adapters.db.connect(dsn=None)`, `load_prices(conn, start, end)`, `load_supply_demand(conn, start, end)`
  - `swing_it.layers.build_layered_dataset(...)`, `layer_features(...)`, `LAYER_LABELS`
  - `swing_it.model.layered.fit_final_model(...)`
  - `swing_it.report.cards.render_candidates(...)`
  - `swing_it.report.forward_log.build_records(...)`, `append_forward_log(...)`
  - 계획 2·3 의 피처 프레임 로더 (CLI 안에서만 import)
- Produces:
  - `render_layer_report(comparison: dict, pead_rows: list[dict], *, baseline_reference: dict | None = None) -> str` — 마크다운 문자열
  - `swing_it.cli_layers.main(argv: list[str] | None = None) -> int`
  - `swing_it.cli_today.main(argv: list[str] | None = None) -> int`

`swing-layers` 는 **판정 결과와 무관하게 종료 코드 0** 을 낸다. FAIL 은 오류가 아니라 결과다(스펙 §10). 종료 코드로 FAIL 을 표현하면 CI 나 래퍼가 "실패했으니 다시 돌리자"는 유혹에 빠진다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_report_layers_md.py`:

```python
from swing_it.report.layers_md import render_layer_report


def _comparison(llm_verdict: str) -> dict:
    return {
        "rows": [
            {"layer": "supply", "label": "① 수급 단독", "n_features": 7,
             "strategy_mean": 0.0031, "null_mean": 0.0002, "null_p95": 0.0044,
             "percentile": 71.0, "net_mean": -0.0015, "verdict": "FAIL"},
            {"layer": "supply_news", "label": "② + 규칙 뉴스", "n_features": 10,
             "strategy_mean": 0.0088, "null_mean": 0.0002, "null_p95": 0.0044,
             "percentile": 98.5, "net_mean": 0.0042, "verdict": "PASS"},
            {"layer": "supply_news_llm", "label": "③ + LLM", "n_features": 16,
             "strategy_mean": 0.0091, "null_mean": 0.0002, "null_p95": 0.0044,
             "percentile": 98.0, "net_mean": 0.0045, "verdict": "PASS"},
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
            "layer3_gate": "PASS", "verdict": llm_verdict,
        },
        "sample": {
            "n_rows": 8421, "n_codes": 913, "date_range": ["2019-03-04", "2026-06-20"],
            "n_folds": 4, "top_n": 15, "n_trials": 200, "n_boot": 2000,
            "cost_one_way": 0.0023, "seed": 0,
        },
    }


def _pead() -> list[dict]:
    return [
        {"layer": "supply", "mean": 0.0031, "sd": 0.061, "n_obs": 412, "n_effective": 10,
         "sharpe_annual": 0.13, "t_stat": 0.16, "pead_sharpe": 0.80, "pead_t": 2.20,
         "verdict": "LOSES_TO_PEAD"},
        {"layer": "supply_news", "mean": 0.0088, "sd": 0.058, "n_obs": 412, "n_effective": 10,
         "sharpe_annual": 0.38, "t_stat": 0.48, "pead_sharpe": 0.80, "pead_t": 2.20,
         "verdict": "LOSES_TO_PEAD"},
        {"layer": "supply_news_llm", "mean": 0.0091, "sd": 0.058, "n_obs": 412,
         "n_effective": 10, "sharpe_annual": 0.39, "t_stat": 0.50, "pead_sharpe": 0.80,
         "pead_t": 2.20, "verdict": "LOSES_TO_PEAD"},
    ]


def test_report_states_the_llm_verdict_up_front():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"), _pead())
    headline = md.split("\n\n")[0:4]
    assert any("LLM_NOT_JUSTIFIED" in part for part in headline)


def test_report_lists_every_layer_with_its_gate_verdict():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"), _pead())
    for label in ["① 수급 단독", "② + 규칙 뉴스", "③ + LLM"]:
        assert label in md
    assert "FAIL" in md
    assert "PASS" in md


def test_report_shows_the_delta_confidence_interval():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"), _pead())
    assert "-0.0019" in md or "-0.19" in md  # ③−② CI 하한이 어떤 형식으로든 찍혀야 한다


def test_report_includes_pead_comparison():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"), _pead())
    assert "PEAD" in md
    assert "LOSES_TO_PEAD" in md
    assert "0.80" in md


def test_report_records_preregistered_parameters():
    md = render_layer_report(_comparison("LLM_NOT_JUSTIFIED"), _pead())
    for token in ["top_n", "15", "n_trials", "200", "0.0023"]:
        assert token in md


def test_report_prints_the_plan1_baseline_when_given():
    md = render_layer_report(
        _comparison("LLM_NOT_JUSTIFIED"),
        _pead(),
        baseline_reference={"strategy_mean": 0.0019, "verdict": "FAIL", "n_events": 24310},
    )
    assert "계획 1 베이스라인" in md
    assert "24,310" in md
    assert "표본이 다르므로" in md


def test_report_does_not_soften_a_justified_verdict():
    md = render_layer_report(_comparison("LLM_JUSTIFIED"), _pead())
    assert "LLM_JUSTIFIED" in md
    assert "LLM_NOT_JUSTIFIED" not in md
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_report_layers_md.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.report.layers_md'`

- [ ] **Step 3: 리포트 렌더러 구현**

`src/swing_it/report/layers_md.py`:

```python
"""층 비교 결과 마크다운 리포트.

판정을 맨 위에 둔다. FAIL 을 뒤쪽 표 안에 묻어두면 사람은 앞의 숫자만 보고
좋은 결과라고 읽는다. 스펙 §10 — 부정 결과도 성공이고, 성공은 명확히 보여야 한다.
"""

from __future__ import annotations

_LLM_MESSAGE = {
    "LLM_JUSTIFIED": (
        "③층이 ②층을 유의하게 이겼고 ③층 자체도 음성대조를 통과했다. "
        "LLM 피처 추출 비용을 계속 지불할 근거가 있다."
    ),
    "LLM_NOT_JUSTIFIED": (
        "③층이 ②층을 유의하게 이기지 못했다(또는 ③층이 음성대조를 못 넘었다). "
        "**LLM 을 쓸 이유가 없다.** 규칙 기반 뉴스 피처까지가 이 데이터에서 나오는 전부다."
    ),
}


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def render_layer_report(
    comparison: dict,
    pead_rows: list[dict],
    *,
    baseline_reference: dict | None = None,
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
        f"③ − ② = {llm['delta']:+.4f} "
        f"(95% CI {llm['ci_low']:+.4f} ~ {llm['ci_high']:+.4f}, "
        f"P(차이>0) = {llm['p_greater']:.3f}, ③층 게이트 {llm['layer3_gate']})",
        "",
        "## 표본",
        "",
        f"- 이벤트 {sample['n_rows']:,}건 · 종목 {sample['n_codes']:,}개 · "
        f"{sample['date_range'][0]} ~ {sample['date_range'][1]}",
        f"- 세 층 모두 **같은 행**을 본다. 한 층에서라도 피처가 결측인 이벤트는 전부 제외됐다.",
        "",
        "### 사전등록 파라미터",
        "",
        f"`top_n={sample['top_n']}` · `n_folds={sample['n_folds']}` · "
        f"`n_trials={sample['n_trials']}` · `n_boot={sample['n_boot']}` · "
        f"`cost_one_way={sample['cost_one_way']}` · `seed={sample['seed']}`",
        "",
        "## 층별 성과 (동일 폴드 · 동일 널)",
        "",
        "| 층 | 피처 수 | 전략 평균 | 널 평균 | 널 p95 | 백분위 | 비용차감 후 | 게이트 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {row['n_features']} | {_pct(row['strategy_mean'])} | "
            f"{_pct(row['null_mean'])} | {_pct(row['null_p95'])} | "
            f"{row['percentile']:.1f} | {_pct(row['net_mean'])} | **{row['verdict']}** |"
        )

    out += [
        "",
        "> 널 평균·널 p95 가 세 층에서 같은 것이 정상이다. 같은 표본 위에서 같은 시드로 "
        "만든 같은 널이기 때문이다. 다르다면 표본이 어긋난 것이고 비교는 무효다.",
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

    out += [
        "",
        "## PEAD 대비 (스펙 §7)",
        "",
        f"기준: kr-quant 검증 PEAD — 연율 Sharpe {pead_rows[0]['pead_sharpe']:.2f}, "
        f"t {pead_rows[0]['pead_t']:.2f}",
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
        "> t 는 겹침 보정본이다. 보유 40거래일이 겹치므로 날짜별 관측은 독립이 아니고, "
        "보정 없이는 t 가 크게 부풀려진다. 유효 표본 = 겹치지 않는 블록 수.",
        "",
    ]

    if baseline_reference:
        out += [
            "## 계획 1 베이스라인 참고값",
            "",
            f"- 전략 평균 {_pct(baseline_reference['strategy_mean'])} · "
            f"게이트 {baseline_reference['verdict']} · "
            f"이벤트 {baseline_reference['n_events']:,}건",
            "",
            "> **표본이 다르므로 위 층별 표와 직접 비교하지 말 것.** 계획 1 은 기사 유무와 "
            "무관한 모든 거래대금 급증 이벤트를 쓰고, 여기 ①층은 세 층 피처가 모두 있는 "
            "이벤트만 쓴다.",
            "",
        ]

    return "\n".join(out)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_report_layers_md.py -v`
Expected: 7 passed

- [ ] **Step 5: `swing-layers` CLI 구현**

`src/swing_it/cli_layers.py`:

```python
"""`swing-layers` — 세 층을 동일 조건에서 비교하고 판정을 출력한다.

**FAIL 이어도 종료 코드는 0 이다.** FAIL 은 오류가 아니라 결과다(스펙 §10).
종료 코드로 FAIL 을 표현하면 래퍼나 CI 가 "실패했으니 파라미터 바꿔 재시도"
경로를 만들게 되고, 그게 정확히 이 프로젝트가 두 번 당한 실패 방식이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.db import connect, load_prices, load_supply_demand
from .layers import layer_features
from .model.baseline import portfolio_returns
from .model.layered import fold_dates, walk_forward_predict_features
from .report.layers_md import render_layer_report
from .validate.compare import compare_layers
from .validate.pead import benchmark_all_layers


def _load_side_frames(args) -> tuple:
    """계획 2·3 의 산출물을 읽는다. 이 함수가 두 계획에 대한 유일한 접점이다."""
    from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
    from .features.news import NEWS_FEATURE_COLUMNS, load_news_features

    news = load_news_features(args.news_path)
    llm = load_llm_features(args.llm_path)
    return news, list(NEWS_FEATURE_COLUMNS), llm, list(LLM_FEATURE_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="층별 기여도 분리 판정")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--news-path", default="data/news_features.parquet")
    parser.add_argument("--llm-path", default="data/llm_features.parquet")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--cost-one-way", type=float, default=0.0023)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-json", default="results/baseline.json")
    parser.add_argument("--out-json", default="results/layers.json")
    parser.add_argument("--out-md", default="results/layers.md")
    args = parser.parse_args(argv)

    if args.top_n < 15:
        parser.error("top_n 하한은 15 다 (스펙 §2). 집중은 이미 죽은 영역이다.")

    from .layers import build_layered_dataset

    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    news, news_columns, llm, llm_columns = _load_side_frames(args)
    print(f"가격 {len(prices):,} · 수급 {len(supply):,} · 뉴스 {len(news):,} · "
          f"LLM {len(llm):,}", file=sys.stderr)

    ds = build_layered_dataset(
        prices,
        supply,
        news=news,
        news_columns=news_columns,
        llm=llm,
        llm_columns=llm_columns,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        hold_end=args.hold_end,
    )
    if ds.empty:
        print("세 층 피처가 모두 있는 이벤트가 없다. 계획 2·3 산출물을 확인하라.",
              file=sys.stderr)
        return 1
    print(f"비교 표본 {len(ds):,} 건 · 종목 {ds['code'].nunique():,}", file=sys.stderr)

    spec = layer_features(news_columns, llm_columns)
    comparison = compare_layers(
        ds,
        spec,
        top_n=args.top_n,
        n_folds=args.folds,
        n_trials=args.trials,
        n_boot=args.boot,
        cost_one_way=args.cost_one_way,
        seed=args.seed,
    )

    folds = fold_dates(ds, n_folds=args.folds)
    curves = {
        layer: portfolio_returns(
            walk_forward_predict_features(ds, columns, folds=folds, seed=args.seed),
            top_n=args.top_n,
        )
        for layer, columns in spec.items()
    }
    pead_rows = benchmark_all_layers(curves, hold_days=args.hold_end)

    baseline_reference = None
    baseline_path = Path(args.baseline_json)
    if baseline_path.exists():
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_reference = {
            "strategy_mean": raw["strategy_mean"],
            "verdict": raw["verdict"],
            "n_events": raw.get("n_events", 0),
        }

    payload = {**comparison, "pead": pead_rows, "baseline_reference": baseline_reference}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    markdown = render_layer_report(comparison, pead_rows,
                                   baseline_reference=baseline_reference)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: `swing-today` CLI 구현**

`src/swing_it/cli_today.py`:

```python
"""`swing-today` — 오늘 후보의 근거 카드를 내고 전진 기록에 남긴다 (스펙 §1·§9).

층은 `--layer` 로 고른다. 기본은 ③이지만, `swing-layers` 판정이
`LLM_NOT_JUSTIFIED` 였다면 ②를 쓰는 것이 옳다. 그 선택은 사람이 한다 —
판정에 따라 자동으로 층을 바꾸면 리포트를 안 읽게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .adapters.db import connect, load_prices, load_supply_demand
from .layers import LAYER_LABELS, build_layered_dataset, layer_features
from .model.layered import fit_final_model
from .report.cards import render_candidates
from .report.forward_log import append_forward_log, build_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="오늘 후보 + 근거 카드 + 전진 기록")
    parser.add_argument("--db", default=None)
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--asof", default=None, help="기준일. 생략 시 데이터 마지막 날")
    parser.add_argument("--layer", default="supply_news_llm",
                        choices=list(LAYER_LABELS))
    parser.add_argument("--news-path", default="data/news_features.parquet")
    parser.add_argument("--llm-path", default="data/llm_features.parquet")
    parser.add_argument("--articles-path", default="data/articles.parquet")
    parser.add_argument("--layers-json", default="results/layers.json")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="data/forward_log.jsonl")
    parser.add_argument("--out-md", default=None, help="생략 시 results/today/<기준일>.md")
    args = parser.parse_args(argv)

    if args.top_n < 15:
        parser.error("top_n 하한은 15 다 (스펙 §2).")

    from .features.llm import LLM_FEATURE_COLUMNS, load_llm_features
    from .features.news import NEWS_FEATURE_COLUMNS, load_news_features
    from .news.store import load_articles

    conn = connect(args.db)
    end = args.asof or str(pd.Timestamp.today().date())
    prices = load_prices(conn, args.start, end)
    supply = load_supply_demand(conn, args.start, end)
    news_columns = list(NEWS_FEATURE_COLUMNS)
    llm_columns = list(LLM_FEATURE_COLUMNS)

    common = dict(
        news=load_news_features(args.news_path),
        news_columns=news_columns,
        llm=load_llm_features(args.llm_path),
        llm_columns=llm_columns,
        hold_end=args.hold_end,
    )
    train = build_layered_dataset(prices, supply, require_label=True, **common)
    pool = build_layered_dataset(prices, supply, require_label=False, **common)
    if train.empty or pool.empty:
        print("학습 표본 또는 후보 풀이 비었다.", file=sys.stderr)
        return 1

    asof = pd.Timestamp(args.asof) if args.asof else pool["date"].max()
    today = pool[pool["date"] == asof].copy()
    if today.empty:
        print(f"{asof.date()} 에 이벤트가 없다.", file=sys.stderr)
        return 1

    columns = layer_features(news_columns, llm_columns)[args.layer]
    model, baseline = fit_final_model(train, columns, seed=args.seed)
    today["prediction"] = model.predict(today[columns])
    candidates = today.nlargest(args.top_n, "prediction").reset_index(drop=True)

    gate_verdict = "UNKNOWN"
    layers_path = Path(args.layers_json)
    if layers_path.exists():
        payload = json.loads(layers_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            if row["layer"] == args.layer:
                gate_verdict = row["verdict"]
    else:
        print("results/layers.json 이 없다. 먼저 swing-layers 를 돌려라.", file=sys.stderr)

    # 진입은 기준일 다음 거래일 시가, 목표 청산은 hold_end 거래일 뒤.
    calendar = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
    idx = int(calendar[calendar == asof].index[0])
    entry = calendar.iloc[min(idx + 1, len(calendar) - 1)]
    exit_offset = idx + 1 + args.hold_end
    exit_date = calendar.iloc[exit_offset] if exit_offset < len(calendar) else pd.NaT
    if pd.isna(exit_date):
        # 달력에 아직 없는 미래 날짜는 영업일 근사로 표시한다
        exit_date = entry + pd.tseries.offsets.BDay(args.hold_end)

    codes = [str(c) for c in candidates["code"]]
    entry_dates = {code: entry for code in codes}
    exit_dates = {code: exit_date for code in codes}

    markdown = "\n".join(
        [
            f"# {asof.date()} 후보 — {LAYER_LABELS[args.layer]}",
            "",
            f"게이트 판정 **{gate_verdict}** · 상위 {len(candidates)}종목 · "
            f"학습 표본 {len(train):,}건",
            "",
            render_candidates(
                candidates,
                model=model,
                baseline=baseline,
                train=train,
                feature_columns=columns,
                articles=load_articles(args.articles_path),
                gate_verdict=gate_verdict,
                layer_label=LAYER_LABELS[args.layer],
                entry_dates=entry_dates,
                exit_dates=exit_dates,
            ),
        ]
    )

    out_md = Path(args.out_md or f"results/today/{asof.date()}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")

    written = append_forward_log(
        args.log_path,
        build_records(
            candidates,
            feature_columns=columns,
            layer=args.layer,
            gate_verdict=gate_verdict,
            logged_at=str(pd.Timestamp.today().date()),
            entry_dates={c: str(pd.Timestamp(entry_dates[c]).date()) for c in codes},
            exit_dates={c: str(pd.Timestamp(exit_dates[c]).date()) for c in codes},
            seed=args.seed,
        ),
    )
    print(f"{out_md} 작성 · 전진 기록 {written} 건 추가", file=sys.stderr)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: `pyproject.toml` 에 엔트리포인트 추가**

`[project.scripts]` 블록을 다음으로 만든다 (기존 `swing-baseline` 줄은 그대로 둔다):

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-layers = "swing_it.cli_layers:main"
swing-today = "swing_it.cli_today:main"
```

- [ ] **Step 8: 전체 테스트와 린트**

Run:
```bash
uv run pytest -v
uv run ruff check src tests
```
Expected: 계획 1 의 27개 + 계획 4 의 60개가 모두 통과. ruff `All checks passed!`
(계획 2·3 테스트는 각 계획 완료 후 합산된다.)

- [ ] **Step 9: 실 데이터로 층 비교 실행**

Run:
```bash
export $(grep -v '^#' ../kr-quant/.env | xargs)
uv run swing-layers --out-json results/layers.json --out-md results/layers.md
```
Expected: `results/layers.md` 가 생성되고 맨 위에 `## 판정: LLM_JUSTIFIED` 또는 `## 판정: LLM_NOT_JUSTIFIED` 가 찍힌다. 종료 코드 0.

**`LLM_NOT_JUSTIFIED` 는 버그가 아니다.** 그건 이 계획이 답하려던 질문에 대한 답이다.
그 결과가 나오면 계획 3(LLM 피처)을 운영에서 내리고 ②층으로 간다. 파라미터를 바꿔
다시 돌리지 마라 — 사전등록 config 는 이미 확정됐다(스펙 §6).

- [ ] **Step 10: 오늘 후보 생성**

Run:
```bash
uv run swing-today --layer supply_news_llm
```
Expected: `results/today/<날짜>.md` 생성, `data/forward_log.jsonl` 에 15건 추가.
같은 명령을 한 번 더 돌리면 `전진 기록 0 건 추가` 가 나와야 한다(append-only 확인).

- [ ] **Step 11: 커밋**

```bash
git add src/swing_it/report/layers_md.py src/swing_it/cli_layers.py \
  src/swing_it/cli_today.py tests/test_report_layers_md.py pyproject.toml
git commit -m "feat(report): 층 비교 리포트 + swing-layers·swing-today CLI"
```

---

## Self-Review 체크

**스펙 커버리지 (계획 4 범위):**

| 스펙 요구 | 태스크 |
|---|---|
| §1 후보 리스트 + 근거 카드 | Task 5, Task 7 (`swing-today`) |
| §1 진입·목표 보유기간 표시 | Task 5 — 카드 표의 진입/목표 청산 행 |
| §1 유사 과거 사례 | Task 5 `find_analogs` |
| §2 top_n 하한 15 | Task 3 기본값, Task 7 CLI 에서 15 미만 거부 |
| §2 손절 없음 | Task 1 — 라벨·카드 모두 중간 청산 개념이 없다 |
| §2 음성대조 필수 | Task 3 — 층마다 `gate_report` 통과 |
| §3 ①층 재사용 | Task 1 `layer_features["supply"] == FEATURE_COLUMNS` |
| §3 ②층 규칙 뉴스 | Task 1 `news_columns` 병합 |
| §3 ③층 LLM | Task 1 `llm_columns` 병합 |
| §3 ③이 ②를 못 이기면 LLM 을 쓸 이유가 없다 | Task 3 `llm_verdict` — 이 계획의 존재 이유 |
| §3 동일 조건 비교 | Task 1 (동일 표본) · Task 2 (동일 폴드) · Task 3 (동일 널, 테스트로 강제) |
| §6 얕은 모델 | Task 2 `MAX_DEPTH=2`, 테스트로 고정 |
| §6 사전등록 | 계획 상단 상수 블록, Task 7 리포트에 파라미터 기록 |
| §7 PEAD 대비 판정 | Task 4 |
| §9 전진 기록 | Task 6 |
| §10.1 층별 기여도 분리 측정 | Task 3 |
| §10.5 PEAD 대비 판정 명시 | Task 4 · Task 7 리포트 |
| §10 부정 결과도 성공 / FAIL 숨기지 않음 | Task 5 (카드에 게이트 판정 강제) · Task 7 (판정을 리포트 최상단에, 종료 코드 0) |
| §8 데이터 커밋 금지 | Task 6 Step 5 — `.gitignore` 확인 |

**계획 4 범위 밖:** 실시간 모니터링, 진입 타이밍, 주문 실행, 조건검색, 뉴스 인제스트의 quant-airflow 이전.

**타입 정합성 확인:**

- `FEATURE_COLUMNS` (계획 1) → `layer_features` → `compare_layers` 의 `spec` → `walk_forward_predict_features` 의 `feature_columns`: 전부 `list[str]`, 이름 일치.
- `portfolio_returns` 결과 `[date, n_positions, mean_excess]` → `paired_bootstrap`(`mean_excess` 사용) · `overlap_stats`(`mean_excess` 사용): 컬럼명 일치.
- `gate_report` 반환 키 `strategy_mean/null_mean/null_p95/percentile/net_mean/verdict` → `compare_layers` 의 `rows` 원소에 그대로 전개 → `render_layer_report` 가 같은 키로 읽음.
- `fit_final_model` 반환 `(model, baseline: pd.Series)` → `feature_contributions(model, row, baseline, feature_columns)` · `render_candidates(model=, baseline=)`: 순서·이름 일치.
- `build_records` 의 `entry_dates`/`exit_dates` 는 `dict[str, str]`(문자열 날짜), `render_card` 의 `entry_date`/`exit_date` 는 `pd.Timestamp`. `cli_today` 가 각각에 맞는 형을 넘긴다.

## 후속

- `swing-layers` 판정이 `LLM_NOT_JUSTIFIED` 면: 계획 3 을 운영 경로에서 내리고 ②층으로 `swing-today` 를 돌린다. 그 사실을 스펙 §10 의 성공 기준 1·5 에 대한 답으로 기록한다.
- 세 층 모두 게이트 `FAIL` 이면: "뉴스 재료 강도는 한국 시장 2~8주 지평에서 재현되는 엣지가 아니다"가 결론이다. 기록하고 끝낸다(스펙 §10). 전진 기록은 계속 쌓되 운영에는 올리지 않는다.
- 6개월 뒤(2027-02 이후): `score_forward_log` 로 전진 기록을 채점한다. 이것이 손 안 댄 유일한 창이다.
