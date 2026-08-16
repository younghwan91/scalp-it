# 계획 3: LLM 구조화 피처 추출 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 계획 2 가 적재한 기사를 provider 교체 가능한 LLM 에 태워 **고정 스키마 7필드**를 채우고, 캐시·재시도·사전 필터를 갖춘 배치로 돌려 `[article_id, material_type, llm_*]` 피처 테이블을 만든다.

**Architecture:** `score/` 는 세 겹으로 나뉜다 — (1) 네트워크를 모르는 순수 계층(스키마 파싱·프롬프트 조립·사전 필터·비교), (2) `LLMProvider` 프로토콜 뒤의 HTTP 계층, (3) 캐시와 배치 러너. 테스트는 `FakeProvider` 주입과 `httpx.MockTransport` 로만 돌며 **실제 API 를 호출하지 않는다.**

**Tech Stack:** Python ≥3.10, uv, httpx, pandas ≥2.0, sqlite3(표준 라이브러리), pytest, ruff

**Spec:** `docs/superpowers/specs/2026-08-17-swing-it-design.md`

**선행 계획:** 계획 1(`docs/superpowers/plans/2026-08-17-baseline-skeleton.md`) — `swing_it.events.extract_events`, `pyproject.toml` 을 쓴다. 계획 2(뉴스 인제스트) — `data/news.db` 를 만든다.

## Global Constraints

- **LLM 은 판정기가 아니라 추출기다.** 프롬프트에 "오를까 / 살 만한가 / 목표주가 / 예상 수익률"을 묻는 문항을 절대 넣지 않는다. 그 답을 믿는 순간 검증이 불가능해진다 (스펙 §3). 이 제약은 Task 1 에서 **테스트로 강제**한다.
- **입력은 제목 + 본문 200자 + 키워드뿐이다.** 전문을 쓰지 않는다. 빅카인즈(200자)와 krx-news-rest-api(전문)의 입력 길이가 다르면 학습·운영이 어긋난다 (스펙 §4).
- **API 키는 환경변수로만.** 코드·설정 파일·테스트 픽스처 어디에도 키 문자열을 두지 않는다. `.env` 는 이미 `.gitignore` 에 있다.
- **기본 모델은 최저가 등급**(Flash-Lite 급). 기사당 작업이 "제목+리드 읽고 고정 스키마 채우기"라 추론 난이도가 낮다 (스펙 §8).
- **단가·모델 ID 를 계획이 단정하지 않는다.** 이 문서의 기본 모델 상수는 착수 시점에 Task 7 Step 1 에서 확인하고 필요하면 갱신한다.
- **같은 기사를 두 번 태우지 않는다.** 캐시 키는 `(article_id, model, prompt_version)`. 모델명·프롬프트 버전을 결과에 함께 기록해야 재현성이 선다 (스펙 §11).
- **부분 실패는 정상이다.** 수천 건 배치에서 일부 실패로 전체를 죽이지 않는다. 실패는 기록하고 다음 실행에서 재시도한다.
- **테스트는 네트워크에 나가지 않는다.** `FakeProvider` 또는 `httpx.MockTransport` 만 쓴다. 실 API 호출은 Task 6·7 의 수동 스모크 스텝에서만, 소량으로.
- **데이터 커밋 금지.** 기사 본문·LLM 원문 응답은 `data/` 아래에만 둔다(`.gitignore` 차단됨). `results/` 에는 **집계 수치만** 쓴다 — 기사 텍스트를 넣지 않는다.
- **DB 는 읽기 전용.** 계획 2 의 `data/news.db` 에 쓰지 않는다. LLM 결과는 별도 파일 `data/llm_features.db` 에 쌓는다.

## 입력 계약 — 계획 2 산출물과 일치해야 한다

이 계획은 로컬 sqlite `data/news.db` 의 `articles` 테이블을 읽는다. **최소 요구 컬럼:**

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `article_id` | TEXT | 기사 고유 키. 캐시 PK 의 일부. **전 기간 유일해야 한다** |
| `date` | TEXT | 기사 게재일 `YYYY-MM-DD`. 이벤트일과 조인하는 키 |
| `code` | TEXT | 6자리 종목코드. 매핑 실패 시 빈 문자열 또는 NULL |
| `press` | TEXT | 매체명 |
| `title` | TEXT | 기사 제목 |
| `body200` | TEXT | 본문 앞 200자 |
| `keywords` | TEXT | 쉼표 구분 키워드·개체명 |

**계획 2 가 만드는 테이블이 이 스키마와 일치해야 한다.** 컬럼명이 다르면 계획 2 쪽을 맞추거나 `adapters/news_db.py` 의 SELECT 별칭에서 흡수한다 — 나머지 모듈은 위 이름만 안다. 여분 컬럼은 있어도 무해하다.

계획 2 가 아직 없어도 Task 1~5 는 진행 가능하다. Task 6 CLI 스모크에서 처음 실 `news.db` 가 필요하다.

## 고정 스키마 — 7필드 (스펙 §3)

| 필드 | 값 | 스펙 §3 대응 |
|---|---|---|
| `material_type` | 수주 / 실적 / 신약 / 증설 / 정책 / 테마 / 루머 / 지분 / 기타 | 재료 유형 |
| `persistence` | `one_off` / `multi_quarter` / `structural` | 지속성 단서 |
| `specificity` | 0 / 1 / 2 / 3 | 구체성 |
| `freshness` | `new` / `follow_up` / `repeat` | 신선도 |
| `sentiment_direction` | -1 / 0 / 1 | 감정 — 방향 |
| `sentiment_strength` | 0 / 1 / 2 / 3 | 감정 — 강도 |
| `scale_vs_size` | `unknown` / `small` / `medium` / `large` | 규모 대비 |

모델이 쓰는 수치 피처는 5개로 축약한다 (`LLM_FEATURE_COLUMNS`). 스펙 §6 의 "피처 10~20개" 상한 때문에 `material_type` 을 원-핫으로 펼치지 않는다 — 문자열 컬럼으로 그대로 넘기고 범주 처리 방식은 계획 4 에서 정한다.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/swing_it/score/__init__.py` | 모듈 설명 + 공개 심볼 |
| `src/swing_it/score/schema.py` | 고정 스키마 정의·파싱·검증·수치 인코딩. **순수** |
| `src/swing_it/score/prompt.py` | 프롬프트 실제 문안과 `PROMPT_VERSION`. **순수** |
| `src/swing_it/score/provider.py` | `LLMProvider` 프로토콜, `FakeProvider`, httpx 기반 3종, env 팩토리 |
| `src/swing_it/score/prefilter.py` | 규칙 기반 사전 필터 (LLM 에 태울 건수 축소). **순수** |
| `src/swing_it/score/cache.py` | `data/llm_features.db` — 결과 캐시, 모델·프롬프트 버전 기록 |
| `src/swing_it/score/batch.py` | 배치 러너 — 캐시 스킵·재시도·부분 실패 집계 |
| `src/swing_it/score/compare.py` | 두 모델 산출물 대조 (일치율·MAE·순위상관). **순수** |
| `src/swing_it/score/cli.py` | `swing-score`, `swing-score-compare` 엔트리포인트 |
| `src/swing_it/adapters/news_db.py` | `news.db` 읽기. **뉴스 SQL 은 여기에만** |
| `src/swing_it/features/llm.py` | 기사 단위 피처를 `(code, date)` 격자로 접는다. **계획 4 의 유일한 접점** |
| `tests/` | 위 각 모듈 대응 테스트 |

---

### Task 1: 고정 스키마와 프롬프트

**Files:**
- Create: `src/swing_it/score/__init__.py`
- Create: `src/swing_it/score/schema.py`
- Create: `src/swing_it/score/prompt.py`
- Test: `tests/test_score_schema.py`
- Test: `tests/test_score_prompt.py`

**Interfaces:**
- Consumes: 없음 (순수 모듈)
- Produces:
  - `MATERIAL_TYPES: tuple[str, ...]`, `PERSISTENCE_LEVELS`, `FRESHNESS_LEVELS`, `SCALE_LEVELS: tuple[str, ...]`
  - `LLM_FEATURE_COLUMNS: list[str]` = `["llm_persistence", "llm_specificity", "llm_freshness", "llm_sentiment", "llm_scale"]`
  - `class SchemaError(ValueError)`
  - `@dataclass(frozen=True) class ArticleFeatures` — 필드 `material_type: str, persistence: str, specificity: int, freshness: str, sentiment_direction: int, sentiment_strength: int, scale_vs_size: str`
  - `parse_features(raw: str | dict) -> ArticleFeatures` — 문자열이면 첫 JSON 객체를 뽑아 파싱. 값이 스키마를 벗어나면 `SchemaError`
  - `encode_features(features: ArticleFeatures) -> dict[str, float]` — 키는 정확히 `LLM_FEATURE_COLUMNS`
  - `PROMPT_VERSION: str` = `"v1"`
  - `build_prompt(article: dict) -> tuple[str, str]` — `(system, user)`. `article` 은 `code, title, body200, keywords` 키를 읽는다
  - `FORBIDDEN_PROMPT_TERMS: tuple[str, ...]` — 프롬프트에 들어가면 안 되는 낱말

- [ ] **Step 1: 실패하는 스키마 테스트 작성**

`tests/test_score_schema.py`:

```python
import pytest

from swing_it.score.schema import (
    LLM_FEATURE_COLUMNS,
    ArticleFeatures,
    SchemaError,
    encode_features,
    parse_features,
)

_VALID = {
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "large",
}


def test_parse_accepts_dict():
    features = parse_features(dict(_VALID))
    assert features == ArticleFeatures(
        material_type="수주",
        persistence="multi_quarter",
        specificity=3,
        freshness="new",
        sentiment_direction=1,
        sentiment_strength=2,
        scale_vs_size="large",
    )


def test_parse_strips_code_fence_and_prose():
    raw = (
        "물론입니다. 아래가 결과입니다.\n"
        "```json\n"
        '{"material_type":"실적","persistence":"one_off","specificity":1,'
        '"freshness":"repeat","sentiment_direction":-1,"sentiment_strength":1,'
        '"scale_vs_size":"unknown"}\n'
        "```\n"
    )
    features = parse_features(raw)
    assert features.material_type == "실적"
    assert features.sentiment_direction == -1


def test_parse_rejects_unknown_material_type():
    bad = dict(_VALID, material_type="어닝서프라이즈")
    with pytest.raises(SchemaError, match="material_type"):
        parse_features(bad)


def test_parse_rejects_out_of_range_specificity():
    with pytest.raises(SchemaError, match="specificity"):
        parse_features(dict(_VALID, specificity=7))


def test_parse_rejects_missing_field():
    partial = {k: v for k, v in _VALID.items() if k != "freshness"}
    with pytest.raises(SchemaError, match="freshness"):
        parse_features(partial)


def test_parse_rejects_text_without_json():
    with pytest.raises(SchemaError):
        parse_features("죄송합니다. 답변할 수 없습니다.")


def test_parse_coerces_numeric_strings():
    # LLM 이 숫자를 문자열로 내는 일이 흔하다. 값만 맞으면 받아준다.
    features = parse_features(dict(_VALID, specificity="3", sentiment_direction="1"))
    assert features.specificity == 3
    assert features.sentiment_direction == 1


def test_encode_returns_exactly_feature_columns():
    encoded = encode_features(parse_features(dict(_VALID)))
    assert list(encoded.keys()) == LLM_FEATURE_COLUMNS


def test_encode_maps_ordinals_to_numbers():
    encoded = encode_features(parse_features(dict(_VALID)))
    assert encoded["llm_persistence"] == 1.0   # multi_quarter
    assert encoded["llm_specificity"] == 3.0
    assert encoded["llm_freshness"] == 2.0     # new
    assert encoded["llm_scale"] == 3.0         # large


def test_sentiment_is_direction_times_strength():
    positive = encode_features(parse_features(dict(_VALID)))
    negative = encode_features(
        parse_features(dict(_VALID, sentiment_direction=-1, sentiment_strength=3))
    )
    neutral = encode_features(
        parse_features(dict(_VALID, sentiment_direction=0, sentiment_strength=3))
    )
    assert positive["llm_sentiment"] == 2.0
    assert negative["llm_sentiment"] == -3.0
    assert neutral["llm_sentiment"] == 0.0
```

- [ ] **Step 2: 실패하는 프롬프트 테스트 작성**

`tests/test_score_prompt.py`:

```python
from swing_it.score.prompt import FORBIDDEN_PROMPT_TERMS, PROMPT_VERSION, build_prompt
from swing_it.score.schema import MATERIAL_TYPES

_ARTICLE = {
    "code": "005930",
    "title": "삼성전자, 미국 데이터센터에 HBM 대규모 공급 계약",
    "body200": "삼성전자가 미국 A사와 3년간 2조원 규모의 HBM 공급 계약을 체결했다고 17일 공시했다.",
    "keywords": "삼성전자,HBM,공급계약,데이터센터",
}


def test_prompt_embeds_the_article_fields():
    system, user = build_prompt(_ARTICLE)
    assert isinstance(system, str) and system
    assert _ARTICLE["title"] in user
    assert _ARTICLE["body200"] in user
    assert _ARTICLE["keywords"] in user
    assert _ARTICLE["code"] in user


def test_prompt_never_asks_for_a_price_verdict():
    """스펙 §3 의 중심축 — LLM 은 '오를까'를 판정하지 않는다."""
    system, user = build_prompt(_ARTICLE)
    whole = system + user
    for term in FORBIDDEN_PROMPT_TERMS:
        assert term not in whole, f"프롬프트에 판정 유도 낱말이 들어갔다: {term}"


def test_forbidden_terms_cover_the_obvious_ones():
    for term in ("수익률", "목표주가", "매수", "투자의견"):
        assert term in FORBIDDEN_PROMPT_TERMS


def test_prompt_lists_every_material_type():
    _, user = build_prompt(_ARTICLE)
    for material in MATERIAL_TYPES:
        assert material in user


def test_prompt_does_not_leak_full_body():
    """입력은 제목 + 200자뿐이다 (스펙 §4). 전문 키가 있어도 무시해야 한다."""
    _, user = build_prompt(dict(_ARTICLE, body="전문 " * 500))
    assert "전문 전문" not in user


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION and isinstance(PROMPT_VERSION, str)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_schema.py tests/test_score_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score'`

- [ ] **Step 4: 스키마 구현**

`src/swing_it/score/__init__.py`:

```python
"""LLM 구조화 피처 추출 (스펙 §3).

LLM 은 판정기가 아니라 추출기다. 기사 하나당 고정 스키마 7필드를 채울 뿐이며,
"이 종목이 오를까"는 묻지 않는다. 그 답을 믿는 순간 검증이 불가능해진다.

계층:
    schema/prompt/prefilter/compare  순수. 네트워크를 모른다.
    provider                          HTTP. httpx 로 직접 호출한다 (SDK 3종 미도입).
    cache/batch                       sqlite 캐시와 배치 러너.
"""
```

`src/swing_it/score/schema.py`:

```python
"""고정 스키마 정의와 파싱 — 순수 모듈.

LLM 출력은 불안정하다. 스키마를 코드로 강제하고, 벗어나면 실패로 처리해
재시도하게 만든다 (스펙 §11 "LLM 출력 불안정 → 고정 스키마 강제").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

MATERIAL_TYPES: tuple[str, ...] = (
    "수주", "실적", "신약", "증설", "정책", "테마", "루머", "지분", "기타",
)
PERSISTENCE_LEVELS: tuple[str, ...] = ("one_off", "multi_quarter", "structural")
FRESHNESS_LEVELS: tuple[str, ...] = ("repeat", "follow_up", "new")
SCALE_LEVELS: tuple[str, ...] = ("unknown", "small", "medium", "large")

LLM_FEATURE_COLUMNS: list[str] = [
    "llm_persistence",
    "llm_specificity",
    "llm_freshness",
    "llm_sentiment",
    "llm_scale",
]


class SchemaError(ValueError):
    """LLM 출력이 고정 스키마를 벗어났다. 배치 러너가 재시도 사유로 쓴다."""


@dataclass(frozen=True)
class ArticleFeatures:
    """기사 하나에서 뽑은 고정 스키마 7필드 (스펙 §3)."""

    material_type: str
    persistence: str
    specificity: int
    freshness: str
    sentiment_direction: int
    sentiment_strength: int
    scale_vs_size: str


_CATEGORICAL = {
    "material_type": MATERIAL_TYPES,
    "persistence": PERSISTENCE_LEVELS,
    "freshness": FRESHNESS_LEVELS,
    "scale_vs_size": SCALE_LEVELS,
}
_NUMERIC = {
    "specificity": (0, 1, 2, 3),
    "sentiment_direction": (-1, 0, 1),
    "sentiment_strength": (0, 1, 2, 3),
}

_PERSISTENCE_SCORE = {"one_off": 0.0, "multi_quarter": 1.0, "structural": 2.0}
_FRESHNESS_SCORE = {"repeat": 0.0, "follow_up": 1.0, "new": 2.0}
# unknown 을 0 에 두는 것은 판정 불가와 '작다'를 실무상 구분할 수 없기 때문이다.
# NaN 으로 두면 dataset 조립 단계의 dropna 가 해당 기사를 통째로 날린다.
_SCALE_SCORE = {"unknown": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}


def _extract_json_object(text: str) -> dict:
    """문자열에서 첫 번째 JSON 객체를 꺼낸다. 코드펜스·앞뒤 산문을 견딘다."""
    start = text.find("{")
    if start < 0:
        raise SchemaError(f"응답에 JSON 객체가 없다: {text[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"JSON 파싱 실패: {exc}") from exc
    raise SchemaError(f"닫히지 않은 JSON 객체: {text[:120]!r}")


def _as_int(name: str, value: object) -> int:
    """LLM 이 숫자를 문자열로 내는 일이 흔하므로 정수로 좁혀 받는다."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} 가 정수가 아니다: {value!r}") from exc


def parse_features(raw: str | dict) -> ArticleFeatures:
    """LLM 응답을 검증된 ``ArticleFeatures`` 로 바꾼다.

    Raises:
        SchemaError: JSON 이 아니거나, 필드가 빠졌거나, 값이 허용 집합 밖일 때.
    """
    payload = raw if isinstance(raw, dict) else _extract_json_object(raw)

    for field in fields(ArticleFeatures):
        if field.name not in payload:
            raise SchemaError(f"필드 누락: {field.name}")

    values: dict[str, object] = {}
    for name, allowed_strings in _CATEGORICAL.items():
        value = str(payload[name]).strip()
        if value not in allowed_strings:
            raise SchemaError(f"{name} 값이 허용 집합 밖이다: {value!r}")
        values[name] = value
    for name, allowed_numbers in _NUMERIC.items():
        number = _as_int(name, payload[name])
        if number not in allowed_numbers:
            raise SchemaError(f"{name} 값이 허용 범위 밖이다: {number}")
        values[name] = number

    return ArticleFeatures(**values)  # type: ignore[arg-type]


def encode_features(features: ArticleFeatures) -> dict[str, float]:
    """모델이 쓰는 수치 피처로 인코딩한다. 키 순서는 ``LLM_FEATURE_COLUMNS``.

    ``material_type`` 은 여기서 원-핫으로 펼치지 않는다. 스펙 §6 의 피처 10~20개
    상한 때문이며, 범주 처리 방식은 계획 4(층별 모델 비교)에서 정한다.
    """
    return {
        "llm_persistence": _PERSISTENCE_SCORE[features.persistence],
        "llm_specificity": float(features.specificity),
        "llm_freshness": _FRESHNESS_SCORE[features.freshness],
        "llm_sentiment": float(features.sentiment_direction * features.sentiment_strength),
        "llm_scale": _SCALE_SCORE[features.scale_vs_size],
    }
```

- [ ] **Step 5: 프롬프트 구현**

`src/swing_it/score/prompt.py`:

```python
"""프롬프트 실제 문안과 버전.

**이 파일을 고치면 ``PROMPT_VERSION`` 을 반드시 올린다.** 캐시 키가
``(article_id, model, prompt_version)`` 이므로, 버전을 안 올리면 옛 프롬프트로
뽑은 결과와 새 프롬프트 결과가 한 테이블에 섞여 재현성이 깨진다 (스펙 §11).

문안 규칙: 여기 어디에도 "오를까 / 살 만한가 / 목표주가"를 묻는 문장을 넣지 않는다.
``FORBIDDEN_PROMPT_TERMS`` 가 테스트로 이를 지킨다 (스펙 §3).
"""

from __future__ import annotations

from .schema import MATERIAL_TYPES

PROMPT_VERSION = "v1"

FORBIDDEN_PROMPT_TERMS: tuple[str, ...] = (
    "수익률",
    "목표주가",
    "매수",
    "매도",
    "투자의견",
    "오를",
    "상승률",
    "유망",
    "추천",
)

_SYSTEM = """너는 한국 상장사 기사에서 고정 스키마 필드를 채우는 추출기다.

규칙:
- 주어진 텍스트에 적힌 내용만 쓴다. 배경지식으로 보충하지 않는다.
- 이 작업에 시세 판단은 포함되지 않는다. 스키마에 없는 것은 답하지 않는다.
- 근거가 없으면 가장 보수적인 값(unknown 또는 0)을 고른다.
- 출력은 JSON 객체 하나뿐이다. 코드펜스·설명·주석을 붙이지 않는다."""

_USER_TEMPLATE = """다음은 종목코드 {code} 관련 기사다.

제목: {title}
본문(앞 200자): {body200}
키워드: {keywords}

아래 7개 필드를 채운 JSON 객체 하나만 출력하라.

material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off(일회성 사건) / multi_quarter(여러 분기에 걸쳐 반영) / structural(사업 구조가 바뀌는 수준)
specificity: 구체성. 0(방향만 언급) / 1(대상과 시기 중 하나만 명시) / 2(둘 다 있으나 금액 없음) / 3(금액·기간·상대방이 모두 명시)
freshness: 정보 신선도. new(처음 알려지는 내용) / follow_up(기존 건의 진행 경과) / repeat(이미 알려진 내용의 재보도)
sentiment_direction: 이 사안이 회사 사업에 미치는 방향. -1(부정) / 0(중립) / 1(긍정)
sentiment_strength: 그 방향의 강도. 0(거의 없음) / 1(작음) / 2(보통) / 3(큼)
scale_vs_size: 회사 규모(시가총액·매출) 대비 사안의 크기. unknown(판단 근거가 본문에 없음) / small / medium / large

출력 형식 예시:
{{"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""


def build_prompt(article: dict) -> tuple[str, str]:
    """기사 한 건의 ``(system, user)`` 프롬프트를 만든다.

    ``article`` 의 ``code``·``title``·``body200``·``keywords`` 만 읽는다. 전문이
    딸려 와도 쓰지 않는다 — 학습(빅카인즈 200자)과 운영(전문)의 입력을 통일하기
    위해서다 (스펙 §4).
    """
    user = _USER_TEMPLATE.format(
        code=str(article.get("code") or "").strip(),
        title=str(article.get("title") or "").strip(),
        body200=str(article.get("body200") or "").strip(),
        keywords=str(article.get("keywords") or "").strip(),
        material_types=" / ".join(MATERIAL_TYPES),
    )
    return _SYSTEM, user
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_schema.py tests/test_score_prompt.py -v`
Expected: 16 passed

- [ ] **Step 7: 커밋**

```bash
git add src/swing_it/score/__init__.py src/swing_it/score/schema.py \
        src/swing_it/score/prompt.py \
        tests/test_score_schema.py tests/test_score_prompt.py
git commit -m "feat(score): 고정 스키마 7필드 + 프롬프트 v1 (판정 문항 금지 테스트 포함)"
```

---

### Task 2: provider 추상화 (httpx 직접 호출)

**Files:**
- Modify: `pyproject.toml` — `dependencies` 에 httpx 추가
- Create: `src/swing_it/score/provider.py`
- Test: `tests/test_score_provider.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `class ProviderError(RuntimeError)` — 네트워크·HTTP·응답 형식 오류. 배치 러너의 재시도 사유
  - `class LLMProvider(Protocol)` — 속성 `name: str`, `model: str`; 메서드 `complete(self, system: str, user: str) -> str`
  - `class FakeProvider` — `__init__(self, *, model: str = "fake-1", responses: list[str | Exception] | None = None, default: str | None = None)`. 속성 `name = "fake"`, `calls: list[tuple[str, str]]`
  - `class HttpProvider` — `__init__(self, name: str, model: str, api_key: str, *, timeout: float = 60.0, client: httpx.Client | None = None)`
  - `provider_from_env(env: Mapping[str, str] | None = None) -> LLMProvider`
  - `DEFAULT_MODELS: dict[str, str]`

**환경변수 계약:**

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SWING_LLM_PROVIDER` | `gemini` | `gemini` / `openai` / `anthropic` / `fake` |
| `SWING_LLM_MODEL` | `DEFAULT_MODELS[provider]` | 모델 ID. openai 는 기본값이 없어 필수 |
| `SWING_LLM_API_KEY` | — | 없으면 `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 순으로 찾는다 |
| `SWING_LLM_TIMEOUT` | `60` | 초 |

**SDK 3종을 넣지 않고 httpx 로 직접 부르는 이유:** 세 provider 의 요청 본문은
각각 20줄 남짓이고 우리가 쓰는 기능은 "system + user → 텍스트 하나"뿐이다.
`google-genai` + `openai` + `anthropic` 을 모두 의존성에 올리면 공개 전환 시
설치 부담이 그 이득을 넘는다. 요청 조립·응답 추출을 순수 함수로 분리해
`httpx.MockTransport` 로 전부 테스트한다.

- [ ] **Step 1: `pyproject.toml` 의존성 수정**

`[project]` 의 `dependencies` 를 아래로 바꾼다 (`httpx` 한 줄 추가):

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "httpx>=0.27",
]
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_score_provider.py`:

```python
import httpx
import pytest

from swing_it.score.provider import (
    DEFAULT_MODELS,
    FakeProvider,
    HttpProvider,
    ProviderError,
    provider_from_env,
)

_JSON = (
    '{"material_type":"수주","persistence":"one_off","specificity":2,'
    '"freshness":"new","sentiment_direction":1,"sentiment_strength":2,'
    '"scale_vs_size":"small"}'
)


# --- FakeProvider ---------------------------------------------------------

def test_fake_provider_returns_default_and_records_calls():
    provider = FakeProvider()
    out = provider.complete("sys", "user")
    assert "material_type" in out
    assert provider.calls == [("sys", "user")]


def test_fake_provider_plays_scripted_responses_in_order():
    provider = FakeProvider(responses=["첫번째", "두번째"])
    assert provider.complete("s", "u") == "첫번째"
    assert provider.complete("s", "u") == "두번째"


def test_fake_provider_raises_scripted_exception():
    provider = FakeProvider(responses=[ProviderError("일시 오류"), _JSON])
    with pytest.raises(ProviderError):
        provider.complete("s", "u")
    assert provider.complete("s", "u") == _JSON


# --- HttpProvider (MockTransport, 네트워크 없음) ---------------------------

def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": _JSON}]}}]},
        )

    provider = HttpProvider("gemini", "flash-lite-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert "flash-lite-test:generateContent" in seen["url"]
    assert seen["key"] == "KEY123"
    assert "시스템" in seen["body"] and "사용자" in seen["body"]


def test_openai_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": _JSON}}]})

    provider = HttpProvider("openai", "mini-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert seen["auth"] == "Bearer KEY123"
    assert "json_object" in seen["body"]


def test_anthropic_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={"content": [{"type": "text", "text": _JSON}]})

    provider = HttpProvider("anthropic", "haiku-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert seen["key"] == "KEY123"
    assert seen["version"] == "2023-06-01"


def test_http_error_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError, match="503"):
        provider.complete("s", "u")


def test_transport_failure_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("연결 실패")

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError):
        provider.complete("s", "u")


def test_unexpected_payload_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": True})

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError, match="응답 형식"):
        provider.complete("s", "u")


# --- env 팩토리 -----------------------------------------------------------

def test_env_factory_builds_fake_provider():
    provider = provider_from_env({"SWING_LLM_PROVIDER": "fake"})
    assert provider.name == "fake"


def test_env_factory_defaults_to_gemini_with_default_model():
    provider = provider_from_env({"GEMINI_API_KEY": "K"})
    assert provider.name == "gemini"
    assert provider.model == DEFAULT_MODELS["gemini"]


def test_env_factory_honours_explicit_model():
    provider = provider_from_env({"GEMINI_API_KEY": "K", "SWING_LLM_MODEL": "custom-1"})
    assert provider.model == "custom-1"


def test_env_factory_requires_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        provider_from_env({"SWING_LLM_PROVIDER": "gemini"})


def test_env_factory_requires_model_for_openai():
    with pytest.raises(ValueError, match="SWING_LLM_MODEL"):
        provider_from_env({"SWING_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "K"})


def test_env_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="grok"):
        provider_from_env({"SWING_LLM_PROVIDER": "grok"})


def test_env_factory_never_returns_the_key_in_repr():
    provider = provider_from_env({"GEMINI_API_KEY": "SECRET-KEY-VALUE"})
    assert "SECRET-KEY-VALUE" not in repr(provider)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score.provider'`

- [ ] **Step 4: 구현**

`src/swing_it/score/provider.py`:

```python
"""LLM provider 추상화 — Gemini / OpenAI / Anthropic 을 환경변수로 교체한다 (스펙 §8).

SDK 를 쓰지 않고 httpx 로 직접 부른다. 우리가 쓰는 기능은 "system + user →
텍스트 하나"뿐이라 provider 당 요청 조립이 20줄 남짓이고, SDK 3종을 의존성에
올리면 공개 전환 시 설치 부담이 그 이득을 넘는다.

API 키는 환경변수로만 받는다. 어떤 경로로도 로그·repr 에 실리지 않게 한다.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import httpx

# 착수 시점에 실제로 서빙되는 ID 인지 확인한다 (Task 7 Step 1).
# 기본은 최저가 등급이다 — 기사당 작업이 "제목+리드 읽고 스키마 채우기"라
# 추론 난이도가 낮다 (스펙 §8).
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash-lite",
    "anthropic": "claude-haiku-4-5-20251001",
    # openai 는 기본값을 두지 않는다. 등급 이름이 자주 바뀌어 낡은 상수가
    # 조용히 비싼 모델을 부르는 사고를 막는다. SWING_LLM_MODEL 로 명시하라.
}

_ENV_KEY_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_DEFAULT_FAKE_JSON = (
    '{"material_type":"기타","persistence":"one_off","specificity":0,'
    '"freshness":"repeat","sentiment_direction":0,"sentiment_strength":0,'
    '"scale_vs_size":"unknown"}'
)

_MAX_OUTPUT_TOKENS = 300  # 7필드 JSON 이면 충분하다. 폭주 과금을 막는 상한.


class ProviderError(RuntimeError):
    """호출이 실패했다. 배치 러너가 재시도 사유로 쓴다."""


@runtime_checkable
class LLMProvider(Protocol):
    """배치 러너가 아는 유일한 인터페이스."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """모델 응답 텍스트를 그대로 돌려준다. 파싱은 호출자 책임."""
        ...


class FakeProvider:
    """테스트·드라이런용. 네트워크에 나가지 않는다.

    ``responses`` 에 예외 인스턴스를 섞으면 실패 시나리오를 재현할 수 있다.
    """

    name = "fake"

    def __init__(
        self,
        *,
        model: str = "fake-1",
        responses: list[str | Exception] | None = None,
        default: str | None = None,
    ) -> None:
        self.model = model
        self._responses: list[str | Exception] = list(responses or [])
        self._default = default if default is not None else _DEFAULT_FAKE_JSON
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return self._default


# --- provider 별 요청 조립 / 응답 추출 (순수 함수) --------------------------

_Request = tuple[str, dict[str, str], dict[str, Any]]


def _gemini_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    return url, headers, body


def _gemini_extract(payload: dict) -> str:
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def _openai_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": _MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    return url, headers, body


def _openai_extract(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


def _anthropic_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "system": system,
        "temperature": 0,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": user}],
    }
    return url, headers, body


def _anthropic_extract(payload: dict) -> str:
    for block in payload["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise KeyError("text 블록이 없다")


_SPECS: dict[str, tuple[Callable[..., _Request], Callable[[dict], str]]] = {
    "gemini": (_gemini_request, _gemini_extract),
    "openai": (_openai_request, _openai_extract),
    "anthropic": (_anthropic_request, _anthropic_extract),
}


class HttpProvider:
    """httpx 로 직접 부르는 provider. ``client`` 를 주입하면 테스트가 가능하다."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if name not in _SPECS:
            raise ValueError(f"모르는 provider: {name}. 가능한 값: {sorted(_SPECS)}")
        self.name = name
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def __repr__(self) -> str:  # 키가 로그·예외에 실리지 않게 한다
        return f"HttpProvider(name={self.name!r}, model={self.model!r})"

    def complete(self, system: str, user: str) -> str:
        build, extract = _SPECS[self.name]
        url, headers, body = build(self.model, self._api_key, system, user)
        try:
            response = self._client.post(url, headers=headers, json=body, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} 요청 실패: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            return extract(response.json())
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name} 응답 형식이 예상과 다르다: {exc}") from exc


def provider_from_env(env: Mapping[str, str] | None = None) -> LLMProvider:
    """환경변수로 provider 를 조립한다. 키는 여기서만 읽는다."""
    source: Mapping[str, str] = env if env is not None else os.environ
    name = source.get("SWING_LLM_PROVIDER", "gemini").strip().lower()

    if name == "fake":
        return FakeProvider(model=source.get("SWING_LLM_MODEL", "fake-1"))
    if name not in _SPECS:
        raise ValueError(
            f"모르는 SWING_LLM_PROVIDER: {name!r}. 가능한 값: fake, {', '.join(sorted(_SPECS))}"
        )

    model = source.get("SWING_LLM_MODEL") or DEFAULT_MODELS.get(name)
    if not model:
        raise ValueError(f"{name} 는 기본 모델이 없다. SWING_LLM_MODEL 을 설정하라.")

    key_name = _ENV_KEY_NAMES[name]
    api_key = source.get("SWING_LLM_API_KEY") or source.get(key_name)
    if not api_key:
        raise ValueError(f"API 키가 없다. SWING_LLM_API_KEY 또는 {key_name} 를 설정하라.")

    timeout = float(source.get("SWING_LLM_TIMEOUT", "60"))
    return HttpProvider(name, model, api_key, timeout=timeout)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_provider.py -v`
Expected: 16 passed

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml src/swing_it/score/provider.py tests/test_score_provider.py
git commit -m "feat(score): provider 추상화 — httpx 직접 호출 3종 + FakeProvider"
```

---

### Task 3: 규칙 기반 사전 필터

**Files:**
- Create: `src/swing_it/score/prefilter.py`
- Test: `tests/test_score_prefilter.py`

**Interfaces:**
- Consumes: 뉴스 테이블 `[article_id, date, code, press, title, body200, keywords]`, 계획 1 `extract_events` 산출 `[code, date, adv, tv_ratio]`
- Produces:
  - `DROP_REASONS: tuple[str, ...]` = `("no_code", "too_short", "no_volume_event", "duplicate")`
  - `normalize_title(title: str) -> str`
  - `prefilter_articles(articles: pd.DataFrame, events: pd.DataFrame, *, min_chars: int = 30) -> pd.DataFrame` — 입력 전 행에 `keep: bool`, `drop_reason: str`(통과 시 `""`), `dup_count: int` 를 붙여 돌려준다. 행을 지우지 않는 이유는 **왜 빠졌는지를 리포트해야** 임계값을 조정할 수 있기 때문이다
  - `prefilter_summary(filtered: pd.DataFrame) -> dict[str, int]` — `{"n_input", "n_kept", "no_code", "too_short", "no_volume_event", "duplicate"}`

**규칙 적용 순서** (앞 규칙에서 탈락하면 뒤는 보지 않는다):

1. `no_code` — 종목 매핑 실패. `code` 가 결측·빈 문자열이면 붙일 라벨이 없다
2. `too_short` — `title + body200` 이 `min_chars` 미만. LLM 이 채울 내용이 없다
3. `no_volume_event` — `(code, date)` 가 거래대금 급증 이벤트에 없다. 시장이 반응하지 않은 기사는 재료가 아니다 (스펙 §4 '핫함')
4. `duplicate` — 같은 `(date, code, 정규화 제목)` 그룹에서 첫 건만 남긴다. `dup_count` 에 그룹 크기를 기록해 뒷단이 '보도 매체 수' 계산에 참고할 수 있게 한다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_score_prefilter.py`:

```python
import pandas as pd

from swing_it.score.prefilter import normalize_title, prefilter_articles, prefilter_summary

_BODY = "삼성전자가 미국 A사와 3년간 2조원 규모의 HBM 공급 계약을 체결했다고 공시했다."


def _articles(rows: list[dict]) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "date": "2026-01-05",
        "code": "005930",
        "press": "한국경제",
        "title": "삼성전자, HBM 대규모 공급 계약",
        "body200": _BODY,
        "keywords": "삼성전자,HBM",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _events(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [c for c, _ in pairs],
            "date": pd.to_datetime([d for _, d in pairs]),
            "adv": [6000.0] * len(pairs),
            "tv_ratio": [4.0] * len(pairs),
        }
    )


def test_normalize_title_strips_prefix_and_punctuation():
    assert normalize_title("[단독] 삼성전자, HBM 공급 계약!") == normalize_title(
        "삼성전자 HBM 공급 계약"
    )
    assert normalize_title("[속보]현대차 수주") == normalize_title("현대차 수주")


def test_clean_article_is_kept():
    out = prefilter_articles(_articles([{}]), _events([("005930", "2026-01-05")]))
    assert bool(out.iloc[0]["keep"]) is True
    assert out.iloc[0]["drop_reason"] == ""
    assert out.iloc[0]["dup_count"] == 1


def test_unmapped_code_is_dropped():
    out = prefilter_articles(
        _articles([{"code": ""}, {"article_id": "a2", "code": None}]),
        _events([("005930", "2026-01-05")]),
    )
    assert list(out["drop_reason"]) == ["no_code", "no_code"]


def test_short_article_is_dropped():
    out = prefilter_articles(
        _articles([{"title": "삼성전자", "body200": ""}]),
        _events([("005930", "2026-01-05")]),
    )
    assert out.iloc[0]["drop_reason"] == "too_short"


def test_article_without_volume_event_is_dropped():
    out = prefilter_articles(_articles([{}]), _events([("000660", "2026-01-05")]))
    assert out.iloc[0]["drop_reason"] == "no_volume_event"


def test_event_on_a_different_day_does_not_count():
    out = prefilter_articles(_articles([{}]), _events([("005930", "2026-01-06")]))
    assert out.iloc[0]["drop_reason"] == "no_volume_event"


def test_duplicate_titles_keep_only_the_first():
    out = prefilter_articles(
        _articles(
            [
                {"article_id": "a1", "press": "한국경제"},
                {"article_id": "a2", "press": "매일경제", "title": "[단독] 삼성전자, HBM 대규모 공급 계약"},
                {"article_id": "a3", "press": "더벨", "title": "삼성전자 HBM 대규모 공급계약!"},
            ]
        ),
        _events([("005930", "2026-01-05")]),
    )
    assert list(out["keep"]) == [True, False, False]
    assert list(out["drop_reason"]) == ["", "duplicate", "duplicate"]
    assert list(out["dup_count"]) == [3, 3, 3]


def test_same_title_different_code_is_not_duplicate():
    out = prefilter_articles(
        _articles(
            [
                {"article_id": "a1", "code": "005930"},
                {"article_id": "a2", "code": "000660"},
            ]
        ),
        _events([("005930", "2026-01-05"), ("000660", "2026-01-05")]),
    )
    assert list(out["keep"]) == [True, True]


def test_summary_counts_every_reason():
    articles = _articles(
        [
            {"article_id": "a1"},
            {"article_id": "a2", "code": ""},
            {"article_id": "a3", "title": "짧다", "body200": ""},
            {"article_id": "a4", "code": "000660"},
            {"article_id": "a5"},
        ]
    )
    out = prefilter_articles(articles, _events([("005930", "2026-01-05")]))
    summary = prefilter_summary(out)
    assert summary == {
        "n_input": 5,
        "n_kept": 1,
        "no_code": 1,
        "too_short": 1,
        "no_volume_event": 1,
        "duplicate": 1,
    }


def test_input_row_order_is_preserved():
    articles = _articles([{"article_id": f"a{i}"} for i in range(5)])
    out = prefilter_articles(articles, _events([("005930", "2026-01-05")]))
    assert list(out["article_id"]) == ["a0", "a1", "a2", "a3", "a4"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_prefilter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score.prefilter'`

- [ ] **Step 3: 구현**

`src/swing_it/score/prefilter.py`:

```python
"""LLM 에 태울 기사를 줄이는 규칙 기반 사전 필터 — 순수 함수.

비용을 줄이는 게 목적이지만 그게 전부는 아니다. 종목이 안 붙은 기사는 라벨을
만들 수 없고, 시장이 거래대금으로 반응하지 않은 기사는 스펙 §4 가 말하는 '핫함'
정의를 못 채운다. 어느 쪽도 LLM 을 태울 가치가 없다.

행을 지우지 않고 ``keep``/``drop_reason`` 을 붙여 돌려준다. 어떤 규칙이 몇 건을
잘랐는지 봐야 임계값을 조정할 수 있기 때문이다.
"""

from __future__ import annotations

import re

import pandas as pd

DROP_REASONS: tuple[str, ...] = ("no_code", "too_short", "no_volume_event", "duplicate")

_BRACKET_PREFIX = re.compile(r"^\s*[\[\(<【][^\]\)>】]{0,12}[\]\)>】]\s*")
_NON_WORD = re.compile(r"[^0-9A-Za-z가-힣]+")


def normalize_title(title: str) -> str:
    """중복 판정용 제목 정규화.

    ``[단독]``·``[속보]`` 류 말머리를 걷어내고, 공백·문장부호를 모두 지운다.
    같은 사안을 여러 매체가 받아쓸 때 남는 차이가 대개 이 둘뿐이다.
    """
    text = str(title or "")
    while True:
        stripped = _BRACKET_PREFIX.sub("", text)
        if stripped == text:
            break
        text = stripped
    return _NON_WORD.sub("", text)


def prefilter_articles(
    articles: pd.DataFrame,
    events: pd.DataFrame,
    *,
    min_chars: int = 30,
) -> pd.DataFrame:
    """규칙 순서대로 탈락 사유를 매긴다. 입력 행 순서를 보존한다."""
    out = articles.copy()
    out["_date"] = pd.to_datetime(out["date"])
    out["_title_key"] = out["title"].map(normalize_title)

    code = out["code"].fillna("").astype(str).str.strip()
    text_len = (
        out["title"].fillna("").astype(str).str.len()
        + out["body200"].fillna("").astype(str).str.len()
    )

    event_keys = set(
        zip(
            events["code"].astype(str),
            pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d"),
        )
    )
    has_event = pd.Series(
        [
            (c, d) in event_keys
            for c, d in zip(code, out["_date"].dt.strftime("%Y-%m-%d"))
        ],
        index=out.index,
    )

    reason = pd.Series("", index=out.index, dtype=object)
    reason[code.eq("")] = "no_code"
    reason[reason.eq("") & text_len.lt(min_chars)] = "too_short"
    reason[reason.eq("") & ~has_event] = "no_volume_event"

    # 중복은 살아남은 행들 사이에서만 따진다. 이미 탈락한 행이 첫 건 자리를
    # 차지해 진짜 기사를 duplicate 로 밀어내는 일이 없어야 한다.
    group_cols = ["_date", "code", "_title_key"]
    survivors = out.loc[reason.eq("")]
    if not survivors.empty:
        dup_rank = survivors.groupby(group_cols, sort=False).cumcount()
        reason.loc[dup_rank[dup_rank > 0].index] = "duplicate"

    # dup_count 는 탈락 행 포함 전체 그룹 크기다. 뒷단의 '보도 매체 수' 참고용.
    out["dup_count"] = out.groupby(group_cols, sort=False)["article_id"].transform("size")
    out["drop_reason"] = reason
    out["keep"] = reason.eq("")
    return out.drop(columns=["_date", "_title_key"])


def prefilter_summary(filtered: pd.DataFrame) -> dict[str, int]:
    """필터 결과를 리포트용 카운트로 접는다."""
    counts = filtered["drop_reason"].value_counts()
    summary = {
        "n_input": int(len(filtered)),
        "n_kept": int(filtered["keep"].sum()),
    }
    for reason in DROP_REASONS:
        summary[reason] = int(counts.get(reason, 0))
    return summary
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_prefilter.py -v`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/score/prefilter.py tests/test_score_prefilter.py
git commit -m "feat(score): 규칙 기반 사전 필터 — 매핑실패·짧은글·거래대금 미동반·중복 제거"
```

---

### Task 4: 결과 캐시

**Files:**
- Create: `src/swing_it/score/cache.py`
- Test: `tests/test_score_cache.py`

**Interfaces:**
- Consumes: `ArticleFeatures`, `LLM_FEATURE_COLUMNS`, `encode_features`
- Produces:
  - `CACHE_TABLE: str` = `"llm_features"`
  - `open_cache(path: str | Path) -> sqlite3.Connection` — 없으면 파일·테이블을 만든다. 여러 번 불러도 안전
  - `get_cached(conn, article_ids: Sequence[str], model: str, prompt_version: str) -> dict[str, ArticleFeatures]` — **성공 행만** 돌려준다
  - `put_result(conn, *, article_id: str, model: str, prompt_version: str, features: ArticleFeatures | None, raw: str | None, error: str | None) -> None` — 같은 키가 있으면 덮어쓴다
  - `load_features_frame(conn, model: str, prompt_version: str) -> pd.DataFrame` — 컬럼 `[article_id, material_type, persistence, specificity, freshness, sentiment_direction, sentiment_strength, scale_vs_size, *LLM_FEATURE_COLUMNS]`. 성공 행만

**설계 결정 — 실패 행도 저장하지만 `get_cached` 는 돌려주지 않는다.** 실패를 기록해야
리포트에 사유가 남고, 캐시 히트로 취급하지 않아야 다음 실행에서 자동 재시도된다.

**설계 결정 — 원문 응답(`raw_json`)을 보관한다.** 스키마를 바꾸거나 파서를 고쳤을 때
API 를 다시 태우지 않고 재파싱할 수 있다. 이 파일은 `data/` 아래라 커밋되지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_score_cache.py`:

```python
import pandas as pd

from swing_it.score.cache import get_cached, load_features_frame, open_cache, put_result
from swing_it.score.schema import LLM_FEATURE_COLUMNS, ArticleFeatures

_F = ArticleFeatures(
    material_type="수주",
    persistence="structural",
    specificity=3,
    freshness="new",
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)


def _conn(tmp_path):
    return open_cache(tmp_path / "llm.db")


def test_open_cache_is_idempotent(tmp_path):
    path = tmp_path / "llm.db"
    open_cache(path).close()
    conn = open_cache(path)
    assert conn.execute("select count(*) from llm_features").fetchone()[0] == 0


def test_put_and_get_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw='{"x":1}', error=None)
    cached = get_cached(conn, ["a1"], "m1", "v1")
    assert cached == {"a1": _F}


def test_error_rows_are_not_cache_hits(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=None, raw=None, error="타임아웃")
    assert get_cached(conn, ["a1"], "m1", "v1") == {}


def test_cache_is_keyed_by_model_and_prompt_version(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="cheap", prompt_version="v1",
               features=_F, raw=None, error=None)
    assert get_cached(conn, ["a1"], "cheap", "v1") == {"a1": _F}
    assert get_cached(conn, ["a1"], "premium", "v1") == {}
    assert get_cached(conn, ["a1"], "cheap", "v2") == {}


def test_put_overwrites_same_key(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=None, raw=None, error="1차 실패")
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw=None, error=None)
    assert get_cached(conn, ["a1"], "m1", "v1") == {"a1": _F}
    assert conn.execute("select count(*) from llm_features").fetchone()[0] == 1


def test_get_cached_handles_large_id_lists(tmp_path):
    """sqlite 변수 한도(999)를 넘는 조회도 나뉘어 처리돼야 한다."""
    conn = _conn(tmp_path)
    ids = [f"a{i}" for i in range(2500)]
    for article_id in ids:
        put_result(conn, article_id=article_id, model="m1", prompt_version="v1",
                   features=_F, raw=None, error=None)
    assert len(get_cached(conn, ids, "m1", "v1")) == 2500


def test_load_features_frame_shape(tmp_path):
    conn = _conn(tmp_path)
    put_result(conn, article_id="a1", model="m1", prompt_version="v1",
               features=_F, raw=None, error=None)
    put_result(conn, article_id="a2", model="m1", prompt_version="v1",
               features=None, raw=None, error="실패")
    frame = load_features_frame(conn, "m1", "v1")
    assert list(frame.columns) == [
        "article_id", "material_type", "persistence", "specificity", "freshness",
        "sentiment_direction", "sentiment_strength", "scale_vs_size",
        *LLM_FEATURE_COLUMNS,
    ]
    assert list(frame["article_id"]) == ["a1"]  # 실패 행은 빠진다
    assert frame.iloc[0]["llm_sentiment"] == 3.0
    assert pd.api.types.is_numeric_dtype(frame["llm_specificity"])


def test_load_features_frame_is_empty_when_nothing_matches(tmp_path):
    frame = load_features_frame(_conn(tmp_path), "m1", "v1")
    assert frame.empty
    assert "llm_sentiment" in frame.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score.cache'`

- [ ] **Step 3: 구현**

`src/swing_it/score/cache.py`:

```python
"""LLM 결과 캐시 — 같은 기사를 두 번 태우지 않는다.

캐시 키는 ``(article_id, model, prompt_version)`` 이다. 모델이나 프롬프트가
바뀌면 다른 결과이므로 다른 행이어야 하고, 결과 테이블에 둘 다 적혀 있어야
"이 피처가 무엇으로 뽑혔는지"를 나중에 재현할 수 있다 (스펙 §11).

실패도 저장하되 캐시 히트로 치지 않는다 — 리포트에는 남고, 다음 실행에서는
자동으로 재시도된다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .schema import LLM_FEATURE_COLUMNS, ArticleFeatures, encode_features

CACHE_TABLE = "llm_features"

_SCHEMA_COLUMNS = [
    "material_type", "persistence", "specificity", "freshness",
    "sentiment_direction", "sentiment_strength", "scale_vs_size",
]

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
    article_id          TEXT    NOT NULL,
    model               TEXT    NOT NULL,
    prompt_version      TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    error               TEXT,
    raw_json            TEXT,
    material_type       TEXT,
    persistence         TEXT,
    specificity         INTEGER,
    freshness           TEXT,
    sentiment_direction INTEGER,
    sentiment_strength  INTEGER,
    scale_vs_size       TEXT,
    PRIMARY KEY (article_id, model, prompt_version)
);
"""

_SQLITE_VARIABLE_LIMIT = 900  # 기본 한도 999 보다 여유를 둔다


def open_cache(path: str | Path) -> sqlite3.Connection:
    """캐시 DB 를 열고 없으면 테이블을 만든다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_CREATE)
    conn.commit()
    return conn


def get_cached(
    conn: sqlite3.Connection,
    article_ids: Sequence[str],
    model: str,
    prompt_version: str,
) -> dict[str, ArticleFeatures]:
    """이미 성공적으로 뽑아둔 기사의 피처를 돌려준다."""
    found: dict[str, ArticleFeatures] = {}
    ids = list(article_ids)
    for start in range(0, len(ids), _SQLITE_VARIABLE_LIMIT):
        chunk = ids[start : start + _SQLITE_VARIABLE_LIMIT]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"select article_id, {', '.join(_SCHEMA_COLUMNS)} from {CACHE_TABLE} "
            f"where model = ? and prompt_version = ? and status = 'ok' "
            f"and article_id in ({marks})",
            (model, prompt_version, *chunk),
        ).fetchall()
        for row in rows:
            found[row[0]] = ArticleFeatures(*row[1:])
    return found


def put_result(
    conn: sqlite3.Connection,
    *,
    article_id: str,
    model: str,
    prompt_version: str,
    features: ArticleFeatures | None,
    raw: str | None,
    error: str | None,
) -> None:
    """성공이든 실패든 한 건을 기록한다. 같은 키가 있으면 덮어쓴다."""
    status = "ok" if features is not None else "error"
    values = (
        [getattr(features, name) for name in _SCHEMA_COLUMNS]
        if features is not None
        else [None] * len(_SCHEMA_COLUMNS)
    )
    conn.execute(
        f"insert or replace into {CACHE_TABLE} "
        f"(article_id, model, prompt_version, created_at, status, error, raw_json, "
        f"{', '.join(_SCHEMA_COLUMNS)}) "
        f"values (?, ?, ?, ?, ?, ?, ?, {', '.join('?' * len(_SCHEMA_COLUMNS))})",
        (
            article_id,
            model,
            prompt_version,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status,
            error,
            raw,
            *values,
        ),
    )
    conn.commit()


def load_features_frame(
    conn: sqlite3.Connection, model: str, prompt_version: str
) -> pd.DataFrame:
    """성공 행을 모델 입력 가능한 DataFrame 으로 꺼낸다."""
    columns = ["article_id", *_SCHEMA_COLUMNS]
    rows = conn.execute(
        f"select {', '.join(columns)} from {CACHE_TABLE} "
        f"where model = ? and prompt_version = ? and status = 'ok' order by article_id",
        (model, prompt_version),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=columns)

    if frame.empty:
        for column in LLM_FEATURE_COLUMNS:
            frame[column] = pd.Series(dtype="float64")
        return frame

    encoded = pd.DataFrame(
        [
            encode_features(ArticleFeatures(*row[1:]))
            for row in rows
        ],
        index=frame.index,
    )
    return pd.concat([frame, encoded[LLM_FEATURE_COLUMNS]], axis=1)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_cache.py -v`
Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/score/cache.py tests/test_score_cache.py
git commit -m "feat(score): 결과 캐시 — (article_id, model, prompt_version) 키, 실패도 기록"
```

---

### Task 5: 배치 러너 — 재시도와 부분 실패

**Files:**
- Create: `src/swing_it/score/batch.py`
- Test: `tests/test_score_batch.py`

**Interfaces:**
- Consumes: `LLMProvider`, `ProviderError`, `build_prompt`, `PROMPT_VERSION`, `parse_features`, `SchemaError`, `get_cached`, `put_result`
- Produces:
  - `class BatchAborted(RuntimeError)`
  - `@dataclass(frozen=True) class BatchResult` — 필드 `n_input: int, n_cached: int, n_called: int, n_ok: int, n_failed: int, failures: list[tuple[str, str]]`
  - `score_articles(articles: pd.DataFrame, provider: LLMProvider, cache: sqlite3.Connection, *, prompt_version: str = PROMPT_VERSION, max_retries: int = 2, backoff: float = 1.0, limit: int | None = None, abort_after_consecutive: int = 20, sleep: Callable[[float], None] = time.sleep, on_progress: Callable[[int, int], None] | None = None) -> BatchResult`

**동작:**

- 캐시에 있는 기사는 API 를 부르지 않는다 (`n_cached`).
- `limit` 은 **API 호출 건수** 상한이다. 비용 사고를 막는 안전핀이라 캐시 히트는 세지 않는다.
- `ProviderError` 와 `SchemaError` 는 재시도 대상이다. 스키마 위반은 온도 0 에서도 가끔 나오며 재요청으로 대개 풀린다. `backoff * 2**attempt` 초를 쉰다.
- 최종 실패는 캐시에 `error` 로 기록하고 다음 기사로 넘어간다. **부분 실패로 배치를 죽이지 않는다.**
- 다만 연속 `abort_after_consecutive` 건이 실패하면 `BatchAborted` 를 던진다. 키 만료·모델 ID 오타로 수천 건을 헛돌리는 일을 막는다.
- `sleep` 을 주입 가능하게 둬서 테스트가 실제로 기다리지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_score_batch.py`:

```python
import pandas as pd
import pytest

from swing_it.score.batch import BatchAborted, score_articles
from swing_it.score.cache import get_cached, open_cache
from swing_it.score.prompt import PROMPT_VERSION
from swing_it.score.provider import FakeProvider, ProviderError

_OK = (
    '{"material_type":"수주","persistence":"multi_quarter","specificity":3,'
    '"freshness":"new","sentiment_direction":1,"sentiment_strength":2,'
    '"scale_vs_size":"medium"}'
)


def _articles(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(n)],
            "date": ["2026-01-05"] * n,
            "code": ["005930"] * n,
            "press": ["한국경제"] * n,
            "title": [f"삼성전자 공급 계약 {i}" for i in range(n)],
            "body200": ["2조원 규모 계약을 체결했다."] * n,
            "keywords": ["삼성전자,수주"] * n,
        }
    )


def _no_sleep(_seconds: float) -> None:
    return None


def test_all_fresh_articles_are_called_once(tmp_path):
    provider = FakeProvider(responses=[_OK] * 3)
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(3), provider, cache, sleep=_no_sleep)
    assert (result.n_input, result.n_cached, result.n_called, result.n_ok) == (3, 0, 3, 3)
    assert result.n_failed == 0
    assert len(provider.calls) == 3


def test_second_run_hits_the_cache(tmp_path):
    cache = open_cache(tmp_path / "llm.db")
    score_articles(_articles(3), FakeProvider(responses=[_OK] * 3), cache, sleep=_no_sleep)

    provider = FakeProvider(responses=[_OK] * 3)
    result = score_articles(_articles(3), provider, cache, sleep=_no_sleep)
    assert (result.n_cached, result.n_called) == (3, 0)
    assert provider.calls == []


def test_transient_failure_is_retried_then_succeeds(tmp_path):
    provider = FakeProvider(responses=[ProviderError("일시 오류"), _OK])
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(1), provider, cache, max_retries=2, sleep=_no_sleep)
    assert (result.n_ok, result.n_failed) == (1, 0)
    assert len(provider.calls) == 2


def test_schema_violation_is_retried(tmp_path):
    provider = FakeProvider(responses=['{"material_type":"미상"}', _OK])
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(1), provider, cache, max_retries=2, sleep=_no_sleep)
    assert result.n_ok == 1
    assert len(provider.calls) == 2


def test_permanent_failure_is_recorded_and_batch_continues(tmp_path):
    provider = FakeProvider(
        responses=[
            ProviderError("깨짐"), ProviderError("깨짐"), ProviderError("깨짐"),
            _OK, _OK,
        ]
    )
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(3), provider, cache, max_retries=2, sleep=_no_sleep)
    assert (result.n_ok, result.n_failed) == (2, 1)
    assert result.failures[0][0] == "a0"
    assert "깨짐" in result.failures[0][1]
    # 실패한 기사는 캐시 히트가 아니므로 다음 실행에서 재시도된다
    assert get_cached(cache, ["a0"], provider.model, PROMPT_VERSION) == {}
    assert set(get_cached(cache, ["a1", "a2"], provider.model, PROMPT_VERSION)) == {"a1", "a2"}


def test_backoff_grows_between_retries(tmp_path):
    slept: list[float] = []
    provider = FakeProvider(responses=[ProviderError("x"), ProviderError("x"), _OK])
    cache = open_cache(tmp_path / "llm.db")
    score_articles(
        _articles(1), provider, cache, max_retries=2, backoff=2.0, sleep=slept.append
    )
    assert slept == [2.0, 4.0]


def test_limit_caps_api_calls(tmp_path):
    provider = FakeProvider(responses=[_OK] * 10)
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(10), provider, cache, limit=4, sleep=_no_sleep)
    assert result.n_called == 4
    assert len(provider.calls) == 4


def test_consecutive_failures_abort_the_batch(tmp_path):
    provider = FakeProvider(responses=[ProviderError("키 만료")] * 100)
    cache = open_cache(tmp_path / "llm.db")
    with pytest.raises(BatchAborted, match="연속"):
        score_articles(
            _articles(50), provider, cache, max_retries=0,
            abort_after_consecutive=3, sleep=_no_sleep,
        )


def test_progress_callback_reports_each_article(tmp_path):
    seen: list[tuple[int, int]] = []
    cache = open_cache(tmp_path / "llm.db")
    score_articles(
        _articles(3), FakeProvider(responses=[_OK] * 3), cache,
        sleep=_no_sleep, on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_empty_input_is_a_no_op(tmp_path):
    provider = FakeProvider()
    cache = open_cache(tmp_path / "llm.db")
    result = score_articles(_articles(0), provider, cache, sleep=_no_sleep)
    assert (result.n_input, result.n_called, result.n_ok) == (0, 0, 0)
    assert provider.calls == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score.batch'`

- [ ] **Step 3: 구현**

`src/swing_it/score/batch.py`:

```python
"""배치 러너 — 캐시 스킵 · 재시도 · 부분 실패 집계.

수천 건 배치에서 일부 실패는 정상이다. 실패 하나로 전체를 죽이지 않고, 사유를
캐시에 남겨 다음 실행에서 자동 재시도되게 한다.

다만 연속 실패가 임계치를 넘으면 멈춘다. 키 만료나 모델 ID 오타로 수천 건을
헛돌리는 것과 진짜 부분 실패는 다른 일이다.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .cache import get_cached, put_result
from .prompt import PROMPT_VERSION, build_prompt
from .provider import LLMProvider, ProviderError
from .schema import SchemaError, parse_features


class BatchAborted(RuntimeError):
    """연속 실패가 임계치를 넘었다. 설정 문제일 가능성이 높다."""


@dataclass(frozen=True)
class BatchResult:
    n_input: int
    n_cached: int
    n_called: int
    n_ok: int
    n_failed: int
    failures: list[tuple[str, str]] = field(default_factory=list)


def score_articles(
    articles: pd.DataFrame,
    provider: LLMProvider,
    cache: sqlite3.Connection,
    *,
    prompt_version: str = PROMPT_VERSION,
    max_retries: int = 2,
    backoff: float = 1.0,
    limit: int | None = None,
    abort_after_consecutive: int = 20,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """기사 프레임을 배치로 태워 캐시에 채운다.

    Args:
        limit: **API 호출 건수** 상한. 비용 사고를 막는 안전핀이므로 캐시
            히트는 세지 않는다. ``None`` 이면 무제한.
        max_retries: 기사 한 건당 추가 시도 횟수. 총 시도는 ``max_retries + 1``.
        backoff: 재시도 대기 초. n 번째 재시도는 ``backoff * 2**n`` 초 쉰다.
        sleep: 주입 가능한 대기 함수. 테스트는 no-op 을 넘긴다.
    """
    total = len(articles)
    if total == 0:
        return BatchResult(0, 0, 0, 0, 0, [])

    ids = articles["article_id"].astype(str).tolist()
    cached = get_cached(cache, ids, provider.model, prompt_version)

    n_called = n_ok = n_failed = 0
    consecutive = 0
    failures: list[tuple[str, str]] = []

    for done, (_, row) in enumerate(articles.iterrows(), start=1):
        article_id = str(row["article_id"])
        if article_id in cached:
            if on_progress:
                on_progress(done, total)
            continue
        if limit is not None and n_called >= limit:
            break

        system, user = build_prompt(row.to_dict())
        last_error = ""
        raw: str | None = None
        for attempt in range(max_retries + 1):
            n_called += 1
            try:
                raw = provider.complete(system, user)
                features = parse_features(raw)
            except (ProviderError, SchemaError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    sleep(backoff * (2**attempt))
                continue
            put_result(
                cache,
                article_id=article_id,
                model=provider.model,
                prompt_version=prompt_version,
                features=features,
                raw=raw,
                error=None,
            )
            n_ok += 1
            consecutive = 0
            break
        else:
            put_result(
                cache,
                article_id=article_id,
                model=provider.model,
                prompt_version=prompt_version,
                features=None,
                raw=raw,
                error=last_error,
            )
            n_failed += 1
            failures.append((article_id, last_error))
            consecutive += 1
            if consecutive >= abort_after_consecutive:
                raise BatchAborted(
                    f"연속 {consecutive} 건 실패로 중단한다. 마지막 사유: {last_error}"
                )

        if on_progress:
            on_progress(done, total)

    return BatchResult(
        n_input=total,
        n_cached=len(cached),
        n_called=n_called,
        n_ok=n_ok,
        n_failed=n_failed,
        failures=failures,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_batch.py -v`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/score/batch.py tests/test_score_batch.py
git commit -m "feat(score): 배치 러너 — 캐시 스킵·지수 백오프 재시도·부분 실패 집계"
```

---

### Task 6: 뉴스 어댑터와 `swing-score` CLI

**Files:**
- Modify: `pyproject.toml` — `[project.scripts]` 에 `swing-score` 추가
- Create: `src/swing_it/adapters/news_db.py`
- Create: `src/swing_it/score/cli.py`
- Test: `tests/test_news_db_adapter.py`
- Test: `tests/test_score_cli.py`

**Interfaces:**
- Consumes: `connect`/`load_prices`(계획 1 `adapters/db.py`), `extract_events`(계획 1 `events.py`), `prefilter_articles`, `prefilter_summary`, `score_articles`, `provider_from_env`, `open_cache`, `load_features_frame`
- Produces:
  - `NEWS_COLUMNS: list[str]` = `["article_id", "date", "code", "press", "title", "body200", "keywords"]`
  - `connect_news(path: str | None = None) -> sqlite3.Connection` — 생략 시 `SWING_NEWS_DB`, 그것도 없으면 `data/news.db`
  - `load_articles(conn, start: str, end: str) -> pd.DataFrame` — 컬럼 `NEWS_COLUMNS`, `date` 는 `datetime64[ns]`
  - `main(argv: list[str] | None = None) -> int` — `swing-score` 진입점

**CLI 인자:**

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--news-db` | `SWING_NEWS_DB` 또는 `data/news.db` | 계획 2 산출물 |
| `--cache-db` | `data/llm_features.db` | LLM 결과 캐시 |
| `--db` | `KR_QUANT_DB` | 가격 DSN (거래대금 이벤트용) |
| `--start` / `--end` | `2016-09-09` / `2026-08-14` | 기간 |
| `--provider` | 미지정 시 `SWING_LLM_PROVIDER` | `fake` 로 두면 네트워크 없이 배선 확인 |
| `--limit` | `None` | **API 호출 상한.** 첫 실행에서는 반드시 작게 준다 |
| `--min-chars` | `30` | 사전 필터 |
| `--event-window` | `20` | 거래대금 평균 구간 |
| `--tv-multiple` / `--adv-floor` | `3.0` / `5000.0` | 계획 1 과 같은 값 |
| `--out` | `results/score_run.json` | 리포트. **집계 수치만 쓴다** |

- [ ] **Step 1: `pyproject.toml` 스크립트 추가**

`[project.scripts]` 를 아래로 바꾼다:

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-score = "swing_it.score.cli:main"
```

- [ ] **Step 2: 실패하는 어댑터 테스트 작성**

`tests/test_news_db_adapter.py`:

```python
import sqlite3

import pandas as pd
import pytest

from swing_it.adapters.news_db import NEWS_COLUMNS, connect_news, load_articles


@pytest.fixture()
def news_db(tmp_path):
    """계획 2 가 만드는 스키마 그대로 — 정규화된 articles + article_codes."""
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE articles (
            news_id  TEXT PRIMARY KEY,
            date     TEXT NOT NULL,
            outlet   TEXT,
            title    TEXT,
            body     TEXT,
            keywords TEXT,
            orgs     TEXT,
            url      TEXT
        );
        CREATE TABLE article_codes (
            news_id    TEXT NOT NULL,
            code       TEXT NOT NULL,
            match_kind TEXT NOT NULL,
            PRIMARY KEY (news_id, code)
        );
        INSERT INTO articles VALUES
            ('bk-1','2026-01-05','한국경제','삼성전자 HBM 공급 계약',
             '2조원 규모 계약을 체결했다.','삼성전자,HBM','삼성전자','http://x/1'),
            ('bk-2','2026-01-06','매일경제','SK하이닉스 증설 발표',
             '청주 M15X 증설을 결정했다.','SK하이닉스,증설','SK하이닉스','http://x/2'),
            ('bk-3','2026-02-01','더벨','삼성전자 후속 보도',
             '앞선 계약의 진행 상황이다.','삼성전자','삼성전자','http://x/3');
        INSERT INTO article_codes VALUES
            ('bk-1','005930','name'),
            ('bk-2','000660','name'),
            ('bk-3','005930','name');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_load_articles_returns_contract_columns(news_db):
    frame = load_articles(connect_news(news_db), "2026-01-01", "2026-12-31")
    assert list(frame.columns) == NEWS_COLUMNS
    assert len(frame) == 3
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_load_articles_respects_date_range(news_db):
    frame = load_articles(connect_news(news_db), "2026-01-01", "2026-01-31")
    assert list(frame["article_id"]) == ["bk-1", "bk-2"]


def test_connect_news_uses_env(news_db, monkeypatch):
    monkeypatch.setenv("SWING_NEWS_DB", news_db)
    frame = load_articles(connect_news(), "2026-01-01", "2026-12-31")
    assert len(frame) == 3
```

- [ ] **Step 3: 실패하는 CLI 테스트 작성**

`tests/test_score_cli.py`:

```python
import json
import sqlite3

import pytest

from swing_it.score.cli import main


@pytest.fixture()
def workspace(tmp_path):
    """가격 DB · 뉴스 DB 를 함께 깐다. 어느 쪽도 네트워크를 쓰지 않는다."""
    price_path = tmp_path / "prices.db"
    conn = sqlite3.connect(price_path)
    conn.executescript(
        """
        CREATE TABLE daily_bars_adjusted (
            code TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, trade_value REAL, source TEXT
        );
        """
    )
    rows = []
    for code in ("005930", "000660"):
        for day in range(1, 6):
            # 5일차만 거래대금 4배로 튄다 → window=3 에서 이벤트가 된다
            trade_value = 24000.0 if day == 5 else 6000.0
            rows.append((code, f"2026-01-0{day}", 100.0, 110.0, 90.0, 105.0,
                         1000, trade_value, "test"))
    conn.executemany(
        "insert into daily_bars_adjusted values (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    news_path = tmp_path / "news.db"
    conn = sqlite3.connect(news_path)
    conn.executescript(
        """
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY, date TEXT, code TEXT, press TEXT,
            title TEXT, body200 TEXT, keywords TEXT
        );
        INSERT INTO articles VALUES
            ('n1','2026-01-05','005930','한국경제','삼성전자 HBM 대규모 공급 계약 체결',
             '삼성전자가 미국 A사와 3년간 2조원 규모의 공급 계약을 맺었다.','삼성전자,HBM'),
            ('n2','2026-01-05','000660','매일경제','SK하이닉스 청주 증설 결정 발표',
             'SK하이닉스가 청주 M15X 라인 증설을 이사회에서 결정했다.','SK하이닉스,증설'),
            ('n3','2026-01-05','','더벨','종목 매핑에 실패한 기사 제목입니다',
             '어느 종목인지 특정되지 않은 시황 기사 본문이다.','시황'),
            ('n4','2026-01-02','005930','서울경제','거래대금이 튀지 않은 날의 기사 제목',
             '평범한 날의 평범한 기사 본문이 여기에 들어간다.','삼성전자');
        """
    )
    conn.commit()
    conn.close()
    return {"prices": str(price_path), "news": str(news_path), "root": tmp_path}


def _run(workspace, *extra):
    out = workspace["root"] / "report.json"
    code = main(
        [
            "--db", workspace["prices"],
            "--news-db", workspace["news"],
            "--cache-db", str(workspace["root"] / "llm.db"),
            "--start", "2026-01-01",
            "--end", "2026-01-31",
            "--provider", "fake",
            "--event-window", "3",
            "--out", str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_cli_runs_end_to_end_with_fake_provider(workspace):
    code, report = _run(workspace)
    assert code == 0
    assert report["prefilter"]["n_input"] == 4
    assert report["prefilter"]["no_code"] == 1
    assert report["prefilter"]["no_volume_event"] == 1
    assert report["prefilter"]["n_kept"] == 2
    assert report["batch"]["n_ok"] == 2
    assert report["batch"]["n_failed"] == 0


def test_cli_records_model_and_prompt_version(workspace):
    _, report = _run(workspace)
    assert report["model"] == "fake-1"
    assert report["prompt_version"]
    assert report["provider"] == "fake"


def test_cli_second_run_is_all_cache(workspace):
    _run(workspace)
    _, report = _run(workspace)
    assert report["batch"]["n_called"] == 0
    assert report["batch"]["n_cached"] == 2


def test_cli_limit_caps_calls(workspace):
    _, report = _run(workspace, "--limit", "1")
    assert report["batch"]["n_called"] == 1


def test_report_contains_no_article_text(workspace):
    """results/ 는 커밋될 수 있다. 기사 본문이 새면 안 된다."""
    _, report = _run(workspace)
    blob = json.dumps(report, ensure_ascii=False)
    assert "삼성전자가 미국 A사와" not in blob
    assert "n1" not in blob
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_db_adapter.py tests/test_score_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.adapters.news_db'`

- [ ] **Step 5: 뉴스 어댑터 구현**

`src/swing_it/adapters/news_db.py`:

```python
"""뉴스 DB 읽기 어댑터 — 뉴스 SQL 은 이 파일에만 존재한다.

입력은 계획 2 가 만드는 로컬 sqlite ``data/news.db`` 다. 계획 2 는 기사를
``articles`` 에, 기사→종목 매핑을 ``article_codes`` 에 정규화해 넣는다(기사 1건이
여러 종목에 걸릴 수 있기 때문이다). 이 어댑터가 조인과 컬럼명 차이를 흡수하고,
나머지 모듈은 평면화된 ``NEWS_COLUMNS`` 만 안다.

계획 2 컬럼 → 이 계약: ``news_id``→``article_id``, ``outlet``→``press``,
``body``→``body200``(계획 2 가 이미 200자로 잘라 넣는다), ``article_codes.code``→``code``.
같은 기사가 두 종목에 걸리면 종목마다 한 행이 된다 — 스코어링 단위가 (기사, 종목) 이므로 의도된 동작이다.

읽기 전용이다. swing-it 의 score 계층은 news.db 에 쓰지 않는다.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

NEWS_COLUMNS: list[str] = [
    "article_id", "date", "code", "press", "title", "body200", "keywords",
]

_DEFAULT_PATH = "data/news.db"


def connect_news(path: str | None = None) -> sqlite3.Connection:
    """뉴스 DB 를 연다. 생략 시 ``SWING_NEWS_DB``, 그것도 없으면 ``data/news.db``."""
    target = path or os.environ.get("SWING_NEWS_DB") or _DEFAULT_PATH
    if not Path(target).exists():
        raise FileNotFoundError(
            f"뉴스 DB 가 없다: {target}. 계획 2(뉴스 인제스트)를 먼저 돌려야 한다."
        )
    return sqlite3.connect(target)


_SELECT_ARTICLES = """
select a.news_id, a.date, ac.code, a.outlet, a.title, a.body, a.keywords
from articles a
join article_codes ac on ac.news_id = a.news_id
where a.date >= ? and a.date <= ?
order by a.date, a.news_id, ac.code
"""


def load_articles(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    """기간 내 (기사, 종목) 을 컬럼 계약에 맞춰 읽는다.

    계획 2 의 정규화 스키마를 조인해 평면화한다. 종목 매핑이 없는 기사는
    조인에서 빠진다 — 스코어링 단위가 (기사, 종목) 이므로 종목 없는 기사는
    애초에 태울 대상이 아니다.
    """
    rows = conn.execute(_SELECT_ARTICLES, (start, end)).fetchall()
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("article_id", "code", "press", "title", "body200", "keywords"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame
```

- [ ] **Step 6: CLI 구현**

`src/swing_it/score/cli.py`:

```python
"""`swing-score` — 뉴스 테이블을 사전 필터에 걸고 LLM 배치로 태운다.

``--provider fake`` 로 두면 네트워크 없이 배선 전체를 확인할 수 있다.
첫 실 호출은 반드시 ``--limit`` 을 작게 잡고 시작한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..adapters.db import connect, load_prices
from ..adapters.news_db import connect_news, load_articles
from ..events import extract_events
from .batch import score_articles
from .cache import open_cache
from .prefilter import prefilter_articles, prefilter_summary
from .prompt import PROMPT_VERSION
from .provider import provider_from_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="뉴스 기사 → LLM 고정 스키마 피처")
    parser.add_argument("--news-db", default=None, help="생략 시 SWING_NEWS_DB 또는 data/news.db")
    parser.add_argument("--cache-db", default="data/llm_features.db")
    parser.add_argument("--db", default=None, help="가격 DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--provider", default=None, help="fake / gemini / openai / anthropic")
    parser.add_argument("--model", default=None, help="생략 시 SWING_LLM_MODEL 또는 기본 모델")
    parser.add_argument("--limit", type=int, default=None, help="API 호출 상한")
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--event-window", type=int, default=20)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--out", default="results/score_run.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    env = dict(os.environ)
    if args.provider:
        env["SWING_LLM_PROVIDER"] = args.provider
    if args.model:
        env["SWING_LLM_MODEL"] = args.model
    provider = provider_from_env(env)

    articles = load_articles(connect_news(args.news_db), args.start, args.end)
    prices = load_prices(connect(args.db), args.start, args.end)
    events = extract_events(
        prices,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        window=args.event_window,
    )
    print(f"기사 {len(articles):,} 건 · 이벤트 {len(events):,} 건", file=sys.stderr)

    filtered = prefilter_articles(articles, events, min_chars=args.min_chars)
    summary = prefilter_summary(filtered)
    kept = filtered.loc[filtered["keep"]].reset_index(drop=True)
    print(f"사전 필터 통과 {summary['n_kept']:,} / {summary['n_input']:,}", file=sys.stderr)

    cache = open_cache(args.cache_db)
    result = score_articles(
        kept,
        provider,
        cache,
        max_retries=args.max_retries,
        limit=args.limit,
        on_progress=lambda done, total: (
            print(f"  {done}/{total}", file=sys.stderr) if done % 100 == 0 else None
        ),
    )

    # results/ 는 커밋될 수 있다. 기사 텍스트·article_id 를 넣지 않는다.
    report = {
        "provider": provider.name,
        "model": provider.model,
        "prompt_version": PROMPT_VERSION,
        "date_range": [args.start, args.end],
        "prefilter": summary,
        "batch": {
            "n_input": result.n_input,
            "n_cached": result.n_cached,
            "n_called": result.n_called,
            "n_ok": result.n_ok,
            "n_failed": result.n_failed,
        },
        "failure_reasons": sorted({reason for _, reason in result.failures})[:10],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_db_adapter.py tests/test_score_cli.py -v`
Expected: 8 passed

- [ ] **Step 8: 전체 테스트와 린트**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: 전부 통과 (계획 1 의 27건 + Task 1~6 의 68건 = 95건)

- [ ] **Step 9: fake provider 로 실 데이터 스모크**

Run:
```bash
export $(grep -v '^#' ../kr-quant/.env | xargs)
uv run swing-score --provider fake --out results/score_smoke.json
```
Expected: 네트워크 호출 0 회. `prefilter` 카운트가 나온다.

**여기서 볼 것은 `n_kept` 다.** 이 숫자가 곧 LLM 비용의 분모다. 스펙 §5 의 수집
범위가 일 30~50건이므로, 3년이면 대략 3~5만 건이 상한이다. `n_kept` 가 그보다
훨씬 크면 사전 필터 임계값(`--tv-multiple`, `--min-chars`)을 조여야 한다.

- [ ] **Step 10: 커밋**

```bash
git add pyproject.toml src/swing_it/adapters/news_db.py src/swing_it/score/cli.py \
        tests/test_news_db_adapter.py tests/test_score_cli.py
git commit -m "feat(score): 뉴스 어댑터 + swing-score CLI (fake provider 로 오프라인 스모크)"
```

---

### Task 7: 모델 대조 — 상위 모델과 저가 모델의 피처 품질 비교

**Files:**
- Modify: `pyproject.toml` — `[project.scripts]` 에 `swing-score-compare` 추가
- Create: `src/swing_it/score/compare.py`
- Test: `tests/test_score_compare.py`

**Interfaces:**
- Consumes: `load_features_frame` 출력 프레임, `open_cache`, `connect_news`/`load_articles`, `prefilter_articles`, `score_articles`, `HttpProvider`/`provider_from_env`
- Produces:
  - `CATEGORICAL_FIELDS: list[str]` = `["material_type", "persistence", "freshness", "scale_vs_size"]`
  - `ORDINAL_FIELDS: list[str]` = `["specificity", "sentiment_direction", "sentiment_strength"]`
  - `compare_frames(baseline: pd.DataFrame, challenger: pd.DataFrame) -> dict` — 키 `n_common`, `agreement`(범주형 필드별 일치율), `ordinal`(필드별 `{"mae": float, "spearman": float}`), `disagreement_examples`(article_id 최대 10개)
  - `main(argv: list[str] | None = None) -> int` — `swing-score-compare` 진입점

**왜 자동 PASS/FAIL 을 두지 않는가.** 이 비교는 "싼 모델로 확정해도 되는가"를
사람이 판단하기 위한 재료다. 진짜 판정은 계획 4 의 층별 비교(②규칙뉴스 vs ③LLM)와
kr-quant 게이트에서 난다. 여기서 임의의 임계선을 세우면 그 선이 근거 없이
설계 결정을 대신하게 된다.

**표본 규모.** 스펙 §8 은 "표본 수백 건"이라고 쓴다. 기본 `--sample 300`.
날짜 전 구간에서 균등 추출해 시기 편향을 피한다.

- [ ] **Step 1: 착수 시점 단가·모델 ID 확인**

이 스텝은 코드가 아니라 확인 작업이다. 아래를 실제로 열어 보고 결과를
`docs/superpowers/plans/2026-08-17-llm-features.md` 하단 "확인 로그"에 적는다.

1. Gemini 가격·모델 목록 — https://ai.google.dev/gemini-api/docs/pricing
2. Anthropic 가격·모델 목록 — https://docs.claude.com/en/docs/about-claude/models
3. OpenAI 가격 — https://platform.openai.com/docs/pricing
4. 각 provider 의 **배치 API 할인** 적용 여부와 최소 대기 시간

확인할 것:

- `provider.py` 의 `DEFAULT_MODELS` 에 적힌 ID 가 아직 서빙되는가. 아니면 갱신한다.
- 최저가 등급이 여전히 Flash-Lite 급인가. 더 싼 등급이 생겼으면 그쪽으로 바꾼다.
- Task 6 Step 9 에서 나온 `n_kept` × (입력 토큰 ≈ 700, 출력 토큰 ≈ 120) 으로
  전체 배치 비용을 추정한다. **이 추정치가 감당 범위를 넘으면 사전 필터를 더 조인다.**
- 배치 API 가 50% 할인이고 대기가 24시간 이내면, 과거 기사 인제스트는 배치로
  돌릴 값이 있다. 이 경우는 이 계획 범위 밖의 후속 작업으로 기록만 한다
  (운영 경로는 실시간이어야 하므로 동기 호출을 유지한다).

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_score_compare.py`:

```python
import numpy as np
import pandas as pd

from swing_it.score.compare import CATEGORICAL_FIELDS, ORDINAL_FIELDS, compare_frames


def _frame(article_ids, materials=None, specificities=None, sentiments=None):
    n = len(article_ids)
    return pd.DataFrame(
        {
            "article_id": article_ids,
            "material_type": materials or ["수주"] * n,
            "persistence": ["multi_quarter"] * n,
            "specificity": specificities or [2] * n,
            "freshness": ["new"] * n,
            "sentiment_direction": [1] * n,
            "sentiment_strength": sentiments or [2] * n,
            "scale_vs_size": ["medium"] * n,
        }
    )


def test_identical_frames_agree_completely():
    ids = ["a1", "a2", "a3", "a4"]
    report = compare_frames(_frame(ids), _frame(ids))
    assert report["n_common"] == 4
    for column in CATEGORICAL_FIELDS:
        assert report["agreement"][column] == 1.0
    for column in ORDINAL_FIELDS:
        assert report["ordinal"][column]["mae"] == 0.0
    assert report["disagreement_examples"] == []


def test_half_disagreement_gives_half_agreement():
    ids = ["a1", "a2", "a3", "a4"]
    baseline = _frame(ids)
    challenger = _frame(ids, materials=["수주", "수주", "실적", "정책"])
    report = compare_frames(baseline, challenger)
    assert report["agreement"]["material_type"] == 0.5
    assert report["agreement"]["freshness"] == 1.0


def test_only_common_article_ids_are_compared():
    report = compare_frames(_frame(["a1", "a2", "a3"]), _frame(["a2", "a3", "a9"]))
    assert report["n_common"] == 2


def test_ordinal_mae_is_mean_absolute_difference():
    ids = ["a1", "a2", "a3", "a4"]
    baseline = _frame(ids, specificities=[0, 1, 2, 3])
    challenger = _frame(ids, specificities=[1, 1, 2, 1])
    # 차이 1, 0, 0, 2 → 평균 0.75
    report = compare_frames(baseline, challenger)
    assert np.isclose(report["ordinal"]["specificity"]["mae"], 0.75)


def test_ordinal_spearman_detects_rank_agreement():
    ids = [f"a{i}" for i in range(4)]
    baseline = _frame(ids, sentiments=[0, 1, 2, 3])
    challenger = _frame(ids, sentiments=[1, 1, 2, 3])
    report = compare_frames(baseline, challenger)
    assert report["ordinal"]["sentiment_strength"]["spearman"] > 0.8


def test_constant_column_reports_nan_spearman_not_a_crash():
    ids = [f"a{i}" for i in range(4)]
    report = compare_frames(_frame(ids), _frame(ids))
    assert np.isnan(report["ordinal"]["sentiment_direction"]["spearman"])


def test_disagreement_examples_are_capped_at_ten():
    ids = [f"a{i}" for i in range(30)]
    baseline = _frame(ids)
    challenger = _frame(ids, materials=["실적"] * 30)
    report = compare_frames(baseline, challenger)
    assert len(report["disagreement_examples"]) == 10


def test_no_common_ids_returns_zero_without_crashing():
    report = compare_frames(_frame(["a1"]), _frame(["b1"]))
    assert report["n_common"] == 0
    assert np.isnan(report["agreement"]["material_type"])
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/test_score_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.score.compare'`

- [ ] **Step 4: 비교 로직 구현**

`src/swing_it/score/compare.py` (파일 앞부분 — 순수 비교 함수):

```python
"""저가 모델과 상위 모델의 피처 품질 대조 (스펙 §8).

"표본 수백 건만 상위 모델로 돌려 피처 품질을 대조한다. 차이가 없으면 싼 것으로
확정한다." 자동 PASS/FAIL 을 두지 않는 이유는, 이 비교가 사람의 모델 선택을 돕는
재료일 뿐이기 때문이다. 진짜 판정은 계획 4 의 층별 비교와 kr-quant 게이트에서 난다.

두 모델이 크게 갈리면 그것 자체가 정보다 — 스키마 문항이 모호하다는 뜻이므로
프롬프트를 고치고 PROMPT_VERSION 을 올린다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_FIELDS: list[str] = [
    "material_type", "persistence", "freshness", "scale_vs_size",
]
ORDINAL_FIELDS: list[str] = [
    "specificity", "sentiment_direction", "sentiment_strength",
]

_MAX_EXAMPLES = 10


def compare_frames(baseline: pd.DataFrame, challenger: pd.DataFrame) -> dict:
    """공통 ``article_id`` 에 대해 필드별 일치도를 잰다.

    범주형은 일치율, 순서형은 평균절대차(MAE)와 순위상관(Spearman)을 쓴다.
    한쪽 컬럼이 상수면 상관이 정의되지 않으므로 ``nan`` 이 나온다 — 정상이다.
    """
    merged = baseline.merge(challenger, on="article_id", suffixes=("_a", "_b"))
    n_common = int(len(merged))

    agreement: dict[str, float] = {}
    for column in CATEGORICAL_FIELDS:
        if n_common == 0:
            agreement[column] = float("nan")
        else:
            agreement[column] = float((merged[f"{column}_a"] == merged[f"{column}_b"]).mean())

    ordinal: dict[str, dict[str, float]] = {}
    for column in ORDINAL_FIELDS:
        if n_common == 0:
            ordinal[column] = {"mae": float("nan"), "spearman": float("nan")}
            continue
        left = pd.to_numeric(merged[f"{column}_a"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_b"], errors="coerce")
        with np.errstate(invalid="ignore"):
            spearman = left.corr(right, method="spearman")
        ordinal[column] = {
            "mae": float((left - right).abs().mean()),
            "spearman": float(spearman) if spearman is not None else float("nan"),
        }

    disagreed = pd.Series(False, index=merged.index)
    for column in CATEGORICAL_FIELDS:
        disagreed |= merged[f"{column}_a"] != merged[f"{column}_b"]
    examples = merged.loc[disagreed, "article_id"].head(_MAX_EXAMPLES).tolist()

    return {
        "n_common": n_common,
        "agreement": agreement,
        "ordinal": ordinal,
        "disagreement_examples": examples,
    }
```

`src/swing_it/score/compare.py` (파일 뒷부분 — CLI):

```python
def _sample_articles(kept: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """날짜 전 구간에서 균등하게 뽑는다. 특정 시기에 몰리면 비교가 왜곡된다."""
    if len(kept) <= size:
        return kept.reset_index(drop=True)
    ordered = kept.sort_values("date").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, size).round().astype(int)
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-1, 2, size=size)
    positions = np.clip(positions + jitter, 0, len(ordered) - 1)
    return ordered.iloc[np.unique(positions)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    from ..adapters.db import connect, load_prices
    from ..adapters.news_db import connect_news, load_articles
    from ..events import extract_events
    from .batch import score_articles
    from .cache import load_features_frame, open_cache
    from .prefilter import prefilter_articles
    from .prompt import PROMPT_VERSION
    from .provider import provider_from_env

    parser = argparse.ArgumentParser(description="두 모델의 피처 품질 대조")
    parser.add_argument("--baseline-provider", default="gemini")
    parser.add_argument("--baseline-model", default=None, help="생략 시 provider 기본 모델")
    parser.add_argument("--challenger-provider", required=True)
    parser.add_argument("--challenger-model", required=True)
    parser.add_argument("--news-db", default=None)
    parser.add_argument("--cache-db", default="data/llm_features.db")
    parser.add_argument("--db", default=None)
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=20)
    parser.add_argument("--out", default="results/model_compare.json")
    args = parser.parse_args(argv)

    articles = load_articles(connect_news(args.news_db), args.start, args.end)
    prices = load_prices(connect(args.db), args.start, args.end)
    events = extract_events(prices, window=args.event_window)
    filtered = prefilter_articles(articles, events)
    sample = _sample_articles(filtered.loc[filtered["keep"]], args.sample, args.seed)
    print(f"대조 표본 {len(sample):,} 건", file=sys.stderr)

    cache = open_cache(args.cache_db)
    frames: dict[str, pd.DataFrame] = {}
    models: dict[str, str] = {}
    for role, provider_name, model_name in (
        ("baseline", args.baseline_provider, args.baseline_model),
        ("challenger", args.challenger_provider, args.challenger_model),
    ):
        env = dict(os.environ, SWING_LLM_PROVIDER=provider_name)
        if model_name:
            env["SWING_LLM_MODEL"] = model_name
        provider = provider_from_env(env)
        result = score_articles(sample, provider, cache)
        print(
            f"{role} {provider.model}: ok {result.n_ok} / 캐시 {result.n_cached} "
            f"/ 실패 {result.n_failed}",
            file=sys.stderr,
        )
        full = load_features_frame(cache, provider.model, PROMPT_VERSION)
        frames[role] = full[full["article_id"].isin(sample["article_id"])]
        models[role] = provider.model

    report = compare_frames(frames["baseline"], frames["challenger"])
    report["baseline_model"] = models["baseline"]
    report["challenger_model"] = models["challenger"]
    report["prompt_version"] = PROMPT_VERSION
    # 비교 표본의 article_id 는 리포트에 남기지 않는다 (results/ 는 커밋될 수 있다)
    report["disagreement_examples"] = len(report["disagreement_examples"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: `pyproject.toml` 스크립트 추가**

`[project.scripts]` 를 아래로 바꾼다:

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-score = "swing_it.score.cli:main"
swing-score-compare = "swing_it.score.compare:main"
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `uv run pytest tests/test_score_compare.py -v`
Expected: 8 passed

- [ ] **Step 7: 전체 테스트와 린트**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: 전부 통과

- [ ] **Step 8: 소량 실 호출 — 저가 모델 30건**

여기서 처음으로 실제 API 를 부른다. **반드시 `--limit` 을 걸고 시작한다.**

```bash
export $(grep -v '^#' .env | xargs)   # GEMINI_API_KEY 등
uv run swing-score --limit 30 --out results/score_pilot.json
```

Expected: `n_ok` 가 25 이상. 실패가 절반을 넘으면 멈추고 원인을 본다 —
`failure_reasons` 에 `SchemaError` 가 몰려 있으면 프롬프트 문제이고,
`ProviderError` 가 몰려 있으면 키·모델 ID·레이트리밋 문제다.

`SchemaError` 가 많으면 프롬프트를 고치고 **`PROMPT_VERSION` 을 `v2` 로 올린 뒤**
다시 돌린다. 버전을 안 올리면 옛 결과가 캐시 히트로 남아 섞인다.

- [ ] **Step 9: 저가 모델 vs 상위 모델 대조 300건**

```bash
uv run swing-score-compare \
  --baseline-provider gemini \
  --challenger-provider anthropic \
  --challenger-model "$SWING_COMPARE_MODEL" \
  --sample 300 \
  --out results/model_compare.json
```

Expected: `results/model_compare.json` 생성.

**읽는 법:**

- 범주형 일치율이 전반적으로 0.8 이상이면 저가 모델로 확정할 근거가 된다.
- `material_type` 만 낮으면 유형 분류 경계가 모호한 것이다 — 프롬프트에서
  유형 정의를 조이거나 유형 수를 줄인다.
- `specificity` MAE 가 1.0 을 넘으면 척도 정의가 모호한 것이다.
- **어느 쪽이 '맞다'고 말할 수 없다.** 정답 라벨이 없다. 여기서 나오는 것은
  일치도뿐이고, 어느 모델이 실제로 쓸모 있는지는 계획 4 의 층별 비교에서
  D+20~40 초과수익과 대조해야 나온다.

- [ ] **Step 10: 확인 로그를 계획 문서에 기록**

Step 1 과 Step 9 의 결과를 이 문서 맨 아래 "확인 로그" 절에 날짜와 함께 적는다.
확정한 모델 ID 를 `provider.py` 의 `DEFAULT_MODELS` 에 반영한다.

- [ ] **Step 11: 커밋**

```bash
git add pyproject.toml src/swing_it/score/compare.py tests/test_score_compare.py \
        docs/superpowers/plans/2026-08-17-llm-features.md
git commit -m "feat(score): 모델 대조 스크립트 + 단가·모델 ID 확인 로그"
```

---

### Task 8: `(code, date)` 격자 어댑터 — 계획 4 인계 지점

**Files:**
- Create: `src/swing_it/features/llm.py`
- Test: `tests/test_features_llm.py`

**Interfaces:**
- Consumes: `load_features_frame`, `CACHE_TABLE`, `LLM_FEATURE_COLUMNS`, `PROMPT_VERSION`, `connect_news`, `load_articles`
- Produces:
  - `LLM_FEATURE_COLUMNS: list[str]` — `score.schema` 에서 **재수출**한다 (재정의 금지)
  - `LLM_FRAME_COLUMNS: list[str]` = `["code", "date", "material_type", "n_articles", *LLM_FEATURE_COLUMNS]`
  - `available_models(conn: sqlite3.Connection) -> list[tuple[str, str, int]]` — `(model, prompt_version, n_ok)`, 건수 내림차순
  - `fold_to_code_date(scores: pd.DataFrame) -> pd.DataFrame` — 입력 `[article_id, code, date, material_type, *LLM_FEATURE_COLUMNS]` → 출력 `LLM_FRAME_COLUMNS`
  - `load_llm_features(cache_path: str | Path = "data/llm_features.db", *, news_path: str | None = None, model: str | None = None, prompt_version: str | None = None, articles: pd.DataFrame | None = None) -> pd.DataFrame`

**왜 이 태스크가 계획 3 에 있는가.** 계획 3 의 캐시는 **기사 단위**(`article_id`)이고
모델은 **`(code, date)` 단위**로 학습한다. 그 사이를 잇는 접기 규칙은 캐시 내부
구조와 `article_id → (code, date)` 조인을 둘 다 알아야 해서, 계획 4 가 아니라
여기가 맞는 자리다. 계획 4 는 `load_llm_features` 하나만 부르면 된다.

**접기 규칙 — 대표 기사 한 건의 행을 통째로 쓴다.** 컬럼별로 `max` 를 따로
집계하면 A기사의 구체성과 B기사의 감정이 섞인, 실재하지 않는 기사가 만들어진다.
대신 그날 **가장 강한 재료 한 건**을 골라 그 행을 그대로 쓴다:

1. `|llm_sentiment|` 최대
2. 동률이면 `llm_specificity` 최대
3. 그래도 동률이면 `article_id` 사전순 최소 (재현성)

`n_articles` 는 그 `(code, date)` 의 전체 기사 수다. **계획 2 가 '기사 건수' 피처를
이미 준다면 계획 4 에서 이 컬럼을 버려라** — 같은 것을 두 번 넣는 셈이다.

**`material_type` 은 수치화하지 않는다.** 9개 값을 원-핫으로 펼치면 LLM 몫이 5개에서
14개로 뛰어 스펙 §6 의 10~20개 상한을 혼자 먹는다. `LLM_FEATURE_COLUMNS` 5개만
모델 행렬에 넣고, `material_type` 은 한국어 문자열 그대로 **근거 카드 표시용**으로
딸려 보낸다. 계획 4 가 굳이 모델에 넣겠다면 walk-forward 폴드 안에서 타깃 인코딩을
해야 하고(폴드 밖 정보가 새면 안 된다), 그 판단은 계획 4 소관이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_features_llm.py`:

```python
import sqlite3

import pandas as pd
import pytest

from swing_it.features.llm import (
    LLM_FEATURE_COLUMNS,
    LLM_FRAME_COLUMNS,
    available_models,
    fold_to_code_date,
    load_llm_features,
)
from swing_it.score.cache import open_cache, put_result
from swing_it.score.schema import ArticleFeatures


def _scores(rows: list[dict]) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "code": "005930",
        "date": pd.Timestamp("2026-01-05"),
        "material_type": "수주",
        "llm_persistence": 1.0,
        "llm_specificity": 2.0,
        "llm_freshness": 2.0,
        "llm_sentiment": 2.0,
        "llm_scale": 2.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_fold_output_contract():
    out = fold_to_code_date(_scores([{}]))
    assert list(out.columns) == LLM_FRAME_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    for column in LLM_FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(out[column])


def test_single_article_passes_through():
    out = fold_to_code_date(_scores([{}]))
    assert len(out) == 1
    assert out.iloc[0]["n_articles"] == 1
    assert out.iloc[0]["llm_sentiment"] == 2.0
    assert out.iloc[0]["material_type"] == "수주"


def test_strongest_absolute_sentiment_wins():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 1.0, "material_type": "테마"},
                {"article_id": "a2", "llm_sentiment": -3.0, "material_type": "루머"},
                {"article_id": "a3", "llm_sentiment": 2.0, "material_type": "수주"},
            ]
        )
    )
    assert len(out) == 1
    assert out.iloc[0]["llm_sentiment"] == -3.0
    assert out.iloc[0]["material_type"] == "루머"
    assert out.iloc[0]["n_articles"] == 3


def test_specificity_breaks_sentiment_ties():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 2.0, "llm_specificity": 1.0},
                {"article_id": "a2", "llm_sentiment": 2.0, "llm_specificity": 3.0},
            ]
        )
    )
    assert out.iloc[0]["llm_specificity"] == 3.0


def test_article_id_breaks_remaining_ties_deterministically():
    rows = [
        {"article_id": "z9", "material_type": "정책"},
        {"article_id": "a1", "material_type": "증설"},
    ]
    assert fold_to_code_date(_scores(rows)).iloc[0]["material_type"] == "증설"
    assert fold_to_code_date(_scores(rows[::-1])).iloc[0]["material_type"] == "증설"


def test_fold_keeps_the_whole_row_not_a_column_wise_max():
    """컬럼별 max 를 섞으면 실재하지 않는 기사가 만들어진다."""
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "llm_sentiment": 3.0, "llm_scale": 0.0},
                {"article_id": "a2", "llm_sentiment": 1.0, "llm_scale": 3.0},
            ]
        )
    )
    assert (out.iloc[0]["llm_sentiment"], out.iloc[0]["llm_scale"]) == (3.0, 0.0)


def test_separate_code_date_pairs_are_independent():
    out = fold_to_code_date(
        _scores(
            [
                {"article_id": "a1", "code": "005930"},
                {"article_id": "a2", "code": "000660"},
                {"article_id": "a3", "code": "005930", "date": pd.Timestamp("2026-01-06")},
            ]
        )
    )
    assert len(out) == 3
    assert list(out["n_articles"]) == [1, 1, 1]


def test_empty_input_returns_empty_contract_frame():
    out = fold_to_code_date(_scores([]).iloc[0:0])
    assert out.empty
    assert list(out.columns) == LLM_FRAME_COLUMNS


# --- load_llm_features (sqlite fixture, 네트워크·API 없음) -------------------

_F = ArticleFeatures(
    material_type="수주",
    persistence="structural",
    specificity=3,
    freshness="new",
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)


@pytest.fixture()
def workspace(tmp_path):
    news_path = tmp_path / "news.db"
    conn = sqlite3.connect(news_path)
    conn.executescript(
        """
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY, date TEXT, code TEXT, press TEXT,
            title TEXT, body200 TEXT, keywords TEXT
        );
        INSERT INTO articles VALUES
            ('n1','2026-01-05','005930','한국경제','제목1','본문1','키1'),
            ('n2','2026-01-05','000660','매일경제','제목2','본문2','키2');
        """
    )
    conn.commit()
    conn.close()

    cache_path = tmp_path / "llm.db"
    cache = open_cache(cache_path)
    for article_id in ("n1", "n2"):
        put_result(cache, article_id=article_id, model="cheap", prompt_version="v1",
                   features=_F, raw=None, error=None)
    cache.close()
    return {"news": str(news_path), "cache": str(cache_path)}


def test_load_llm_features_end_to_end(workspace):
    frame = load_llm_features(workspace["cache"], news_path=workspace["news"])
    assert list(frame.columns) == LLM_FRAME_COLUMNS
    assert sorted(frame["code"]) == ["000660", "005930"]
    assert frame["llm_sentiment"].tolist() == [3.0, 3.0]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_load_llm_features_auto_detects_the_only_model(workspace):
    """모델이 하나뿐이면 인자 없이도 찾아낸다."""
    assert len(load_llm_features(workspace["cache"], news_path=workspace["news"])) == 2


def test_load_llm_features_refuses_to_guess_between_models(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n1", model="premium", prompt_version="v1",
               features=_F, raw=None, error=None)
    cache.close()
    with pytest.raises(ValueError, match="cheap"):
        load_llm_features(workspace["cache"], news_path=workspace["news"])


def test_load_llm_features_honours_explicit_model(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n1", model="premium", prompt_version="v1",
               features=_F, raw=None, error=None)
    cache.close()
    frame = load_llm_features(
        workspace["cache"], news_path=workspace["news"],
        model="premium", prompt_version="v1",
    )
    assert list(frame["code"]) == ["005930"]


def test_load_llm_features_accepts_an_articles_frame(workspace):
    """계획 4 가 이미 기사 프레임을 갖고 있으면 news.db 를 다시 열지 않는다."""
    articles = pd.DataFrame(
        {
            "article_id": ["n1"],
            "date": pd.to_datetime(["2026-01-05"]),
            "code": ["005930"],
        }
    )
    frame = load_llm_features(workspace["cache"], articles=articles)
    assert list(frame["code"]) == ["005930"]


def test_available_models_counts_only_success_rows(workspace):
    cache = open_cache(workspace["cache"])
    put_result(cache, article_id="n3", model="cheap", prompt_version="v1",
               features=None, raw=None, error="실패")
    assert available_models(cache) == [("cheap", "v1", 2)]


def test_empty_cache_raises_a_useful_message(tmp_path):
    open_cache(tmp_path / "empty.db").close()
    with pytest.raises(ValueError, match="비어 있다"):
        load_llm_features(tmp_path / "empty.db", articles=pd.DataFrame())
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_features_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.features.llm'`

- [ ] **Step 3: 구현**

`src/swing_it/features/llm.py`:

```python
"""LLM 피처를 ``(code, date)`` 격자로 접는 어댑터.

계획 3 의 캐시는 기사 단위이고 모델은 ``(code, date)`` 단위로 학습한다.
그 사이를 잇는 것이 이 파일이며, 계획 4(층별 모델 비교)가 계획 3 에 닿는
유일한 지점이다.

접기 규칙은 "그날 가장 강한 재료 한 건의 행을 통째로 쓴다"이다. 컬럼별로 max 를
따로 집계하면 A기사의 구체성과 B기사의 감정이 섞인, 실재하지 않는 기사가 만들어진다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..adapters.news_db import connect_news, load_articles
from ..score.cache import CACHE_TABLE, load_features_frame
from ..score.prompt import PROMPT_VERSION
from ..score.schema import LLM_FEATURE_COLUMNS

__all__ = [
    "LLM_FEATURE_COLUMNS",
    "LLM_FRAME_COLUMNS",
    "available_models",
    "fold_to_code_date",
    "load_llm_features",
]

# material_type 은 모델 행렬에 넣지 않는다 (9개 원-핫이면 스펙 §6 상한을 혼자 먹는다).
# 한국어 문자열 그대로 근거 카드 표시용으로 딸려 보낸다.
LLM_FRAME_COLUMNS: list[str] = [
    "code", "date", "material_type", "n_articles", *LLM_FEATURE_COLUMNS,
]

_SORT_KEYS = ["code", "date", "_abs_sentiment", "llm_specificity", "article_id"]
_SORT_ORDER = [True, True, False, False, True]


def available_models(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """캐시에 들어 있는 ``(model, prompt_version, 성공 건수)`` 를 건수 내림차순으로."""
    rows = conn.execute(
        f"select model, prompt_version, count(*) from {CACHE_TABLE} "
        "where status = 'ok' group by model, prompt_version "
        "order by count(*) desc, model asc"
    ).fetchall()
    return [(str(model), str(version), int(count)) for model, version, count in rows]


def fold_to_code_date(scores: pd.DataFrame) -> pd.DataFrame:
    """같은 ``(code, date)`` 의 여러 기사를 대표 한 건으로 접는다.

    우선순위: ``|llm_sentiment|`` 최대 → ``llm_specificity`` 최대 →
    ``article_id`` 사전순 최소. 마지막 단계는 재현성을 위한 것이다.
    """
    if scores.empty:
        return pd.DataFrame(columns=LLM_FRAME_COLUMNS)

    work = scores.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["_abs_sentiment"] = work["llm_sentiment"].abs()
    work["n_articles"] = work.groupby(["code", "date"], sort=False)["article_id"].transform("size")

    ordered = work.sort_values(_SORT_KEYS, ascending=_SORT_ORDER, kind="mergesort")
    picked = ordered.groupby(["code", "date"], sort=True, as_index=False).head(1)
    return picked[LLM_FRAME_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _resolve_version(
    conn: sqlite3.Connection, model: str | None, prompt_version: str | None
) -> tuple[str, str]:
    """모델·프롬프트 버전이 생략되면 캐시에서 유일한 조합을 찾는다.

    여러 조합이 섞여 있으면 **고르지 않고 실패한다.** 저가 모델과 상위 모델의
    결과가 한 파일에 있는 게 정상 상태이고(계획 3 Task 7 대조), 아무거나 집으면
    어떤 모델로 학습했는지 모르는 채 결과가 나온다.
    """
    if model is not None:
        return model, prompt_version or PROMPT_VERSION
    combos = available_models(conn)
    if not combos:
        raise ValueError("LLM 피처 캐시가 비어 있다. swing-score 를 먼저 돌려라.")
    if prompt_version is not None:
        combos = [c for c in combos if c[1] == prompt_version]
    if len(combos) > 1:
        listed = ", ".join(f"{m}@{v}({n}건)" for m, v, n in combos)
        raise ValueError(
            f"캐시에 여러 모델이 있다: {listed}. model= 로 명시하라."
        )
    return combos[0][0], combos[0][1]


def load_llm_features(
    cache_path: str | Path = "data/llm_features.db",
    *,
    news_path: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    articles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """LLM 피처를 ``(code, date)`` 격자 프레임으로 읽는다.

    Args:
        cache_path: 계획 3 이 만든 sqlite 캐시. 기본 ``data/llm_features.db``.
        news_path: ``article_id → (code, date)`` 조인용 뉴스 DB. 생략 시
            ``SWING_NEWS_DB``, 그것도 없으면 ``data/news.db``.
        model / prompt_version: 생략하면 캐시에 조합이 하나뿐일 때만 자동 선택한다.
        articles: 이미 읽어둔 기사 프레임(``article_id, date, code``)이 있으면
            넘긴다. news DB 를 다시 열지 않는다.

    Returns:
        컬럼 ``LLM_FRAME_COLUMNS``. ``date`` 는 ``datetime64[ns]``,
        ``LLM_FEATURE_COLUMNS`` 는 전부 수치형.
    """
    conn = sqlite3.connect(Path(cache_path))
    try:
        resolved_model, resolved_version = _resolve_version(conn, model, prompt_version)
        features = load_features_frame(conn, resolved_model, resolved_version)
    finally:
        conn.close()

    if articles is None:
        articles = load_articles(connect_news(news_path), "0001-01-01", "9999-12-31")
    keys = articles[["article_id", "code", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])

    joined = features.merge(keys, on="article_id", how="inner")
    joined = joined[joined["code"].astype(str).str.strip().ne("")]
    return fold_to_code_date(
        joined[["article_id", "code", "date", "material_type", *LLM_FEATURE_COLUMNS]]
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_features_llm.py -v`
Expected: 16 passed

- [ ] **Step 5: 전체 테스트와 린트**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: 전부 통과 (계획 1 의 27건 + 이 계획의 92건 = 119건)

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/features/llm.py tests/test_features_llm.py
git commit -m "feat(features): LLM 피처를 (code, date) 격자로 접는 계획 4 인계 어댑터"
```

---

## Self-Review 체크

**스펙 커버리지 (계획 3 범위):**

| 스펙 요구 | 태스크 |
|---|---|
| §3 고정 스키마 6축(재료유형·지속성·구체성·신선도·감정·규모대비) | Task 1 — 7필드(감정을 방향·강도로 분리) |
| §3 LLM 은 판정하지 않는다 | Task 1 — `FORBIDDEN_PROMPT_TERMS` 테스트로 강제 |
| §4 입력은 제목 + 200자 + 키워드 | Task 1 `build_prompt` — 전문 키를 무시하는 테스트 포함 |
| §4 '핫함' — 거래대금 급증 동반 | Task 3 `no_volume_event` 규칙 |
| §8 provider 추상화 (Gemini/GPT/Claude 교체) | Task 2 |
| §8 기본은 최저가 등급 | Task 2 `DEFAULT_MODELS`, Task 7 Step 1 |
| §8 표본 수백 건 상위 모델 대조 | Task 7 |
| §8 실제 단가는 착수 시점 확인, 배치 API 검토 | Task 7 Step 1 |
| §8 API 키는 환경변수로만 | Task 2 `provider_from_env`, repr 마스킹 테스트 |
| §6 피처 10~20개 | Task 1 — `LLM_FEATURE_COLUMNS` 5개, 원-핫 금지 |
| §11 LLM 출력 불안정 → 고정 스키마·캐싱·버전 기록 | Task 1 `SchemaError`, Task 4 캐시 PK |
| §9 데이터 커밋 금지 | Task 6 — 리포트에 기사 텍스트·article_id 없음(테스트로 확인) |
| 부분 실패 처리 | Task 5 — 실패 기록 + 계속 진행 + 연속 실패 중단 |
| 테스트가 API 를 부르지 않음 | Task 2 `MockTransport`, Task 5·6 `FakeProvider` |
| 계획 4 인계 — `(code, date)` 격자 프레임 | Task 8 `load_llm_features` |

**계획 4 와의 계약 (Task 8 에서 확정):**

계획 4(`docs/superpowers/plans/2026-08-17-layers-and-report.md`)는 아래 두 이름만
import 한다. 다른 접점은 없다.

```python
from swing_it.features.llm import LLM_FEATURE_COLUMNS, load_llm_features
```

- `LLM_FEATURE_COLUMNS` 5개 — 전부 수치형. 계획 4 는 GBM 에 그대로 넣는다
- `load_llm_features(...)` 반환 컬럼 `["code", "date", "material_type", "n_articles", *LLM_FEATURE_COLUMNS]`
- 저장소는 **sqlite** `data/llm_features.db` 다 (parquet 아님). 첫 인자가 그 경로
- `material_type` 은 문자열이며 **모델 행렬 밖**이다 — 근거 카드 표시용
- `n_articles` 는 계획 2 가 '기사 건수' 피처를 주면 계획 4 에서 버린다 (중복)

**계획 3 범위 밖 (후속):**

- LLM 피처를 `build_dataset` 에 결합하는 일 — 계획 4. Task 8 이 격자까지 맞춰주므로
  계획 4 는 `(code, date)` 로 머지만 하면 된다
- 층별 기여도 비교(①수급 / ②+규칙뉴스 / ③+LLM) — 계획 4
- `material_type` 을 모델 피처로 쓸지 여부 — 계획 4. 쓰려면 walk-forward 폴드 안에서
  타깃 인코딩을 해야 한다 (폴드 밖 정보가 새면 안 된다)
- 배치 API(비동기 할인) 경로 — Task 7 Step 1 에서 값이 있다고 판단되면 별도 작업

**타입 일관성 확인:**

- `ArticleFeatures` 필드명이 Task 1 정의 → Task 4 `_SCHEMA_COLUMNS` → Task 7
  `CATEGORICAL_FIELDS`/`ORDINAL_FIELDS` 에서 동일하다.
- `LLM_FEATURE_COLUMNS` 는 Task 1 (`score/schema.py`) 에서만 정의한다. Task 4 가 쓰고,
  Task 8 은 **재수출만** 한다 — `features/llm.py` 에서 다시 정의하면 두 곳이 갈린다.
- `provider.model` 은 캐시 키에 그대로 들어간다 — `FakeProvider.model` 기본값
  `"fake-1"` 이 Task 6 CLI 테스트의 `report["model"]` 기대값과 같다.
- `prefilter_articles` 가 돌려주는 `keep`/`drop_reason`/`dup_count` 를 Task 6 CLI 가 쓴다.
- `extract_events(prices, *, adv_floor, tv_multiple, window)` 는 계획 1 Task 2 의
  시그니처 그대로다.

## 확인 로그

*(Task 7 Step 1·9·10 에서 채운다. 확인 날짜 · provider · 모델 ID · 단가 · 배치 API
지원 여부 · 대조 결과 요약을 적는다.)*

## 후속 계획

- 계획 4: 층별 모델 비교 (①수급 / ②+규칙뉴스 / ③+LLM) + kr-quant 게이트 + 근거 카드
