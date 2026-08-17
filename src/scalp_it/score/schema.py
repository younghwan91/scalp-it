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
