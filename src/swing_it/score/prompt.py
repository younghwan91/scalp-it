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
