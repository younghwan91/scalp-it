"""공시를 LLM 으로 채점해 '재료'를 판단한다 — 만쥬 §2-1.

만쥬 주도주 3대 요건의 세 번째는 **"명확한 재료 및 미래 가치"**, 그리고 **"단순 수급뿐
아니라 시장의 관심을 지속시킬 강력한 스토리"** 다. 이건 유형별 가중치표로는 안 된다 —
`universe.MATERIAL_WEIGHTS` 는 "공급계약"이 1억짜리인지 1조짜리인지 모르고, 같은
"기술_인허가"라도 임상 1상인지 품목허가인지 구분하지 못한다.

레포의 LLM 채점기(`score/`)가 뽑는 필드가 정확히 그 판단이다:

    llm_persistence  ← "시장의 관심을 **지속**시킬"   (§2-1)
    llm_specificity  ← "**명확한** 재료"              (§2-1)
    llm_scale        ← 규모
    llm_freshness    ← 테마 **신선도**                (§5-5)
    llm_sentiment    ← 방향(악재 판별)

그 채점기는 뉴스 기사용으로 만들어졌지만 입력이 ``code``·``title``·``body200``·
``keywords`` 넷뿐이라 **공시에 그대로 쓸 수 있다**. 새로 짜지 않고 재사용한다.

뉴스가 아니라 공시인 이유는 `docs/research/manju/17` 의 정책이다 — "1년치 종목 뉴스를
프로그램으로 합법 확보할 경로는 없다", "직접 크롤러를 새로 짜지 않는다(ToS·저작권)".
DART 공시가 그 문서가 지목한 현실 경로다.

**운영 전제**: LLM 제공자 키(`ANTHROPIC_API_KEY` 등)가 필요하다. 키가 없으면 이 경로는
쓰이지 않고 `universe.material_score` 의 규칙 기반 가중치가 그대로 돈다 — 재료 판단이
없는 것보다 거친 판단이 낫고, 거친 판단보다 LLM 판단이 낫다는 순서다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "disclosure_to_article",
    "llm_material_score",
    "score_disclosures",
]

#: 재료 점수에서 각 축이 갖는 무게. 지속성이 가장 크다 — 만쥬가 "**지속**시킬 스토리"를
#: 명시적으로 요구하기 때문이다. 규모는 그다음(수급을 부른다), 구체성은 보조다.
_WEIGHTS = {"llm_persistence": 0.5, "llm_scale": 0.3, "llm_specificity": 0.2}


def disclosure_to_article(row: Mapping[str, Any]) -> dict[str, str]:
    """DART 공시 한 건을 채점기가 읽는 형태로 접는다.

    공시는 **제목과 분류만** 있고 본문이 없다(`dart/store.py` 스키마). 그래서
    ``body200`` 은 빈 문자열이다 — 프롬프트 템플릿이 빈 값을 견디도록 돼 있다.
    종목명은 제목에 붙인다. 어느 회사 일인지가 재료 판단에 필요하다.
    """
    corp = str(row.get("corp_name") or "").strip()
    title = str(row.get("report_nm") or "").strip()
    return {
        "code": str(row.get("code") or "").strip(),
        "title": f"{corp} {title}".strip() if corp else title,
        "body200": "",
        "keywords": str(row.get("material_type") or "").strip(),
    }


def llm_material_score(features: Mapping[str, Any]) -> float:
    """LLM 피처 → 재료 점수 하나. 방향(감정)은 곱으로 쓴다.

    지속성·규모·구체성을 가중합한 뒤 감정 부호를 곱한다 — **악재면 음수**여야 한다
    (재료가 아니라 반대 방향이다, `universe.MATERIAL_WEIGHTS` 와 같은 원칙).
    필드가 비면 0 으로 본다(억지 추정 금지).
    """
    total = 0.0
    for key, w in _WEIGHTS.items():
        try:
            total += w * float(features.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    if total <= 0:
        return 0.0
    try:
        sentiment = float(features.get("llm_sentiment") or 0.0)
    except (TypeError, ValueError):
        sentiment = 0.0
    if sentiment < 0:
        return -total
    return total


def score_disclosures(rows: Sequence[Mapping[str, Any]], *, cache_path: str,
                      env: Mapping[str, str] | None = None,
                      limit: int | None = None) -> dict[str, float]:
    """공시들을 LLM 으로 채점해 ``{종목코드: 재료점수}`` 로.

    **어떤 실패도 기동을 막지 않는다.** 키가 없거나, 제공자가 죽거나, 캐시를 못 열면
    빈 dict 를 돌려주고 호출부는 규칙 기반(`universe.material_score`)으로 간다.
    재료 판단이 없는 것보다 거친 판단이 낫고, 거친 판단보다 LLM 판단이 낫다 — 그
    순서를 지키되 어느 단계도 장 시작을 막지 못하게 한다.

    같은 공시를 두 번 부르지 않도록 채점 결과는 캐시에 남는다(`score/cache.py`).
    """
    if not rows:
        return {}
    try:
        import pandas as pd

        from .score.batch import score_articles
        from .score.cache import load_features_frame, open_cache
        from .score.prompt import PROMPT_VERSION
        from .score.provider import provider_from_env

        provider = provider_from_env(env)
        frame = pd.DataFrame([
            {"article_id": str(r.get("rcept_no") or ""), **disclosure_to_article(r)}
            for r in rows
        ])
        with open_cache(cache_path) as cache:
            score_articles(frame, provider, cache, limit=limit)
            feats = load_features_frame(cache, provider.model, PROMPT_VERSION)
    except Exception:  # noqa: BLE001 — 재료 채점 실패로 장을 놓치지 않는다
        return {}
    if feats is None or getattr(feats, "empty", True):
        return {}
    out: dict[str, float] = {}
    id_to_code = {str(r.get("rcept_no") or ""): str(r.get("code") or "") for r in rows}
    for rec in feats.to_dict("records"):
        code = id_to_code.get(str(rec.get("article_id") or ""))
        if not code:
            continue
        score = llm_material_score(rec)
        # 한 종목에 공시 여러 건이면 **가장 강한 재료 하나**를 쓴다. 합치면 잡공시
        # 여러 건이 강한 재료 하나를 이긴다.
        if code not in out or abs(score) > abs(out[code]):
            out[code] = score
    return out
