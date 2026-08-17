from scalp_it.score.prompt import FORBIDDEN_PROMPT_TERMS, PROMPT_VERSION, build_prompt
from scalp_it.score.schema import MATERIAL_TYPES

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
