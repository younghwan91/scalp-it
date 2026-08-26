"""공시 → LLM 재료 채점 어댑터 테스트(만쥬 §2-1).

만쥬 요건은 "명확한 재료 및 **미래 가치**", "시장의 관심을 **지속**시킬 강력한 스토리"다.
유형별 가중치표(MATERIAL_WEIGHTS)로는 그 판단이 안 된다 — "공급계약"이 1억인지 1조인지
표는 모른다. 레포의 LLM 채점기가 뽑는 필드가 정확히 그 판단이다:

    llm_persistence  ← "관심을 지속시킬"
    llm_specificity  ← "명확한"
    llm_scale        ← 규모
    llm_freshness    ← 테마 신선도(§5-5)

기사용으로 만들어진 채점기지만 입력이 code·title·body200·keywords 뿐이라 공시에 그대로
쓸 수 있다. 새로 짜지 않고 재사용한다.
"""

from scalp_it.material_llm import (
    disclosure_to_article,
    llm_material_score,
)


def _row(**kw):
    base = {"rcept_no": "20260826000001", "code": "100000", "corp_name": "테스트전자",
            "report_nm": "단일판매ㆍ공급계약체결", "material_type": "공급계약"}
    base.update(kw)
    return base


def test_disclosure_becomes_article_shaped_input():
    """채점기가 읽는 네 필드로 접는다 — 나머지는 버린다."""
    a = disclosure_to_article(_row())
    assert a["code"] == "100000"
    assert "공급계약" in a["title"]
    assert a["keywords"] == "공급계약"          # 재료 유형을 키워드로 넘긴다
    # 공시는 본문이 없다(제목+분류만) — 빈 문자열이어야 프롬프트가 깨지지 않는다
    assert a["body200"] == ""


def test_disclosure_article_carries_corp_name():
    """종목명이 있으면 제목에 붙인다 — 어느 회사 일인지가 재료 판단에 필요하다."""
    a = disclosure_to_article(_row())
    assert "테스트전자" in a["title"]


def test_llm_material_score_combines_fields():
    """지속성·구체성·규모를 합쳐 하나의 재료 점수로.

    감정(sentiment)은 방향이라 곱으로 쓴다 — 악재면 점수가 음수여야 한다.
    """
    strong = llm_material_score({"llm_persistence": 3, "llm_specificity": 3,
                                 "llm_scale": 3, "llm_sentiment": 1})
    weak = llm_material_score({"llm_persistence": 1, "llm_specificity": 1,
                               "llm_scale": 1, "llm_sentiment": 1})
    bad = llm_material_score({"llm_persistence": 3, "llm_specificity": 3,
                              "llm_scale": 3, "llm_sentiment": -1})
    assert strong > weak > 0
    assert bad < 0, "악재는 감점이어야 한다"


def test_llm_material_score_handles_missing_fields():
    """필드가 비면 0 — 억지 추정 금지."""
    assert llm_material_score({}) == 0.0
    assert llm_material_score({"llm_persistence": 2}) >= 0.0


# ----- 폴백 — 키가 없어도 죽지 않는다 -----------------------------------------

def test_score_disclosures_falls_back_without_provider(tmp_path):
    """LLM 키가 없으면 빈 결과를 돌려준다 — 호출부는 규칙 기반으로 간다.

    재료 판단이 없는 것보다 거친 판단(가중치표)이 낫고, 거친 판단보다 LLM 판단이 낫다.
    그 순서를 지키되 **어느 단계도 기동을 막지 않는다**.
    """
    from scalp_it.material_llm import score_disclosures

    got = score_disclosures([{"rcept_no": "1", "code": "100000",
                              "report_nm": "단일판매ㆍ공급계약체결",
                              "material_type": "공급계약"}],
                            cache_path=str(tmp_path / "c.db"), env={})
    assert got == {}


def test_score_disclosures_uses_fake_provider(tmp_path):
    """fake provider 로 경로 전체가 도는지 — 키 없이 검증 가능한 유일한 방법."""
    from scalp_it.material_llm import score_disclosures

    got = score_disclosures(
        [{"rcept_no": "20260826000001", "code": "100000", "corp_name": "테스트전자",
          "report_nm": "단일판매ㆍ공급계약체결", "material_type": "공급계약"}],
        cache_path=str(tmp_path / "c.db"),
        env={"SWING_LLM_PROVIDER": "fake"},
    )
    assert "100000" in got
    assert isinstance(got["100000"], float)
