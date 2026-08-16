"""LLM 구조화 피처 추출 (스펙 §3).

LLM 은 판정기가 아니라 추출기다. 기사 하나당 고정 스키마 7필드를 채울 뿐이며,
"이 종목이 오를까"는 묻지 않는다. 그 답을 믿는 순간 검증이 불가능해진다.

계층:
    schema/prompt/prefilter/compare  순수. 네트워크를 모른다.
    provider                          HTTP. httpx 로 직접 호출한다 (SDK 3종 미도입).
    cache/batch                       sqlite 캐시와 배치 러너.
"""
