# NOTE: 분류 규칙은 Task 2 담당이 같은 파일에 채운다. 여기는 상수만.
"""공시 유형 상수."""

from __future__ import annotations

MATERIAL_TYPES: list[str] = [
    "악재", "내부자", "공급계약", "시설투자", "주주환원", "자본조달",
    "지분변동", "실적", "기술_인허가", "구조조정", "재량공시", "정기보고",
    "행정", "기타",
]
EXCLUDED_TYPES: frozenset[str] = frozenset({"악재"})
