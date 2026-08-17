"""공시 제목을 재료 유형으로 분류한다 — 규칙 기반, LLM 불필요.

공시는 뉴스와 달리 **유형이 제목에 규격화돼 있다.** 금융감독원 서식이 정한
이름이라 표현이 흔들리지 않는다. 그래서 LLM 없이도 재료 유형을 얻을 수 있고,
이것이 ②층을 무료로 만드는 근거다.

규칙은 사전등록이다. 성과를 보고 바꾸면 그 순간 검증이 무의미해진다.
"""

from __future__ import annotations

import pandas as pd

MATERIAL_TYPES: list[str] = [
    "악재", "내부자", "공급계약", "시설투자", "주주환원", "자본조달",
    "지분변동", "실적", "기술_인허가", "구조조정", "재량공시", "정기보고",
    "행정", "기타",
]

#: 이벤트에서 배제하는 유형. 롱온리라 공매도로 못 먹는다 (스펙 §5).
EXCLUDED_TYPES: frozenset[str] = frozenset({"악재"})

# 위에서 아래로 처음 맞는 것을 쓴다. 순서에 의미가 있다:
#  - '악재'가 맨 위 — 방향을 결정한다. "소송등의판결"은 구조조정과 겹치지만 악재다.
#  - '내부자'가 그다음 — "임원ㆍ주요주주…"가 지분변동의 '최대주주'와 겹친다.
#    실측 하루 60건으로 전체 최다라 별도 축으로 봐야 한다.
#  - '주주환원'이 '자본조달'보다 위 — "자기주식취득 신탁계약"류가 증권 키워드와 겹친다.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("악재", ("감사의견", "의견거절", "부적정", "자본잠식", "상장폐지", "거래정지",
              "회생절차", "파산", "부도", "횡령", "배임", "소송", "손실발생",
              "실질심사", "미달사실발생", "영업정지")),
    ("내부자", ("임원ㆍ주요주주", "임원·주요주주", "특정증권등소유상황",
                "특정증권등거래계획")),
    ("공급계약", ("공급계약", "수주", "판매계약")),
    ("시설투자", ("신규시설투자", "유형자산", "공장", "증설")),
    ("주주환원", ("자기주식", "주식소각", "배당")),
    ("자본조달", ("유상증자", "무상증자", "전환사채", "신주인수권", "교환사채",
                 "전환가액", "증권신고서", "일괄신고")),
    ("지분변동", ("대량보유", "최대주주", "주식매수선택권")),
    ("실적", ("매출액또는손익구조", "영업(잠정)실적", "연결재무제표기준영업",
              "결산실적", "매출액미달")),
    ("기술_인허가", ("특허", "임상", "품목허가", "승인", "인증")),
    ("구조조정", ("합병", "분할", "양수", "양도", "채무보증")),
    ("재량공시", ("투자판단관련주요경영사항", "기업설명회", "기타시장안내")),
    ("정기보고", ("사업보고서", "반기보고서", "분기보고서", "감사보고서", "검토",
                 "투자설명서")),
    ("행정", ("주주총회", "주주명부", "의결권대리행사", "기준일",
              "증권발행실적", "증권발행결과")),
]

_CORRECTION_MARKERS = ("[기재정정]", "[첨부정정]", "[첨부추가]", "[정정]")


def classify_report(report_nm: str) -> str:
    """공시 제목을 재료 유형 하나로 매핑한다. 못 맞추면 ``기타``."""
    text = str(report_nm or "").strip()
    for marker in _CORRECTION_MARKERS:
        text = text.replace(marker, "")
    text = text.strip()
    for material_type, keywords in _RULES:
        if any(keyword in text for keyword in keywords):
            return material_type
    return "기타"


def is_correction(report_nm: str) -> bool:
    """정정·첨부 공시인가. 원 공시의 재탕이라 신선도가 낮다."""
    text = str(report_nm or "").strip()
    return any(text.startswith(marker) for marker in _CORRECTION_MARKERS)


def classify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """``report_nm`` 컬럼에 ``material_type``·``is_correction`` 을 붙인다."""
    out = frame.copy()
    out["material_type"] = [classify_report(v) for v in out.get("report_nm", [])]
    out["is_correction"] = [is_correction(v) for v in out.get("report_nm", [])]
    return out
