import pandas as pd
import pytest

from swing_it.dart.classify import (
    MATERIAL_TYPES,
    classify_frame,
    classify_report,
    is_correction,
)


@pytest.mark.parametrize(
    ("report_nm", "expected"),
    [
        # 실측 상위 유형 — 2026-05~07 표본에서 실제로 나온 제목들이다
        ("임원ㆍ주요주주특정증권등소유상황보고서", "내부자"),
        ("임원ㆍ주요주주특정증권등거래계획보고서", "내부자"),
        ("단일판매ㆍ공급계약체결", "공급계약"),
        ("신규시설투자등", "시설투자"),
        ("자기주식취득결정", "주주환원"),
        ("자기주식처분결정", "주주환원"),
        ("주식소각결정", "주주환원"),
        ("현금ㆍ현물배당결정", "주주환원"),
        ("유상증자결정", "자본조달"),
        ("무상증자결정", "자본조달"),
        ("전환가액의조정", "자본조달"),
        ("증권신고서", "자본조달"),
        ("주식등의대량보유상황보고서", "지분변동"),
        ("주식매수선택권부여에관한신고", "지분변동"),
        ("매출액또는손익구조30%이상변동", "실적"),
        ("연결재무제표기준영업(잠정)실적", "실적"),
        ("결산실적공시예고", "실적"),
        ("특허권취득", "기술_인허가"),
        ("회사합병결정", "구조조정"),
        ("타법인주식및출자증권양수결정", "구조조정"),
        ("타인에대한채무보증결정", "구조조정"),
        ("투자판단관련주요경영사항", "재량공시"),
        ("기업설명회", "재량공시"),
        ("기타시장안내", "재량공시"),
        ("반기보고서", "정기보고"),
        ("사업보고서", "정기보고"),
        ("투자설명서", "정기보고"),
        ("주주총회소집결의", "행정"),
        ("주주명부폐쇄기간또는기준일설정", "행정"),
        ("의결권대리행사권유참고서류", "행정"),
        ("증권발행실적보고서", "행정"),
        ("반기검토의견부적정,의견거절또는완전자본잠식사실발생", "악재"),
        ("주권매매거래정지", "악재"),
        ("파생상품거래손실발생", "악재"),
    ],
)
def test_classify_report(report_nm, expected):
    assert classify_report(report_nm) == expected


def test_insider_wins_over_major_shareholder_keyword():
    """'임원ㆍ주요주주…'는 지분변동의 '최대주주'와 겹칠 수 있다. 내부자가 우선한다."""
    assert classify_report("임원ㆍ주요주주특정증권등소유상황보고서") == "내부자"


def test_coverage_on_real_titles_is_high():
    """실측 상위 제목 중 '기타'로 빠지는 비율이 낮아야 한다.

    초안 규칙에서는 '기타'가 66% 였다. 유형이 피처인 설계에서 3분의 2가
    한 값에 몰리면 그 피처는 정보를 거의 담지 못한다.
    """
    titles = [
        "임원ㆍ주요주주특정증권등소유상황보고서", "연결재무제표기준영업(잠정)실적",
        "투자설명서", "기업설명회", "증권발행실적보고서", "일괄신고추가서류",
        "주주총회소집결의", "결산실적공시예고", "주주명부폐쇄기간또는기준일설정",
        "임시주주총회결과", "주주총회소집공고", "현금ㆍ현물배당결정", "주식소각결정",
        "투자판단관련주요경영사항", "타인에대한채무보증결정", "증권발행결과",
        "증권신고서", "주식매수선택권부여에관한신고", "타법인주식및출자증권취득결정",
        "전환가액의조정", "기타시장안내", "단일판매ㆍ공급계약체결", "반기보고서",
    ]
    etc = sum(1 for t in titles if classify_report(t) == "기타")
    assert etc / len(titles) < 0.15


def test_every_result_is_a_declared_type():
    for nm in ("단일판매ㆍ공급계약체결", "듣도보도못한공시", ""):
        assert classify_report(nm) in MATERIAL_TYPES


def test_bad_news_wins_over_other_keywords():
    """'소송 등의 판결'은 구조조정 키워드(양도)와 겹칠 수 있다. 악재가 우선한다."""
    assert classify_report("소송등의판결ㆍ결정") == "악재"


def test_correction_prefix_detected():
    assert is_correction("[기재정정]단일판매ㆍ공급계약체결")
    assert is_correction("[첨부추가]반기보고서")
    assert not is_correction("단일판매ㆍ공급계약체결")


def test_correction_does_not_change_type():
    assert classify_report("[기재정정]단일판매ㆍ공급계약체결") == "공급계약"


def test_whitespace_is_tolerated():
    assert classify_report("  단일판매ㆍ공급계약체결   ") == "공급계약"


def test_classify_frame_adds_two_columns():
    frame = pd.DataFrame({"report_nm": ["단일판매ㆍ공급계약체결", "[기재정정]반기보고서"]})
    out = classify_frame(frame)
    assert list(out["material_type"]) == ["공급계약", "정기보고"]
    assert list(out["is_correction"]) == [False, True]


def test_classify_frame_keeps_other_columns():
    frame = pd.DataFrame({"report_nm": ["유상증자결정"], "rcept_no": ["x"]})
    out = classify_frame(frame)
    assert out.iloc[0]["rcept_no"] == "x"


def test_classify_frame_on_empty():
    out = classify_frame(pd.DataFrame({"report_nm": []}))
    assert list(out.columns) == ["report_nm", "material_type", "is_correction"]
    assert out.empty
