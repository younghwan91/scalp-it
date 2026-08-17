import pandas as pd
import pytest

from swing_it.dart.store import (
    DISC_COLUMNS,
    collected_range,
    init_dart_db,
    load_disclosures,
    rows_to_frame,
    upsert_disclosures,
)

_ROWS = [
    {"corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
     "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
     "rcept_no": "20260814000123", "flr_nm": "에프알텍", "rcept_dt": "20260814", "rm": ""},
    {"corp_cls": "Y", "corp_name": "삼성전자", "corp_code": "00126380",
     "stock_code": "005930", "report_nm": "반기보고서",
     "rcept_no": "20260814000999", "flr_nm": "삼성전자", "rcept_dt": "20260814", "rm": ""},
    # 비상장 법인 — stock_code 가 비어 있다
    {"corp_cls": "E", "corp_name": "비상장회사", "corp_code": "00999999",
     "stock_code": " ", "report_nm": "감사보고서제출",
     "rcept_no": "20260814000777", "flr_nm": "비상장회사", "rcept_dt": "20260814", "rm": ""},
]


@pytest.fixture()
def conn(tmp_path):
    return init_dart_db(tmp_path / "dart.db")


def test_rows_to_frame_has_contract_columns():
    frame = rows_to_frame(_ROWS)
    assert list(frame.columns) == DISC_COLUMNS


def test_rows_to_frame_drops_unlisted():
    """상장 종목만 쓴다. stock_code 없는 법인은 가격이 없어 라벨을 못 만든다."""
    frame = rows_to_frame(_ROWS)
    assert len(frame) == 2
    assert set(frame["code"]) == {"073540", "005930"}


def test_rows_to_frame_classifies():
    frame = rows_to_frame(_ROWS).set_index("code")
    assert frame.loc["073540", "material_type"] == "공급계약"
    assert frame.loc["005930", "material_type"] == "정기보고"


def test_rows_to_frame_on_empty():
    frame = rows_to_frame([])
    assert list(frame.columns) == DISC_COLUMNS
    assert frame.empty


def test_upsert_then_load(conn):
    n = upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert n == 2
    got = load_disclosures(conn, "20260801", "20260831")
    assert len(got) == 2
    assert pd.api.types.is_datetime64_any_dtype(got["rcept_dt"])


def test_upsert_is_idempotent(conn):
    frame = rows_to_frame(_ROWS)
    upsert_disclosures(conn, frame)
    upsert_disclosures(conn, frame)
    assert len(load_disclosures(conn, "20260801", "20260831")) == 2


def test_load_respects_range(conn):
    upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert load_disclosures(conn, "20260815", "20260831").empty


def test_collected_range_reports_bounds(conn):
    assert collected_range(conn) is None
    upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert collected_range(conn) == ("20260814", "20260814")
