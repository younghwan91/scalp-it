"""PEAD 롱 리스크 오버레이 — 키워드 필터·정정공시 제외·최신성.

실 DART DB 를 안 쓴다. 합성 SQLite 를 만들어 스캔 규칙만 고정한다.
"""
import sqlite3

from scalp_it import pead_risk


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "create table disclosures (rcept_no text, rcept_dt text, code text, "
        "corp_name text, corp_cls text, report_nm text, material_type text, "
        "is_correction integer)"
    )
    conn.executemany(
        "insert into disclosures values (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


def test_missing_db_is_not_covered(tmp_path):
    scan = pead_risk.scan_longs(["000001"], tmp_path / "nope.db")
    assert scan.covered is False
    assert scan.flagged_codes == set()


def test_keyword_in_report_name_flags(tmp_path):
    db = tmp_path / "d.db"
    _make_db(
        db,
        [
            ("1", "20260101", "000001", "A", "Y", "유상증자결정", "자본조달", 0),
            ("2", "20260101", "000002", "B", "Y", "분기보고서", "정기보고", 0),
        ],
    )
    scan = pead_risk.scan_longs(["000001", "000002"], db)
    assert scan.covered is True
    assert scan.flagged_codes == {"000001"}
    assert "유상증자" in scan.hits["000001"][0].report_nm


def test_material_type_akjae_flags(tmp_path):
    db = tmp_path / "d.db"
    _make_db(
        db,
        [("3", "20260101", "000003", "C", "Y", "중요한 무언가", "악재", 0)],
    )
    scan = pead_risk.scan_longs(["000003"], db)
    assert scan.flagged_codes == {"000003"}


def test_correction_disclosures_skipped(tmp_path):
    db = tmp_path / "d.db"
    _make_db(
        db,
        [("4", "20260101", "000004", "D", "Y", "[정정]감자결정", "구조조정", 1)],
    )
    scan = pead_risk.scan_longs(["000004"], db)
    assert scan.flagged_codes == set()


def test_since_filter_excludes_old(tmp_path):
    db = tmp_path / "d.db"
    _make_db(
        db,
        [("5", "20200101", "000005", "E", "Y", "횡령ㆍ배임", "악재", 0)],
    )
    old = pead_risk.scan_longs(["000005"], db, since="20260101")
    assert old.flagged_codes == set()  # 2020 공시는 since=2026 밖
    recent = pead_risk.scan_longs(["000005"], db, since="20190101")
    assert recent.flagged_codes == {"000005"}


def test_freshness_note_warns_when_db_is_stale(tmp_path):
    db = tmp_path / "d.db"
    _make_db(db, [("6", "20231229", "000006", "F", "Y", "분기보고서", "정기보고", 0)])
    scan = pead_risk.scan_longs(["000006"], db, since="20260101")
    note = pead_risk.freshness_note(scan, "2026-08-24")
    assert "안전 아님" in note
    assert "20231229" in note


def test_fetch_recent_hook_is_uncovered():
    scan = pead_risk.fetch_recent_disclosures(["000001"], as_of="2026-08-24")
    assert scan.covered is False
