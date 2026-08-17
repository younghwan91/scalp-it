"""공시 로컬 저장소 — sqlite ``data/dart.db``.

TimescaleDB 는 quant-airflow 소관이라 여기서 쓰지 않는다. 수집 결과는
로컬에만 둔다(``data/`` 는 gitignore 차단).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .classify import classify_report, is_correction

DISC_COLUMNS: list[str] = [
    "rcept_no", "rcept_dt", "code", "corp_name", "corp_cls",
    "report_nm", "material_type", "is_correction",
]

_CREATE = """
CREATE TABLE IF NOT EXISTS disclosures (
    rcept_no      TEXT PRIMARY KEY,
    rcept_dt      TEXT NOT NULL,
    code          TEXT NOT NULL,
    corp_name     TEXT,
    corp_cls      TEXT,
    report_nm     TEXT,
    material_type TEXT,
    is_correction INTEGER
);
CREATE INDEX IF NOT EXISTS idx_disc_dt ON disclosures(rcept_dt);
CREATE INDEX IF NOT EXISTS idx_disc_code ON disclosures(code);
"""


def init_dart_db(path: str | Path) -> sqlite3.Connection:
    """스키마를 만들고 연결을 돌려준다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_CREATE)
    conn.commit()
    return conn


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """공시검색 응답 항목을 저장 계약에 맞춘다.

    ``stock_code`` 가 빈 항목은 버린다. 비상장 법인이라 가격이 없고,
    가격이 없으면 라벨을 만들 수 없다.
    """
    records = []
    for row in rows:
        code = str(row.get("stock_code") or "").strip()
        if not code:
            continue
        report_nm = str(row.get("report_nm") or "").strip()
        records.append(
            {
                "rcept_no": str(row.get("rcept_no") or "").strip(),
                "rcept_dt": str(row.get("rcept_dt") or "").strip(),
                "code": code,
                "corp_name": str(row.get("corp_name") or "").strip(),
                "corp_cls": str(row.get("corp_cls") or "").strip(),
                "report_nm": report_nm,
                "material_type": classify_report(report_nm),
                "is_correction": is_correction(report_nm),
            }
        )
    return pd.DataFrame(records, columns=DISC_COLUMNS)


def upsert_disclosures(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """``rcept_no`` 기준으로 덮어쓴다."""
    if frame.empty:
        return 0
    payload = [
        (
            r["rcept_no"], r["rcept_dt"], r["code"], r["corp_name"],
            r["corp_cls"], r["report_nm"], r["material_type"], int(bool(r["is_correction"])),
        )
        for _, r in frame.iterrows()
    ]
    conn.executemany(
        "INSERT INTO disclosures "
        "(rcept_no, rcept_dt, code, corp_name, corp_cls, report_nm, material_type, is_correction) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(rcept_no) DO UPDATE SET "
        "rcept_dt=excluded.rcept_dt, code=excluded.code, corp_name=excluded.corp_name, "
        "corp_cls=excluded.corp_cls, report_nm=excluded.report_nm, "
        "material_type=excluded.material_type, is_correction=excluded.is_correction",
        payload,
    )
    conn.commit()
    return len(payload)


def load_disclosures(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    """``rcept_dt`` 구간(YYYYMMDD 문자열 비교)의 공시를 읽는다."""
    rows = conn.execute(
        f"select {', '.join(DISC_COLUMNS)} from disclosures "
        "where rcept_dt >= ? and rcept_dt <= ? order by rcept_dt, rcept_no",
        (start, end),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=DISC_COLUMNS)
    frame["rcept_dt"] = pd.to_datetime(frame["rcept_dt"], format="%Y%m%d", errors="coerce")
    frame["is_correction"] = frame["is_correction"].astype(bool)
    return frame


def collected_range(conn: sqlite3.Connection) -> tuple[str, str] | None:
    """적재된 접수일자의 최소·최대. 비어 있으면 ``None``."""
    row = conn.execute("select min(rcept_dt), max(rcept_dt) from disclosures").fetchone()
    if not row or row[0] is None:
        return None
    return (str(row[0]), str(row[1]))
