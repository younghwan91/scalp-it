"""PEAD 롱 리스크 오버레이 — 명백한 악재성 공시가 붙은 롱 종목을 플래그한다.

이것은 **수익 예측이 아니다.** 24번 검증에서 DART 유형층은 수익 신호가 없었다
(유형만으로는 초과수익을 못 만든다). 여기 목적은 다르다 — "명백한 악재(감자·
유상증자·횡령·배임·상장폐지 실질심사 등)가 걸린 종목을 롱에서 회피"하는 **리스크
필터**다. 값을 더하면만 쓰고, 애매하면 아무것도 빼지 않는다. LLM 이 필요 없다 —
공시 제목(report_nm)/유형(material_type) 키워드로 충분하다.

**최신성 한계(정직히).** ``data/dart_full.db`` 는 공시가 2016~2023 에서 끝난다.
현재 분기 롱 종목의 **최근** 악재를 이 DB 로는 못 잡는다. 그러므로 플래그가
없다는 것이 "안전"의 증거가 아니다. 실운영 시엔 진입 당일 DART Open API 를
조회해야 한다(``fetch_recent_disclosures`` 훅 자리만 만들어 둔다). 이 모듈은
있는 데이터로 과거 악재 이력을 붙이고, DB 커버리지를 함께 알린다.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: 롱에서 회피하고 싶은 명백한 악재 키워드. 공시 제목(report_nm)에서 부분일치.
#: 유상증자는 희석이라 롱 관점 악재로 포함(과제 스펙: 감자·유증).
BAD_KEYWORDS: tuple[str, ...] = (
    "감자",
    "유상증자",
    "횡령",
    "배임",
    "상장폐지",
    "관리종목",
    "거래정지",
    "실질심사",
    "감사의견거절",
    "감사의견 거절",
    "의견거절",
    "자본잠식",
    "불성실공시",
    "회생절차",
    "파산",
    "영업정지",
)

#: material_type 중 그 자체로 악재 신호인 것.
BAD_TYPES: tuple[str, ...] = ("악재",)

DEFAULT_DB = "data/dart_full.db"


@dataclass
class RiskHit:
    code: str
    rcept_dt: str
    report_nm: str
    material_type: str


@dataclass
class RiskScan:
    """롱 종목 악재 스캔 결과.

    ``hits`` 는 종목코드 → 발견 목록. ``db_max_date`` 는 DB 가 커버하는 마지막
    공시일(YYYYMMDD) — 최신성 판단용. ``covered`` 는 스캔이 실제로 실행됐는지.
    """

    hits: dict[str, list[RiskHit]] = field(default_factory=dict)
    db_max_date: str | None = None
    covered: bool = False

    @property
    def flagged_codes(self) -> set[str]:
        return set(self.hits)


def _match_keywords(text: str) -> bool:
    return any(kw in text for kw in BAD_KEYWORDS)


def scan_longs(
    codes: list[str],
    db_path: str | Path = DEFAULT_DB,
    *,
    since: str | None = None,
) -> RiskScan:
    """롱 종목코드 목록에 대해 dart_full.db 에서 악재성 공시를 찾는다.

    Args:
        codes: 롱 종목코드(6자리 문자열) 목록.
        db_path: DART 공시 SQLite. 없으면 빈 스캔(covered=False)을 돌려준다.
        since: YYYYMMDD 이상만 스캔(생략 시 전 구간). 정정공시 제외.

    Returns:
        :class:`RiskScan`. DB 가 없으면 ``covered=False`` 로, 훅만 남겨둔다.
    """
    scan = RiskScan()
    if not codes:
        scan.covered = Path(db_path).exists()
        return scan
    path = Path(db_path)
    if not path.exists():
        return scan  # covered=False — 실운영 시 당일 DART 조회 필요

    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("select max(rcept_dt) from disclosures").fetchone()
        scan.db_max_date = row[0] if row else None
        scan.covered = True

        placeholders = ",".join("?" * len(codes))
        params: list[str] = list(codes)
        sql = (
            "select code, rcept_dt, report_nm, material_type, is_correction "
            f"from disclosures where code in ({placeholders})"
        )
        if since:
            sql += " and rcept_dt >= ?"
            params.append(since)
        sql += " order by rcept_dt desc"

        for code, rcept_dt, report_nm, mtype, is_corr in conn.execute(sql, params):
            if is_corr:
                continue  # 정정공시는 중복이라 건너뛴다
            report_nm = report_nm or ""
            mtype = mtype or ""
            if _match_keywords(report_nm) or mtype in BAD_TYPES:
                scan.hits.setdefault(str(code), []).append(
                    RiskHit(str(code), str(rcept_dt), report_nm, mtype)
                )
    finally:
        conn.close()
    return scan


def fetch_recent_disclosures(codes: list[str], *, as_of: str) -> RiskScan:
    """실운영 훅 — 진입 당일 DART Open API 로 최근 악재 공시를 조회한다.

    현재는 미구현이다. ``dart_full.db`` 가 2023 에서 끝나 최신 분기를 못 덮기
    때문에, 실전 투입 시 이 자리에 DART Open API(list.json, 최근 N일) 조회를
    붙여야 한다. 지금은 ``covered=False`` 로 돌려주어 "최신성 부족"을 드러낸다.
    """
    return RiskScan(covered=False)


def freshness_note(scan: RiskScan, as_of: str) -> str:
    """스캔의 최신성을 한 줄로 정직하게 요약한다."""
    if not scan.covered:
        return (
            "리스크 스캔 미실행 (DART DB 없음) — 실운영 시 진입 당일 DART Open API "
            "조회 필요."
        )
    if scan.db_max_date and scan.db_max_date < as_of.replace("-", ""):
        return (
            f"DART DB 커버리지 ~{scan.db_max_date} (기준일 {as_of}보다 과거). "
            "**플래그 없음 = 안전 아님.** 최근 악재는 이 DB 로 못 잡는다 — "
            "실운영 시 진입 당일 DART 조회 필수."
        )
    return f"DART DB 커버리지 ~{scan.db_max_date}."
