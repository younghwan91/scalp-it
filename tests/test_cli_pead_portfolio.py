"""`scalp-pead-portfolio` — 실전 롱숏 생성·CSV 저장·리스크 플래그.

포트폴리오 선정은 :mod:`scalp_it.cli_pead_watch` 와 같은 함수라 여기선 표/비중/
리스크 병합과 CSV 저장, 그리고 로더 monkeypatch 로 main 통합만 고정한다.
"""
import sqlite3

import pandas as pd

from scalp_it import cli_pead_portfolio as P
from scalp_it import pead_risk
from scalp_it.cli_pead_watch import Portfolio

CODES = [f"{i:06d}" for i in range(1, 41)]
DATES = pd.bdate_range("2026-01-02", "2026-05-01")
ASOF = str(DATES[-1].date())


def _side_frame(codes, decile, base_sue):
    return pd.DataFrame(
        {
            "code": codes,
            "name": [f"종목{c}" for c in codes],
            "market": ["KOSPI"] * len(codes),
            "decile": [decile] * len(codes),
            "sue_raw": [base_sue + i for i in range(len(codes))],
            "entry_date": [pd.Timestamp("2026-04-20")] * len(codes),
            "entry_open": [10000.0] * len(codes),
            "exit_date_planned": [pd.Timestamp("2026-06-15")] * len(codes),
            "liq": [6000.0] * len(codes),
        }
    )


def _portfolio(n=4):
    longs = _side_frame(CODES[-n:], 10, 5.0)
    shorts = _side_frame(CODES[:n], 1, -5.0)
    return Portfolio(
        period="2026Q1",
        avail_date=pd.Timestamp("2026-04-17"),
        longs=longs.reset_index(drop=True),
        shorts=shorts.reset_index(drop=True),
    )


def test_build_frame_equal_weight_both_sides():
    pf = _portfolio(n=4)
    frame = P.build_frame(pf, pead_risk.RiskScan(covered=True))
    assert len(frame) == 8
    longs = frame[frame["side"] == "롱"]
    assert (longs["weight_pct"] == 25.0).all()  # 100/4
    assert set(frame["side"]) == {"롱", "숏"}


def test_risk_flag_merged_onto_longs_only():
    pf = _portfolio(n=4)
    flagged = pf.longs["code"].iloc[0]
    scan = pead_risk.RiskScan(covered=True)
    scan.hits[flagged] = [
        pead_risk.RiskHit(flagged, "20260410", "유상증자결정", "자본조달")
    ]
    frame = P.build_frame(pf, scan)
    long_row = frame[(frame["side"] == "롱") & (frame["code"] == flagged)].iloc[0]
    assert "유상증자" in long_row["risk_flag"]
    # 숏에는 리스크 컬럼을 채우지 않는다
    assert (frame[frame["side"] == "숏"]["risk_flag"] == "").all()


def test_format_report_has_both_tables_and_freshness():
    pf = _portfolio()
    scan = pead_risk.RiskScan(covered=True, db_max_date="20231229")
    frame = P.build_frame(pf, scan)
    txt = P.format_report(pf, frame, scan, ASOF)
    assert "롱 (Q10)" in txt
    assert "숏 (Q1)" in txt
    assert "리스크 오버레이" in txt
    assert "안전 아님" in txt  # 최신성 경고


def _install_db(monkeypatch, tmp_path):
    from scalp_it.cli_pead_watch import _build_events  # noqa: F401

    n = 40
    codes = CODES
    bars = pd.DataFrame(
        [
            {"code": c, "date": d, "open": 10000.0, "close": 10000.0, "trade_value": 6000.0}
            for c in codes
            for d in DATES
        ]
    )
    raw = pd.DataFrame(
        [
            {"code": c, "date": d, "open": 10000.0, "close": 10000.0}
            for c in codes
            for d in DATES
        ]
    )
    earn = pd.DataFrame(
        [
            {
                "code": c, "period": "2026Q1", "avail_date": DATES[-15],
                "op_income": 100.0 + i * 10.0, "op_income_prior": 100.0,
            }
            for i, c in enumerate(codes)
        ]
    )
    meta = pd.DataFrame(
        {
            "code": codes,
            "name": [f"종목{c}" for c in codes],
            "market": ["KOSPI"] * n,
            "sector": ["섹터"] * n,
        }
    )

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(P, "connect", lambda dsn=None: _Conn())
    monkeypatch.setattr(P, "load_earnings", lambda conn: earn)
    monkeypatch.setattr(P, "load_prices", lambda conn, s, e: bars)
    monkeypatch.setattr(P, "load_daily_bars_raw", lambda conn, s, e: raw)
    monkeypatch.setattr(P, "load_stock_master", lambda conn: meta)


def test_main_writes_csv(monkeypatch, tmp_path):
    _install_db(monkeypatch, tmp_path)
    # DART DB 없음 → 리스크 미커버(정직)
    rc = P.main(
        [
            "--asof", ASOF,
            "--out-dir", str(tmp_path),
            "--dart-db", str(tmp_path / "no.db"),
        ]
    )
    assert rc == 0
    csv = tmp_path / "pead_portfolio_2026Q1.csv"
    assert csv.exists()
    df = pd.read_csv(csv, dtype={"code": str})
    assert len(df) == 8  # 40/10 = 4/side
    assert set(df["side"]) == {"롱", "숏"}
    assert (df[df["side"] == "롱"]["weight_pct"] == 25.0).all()


def test_main_risk_scan_reads_dart_db(monkeypatch, tmp_path):
    _install_db(monkeypatch, tmp_path)
    db = tmp_path / "dart.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "create table disclosures (rcept_no text, rcept_dt text, code text, "
        "corp_name text, corp_cls text, report_nm text, material_type text, "
        "is_correction integer)"
    )
    # 롱 최상위 종목(마지막 코드)에 최근 악재
    top_long = CODES[-1]
    conn.execute(
        "insert into disclosures values (?,?,?,?,?,?,?,?)",
        ("1", DATES[-20].strftime("%Y%m%d"), top_long, "X", "Y", "유상증자결정",
         "자본조달", 0),
    )
    conn.commit()
    conn.close()

    rc = P.main(
        ["--asof", ASOF, "--out-dir", str(tmp_path), "--dart-db", str(db), "--no-save"]
    )
    assert rc == 0
