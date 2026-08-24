"""`scalp-pead-report` — 전진 기록 손익 채점.

DB 없이 순수 함수(``evaluate_log``·``summarize``)만 고정한다. 청산확정/보유중/
가격없음 세 상태와 롱숏 spread 계산, 빈 로그를 검증한다.
"""
import pandas as pd

from scalp_it import cli_pead_report as R

ASOF = pd.Timestamp("2026-03-01")


def _log():
    """롱 하나(청산 도래·상승), 숏 하나(보유중·상승=숏손실)."""
    return pd.DataFrame(
        [
            {
                "strategy": "pead", "period": "2026Q1", "side": "long",
                "code": "000001", "name": "롱A", "entry_open": 100.0,
                "entry_date": "2026-01-05", "exit_date_planned": "2026-02-15",
            },
            {
                "strategy": "pead", "period": "2026Q1", "side": "short",
                "code": "000002", "name": "숏B", "entry_open": 200.0,
                "entry_date": "2026-01-05", "exit_date_planned": "2026-05-15",
            },
        ]
    )


def _prices():
    rows = [
        # 롱: 청산일(02-15) 종가 110 → +10%
        {"code": "000001", "date": pd.Timestamp("2026-02-15"), "close": 110.0},
        {"code": "000001", "date": pd.Timestamp("2026-03-01"), "close": 999.0},
        # 숏: asof(03-01) 최근 종가 220 → 원 +10% → 숏 pnl -10%
        {"code": "000002", "date": pd.Timestamp("2026-03-01"), "close": 220.0},
    ]
    return pd.DataFrame(rows)


def test_realized_uses_exit_date_price():
    scored = R.evaluate_log(_log(), _prices(), ASOF)
    lo = scored[scored["code"] == "000001"].iloc[0]
    assert lo["status"] == "realized"
    assert lo["exit_price"] == 110.0  # asof(999)가 아니라 청산일 종가
    assert abs(lo["pnl_gross"] - 0.10) < 1e-9


def test_held_uses_latest_price_and_short_sign():
    scored = R.evaluate_log(_log(), _prices(), ASOF)
    sh = scored[scored["code"] == "000002"].iloc[0]
    assert sh["status"] == "held"
    # 원 +10% 상승 → 숏은 손실 -10%
    assert abs(sh["pnl_gross"] + 0.10) < 1e-9


def test_no_price_marked():
    scored = R.evaluate_log(_log(), pd.DataFrame(columns=["code", "date", "close"]), ASOF)
    assert set(scored["status"]) == {"no_price"}
    assert scored["pnl_gross"].isna().all()


def test_summary_spread_is_long_plus_short():
    scored = R.evaluate_log(_log(), _prices(), ASOF)
    summ = R.summarize(scored).iloc[0]
    assert summ["n_long"] == 1
    assert summ["n_short"] == 1
    # 롱 +10% + 숏 -10% = 0% (gross spread)
    assert abs(summ["longshort_gross"]) < 1e-9
    # net 은 비용 차감이라 gross 보다 작다
    assert summ["longshort_net"] < summ["longshort_gross"]


def test_empty_log_reports_gracefully():
    txt = R.format_report(pd.DataFrame(), pd.DataFrame(), ASOF)
    assert "비어 있다" in txt


def test_format_report_shows_quarter_and_expectation():
    scored = R.evaluate_log(_log(), _prices(), ASOF)
    summ = R.summarize(scored)
    txt = R.format_report(scored, summ, ASOF)
    assert "2026Q1" in txt
    assert "4.47%" in txt  # 백테스트 기대 병기
    assert "누적" in txt
