"""결합 북 대시보드 — 합산 익스포저·비중 고정.

선정은 PEAD·저변동 각 배포 함수를 재사용하므로 여기선 노출 합산(비중 적용)과
대시보드 렌더, 로더 monkeypatch main 통합만 고정한다.
"""
import numpy as np
import pandas as pd

from scalp_it import cli_book_report as B
from scalp_it.cli_pead_watch import Portfolio as PeadPortfolio
from scalp_it.lowvol import LowvolPortfolio


def _pead(n=4):
    def side(codes, dec):
        return pd.DataFrame({
            "code": codes, "name": [f"P{c}" for c in codes],
            "market": ["KOSPI"] * len(codes), "decile": [dec] * len(codes),
            "sue_raw": [1.0] * len(codes),
            "entry_date": [pd.Timestamp("2026-01-05")] * len(codes),
            "entry_open": [10000.0] * len(codes),
            "exit_date_planned": [pd.Timestamp("2026-03-05")] * len(codes),
            "liq": [6000.0] * len(codes),
        })
    longs = side([f"L{i}" for i in range(n)], 10)
    shorts = side([f"S{i}" for i in range(n)], 1)
    return PeadPortfolio("2025Q4", pd.Timestamp("2026-01-02"),
                         longs.reset_index(drop=True), shorts.reset_index(drop=True))


def _lowvol(n=4):
    def side(codes, dec):
        return pd.DataFrame({
            "code": codes, "vol60": [0.02] * len(codes), "liq": [6000.0] * len(codes),
            "close": [5000.0] * len(codes), "decile": [dec] * len(codes),
            "vol_ann": [0.3] * len(codes), "name": [f"V{c}" for c in codes],
            "market": ["KOSDAQ"] * len(codes), "sector": ["s"] * len(codes),
        })
    longs = side([f"LV{i}" for i in range(n)], 10)
    shorts = side([f"HV{i}" for i in range(n)], 1)
    return LowvolPortfolio("2026-01", pd.Timestamp("2026-01-30"),
                           longs.reset_index(drop=True), shorts.reset_index(drop=True))


def test_notional_market_neutral_and_weighted():
    pead, lowvol = _pead(4), _lowvol(4)
    long_p = B._name_notional(pead, "longs", B.COMBO_W_PEAD)
    # PEAD 롱 다리 = book_weight/2 * 100 = 35, 4종목 동일가중 → 각 8.75
    assert np.isclose(long_p["notional"].sum(), B.COMBO_W_PEAD / 2 * 100)
    assert np.isclose(long_p["notional"].iloc[0], B.COMBO_W_PEAD / 2 * 100 / 4)
    long_v = B._name_notional(lowvol, "longs", B.COMBO_W_LOWVOL)
    assert np.isclose(long_v["notional"].sum(), B.COMBO_W_LOWVOL / 2 * 100)


def test_dashboard_has_both_strategies_and_exposure():
    txt = B.format_dashboard(_pead(4), _lowvol(4), "2026-01-30")
    assert "PEAD" in txt and "저변동" in txt
    assert "합산 익스포저" in txt
    assert "롱 북" in txt and "숏 북" in txt
    assert f"{B.COMBO_W_PEAD:.0%}" in txt


def test_dashboard_handles_missing_strategy():
    txt = B.format_dashboard(None, _lowvol(4), "2026-01-30")
    assert "형성 불가" in txt
    # 저변동만으로도 합산 북이 그려진다
    assert "롱 북" in txt


def test_overlap_marked():
    # 같은 코드가 PEAD 롱 & 저변동 숏 → 상계 표시
    pead = _pead(4)
    lowvol = _lowvol(4)
    lowvol.shorts.loc[0, "code"] = pead.longs["code"].iloc[0]
    txt = B.format_dashboard(pead, lowvol, "2026-01-30")
    assert "상계" in txt


def _install_db(monkeypatch):
    n = 40
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    dates = pd.bdate_range("2025-06-02", "2026-01-30")
    rng = np.random.default_rng(1)
    rows = []
    for i, c in enumerate(codes):
        vol = 0.005 + 0.002 * i
        price = 10000.0
        for d in dates:
            price *= 1.0 + rng.normal(0, vol)
            rows.append({"code": c, "date": d, "open": price, "close": price,
                         "trade_value": 6000.0})
    adj = pd.DataFrame(rows)
    raw = adj[["code", "date", "open", "close"]].copy()
    earn = pd.DataFrame([
        {"code": c, "period": "2025Q4", "avail_date": dates[-15],
         "op_income": 100.0 + i * 10.0, "op_income_prior": 100.0}
        for i, c in enumerate(codes)
    ])
    meta = pd.DataFrame({"code": codes, "name": [f"종목{c}" for c in codes],
                         "market": ["KOSPI"] * n, "sector": ["s"] * n})

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(B, "connect", lambda dsn=None: _Conn())
    monkeypatch.setattr(B, "load_earnings", lambda conn: earn)
    monkeypatch.setattr(B, "load_prices", lambda conn, s, e: adj)
    monkeypatch.setattr(B, "load_daily_bars_raw", lambda conn, s, e: raw)
    monkeypatch.setattr(B, "load_stock_master", lambda conn: meta)


def test_main_runs(monkeypatch, capsys):
    _install_db(monkeypatch)
    rc = B.main(["--asof", "2026-01-30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "결합 북 대시보드" in out
    assert "합산 익스포저" in out
