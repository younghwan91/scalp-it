"""저변동성 포트폴리오 선정·표·CSV — 배포 산출물 고정.

선정은 :mod:`scalp_it.lowvol.select_lowvol_portfolio` 한 곳에서만 정의된다. 여기선
데실 방향(Q10=저변동=롱), 동일가중 비중, CSV 저장, 로더 monkeypatch main 통합을 고정한다.
"""
import numpy as np
import pandas as pd

from scalp_it import cli_lowvol_portfolio as C
from scalp_it.lowvol import build_frame, select_lowvol_portfolio

N = 40
CODES = [f"{i:06d}" for i in range(1, N + 1)]
DATES = pd.bdate_range("2025-06-02", "2026-01-30")  # >120 거래일
ASOF = str(DATES[-1].date())


def _bars():
    """종목 i 는 변동성이 코드 순으로 증가 — 저변동(코드 작음)=Q10 롱이 되도록."""
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(CODES):
        vol = 0.005 + 0.002 * i          # 코드 클수록 고변동
        price = 10000.0
        for d in DATES:
            price *= 1.0 + rng.normal(0, vol)
            rows.append({"code": c, "date": d, "open": price, "close": price,
                         "trade_value": 6000.0})
    return pd.DataFrame(rows)


def _meta():
    return pd.DataFrame({
        "code": CODES,
        "name": [f"종목{c}" for c in CODES],
        "market": ["KOSPI"] * N,
        "sector": ["섹터"] * N,
    })


def test_low_vol_is_long_high_vol_is_short():
    pf = select_lowvol_portfolio(_bars(), _meta(), pd.Timestamp(ASOF))
    assert pf is not None
    # 롱(Q10)의 평균 변동성 < 숏(Q1)의 평균 변동성
    assert pf.longs["vol60"].mean() < pf.shorts["vol60"].mean()
    # 저변동 종목(코드 앞쪽)이 롱, 고변동(코드 뒤쪽)이 숏
    assert pf.longs["code"].min() < pf.shorts["code"].min()


def test_build_frame_equal_weight_both_sides():
    pf = select_lowvol_portfolio(_bars(), _meta(), pd.Timestamp(ASOF))
    frame = build_frame(pf)
    longs = frame[frame["side"] == "롱"]
    shorts = frame[frame["side"] == "숏"]
    assert len(longs) == 4 and len(shorts) == 4  # 40/10 = 4/side
    assert np.isclose(longs["weight_pct"].iloc[0], 25.0)
    assert (longs["weight_pct"] == longs["weight_pct"].iloc[0]).all()


def test_spac_reit_excluded():
    meta = _meta()
    meta.loc[0, "name"] = "테스트리츠"
    meta.loc[1, "name"] = "테스트스팩3호"
    pf = select_lowvol_portfolio(_bars(), meta, pd.Timestamp(ASOF))
    codes = set(pf.longs["code"]) | set(pf.shorts["code"])
    assert CODES[0] not in codes and CODES[1] not in codes


def _install_db(monkeypatch):
    bars, meta = _bars(), _meta()

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(C, "connect", lambda dsn=None: _Conn())
    monkeypatch.setattr(C, "load_prices", lambda conn, s, e: bars)
    monkeypatch.setattr(C, "load_stock_master", lambda conn: meta)


def test_main_writes_csv(monkeypatch, tmp_path):
    _install_db(monkeypatch)
    rc = C.main(["--asof", ASOF, "--out-dir", str(tmp_path)])
    assert rc == 0
    csvs = list(tmp_path.glob("lowvol_portfolio_*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0], dtype={"code": str})
    assert len(df) == 8
    assert set(df["side"]) == {"롱", "숏"}


def test_main_no_save(monkeypatch, tmp_path):
    _install_db(monkeypatch)
    rc = C.main(["--asof", ASOF, "--out-dir", str(tmp_path), "--no-save"])
    assert rc == 0
    assert list(tmp_path.glob("*.csv")) == []
