import sqlite3

import pandas as pd
import pytest

from swing_it.adapters.db import connect, load_prices, load_supply_demand


@pytest.fixture()
def fake_db(tmp_path):
    """실 DB 와 같은 컬럼 구조를 가진 sqlite fixture."""
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE daily_bars_adjusted (
            code TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, trade_value REAL, source TEXT
        );
        CREATE TABLE supply_demand (
            code TEXT, date TEXT, close REAL, flu_rt REAL, acc_trde_qty INTEGER,
            individual INTEGER, foreign_ INTEGER, institution INTEGER,
            fnnc_invt INTEGER, insrnc INTEGER, invtrt INTEGER, bank INTEGER,
            penfnd_etc INTEGER, samo_fund INTEGER, natn INTEGER, etc_corp INTEGER,
            source TEXT
        );
        INSERT INTO daily_bars_adjusted VALUES
            ('005930','2026-01-02',100,110,90,105,1000,5000,'kiwoom'),
            ('005930','2026-01-03',105,115,95,110,1200,6000,'kiwoom'),
            ('000660','2026-01-02',200,210,190,205,500,3000,'kiwoom');
        INSERT INTO supply_demand VALUES
            ('005930','2026-01-02',105,1.0,1000,-500,300,200,0,0,0,0,100,0,0,0,'kiwoom'),
            ('005930','2026-01-03',110,4.8,1200,-700,500,200,0,0,0,0,150,0,0,0,'kiwoom');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_connect_opens_sqlite(fake_db):
    conn = connect(fake_db)
    assert conn.execute("select count(*) from daily_bars_adjusted").fetchone()[0] == 3


def test_load_prices_returns_typed_frame(fake_db):
    df = load_prices(connect(fake_db), "2026-01-01", "2026-01-31")
    assert list(df.columns) == ["code", "date", "open", "close", "trade_value"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_prices_respects_date_range(fake_db):
    df = load_prices(connect(fake_db), "2026-01-03", "2026-01-31")
    assert len(df) == 1
    assert df.iloc[0]["code"] == "005930"


def test_load_supply_demand_returns_investor_columns(fake_db):
    df = load_supply_demand(connect(fake_db), "2026-01-01", "2026-01-31")
    assert list(df.columns) == [
        "code", "date", "individual", "foreign_", "institution",
        "penfnd_etc", "acc_trde_qty",
    ]
    assert len(df) == 2
