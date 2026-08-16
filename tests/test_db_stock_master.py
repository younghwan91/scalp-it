import sqlite3

import pandas as pd
import pytest

from swing_it.adapters.db import connect, load_stock_master


@pytest.fixture()
def master_db(tmp_path):
    path = tmp_path / "master.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE stocks (
            code TEXT, name TEXT, market TEXT, sector TEXT, kind TEXT
        );
        CREATE TABLE delisted_stocks (
            code TEXT, name TEXT, market TEXT, last_trade_date TEXT, naver_checked TEXT
        );
        INSERT INTO stocks VALUES
            ('005930','삼성전자','거래소','전기전자','보통주'),
            ('000660','SK하이닉스','거래소','전기전자','보통주');
        INSERT INTO delisted_stocks VALUES
            ('037550','쌍방울','코스닥','2023-06-30',NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_master_returns_expected_columns(master_db):
    df = load_stock_master(connect(master_db))
    assert list(df.columns) == ["code", "name", "market", "sector", "last_trade_date"]


def test_master_unions_listed_and_delisted(master_db):
    df = load_stock_master(connect(master_db))
    assert set(df["code"]) == {"005930", "000660", "037550"}


def test_listed_rows_have_no_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert pd.isna(df.loc["005930", "last_trade_date"])


def test_delisted_row_keeps_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert df.loc["037550", "last_trade_date"] == pd.Timestamp("2023-06-30")


def test_last_trade_date_is_datetime_dtype(master_db):
    df = load_stock_master(connect(master_db))
    assert pd.api.types.is_datetime64_any_dtype(df["last_trade_date"])
