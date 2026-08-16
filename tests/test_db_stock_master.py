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
            ('037550','쌍방울','코스닥','2023-06-30',NULL),
            -- 실 DB 에서 폐지 종목의 89% 가 last_trade_date 가 NULL 이다
            ('000010','조흥은행','유가증권',NULL,NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_master_returns_expected_columns(master_db):
    df = load_stock_master(connect(master_db))
    assert list(df.columns) == [
        "code", "name", "market", "sector", "last_trade_date", "is_delisted",
    ]


def test_master_unions_listed_and_delisted(master_db):
    df = load_stock_master(connect(master_db))
    assert set(df["code"]) == {"005930", "000660", "037550", "000010"}


def test_listed_rows_have_no_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert pd.isna(df.loc["005930", "last_trade_date"])


def test_is_delisted_comes_from_source_table_not_from_date():
    """폐지 여부는 출처 테이블로 판정한다.

    실 DB 에서 ``delisted_stocks`` 의 4,162 건 중 3,693 건(89%)이
    ``last_trade_date`` 가 NULL 이다. 날짜 유무로 상장 여부를 판정하면
    폐지 종목 대부분을 상장으로 오인하고 생존편향이 되살아난다.
    """
    import sqlite3
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE stocks (code TEXT, name TEXT, market TEXT, sector TEXT, kind TEXT);
            CREATE TABLE delisted_stocks (
                code TEXT, name TEXT, market TEXT, last_trade_date TEXT, naver_checked TEXT
            );
            INSERT INTO stocks VALUES ('005930','삼성전자','거래소','전기전자','보통주');
            INSERT INTO delisted_stocks VALUES ('000010','조흥은행','유가증권',NULL,NULL);
            """
        )
        conn.commit()
        conn.close()
        df = load_stock_master(connect(str(path))).set_index("code")

    assert df.loc["005930", "is_delisted"] is False or not df.loc["005930", "is_delisted"]
    # 날짜가 없어도 폐지로 잡혀야 한다
    assert pd.isna(df.loc["000010", "last_trade_date"])
    assert bool(df.loc["000010", "is_delisted"]) is True


def test_listed_rows_are_not_marked_delisted(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert not bool(df.loc["005930", "is_delisted"])
    assert bool(df.loc["037550", "is_delisted"])


def test_delisted_row_keeps_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert df.loc["037550", "last_trade_date"] == pd.Timestamp("2023-06-30")


def test_last_trade_date_is_datetime_dtype(master_db):
    df = load_stock_master(connect(master_db))
    assert pd.api.types.is_datetime64_any_dtype(df["last_trade_date"])
