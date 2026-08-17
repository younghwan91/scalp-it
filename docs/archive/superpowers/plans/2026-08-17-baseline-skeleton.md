# 계획 1: 베이스라인 워킹 스켈레톤 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 뉴스·LLM 없이 수급 데이터만으로 "거래대금 급증 이벤트 → D+20~40 초과수익" 파이프라인 전체를 세우고, kr-quant 음성대조 게이트로 판정한 베이스라인 성과를 산출한다.

**Architecture:** 순수 함수 계층(pandas DataFrame in → DataFrame out)과 I/O 계층(`adapters/`)을 분리한다. 모든 학습·검증 로직은 DB 를 모른 채 DataFrame 만 받으므로 sqlite fixture 로 단위 테스트가 가능하다. 실 DB 는 CLI 스모크 실행에서만 닿는다.

**Tech Stack:** Python ≥3.10, uv, pandas ≥2.0, scikit-learn, psycopg2-binary, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-08-17-scalp-it-design.md`

## Global Constraints

- **수정주가만 사용한다.** 가격은 `daily_bars_adjusted` 에서만 읽는다. `daily_bars` 는 분할 미반영이며 절대 쓰지 않는다.
- **point-in-time 엄수.** 시점 t 의 피처는 t 시점에 알 수 있는 정보만 쓴다. 롤링 통계는 반드시 `.shift(1)` 후 계산한다.
- **유동성 하한.** 이벤트 후보는 진입 직전 20거래일 평균 `trade_value` ≥ 5000 (단위: 백만원 = 50억원). 스펙 §5.
- **드리프트를 자르지 않는다.** 라벨은 D+21 ~ D+41 구간 보유. 중간 손절·이익실현 없음. 스펙 §2.
- **음성대조 없이 승격 없음.** 모든 성과 주장은 랜덤 진입 널 대비로 보고한다. 스펙 §2.
- **데이터 커밋 금지.** 결과물은 `data/`·`results/` 아래 두며 `.gitignore` 로 차단돼 있다.
- **DB 는 읽기 전용.** scalp-it 은 어떤 테이블에도 쓰지 않는다.
- `trade_value` 단위는 **백만원**. `supply_demand` 순매수 컬럼 단위는 **주식 수**(확인 필요 → Task 2 에서 검증).

## 데이터 실측 (2026-08-17 확인)

| 테이블 | 기간 | 비고 |
|---|---|---|
| `daily_bars_adjusted` | 2016-09-09 ~ 2026-08-14, 3,093 종목 | 수정주가 반영본 |
| `supply_demand` | 2016-09-09 ~ 2026-08-14 | 11개 투자자 분류 |
| `earnings` | `avail_date`, `knowledge_date` 보유 | point-in-time 정상 |
| `delisted_stocks` | 4,162 종목 | 생존편향 대응 |
| `short_selling` | 2024-05-09 ~ | **2년 3개월뿐** — 후기 전용 피처 |
| `credit_balance` | 2026-02-20 ~ | **6개월뿐 — 이번 범위에서 제외** |
| `sector_index` | 2024-01-26 ~ | **2년 7개월뿐 — 업종 벤치마크 불가** |

**결정:** 초과수익 벤치마크는 업종이 아니라 **당일 유니버스 동일가중 평균**(횡단면 디민)으로 한다. `sector_index` 가 전 기간을 못 덮기 때문이다.

**접속:** `KR_QUANT_DB` 환경변수. 컨테이너 `quant-airflow-timescaledb-1`, DB `kr_quant`, 유저 `kr_quant`.

## File Structure

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | 패키징·의존성·ruff·pytest 설정 |
| `src/scalp_it/adapters/db.py` | DB 접속과 읽기 쿼리. **여기서만 SQL 을 쓴다** |
| `src/scalp_it/events.py` | 유니버스 필터 + 거래대금 급증 이벤트 추출 (순수 함수) |
| `src/scalp_it/labels.py` | D+1 진입 → D+21~41 초과수익 라벨 (순수 함수) |
| `src/scalp_it/features/supply.py` | 수급 피처 (순수 함수) |
| `src/scalp_it/dataset.py` | 이벤트 + 피처 + 라벨 조립 |
| `src/scalp_it/model/baseline.py` | walk-forward 학습·예측 |
| `src/scalp_it/validate/gate.py` | 음성대조 널 + 판정 리포트 |
| `src/scalp_it/cli.py` | `scalp-baseline` 엔트리포인트 |
| `tests/` | 위 각 모듈 대응 테스트 |

---

### Task 1: 패키지 스캐폴딩과 DB 어댑터

**Files:**
- Create: `pyproject.toml`
- Create: `src/scalp_it/__init__.py`
- Create: `src/scalp_it/adapters/__init__.py`
- Create: `src/scalp_it/adapters/db.py`
- Test: `tests/test_db_adapter.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `connect(dsn: str | None = None) -> Connection` — `dsn` 생략 시 `KR_QUANT_DB` 환경변수. `postgresql://`/`postgres://` 면 psycopg2, 아니면 sqlite3.
  - `load_prices(conn, start: str, end: str) -> pd.DataFrame` — 컬럼 `[code, date, open, close, trade_value]`, `date` 는 `datetime64[ns]`
  - `load_supply_demand(conn, start: str, end: str) -> pd.DataFrame` — 컬럼 `[code, date, individual, foreign_, institution, penfnd_etc, acc_trde_qty]`

- [ ] **Step 1: `pyproject.toml` 작성**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "scalp-it"
version = "0.1.0"
description = "한국 주식 2~8주 스윙 후보 리서치 툴 — 뉴스 재료 강도 + 수급 확증"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Younghwan Chae" }]
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
pg = ["psycopg2-binary>=2.9"]
dev = ["pytest>=8.0", "ruff==0.16.2", "psycopg2-binary>=2.9"]

[project.scripts]
scalp-baseline = "scalp_it.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/scalp_it"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_db_adapter.py`:

```python
import sqlite3

import pandas as pd
import pytest

from scalp_it.adapters.db import connect, load_prices, load_supply_demand


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
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/test_db_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it'`

- [ ] **Step 4: 어댑터 구현**

`src/scalp_it/__init__.py`:

```python
"""scalp-it — 한국 주식 2~8주 스윙 후보 리서치 툴."""

__version__ = "0.1.0"
```

`src/scalp_it/adapters/__init__.py`:

```python
"""외부 시스템 접근을 격리하는 계층.

kr-quant / kiwoom / krx-news / DB 에 손을 뻗는 코드는 전부 여기에만 둔다.
나머지 모듈은 이 레포들이 존재하는지도 모른다 (스펙 §8).
"""
```

`src/scalp_it/adapters/db.py`:

```python
"""DB 읽기 어댑터 — SQL 은 이 파일에만 존재한다.

읽기 전용이다. scalp-it 은 어떤 테이블에도 쓰지 않는다.
가격은 반드시 ``daily_bars_adjusted`` 에서 읽는다. ``daily_bars`` 는 분할 미반영이다.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pandas as pd

_PG_PREFIXES = ("postgresql://", "postgres://")

_PRICE_COLUMNS = ["code", "date", "open", "close", "trade_value"]
_SUPPLY_COLUMNS = [
    "code", "date", "individual", "foreign_", "institution",
    "penfnd_etc", "acc_trde_qty",
]


def connect(dsn: str | None = None) -> Any:
    """DSN 또는 ``KR_QUANT_DB`` 로 접속한다. postgres 면 psycopg2, 아니면 sqlite."""
    target = dsn or os.environ.get("KR_QUANT_DB")
    if not target:
        raise ValueError("DSN 이 없다. 인자로 넘기거나 KR_QUANT_DB 를 설정하라.")
    if target.startswith(_PG_PREFIXES):
        import psycopg2  # 지연 임포트 — sqlite 전용 사용자는 설치할 필요가 없다

        return psycopg2.connect(target)
    return sqlite3.connect(target)


def _placeholder(conn: Any) -> str:
    return "%s" if conn.__class__.__module__.startswith("psycopg2") else "?"


def _query(conn: Any, sql: str, params: tuple, columns: list[str]) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=columns)
    cur.close()
    df["date"] = pd.to_datetime(df["date"])
    for col in columns:
        if col not in ("code", "date"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_prices(conn: Any, start: str, end: str) -> pd.DataFrame:
    """``daily_bars_adjusted`` 에서 수정주가를 읽는다 (분할 반영본)."""
    ph = _placeholder(conn)
    sql = (
        "select code, date, open, close, trade_value from daily_bars_adjusted "
        f"where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), _PRICE_COLUMNS)


def load_supply_demand(conn: Any, start: str, end: str) -> pd.DataFrame:
    """투자자별 순매수를 읽는다. 개인·외국인·기관계·연기금만 쓴다."""
    ph = _placeholder(conn)
    sql = (
        "select code, date, individual, foreign_, institution, penfnd_etc, acc_trde_qty "
        f"from supply_demand where date >= {ph} and date <= {ph} order by code, date"
    )
    return _query(conn, sql, (start, end), _SUPPLY_COLUMNS)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_db_adapter.py -v`
Expected: 4 passed

- [ ] **Step 6: 실 DB 스모크 확인**

Run:
```bash
docker exec quant-airflow-timescaledb-1 psql -U kr_quant -d kr_quant -tAc \
  "select count(*) from daily_bars_adjusted where date between '2026-08-01' and '2026-08-14'"
```
Expected: 0 이 아닌 수. (파이썬에서 직접 붙는 경로는 Task 7 CLI 에서 확인한다.)

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml src/scalp_it tests/test_db_adapter.py
git commit -m "feat(adapters): 읽기 전용 DB 어댑터 — 수정주가·수급 로더"
```

---

### Task 2: 유니버스 필터와 거래대금 급증 이벤트

**Files:**
- Create: `src/scalp_it/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `load_prices` 가 주는 `[code, date, open, close, trade_value]`
- Produces:
  - `add_liquidity_stats(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame` — 입력에 `adv` (직전 `window` 일 평균 거래대금, **shift(1) 적용**) 와 `tv_ratio` (`trade_value / adv`) 컬럼 추가
  - `extract_events(prices: pd.DataFrame, *, adv_floor: float = 5000.0, tv_multiple: float = 3.0, window: int = 20) -> pd.DataFrame` — 컬럼 `[code, date, adv, tv_ratio]`, 조건 `adv >= adv_floor and tv_ratio >= tv_multiple`

`adv` 가 shift(1) 인 것이 핵심이다. 당일 거래대금을 자기 평균에 포함시키면 lookahead 다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_events.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.events import add_liquidity_stats, extract_events


def _prices(code: str, trade_values: list[float]) -> pd.DataFrame:
    n = len(trade_values)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [100.0] * n,
            "close": [100.0] * n,
            "trade_value": trade_values,
        }
    )


def test_adv_excludes_current_day():
    df = _prices("A", [10.0, 20.0, 30.0, 999.0])
    out = add_liquidity_stats(df, window=3)
    # 4번째 행의 adv 는 앞의 3일 평균 (10+20+30)/3 = 20 이어야 한다. 999 가 섞이면 lookahead.
    assert out.iloc[3]["adv"] == 20.0
    assert out.iloc[3]["tv_ratio"] == 999.0 / 20.0


def test_adv_is_nan_before_window_fills():
    df = _prices("A", [10.0, 20.0, 30.0])
    out = add_liquidity_stats(df, window=3)
    assert np.isnan(out.iloc[0]["adv"])
    assert np.isnan(out.iloc[2]["adv"])  # 앞 2일뿐이라 window=3 미충족


def test_stats_computed_per_code_independently():
    df = pd.concat([_prices("A", [10.0] * 4), _prices("B", [1000.0] * 4)], ignore_index=True)
    out = add_liquidity_stats(df, window=3)
    a = out[out["code"] == "A"].iloc[3]
    b = out[out["code"] == "B"].iloc[3]
    assert a["adv"] == 10.0
    assert b["adv"] == 1000.0


def test_extract_events_applies_both_gates():
    # adv 6000 (>= 5000 통과), 마지막 날 거래대금 24000 → 배수 4.0 (>= 3.0 통과)
    df = _prices("A", [6000.0, 6000.0, 6000.0, 24000.0])
    events = extract_events(df, adv_floor=5000.0, tv_multiple=3.0, window=3)
    assert list(events.columns) == ["code", "date", "adv", "tv_ratio"]
    assert len(events) == 1
    assert events.iloc[0]["tv_ratio"] == 4.0


def test_extract_events_rejects_illiquid_even_when_spiking():
    # 배수는 10배지만 adv 가 100 뿐이라 유동성 하한에서 탈락해야 한다
    df = _prices("A", [100.0, 100.0, 100.0, 1000.0])
    events = extract_events(df, adv_floor=5000.0, tv_multiple=3.0, window=3)
    assert events.empty
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.events'`

- [ ] **Step 3: 구현**

`src/scalp_it/events.py`:

```python
"""유니버스 필터와 이벤트 정의 — 순수 함수, DB 를 모른다.

이벤트는 '거래대금 급증'이다. 스펙 §4 의 '핫함' 정의 중 과거·현재 양쪽에서
동일하게 계산 가능한 축이며, 뉴스 없이도 성립한다(계획 1 베이스라인).
"""

from __future__ import annotations

import pandas as pd


def add_liquidity_stats(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """종목별 직전 ``window`` 일 평균 거래대금(``adv``)과 당일 배수(``tv_ratio``)를 붙인다.

    ``adv`` 는 반드시 ``shift(1)`` 이후 계산한다. 당일 거래대금을 자기 평균에
    포함하면 lookahead 이며, 배수가 구조적으로 축소된다.
    """
    out = prices.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)["trade_value"]
    out["adv"] = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )
    out["tv_ratio"] = out["trade_value"] / out["adv"]
    return out


def extract_events(
    prices: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    window: int = 20,
) -> pd.DataFrame:
    """유동성 하한을 통과하면서 거래대금이 급증한 (종목, 날짜) 를 추린다.

    Args:
        adv_floor: 직전 평균 거래대금 하한. 단위 백만원. 기본 5000 = 50억원
            (kr-quant PEAD 검증에서 필수였던 point-in-time 유동성 하한과 같은 값).
        tv_multiple: 평균 대비 당일 거래대금 배수 하한.
        window: 평균 계산 구간(거래일).
    """
    stats = add_liquidity_stats(prices, window=window)
    hit = (stats["adv"] >= adv_floor) & (stats["tv_ratio"] >= tv_multiple)
    return stats.loc[hit, ["code", "date", "adv", "tv_ratio"]].reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_events.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/events.py tests/test_events.py
git commit -m "feat(events): 거래대금 급증 이벤트 추출 + point-in-time 유동성 하한"
```

---

### Task 3: D+20~40 초과수익 라벨

**Files:**
- Create: `src/scalp_it/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `load_prices` 의 `[code, date, open, close]`, `extract_events` 의 `[code, date]`
- Produces:
  - `forward_return(prices: pd.DataFrame, *, entry_offset: int = 1, hold_start: int = 20, hold_end: int = 40) -> pd.DataFrame` — 컬럼 `[code, date, entry_price, exit_price, fwd_return]`. 진입은 `date` 로부터 `entry_offset` 거래일 뒤의 **시가**, 청산은 `hold_end` 거래일 뒤의 **종가**. `hold_start` 는 최소 보유 확인용으로 데이터가 모자라면 `NaN`.
  - `excess_return(fwd: pd.DataFrame) -> pd.DataFrame` — `fwd_return` 에서 **같은 진입일 유니버스 동일가중 평균**을 뺀 `excess_return` 컬럼 추가

업종 지수가 전 기간을 못 덮으므로(2024-01-26~) 벤치마크는 횡단면 평균이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_labels.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.labels import excess_return, forward_return


def _series(code: str, opens: list[float], closes: list[float]) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": opens,
            "close": closes,
        }
    )


def test_entry_is_next_day_open_and_exit_is_hold_end_close():
    # 0일차 이벤트 → 1일차 시가 진입(=110), 3일차 종가 청산(=132) → 20%
    opens = [100.0, 110.0, 120.0, 130.0]
    closes = [105.0, 115.0, 125.0, 132.0]
    fwd = forward_return(_series("A", opens, closes), entry_offset=1, hold_start=2, hold_end=3)
    row = fwd.iloc[0]
    assert row["entry_price"] == 110.0
    assert row["exit_price"] == 132.0
    assert row["fwd_return"] == (132.0 - 110.0) / 110.0


def test_rows_without_enough_future_are_nan():
    opens = [100.0, 110.0, 120.0, 130.0]
    closes = [105.0, 115.0, 125.0, 132.0]
    fwd = forward_return(_series("A", opens, closes), entry_offset=1, hold_start=2, hold_end=3)
    # 마지막 행은 미래가 없다
    assert np.isnan(fwd.iloc[-1]["fwd_return"])


def test_excess_return_demeans_within_entry_date():
    fwd = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "entry_price": [100.0, 100.0, 100.0],
            "exit_price": [110.0, 120.0, 130.0],
            "fwd_return": [0.10, 0.20, 0.30],
        }
    )
    out = excess_return(fwd)
    # 평균 0.20 → 초과수익 -0.10 / 0.00 / +0.10
    assert np.allclose(out["excess_return"].tolist(), [-0.10, 0.0, 0.10])


def test_excess_return_ignores_nan_rows_in_mean():
    fwd = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "entry_price": [100.0, 100.0, 100.0],
            "exit_price": [110.0, 130.0, np.nan],
            "fwd_return": [0.10, 0.30, np.nan],
        }
    )
    out = excess_return(fwd)
    assert np.allclose(out["excess_return"].iloc[:2].tolist(), [-0.10, 0.10])
    assert np.isnan(out["excess_return"].iloc[2])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.labels'`

- [ ] **Step 3: 구현**

`src/scalp_it/labels.py`:

```python
"""라벨 생성 — 순수 함수.

라벨은 D+1 시가 진입 → D+``hold_end`` 종가 청산의 초과수익이다.
중간 손절·이익실현은 없다. 프랍 스윙 탐색에서 PEAD-집중을 죽인 직접 원인이
10% 손절 + 10일 캡으로 드리프트를 자른 것이었다 (스펙 §2).
"""

from __future__ import annotations

import pandas as pd


def forward_return(
    prices: pd.DataFrame,
    *,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """(종목, 날짜) 마다 진입가·청산가·수익률을 계산한다.

    Args:
        entry_offset: 이벤트일로부터 진입까지의 거래일 수. 1 = 다음날 시가.
        hold_start: 최소 보유 거래일. 현재는 데이터 충분성 확인에만 쓴다.
        hold_end: 청산 시점(이벤트일로부터의 거래일 수). 종가로 청산.
    """
    out = prices.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)
    out["entry_price"] = grouped["open"].shift(-entry_offset)
    out["exit_price"] = grouped["close"].shift(-hold_end)
    # 보유 구간 자체가 확보되지 않은 행은 라벨을 만들지 않는다
    has_min_hold = grouped["close"].shift(-hold_start).notna()
    out.loc[~has_min_hold, "exit_price"] = pd.NA
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["fwd_return"] = (out["exit_price"] - out["entry_price"]) / out["entry_price"]
    return out[["code", "date", "entry_price", "exit_price", "fwd_return"]].reset_index(drop=True)


def excess_return(fwd: pd.DataFrame) -> pd.DataFrame:
    """같은 날짜 유니버스 동일가중 평균 대비 초과수익을 붙인다.

    업종 지수(``sector_index``)가 2024-01-26 부터만 존재해 전 기간 벤치마크로
    쓸 수 없으므로 횡단면 디민으로 대신한다 (계획 1 데이터 실측).
    """
    out = fwd.copy()
    market = out.groupby("date")["fwd_return"].transform("mean")
    out["excess_return"] = out["fwd_return"] - market
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_labels.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/labels.py tests/test_labels.py
git commit -m "feat(labels): D+1 진입 → D+40 청산 초과수익 라벨"
```

---

### Task 4: 수급 피처

**Files:**
- Create: `src/scalp_it/features/__init__.py`
- Create: `src/scalp_it/features/supply.py`
- Test: `tests/test_features_supply.py`

**Interfaces:**
- Consumes: `load_supply_demand` 의 `[code, date, individual, foreign_, institution, penfnd_etc, acc_trde_qty]`
- Produces:
  - `supply_features(sd: pd.DataFrame, *, window: int = 5) -> pd.DataFrame` — 컬럼 `[code, date, foreign_net_5d, inst_net_5d, indiv_net_5d, pension_net_5d, foreign_net_ratio, inst_net_ratio, smart_money_net_5d]`

정의:
- `*_net_5d` — 해당 투자자의 직전 `window` 일 순매수 합 (**당일 포함**. 이벤트일 종가 이후 확정되며 D+1 시가 진입 시점에는 알려져 있다)
- `*_net_ratio` — `*_net_5d / (직전 window 일 acc_trde_qty 합)`
- `smart_money_net_5d` — `foreign_net_5d + inst_net_5d`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_features_supply.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.features.supply import supply_features


def _sd(code: str, foreign: list[int], inst: list[int], indiv: list[int], vol: list[int]):
    n = len(foreign)
    return pd.DataFrame(
        {
            "code": [code] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "individual": indiv,
            "foreign_": foreign,
            "institution": inst,
            "penfnd_etc": [0] * n,
            "acc_trde_qty": vol,
        }
    )


def test_net_sums_include_current_day():
    df = _sd("A", foreign=[10, 20, 30], inst=[1, 2, 3], indiv=[-11, -22, -33], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["foreign_net_5d"] == 60  # 10+20+30, 당일 포함
    assert out.iloc[2]["inst_net_5d"] == 6


def test_window_not_filled_is_nan():
    df = _sd("A", foreign=[10, 20], inst=[1, 2], indiv=[-11, -22], vol=[100, 100])
    out = supply_features(df, window=3)
    assert np.isnan(out.iloc[0]["foreign_net_5d"])
    assert np.isnan(out.iloc[1]["foreign_net_5d"])


def test_ratio_divides_by_volume_sum():
    df = _sd("A", foreign=[10, 20, 30], inst=[0, 0, 0], indiv=[0, 0, 0], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["foreign_net_ratio"] == 60 / 300


def test_smart_money_is_foreign_plus_institution():
    df = _sd("A", foreign=[10, 20, 30], inst=[1, 2, 3], indiv=[0, 0, 0], vol=[100, 100, 100])
    out = supply_features(df, window=3)
    assert out.iloc[2]["smart_money_net_5d"] == 66


def test_features_are_per_code():
    df = pd.concat(
        [
            _sd("A", [10, 10, 10], [0, 0, 0], [0, 0, 0], [100, 100, 100]),
            _sd("B", [99, 99, 99], [0, 0, 0], [0, 0, 0], [100, 100, 100]),
        ],
        ignore_index=True,
    )
    out = supply_features(df, window=3)
    assert out[out["code"] == "A"].iloc[2]["foreign_net_5d"] == 30
    assert out[out["code"] == "B"].iloc[2]["foreign_net_5d"] == 297
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_features_supply.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.features'`

- [ ] **Step 3: 구현**

`src/scalp_it/features/__init__.py`:

```python
"""피처 계산 — 전부 순수 함수. DataFrame in → DataFrame out."""
```

`src/scalp_it/features/supply.py`:

```python
"""수급 피처 — 텍스트와 독립된 두 번째 증거 (스펙 §3).

핵심 가설: 재료가 강하다면 기관·외국인이 따라 들어온다. 기사 텍스트만으로
강도를 판단하면 결국 기자의 어조를 읽는 것에 가깝다.

당일 순매수를 포함하는 이유: 수급은 장마감 후 확정되고 진입은 D+1 시가이므로
시점 안전하다 (스펙 §3).
"""

from __future__ import annotations

import pandas as pd

_INVESTORS = {
    "foreign_net_5d": "foreign_",
    "inst_net_5d": "institution",
    "indiv_net_5d": "individual",
    "pension_net_5d": "penfnd_etc",
}


def supply_features(sd: pd.DataFrame, *, window: int = 5) -> pd.DataFrame:
    """투자자별 누적 순매수와 거래량 대비 비율을 계산한다."""
    out = sd.sort_values(["code", "date"]).copy()
    grouped = out.groupby("code", sort=False)

    for name, column in _INVESTORS.items():
        out[name] = grouped[column].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )

    volume_sum = grouped["acc_trde_qty"].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )
    out["foreign_net_ratio"] = out["foreign_net_5d"] / volume_sum
    out["inst_net_ratio"] = out["inst_net_5d"] / volume_sum
    out["smart_money_net_5d"] = out["foreign_net_5d"] + out["inst_net_5d"]

    columns = ["code", "date", *_INVESTORS, "foreign_net_ratio", "inst_net_ratio",
               "smart_money_net_5d"]
    return out[columns].reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_features_supply.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/features tests/test_features_supply.py
git commit -m "feat(features): 수급 피처 — 투자자별 순매수·거래량 대비 비율"
```

---

### Task 5: 데이터셋 조립

**Files:**
- Create: `src/scalp_it/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `extract_events`, `supply_features`, `forward_return`, `excess_return`
- Produces:
  - `FEATURE_COLUMNS: list[str]` — 모델이 쓰는 피처 컬럼 이름 목록 (고정 순서)
  - `build_dataset(prices, supply_demand, *, adv_floor=5000.0, tv_multiple=3.0, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, fwd_return, excess_return]`. 라벨이 `NaN` 인 행과 피처가 하나라도 `NaN` 인 행은 제거한다.

`FEATURE_COLUMNS` = `["tv_ratio", "log_adv", "foreign_net_ratio", "inst_net_ratio", "smart_money_net_5d_scaled", "indiv_net_5d_scaled", "pension_net_5d_scaled"]`

`*_scaled` 는 `net_5d / (adv * 1e6 / entry_price)` 즉 **평균 거래대금 대비 순매수 주식 수 비율**로, 종목 규모 차이를 제거한다. `log_adv` 는 `numpy.log10(adv)`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dataset.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.dataset import FEATURE_COLUMNS, build_dataset


def _prices(codes: list[str], n: int, spike_at: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            tv = 6000.0 * (5.0 if i == spike_at else 1.0)
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "open": 100.0 + i,
                    "close": 100.0 + i,
                    "trade_value": tv,
                }
            )
    return pd.DataFrame(rows)


def _supply(codes: list[str], n: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        for i in range(n):
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "individual": -100,
                    "foreign_": 60,
                    "institution": 40,
                    "penfnd_etc": 10,
                    "acc_trde_qty": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_has_expected_columns():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, tv_multiple=3.0, adv_floor=5000.0,
                       hold_start=10, hold_end=20)
    assert list(ds.columns) == ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]


def test_dataset_drops_rows_without_labels():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    assert ds["excess_return"].notna().all()
    assert ds[FEATURE_COLUMNS].notna().all().all()


def test_dataset_only_contains_event_rows():
    prices = _prices(["A"], 80, spike_at=25)
    sd = _supply(["A"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    # 급증일은 하나뿐이므로 이벤트도 최대 하나
    assert len(ds) <= 1
    if len(ds) == 1:
        assert ds.iloc[0]["date"] == pd.Timestamp("2026-01-01") + pd.Timedelta(days=25)


def test_log_adv_is_log10_of_adv():
    prices = _prices(["A", "B"], 80, spike_at=25)
    sd = _supply(["A", "B"], 80)
    ds = build_dataset(prices, sd, hold_start=10, hold_end=20)
    if not ds.empty:
        assert np.isclose(ds.iloc[0]["log_adv"], np.log10(6000.0), atol=1e-6)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.dataset'`

- [ ] **Step 3: 구현**

`src/scalp_it/dataset.py`:

```python
"""이벤트 · 피처 · 라벨을 하나의 학습 테이블로 조립한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import extract_events
from .features.supply import supply_features
from .labels import excess_return, forward_return

FEATURE_COLUMNS: list[str] = [
    "tv_ratio",
    "log_adv",
    "foreign_net_ratio",
    "inst_net_ratio",
    "smart_money_net_5d_scaled",
    "indiv_net_5d_scaled",
    "pension_net_5d_scaled",
]

_SCALED = {
    "smart_money_net_5d_scaled": "smart_money_net_5d",
    "indiv_net_5d_scaled": "indiv_net_5d",
    "pension_net_5d_scaled": "pension_net_5d",
}


def build_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    tv_multiple: float = 3.0,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """이벤트 행에 피처와 라벨을 붙여 학습 가능한 테이블을 만든다."""
    events = extract_events(
        prices, adv_floor=adv_floor, tv_multiple=tv_multiple
    )
    if events.empty:
        return pd.DataFrame(columns=["code", "date", *FEATURE_COLUMNS,
                                     "fwd_return", "excess_return"])

    supply = supply_features(supply_demand)
    labels = excess_return(
        forward_return(
            prices,
            entry_offset=entry_offset,
            hold_start=hold_start,
            hold_end=hold_end,
        )
    )

    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    # 순매수 주식 수를 '평균 거래대금이 감당하는 주식 수'로 나눠 규모 차이를 제거한다.
    # adv 단위는 백만원이므로 1e6 을 곱해 원 단위로 되돌린 뒤 진입가로 나눈다.
    shares_capacity = (df["adv"] * 1e6) / df["entry_price"]
    for scaled, raw in _SCALED.items():
        df[scaled] = df[raw] / shares_capacity

    df["log_adv"] = np.log10(df["adv"])

    columns = ["code", "date", *FEATURE_COLUMNS, "fwd_return", "excess_return"]
    out = df[columns].replace([np.inf, -np.inf], np.nan)
    return out.dropna().sort_values(["date", "code"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): 이벤트·피처·라벨 조립"
```

---

### Task 6: walk-forward 베이스라인 모델

**Files:**
- Create: `src/scalp_it/model/__init__.py`
- Create: `src/scalp_it/model/baseline.py`
- Test: `tests/test_model_baseline.py`

**Interfaces:**
- Consumes: `build_dataset` 결과, `FEATURE_COLUMNS`
- Produces:
  - `walk_forward_predict(ds: pd.DataFrame, *, n_folds: int = 4, min_train: int = 200, seed: int = 0) -> pd.DataFrame` — 컬럼 `[code, date, excess_return, prediction, fold]`. 시간순으로 폴드를 나누고 각 폴드는 **그 이전 데이터만으로** 학습한다.
  - `portfolio_returns(pred: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame` — 컬럼 `[date, n_positions, mean_excess]`. 날짜별 예측 상위 `top_n` 종목의 평균 초과수익.

모델은 `sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=seed)`. 스펙 §6 의 표본 제약에 따라 얕게 고정한다.

`top_n=15` 는 스펙 §2 의 "동시 보유 15종목 이상" 제약을 반영한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_model_baseline.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.dataset import FEATURE_COLUMNS
from scalp_it.model.baseline import portfolio_returns, walk_forward_predict


def _dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    )
    df["code"] = [f"{i % 40:06d}" for i in range(n)]
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    # 첫 피처가 라벨을 결정하는 학습 가능한 구조를 넣는다
    df["excess_return"] = df[FEATURE_COLUMNS[0]] * 0.05 + rng.normal(scale=0.01, size=n)
    return df[["code", "date", *FEATURE_COLUMNS, "excess_return"]]


def test_walk_forward_returns_expected_columns():
    pred = walk_forward_predict(_dataset(600), n_folds=3, min_train=100)
    assert list(pred.columns) == ["code", "date", "excess_return", "prediction", "fold"]


def test_walk_forward_never_predicts_on_training_dates():
    ds = _dataset(600)
    pred = walk_forward_predict(ds, n_folds=3, min_train=100)
    # 각 폴드의 최소 예측일은 이전 폴드의 최대 예측일보다 커야 한다
    bounds = pred.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    for i in range(1, len(bounds)):
        assert bounds.iloc[i]["min"] > bounds.iloc[i - 1]["max"]


def test_walk_forward_learns_a_real_signal():
    pred = walk_forward_predict(_dataset(1200), n_folds=4, min_train=200)
    corr = pred["prediction"].corr(pred["excess_return"])
    assert corr > 0.2  # 구조를 심어둔 데이터이므로 양의 상관이 나와야 한다


def test_portfolio_returns_picks_top_n_per_date():
    pred = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "excess_return": [0.1, 0.2, 0.3, 0.4],
            "prediction": [0.4, 0.3, 0.2, 0.1],
            "fold": [0] * 4,
        }
    )
    out = portfolio_returns(pred, top_n=2)
    assert len(out) == 1
    assert out.iloc[0]["n_positions"] == 2
    # 예측 상위 2개는 A(0.1) 와 B(0.2) → 평균 0.15
    assert np.isclose(out.iloc[0]["mean_excess"], 0.15)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_model_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.model'`

- [ ] **Step 3: 구현**

`src/scalp_it/model/__init__.py`:

```python
"""학습·예측 계층.

스펙 §6: 실효 독립 표본이 작으므로 모델은 얕게 고정한다.
딥러닝·대규모 앙상블은 이 표본 크기에서 과적합 기계다.
"""
```

`src/scalp_it/model/baseline.py`:

```python
"""walk-forward 베이스라인 — 수급 피처만으로 초과수익을 예측한다 (스펙 §3 ①층)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ..dataset import FEATURE_COLUMNS


def walk_forward_predict(
    ds: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """시간순 폴드로 나눠 각 폴드를 그 이전 데이터만으로 학습·예측한다.

    폴드 경계는 날짜 단위로 끊는다. 같은 날짜가 학습과 예측에 동시에 들어가면
    같은 날 다른 종목의 결과가 새어 들어온다.
    """
    df = ds.sort_values(["date", "code"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    if len(df) <= min_train:
        raise ValueError(f"표본이 {len(df)} 행뿐이라 min_train={min_train} 을 못 채운다")

    # min_train 행을 채우는 최초 날짜 이후를 n_folds 로 등분한다
    train_end_idx = int(np.searchsorted(df.index.to_numpy(), min_train))
    first_test_date = df.loc[train_end_idx, "date"]
    test_dates = dates[dates > first_test_date]
    if len(test_dates) < n_folds:
        raise ValueError(f"검증 가능한 날짜가 {len(test_dates)} 개뿐이라 {n_folds} 폴드를 못 만든다")

    chunks = np.array_split(test_dates, n_folds)
    results = []
    for fold, chunk in enumerate(chunks):
        fold_start = chunk[0]
        train = df[df["date"] < fold_start]
        test = df[df["date"].isin(chunk)]
        if train.empty or test.empty:
            continue
        model = GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=seed)
        model.fit(train[FEATURE_COLUMNS], train["excess_return"])
        block = test[["code", "date", "excess_return"]].copy()
        block["prediction"] = model.predict(test[FEATURE_COLUMNS])
        block["fold"] = fold
        results.append(block)

    return pd.concat(results, ignore_index=True)


def portfolio_returns(pred: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    """날짜별 예측 상위 ``top_n`` 종목의 평균 초과수익.

    ``top_n`` 하한이 15 인 이유는 스펙 §2 — 4~8 종목 집중은 이미 죽은 영역이다.
    """
    rows = []
    for date, group in pred.groupby("date"):
        picked = group.nlargest(top_n, "prediction")
        rows.append(
            {
                "date": date,
                "n_positions": len(picked),
                "mean_excess": picked["excess_return"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_model_baseline.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/model tests/test_model_baseline.py
git commit -m "feat(model): walk-forward 베이스라인 (수급 단독)"
```

---

### Task 7: 음성대조 게이트와 CLI

**Files:**
- Create: `src/scalp_it/validate/__init__.py`
- Create: `src/scalp_it/validate/gate.py`
- Create: `src/scalp_it/cli.py`
- Test: `tests/test_validate_gate.py`

**Interfaces:**
- Consumes: `walk_forward_predict` 결과, `portfolio_returns`
- Produces:
  - `random_null(pred: pd.DataFrame, *, top_n: int = 15, n_trials: int = 200, seed: int = 0) -> np.ndarray` — 각 시행마다 날짜별로 `top_n` 종목을 **무작위** 선택했을 때의 전체 평균 초과수익. 길이 `n_trials`.
  - `gate_report(pred: pd.DataFrame, *, top_n: int = 15, n_trials: int = 200, cost_one_way: float = 0.0023, seed: int = 0) -> dict` — 키 `strategy_mean`, `null_mean`, `null_p95`, `percentile`, `net_mean`, `verdict`. `verdict` 는 `"PASS"` (널 95 백분위 초과 **그리고** 비용 차감 후 양수) 또는 `"FAIL"`.
  - `main(argv: list[str] | None = None) -> int` — CLI 진입점

비용은 왕복이므로 `cost_one_way * 2` 를 차감한다. 기본 0.0023 은 kr-quant PEAD 검증에서 쓴 편도 비용이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validate_gate.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.validate.gate import gate_report, random_null


def _pred(n_dates: int, n_per_date: int, signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n_per_date):
            excess = rng.normal(scale=0.05)
            prediction = excess if signal else rng.normal()
            rows.append(
                {
                    "code": f"{i:06d}",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                    "excess_return": excess,
                    "prediction": prediction,
                    "fold": 0,
                }
            )
    return pd.DataFrame(rows)


def test_random_null_has_requested_length():
    null = random_null(_pred(20, 30, signal=False), top_n=15, n_trials=50)
    assert null.shape == (50,)


def test_random_null_centres_near_zero():
    # 초과수익이 날짜별로 디민돼 있지 않아도, 무작위 선택의 기대값은 전체 평균 근처다
    pred = _pred(30, 40, signal=False)
    null = random_null(pred, top_n=15, n_trials=200)
    assert abs(null.mean() - pred["excess_return"].mean()) < 0.01


def test_gate_fails_when_prediction_is_noise():
    report = gate_report(_pred(30, 40, signal=False), top_n=15, n_trials=200)
    assert report["verdict"] == "FAIL"


def test_gate_passes_when_prediction_is_perfect_and_beats_cost():
    # 예측 = 실제 초과수익이므로 상위 15개는 확실히 널을 이긴다
    report = gate_report(_pred(30, 40, signal=True), top_n=15, n_trials=200, cost_one_way=0.0)
    assert report["verdict"] == "PASS"
    assert report["percentile"] > 95


def test_gate_report_has_all_keys():
    report = gate_report(_pred(20, 30, signal=False), top_n=15, n_trials=50)
    for key in ["strategy_mean", "null_mean", "null_p95", "percentile", "net_mean", "verdict"]:
        assert key in report
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_validate_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.validate'`

- [ ] **Step 3: 게이트 구현**

`src/scalp_it/validate/__init__.py`:

```python
"""판정 계층.

스펙 §8: scalp-it 은 자기 검증을 하지 않는다. 이 모듈은 kr-quant 가 확립한
판정 기준(음성대조·비용스윕)을 같은 형태로 적용할 뿐이며, 여기서 통과한 것만
운영에 등록한다.

프랍 스윙 탐색(2026-07-18)에서 진짜 판별기는 폴드 수가 아니라 랜덤 널 대비였다.
"""
```

`src/scalp_it/validate/gate.py`:

```python
"""음성대조 게이트 — 랜덤 진입 널을 이기지 못하면 아무것도 승격하지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.baseline import portfolio_returns


def random_null(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """날짜마다 ``top_n`` 종목을 무작위로 골랐을 때의 평균 초과수익 분포."""
    rng = np.random.default_rng(seed)
    by_date = [group["excess_return"].to_numpy() for _, group in pred.groupby("date")]
    out = np.empty(n_trials)
    for trial in range(n_trials):
        daily = []
        for values in by_date:
            k = min(top_n, len(values))
            picked = rng.choice(values, size=k, replace=False)
            daily.append(picked.mean())
        out[trial] = float(np.mean(daily))
    return out


def gate_report(
    pred: pd.DataFrame,
    *,
    top_n: int = 15,
    n_trials: int = 200,
    cost_one_way: float = 0.0023,
    seed: int = 0,
) -> dict:
    """전략 성과를 랜덤 널 및 비용과 대조해 판정한다."""
    strategy = portfolio_returns(pred, top_n=top_n)["mean_excess"].mean()
    null = random_null(pred, top_n=top_n, n_trials=n_trials, seed=seed)
    percentile = float((null < strategy).mean() * 100.0)
    net = float(strategy - cost_one_way * 2)
    passed = percentile > 95.0 and net > 0.0
    return {
        "strategy_mean": float(strategy),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "percentile": percentile,
        "net_mean": net,
        "verdict": "PASS" if passed else "FAIL",
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_validate_gate.py -v`
Expected: 5 passed

- [ ] **Step 5: CLI 구현**

`src/scalp_it/cli.py`:

```python
"""`scalp-baseline` — 베이스라인 파이프라인 전체를 실 DB 로 한 번 돌린다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.db import connect, load_prices, load_supply_demand
from .dataset import build_dataset
from .model.baseline import portfolio_returns, walk_forward_predict
from .validate.gate import gate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="수급 단독 베이스라인 판정")
    parser.add_argument("--db", default=None, help="DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--tv-multiple", type=float, default=3.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args(argv)

    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    print(f"가격 {len(prices):,} 행 · 수급 {len(supply):,} 행", file=sys.stderr)

    ds = build_dataset(
        prices,
        supply,
        adv_floor=args.adv_floor,
        tv_multiple=args.tv_multiple,
        hold_end=args.hold_end,
    )
    print(f"이벤트 {len(ds):,} 건 · 종목 {ds['code'].nunique():,}", file=sys.stderr)
    if ds.empty:
        print("이벤트가 없다. 임계값을 확인하라.", file=sys.stderr)
        return 1

    pred = walk_forward_predict(ds, n_folds=args.folds)
    report = gate_report(pred, top_n=args.top_n)
    report["n_events"] = int(len(ds))
    report["n_codes"] = int(ds["code"].nunique())
    report["date_range"] = [str(ds["date"].min().date()), str(ds["date"].max().date())]
    report["per_fold"] = (
        pred.groupby("fold")
        .apply(lambda g: portfolio_returns(g, top_n=args.top_n)["mean_excess"].mean(),
               include_groups=False)
        .round(6)
        .to_dict()
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 전체 테스트 통과 확인**

Run: `uv run pytest -v`
Expected: 27 passed

- [ ] **Step 7: 실 DB 로 파이프라인 실행**

Run:
```bash
export $(grep -v '^#' ../kr-quant/.env | xargs)
uv run scalp-baseline --out results/baseline.json
```
Expected: `results/baseline.json` 생성, `verdict` 가 `PASS` 또는 `FAIL` 로 출력됨.

**FAIL 은 버그가 아니다.** 수급만으로 20~40일 초과수익이 나오면 그건 이미 알려진 큰 알파일 것이다. 베이스라인의 목적은 다음 층(뉴스·LLM)이 넘어야 할 선을 확정하는 것이다 (스펙 §3).

- [ ] **Step 8: 커밋**

```bash
git add src/scalp_it/validate src/scalp_it/cli.py tests/test_validate_gate.py
git commit -m "feat(validate): 음성대조 게이트 + scalp-baseline CLI"
```

---

## Self-Review 체크

**스펙 커버리지 (계획 1 범위):**

| 스펙 요구 | 태스크 |
|---|---|
| §3 ①층 수급 단독 베이스라인 | Task 4·5·6 |
| §3 D+20~40 초과수익 라벨 | Task 3 |
| §2 드리프트 자르지 않음 (손절 없음) | Task 3 — 중간 청산 로직 자체가 없음 |
| §2 집중 금지 (15종목 이상) | Task 6 `top_n=15` |
| §2 음성대조 필수 | Task 7 |
| §5 point-in-time 유동성 하한 | Task 2 `adv_floor`, shift(1) |
| §6 단순 모델 강제 | Task 6 `max_depth=2` |
| §8 어댑터 경계 | Task 1 — SQL 은 `adapters/db.py` 에만 |
| §11 수정주가 | Task 1 — `daily_bars_adjusted` 만 조회 |
| §11 생존편향 | `daily_bars_adjusted` 에 폐지 종목 포함 (kr-quant 수집 정책) |

**계획 1 범위 밖 (후속 계획):** 뉴스 인제스트, LLM 피처, 층별 비교, 근거 카드, 전진 기록.

## 후속 계획

- 계획 2: 뉴스 인제스트 (빅카인즈 적재 · 종목 매핑 · '핫함')
- 계획 3: LLM 피처 추출 (provider 추상화 · 배치)
- 계획 4: 층별 모델 비교 + 근거 카드 리포트
