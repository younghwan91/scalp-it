# 계획 2: 뉴스 인제스트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 빅카인즈에서 수동 다운로드한 엑셀을 point-in-time 뉴스 테이블로 정규화·적재하고, 종목 매핑·중복 제거·'핫함' 계산을 거쳐 **일별 학습셋(상위 50) / 운영셋(상위 10)** 을 로컬 sqlite 에 남긴다.

**Architecture:** 계획 1 과 같은 분리를 유지한다 — 파싱·매핑·중복제거·핫함은 전부 순수 함수(DataFrame in → DataFrame out)이고, I/O 는 `adapters/db.py`(TimescaleDB 읽기)와 `news/store.py`(로컬 sqlite 쓰기) 두 곳에만 있다. 빅카인즈 엑셀의 실제 컬럼 구성은 확정적으로 모르므로 Task 1 을 사람 개입 단계로 두고, 거기서 확정한 이름만 이후 태스크가 참조한다.

**Tech Stack:** Python ≥3.10, uv, pandas ≥2.0, openpyxl, sqlite3(표준 라이브러리), pytest, ruff

**Spec:** `docs/superpowers/specs/2026-08-17-swing-it-design.md`

**선행 계획:** `docs/superpowers/plans/2026-08-17-baseline-skeleton.md` (계획 1). 이 계획은 계획 1 Task 1 의 `src/swing_it/adapters/db.py` 와 Task 2 의 `src/swing_it/events.add_liquidity_stats` 를 소비한다. 계획 1 이 끝나 있어야 실행 가능하다.

## Global Constraints

- **본문은 200자만 저장한다.** 빅카인즈 제약이자 학습/운영 입력 통일 결정이다. 운영(krx-news-rest-api)은 전문을 주지만 모델 입력은 200자로 절단한다. 스펙 §4.
- **데이터 커밋 금지.** 엑셀 원본은 `data/bigkinds/`, 산출 DB 는 `data/news.db`. `.gitignore` 가 `data/`·`*.xlsx`·`*.db` 를 이미 차단한다. 스펙 §8.
- **DB 쓰기 금지.** TimescaleDB(`kr_quant`)는 읽기 전용이다. swing-it 의 모든 쓰기는 로컬 sqlite `data/news.db` 로만 간다.
- **외부 접근은 어댑터에만.** TimescaleDB SQL 은 `src/swing_it/adapters/db.py` 에만 존재한다. `news/` 모듈은 DataFrame 만 받는다. 스펙 §8.
- **'핫함'은 과거·현재 동일 계산 가능한 것만.** ①기사 건수 급증 배수 ②보도 매체 수 ③거래대금 급증 배수. 조회수·댓글수처럼 과거 데이터가 없는 지표는 쓰지 않는다. 스펙 §4.
- **point-in-time 엄수.** 기사 건수 baseline 은 반드시 `.shift(1)` 이후 계산한다. 종목 매핑은 기사 시점에 상장돼 있던 종목만 인정한다(폐지 종목 포함 — 생존편향 대응, 스펙 §11).
- **유동성 하한.** 랭킹 대상은 `adv >= 5000` (단위 백만원 = 50억원). 계획 1 과 같은 값. 스펙 §5.
- **학습 30~50건/일, 운영 10건/일.** 기본값 `train_top=50`, `ops_top=10`. 스펙 §5.
- **가중치는 사전등록.** 핫함 가중치 `(0.4, 0.3, 0.3)` 은 코드 상수로 고정하고, 튜닝하려면 계획 4 의 사전등록 절차를 거친다. 스펙 §6.

## 종목 마스터 실측 (2026-08-17 확인)

| 테이블 | 컬럼 | 행 수 |
|---|---|---|
| `stocks` | `code, name, market, sector, kind` | 2,641 (현재 상장) |
| `delisted_stocks` | `code, name, market, last_trade_date, naver_checked` | 4,162 |

`market` 값은 `거래소` / `코스닥`.

> **구현 중 정정 (2026-08-17).** `last_trade_date` 로 상장 여부를 판정하면 안 된다.
> 실 DB 스모크 결과 `delisted_stocks` 4,162 건 중 **3,693 건(89%)이 이 값이 NULL** 이다.
> 날짜 유무로 판정하면 폐지 종목 대부분을 상장으로 오인해, 생존편향을 막으려고
> 폐지 종목을 넣어놓고 정작 그 판정이 틀리게 된다. `load_stock_master` 는
> 출처 테이블 기준 `is_delisted: bool` 을 함께 돌려주며, **Task 4 의 매핑은 이 플래그를 쓴다.**
> 같은 코드가 양쪽에 있으면(17 건) 상장 쪽을 남긴다.
> 확인된 규모: 마스터 6,786 = 상장 2,641 + 폐지 4,145.

`last_trade_date` 는 `text` 타입이며 값이 있는 폐지 종목(452 건)에서만 참고값으로 쓴다. 현재 상장 종목에는 상장일 컬럼이 **없으므로**, 상장 이전 구간의 오매핑은 이름 규칙이 아니라 가격 데이터 조인으로 걸러낸다(그날 `daily_bars_adjusted` 행이 없으면 유동성 하한에서 탈락한다).

## 빅카인즈 다운로드 절차 (사람이 하는 부분)

1. `bigkinds.or.kr` 로그인 → 뉴스분석 → 기간·언론사 지정 → 엑셀 다운로드.
2. **1회 20,000건 제한**이 있으므로 분기 단위로 쪼개 받는다(3년치면 12개 파일). 파일이 20,000행에 딱 붙으면 잘렸다는 뜻이니 월 단위로 더 쪼갠다.
3. 받은 파일을 `data/bigkinds/` 에 그대로 둔다. 파일명은 자유다 — Task 2 의 로더가 디렉터리를 통째로 읽고 `news_id` 로 중복을 제거한다.

## File Structure

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | openpyxl 의존성 · `swing-news` 엔트리포인트 추가 (수정) |
| `scripts/inspect_bigkinds.py` | 엑셀 실제 헤더를 찍어보는 일회성 조사 스크립트 |
| `src/swing_it/news/schema.py` | 빅카인즈 컬럼명 ↔ 내부 정규 컬럼명 매핑. **Task 1 의 사람 확인 산출물** |
| `src/swing_it/news/bigkinds.py` | 엑셀 → 정규화 기사 테이블 (순수 함수 + 파일 읽기) |
| `src/swing_it/adapters/db.py` | `load_stock_master` 추가 (수정) |
| `src/swing_it/news/mapping.py` | 기사 → 종목 매핑 (순수 함수) |
| `src/swing_it/news/dedupe.py` | 같은 사건 중복 기사 클러스터링 (순수 함수) |
| `src/swing_it/news/hotness.py` | 핫함 3축 계산 + 일별 랭킹 (순수 함수) |
| `src/swing_it/news/store.py` | 로컬 sqlite 적재·조회 |
| `src/swing_it/cli_news.py` | `swing-news` 엔트리포인트 |
| `tests/` | 위 각 모듈 대응 테스트 |

---

### Task 1: 빅카인즈 엑셀 컬럼 구조 확정

이 태스크만 사람 개입이 필요하다. 이후 모든 태스크는 여기서 확정된 이름을 쓴다.

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/inspect_bigkinds.py`
- Create: `src/swing_it/news/__init__.py`
- Create: `src/swing_it/news/schema.py`
- Test: `tests/test_news_schema.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `BODY_MAX_CHARS: int = 200`
  - `CANONICAL_COLUMNS: list[str] = ["news_id", "date", "outlet", "title", "body", "keywords", "orgs", "url"]`
  - `COLUMN_MAP: dict[str, str]` — 빅카인즈 원본 헤더 → 정규 컬럼명
  - `REQUIRED_SOURCE_COLUMNS: tuple[str, ...]` — 없으면 파싱을 거부할 원본 헤더
  - `inspect_columns(path: str | Path) -> list[str]` — 엑셀 첫 시트의 헤더 목록

- [ ] **Step 1: `pyproject.toml` 에 의존성과 엔트리포인트 추가**

`[project] dependencies` 에 한 줄, `[project.scripts]` 에 한 줄을 더한다.

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "openpyxl>=3.1",
]
```

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-news = "swing_it.cli_news:main"
```

Run: `uv sync`
Expected: openpyxl 설치 완료.

- [ ] **Step 2: 조사 스크립트 작성**

`scripts/inspect_bigkinds.py`:

```python
"""빅카인즈 엑셀의 실제 헤더를 확인하는 일회성 조사 스크립트.

빅카인즈는 UI 개편에 따라 컬럼 구성이 바뀐다. 추측하지 말고 실물을 본다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("사용법: python scripts/inspect_bigkinds.py <엑셀경로>", file=sys.stderr)
        return 2
    path = Path(args[0])
    df = pd.read_excel(path, dtype=str, engine="openpyxl", nrows=3)
    print(f"파일: {path}")
    print(f"컬럼 {len(df.columns)}개:")
    for i, col in enumerate(df.columns):
        sample = "" if df.empty else str(df.iloc[0][col])[:60].replace("\n", " ")
        print(f"  [{i:2d}] {col!r} — 예: {sample!r}")
    full = pd.read_excel(path, dtype=str, engine="openpyxl")
    print(f"행 수: {len(full):,}")
    if len(full) >= 20000:
        print("경고: 20,000행에 도달했다. 다운로드가 잘렸을 수 있으니 기간을 더 쪼개라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 사람 — 파일 1개를 받아 헤더를 확인한다**

`data/bigkinds/` 에 엑셀 파일 1개를 놓고 실행한다.

Run:
```bash
mkdir -p data/bigkinds
uv run python scripts/inspect_bigkinds.py data/bigkinds/*.xlsx
```
Expected: 컬럼 목록과 샘플 값이 출력된다.

**출력된 헤더를 그대로 Step 4 의 `COLUMN_MAP` 왼쪽에 옮긴다.** 아래 기본값은 흔히 관찰되는 구성이며 **Task 1 에서 검증**해야 한다. 실물과 다르면 왼쪽 문자열만 고친다 — 오른쪽(정규 컬럼명)은 이후 태스크가 전부 참조하므로 **절대 바꾸지 않는다.**

- [ ] **Step 4: 실패하는 테스트 작성**

`tests/test_news_schema.py`:

```python
from openpyxl import Workbook

from swing_it.news.schema import (
    BODY_MAX_CHARS,
    CANONICAL_COLUMNS,
    COLUMN_MAP,
    REQUIRED_SOURCE_COLUMNS,
    inspect_columns,
)


def test_body_limit_is_two_hundred():
    # 빅카인즈 제약이자 학습/운영 입력 통일 결정 (스펙 §4)
    assert BODY_MAX_CHARS == 200


def test_column_map_covers_every_required_source_column():
    for source in REQUIRED_SOURCE_COLUMNS:
        assert source in COLUMN_MAP


def test_required_columns_map_onto_canonical_names():
    required_targets = {COLUMN_MAP[src] for src in REQUIRED_SOURCE_COLUMNS}
    assert {"news_id", "date", "outlet", "title", "body"} <= required_targets


def test_every_mapped_target_is_canonical():
    for target in COLUMN_MAP.values():
        assert target in CANONICAL_COLUMNS


def test_canonical_columns_have_no_duplicates():
    assert len(CANONICAL_COLUMNS) == len(set(CANONICAL_COLUMNS))


def test_inspect_columns_reads_header_row(tmp_path):
    path = tmp_path / "probe.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["뉴스 식별자", "일자", "제목"])
    ws.append(["01100101.20260817", "20260817", "테스트"])
    wb.save(path)
    assert inspect_columns(path) == ["뉴스 식별자", "일자", "제목"]
```

- [ ] **Step 5: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news'`

- [ ] **Step 6: 구현**

`src/swing_it/news/__init__.py`:

```python
"""뉴스 인제스트 — 기사 수집·종목 매핑·중복 제거·point-in-time 저장 (스펙 §8).

이 패키지는 DB 를 모른다. 외부 데이터는 전부 DataFrame 으로 주입받는다.
유일한 예외는 ``store`` 로, 로컬 sqlite 산출물만 다룬다.
"""
```

`src/swing_it/news/schema.py`:

```python
"""빅카인즈 엑셀의 컬럼 계약.

왼쪽(빅카인즈 원본 헤더)은 실물 파일에서 확인한 값이다 (계획 2 Task 1).
빅카인즈는 UI 개편에 따라 헤더가 바뀌므로, 파싱이 깨지면 먼저
``scripts/inspect_bigkinds.py`` 로 실물을 다시 확인하고 이 표만 고친다.
오른쪽(정규 컬럼명)은 하위 모듈 전체가 참조하므로 바꾸지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: 본문 저장 길이. 빅카인즈가 200자만 제공하고, 운영(전문)도 여기에 맞춰
#: 절단한다. 학습·운영 입력 길이가 다르면 모델이 깨진다 (스펙 §4).
BODY_MAX_CHARS = 200

CANONICAL_COLUMNS: list[str] = [
    "news_id",
    "date",
    "outlet",
    "title",
    "body",
    "keywords",
    "orgs",
    "url",
]

#: 빅카인즈 원본 헤더 → 정규 컬럼명.
COLUMN_MAP: dict[str, str] = {
    "뉴스 식별자": "news_id",
    "일자": "date",
    "언론사": "outlet",
    "제목": "title",
    "본문": "body",
    "키워드": "keywords",
    "기관": "orgs",
    "URL": "url",
}

#: 이 중 하나라도 없으면 파일을 거부한다. 나머지는 없으면 빈 문자열로 채운다.
REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "뉴스 식별자",
    "일자",
    "언론사",
    "제목",
    "본문",
)


def inspect_columns(path: str | Path) -> list[str]:
    """엑셀 첫 시트의 헤더 목록을 돌려준다."""
    df = pd.read_excel(path, dtype=str, engine="openpyxl", nrows=0)
    return [str(c) for c in df.columns]
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_schema.py -v`
Expected: 6 passed

- [ ] **Step 8: 실물 파일로 계약 확인**

Run:
```bash
uv run python -c "
from swing_it.news.schema import COLUMN_MAP, REQUIRED_SOURCE_COLUMNS, inspect_columns
import glob
cols = inspect_columns(sorted(glob.glob('data/bigkinds/*.xlsx'))[0])
missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in cols]
print('필수 누락:', missing)
print('매핑 안 된 원본 컬럼:', [c for c in cols if c not in COLUMN_MAP])
"
```
Expected: `필수 누락: []`. 비어 있지 않으면 Step 6 의 `COLUMN_MAP` 왼쪽을 실물에 맞춰 고치고 Step 7 부터 다시 한다. 매핑 안 된 컬럼이 남는 것은 정상이다(쓰지 않는 컬럼).

- [ ] **Step 9: 커밋**

```bash
git add pyproject.toml scripts/inspect_bigkinds.py src/swing_it/news tests/test_news_schema.py
git commit -m "feat(news): 빅카인즈 컬럼 계약 확정 + 헤더 조사 스크립트"
```

---

### Task 2: 엑셀 → 정규화 기사 테이블

**Files:**
- Create: `src/swing_it/news/bigkinds.py`
- Test: `tests/test_news_bigkinds.py`

**Interfaces:**
- Consumes: `schema.COLUMN_MAP`, `schema.CANONICAL_COLUMNS`, `schema.REQUIRED_SOURCE_COLUMNS`, `schema.BODY_MAX_CHARS`
- Produces:
  - `parse_dates(series: pd.Series) -> pd.Series` — `20260817` / `2026-08-17` / `2026.08.17 12:34` 를 모두 `datetime64[ns]` 로. 실패는 `NaT`
  - `normalize_articles(raw: pd.DataFrame, *, body_max_chars: int = BODY_MAX_CHARS) -> pd.DataFrame` — 컬럼 `CANONICAL_COLUMNS`. 필수 원본 컬럼 누락 시 `ValueError`
  - `load_bigkinds_excel(path: str | Path) -> pd.DataFrame` — 파일 1개
  - `load_bigkinds_dir(directory: str | Path) -> pd.DataFrame` — `*.xlsx` 전부 읽어 `news_id` 로 중복 제거

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_bigkinds.py`:

```python
import pandas as pd
import pytest
from openpyxl import Workbook

from swing_it.news.bigkinds import (
    load_bigkinds_dir,
    load_bigkinds_excel,
    normalize_articles,
    parse_dates,
)
from swing_it.news.schema import CANONICAL_COLUMNS

_HEADER = ["뉴스 식별자", "일자", "언론사", "제목", "본문", "키워드", "기관", "URL"]


def _write_xlsx(path, rows: list[list[str]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(_HEADER)
    for row in rows:
        ws.append(row)
    wb.save(path)


def _raw(rows: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_HEADER)


def test_parse_dates_handles_three_formats():
    out = parse_dates(pd.Series(["20260817", "2026-08-17", "2026.08.17 12:34"]))
    assert (out == pd.Timestamp("2026-08-17")).all()


def test_parse_dates_returns_nat_for_garbage():
    out = parse_dates(pd.Series(["없음"]))
    assert out.isna().all()


def test_normalize_maps_columns_to_canonical_names():
    raw = _raw([["01100101.20260817", "20260817", "한국경제", "제목", "본문",
                 "키워드", "삼성전자", "http://x"]])
    out = normalize_articles(raw)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.iloc[0]["outlet"] == "한국경제"
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_body_is_truncated_to_two_hundred_chars():
    raw = _raw([["a", "20260817", "매체", "제목", "가" * 500, "", "", ""]])
    out = normalize_articles(raw)
    assert len(out.iloc[0]["body"]) == 200


def test_missing_required_column_raises():
    raw = _raw([["a", "20260817", "매체", "제목", "본문", "", "", ""]]).drop(columns=["본문"])
    with pytest.raises(ValueError, match="본문"):
        normalize_articles(raw)


def test_optional_column_absent_becomes_empty_string():
    raw = _raw([["a", "20260817", "매체", "제목", "본문", "", "", ""]]).drop(columns=["기관"])
    out = normalize_articles(raw)
    assert out.iloc[0]["orgs"] == ""


def test_rows_with_unparseable_date_are_dropped():
    raw = _raw([
        ["a", "20260817", "매체", "제목", "본문", "", "", ""],
        ["b", "없음", "매체", "제목", "본문", "", "", ""],
    ])
    out = normalize_articles(raw)
    assert out["news_id"].tolist() == ["a"]


def test_duplicate_news_id_kept_once():
    raw = _raw([
        ["a", "20260817", "매체", "제목1", "본문", "", "", ""],
        ["a", "20260817", "매체", "제목2", "본문", "", "", ""],
    ])
    out = normalize_articles(raw)
    assert len(out) == 1


def test_load_excel_reads_file(tmp_path):
    path = tmp_path / "one.xlsx"
    _write_xlsx(path, [["a", "20260817", "매체", "제목", "본문", "", "", ""]])
    out = load_bigkinds_excel(path)
    assert out.iloc[0]["news_id"] == "a"


def test_load_dir_merges_files_and_dedupes_across_them(tmp_path):
    _write_xlsx(tmp_path / "q1.xlsx", [["a", "20260817", "매체", "제목", "본문", "", "", ""]])
    _write_xlsx(tmp_path / "q2.xlsx", [
        ["a", "20260817", "매체", "제목", "본문", "", "", ""],
        ["b", "20260818", "매체", "제목", "본문", "", "", ""],
    ])
    out = load_bigkinds_dir(tmp_path)
    assert out["news_id"].tolist() == ["a", "b"]


def test_load_dir_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bigkinds_dir(tmp_path)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_bigkinds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news.bigkinds'`

- [ ] **Step 3: 구현**

`src/swing_it/news/bigkinds.py`:

```python
"""빅카인즈 엑셀 → 정규화 기사 테이블.

수동 다운로드를 전제한 반자동 경로다. 자동 크롤러는 짜지 않는다 —
ToS·저작권 리스크가 공개 전환 계획과 충돌하고, 실시간 수집은
krx-news-rest-api 가 이미 한다 (스펙 §4).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import BODY_MAX_CHARS, CANONICAL_COLUMNS, COLUMN_MAP, REQUIRED_SOURCE_COLUMNS

_TEXT_COLUMNS = ("news_id", "outlet", "title", "body", "keywords", "orgs", "url")


def parse_dates(series: pd.Series) -> pd.Series:
    """빅카인즈 일자 컬럼을 ``datetime64[ns]`` 로 바꾼다.

    ``20260817`` · ``2026-08-17`` · ``2026.08.17 12:34`` 를 모두 받는다.
    구분자를 지우고 앞 8자리만 취하면 세 형태가 같은 문자열이 된다.
    """
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[.\-/]", "", regex=True)
        .str.slice(0, 8)
    )
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def normalize_articles(
    raw: pd.DataFrame,
    *,
    body_max_chars: int = BODY_MAX_CHARS,
) -> pd.DataFrame:
    """원본 엑셀 프레임을 정규 컬럼 구조로 바꾼다."""
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"빅카인즈 필수 컬럼이 없다: {missing}. schema.COLUMN_MAP 을 확인하라.")

    out = pd.DataFrame(index=raw.index)
    for source, target in COLUMN_MAP.items():
        if source in raw.columns:
            out[target] = raw[source]
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = ""

    out["date"] = parse_dates(out["date"])
    for column in _TEXT_COLUMNS:
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["body"] = out["body"].str.slice(0, body_max_chars)

    out = out[out["news_id"] != ""]
    out = out[out["date"].notna()]
    out = out.drop_duplicates(subset="news_id", keep="first")
    return out[CANONICAL_COLUMNS].sort_values(["date", "news_id"]).reset_index(drop=True)


def load_bigkinds_excel(path: str | Path) -> pd.DataFrame:
    """엑셀 파일 하나를 읽어 정규화한다."""
    raw = pd.read_excel(path, dtype=str, engine="openpyxl")
    return normalize_articles(raw)


def load_bigkinds_dir(directory: str | Path) -> pd.DataFrame:
    """디렉터리의 ``*.xlsx`` 를 전부 읽어 ``news_id`` 로 중복을 제거한다.

    20,000건 제한 때문에 기간을 쪼개 받으면 경계에서 같은 기사가 겹친다.
    """
    paths = sorted(Path(directory).glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"{directory} 에 xlsx 파일이 없다")
    frames = [load_bigkinds_excel(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="news_id", keep="first")
    return merged.sort_values(["date", "news_id"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_bigkinds.py -v`
Expected: 11 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/news/bigkinds.py tests/test_news_bigkinds.py
git commit -m "feat(news): 빅카인즈 엑셀 파서 — 본문 200자 절단·news_id 중복 제거"
```

---

### Task 3: 종목 마스터 로더

**Files:**
- Modify: `src/swing_it/adapters/db.py` (함수 추가만. 기존 함수는 건드리지 않는다)
- Test: `tests/test_db_stock_master.py`

**Interfaces:**
- Consumes: `adapters.db.connect(dsn: str | None = None)` (계획 1 Task 1)
- Produces:
  - `load_stock_master(conn) -> pd.DataFrame` — 컬럼 `[code, name, market, sector, last_trade_date, is_delisted]`. `stocks` + `delisted_stocks` 를 합친다. 폐지 종목의 `sector` 는 `None`, `last_trade_date` 는 `datetime64[ns]`. **상장 여부는 `is_delisted` 로만 판정한다** — 위 정정 참고. *(이 태스크는 구현 완료됨)*

생존편향 대응이다(스펙 §11). 폐지 종목을 빼면 과거 기사의 상당수가 매핑 실패로 사라지고, 남은 표본이 "살아남은 회사" 쪽으로 기운다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_db_stock_master.py`:

```python
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
    assert list(df.columns) == [
        "code", "name", "market", "sector", "last_trade_date", "is_delisted",
    ]


def test_master_unions_listed_and_delisted(master_db):
    df = load_stock_master(connect(master_db))
    assert set(df["code"]) == {"005930", "000660", "037550"}


def test_is_delisted_comes_from_source_table_not_from_date(master_db):
    # delisted_stocks 4,162 건 중 89% 는 last_trade_date 가 NULL 이다.
    # 날짜 유무로 판정하면 폐지 종목 대부분을 상장으로 오인한다.
    df = load_stock_master(connect(master_db)).set_index("code")
    assert bool(df.loc["005930", "is_delisted"]) is False
    assert bool(df.loc["037550", "is_delisted"]) is True


def test_delisted_without_last_trade_date_is_still_flagged(master_db):
    conn = connect(master_db)
    conn.execute("INSERT INTO delisted_stocks VALUES ('900110','이스트아시아','코스닥',NULL,NULL)")
    conn.commit()
    df = load_stock_master(conn).set_index("code")
    assert bool(df.loc["900110", "is_delisted"]) is True
    assert pd.isna(df.loc["900110", "last_trade_date"])


def test_listed_row_wins_when_code_is_in_both_tables(master_db):
    conn = connect(master_db)
    conn.execute("INSERT INTO delisted_stocks VALUES ('005930','삼성전자','거래소','2020-01-01',NULL)")
    conn.commit()
    df = load_stock_master(conn).set_index("code")
    assert bool(df.loc["005930", "is_delisted"]) is False


def test_listed_rows_have_no_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert pd.isna(df.loc["005930", "last_trade_date"])


def test_delisted_row_keeps_last_trade_date(master_db):
    df = load_stock_master(connect(master_db)).set_index("code")
    assert df.loc["037550", "last_trade_date"] == pd.Timestamp("2023-06-30")


def test_last_trade_date_is_datetime_dtype(master_db):
    df = load_stock_master(connect(master_db))
    assert pd.api.types.is_datetime64_any_dtype(df["last_trade_date"])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_db_stock_master.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_stock_master'`

- [ ] **Step 3: 구현**

`src/swing_it/adapters/db.py` 끝에 상수 하나와 함수 하나를 추가한다. 기존 `_query` 는 `date` 컬럼을 요구하므로 여기서는 쓰지 않는다.

```python
_MASTER_COLUMNS = ["code", "name", "market", "sector", "last_trade_date", "is_delisted"]


def load_stock_master(conn: Any) -> pd.DataFrame:
    """상장·폐지 종목을 합친 종목 마스터.

    폐지 종목을 포함하는 이유는 생존편향이다 (스펙 §11). 과거 기사에 나오는
    회사 중 상당수는 지금 상장돼 있지 않고, 그들을 빼면 표본이
    '살아남은 회사' 쪽으로 기운다.

    **``last_trade_date`` 로 상장 여부를 판정하지 마라.** 실 DB 에서
    ``delisted_stocks`` 4,162 건 중 3,693 건(89%)이 이 값이 NULL 이다.
    폐지 여부는 출처 테이블로 정하며 ``is_delisted`` 로 노출한다.
    ``last_trade_date`` 는 있으면 쓰는 참고값이다.

    같은 코드가 양쪽에 있으면(실 DB 기준 17 건) 상장 쪽을 남긴다 — 재상장·코드
    재사용 사례이고, 현재 거래 가능한 것이 우선이다.
    """
    sql = (
        "select code, name, market, sector, null as last_trade_date, 0 as is_delisted "
        "from stocks "
        "union all "
        "select code, name, market, null as sector, last_trade_date, 1 as is_delisted "
        "from delisted_stocks"
    )
    cur = conn.cursor()
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall(), columns=_MASTER_COLUMNS)
    cur.close()
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce")
    df["is_delisted"] = df["is_delisted"].astype(bool)
    # 상장 행이 먼저 오므로 keep="first" 가 곧 '상장 우선'이다.
    df = df[df["code"] != ""].drop_duplicates(subset="code", keep="first")
    return df.sort_values("code").reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_db_stock_master.py -v`
Expected: 7 passed

- [ ] **Step 5: 실 DB 스모크 확인**

Run:
```bash
docker exec quant-airflow-timescaledb-1 psql -U kr_quant -d kr_quant -tAc \
  "select (select count(*) from stocks), (select count(*) from delisted_stocks)"
```
Expected: `2641|4162` 부근의 값.

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/adapters/db.py tests/test_db_stock_master.py
git commit -m "feat(adapters): 종목 마스터 로더 — 상장+폐지 합본"
```

---

### Task 4: 기사 → 종목 매핑

**Files:**
- Create: `src/swing_it/news/mapping.py`
- Test: `tests/test_news_mapping.py`

**Interfaces:**
- Consumes: `normalize_articles` 의 `[news_id, date, outlet, title, body, orgs]`, `load_stock_master` 의 `[code, name, last_trade_date, is_delisted]`
- Produces:
  - `normalize_name(text: str) -> str` — `주식회사`·`(주)`·`㈜` 제거, 한글/영숫자 외 전부 제거, 소문자화. **공백도 제거한다**(부분문자열 매칭용)
  - `AMBIGUOUS_NAMES: frozenset[str]` — 정규화된 형태의 동음이의 사명 목록
  - `build_name_index(master: pd.DataFrame) -> dict[str, list[str]]` — 정규화 사명 → 종목코드 목록. 우선주(코드 끝자리 ≠ `0`)와 2자 미만 이름은 제외
  - `map_articles(articles: pd.DataFrame, master: pd.DataFrame, *, ambiguous_names: frozenset[str] = AMBIGUOUS_NAMES) -> pd.DataFrame` — 컬럼 `[news_id, date, code, match_kind, outlet, title]`. `match_kind` 는 `"code"` 또는 `"name"`

매핑 규칙 네 가지:

1. **종목코드 우선.** 제목·본문에 6자리 숫자가 단독으로 나오고 그것이 마스터의 코드면 그대로 인정한다(`match_kind="code"`).
2. **최장 사명 우선.** 한 기사에서 매칭된 이름 중 더 긴 이름의 부분문자열인 짧은 이름은 버린다. `한화솔루션` 기사가 `한화` 로도 잡히는 계열사 혼동을 막는다.
3. **동음이의는 기관 컬럼으로 확증.** 정규화 길이 2자 이하이거나 `AMBIGUOUS_NAMES` 에 있는 이름은 빅카인즈 `기관` 컬럼에 같이 나올 때만 인정한다. `대상`(對象)·`삼천리` 같은 일반명사 사명이 여기 해당한다.
4. **point-in-time.** 상장 종목(`is_delisted == False`)은 항상 통과시킨다 — 상장 이전 구간은 가격 조인이 거른다. 폐지 종목은 `last_trade_date` 가 **있을 때만** 그 날짜로 자른다. 89% 는 NULL 이라 여기서 버리면 생존편향이 그대로 돌아오므로, 그 경우도 통과시키고 폐지 이후 구간은 가격 조인에 맡긴다(위 「구현 중 정정」). 같은 정규화 사명에 코드가 둘 이상 붙어 있으면 이 필터를 적용한 뒤 **상장 종목을 우선**하고, 그래도 둘 이상 남으면 **버린다.**

`AMBIGUOUS_NAMES` 초기 목록은 관찰된 것들이며, Task 7 스모크 실행에서 오매핑이 보이면 여기에 추가한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_mapping.py`:

```python
import pandas as pd

from swing_it.news.mapping import build_name_index, map_articles, normalize_name


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("005930", "삼성전자", pd.NaT, False),
            ("005935", "삼성전자우", pd.NaT, False),
            ("009150", "삼성전기", pd.NaT, False),
            ("000880", "한화", pd.NaT, False),
            ("009830", "한화솔루션", pd.NaT, False),
            ("001680", "대상", pd.NaT, False),
            ("037550", "쌍방울", pd.Timestamp("2023-06-30"), True),
            ("900110", "이스트아시아", pd.NaT, True),  # 폐지일 미상 — 89% 가 이 경우다
        ],
        columns=["code", "name", "last_trade_date", "is_delisted"],
    )


def _articles(rows: list[dict]) -> pd.DataFrame:
    base = {"news_id": "n", "date": pd.Timestamp("2026-08-17"), "outlet": "매체",
            "title": "", "body": "", "orgs": ""}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_normalize_name_strips_corporate_suffix_and_spaces():
    assert normalize_name("주식회사 한 화 솔루션") == "한화솔루션"
    assert normalize_name("(주)대상") == "대상"


def test_name_index_excludes_preferred_shares():
    index = build_name_index(_master())
    assert "삼성전자" in index
    assert "삼성전자우" not in index  # 코드 005935 — 끝자리가 0 이 아니다


def test_code_in_body_maps_directly():
    out = map_articles(_articles([{"body": "종목코드 005930 강세"}]), _master())
    assert out["code"].tolist() == ["005930"]
    assert out.iloc[0]["match_kind"] == "code"


def test_name_in_title_maps():
    out = map_articles(_articles([{"title": "삼성전자 3분기 실적 발표"}]), _master())
    assert out["code"].tolist() == ["005930"]
    assert out.iloc[0]["match_kind"] == "name"


def test_longest_name_wins_over_affiliate_substring():
    out = map_articles(_articles([{"title": "한화솔루션 태양광 증설"}]), _master())
    assert out["code"].tolist() == ["009830"]  # 한화(000880) 는 붙지 않는다


def test_similar_sibling_names_are_distinct():
    out = map_articles(_articles([{"title": "삼성전기 MLCC 수주"}]), _master())
    assert out["code"].tolist() == ["009150"]


def test_ambiguous_name_needs_org_column():
    without = map_articles(_articles([{"title": "규제 대상 확대"}]), _master())
    assert without.empty
    with_org = map_articles(
        _articles([{"title": "대상 영업이익 급증", "orgs": "대상"}]), _master()
    )
    assert with_org["code"].tolist() == ["001680"]


def test_delisted_stock_maps_before_last_trade_date():
    articles = _articles([{"title": "쌍방울 유상증자", "date": pd.Timestamp("2022-05-02")}])
    out = map_articles(articles, _master())
    assert out["code"].tolist() == ["037550"]


def test_delisted_stock_does_not_map_after_last_trade_date():
    articles = _articles([{"title": "쌍방울 관련 재판", "date": pd.Timestamp("2026-08-17")}])
    out = map_articles(articles, _master())
    assert out.empty


def test_delisted_without_last_trade_date_still_maps():
    # 폐지일을 모르는 종목을 여기서 버리면 생존편향이 그대로 돌아온다.
    # 폐지 이후 구간은 가격 조인(rank_daily 의 유동성 하한)이 거른다.
    articles = _articles([{"title": "이스트아시아 유상증자", "date": pd.Timestamp("2018-03-02")}])
    out = map_articles(articles, _master())
    assert out["code"].tolist() == ["900110"]


def test_one_article_can_map_to_multiple_codes():
    out = map_articles(
        _articles([{"title": "삼성전자 삼성전기 동반 상승"}]), _master()
    )
    assert sorted(out["code"]) == ["005930", "009150"]


def test_output_columns_and_no_duplicate_pairs():
    out = map_articles(
        _articles([{"title": "삼성전자 실적", "body": "삼성전자 005930 강세"}]), _master()
    )
    assert list(out.columns) == ["news_id", "date", "code", "match_kind", "outlet", "title"]
    assert len(out) == 1
    assert out.iloc[0]["match_kind"] == "code"  # 코드 매칭이 이름 매칭보다 우선


def test_empty_input_returns_empty_frame_with_columns():
    out = map_articles(_articles([]), _master())
    assert out.empty
    assert list(out.columns) == ["news_id", "date", "code", "match_kind", "outlet", "title"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_mapping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news.mapping'`

- [ ] **Step 3: 구현**

`src/swing_it/news/mapping.py`:

```python
"""기사 → 종목 매핑.

기사가 유니버스를 정하므로(스펙 §5) 이 매핑의 정밀도가 곧 표본의 품질이다.
재현율보다 정밀도를 택한다 — 애매하면 버린다. 오매핑된 기사는 라벨에
남의 수익률을 붙여 학습을 오염시키지만, 놓친 기사는 표본을 줄일 뿐이다.
"""

from __future__ import annotations

import re

import pandas as pd

_CODE_RE = re.compile(r"(?<![0-9A-Za-z])([0-9]{6})(?![0-9])")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
_CORPORATE_SUFFIXES = ("주식회사", "(주)", "㈜")

#: 일반명사와 겹쳐 이름만으로는 못 믿는 사명들 (정규화된 형태).
#: 기사 '기관' 컬럼에 같이 나올 때만 인정한다. Task 7 스모크에서
#: 오매핑이 관찰되면 여기에 추가한다.
AMBIGUOUS_NAMES: frozenset[str] = frozenset(
    {
        "대상", "동양", "국제", "삼천리", "남성", "대성", "한창", "조광",
        "세방", "이수", "성창", "서연", "우리들", "신세계",
    }
)

_OUTPUT_COLUMNS = ["news_id", "date", "code", "match_kind", "outlet", "title"]


def normalize_name(text: str) -> str:
    """사명·본문을 부분문자열 비교가 가능한 형태로 정규화한다.

    공백까지 지우는 이유는 기사에서 사명이 띄어 쓰이는 경우가 있어서다
    ('한화 솔루션'). 사명 쪽도 같은 함수로 정규화하므로 양쪽이 일치한다.
    """
    out = str(text)
    for suffix in _CORPORATE_SUFFIXES:
        out = out.replace(suffix, "")
    return _NON_WORD_RE.sub("", out).lower()


def build_name_index(master: pd.DataFrame) -> dict[str, list[str]]:
    """정규화 사명 → 종목코드 목록.

    우선주는 제외한다. KRX 보통주는 종목코드 끝자리가 ``0`` 이며,
    우선주는 이름이 보통주의 부분문자열이라 그대로 두면 항상 같이 잡힌다.
    """
    index: dict[str, list[str]] = {}
    for code, name in zip(master["code"], master["name"]):
        if not str(code).endswith("0"):
            continue
        key = normalize_name(name)
        if len(key) < 2:
            continue
        index.setdefault(key, []).append(str(code))
    return index


def _mappable_on(
    is_delisted: dict[str, bool],
    last_trade: dict[str, pd.Timestamp],
    code: str,
    date: pd.Timestamp,
) -> bool:
    """기사 시점에 매핑을 허용할 종목인가.

    상장 종목은 항상 통과시킨다 — 상장 이전 구간은 가격 조인이 거른다.
    폐지 종목은 ``last_trade_date`` 가 있을 때만 그 날짜로 자른다.
    ``delisted_stocks`` 의 89% 는 이 값이 NULL 이고, 그들을 여기서 버리면
    생존편향을 막으려고 폐지 종목을 넣어놓고 정작 다시 떨어뜨리는 꼴이 된다.
    폐지 이후 구간은 ``rank_daily`` 의 유동성 하한이 거른다.
    """
    if not is_delisted.get(code, False):
        return True
    value = last_trade.get(code)
    return pd.isna(value) or pd.Timestamp(value) >= pd.Timestamp(date)


def map_articles(
    articles: pd.DataFrame,
    master: pd.DataFrame,
    *,
    ambiguous_names: frozenset[str] = AMBIGUOUS_NAMES,
) -> pd.DataFrame:
    """기사마다 언급된 종목을 찾아 (기사, 종목) 쌍으로 펼친다."""
    index = build_name_index(master)
    all_codes = {str(c) for c in master["code"]}
    last_trade = dict(zip(master["code"].astype(str), master["last_trade_date"]))
    is_delisted = dict(zip(master["code"].astype(str), master["is_delisted"].astype(bool)))
    names_by_length = sorted(index, key=len, reverse=True)

    rows: list[dict] = []
    for article in articles.itertuples(index=False):
        raw_text = f"{article.title} {article.body}"
        norm_text = normalize_name(raw_text)
        norm_orgs = normalize_name(article.orgs)
        hits: dict[str, str] = {}

        for code in _CODE_RE.findall(raw_text):
            if code in all_codes and _mappable_on(is_delisted, last_trade, code, article.date):
                hits[code] = "code"

        present = [name for name in names_by_length if name in norm_text]
        for i, name in enumerate(present):
            if any(name in longer for longer in present[:i]):
                continue  # 더 긴 사명이 이미 잡혔다 — 계열사 혼동 방지
            if len(name) <= 2 or name in ambiguous_names:
                if name not in norm_orgs:
                    continue  # 동음이의 위험 — 기관 컬럼 확증이 없으면 버린다
            codes = [
                c for c in index[name]
                if _mappable_on(is_delisted, last_trade, c, article.date)
            ]
            if len(codes) > 1:
                # 동명 종목이 남으면 상장 종목을 우선한다. 그래도 갈리면 버린다.
                codes = [c for c in codes if not is_delisted.get(c, False)]
            if len(codes) != 1:
                continue
            hits.setdefault(codes[0], "name")

        for code, kind in hits.items():
            rows.append(
                {
                    "news_id": article.news_id,
                    "date": article.date,
                    "code": code,
                    "match_kind": kind,
                    "outlet": article.outlet,
                    "title": article.title,
                }
            )

    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    out = pd.DataFrame(rows)[_OUTPUT_COLUMNS]
    return out.sort_values(["date", "code", "news_id"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_mapping.py -v`
Expected: 13 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/news/mapping.py tests/test_news_mapping.py
git commit -m "feat(news): 기사→종목 매핑 — 코드 우선·최장 사명·동음이의 확증"
```

---

### Task 5: 중복 기사 제거

**Files:**
- Create: `src/swing_it/news/dedupe.py`
- Test: `tests/test_news_dedupe.py`

**Interfaces:**
- Consumes: `map_articles` 의 `[news_id, date, code, match_kind, outlet, title]`
- Produces:
  - `normalize_title(title: str) -> str` — 대괄호·소괄호 안 내용 제거(`[특징주]` 등), 한글/영숫자 외를 공백으로, 소문자화, 공백 정리
  - `title_tokens(title: str) -> set[str]` — 정규화 제목의 2자 이상 토큰 집합
  - `jaccard(a: set[str], b: set[str]) -> float` — 한쪽이라도 비면 `0.0`
  - `cluster_duplicates(mapped: pd.DataFrame, *, threshold: float = 0.6) -> pd.DataFrame` — 입력 컬럼 + `cluster_id`(int, `(date, code)` 안에서 0부터) + `is_primary`(bool)

같은 사건을 여러 매체가 받아쓴 것은 **하나의 사건 1건**으로 세되, **매체 수는 보존한다.** 매체 수 자체가 핫함의 한 축이기 때문이다(스펙 §4). 그래서 여기서 행을 지우지 않고 `cluster_id` 로 묶기만 한다 — 삭제는 정보를 잃는다.

비교는 `(date, code)` 그룹 안에서만 한다. 다른 종목·다른 날의 비슷한 제목은 다른 사건이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_dedupe.py`:

```python
import pandas as pd

from swing_it.news.dedupe import cluster_duplicates, jaccard, normalize_title, title_tokens


def _mapped(rows: list[tuple[str, str, str]], date="2026-08-17", code="005930"):
    return pd.DataFrame(
        [
            {
                "news_id": news_id,
                "date": pd.Timestamp(date),
                "code": code,
                "match_kind": "name",
                "outlet": outlet,
                "title": title,
            }
            for news_id, outlet, title in rows
        ]
    )


def test_normalize_title_drops_bracket_prefix():
    assert normalize_title("[특징주] 삼성전자, 3% 상승") == "삼성전자 3 상승"


def test_title_tokens_drop_one_char_tokens():
    assert title_tokens("삼성전자 3% 상승") == {"삼성전자", "상승"}


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_with_empty_set_is_zero():
    assert jaccard(set(), {"a"}) == 0.0


def test_jaccard_of_identical_sets_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_same_event_from_two_outlets_forms_one_cluster():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 미국 반도체 공장 증설 발표한다"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out["cluster_id"].nunique() == 1
    assert out["is_primary"].sum() == 1


def test_different_events_stay_separate():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 배당 정책 변경 검토"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out["cluster_id"].nunique() == 2


def test_clusters_do_not_cross_codes():
    df = pd.concat([
        _mapped([("n1", "한국경제", "실적 개선 전망")], code="005930"),
        _mapped([("n2", "매일경제", "실적 개선 전망")], code="000660"),
    ], ignore_index=True)
    out = cluster_duplicates(df, threshold=0.6)
    assert len(out) == 2
    assert out["is_primary"].sum() == 2  # 종목이 다르면 각자 대표


def test_clusters_do_not_cross_dates():
    df = pd.concat([
        _mapped([("n1", "한국경제", "실적 개선 전망")], date="2026-08-17"),
        _mapped([("n2", "매일경제", "실적 개선 전망")], date="2026-08-18"),
    ], ignore_index=True)
    out = cluster_duplicates(df, threshold=0.6)
    assert out["is_primary"].sum() == 2


def test_primary_is_lowest_news_id_in_cluster():
    df = _mapped([
        ("n9", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n1", "매일경제", "삼성전자 미국 반도체 공장 증설 발표"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert out[out["is_primary"]]["news_id"].tolist() == ["n1"]


def test_all_rows_are_kept():
    df = _mapped([
        ("n1", "한국경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n2", "매일경제", "삼성전자 미국 반도체 공장 증설 발표"),
        ("n3", "머니투데이", "삼성전자 미국 반도체 공장 증설 발표"),
    ])
    out = cluster_duplicates(df, threshold=0.6)
    assert len(out) == 3  # 매체 수를 세야 하므로 행은 지우지 않는다


def test_empty_input_keeps_columns():
    df = _mapped([])
    out = cluster_duplicates(df)
    assert out.empty
    assert "cluster_id" in out.columns and "is_primary" in out.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_dedupe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news.dedupe'`

- [ ] **Step 3: 구현**

`src/swing_it/news/dedupe.py`:

```python
"""같은 사건을 다룬 중복 기사 클러스터링.

통신사 기사를 여러 매체가 받아쓰면 한 사건이 수십 건으로 부풀어 기사 건수
급증 배수가 왜곡된다. 그래서 사건 단위로 묶어 센다.

행을 지우지는 않는다. 보도 매체 수가 핫함의 독립적인 한 축이기 때문이다
(스펙 §4) — 삭제하면 그 정보가 사라진다.
"""

from __future__ import annotations

import re

import pandas as pd

_BRACKET_RE = re.compile(r"[\[\(【][^\]\)】]*[\]\)】]")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

_EXTRA_COLUMNS = ["cluster_id", "is_primary"]


def normalize_title(title: str) -> str:
    """괄호 안 머리표를 떼고 토큰만 남긴다."""
    text = _BRACKET_RE.sub(" ", str(title))
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.lower().split())


def title_tokens(title: str) -> set[str]:
    """2자 이상 토큰 집합. 1자 토큰은 조사·수치 파편이라 변별력이 없다."""
    return {token for token in normalize_title(title).split() if len(token) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    """토큰 집합의 자카드 유사도."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_duplicates(mapped: pd.DataFrame, *, threshold: float = 0.6) -> pd.DataFrame:
    """``(date, code)`` 안에서 제목이 유사한 기사를 한 클러스터로 묶는다.

    ``news_id`` 순으로 훑으며 기존 클러스터의 **대표 기사** 제목과 비교하는
    탐욕적 방식이다. 그룹 크기가 하루 수십 건이라 O(n²) 로 충분하고,
    입력 순서가 ``news_id`` 로 고정돼 결과가 재현된다.
    """
    if mapped.empty:
        out = mapped.copy()
        out["cluster_id"] = pd.Series(dtype="int64")
        out["is_primary"] = pd.Series(dtype="bool")
        return out

    blocks = []
    for _, group in mapped.groupby(["date", "code"], sort=False):
        group = group.sort_values("news_id")
        representatives: list[set[str]] = []
        assigned: list[int] = []
        for title in group["title"]:
            tokens = title_tokens(title)
            match = None
            for cluster_id, rep_tokens in enumerate(representatives):
                if jaccard(tokens, rep_tokens) >= threshold:
                    match = cluster_id
                    break
            if match is None:
                match = len(representatives)
                representatives.append(tokens)
            assigned.append(match)
        block = group.copy()
        block["cluster_id"] = assigned
        block["is_primary"] = ~block.duplicated(subset="cluster_id", keep="first")
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["date", "code", "news_id"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_dedupe.py -v`
Expected: 12 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/news/dedupe.py tests/test_news_dedupe.py
git commit -m "feat(news): 제목 자카드 기반 중복 기사 클러스터링"
```

---

### Task 6: '핫함' 계산과 일별 랭킹

**Files:**
- Create: `src/swing_it/news/hotness.py`
- Test: `tests/test_news_hotness.py`

**Interfaces:**
- Consumes: `cluster_duplicates` 결과, 계획 1 `events.add_liquidity_stats(prices, window=20)` 의 `[code, date, adv, tv_ratio]`
- Produces:
  - `W_ARTICLES: float = 0.4`, `W_OUTLETS: float = 0.3`, `W_TRADE: float = 0.3` — 사전등록 가중치
  - `daily_counts(clustered: pd.DataFrame) -> pd.DataFrame` — 컬럼 `[date, code, n_articles, n_outlets, primary_news_id]`. `n_articles` 는 **클러스터 수**(사건 수), `n_outlets` 는 고유 매체 수
  - `add_hotness(counts: pd.DataFrame, liquidity: pd.DataFrame, *, baseline_window: int = 60, min_periods: int = 20, baseline_floor: float = 0.5) -> pd.DataFrame` — 입력 + `[baseline_articles, article_ratio, adv, tv_ratio, hotness]`
  - `rank_daily(hot: pd.DataFrame, *, adv_floor: float = 5000.0, train_top: int = 50, ops_top: int = 10) -> pd.DataFrame` — 컬럼 `[date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, hotness, rank, in_train, in_ops, primary_news_id]`. 학습셋(`rank <= train_top`)만 반환

핫함 3축은 전부 빅카인즈·krx-news-rest-api·가격DB 양쪽에서 같은 방식으로 계산된다(스펙 §4):

```
hotness = 0.4·log1p(기사 급증 배수) + 0.3·log1p(매체 수) + 0.3·log1p(거래대금 급증 배수)
```

**baseline 은 달력일 기준으로 센다.** 기사가 난 날만 모아 평균 내면 "평소"가 과대평가된다 — 평소에 기사가 안 나는 종목일수록 급증 배수가 낮게 나와 정반대 방향의 편향이 생긴다. 그래서 `(전 종목 × 전 날짜)` 격자로 펼쳐 0 을 채운 뒤 롤링한다. `.shift(1)` 로 당일을 제외하는 것은 계획 1 의 `adv` 와 같은 이유다.

`baseline_floor=0.5` 는 평소 기사가 0 인 종목의 분모를 막는다. 이 값에서 "평소 0건이던 종목에 1건" = 배수 2.0 이 된다.

랭킹 대상은 `adv >= 5000` 만 남긴다. 가격 행이 없는 종목(상장 전·거래정지)은 `adv` 가 `NaN` 이라 자동으로 탈락한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_hotness.py`:

```python
import numpy as np
import pandas as pd

from swing_it.news.hotness import (
    W_ARTICLES,
    W_OUTLETS,
    W_TRADE,
    add_hotness,
    daily_counts,
    rank_daily,
)


def _clustered(rows: list[dict]) -> pd.DataFrame:
    base = {"news_id": "n", "date": pd.Timestamp("2026-08-17"), "code": "005930",
            "match_kind": "name", "outlet": "매체", "title": "제목",
            "cluster_id": 0, "is_primary": True}
    return pd.DataFrame([{**base, **row} for row in rows])


def _liquidity(codes: list[str], dates: list[str], adv: float, tv_ratio: float):
    return pd.DataFrame(
        [
            {"code": c, "date": pd.Timestamp(d), "adv": adv, "tv_ratio": tv_ratio}
            for c in codes
            for d in dates
        ]
    )


def test_counts_use_clusters_not_rows():
    df = _clustered([
        {"news_id": "n1", "outlet": "A", "cluster_id": 0, "is_primary": True},
        {"news_id": "n2", "outlet": "B", "cluster_id": 0, "is_primary": False},
        {"news_id": "n3", "outlet": "C", "cluster_id": 1, "is_primary": True},
    ])
    out = daily_counts(df)
    assert out.iloc[0]["n_articles"] == 2  # 사건 2건
    assert out.iloc[0]["n_outlets"] == 3   # 매체 3곳


def test_counts_pick_first_primary_as_representative():
    df = _clustered([
        {"news_id": "n5", "outlet": "A", "cluster_id": 0, "is_primary": True},
        {"news_id": "n2", "outlet": "B", "cluster_id": 1, "is_primary": True},
    ])
    out = daily_counts(df)
    assert out.iloc[0]["primary_news_id"] == "n2"


def test_baseline_counts_zero_article_days():
    # 3일 중 2일은 기사가 없다. baseline 은 달력일 평균이어야 한다.
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 3,
         "n_outlets": 3, "primary_news_id": "n1"},
        {"date": pd.Timestamp("2026-01-04"), "code": "A", "n_articles": 4,
         "n_outlets": 4, "primary_news_id": "n2"},
    ])
    liq = _liquidity(["A"], dates, adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=3, min_periods=3)
    row = out[out["date"] == pd.Timestamp("2026-01-04")].iloc[0]
    # 1/1~1/3 의 기사 건수는 3, 0, 0 → 평균 1.0
    assert row["baseline_articles"] == 1.0
    assert row["article_ratio"] == 4.0


def test_baseline_floor_applies_when_history_is_short():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 2,
         "n_outlets": 2, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    assert out.iloc[0]["baseline_articles"] == 0.5
    assert out.iloc[0]["article_ratio"] == 4.0


def test_baseline_excludes_current_day():
    dates = ["2026-01-01", "2026-01-02"]
    counts = pd.DataFrame([
        {"date": pd.Timestamp(d), "code": "A", "n_articles": 2,
         "n_outlets": 1, "primary_news_id": "n"} for d in dates
    ])
    liq = _liquidity(["A"], dates, adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=1, min_periods=1)
    second = out[out["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert second["baseline_articles"] == 2.0  # 전날 값. 당일 2 가 섞이면 lookahead


def test_hotness_matches_weighted_formula():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 2,
         "n_outlets": 4, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=3.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    expected = (
        W_ARTICLES * np.log1p(4.0) + W_OUTLETS * np.log1p(4.0) + W_TRADE * np.log1p(3.0)
    )
    assert np.isclose(out.iloc[0]["hotness"], expected)


def test_missing_liquidity_row_yields_nan_adv():
    counts = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "Z", "n_articles": 1,
         "n_outlets": 1, "primary_news_id": "n1"},
    ])
    liq = _liquidity(["A"], ["2026-01-01"], adv=10000.0, tv_ratio=1.0)
    out = add_hotness(counts, liq, baseline_window=60, min_periods=20)
    assert np.isnan(out.iloc[0]["adv"])
    assert out.iloc[0]["tv_ratio"] == 1.0  # 거래대금 급증 없음으로 간주


def test_rank_drops_illiquid_and_missing_price_rows():
    hot = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 5, "n_outlets": 5,
         "article_ratio": 5.0, "adv": 10000.0, "tv_ratio": 3.0, "hotness": 2.0,
         "primary_news_id": "n1"},
        {"date": pd.Timestamp("2026-01-01"), "code": "B", "n_articles": 9, "n_outlets": 9,
         "article_ratio": 9.0, "adv": 100.0, "tv_ratio": 9.0, "hotness": 3.0,
         "primary_news_id": "n2"},
        {"date": pd.Timestamp("2026-01-01"), "code": "C", "n_articles": 9, "n_outlets": 9,
         "article_ratio": 9.0, "adv": np.nan, "tv_ratio": 1.0, "hotness": 3.0,
         "primary_news_id": "n3"},
    ])
    out = rank_daily(hot, adv_floor=5000.0)
    assert out["code"].tolist() == ["A"]


def test_rank_is_per_date_and_flags_train_and_ops():
    rows = []
    for day in ("2026-01-01", "2026-01-02"):
        for i in range(60):
            rows.append({
                "date": pd.Timestamp(day), "code": f"{i:06d}", "n_articles": 1,
                "n_outlets": 1, "article_ratio": 1.0, "adv": 10000.0,
                "tv_ratio": 1.0, "hotness": float(60 - i), "primary_news_id": f"n{i}",
            })
    out = rank_daily(pd.DataFrame(rows), train_top=50, ops_top=10)
    assert len(out) == 100                       # 날짜당 50건
    assert out["rank"].min() == 1
    assert out["rank"].max() == 50
    assert out.groupby("date")["in_ops"].sum().tolist() == [10, 10]


def test_rank_orders_by_hotness_descending():
    rows = [
        {"date": pd.Timestamp("2026-01-01"), "code": c, "n_articles": 1, "n_outlets": 1,
         "article_ratio": 1.0, "adv": 10000.0, "tv_ratio": 1.0, "hotness": h,
         "primary_news_id": "n"}
        for c, h in [("A", 1.0), ("B", 3.0), ("C", 2.0)]
    ]
    out = rank_daily(pd.DataFrame(rows))
    assert out["code"].tolist() == ["B", "C", "A"]


def test_rank_output_columns():
    rows = [{"date": pd.Timestamp("2026-01-01"), "code": "A", "n_articles": 1,
             "n_outlets": 1, "article_ratio": 1.0, "adv": 10000.0, "tv_ratio": 1.0,
             "hotness": 1.0, "primary_news_id": "n1"}]
    out = rank_daily(pd.DataFrame(rows))
    assert list(out.columns) == [
        "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
        "tv_ratio", "hotness", "rank", "in_train", "in_ops", "primary_news_id",
    ]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_hotness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news.hotness'`

- [ ] **Step 3: 구현**

`src/swing_it/news/hotness.py`:

```python
"""'핫함' 3축 계산과 일별 랭킹.

백테스트 유효성의 핵심은 과거와 현재가 같은 방식으로 계산되는 것이다
(스펙 §4). 조회수·댓글수처럼 과거 데이터가 없는 지표는 쓰지 않는다.
여기 쓰는 세 축은 빅카인즈·krx-news-rest-api·가격DB 양쪽에서 모두 나온다.

가중치는 사전등록 값이다 (스펙 §6). 성과를 보고 조정하기 시작하면
그 순간 검증이 무의미해진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

W_ARTICLES = 0.4
W_OUTLETS = 0.3
W_TRADE = 0.3

_COUNT_COLUMNS = ["date", "code", "n_articles", "n_outlets", "primary_news_id"]
_RANK_COLUMNS = [
    "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
    "tv_ratio", "hotness", "rank", "in_train", "in_ops", "primary_news_id",
]


def daily_counts(clustered: pd.DataFrame) -> pd.DataFrame:
    """(날짜, 종목) 별 사건 수와 매체 수.

    ``n_articles`` 는 기사 행 수가 아니라 **클러스터 수**다. 통신사 기사를
    여러 매체가 받아쓴 것은 사건 하나로 센다.
    """
    if clustered.empty:
        return pd.DataFrame(columns=_COUNT_COLUMNS)

    counts = (
        clustered.groupby(["date", "code"])
        .agg(n_articles=("cluster_id", "nunique"), n_outlets=("outlet", "nunique"))
        .reset_index()
    )
    primary = (
        clustered[clustered["is_primary"]]
        .sort_values(["date", "code", "news_id"])
        .groupby(["date", "code"], as_index=False)["news_id"]
        .first()
        .rename(columns={"news_id": "primary_news_id"})
    )
    out = counts.merge(primary, on=["date", "code"], how="left")
    return out[_COUNT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _dense_grid(counts: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """(전 종목 × 전 거래일) 격자로 펼치고 기사 없는 날을 0 으로 채운다.

    기사가 난 날만 모아 롤링하면 '평소'가 과대평가된다. 평소에 조용한
    종목일수록 급증 배수가 낮게 나오는, 정확히 반대 방향의 편향이다.

    날짜 축은 가격 데이터의 거래일에서 가져온다. 기사가 난 날만으로
    축을 만들면 그것 자체가 같은 편향을 다시 만든다.
    """
    axis = pd.Index(sorted(set(calendar) | set(counts["date"])), name="date")
    dates = axis
    codes = pd.Index(sorted(counts["code"].unique()), name="code")
    grid = pd.MultiIndex.from_product([codes, dates]).to_frame(index=False)
    dense = grid.merge(counts, on=["code", "date"], how="left")
    dense["n_articles"] = dense["n_articles"].fillna(0.0)
    return dense.sort_values(["code", "date"])


def add_hotness(
    counts: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    baseline_window: int = 60,
    min_periods: int = 20,
    baseline_floor: float = 0.5,
) -> pd.DataFrame:
    """기사 급증 배수·매체 수·거래대금 급증 배수를 합쳐 핫함 점수를 만든다.

    Args:
        liquidity: 계획 1 ``events.add_liquidity_stats`` 의 ``[code, date, adv, tv_ratio]``.
            baseline 의 달력 축도 여기서 가져온다 — 거래일 목록의 유일한 출처다.
        baseline_window: 평소 기사 건수를 재는 달력일 구간.
        min_periods: 이 일수를 못 채우면 baseline 을 ``baseline_floor`` 로 둔다.
        baseline_floor: 평소 0건인 종목의 분모 하한. 0.5 이면 '평소 0건 → 오늘 1건'이
            배수 2.0 이 된다.
    """
    if counts.empty:
        return pd.DataFrame(
            columns=[*_COUNT_COLUMNS, "baseline_articles", "article_ratio",
                     "adv", "tv_ratio", "hotness"]
        )

    dense = _dense_grid(counts, liquidity["date"])
    baseline = dense.groupby("code", sort=False)["n_articles"].transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=min_periods).mean()
    )
    dense["baseline_articles"] = baseline.fillna(baseline_floor).clip(lower=baseline_floor)

    out = counts.merge(
        dense[["code", "date", "baseline_articles"]], on=["code", "date"], how="left"
    )
    out["article_ratio"] = out["n_articles"] / out["baseline_articles"]

    out = out.merge(
        liquidity[["code", "date", "adv", "tv_ratio"]], on=["code", "date"], how="left"
    )
    # 가격 행이 없으면 거래대금 급증도 없었던 것으로 본다. adv 는 NaN 으로 남겨
    # 랭킹 단계의 유동성 하한에서 탈락시킨다.
    out["tv_ratio"] = out["tv_ratio"].fillna(1.0)

    out["hotness"] = (
        W_ARTICLES * np.log1p(out["article_ratio"])
        + W_OUTLETS * np.log1p(out["n_outlets"])
        + W_TRADE * np.log1p(out["tv_ratio"])
    )
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def rank_daily(
    hot: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    train_top: int = 50,
    ops_top: int = 10,
) -> pd.DataFrame:
    """유동성 하한을 통과한 종목을 날짜별 핫함 순으로 세우고 학습셋을 자른다.

    학습은 일 50건, 운영은 상위 10건이다 (스펙 §5). 학습 표본을 넉넉히
    잡는 이유는 두 가지다 — 표본이 4배가 되고, '핫함 순위' 자체가 피처가 된다.

    ``adv`` 가 ``NaN`` 인 행(상장 전·거래정지)은 비교가 False 라 자동 탈락한다.
    """
    if hot.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)

    out = hot[hot["adv"] >= adv_floor].copy()
    out = out.sort_values(["date", "hotness", "code"], ascending=[True, False, True])
    out["rank"] = out.groupby("date").cumcount() + 1
    out["in_train"] = out["rank"] <= train_top
    out["in_ops"] = out["rank"] <= ops_top
    out = out[out["in_train"]]
    return out[_RANK_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_hotness.py -v`
Expected: 11 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/news/hotness.py tests/test_news_hotness.py
git commit -m "feat(news): 핫함 3축 계산 + 학습50/운영10 일별 랭킹"
```

---

### Task 7: sqlite 적재와 `swing-news` CLI

**Files:**
- Create: `src/swing_it/news/store.py`
- Create: `src/swing_it/cli_news.py`
- Test: `tests/test_news_store.py`

**Interfaces:**
- Consumes: `load_bigkinds_dir`, `load_stock_master`, `map_articles`, `cluster_duplicates`, `daily_counts`, `add_hotness`, `rank_daily`, 계획 1 의 `connect` · `load_prices` · `events.add_liquidity_stats`
- Produces:
  - `init_db(path: str | Path) -> sqlite3.Connection` — 테이블 3개를 만들고 커넥션을 돌려준다
  - `upsert_articles(conn: sqlite3.Connection, articles: pd.DataFrame) -> int` — `news_id` 기준 덮어쓰기. 반환은 기록한 행 수
  - `upsert_article_codes(conn: sqlite3.Connection, mapped: pd.DataFrame) -> int` — `(news_id, code)` 기준 덮어쓰기
  - `replace_daily_news(conn: sqlite3.Connection, ranked: pd.DataFrame) -> int` — 등장한 날짜 구간을 통째로 지우고 다시 넣는다(재실행 안전)
  - `load_daily_news(conn: sqlite3.Connection, start: str, end: str, *, ops_only: bool = False) -> pd.DataFrame` — 컬럼 `[date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, hotness, rank, in_ops, primary_news_id]`, `date` 는 `datetime64[ns]`
  - `main(argv: list[str] | None = None) -> int` — CLI 진입점

sqlite 스키마:

| 테이블 | 키 | 내용 |
|---|---|---|
| `articles` | `news_id` | 기사 원문 (본문 200자) |
| `article_codes` | `(news_id, code)` | 종목 매핑 |
| `daily_news` | `(date, code)` | 일별 핫함·랭킹 |

`replace_daily_news` 가 날짜 구간을 지우고 다시 넣는 이유는, 핫함 baseline 이 뒤에 들어온 데이터에 따라 바뀌기 때문이다. 부분 갱신은 과거 행과 새 행의 계산 기준이 달라져 재현이 깨진다.

> **계획 3 과의 계약 주의 (2026-08-17 구현 중 발견).** 계획 3 의 `adapters/news_db.py` 는
> `articles JOIN article_codes` 로 읽는다. 따라서 **`article_codes` 에 행이 없는 기사는
> 조인에서 통째로 빠진다.** 그러면 계획 3 의 사전 필터가 세는 `no_code` 카운트가 실
> 데이터에서 항상 0 이 되어, "종목 매핑에 몇 건이나 실패했는가"를 아무도 볼 수 없다.
>
> 매핑 실패를 관측 가능하게 하려면 둘 중 하나를 택해야 한다:
> (a) 매핑 실패 기사에 `code = ''`, `match_kind = 'none'` 행을 넣는다 — 계획 3 의
> 사전 필터가 `no_code` 로 집계한다, 또는
> (b) 인제스트 리포트에서 매핑 실패 건수를 따로 출력한다.
>
> **(a) 를 권한다.** 실패가 파이프라인 뒤쪽까지 눈에 보이는 편이 낫고, 계획 3 의
> 사전 필터가 이미 그 경우를 처리한다. 어느 쪽이든 Task 4·7 에서 명시적으로 정해라 —
> 정하지 않으면 매핑 실패가 조용히 사라진다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_store.py`:

```python
import pandas as pd

from swing_it.news.store import (
    init_db,
    load_daily_news,
    replace_daily_news,
    upsert_article_codes,
    upsert_articles,
)


def _articles(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"news_id": nid, "date": pd.Timestamp(day), "outlet": "매체",
             "title": title, "body": "본문", "keywords": "", "orgs": "", "url": ""}
            for nid, day, title in rows
        ]
    )


def _ranked(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(day), "code": code, "n_articles": 3, "n_outlets": 5,
             "article_ratio": 6.0, "adv": 10000.0, "tv_ratio": 2.0, "hotness": hotness,
             "rank": rank, "in_train": True, "in_ops": rank <= 10,
             "primary_news_id": "n1"}
            for day, code, rank, hotness in rows
        ]
    )


def test_init_creates_three_tables(tmp_path):
    conn = init_db(tmp_path / "news.db")
    names = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='table'")
    }
    assert {"articles", "article_codes", "daily_news"} <= names


def test_articles_roundtrip(tmp_path):
    conn = init_db(tmp_path / "news.db")
    written = upsert_articles(conn, _articles([("a", "2026-08-17", "제목")]))
    assert written == 1
    row = conn.execute("select news_id, date, title from articles").fetchone()
    assert row == ("a", "2026-08-17", "제목")


def test_articles_upsert_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "news.db")
    upsert_articles(conn, _articles([("a", "2026-08-17", "옛 제목")]))
    upsert_articles(conn, _articles([("a", "2026-08-17", "새 제목")]))
    rows = conn.execute("select title from articles").fetchall()
    assert rows == [("새 제목",)]


def test_article_codes_upsert_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "news.db")
    mapped = pd.DataFrame([{"news_id": "a", "date": pd.Timestamp("2026-08-17"),
                            "code": "005930", "match_kind": "name",
                            "outlet": "매체", "title": "제목"}])
    upsert_article_codes(conn, mapped)
    upsert_article_codes(conn, mapped)
    assert conn.execute("select count(*) from article_codes").fetchone()[0] == 1


def test_daily_news_roundtrip_types(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([("2026-08-17", "005930", 1, 2.5)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out.iloc[0]["code"] == "005930"
    assert out.iloc[0]["rank"] == 1
    assert bool(out.iloc[0]["in_ops"]) is True


def test_replace_wipes_the_date_range_before_inserting(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-17", "005930", 1, 2.5),
        ("2026-08-17", "000660", 2, 2.0),
    ]))
    replace_daily_news(conn, _ranked([("2026-08-17", "005930", 1, 9.9)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert out["code"].tolist() == ["005930"]
    assert out.iloc[0]["hotness"] == 9.9


def test_replace_leaves_other_dates_alone(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([("2026-08-10", "005930", 1, 1.0)]))
    replace_daily_news(conn, _ranked([("2026-08-17", "000660", 1, 2.0)]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert len(out) == 2


def test_load_respects_date_range(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-10", "005930", 1, 1.0),
        ("2026-08-17", "000660", 1, 2.0),
    ]))
    out = load_daily_news(conn, "2026-08-15", "2026-08-31")
    assert out["code"].tolist() == ["000660"]


def test_ops_only_filters_to_top_ten(tmp_path):
    conn = init_db(tmp_path / "news.db")
    replace_daily_news(conn, _ranked([
        ("2026-08-17", f"{i:06d}", i + 1, float(50 - i)) for i in range(30)
    ]))
    out = load_daily_news(conn, "2026-08-01", "2026-08-31", ops_only=True)
    assert len(out) == 10


def test_load_on_empty_db_returns_typed_empty_frame(tmp_path):
    conn = init_db(tmp_path / "news.db")
    out = load_daily_news(conn, "2026-08-01", "2026-08-31")
    assert out.empty
    assert "hotness" in out.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_news_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.news.store'`

- [ ] **Step 3: 저장소 구현**

`src/swing_it/news/store.py`:

```python
"""로컬 sqlite 적재.

실 DB(TimescaleDB)는 읽기 전용이므로 뉴스 산출물은 전부 여기로 간다.
파일은 ``data/news.db`` 이고 ``.gitignore`` 가 차단한다 — 빅카인즈는
재배포를 금지하며 기사에는 저작권이 있다 (스펙 §8).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    news_id  TEXT PRIMARY KEY,
    date     TEXT NOT NULL,
    outlet   TEXT,
    title    TEXT,
    body     TEXT,
    keywords TEXT,
    orgs     TEXT,
    url      TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);

CREATE TABLE IF NOT EXISTS article_codes (
    news_id    TEXT NOT NULL,
    code       TEXT NOT NULL,
    match_kind TEXT NOT NULL,
    PRIMARY KEY (news_id, code)
);
CREATE INDEX IF NOT EXISTS idx_article_codes_code ON article_codes(code);

CREATE TABLE IF NOT EXISTS daily_news (
    date             TEXT NOT NULL,
    code             TEXT NOT NULL,
    n_articles       INTEGER,
    n_outlets        INTEGER,
    article_ratio    REAL,
    adv              REAL,
    tv_ratio         REAL,
    hotness          REAL,
    "rank"           INTEGER,
    in_ops           INTEGER,
    primary_news_id  TEXT,
    PRIMARY KEY (date, code)
);
"""

_ARTICLE_COLUMNS = ["news_id", "date", "outlet", "title", "body", "keywords", "orgs", "url"]
_DAILY_COLUMNS = [
    "date", "code", "n_articles", "n_outlets", "article_ratio", "adv",
    "tv_ratio", "hotness", "rank", "in_ops", "primary_news_id",
]


def init_db(path: str | Path) -> sqlite3.Connection:
    """sqlite 파일을 열고 스키마를 보장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _date_strings(series: pd.Series) -> list[str]:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").tolist()


def upsert_articles(conn: sqlite3.Connection, articles: pd.DataFrame) -> int:
    """기사를 ``news_id`` 기준으로 덮어쓴다."""
    if articles.empty:
        return 0
    df = articles.copy()
    df["date"] = _date_strings(df["date"])
    rows = list(df[_ARTICLE_COLUMNS].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT OR REPLACE INTO articles "
        "(news_id, date, outlet, title, body, keywords, orgs, url) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_article_codes(conn: sqlite3.Connection, mapped: pd.DataFrame) -> int:
    """(기사, 종목) 매핑을 덮어쓴다."""
    if mapped.empty:
        return 0
    rows = list(
        mapped[["news_id", "code", "match_kind"]].itertuples(index=False, name=None)
    )
    conn.executemany(
        "INSERT OR REPLACE INTO article_codes (news_id, code, match_kind) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def replace_daily_news(conn: sqlite3.Connection, ranked: pd.DataFrame) -> int:
    """등장한 날짜들을 통째로 지우고 다시 넣는다.

    부분 갱신을 하지 않는 이유는 핫함 baseline 이 이후에 들어온 데이터에
    따라 달라지기 때문이다. 계산 기준이 다른 행이 섞이면 재현이 깨진다.
    """
    if ranked.empty:
        return 0
    df = ranked.copy()
    df["date"] = _date_strings(df["date"])
    df["in_ops"] = df["in_ops"].astype(int)

    dates = sorted(set(df["date"]))
    conn.executemany("DELETE FROM daily_news WHERE date = ?", [(d,) for d in dates])
    rows = list(df[_DAILY_COLUMNS].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO daily_news "
        "(date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, "
        " hotness, \"rank\", in_ops, primary_news_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def load_daily_news(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    *,
    ops_only: bool = False,
) -> pd.DataFrame:
    """일별 핫함 테이블을 읽는다. ``ops_only`` 면 운영 상위 10건만."""
    sql = (
        "SELECT date, code, n_articles, n_outlets, article_ratio, adv, tv_ratio, "
        "hotness, \"rank\", in_ops, primary_news_id FROM daily_news "
        "WHERE date >= ? AND date <= ?"
    )
    if ops_only:
        sql += " AND in_ops = 1"
    sql += " ORDER BY date, \"rank\""
    rows = conn.execute(sql, (start, end)).fetchall()
    df = pd.DataFrame(rows, columns=_DAILY_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["in_ops"] = df["in_ops"].astype(bool)
    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_news_store.py -v`
Expected: 10 passed

- [ ] **Step 5: CLI 구현**

`src/swing_it/cli_news.py`:

```python
"""`swing-news` — 빅카인즈 엑셀 → 종목 매핑 → 핫함 → 로컬 sqlite.

수동 다운로드를 전제한 반자동 파이프라인이다. 사람이 하는 일은
`data/bigkinds/` 에 엑셀을 놓는 것뿐이고 나머지는 이 명령이 한다.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .adapters.db import connect, load_prices, load_stock_master
from .events import add_liquidity_stats
from .news.bigkinds import load_bigkinds_dir
from .news.dedupe import cluster_duplicates
from .news.hotness import add_hotness, daily_counts, rank_daily
from .news.mapping import map_articles
from .news.store import init_db, replace_daily_news, upsert_article_codes, upsert_articles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="빅카인즈 뉴스 인제스트")
    parser.add_argument("--xlsx-dir", default="data/bigkinds", help="엑셀 다운로드 디렉터리")
    parser.add_argument("--news-db", default="data/news.db", help="산출 sqlite 경로")
    parser.add_argument("--db", default=None, help="TimescaleDB DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--train-top", type=int, default=50)
    parser.add_argument("--ops-top", type=int, default=10)
    args = parser.parse_args(argv)

    articles = load_bigkinds_dir(args.xlsx_dir)
    print(f"기사 {len(articles):,} 건", file=sys.stderr)
    if articles.empty:
        print("읽은 기사가 없다. --xlsx-dir 를 확인하라.", file=sys.stderr)
        return 1

    start = articles["date"].min().strftime("%Y-%m-%d")
    end = articles["date"].max().strftime("%Y-%m-%d")
    print(f"기간 {start} ~ {end}", file=sys.stderr)

    source = connect(args.db)
    master = load_stock_master(source)
    # adv 는 직전 20거래일 평균이라 앞쪽 여유가 필요하다. 60 달력일이면 충분하다.
    price_start = (articles["date"].min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    prices = load_prices(source, price_start, end)
    print(f"종목 마스터 {len(master):,} · 가격 {len(prices):,} 행", file=sys.stderr)

    mapped = map_articles(articles, master)
    print(f"매핑 {len(mapped):,} 쌍 · 종목 {mapped['code'].nunique():,}", file=sys.stderr)
    if mapped.empty:
        print("매핑된 기사가 없다. schema.COLUMN_MAP 과 종목 마스터를 확인하라.", file=sys.stderr)
        return 1

    clustered = cluster_duplicates(mapped)
    counts = daily_counts(clustered)
    hot = add_hotness(counts, add_liquidity_stats(prices))
    ranked = rank_daily(
        hot, adv_floor=args.adv_floor, train_top=args.train_top, ops_top=args.ops_top
    )
    print(f"학습셋 {len(ranked):,} 건 · 날짜 {ranked['date'].nunique():,}", file=sys.stderr)

    store = init_db(args.news_db)
    upsert_articles(store, articles)
    upsert_article_codes(store, mapped)
    replace_daily_news(store, ranked)
    store.close()

    per_day = len(ranked) / max(ranked["date"].nunique(), 1)
    print(f"적재 완료: {args.news_db} (일평균 {per_day:.1f} 건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 전체 테스트 통과 확인**

Run: `uv run pytest -v`
Expected: 계획 2 몫 70 passed (6+11+7+13+12+11+10), 그리고 기존 테스트 전부 통과. 계획 1·3 테스트가 이미 들어와 있으므로 총계는 그보다 크다 — 중요한 건 **실패가 0** 이라는 것이다.

- [ ] **Step 7: 실 데이터로 파이프라인 실행**

Run:
```bash
export $(grep -v '^#' ../kr-quant/.env | xargs)
uv run swing-news --xlsx-dir data/bigkinds --news-db data/news.db
```
Expected: `data/news.db` 생성, 일평균 건수가 출력된다.

- [ ] **Step 8: 매핑 품질 육안 확인**

Run:
```bash
uv run python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/news.db')
q = '''select d.date, d.code, s.title, d.n_articles, d.n_outlets,
              round(d.article_ratio,1) ar, round(d.tv_ratio,1) tv, round(d.hotness,2) h
       from daily_news d join articles s on s.news_id = d.primary_news_id
       where d.in_ops = 1 order by d.date desc, d.\"rank\" limit 30'''
pd.set_option('display.width', 200, 'display.max_colwidth', 45)
print(pd.read_sql(q, conn))
print()
print('match_kind 분포:', pd.read_sql(
    'select match_kind, count(*) n from article_codes group by 1', conn).to_dict('records'))
"
```
Expected: 상위 30건의 제목과 종목이 실제로 맞아떨어진다.

**오매핑이 보이면 고칠 곳은 한 군데다** — `news/mapping.py` 의 `AMBIGUOUS_NAMES` 에 그 사명을 추가한다(정규화된 형태로). 고친 뒤 Task 4 테스트에 그 사례를 케이스로 추가하고 다시 돌린다. 일평균이 50건에 크게 못 미치면 유동성 하한(`--adv-floor`)이 아니라 다운로드 범위가 좁은 것이니 엑셀을 더 받는다.

- [ ] **Step 9: 커밋**

```bash
git add src/swing_it/news/store.py src/swing_it/cli_news.py tests/test_news_store.py
git commit -m "feat(news): 로컬 sqlite 적재 + swing-news 인제스트 CLI"
```

---

## Self-Review 체크

**스펙 커버리지 (계획 2 범위):**

| 스펙 요구 | 태스크 |
|---|---|
| §4 빅카인즈 수동 다운로드 → 정규화 적재 | Task 1·2·7 |
| §4 본문 200자 통일 | Task 1 `BODY_MAX_CHARS`, Task 2 절단 |
| §4 핫함 ① 기사 건수 급증 배수 | Task 6 `article_ratio` (달력일 baseline · shift(1)) |
| §4 핫함 ② 보도 매체 수 | Task 5 매체 보존 → Task 6 `n_outlets` |
| §4 핫함 ③ 거래대금 급증 배수 | Task 6 — 계획 1 `add_liquidity_stats` 의 `tv_ratio` 재사용 |
| §5 기사가 유니버스를 정한다 | Task 4 매핑 결과가 곧 후보 풀 |
| §5 point-in-time 유동성 하한 | Task 6 `rank_daily(adv_floor=5000)` |
| §5 학습 30~50건 / 운영 10건 | Task 6 `train_top=50`, `ops_top=10` |
| §6 사전등록 (가중치 고정) | Task 6 `W_ARTICLES/W_OUTLETS/W_TRADE` 상수 |
| §8 어댑터 경계 | Task 3 — TimescaleDB SQL 은 `adapters/db.py` 에만 |
| §8 데이터 커밋 금지 | `.gitignore` 기존 규칙 + 산출물은 `data/` 아래로만 |
| §11 생존편향 | Task 3 `delisted_stocks` 합본 + `is_delisted`, Task 4 `_mappable_on` |
| §12 열린 질문 — 다운로드 쿼리 구성 | 「빅카인즈 다운로드 절차」 — 분기 단위 분할, 20,000행 경고 |
| §12 열린 질문 — 동음이의·계열사 | Task 4 규칙 2·3, Task 7 Step 8 로 피드백 루프 |

**계획 2 범위 밖 (후속 계획):** LLM 피처 추출, 층별 모델 비교, 라벨 결합, 근거 카드, krx-news-rest-api 실시간 경로 연결.

**미해소로 남기는 것:** 빅카인즈 엑셀의 실제 헤더는 Task 1 Step 3 에서 사람이 확인한다. 그 전까지 `COLUMN_MAP` 왼쪽 값은 **가정**이며, Step 8 의 계약 확인을 통과해야 확정이다.

## 후속 계획

- 계획 3: LLM 피처 추출 (provider 추상화 · 배치 · 결과 캐싱)
- 계획 4: 층별 모델 비교(①수급 → ②+규칙뉴스 → ③+LLM) + 근거 카드 리포트
