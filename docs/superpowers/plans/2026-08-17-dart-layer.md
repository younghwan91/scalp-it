# 계획 5: DART 공시 층 (②층) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DART 전자공시를 무료·완전·point-in-time 으로 수집해 ②층(수급 + 공시) 피처를 만들고, 계획 4 의 층 비교 하네스로 ①층 대비 기여를 판정한다.

**Architecture:** `adapters/dart.py` 가 OpenDART REST 를 감싸고, `dart/` 가 순수 함수로 유형 분류·구조화 파싱·피처를 만든다. 저장은 로컬 sqlite(`data/dart.db`). 계획 4 의 `build_layered_dataset` 에 `news=` 대신 `disc=` 프레임을 넘겨 층을 구성한다.

**Tech Stack:** Python ≥3.10, httpx(이미 의존성), pandas, sqlite3, pytest

**Spec:** `docs/superpowers/specs/2026-08-17-swing-it-design.md`

## 왜 이 계획이 생겼나

뉴스 경로가 둘 다 막히거나 비쌌다.

- **빅카인즈 웹**: 무료지만 1회 20,000건 상한. 경제 전체가 하루 6,000건이라 1년치에 수동 다운로드 36회+. 게다가 요청 기간의 앞부분이 통째로 잘린다(8/10~8/17 요청에 8/10·8/11 소실 확인).
- **뉴스토어 API**: 10년치 가능하고 본문 전문을 주지만 **유료**(견적제). 검증 전 지출은 이 프로젝트가 두 번 실패한 순서다.
- **`krx-news-rest-api`**: 자체 보유지만 Redis TTL 1시간이라 과거가 없다.

DART 는 이 셋의 문제가 전부 없다.

| | DART |
|---|---|
| 비용 | 무료 |
| 기간 | 10년+ |
| 시점 정확성 | 접수일자까지 정확 — 뉴스보다 깨끗하다 |
| 인증 | `quant-airflow/.env` 에 **키 3개 이미 보유** (호출 한도 분산용) |
| 원문 | 무료 제공 (`document.xml`) |

**결정적으로, 우리가 LLM 에게 추정시키려던 피처가 공시에는 사실로 적혀 있다.** 실측 예 (에프알텍 073540, 2026-08-14 단일판매·공급계약체결, 정제 후 658자):

```
계약금액 총액   5,584,228,670원
최근 매출액    25,263,688,167원
매출액 대비        22.1%              → 스펙 §3 "규모 대비"
계약기간      2026-08-12 ~ 2026-11-04  → 스펙 §3 "지속성 단서"
3년간 동종계약   해당                   → 스펙 §3 "신선도"
계약상대방     (주)케이티 (매출 28조)      → 상대방 신뢰도
```

**한계는 분명하다.** 공시는 재료의 부분집합이다 — 테마·정책·해외 수요·애널리스트 리포트·루머는 공시가 없다. 시장 관심도(보도 매체 수) 축도 없다. **뉴스의 대체가 아니라, 뉴스를 사기 전에 무료로 던지는 질문이다.**

②층이 ①층을 못 이기면 뉴스 구매의 근거가 약해진다. 이기면 "재료가 D+20~40 을 설명한다"가 무료로 증명되고, 그때 뉴스 지출은 근거 있는 결정이 된다.

## Global Constraints

- **DART 는 읽기 전용.** 수집 결과는 로컬 `data/dart.db` 에만 쓴다. TimescaleDB 에 쓰지 않는다.
- **API 키는 환경변수로만.** `quant-airflow/.env` 의 `DART_API_KEY`·`DART_API_KEY_2`·`DART_API_KEY_3` 을 읽는다. 코드·테스트·커밋에 키를 넣지 않는다.
- **point-in-time 엄수.** 이벤트일 t 의 피처는 `rcept_dt <= t` 인 공시만 쓴다. 공시는 접수일자가 곧 공개 시점이다.
- **거래일 이월.** 휴장일 접수 공시는 다음 거래일로 옮긴다. 계획 2 의 `news.calendar.align_to_trading_days` 를 **그대로 재사용한다** — 같은 문제를 두 번 풀지 않는다.
- **테스트는 네트워크에 나가지 않는다.** 전부 주입된 fake transport 와 sqlite fixture 로 한다.
- **데이터 커밋 금지.** `data/` 는 이미 `.gitignore` 차단.
- **`top_n` 은 동시 보유 기준.** 스펙 §2 는 동시 보유 15종목 이상이고, 동시 보유 ≈ `top_n × hold_end` 다.
  → `docs/research/2026-08-17-topn-preregistration.md`
- **부정 결과를 숨기는 경로를 만들지 않는다.** FAIL 이 나오면 파라미터를 바꿔 재시도하지 않는다 (스펙 §10).

## 확인된 API 사실 (2026-08-17 실호출)

**공시검색 `GET https://opendart.fss.or.kr/api/list.json`**

| 파라미터 | 값 |
|---|---|
| `crtfc_key` | 인증키 (필수) |
| `bgn_de` · `end_de` | YYYYMMDD. **`corp_code` 없이는 검색기간 3개월 제한** |
| `corp_cls` | `Y`(유가) `K`(코스닥) `N`(코넥스) `E`(기타) |
| `pblntf_ty` | `A`정기 `B`주요사항 `C`발행 `D`지분 `E`기타 `F`외부감사 `G`펀드 `H`자산유동화 `I`거래소 `J`공정위 |
| `page_count` | **최대 100** |
| `page_no` | 1부터 |
| `sort` / `sort_mth` | `date` / `asc`·`desc` |

응답: `status`(정상은 `"000"`), `total_count`, `total_page`, `list[]` 의 `corp_cls`·`corp_name`·`corp_code`·`stock_code`·`report_nm`·`rcept_no`·`flr_nm`·`rcept_dt`·`rm`.

실측: 2026-08-10~08-14 유가증권 `total_count` 2,302. 유가+코스닥 합쳐 대략 하루 1,000~1,500건.

**원문 `GET https://opendart.fss.or.kr/api/document.xml?rcept_no=...`**
ZIP 바이트를 돌려준다. 안에 XML 파일 하나. 앞부분에 `<STYLE>` CSS 블록이 있으므로 **먼저 걷어내고** 태그를 제거해야 한다. 위 예시는 정제 후 658자.

**호출 한도:** 문서에 일 20,000건 언급. 키가 3개이므로 실질 60,000건/일.

**수집량 산정:** 하루 1,200건 × 250거래일 × 10년 = 300만 건. `page_count=100` 이면 30,000 콜. 3개월 제한 때문에 구간을 40개로 나눠야 하지만 콜 수는 같다. **키 3개로 하루 안에 10년치가 들어온다.**

## File Structure

| 파일 | 책임 |
|---|---|
| `src/swing_it/adapters/dart.py` | OpenDART HTTP. **DART 호출은 이 파일에만** |
| `src/swing_it/dart/__init__.py` | 패키지 |
| `src/swing_it/dart/classify.py` | `report_nm` → 재료 유형 (순수 함수) |
| `src/swing_it/dart/contract.py` | 공급계약 원문 → 구조화 필드 (순수 함수) |
| `src/swing_it/dart/store.py` | `data/dart.db` 스키마·적재·조회 |
| `src/swing_it/features/disclosure.py` | ②층 피처 (순수 함수) |
| `src/swing_it/cli_dart.py` | `swing-dart` — 수집 CLI |
| `tests/` | 각 모듈 대응 |

---

### Task 1: OpenDART 어댑터 — 공시검색

**Files:**
- Create: `src/swing_it/adapters/dart.py`
- Test: `tests/test_dart_adapter.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `DART_BASE: str` = `"https://opendart.fss.or.kr/api"`
  - `class DartError(Exception)`
  - `api_keys(env: dict | None = None) -> list[str]` — `DART_API_KEY`, `DART_API_KEY_2`, `DART_API_KEY_3` 중 설정된 것만 순서대로. 하나도 없으면 `DartError`
  - `class DartClient` — `__init__(self, keys: list[str], *, client: httpx.Client | None = None, sleep: float = 0.0)`
    - `search(self, *, bgn_de: str, end_de: str, corp_cls: str, page_no: int = 1, page_count: int = 100) -> dict` — 한 페이지 원본 응답. `status != "000"` 이면 `DartError`. 단 `"013"`(조회 데이터 없음)은 빈 결과로 정상 처리
    - `search_all(self, *, bgn_de: str, end_de: str, corp_cls: str) -> list[dict]` — 전 페이지를 모아 `list` 항목들을 반환
  - 키 로테이션: 호출마다 다음 키로 돌린다(라운드로빈). 한도 초과 코드(`"020"`)면 다음 키로 즉시 재시도하고, 모든 키가 막히면 `DartError`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dart_adapter.py`:

```python
import httpx
import pytest

from swing_it.adapters.dart import DartClient, DartError, api_keys


def _resp(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _page(items, total_page=1, status="000"):
    return {
        "status": status,
        "message": "정상" if status == "000" else "오류",
        "page_no": 1,
        "page_count": 100,
        "total_count": len(items),
        "total_page": total_page,
        "list": items,
    }


_ITEM = {
    "corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
    "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
    "rcept_no": "20260814000123", "flr_nm": "에프알텍", "rcept_dt": "20260814", "rm": "",
}


def test_api_keys_reads_all_three_in_order():
    env = {"DART_API_KEY": "a", "DART_API_KEY_2": "b", "DART_API_KEY_3": "c"}
    assert api_keys(env) == ["a", "b", "c"]


def test_api_keys_skips_missing_and_blank():
    env = {"DART_API_KEY": "a", "DART_API_KEY_2": "", "DART_API_KEY_3": "c"}
    assert api_keys(env) == ["a", "c"]


def test_api_keys_raises_when_none_set():
    with pytest.raises(DartError):
        api_keys({})


def test_search_returns_payload():
    transport = httpx.MockTransport(lambda request: _resp(_page([_ITEM])))
    client = DartClient(["k1"], client=httpx.Client(transport=transport))
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    assert got["list"][0]["stock_code"] == "073540"


def test_search_sends_expected_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return _resp(_page([]))

    client = DartClient(["k1"], client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.search(bgn_de="20260101", end_de="20260331", corp_cls="Y", page_no=3)
    assert seen["bgn_de"] == "20260101"
    assert seen["end_de"] == "20260331"
    assert seen["corp_cls"] == "Y"
    assert seen["page_no"] == "3"
    assert seen["page_count"] == "100"
    assert seen["crtfc_key"] == "k1"


def test_status_013_is_empty_not_error():
    """조회 데이터 없음은 오류가 아니다. 공시 없는 날이 실제로 있다."""
    payload = {"status": "013", "message": "조회된 데이터가 없습니다."}
    client = DartClient(["k1"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp(payload))))
    assert client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")["list"] == []


def test_other_error_status_raises():
    payload = {"status": "010", "message": "등록되지 않은 키입니다."}
    client = DartClient(["k1"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp(payload))))
    with pytest.raises(DartError):
        client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")


def test_rate_limited_key_falls_through_to_next():
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.params["crtfc_key"]
        used.append(key)
        if key == "k1":
            return _resp({"status": "020", "message": "요청 제한을 초과하였습니다."})
        return _resp(_page([_ITEM]))

    client = DartClient(["k1", "k2"], client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    assert used == ["k1", "k2"]


def test_all_keys_rate_limited_raises():
    client = DartClient(["k1", "k2"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp({"status": "020", "message": "초과"}))))
    with pytest.raises(DartError):
        client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")


def test_search_all_walks_every_page():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page_no"])
        item = dict(_ITEM, rcept_no=f"2026081400{page:04d}")
        return _resp(_page([item], total_page=3))

    client = DartClient(["k1"], client=httpx.Client(transport=httpx.MockTransport(handler)))
    rows = client.search_all(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert len(rows) == 3
    assert [r["rcept_no"][-4:] for r in rows] == ["0001", "0002", "0003"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_dart_adapter.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.adapters.dart'`

- [ ] **Step 3: 구현**

`src/swing_it/adapters/dart.py`:

```python
"""OpenDART HTTP 어댑터 — DART 호출은 이 파일에만 존재한다.

무료·공식·10년+·접수일자 단위 point-in-time. 인증키는 환경변수로만 받는다.

키를 여러 개 돌리는 이유는 일 20,000건 한도 때문이다. quant-airflow 가 이미
3개를 발급받아 두었고, 10년치 수집에 약 30,000 콜이 필요하다.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

DART_BASE = "https://opendart.fss.or.kr/api"

_KEY_ENV = ("DART_API_KEY", "DART_API_KEY_2", "DART_API_KEY_3")
_OK = "000"
_NO_DATA = "013"
_RATE_LIMITED = "020"


class DartError(Exception):
    """OpenDART 가 오류를 돌려주었거나 인증키가 없다."""


def api_keys(env: dict | None = None) -> list[str]:
    """설정된 DART 인증키를 순서대로 모은다."""
    source = os.environ if env is None else env
    keys = [str(source.get(name) or "").strip() for name in _KEY_ENV]
    keys = [k for k in keys if k]
    if not keys:
        raise DartError(
            "DART 인증키가 없다. DART_API_KEY 를 설정하라 "
            "(quant-airflow/.env 에 이미 있다면 export 해서 쓰면 된다)."
        )
    return keys


class DartClient:
    """공시검색 페이지를 읽는다. 키 로테이션과 한도 초과 폴백을 포함한다."""

    def __init__(
        self,
        keys: list[str],
        *,
        client: httpx.Client | None = None,
        sleep: float = 0.0,
    ) -> None:
        if not keys:
            raise DartError("인증키가 비어 있다.")
        self._keys = list(keys)
        self._cursor = 0
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep

    def _next_key(self) -> str:
        key = self._keys[self._cursor % len(self._keys)]
        self._cursor += 1
        return key

    def search(
        self,
        *,
        bgn_de: str,
        end_de: str,
        corp_cls: str,
        page_no: int = 1,
        page_count: int = 100,
    ) -> dict[str, Any]:
        """공시검색 한 페이지. 한도 초과면 다음 키로 넘어간다."""
        last_message = ""
        for _ in range(len(self._keys)):
            params = {
                "crtfc_key": self._next_key(),
                "bgn_de": bgn_de,
                "end_de": end_de,
                "corp_cls": corp_cls,
                "page_no": str(page_no),
                "page_count": str(page_count),
            }
            if self._sleep:
                time.sleep(self._sleep)
            payload = self._client.get(f"{DART_BASE}/list.json", params=params).json()
            status = str(payload.get("status"))
            if status == _OK:
                payload.setdefault("list", [])
                return payload
            if status == _NO_DATA:
                # 공시가 하루도 없는 구간은 실제로 존재한다. 오류가 아니다.
                return {**payload, "list": [], "total_page": 0, "total_count": 0}
            last_message = f"{status} {payload.get('message')}"
            if status != _RATE_LIMITED:
                raise DartError(f"OpenDART 오류: {last_message}")
        raise DartError(f"모든 인증키가 한도 초과다: {last_message}")

    def search_all(self, *, bgn_de: str, end_de: str, corp_cls: str) -> list[dict[str, Any]]:
        """구간의 모든 페이지를 모은다."""
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.search(
                bgn_de=bgn_de, end_de=end_de, corp_cls=corp_cls, page_no=page
            )
            rows.extend(payload.get("list") or [])
            total_page = int(payload.get("total_page") or 0)
            if page >= total_page:
                return rows
            page += 1
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_dart_adapter.py -q`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/adapters/dart.py tests/test_dart_adapter.py
git commit -m "feat(adapters): OpenDART 공시검색 어댑터 — 키 로테이션·한도 폴백"
```

---

### Task 2: 공시 유형 분류

**Files:**
- Create: `src/swing_it/dart/__init__.py`
- Create: `src/swing_it/dart/classify.py`
- Test: `tests/test_dart_classify.py`

**Interfaces:**
- Consumes: `report_nm` 문자열
- Produces:
  - `MATERIAL_TYPES: list[str]` = `["공급계약", "시설투자", "지분변동", "자사주", "실적변동", "기술_인허가", "구조조정", "악재", "정기보고", "기타"]`
  - `classify_report(report_nm: str) -> str` — 위 중 하나
  - `is_correction(report_nm: str) -> bool` — `[기재정정]`·`[첨부정정]`·`[첨부추가]` 접두 여부
  - `classify_frame(frame: pd.DataFrame) -> pd.DataFrame` — `report_nm` 컬럼을 읽어 `material_type`·`is_correction` 을 붙인다

**분류 규칙 (사전등록).** 성과를 보고 바꾸지 않는다. 규칙은 위에서 아래로 처음 맞는 것을 쓴다.

| 유형 | 매칭 키워드 |
|---|---|
| `악재` | 감사의견, 의견거절, 자본잠식, 상장폐지, 거래정지, 회생절차, 파산, 부도, 횡령, 배임, 소송, 손실발생 |
| `공급계약` | 공급계약, 수주, 판매계약 |
| `시설투자` | 신규시설투자, 유형자산, 공장, 증설 |
| `지분변동` | 유상증자, 무상증자, 전환사채, 신주인수권, 교환사채, 대량보유, 최대주주 |
| `자사주` | 자기주식 |
| `실적변동` | 매출액또는손익구조, 영업실적, 매출액미달 |
| `기술_인허가` | 특허, 임상, 품목허가, 승인, 인증 |
| `구조조정` | 합병, 분할, 양수, 양도, 영업정지 |
| `정기보고` | 사업보고서, 반기보고서, 분기보고서, 감사보고서, 검토 |
| `기타` | 나머지 |

`악재` 를 맨 위에 두는 이유: "소송 등의 판결"처럼 다른 키워드와 겹치는 제목이 있는데, 악재 여부가 방향을 결정하므로 우선한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dart_classify.py`:

```python
import pandas as pd
import pytest

from swing_it.dart.classify import (
    MATERIAL_TYPES,
    classify_frame,
    classify_report,
    is_correction,
)


@pytest.mark.parametrize(
    ("report_nm", "expected"),
    [
        ("단일판매ㆍ공급계약체결", "공급계약"),
        ("신규시설투자등", "시설투자"),
        ("유상증자결정", "지분변동"),
        ("무상증자결정", "지분변동"),
        ("주식등의대량보유상황보고서", "지분변동"),
        ("자기주식취득결정", "자사주"),
        ("자기주식처분결정", "자사주"),
        ("매출액또는손익구조30%이상변동", "실적변동"),
        ("특허권취득", "기술_인허가"),
        ("반기보고서", "정기보고"),
        ("사업보고서", "정기보고"),
        ("회사합병결정", "구조조정"),
        ("타법인주식및출자증권양수결정", "구조조정"),
        ("반기검토의견부적정,의견거절또는완전자본잠식사실발생", "악재"),
        ("주권매매거래정지", "악재"),
        ("기업설명회", "기타"),
    ],
)
def test_classify_report(report_nm, expected):
    assert classify_report(report_nm) == expected


def test_every_result_is_a_declared_type():
    for nm in ("단일판매ㆍ공급계약체결", "듣도보도못한공시", ""):
        assert classify_report(nm) in MATERIAL_TYPES


def test_bad_news_wins_over_other_keywords():
    """'소송 등의 판결'은 구조조정 키워드(양도)와 겹칠 수 있다. 악재가 우선한다."""
    assert classify_report("소송등의판결ㆍ결정") == "악재"


def test_correction_prefix_detected():
    assert is_correction("[기재정정]단일판매ㆍ공급계약체결")
    assert is_correction("[첨부추가]반기보고서")
    assert not is_correction("단일판매ㆍ공급계약체결")


def test_correction_does_not_change_type():
    assert classify_report("[기재정정]단일판매ㆍ공급계약체결") == "공급계약"


def test_whitespace_is_tolerated():
    assert classify_report("  단일판매ㆍ공급계약체결   ") == "공급계약"


def test_classify_frame_adds_two_columns():
    frame = pd.DataFrame({"report_nm": ["단일판매ㆍ공급계약체결", "[기재정정]반기보고서"]})
    out = classify_frame(frame)
    assert list(out["material_type"]) == ["공급계약", "정기보고"]
    assert list(out["is_correction"]) == [False, True]


def test_classify_frame_keeps_other_columns():
    frame = pd.DataFrame({"report_nm": ["유상증자결정"], "rcept_no": ["x"]})
    out = classify_frame(frame)
    assert out.iloc[0]["rcept_no"] == "x"


def test_classify_frame_on_empty():
    out = classify_frame(pd.DataFrame({"report_nm": []}))
    assert list(out.columns) == ["report_nm", "material_type", "is_correction"]
    assert out.empty
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_dart_classify.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.dart'`

- [ ] **Step 3: 구현**

`src/swing_it/dart/__init__.py`:

```python
"""DART 공시 처리 — 수집은 adapters/dart.py, 여기는 순수 함수."""
```

`src/swing_it/dart/classify.py`:

```python
"""공시 제목을 재료 유형으로 분류한다 — 규칙 기반, LLM 불필요.

공시는 뉴스와 달리 **유형이 제목에 규격화돼 있다.** 금융감독원 서식이 정한
이름이라 표현이 흔들리지 않는다. 그래서 LLM 없이도 재료 유형을 얻을 수 있고,
이것이 ②층을 무료로 만드는 근거다.

규칙은 사전등록이다. 성과를 보고 바꾸면 그 순간 검증이 무의미해진다.
"""

from __future__ import annotations

import pandas as pd

MATERIAL_TYPES: list[str] = [
    "공급계약", "시설투자", "지분변동", "자사주", "실적변동",
    "기술_인허가", "구조조정", "악재", "정기보고", "기타",
]

# 위에서 아래로 처음 맞는 것을 쓴다. '악재'가 맨 위인 이유는 방향을 결정하기
# 때문이다 — "소송등의판결"은 구조조정 키워드와 겹치지만 악재로 봐야 한다.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("악재", ("감사의견", "의견거절", "부적정", "자본잠식", "상장폐지", "거래정지",
              "회생절차", "파산", "부도", "횡령", "배임", "소송", "손실발생",
              "실질심사", "미달사실발생")),
    ("공급계약", ("공급계약", "수주", "판매계약")),
    ("시설투자", ("신규시설투자", "유형자산", "공장", "증설")),
    ("지분변동", ("유상증자", "무상증자", "전환사채", "신주인수권", "교환사채",
                 "대량보유", "최대주주")),
    ("자사주", ("자기주식",)),
    ("실적변동", ("매출액또는손익구조", "영업실적", "매출액미달")),
    ("기술_인허가", ("특허", "임상", "품목허가", "승인", "인증")),
    ("구조조정", ("합병", "분할", "양수", "양도", "영업정지")),
    ("정기보고", ("사업보고서", "반기보고서", "분기보고서", "감사보고서", "검토")),
]

_CORRECTION_MARKERS = ("[기재정정]", "[첨부정정]", "[첨부추가]", "[정정]")


def classify_report(report_nm: str) -> str:
    """공시 제목을 재료 유형 하나로 매핑한다. 못 맞추면 ``기타``."""
    text = str(report_nm or "").strip()
    for marker in _CORRECTION_MARKERS:
        text = text.replace(marker, "")
    text = text.strip()
    for material_type, keywords in _RULES:
        if any(keyword in text for keyword in keywords):
            return material_type
    return "기타"


def is_correction(report_nm: str) -> bool:
    """정정·첨부 공시인가. 원 공시의 재탕이라 신선도가 낮다."""
    text = str(report_nm or "").strip()
    return any(text.startswith(marker) for marker in _CORRECTION_MARKERS)


def classify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """``report_nm`` 컬럼에 ``material_type``·``is_correction`` 을 붙인다."""
    out = frame.copy()
    out["material_type"] = [classify_report(v) for v in out.get("report_nm", [])]
    out["is_correction"] = [is_correction(v) for v in out.get("report_nm", [])]
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_dart_classify.py -q`
Expected: 24 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/dart tests/test_dart_classify.py
git commit -m "feat(dart): 공시 제목 → 재료 유형 규칙 분류 (사전등록)"
```

---

### Task 3: 공시 저장소

**Files:**
- Create: `src/swing_it/dart/store.py`
- Test: `tests/test_dart_store.py`

**Interfaces:**
- Consumes: `classify_frame` 결과
- Produces:
  - `DISC_COLUMNS: list[str]` = `["rcept_no", "rcept_dt", "code", "corp_name", "corp_cls", "report_nm", "material_type", "is_correction"]`
  - `init_dart_db(path: str | Path) -> sqlite3.Connection` — 스키마 생성 후 연결 반환
  - `rows_to_frame(rows: list[dict]) -> pd.DataFrame` — API 응답 항목 → `DISC_COLUMNS` 프레임. **`stock_code` 가 빈 항목은 버린다**(비상장 법인)
  - `upsert_disclosures(conn, frame) -> int` — `rcept_no` 기준 덮어쓰기, 반영 건수 반환
  - `load_disclosures(conn, start: str, end: str) -> pd.DataFrame` — `DISC_COLUMNS`, `rcept_dt` 는 `datetime64[ns]`
  - `collected_range(conn) -> tuple[str, str] | None` — 적재된 `rcept_dt` 최소·최대

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dart_store.py`:

```python
import pandas as pd
import pytest

from swing_it.dart.store import (
    DISC_COLUMNS,
    collected_range,
    init_dart_db,
    load_disclosures,
    rows_to_frame,
    upsert_disclosures,
)

_ROWS = [
    {"corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
     "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
     "rcept_no": "20260814000123", "flr_nm": "에프알텍", "rcept_dt": "20260814", "rm": ""},
    {"corp_cls": "Y", "corp_name": "삼성전자", "corp_code": "00126380",
     "stock_code": "005930", "report_nm": "반기보고서",
     "rcept_no": "20260814000999", "flr_nm": "삼성전자", "rcept_dt": "20260814", "rm": ""},
    # 비상장 법인 — stock_code 가 비어 있다
    {"corp_cls": "E", "corp_name": "비상장회사", "corp_code": "00999999",
     "stock_code": " ", "report_nm": "감사보고서제출",
     "rcept_no": "20260814000777", "flr_nm": "비상장회사", "rcept_dt": "20260814", "rm": ""},
]


@pytest.fixture()
def conn(tmp_path):
    return init_dart_db(tmp_path / "dart.db")


def test_rows_to_frame_has_contract_columns():
    frame = rows_to_frame(_ROWS)
    assert list(frame.columns) == DISC_COLUMNS


def test_rows_to_frame_drops_unlisted():
    """상장 종목만 쓴다. stock_code 없는 법인은 가격이 없어 라벨을 못 만든다."""
    frame = rows_to_frame(_ROWS)
    assert len(frame) == 2
    assert set(frame["code"]) == {"073540", "005930"}


def test_rows_to_frame_classifies():
    frame = rows_to_frame(_ROWS).set_index("code")
    assert frame.loc["073540", "material_type"] == "공급계약"
    assert frame.loc["005930", "material_type"] == "정기보고"


def test_rows_to_frame_on_empty():
    frame = rows_to_frame([])
    assert list(frame.columns) == DISC_COLUMNS
    assert frame.empty


def test_upsert_then_load(conn):
    n = upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert n == 2
    got = load_disclosures(conn, "20260801", "20260831")
    assert len(got) == 2
    assert pd.api.types.is_datetime64_any_dtype(got["rcept_dt"])


def test_upsert_is_idempotent(conn):
    frame = rows_to_frame(_ROWS)
    upsert_disclosures(conn, frame)
    upsert_disclosures(conn, frame)
    assert len(load_disclosures(conn, "20260801", "20260831")) == 2


def test_load_respects_range(conn):
    upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert load_disclosures(conn, "20260815", "20260831").empty


def test_collected_range_reports_bounds(conn):
    assert collected_range(conn) is None
    upsert_disclosures(conn, rows_to_frame(_ROWS))
    assert collected_range(conn) == ("20260814", "20260814")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_dart_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.dart.store'`

- [ ] **Step 3: 구현**

`src/swing_it/dart/store.py`:

```python
"""공시 로컬 저장소 — sqlite ``data/dart.db``.

TimescaleDB 는 quant-airflow 소관이라 여기서 쓰지 않는다. 수집 결과는
로컬에만 둔다(``data/`` 는 gitignore 차단).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .classify import classify_report, is_correction

DISC_COLUMNS: list[str] = [
    "rcept_no", "rcept_dt", "code", "corp_name", "corp_cls",
    "report_nm", "material_type", "is_correction",
]

_CREATE = """
CREATE TABLE IF NOT EXISTS disclosures (
    rcept_no      TEXT PRIMARY KEY,
    rcept_dt      TEXT NOT NULL,
    code          TEXT NOT NULL,
    corp_name     TEXT,
    corp_cls      TEXT,
    report_nm     TEXT,
    material_type TEXT,
    is_correction INTEGER
);
CREATE INDEX IF NOT EXISTS idx_disc_dt ON disclosures(rcept_dt);
CREATE INDEX IF NOT EXISTS idx_disc_code ON disclosures(code);
"""


def init_dart_db(path: str | Path) -> sqlite3.Connection:
    """스키마를 만들고 연결을 돌려준다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(_CREATE)
    conn.commit()
    return conn


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """공시검색 응답 항목을 저장 계약에 맞춘다.

    ``stock_code`` 가 빈 항목은 버린다. 비상장 법인이라 가격이 없고,
    가격이 없으면 라벨을 만들 수 없다.
    """
    records = []
    for row in rows:
        code = str(row.get("stock_code") or "").strip()
        if not code:
            continue
        report_nm = str(row.get("report_nm") or "").strip()
        records.append(
            {
                "rcept_no": str(row.get("rcept_no") or "").strip(),
                "rcept_dt": str(row.get("rcept_dt") or "").strip(),
                "code": code,
                "corp_name": str(row.get("corp_name") or "").strip(),
                "corp_cls": str(row.get("corp_cls") or "").strip(),
                "report_nm": report_nm,
                "material_type": classify_report(report_nm),
                "is_correction": is_correction(report_nm),
            }
        )
    return pd.DataFrame(records, columns=DISC_COLUMNS)


def upsert_disclosures(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """``rcept_no`` 기준으로 덮어쓴다."""
    if frame.empty:
        return 0
    payload = [
        (
            r["rcept_no"], r["rcept_dt"], r["code"], r["corp_name"],
            r["corp_cls"], r["report_nm"], r["material_type"], int(bool(r["is_correction"])),
        )
        for _, r in frame.iterrows()
    ]
    conn.executemany(
        "INSERT INTO disclosures "
        "(rcept_no, rcept_dt, code, corp_name, corp_cls, report_nm, material_type, is_correction) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(rcept_no) DO UPDATE SET "
        "rcept_dt=excluded.rcept_dt, code=excluded.code, corp_name=excluded.corp_name, "
        "corp_cls=excluded.corp_cls, report_nm=excluded.report_nm, "
        "material_type=excluded.material_type, is_correction=excluded.is_correction",
        payload,
    )
    conn.commit()
    return len(payload)


def load_disclosures(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    """``rcept_dt`` 구간(YYYYMMDD 문자열 비교)의 공시를 읽는다."""
    rows = conn.execute(
        f"select {', '.join(DISC_COLUMNS)} from disclosures "
        "where rcept_dt >= ? and rcept_dt <= ? order by rcept_dt, rcept_no",
        (start, end),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=DISC_COLUMNS)
    frame["rcept_dt"] = pd.to_datetime(frame["rcept_dt"], format="%Y%m%d", errors="coerce")
    frame["is_correction"] = frame["is_correction"].astype(bool)
    return frame


def collected_range(conn: sqlite3.Connection) -> tuple[str, str] | None:
    """적재된 접수일자의 최소·최대. 비어 있으면 ``None``."""
    row = conn.execute("select min(rcept_dt), max(rcept_dt) from disclosures").fetchone()
    if not row or row[0] is None:
        return None
    return (str(row[0]), str(row[1]))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_dart_store.py -q`
Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/dart/store.py tests/test_dart_store.py
git commit -m "feat(dart): 공시 로컬 저장소 — 상장 종목만, rcept_no 멱등 적재"
```

---

### Task 4: ②층 공시 피처

**Files:**
- Create: `src/swing_it/features/disclosure.py`
- Test: `tests/test_features_disclosure.py`

**Interfaces:**
- Consumes: `load_disclosures` 결과, 계획 1 `events.add_liquidity_stats` 의 `[code, date, adv, tv_ratio]`, 계획 2 `news.calendar.align_to_trading_days`
- Produces:
  - `DISC_FEATURE_COLUMNS: list[str]` = `["disc_count", "disc_ratio", "disc_is_bad", "disc_is_contract", "disc_is_capex", "disc_is_equity", "disc_correction_ratio", "days_since_disc"]`
  - `daily_disclosures(disc: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame` — 거래일 이월 후 `(code, date)` 로 집계. 컬럼 `[code, date, disc_count, n_bad, n_contract, n_capex, n_equity, n_correction, top_material_type]`
  - `disclosure_features(daily: pd.DataFrame, calendar: pd.Series, *, baseline_window: int = 60, min_periods: int = 20, baseline_floor: float = 0.2) -> pd.DataFrame` — 컬럼 `[code, date, *DISC_FEATURE_COLUMNS, top_material_type]`

정의:

| 피처 | 정의 |
|---|---|
| `disc_count` | 그날 그 종목 공시 수 |
| `disc_ratio` | `disc_count / baseline` — baseline 은 직전 `baseline_window` 거래일 평균(**shift(1)**), 하한 `baseline_floor` |
| `disc_is_bad` | 그날 악재 공시 비율 |
| `disc_is_contract` | 공급계약 비율 |
| `disc_is_capex` | 시설투자 비율 |
| `disc_is_equity` | 지분변동 비율 |
| `disc_correction_ratio` | 정정 공시 비율 — 재탕은 신선도가 낮다 |
| `days_since_disc` | 직전 공시일로부터 경과 거래일. 첫 공시는 `baseline_window` 로 채운다 |

`top_material_type` 은 모델에 넣지 않는다 — 근거 카드 표시용이다 (스펙 §6 의 피처 상한).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_features_disclosure.py`:

```python
import numpy as np
import pandas as pd

from swing_it.features.disclosure import (
    DISC_FEATURE_COLUMNS,
    daily_disclosures,
    disclosure_features,
)


def _calendar(days: list[str]) -> pd.Series:
    return pd.Series(pd.to_datetime(days))


def _disc(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(rcept_dt, code, material_type) 목록."""
    return pd.DataFrame(
        {
            "rcept_no": [f"r{i}" for i in range(len(rows))],
            "rcept_dt": pd.to_datetime([r[0] for r in rows]),
            "code": [r[1] for r in rows],
            "corp_name": ["회사"] * len(rows),
            "corp_cls": ["K"] * len(rows),
            "report_nm": [r[2] for r in rows],
            "material_type": [r[2] for r in rows],
            "is_correction": [False] * len(rows),
        }
    )


CAL = _calendar(["2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18"])


def test_daily_counts_group_by_code_and_date():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "시설투자"),
                  ("2026-08-14", "B", "정기보고")])
    out = daily_disclosures(disc, CAL).set_index(["code", "date"])
    assert out.loc[("A", pd.Timestamp("2026-08-14")), "disc_count"] == 2
    assert out.loc[("B", pd.Timestamp("2026-08-14")), "disc_count"] == 1


def test_holiday_disclosure_moves_to_next_trading_day():
    """토요일 접수 공시가 사라지면 안 된다 (계획 2 에서 뉴스로 같은 버그를 겪었다)."""
    disc = _disc([("2026-08-15", "A", "공급계약")])
    out = daily_disclosures(disc, CAL)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_type_ratios_sum_correctly():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "악재")])
    daily = daily_disclosures(disc, CAL)
    feats = disclosure_features(daily, CAL, min_periods=1).set_index("code")
    assert feats.loc["A", "disc_is_contract"] == 0.5
    assert feats.loc["A", "disc_is_bad"] == 0.5


def test_baseline_uses_only_the_past():
    """오늘 공시가 자기 baseline 에 들어가면 lookahead 다."""
    disc = _disc([("2026-08-13", "A", "정기보고"), ("2026-08-14", "A", "정기보고"),
                  ("2026-08-17", "A", "공급계약"), ("2026-08-17", "A", "공급계약"),
                  ("2026-08-17", "A", "공급계약")])
    daily = daily_disclosures(disc, CAL)
    feats = disclosure_features(daily, CAL, baseline_window=2, min_periods=2)
    row = feats[feats["date"] == pd.Timestamp("2026-08-17")].iloc[0]
    # 직전 2거래일 평균은 (1+1)/2 = 1.0 → 배수 3.0
    assert np.isclose(row["disc_ratio"], 3.0)


def test_days_since_disc_counts_trading_days():
    disc = _disc([("2026-08-13", "A", "정기보고"), ("2026-08-18", "A", "공급계약")])
    daily = daily_disclosures(disc, CAL)
    feats = disclosure_features(daily, CAL, min_periods=1).set_index("date")
    # 8/13 → 8/18 사이 거래일: 8/14, 8/17, 8/18 → 3 거래일 뒤
    assert feats.loc[pd.Timestamp("2026-08-18"), "days_since_disc"] == 3


def test_first_disclosure_gets_window_as_days_since():
    disc = _disc([("2026-08-14", "A", "공급계약")])
    daily = daily_disclosures(disc, CAL)
    feats = disclosure_features(daily, CAL, baseline_window=60, min_periods=1)
    assert feats.iloc[0]["days_since_disc"] == 60


def test_feature_columns_contract():
    disc = _disc([("2026-08-14", "A", "공급계약")])
    feats = disclosure_features(daily_disclosures(disc, CAL), CAL, min_periods=1)
    assert list(feats.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def test_top_material_type_is_the_most_common():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "공급계약"),
                  ("2026-08-14", "A", "정기보고")])
    feats = disclosure_features(daily_disclosures(disc, CAL), CAL, min_periods=1)
    assert feats.iloc[0]["top_material_type"] == "공급계약"


def test_empty_input_returns_empty_with_columns():
    empty = _disc([])
    feats = disclosure_features(daily_disclosures(empty, CAL), CAL)
    assert feats.empty
    assert list(feats.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_features_disclosure.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.features.disclosure'`

- [ ] **Step 3: 구현**

`src/swing_it/features/disclosure.py`:

```python
"""②층 공시 피처 — 무료·규칙 기반, LLM 불필요.

핵심 가설: 재료가 강하면 공시로 남는다. 뉴스와 달리 공시는 유형이 규격화돼
있어 텍스트 이해 없이도 재료 유형을 얻는다.

거래일 이월은 계획 2 의 ``news.calendar.align_to_trading_days`` 를 재사용한다.
휴장일 접수 공시가 사라지는 버그를 뉴스에서 이미 겪었고, 같은 문제를 두 번
풀 이유가 없다.
"""

from __future__ import annotations

import pandas as pd

from ..news.calendar import align_to_trading_days

DISC_FEATURE_COLUMNS: list[str] = [
    "disc_count",
    "disc_ratio",
    "disc_is_bad",
    "disc_is_contract",
    "disc_is_capex",
    "disc_is_equity",
    "disc_correction_ratio",
    "days_since_disc",
]

_DAILY_COLUMNS = [
    "code", "date", "disc_count", "n_bad", "n_contract", "n_capex",
    "n_equity", "n_correction", "top_material_type",
]
_OUT_COLUMNS = ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def daily_disclosures(disc: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """공시를 거래일로 이월한 뒤 ``(code, date)`` 로 집계한다."""
    if disc.empty:
        return pd.DataFrame(columns=_DAILY_COLUMNS)

    frame = disc.rename(columns={"rcept_dt": "date"}).copy()
    frame = align_to_trading_days(frame, calendar)
    if frame.empty:
        return pd.DataFrame(columns=_DAILY_COLUMNS)

    frame["is_correction"] = frame["is_correction"].astype(bool)
    grouped = frame.groupby(["code", "date"], sort=False)
    out = grouped.agg(
        disc_count=("rcept_no", "size"),
        n_bad=("material_type", lambda s: (s == "악재").sum()),
        n_contract=("material_type", lambda s: (s == "공급계약").sum()),
        n_capex=("material_type", lambda s: (s == "시설투자").sum()),
        n_equity=("material_type", lambda s: (s == "지분변동").sum()),
        n_correction=("is_correction", "sum"),
        top_material_type=("material_type", lambda s: s.value_counts().index[0]),
    ).reset_index()
    return out[_DAILY_COLUMNS].sort_values(["code", "date"]).reset_index(drop=True)


def disclosure_features(
    daily: pd.DataFrame,
    calendar: pd.Series,
    *,
    baseline_window: int = 60,
    min_periods: int = 20,
    baseline_floor: float = 0.2,
) -> pd.DataFrame:
    """공시 건수 급증 배수·유형 비율·신선도를 만든다.

    baseline 은 반드시 ``shift(1)`` 이후 계산한다. 오늘 공시를 자기 평균에
    포함하면 lookahead 이고, 배수가 구조적으로 축소된다.
    """
    if daily.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    trading = pd.Index(pd.to_datetime(pd.Series(calendar)).unique()).sort_values()
    codes = pd.Index(sorted(daily["code"].unique()), name="code")
    grid = pd.MultiIndex.from_product([codes, trading], names=["code", "date"]).to_frame(
        index=False
    )
    dense = grid.merge(daily[["code", "date", "disc_count"]], on=["code", "date"], how="left")
    dense["disc_count"] = dense["disc_count"].fillna(0.0)
    dense = dense.sort_values(["code", "date"])

    grouped = dense.groupby("code", sort=False)
    dense["baseline"] = grouped["disc_count"].transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=min_periods).mean()
    )
    dense["baseline"] = dense["baseline"].fillna(baseline_floor).clip(lower=baseline_floor)

    # 직전 공시로부터 경과 거래일. 공시가 없던 날은 이전 값을 물고 내려간다.
    dense["_had"] = dense["disc_count"] > 0
    dense["_ordinal"] = grouped.cumcount()
    dense["_last"] = dense["_ordinal"].where(dense["_had"])
    dense["_prev"] = grouped["_last"].transform(lambda s: s.shift(1).ffill())
    dense["days_since_disc"] = (dense["_ordinal"] - dense["_prev"]).fillna(baseline_window)

    out = daily.merge(
        dense[["code", "date", "baseline", "days_since_disc"]], on=["code", "date"], how="left"
    )
    count = out["disc_count"].astype(float)
    out["disc_ratio"] = count / out["baseline"]
    out["disc_is_bad"] = out["n_bad"] / count
    out["disc_is_contract"] = out["n_contract"] / count
    out["disc_is_capex"] = out["n_capex"] / count
    out["disc_is_equity"] = out["n_equity"] / count
    out["disc_correction_ratio"] = out["n_correction"] / count
    out["days_since_disc"] = out["days_since_disc"].fillna(baseline_window)

    return out[_OUT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_features_disclosure.py -q`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add src/swing_it/features/disclosure.py tests/test_features_disclosure.py
git commit -m "feat(features): ②층 공시 피처 — 건수 급증·유형 비율·신선도"
```

---

### Task 5: 수집 CLI

**Files:**
- Create: `src/swing_it/cli_dart.py`
- Modify: `pyproject.toml` — `[project.scripts]` 에 `swing-dart` 추가
- Test: `tests/test_cli_dart.py`

**Interfaces:**
- Consumes: `DartClient`, `api_keys`, `rows_to_frame`, `upsert_disclosures`, `init_dart_db`, `collected_range`
- Produces:
  - `quarter_ranges(start: str, end: str) -> list[tuple[str, str]]` — `corp_code` 없는 조회는 3개월 제한이므로 구간을 쪼갠다. 각 구간은 90일 이하
  - `main(argv: list[str] | None = None) -> int`

**CLI 인자:** `--start`(기본 `20160901`) `--end`(기본 오늘) `--db`(기본 `data/dart.db`) `--corp-cls`(기본 `Y,K`) `--sleep`(기본 `0.05`) `--dry-run`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_dart.py`:

```python
import httpx
import pytest

from swing_it.cli_dart import main, quarter_ranges


def test_quarter_ranges_never_exceed_ninety_days():
    ranges = quarter_ranges("20240101", "20261231")
    import pandas as pd

    for bgn, end in ranges:
        span = (pd.Timestamp(end) - pd.Timestamp(bgn)).days
        assert 0 <= span <= 90


def test_quarter_ranges_cover_the_whole_span_without_gaps():
    import pandas as pd

    ranges = quarter_ranges("20240101", "20241231")
    assert ranges[0][0] == "20240101"
    assert ranges[-1][1] == "20241231"
    for (_, prev_end), (next_bgn, _) in zip(ranges, ranges[1:]):
        assert pd.Timestamp(next_bgn) == pd.Timestamp(prev_end) + pd.Timedelta(days=1)


def test_quarter_ranges_single_short_span():
    assert quarter_ranges("20260801", "20260814") == [("20260801", "20260814")]


def test_cli_collects_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "k1")
    item = {
        "corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
        "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
        "rcept_no": "20260814000123", "flr_nm": "에프알텍",
        "rcept_dt": "20260814", "rm": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "000", "message": "정상", "page_no": 1, "page_count": 100,
            "total_count": 1, "total_page": 1, "list": [item],
        })

    db = tmp_path / "dart.db"
    rc = main([
        "--start", "20260801", "--end", "20260814", "--db", str(db),
        "--corp-cls", "K", "--sleep", "0",
    ], client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert rc == 0

    from swing_it.dart.store import init_dart_db, load_disclosures

    got = load_disclosures(init_dart_db(db), "20260801", "20260831")
    assert len(got) == 1
    assert got.iloc[0]["material_type"] == "공급계약"


def test_cli_dry_run_makes_no_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "k1")
    db = tmp_path / "dart.db"
    rc = main(["--start", "20260801", "--end", "20260814", "--db", str(db), "--dry-run"])
    assert rc == 0
    assert not db.exists()


def test_cli_without_key_fails_clearly(tmp_path, monkeypatch):
    for name in ("DART_API_KEY", "DART_API_KEY_2", "DART_API_KEY_3"):
        monkeypatch.delenv(name, raising=False)
    rc = main(["--start", "20260801", "--end", "20260814", "--db", str(tmp_path / "d.db")])
    assert rc == 1
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_cli_dart.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.cli_dart'`

- [ ] **Step 3: `pyproject.toml` 수정**

`[project.scripts]` 를 아래로 바꾼다:

```toml
[project.scripts]
swing-baseline = "swing_it.cli:main"
swing-news = "swing_it.cli_news:main"
swing-score = "swing_it.score.cli:main"
swing-layers = "swing_it.cli_layers:main"
swing-today = "swing_it.cli_today:main"
swing-dart = "swing_it.cli_dart:main"
```

Run: `uv sync --extra dev`

- [ ] **Step 4: 구현**

`src/swing_it/cli_dart.py`:

```python
"""`swing-dart` — OpenDART 공시를 로컬 sqlite 에 수집한다.

``corp_code`` 없는 공시검색은 기간이 3개월로 제한되므로 구간을 쪼개 돈다.
하루 1,000~1,500건, ``page_count=100`` 이므로 10년치가 약 30,000 콜이다.
인증키 3개면 하루 안에 끝난다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

from .adapters.dart import DartClient, DartError, api_keys
from .dart.store import collected_range, init_dart_db, rows_to_frame, upsert_disclosures

_MAX_SPAN_DAYS = 90


def quarter_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """3개월 제한에 맞춰 [start, end] 를 90일 이하 구간으로 쪼갠다."""
    begin = pd.Timestamp(start)
    finish = pd.Timestamp(end)
    ranges: list[tuple[str, str]] = []
    cursor = begin
    while cursor <= finish:
        stop = min(cursor + pd.Timedelta(days=_MAX_SPAN_DAYS), finish)
        ranges.append((cursor.strftime("%Y%m%d"), stop.strftime("%Y%m%d")))
        cursor = stop + pd.Timedelta(days=1)
    return ranges


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenDART 공시 수집")
    parser.add_argument("--start", default="20160901")
    parser.add_argument("--end", default=None, help="생략 시 오늘")
    parser.add_argument("--db", default="data/dart.db")
    parser.add_argument("--corp-cls", default="Y,K", help="쉼표 구분. Y=유가 K=코스닥")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true",
                        help="구간만 출력하고 호출하지 않는다")
    args = parser.parse_args(argv)

    end = args.end or pd.Timestamp.today().strftime("%Y%m%d")
    ranges = quarter_ranges(args.start, end)
    classes = [c.strip() for c in args.corp_cls.split(",") if c.strip()]
    print(f"구간 {len(ranges)}개 × 법인구분 {len(classes)}개 = {len(ranges)*len(classes)} 회 조회",
          file=sys.stderr)

    if args.dry_run:
        for bgn, stop in ranges:
            print(f"  {bgn} ~ {stop}", file=sys.stderr)
        return 0

    try:
        keys = api_keys()
    except DartError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    dart = DartClient(keys, client=client, sleep=args.sleep)
    conn = init_dart_db(args.db)
    total = 0
    try:
        for bgn, stop in ranges:
            for cls in classes:
                rows = dart.search_all(bgn_de=bgn, end_de=stop, corp_cls=cls)
                n = upsert_disclosures(conn, rows_to_frame(rows))
                total += n
                print(f"  {bgn}~{stop} {cls}: 수신 {len(rows):,} · 적재 {n:,}", file=sys.stderr)
    except DartError as exc:
        print(f"\n중단: {exc}", file=sys.stderr)
        print(f"여기까지 {total:,}건 적재됨. 같은 명령을 다시 돌리면 이어받는다.",
              file=sys.stderr)
        return 1
    finally:
        span = collected_range(conn)
        conn.close()

    print(f"\n적재 {total:,}건 · 범위 {span}", file=sys.stderr)
    print(f"저장: {Path(args.db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_cli_dart.py -q`
Expected: 6 passed

- [ ] **Step 6: 실 API 스모크 (짧게)**

```bash
export $(grep -v '^#' ../quant-airflow/.env | grep DART | xargs)
uv run swing-dart --start 20260801 --end 20260814 --db data/dart.db
```
Expected: 수천 건 적재. **여기서 멈추고 결과를 확인한 뒤** 전 기간을 돌린다.

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml src/swing_it/cli_dart.py tests/test_cli_dart.py
git commit -m "feat(cli): swing-dart — 3개월 구간 분할 공시 수집"
```

---

### Task 6: ②층 통합과 판정

**Files:**
- Modify: `src/swing_it/layers.py` — 공시 층을 층 정의에 추가
- Create: `src/swing_it/cli_dart_layers.py`
- Test: `tests/test_layers_disclosure.py`

**Interfaces:**
- Consumes: `build_layered_dataset`(계획 4), `disclosure_features`, `compare_layers`(계획 4)
- Produces:
  - `layers.LAYER_NAMES` 에 `"supply_disc"` 추가 — 순서는 `["supply", "supply_disc", "supply_disc_news", "supply_disc_news_llm"]` 이 아니라, **이번 계획 범위에서는 `["supply", "supply_disc"]` 두 층만** 쓴다. 뉴스·LLM 층은 계획 4 가 이미 정의했고 데이터가 없으므로 건드리지 않는다
  - `build_disclosure_dataset(prices, supply_demand, disc_features, **kwargs) -> pd.DataFrame` — 계획 4 `build_layered_dataset` 에 `news=disc_features, news_columns=DISC_FEATURE_COLUMNS` 로 위임한다. **계획 4 의 함수를 고치지 않는다** — 뉴스 자리에 공시를 넣는 얇은 래퍼다
  - `main(argv) -> int` — `swing-dart-layers`

**왜 래퍼인가.** 계획 4 의 `build_layered_dataset(news=, news_columns=)` 은 이미 "이벤트 행에 (code,date) 프레임을 붙이고 결측 행을 떨어뜨린다"는 일반 동작이다. 공시도 같은 모양이라 새 경로를 만들 이유가 없다. 이름만 뉴스일 뿐이고, 그 사실을 래퍼 독스트링에 적는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_layers_disclosure.py`:

```python
import numpy as np
import pandas as pd

from swing_it.dataset import FEATURE_COLUMNS
from swing_it.features.disclosure import DISC_FEATURE_COLUMNS
from swing_it.layers_disclosure import build_disclosure_dataset


def _prices(codes, n, spike_at):
    rows = []
    for code in codes:
        for i in range(n):
            tv = 6000.0 * (5.0 if i == spike_at else 1.0)
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "open": 100.0 + i, "close": 100.0 + i, "trade_value": tv})
    return pd.DataFrame(rows)


def _supply(codes, n):
    rows = []
    for code in codes:
        for i in range(n):
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "individual": -100, "foreign_": 60, "institution": 40,
                         "penfnd_etc": 10, "acc_trde_qty": 1000})
    return pd.DataFrame(rows)


def _disc_features(codes, n):
    rows = []
    for code in codes:
        for i in range(n):
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "disc_count": 1.0, "disc_ratio": 2.0, "disc_is_bad": 0.0,
                         "disc_is_contract": 1.0, "disc_is_capex": 0.0,
                         "disc_is_equity": 0.0, "disc_correction_ratio": 0.0,
                         "days_since_disc": 5.0})
    return pd.DataFrame(rows)


def test_dataset_has_supply_and_disclosure_columns():
    ds = build_disclosure_dataset(
        _prices(["A", "B"], 80, 25), _supply(["A", "B"], 80),
        _disc_features(["A", "B"], 80), hold_start=10, hold_end=20,
    )
    expected = ["code", "date", *FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS,
                "fwd_return", "excess_return"]
    assert list(ds.columns) == expected


def test_rows_without_disclosure_are_dropped():
    """세 층이 같은 표본을 봐야 비교가 성립한다 (계획 4 결측 정책)."""
    ds = build_disclosure_dataset(
        _prices(["A", "B"], 80, 25), _supply(["A", "B"], 80),
        _disc_features(["A"], 80), hold_start=10, hold_end=20,
    )
    assert set(ds["code"]) <= {"A"}


def test_no_column_collision_with_plan1_features():
    overlap = set(FEATURE_COLUMNS) & set(DISC_FEATURE_COLUMNS)
    assert overlap == set()


def test_feature_budget_within_spec_limit():
    """스펙 §6: 피처 10~20개."""
    assert 10 <= len(FEATURE_COLUMNS) + len(DISC_FEATURE_COLUMNS) <= 20
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_layers_disclosure.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'swing_it.layers_disclosure'`

- [ ] **Step 3: 구현**

`src/swing_it/layers_disclosure.py`:

```python
"""공시 층(②) 데이터셋 — 계획 4 하네스에 얇게 위임한다.

계획 4 의 ``build_layered_dataset(news=, news_columns=)`` 은 사실 "이벤트 행에
``(code, date)`` 프레임을 붙이고 결측 행을 떨어뜨린다"는 일반 동작이다.
공시도 모양이 같으므로 새 경로를 만들지 않고 그 자리에 넣는다.
인자 이름만 뉴스일 뿐이다.

계획 4 의 함수는 고치지 않는다 — 계획 4 테스트가 계속 통과해야 한다.
"""

from __future__ import annotations

import pandas as pd

from .features.disclosure import DISC_FEATURE_COLUMNS
from .layers import build_layered_dataset

LAYER_NAMES: list[str] = ["supply", "supply_disc"]
LAYER_LABELS: dict[str, str] = {
    "supply": "① 수급 단독",
    "supply_disc": "② + 공시(DART)",
}


def build_disclosure_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    disc_features: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """①층 피처 + 공시 피처 + 라벨을 한 표에 담는다."""
    return build_layered_dataset(
        prices,
        supply_demand,
        news=disc_features,
        news_columns=DISC_FEATURE_COLUMNS,
        **kwargs,
    )


def layer_features_disc() -> dict[str, list[str]]:
    """층 이름 → 그 층이 쓰는 피처 컬럼(누적)."""
    from .dataset import FEATURE_COLUMNS

    return {
        "supply": list(FEATURE_COLUMNS),
        "supply_disc": [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS],
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_layers_disclosure.py -q`
Expected: 4 passed

- [ ] **Step 5: 전체 테스트와 lint**

Run: `uv run --extra dev pytest -q && uv run --extra dev ruff check src tests`
Expected: 전부 통과. 기존 테스트가 하나도 깨지지 않아야 한다.

- [ ] **Step 6: 커밋**

```bash
git add src/swing_it/layers_disclosure.py tests/test_layers_disclosure.py
git commit -m "feat(layers): ②층 공시 데이터셋 — 계획 4 하네스에 위임"
```

---

## 이 계획이 끝나면 답할 수 있는 것

> **DART 공시가 수급 위에 D+20~40 초과수익 예측력을 더하는가?**

판정은 계획 4 의 `layer_verdict` 를 그대로 쓴다 — 전체 널 대비 / 모든 폴드 널 대비 / 손 안 댄 최종창 / 플라시보, 그리고 하방 회피(`p10`, `tail_capture_rate`)와 선별도(`selection_ratio`).

**어느 쪽이 나오든 기록하고 끝낸다.**

- **②가 ①을 이기면** — "재료가 D+20~40 을 설명한다"가 무료로 증명된다. 뉴스 구매(뉴스토어 견적)가 근거 있는 지출이 되고, 공시가 없는 재료(테마·정책·해외수요)를 뉴스가 메울 수 있는지가 다음 질문이 된다.
- **②가 ①을 못 이기면** — 재료 축 자체가 이 지평에서 약하다는 신호다. 뉴스에 돈을 쓰기 전에 그 사실을 알게 된다. 이것이 이 계획의 존재 이유다.

## 범위 밖

- **공시 원문 수집과 LLM 피처 (③층).** 원문은 무료로 받을 수 있고(`document.xml`, ZIP 안 XML, CSS 블록 제거 필요) 계약금액·매출액 대비·계약기간이 텍스트에 있다. ②층 결과를 보고 계획 6 으로 다룬다. **여기서 미리 만들지 않는다** — ②가 실패하면 필요 없다.
- 뉴스 수집(계획 2 나머지), 뉴스 LLM(계획 3 Task 7).
- 포지션 예산 시뮬레이션.
