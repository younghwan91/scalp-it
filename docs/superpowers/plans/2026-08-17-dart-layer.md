# 계획 5: DART 공시 층 (②층) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DART 전자공시를 무료·완전·point-in-time 으로 수집해 **공시를 이벤트로 삼는** 데이터셋을 만들고, ①수급 대비 ②공시 층의 기여를 판정한다. 절대 임계와 십분위 단조성까지 붙여 "확실한 것만" 이 성립하는지 함께 본다.

**Architecture:** `adapters/dart.py` 가 OpenDART REST 를 감싸고, `dart/` 가 순수 함수로 유형 분류·이벤트 구성·피처를 만든다. 저장은 로컬 sqlite(`data/dart.db`). 계획 1 의 `events.extract_events`(거래대금 급증)를 **대체하는 새 이벤트 빌더**를 두고, 거래대금 급증은 피처로 내려간다.

> **스펙 rev2 기준이다.** 이벤트가 "거래대금 급증"에서 "공시 접수"로 뒤집혔다.
> 이 계획의 첫 판(rev1 기준)은 공시를 *피처*로만 붙였는데, 그 설계는 폐기됐다.

**Tech Stack:** Python ≥3.10, httpx(이미 의존성), pandas, sqlite3, pytest

**Spec:** `docs/superpowers/specs/2026-08-17-scalp-it-design.md`

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
| `src/scalp_it/adapters/dart.py` | OpenDART HTTP. **DART 호출은 이 파일에만** |
| `src/scalp_it/dart/__init__.py` | 패키지 |
| `src/scalp_it/dart/classify.py` | `report_nm` → 재료 유형 (순수 함수) |
| `src/scalp_it/dart/events.py` | **공시 → 이벤트** (계획 1 거래대금 급증을 대체) |
| `src/scalp_it/dart/store.py` | `data/dart.db` 스키마·적재·조회 |
| `src/scalp_it/features/disclosure.py` | ②층 피처 (순수 함수) |
| `src/scalp_it/dart/dataset.py` | 이벤트 + 피처 + 라벨 조립 |
| `src/scalp_it/validate/monotonic.py` | 십분위 단조성 |
| `src/scalp_it/report/threshold.py` | 절대 임계 (동시 보유 역산) |
| `src/scalp_it/cli_dart.py` | `scalp-dart` — 수집 CLI |
| `src/scalp_it/cli_dart_layers.py` | `scalp-dart-layers` — 판정 CLI |
| `tests/` | 각 모듈 대응 |

---

### Task 1: OpenDART 어댑터 — 공시검색

**Files:**
- Create: `src/scalp_it/adapters/dart.py`
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

from scalp_it.adapters.dart import DartClient, DartError, api_keys


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
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.adapters.dart'`

- [ ] **Step 3: 구현**

`src/scalp_it/adapters/dart.py`:

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
git add src/scalp_it/adapters/dart.py tests/test_dart_adapter.py
git commit -m "feat(adapters): OpenDART 공시검색 어댑터 — 키 로테이션·한도 폴백"
```

---

### Task 2: 공시 유형 분류

**Files:**
- Create: `src/scalp_it/dart/__init__.py`
- Create: `src/scalp_it/dart/classify.py`
- Test: `tests/test_dart_classify.py`

**Interfaces:**
- Consumes: `report_nm` 문자열
- Produces:
  - `MATERIAL_TYPES: list[str]` = `["공급계약", "시설투자", "지분변동", "자사주", "실적변동", "기술_인허가", "구조조정", "악재", "정기보고", "기타"]`
  - `classify_report(report_nm: str) -> str` — 위 중 하나
  - `is_correction(report_nm: str) -> bool` — `[기재정정]`·`[첨부정정]`·`[첨부추가]` 접두 여부
  - `classify_frame(frame: pd.DataFrame) -> pd.DataFrame` — `report_nm` 컬럼을 읽어 `material_type`·`is_correction` 을 붙인다

**분류 규칙 (사전등록).** 성과를 보고 바꾸지 않는다. 규칙은 위에서 아래로 처음 맞는 것을 쓴다.

실측(2026-05~07, 상장종목 공시 하루 462건)으로 커버리지를 확인하고 정한 목록이다.
초안 규칙에서는 '기타'가 66% 였고, 그 안에 **임원·주요주주 소유상황보고서(하루 60건,
전체 최다)**, 연결재무제표기준 영업(잠정)실적, 배당·주식소각, 투자판단관련주요경영사항이
들어 있었다. 아래 규칙은 그것들을 끌어올린 판이다.

| 유형 | 매칭 키워드 |
|---|---|
| `악재` | 감사의견, 의견거절, 부적정, 자본잠식, 상장폐지, 거래정지, 회생절차, 파산, 부도, 횡령, 배임, 소송, 손실발생, 실질심사, 미달사실발생, 영업정지 |
| `내부자` | 임원ㆍ주요주주, 임원·주요주주, 특정증권등소유상황, 특정증권등거래계획 |
| `공급계약` | 공급계약, 수주, 판매계약 |
| `시설투자` | 신규시설투자, 유형자산, 공장, 증설 |
| `주주환원` | 자기주식, 주식소각, 배당 |
| `자본조달` | 유상증자, 무상증자, 전환사채, 신주인수권, 교환사채, 전환가액, 증권신고서, 일괄신고 |
| `지분변동` | 대량보유, 최대주주, 주식매수선택권 |
| `실적` | 매출액또는손익구조, 영업(잠정)실적, 연결재무제표기준영업, 결산실적, 매출액미달 |
| `기술_인허가` | 특허, 임상, 품목허가, 승인, 인증 |
| `구조조정` | 합병, 분할, 양수, 양도, 채무보증 |
| `재량공시` | 투자판단관련주요경영사항, 기업설명회, 기타시장안내 |
| `정기보고` | 사업보고서, 반기보고서, 분기보고서, 감사보고서, 검토, 투자설명서 |
| `행정` | 주주총회, 주주명부, 의결권대리행사, 기준일, 증권발행실적, 증권발행결과 |
| `기타` | 나머지 |

**순서가 의미를 가진다:**

- `악재` 가 맨 위인 이유는 방향을 결정하기 때문이다. "소송등의판결"은 구조조정 키워드와
  겹치지만 악재로 봐야 한다.
- `내부자` 가 그다음인 이유는 "임원ㆍ주요주주특정증권등소유상황보고서"가 `지분변동` 의
  '최대주주'와 겹칠 수 있어서다. 내부자 거래는 별도 축으로 봐야 할 만큼 건수가 많다.
- `주주환원` 이 `자본조달` 보다 위인 이유는 "자기주식취득 신탁계약"류가 증권 관련
  키워드와 겹칠 수 있어서다.

**`악재` 는 이벤트에서 배제된다** (스펙 §5, 롱온리). 나머지 유형은 전부 피처로 남는다.
`정기보고`·`행정` 도 배제하지 않는다 — "행정 공시는 무의미하다"는 검증되지 않은 가정이고,
유형을 피처로 넣으면 모델이 가려낸다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dart_classify.py`:

```python
import pandas as pd
import pytest

from scalp_it.dart.classify import (
    MATERIAL_TYPES,
    classify_frame,
    classify_report,
    is_correction,
)


@pytest.mark.parametrize(
    ("report_nm", "expected"),
    [
        # 실측 상위 유형 — 2026-05~07 표본에서 실제로 나온 제목들이다
        ("임원ㆍ주요주주특정증권등소유상황보고서", "내부자"),
        ("임원ㆍ주요주주특정증권등거래계획보고서", "내부자"),
        ("단일판매ㆍ공급계약체결", "공급계약"),
        ("신규시설투자등", "시설투자"),
        ("자기주식취득결정", "주주환원"),
        ("자기주식처분결정", "주주환원"),
        ("주식소각결정", "주주환원"),
        ("현금ㆍ현물배당결정", "주주환원"),
        ("유상증자결정", "자본조달"),
        ("무상증자결정", "자본조달"),
        ("전환가액의조정", "자본조달"),
        ("증권신고서", "자본조달"),
        ("주식등의대량보유상황보고서", "지분변동"),
        ("주식매수선택권부여에관한신고", "지분변동"),
        ("매출액또는손익구조30%이상변동", "실적"),
        ("연결재무제표기준영업(잠정)실적", "실적"),
        ("결산실적공시예고", "실적"),
        ("특허권취득", "기술_인허가"),
        ("회사합병결정", "구조조정"),
        ("타법인주식및출자증권양수결정", "구조조정"),
        ("타인에대한채무보증결정", "구조조정"),
        ("투자판단관련주요경영사항", "재량공시"),
        ("기업설명회", "재량공시"),
        ("기타시장안내", "재량공시"),
        ("반기보고서", "정기보고"),
        ("사업보고서", "정기보고"),
        ("투자설명서", "정기보고"),
        ("주주총회소집결의", "행정"),
        ("주주명부폐쇄기간또는기준일설정", "행정"),
        ("의결권대리행사권유참고서류", "행정"),
        ("증권발행실적보고서", "행정"),
        ("반기검토의견부적정,의견거절또는완전자본잠식사실발생", "악재"),
        ("주권매매거래정지", "악재"),
        ("파생상품거래손실발생", "악재"),
    ],
)
def test_classify_report(report_nm, expected):
    assert classify_report(report_nm) == expected


def test_insider_wins_over_major_shareholder_keyword():
    """'임원ㆍ주요주주…'는 지분변동의 '최대주주'와 겹칠 수 있다. 내부자가 우선한다."""
    assert classify_report("임원ㆍ주요주주특정증권등소유상황보고서") == "내부자"


def test_coverage_on_real_titles_is_high():
    """실측 상위 제목 중 '기타'로 빠지는 비율이 낮아야 한다.

    초안 규칙에서는 '기타'가 66% 였다. 유형이 피처인 설계에서 3분의 2가
    한 값에 몰리면 그 피처는 정보를 거의 담지 못한다.
    """
    titles = [
        "임원ㆍ주요주주특정증권등소유상황보고서", "연결재무제표기준영업(잠정)실적",
        "투자설명서", "기업설명회", "증권발행실적보고서", "일괄신고추가서류",
        "주주총회소집결의", "결산실적공시예고", "주주명부폐쇄기간또는기준일설정",
        "임시주주총회결과", "주주총회소집공고", "현금ㆍ현물배당결정", "주식소각결정",
        "투자판단관련주요경영사항", "타인에대한채무보증결정", "증권발행결과",
        "증권신고서", "주식매수선택권부여에관한신고", "타법인주식및출자증권취득결정",
        "전환가액의조정", "기타시장안내", "단일판매ㆍ공급계약체결", "반기보고서",
    ]
    etc = sum(1 for t in titles if classify_report(t) == "기타")
    assert etc / len(titles) < 0.15


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
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.dart'`

- [ ] **Step 3: 구현**

`src/scalp_it/dart/__init__.py`:

```python
"""DART 공시 처리 — 수집은 adapters/dart.py, 여기는 순수 함수."""
```

`src/scalp_it/dart/classify.py`:

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
    "악재", "내부자", "공급계약", "시설투자", "주주환원", "자본조달",
    "지분변동", "실적", "기술_인허가", "구조조정", "재량공시", "정기보고",
    "행정", "기타",
]

#: 이벤트에서 배제하는 유형. 롱온리라 공매도로 못 먹는다 (스펙 §5).
EXCLUDED_TYPES: frozenset[str] = frozenset({"악재"})

# 위에서 아래로 처음 맞는 것을 쓴다. 순서에 의미가 있다:
#  - '악재'가 맨 위 — 방향을 결정한다. "소송등의판결"은 구조조정과 겹치지만 악재다.
#  - '내부자'가 그다음 — "임원ㆍ주요주주…"가 지분변동의 '최대주주'와 겹친다.
#    실측 하루 60건으로 전체 최다라 별도 축으로 봐야 한다.
#  - '주주환원'이 '자본조달'보다 위 — "자기주식취득 신탁계약"류가 증권 키워드와 겹친다.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("악재", ("감사의견", "의견거절", "부적정", "자본잠식", "상장폐지", "거래정지",
              "회생절차", "파산", "부도", "횡령", "배임", "소송", "손실발생",
              "실질심사", "미달사실발생", "영업정지")),
    ("내부자", ("임원ㆍ주요주주", "임원·주요주주", "특정증권등소유상황",
                "특정증권등거래계획")),
    ("공급계약", ("공급계약", "수주", "판매계약")),
    ("시설투자", ("신규시설투자", "유형자산", "공장", "증설")),
    ("주주환원", ("자기주식", "주식소각", "배당")),
    ("자본조달", ("유상증자", "무상증자", "전환사채", "신주인수권", "교환사채",
                 "전환가액", "증권신고서", "일괄신고")),
    ("지분변동", ("대량보유", "최대주주", "주식매수선택권")),
    ("실적", ("매출액또는손익구조", "영업(잠정)실적", "연결재무제표기준영업",
              "결산실적", "매출액미달")),
    ("기술_인허가", ("특허", "임상", "품목허가", "승인", "인증")),
    ("구조조정", ("합병", "분할", "양수", "양도", "채무보증")),
    ("재량공시", ("투자판단관련주요경영사항", "기업설명회", "기타시장안내")),
    ("정기보고", ("사업보고서", "반기보고서", "분기보고서", "감사보고서", "검토",
                 "투자설명서")),
    ("행정", ("주주총회", "주주명부", "의결권대리행사", "기준일",
              "증권발행실적", "증권발행결과")),
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
git add src/scalp_it/dart tests/test_dart_classify.py
git commit -m "feat(dart): 공시 제목 → 재료 유형 규칙 분류 (사전등록)"
```

---

### Task 3: 공시 저장소

**Files:**
- Create: `src/scalp_it/dart/store.py`
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

from scalp_it.dart.store import (
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
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.dart.store'`

- [ ] **Step 3: 구현**

`src/scalp_it/dart/store.py`:

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
git add src/scalp_it/dart/store.py tests/test_dart_store.py
git commit -m "feat(dart): 공시 로컬 저장소 — 상장 종목만, rcept_no 멱등 적재"
```

---

### Task 4: 수집 CLI

**Files:**
- Create: `src/scalp_it/cli_dart.py`
- Modify: `pyproject.toml` — `[project.scripts]` 에 `scalp-dart` 추가
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

from scalp_it.cli_dart import main, quarter_ranges


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

    from scalp_it.dart.store import init_dart_db, load_disclosures

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
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.cli_dart'`

- [ ] **Step 3: `pyproject.toml` 수정**

`[project.scripts]` 를 아래로 바꾼다:

```toml
[project.scripts]
scalp-baseline = "scalp_it.cli:main"
scalp-news = "scalp_it.cli_news:main"
scalp-score = "scalp_it.score.cli:main"
scalp-layers = "scalp_it.cli_layers:main"
scalp-today = "scalp_it.cli_today:main"
scalp-dart = "scalp_it.cli_dart:main"
```

Run: `uv sync --extra dev`

- [ ] **Step 4: 구현**

`src/scalp_it/cli_dart.py`:

```python
"""`scalp-dart` — OpenDART 공시를 로컬 sqlite 에 수집한다.

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
uv run scalp-dart --start 20260801 --end 20260814 --db data/dart.db
```
Expected: 수천 건 적재. **여기서 멈추고 결과를 확인한 뒤** 전 기간을 돌린다.

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml src/scalp_it/cli_dart.py tests/test_cli_dart.py
git commit -m "feat(cli): scalp-dart — 3개월 구간 분할 공시 수집"
```

---

### Task 5: 이벤트 빌더 — 공시가 이벤트다

**Files:**
- Create: `src/scalp_it/dart/events.py`
- Test: `tests/test_dart_events.py`

**Interfaces:**
- Consumes: `load_disclosures`(Task 3), 계획 1 `events.add_liquidity_stats`, 계획 2 `news.calendar.align_to_trading_days`, `classify.EXCLUDED_TYPES`
- Produces:
  - `EVENT_COLUMNS: list[str]` = `["code", "date", "adv", "tv_ratio"]`
  - `build_disclosure_events(disc: pd.DataFrame, prices: pd.DataFrame, *, adv_floor: float = 5000.0, adv_window: int = 20) -> pd.DataFrame`

**이 함수가 계획 1 의 `extract_events`(거래대금 급증)를 대체한다.** 거래대금 급증은
폐기되지 않고 `tv_ratio` 로 남아 피처가 된다 (스펙 rev2 §3).

처리 순서 — 순서가 결과를 바꾸므로 고정한다:

1. **악재 배제** — `material_type in EXCLUDED_TYPES` 인 행을 버린다. 롱온리라 못 먹는다.
2. **거래일 이월** — 휴장일 접수 공시를 다음 거래일로. 앞으로만 (뒤로 당기면 lookahead).
   달력은 `prices["date"]` 에서 가져온다.
3. **`(code, date)` 중복 제거** — 같은 종목에 하루 여러 건이 나도 이벤트는 하나다.
   건수는 피처(Task 6)에서 센다.
4. **유동성 하한** — `adv >= adv_floor`. `adv` 는 직전 `adv_window` 거래일 평균이며
   **`shift(1)` 이후** 계산된다(계획 1 `add_liquidity_stats` 재사용).
5. **가격 없는 (code, date) 제거** — 거래정지·상장 전이면 진입할 수 없다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dart_events.py`:

```python
import pandas as pd
import pytest

from scalp_it.dart.events import EVENT_COLUMNS, build_disclosure_events


def _prices(codes, n, adv=9000.0, spike_at=None):
    rows = []
    for code in codes:
        for i in range(n):
            tv = adv * (5.0 if i == spike_at else 1.0)
            rows.append({
                "code": code,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "open": 100.0, "close": 100.0, "trade_value": tv,
            })
    return pd.DataFrame(rows)


def _disc(rows):
    """(rcept_dt, code, material_type)."""
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(rows))],
        "rcept_dt": pd.to_datetime([r[0] for r in rows]),
        "code": [r[1] for r in rows],
        "corp_name": ["회사"] * len(rows),
        "corp_cls": ["K"] * len(rows),
        "report_nm": [r[2] for r in rows],
        "material_type": [r[2] for r in rows],
        "is_correction": [False] * len(rows),
    })


def test_event_columns_contract():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert list(out.columns) == EVENT_COLUMNS


def test_disclosure_becomes_an_event():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert len(out) == 1
    assert out.iloc[0]["code"] == "A"
    assert out.iloc[0]["date"] == pd.Timestamp("2026-01-30")


def test_bad_news_is_excluded():
    """롱온리라 악재는 후보가 될 수 없다."""
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "악재")])
    assert build_disclosure_events(disc, prices).empty


def test_same_code_same_day_collapses_to_one_event():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약"), ("2026-01-30", "A", "시설투자")])
    assert len(build_disclosure_events(disc, prices)) == 1


def test_illiquid_stock_is_dropped():
    """PEAD 검증에서 유동성 하한 없이는 소형주 꼬리가 시그널을 음수로 뒤집었다."""
    prices = _prices(["A"], 40, adv=100.0)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    assert build_disclosure_events(disc, prices, adv_floor=5000.0).empty


def test_date_without_price_row_is_dropped():
    """거래정지·상장 전이면 진입할 수 없다."""
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "B", "공급계약")])  # B 는 가격이 없다
    assert build_disclosure_events(disc, prices).empty


def test_tv_ratio_is_carried_as_a_feature_not_a_filter():
    """거래대금 급증은 rev2 에서 이벤트가 아니라 피처다. 급증이 없어도 이벤트다."""
    prices = _prices(["A"], 40)  # 급증 없음
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert len(out) == 1
    assert out.iloc[0]["tv_ratio"] == pytest.approx(1.0)


def test_spike_is_reflected_in_tv_ratio():
    prices = _prices(["A"], 40, spike_at=35)
    spike_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=35)
    disc = _disc([(str(spike_date.date()), "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert out.iloc[0]["tv_ratio"] > 4.0


def test_holiday_disclosure_moves_to_next_trading_day():
    """가격 달력에 없는 날 접수된 공시가 사라지면 안 된다."""
    prices = pd.DataFrame({
        "code": ["A"] * 3,
        "date": pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-17"]),
        "open": [100.0] * 3, "close": [100.0] * 3, "trade_value": [9000.0] * 3,
    })
    disc = _disc([("2026-08-15", "A", "공급계약")])   # 토요일
    out = build_disclosure_events(disc, prices, adv_window=1, adv_floor=0.0)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_empty_disclosures_returns_empty_with_columns():
    out = build_disclosure_events(_disc([]), _prices(["A"], 40))
    assert out.empty
    assert list(out.columns) == EVENT_COLUMNS
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_dart_events.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalp_it.dart.events'`

- [ ] **Step 3: 구현**

`src/scalp_it/dart/events.py`:

```python
"""이벤트 정의 — 공시가 난 (종목, 날짜) 다 (스펙 rev2 §3).

계획 1 의 ``events.extract_events``(거래대금 급증)를 대체한다. 거래대금 급증은
폐기되지 않고 ``tv_ratio`` 로 남아 "시장이 이 재료에 반응했는가"를 재는 피처가 된다.

왜 뒤집었나 (스펙 §0):
  - 검증된 PEAD 가 이미 '재료 → 드리프트' 구조다.
  - 공시 접수일자는 정확한 정보 공개 시점이라 자연 실험이 깨끗하다.
  - 급증 종목의 재료를 찾는 것은 이미 오른 걸 보고 이유를 붙이는 역인과다.
"""

from __future__ import annotations

import pandas as pd

from ..events import add_liquidity_stats
from ..news.calendar import align_to_trading_days
from .classify import EXCLUDED_TYPES

EVENT_COLUMNS: list[str] = ["code", "date", "adv", "tv_ratio"]


def build_disclosure_events(
    disc: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
) -> pd.DataFrame:
    """공시를 이벤트로 바꾼다.

    순서가 결과를 바꾸므로 고정한다: 악재 배제 → 거래일 이월 → (code,date) 중복
    제거 → 유동성 하한 → 가격 없는 행 제거.

    Args:
        adv_floor: 직전 ``adv_window`` 거래일 평균 거래대금 하한. 단위 백만원.
            기본 5000 = 50억. PEAD 검증에서 이 조건 없이는 소형주 꼬리가
            시그널을 음수로 뒤집었다.
    """
    if disc.empty or prices.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    kept = disc[~disc["material_type"].isin(EXCLUDED_TYPES)].copy()
    if kept.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    kept = kept.rename(columns={"rcept_dt": "date"})
    kept = align_to_trading_days(kept, prices["date"])
    if kept.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    pairs = kept[["code", "date"]].drop_duplicates()

    stats = add_liquidity_stats(prices, window=adv_window)
    events = pairs.merge(
        stats[["code", "date", "adv", "tv_ratio"]], on=["code", "date"], how="inner"
    )
    events = events[events["adv"] >= adv_floor]
    return events[EVENT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_dart_events.py -q`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add src/scalp_it/dart/events.py tests/test_dart_events.py
git commit -m "feat(dart): 이벤트 정의를 공시로 뒤집는다 (스펙 rev2)"
```

---

### Task 6: 공시 피처와 데이터셋

**Files:**
- Create: `src/scalp_it/features/disclosure.py`
- Create: `src/scalp_it/dart/dataset.py`
- Test: `tests/test_features_disclosure.py`
- Test: `tests/test_dart_dataset.py`

**Interfaces:**
- Consumes: `load_disclosures`, `build_disclosure_events`, 계획 1 `labels.forward_return`·`excess_return`·`features.supply.supply_features`·`dataset.FEATURE_COLUMNS`
- Produces:
  - `DISC_FEATURE_COLUMNS: list[str]` = `["disc_count", "disc_ratio", "disc_correction_ratio", "days_since_disc", "is_insider", "is_contract", "is_capex", "is_payout", "is_funding", "is_earnings"]`
  - `disclosure_features(disc, calendar, *, baseline_window=60, min_periods=20, baseline_floor=0.2) -> pd.DataFrame` — 컬럼 `[code, date, *DISC_FEATURE_COLUMNS, top_material_type]`
  - `DISC_DATASET_COLUMNS: list[str]`
  - `build_dart_dataset(prices, supply_demand, disc, *, adv_floor=5000.0, adv_window=20, entry_offset=1, hold_start=20, hold_end=40) -> pd.DataFrame` — 컬럼 `[code, date, *FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS, fwd_return, excess_return, top_material_type]`

**피처 정의**

| 피처 | 정의 |
|---|---|
| `disc_count` | 그날 그 종목 공시 수 (악재 제외 후) |
| `disc_ratio` | `disc_count / baseline`. baseline 은 직전 `baseline_window` 거래일 평균(**shift(1)**), 하한 `baseline_floor` |
| `disc_correction_ratio` | 정정 공시 비율 — 재탕은 신선도가 낮다 |
| `days_since_disc` | 직전 공시일로부터 경과 거래일. 첫 공시는 `baseline_window` |
| `is_insider` … `is_earnings` | 유형별 그날 비율 (내부자 / 공급계약 / 시설투자 / 주주환원 / 자본조달 / 실적) |

`top_material_type` 은 **모델 행렬에 넣지 않는다** — 근거 카드 표시용이다.
원-핫으로 펼치면 14개 값이 스펙 §7 의 10~20개 상한을 혼자 먹는다. 대신 비율 6개로 요약한다.

**피처 예산:** 계획 1 수급 7개 + 공시 10개 = **17개**. 상한 안이다.

**계획 1 `FEATURE_COLUMNS` 와의 충돌 없음:** `tv_ratio`·`log_adv` 는 계획 1 소유이고
공시 피처 이름과 겹치지 않는다. 테스트로 고정한다.

- [ ] **Step 1: 실패하는 피처 테스트 작성**

`tests/test_features_disclosure.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.features.disclosure import DISC_FEATURE_COLUMNS, disclosure_features


CAL = pd.Series(pd.to_datetime(
    ["2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19"]))


def _disc(rows, corrections=None):
    corrections = corrections or [False] * len(rows)
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(rows))],
        "rcept_dt": pd.to_datetime([r[0] for r in rows]),
        "code": [r[1] for r in rows],
        "corp_name": ["회사"] * len(rows),
        "corp_cls": ["K"] * len(rows),
        "report_nm": [r[2] for r in rows],
        "material_type": [r[2] for r in rows],
        "is_correction": corrections,
    })


def test_columns_contract():
    out = disclosure_features(_disc([("2026-08-14", "A", "공급계약")]), CAL, min_periods=1)
    assert list(out.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def test_count_and_type_ratios():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "내부자")])
    row = disclosure_features(disc, CAL, min_periods=1).iloc[0]
    assert row["disc_count"] == 2
    assert row["is_contract"] == 0.5
    assert row["is_insider"] == 0.5
    assert row["is_capex"] == 0.0


def test_holiday_disclosure_moves_forward():
    disc = _disc([("2026-08-15", "A", "공급계약")])
    out = disclosure_features(disc, CAL, min_periods=1)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_baseline_uses_only_the_past():
    disc = _disc([("2026-08-13", "A", "행정"), ("2026-08-14", "A", "행정"),
                  ("2026-08-17", "A", "공급계약"), ("2026-08-17", "A", "공급계약"),
                  ("2026-08-17", "A", "공급계약")])
    out = disclosure_features(disc, CAL, baseline_window=2, min_periods=2)
    row = out[out["date"] == pd.Timestamp("2026-08-17")].iloc[0]
    assert np.isclose(row["disc_ratio"], 3.0)   # 직전 2일 평균 1.0 → 3배


def test_days_since_counts_trading_days():
    disc = _disc([("2026-08-13", "A", "행정"), ("2026-08-19", "A", "공급계약")])
    out = disclosure_features(disc, CAL, min_periods=1).set_index("date")
    assert out.loc[pd.Timestamp("2026-08-19"), "days_since_disc"] == 4


def test_correction_ratio():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "공급계약")],
                 corrections=[True, False])
    assert disclosure_features(disc, CAL, min_periods=1).iloc[0]["disc_correction_ratio"] == 0.5


def test_top_material_type_is_most_common_and_not_a_model_feature():
    disc = _disc([("2026-08-14", "A", "공급계약"), ("2026-08-14", "A", "공급계약"),
                  ("2026-08-14", "A", "행정")])
    out = disclosure_features(disc, CAL, min_periods=1)
    assert out.iloc[0]["top_material_type"] == "공급계약"
    assert "top_material_type" not in DISC_FEATURE_COLUMNS


def test_empty_input():
    out = disclosure_features(_disc([]), CAL)
    assert out.empty
    assert list(out.columns) == ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]
```

- [ ] **Step 2: 실패하는 데이터셋 테스트 작성**

`tests/test_dart_dataset.py`:

```python
import pandas as pd

from scalp_it.dart.dataset import DISC_DATASET_COLUMNS, build_dart_dataset
from scalp_it.dataset import FEATURE_COLUMNS
from scalp_it.features.disclosure import DISC_FEATURE_COLUMNS


def _prices(codes, n):
    rows = []
    for code in codes:
        for i in range(n):
            rows.append({"code": code,
                         "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "open": 100.0 + i, "close": 100.0 + i, "trade_value": 9000.0})
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


def _disc(pairs):
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(pairs))],
        "rcept_dt": pd.to_datetime([p[0] for p in pairs]),
        "code": [p[1] for p in pairs],
        "corp_name": ["회사"] * len(pairs),
        "corp_cls": ["K"] * len(pairs),
        "report_nm": [p[2] for p in pairs],
        "material_type": [p[2] for p in pairs],
        "is_correction": [False] * len(pairs),
    })


def test_dataset_columns_contract():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약"), ("2026-01-30", "B", "내부자")]),
        hold_start=10, hold_end=20,
    )
    assert list(ds.columns) == DISC_DATASET_COLUMNS


def test_feature_budget_is_within_spec_limit():
    """스펙 §7: 피처 10~20개."""
    assert 10 <= len(FEATURE_COLUMNS) + len(DISC_FEATURE_COLUMNS) <= 20


def test_no_column_collision_between_plan1_and_disclosure():
    assert set(FEATURE_COLUMNS).isdisjoint(DISC_FEATURE_COLUMNS)


def test_rows_are_events_only():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약")]), hold_start=10, hold_end=20,
    )
    assert set(ds["code"]) <= {"A"}


def test_labels_and_features_have_no_nan():
    ds = build_dart_dataset(
        _prices(["A", "B"], 80), _supply(["A", "B"], 80),
        _disc([("2026-01-30", "A", "공급계약"), ("2026-01-31", "B", "시설투자")]),
        hold_start=10, hold_end=20,
    )
    cols = [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS, "excess_return"]
    assert ds[cols].notna().all().all()
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_features_disclosure.py tests/test_dart_dataset.py -q`
Expected: FAIL — 두 모듈 모두 `ModuleNotFoundError`

- [ ] **Step 4: 피처 구현**

`src/scalp_it/features/disclosure.py`:

```python
"""②층 공시 피처 — 무료·규칙 기반, LLM 불필요.

거래일 이월은 계획 2 의 ``news.calendar.align_to_trading_days`` 를 재사용한다.
휴장일 접수분이 사라지는 버그를 뉴스에서 이미 겪었고, 같은 문제를 두 번 풀지 않는다.
"""

from __future__ import annotations

import pandas as pd

from ..news.calendar import align_to_trading_days

_TYPE_FLAGS: dict[str, str] = {
    "is_insider": "내부자",
    "is_contract": "공급계약",
    "is_capex": "시설투자",
    "is_payout": "주주환원",
    "is_funding": "자본조달",
    "is_earnings": "실적",
}

DISC_FEATURE_COLUMNS: list[str] = [
    "disc_count",
    "disc_ratio",
    "disc_correction_ratio",
    "days_since_disc",
    *_TYPE_FLAGS,
]

_OUT_COLUMNS = ["code", "date", *DISC_FEATURE_COLUMNS, "top_material_type"]


def disclosure_features(
    disc: pd.DataFrame,
    calendar: pd.Series,
    *,
    baseline_window: int = 60,
    min_periods: int = 20,
    baseline_floor: float = 0.2,
) -> pd.DataFrame:
    """``(code, date)`` 별 공시 건수·급증 배수·유형 비율·신선도."""
    if disc.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    frame = disc.rename(columns={"rcept_dt": "date"}).copy()
    frame = align_to_trading_days(frame, calendar)
    if frame.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)
    frame["is_correction"] = frame["is_correction"].astype(bool)

    aggs = {
        "disc_count": ("rcept_no", "size"),
        "n_correction": ("is_correction", "sum"),
        "top_material_type": ("material_type", lambda s: s.value_counts().index[0]),
    }
    daily = frame.groupby(["code", "date"], sort=False).agg(**aggs).reset_index()
    for flag, label in _TYPE_FLAGS.items():
        hits = (
            frame.assign(_hit=(frame["material_type"] == label).astype(float))
            .groupby(["code", "date"], sort=False)["_hit"].sum()
            .reset_index(name=flag)
        )
        daily = daily.merge(hits, on=["code", "date"], how="left")

    # 전 거래일 격자로 펼쳐야 '평소'가 과대평가되지 않는다.
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

    dense["_ordinal"] = grouped.cumcount()
    dense["_last"] = dense["_ordinal"].where(dense["disc_count"] > 0)
    dense["_prev"] = grouped["_last"].transform(lambda s: s.shift(1).ffill())
    dense["days_since_disc"] = (dense["_ordinal"] - dense["_prev"]).fillna(baseline_window)

    out = daily.merge(
        dense[["code", "date", "baseline", "days_since_disc"]], on=["code", "date"], how="left"
    )
    count = out["disc_count"].astype(float)
    out["disc_ratio"] = count / out["baseline"]
    out["disc_correction_ratio"] = out["n_correction"] / count
    for flag in _TYPE_FLAGS:
        out[flag] = out[flag].fillna(0.0) / count
    out["days_since_disc"] = out["days_since_disc"].fillna(baseline_window)

    return out[_OUT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
```

- [ ] **Step 5: 데이터셋 구현**

`src/scalp_it/dart/dataset.py`:

```python
"""공시 이벤트 + 수급 피처 + 공시 피처 + 라벨을 한 표로 조립한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset import FEATURE_COLUMNS
from ..features.disclosure import DISC_FEATURE_COLUMNS, disclosure_features
from ..features.supply import supply_features
from ..labels import excess_return, forward_return
from .events import build_disclosure_events

DISC_DATASET_COLUMNS: list[str] = [
    "code", "date", *FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS,
    "fwd_return", "excess_return", "top_material_type",
]

_SCALED = {
    "smart_money_net_5d_scaled": "smart_money_net_5d",
    "indiv_net_5d_scaled": "indiv_net_5d",
    "pension_net_5d_scaled": "pension_net_5d",
}


def build_dart_dataset(
    prices: pd.DataFrame,
    supply_demand: pd.DataFrame,
    disc: pd.DataFrame,
    *,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    entry_offset: int = 1,
    hold_start: int = 20,
    hold_end: int = 40,
) -> pd.DataFrame:
    """공시 이벤트 행에 피처와 라벨을 붙인다."""
    events = build_disclosure_events(
        disc, prices, adv_floor=adv_floor, adv_window=adv_window
    )
    if events.empty:
        return pd.DataFrame(columns=DISC_DATASET_COLUMNS)

    supply = supply_features(supply_demand)
    disc_feats = disclosure_features(disc, prices["date"])
    labels = excess_return(
        forward_return(
            prices, entry_offset=entry_offset, hold_start=hold_start, hold_end=hold_end
        )
    )

    df = events.merge(supply, on=["code", "date"], how="left")
    df = df.merge(disc_feats, on=["code", "date"], how="left")
    df = df.merge(labels, on=["code", "date"], how="left")

    shares_capacity = (df["adv"] * 1e6) / df["entry_price"]
    for scaled, raw in _SCALED.items():
        df[scaled] = df[raw] / shares_capacity
    df["log_adv"] = np.log10(df["adv"])

    out = df[DISC_DATASET_COLUMNS].replace([np.inf, -np.inf], np.nan)
    numeric = [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS, "excess_return"]
    out = out.dropna(subset=numeric)
    return out.sort_values(["date", "code"]).reset_index(drop=True)
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `uv run --extra dev pytest tests/test_features_disclosure.py tests/test_dart_dataset.py -q`
Expected: 13 passed

- [ ] **Step 7: 커밋**

```bash
git add src/scalp_it/features/disclosure.py src/scalp_it/dart/dataset.py \
        tests/test_features_disclosure.py tests/test_dart_dataset.py
git commit -m "feat(dart): 공시 피처 10종과 이벤트 데이터셋 조립"
```

---

### Task 7: 판정 — 층 비교 · 십분위 단조성 · 절대 임계

**Files:**
- Create: `src/scalp_it/validate/monotonic.py`
- Create: `src/scalp_it/report/threshold.py`
- Create: `src/scalp_it/cli_dart_layers.py`
- Modify: `pyproject.toml` — `scalp-dart-layers` 추가
- Test: `tests/test_validate_monotonic.py`
- Test: `tests/test_report_threshold.py`

**Interfaces:**
- Consumes: `build_dart_dataset`, 계획 4 `model.layered.walk_forward_predict_features`·`fold_dates`, `validate.layer_verdict.layer_verdict`, `validate.compare.compare_layers`
- Produces:
  - `decile_table(pred: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame` — 컬럼 `[decile, n, mean_excess, median_excess]`. `decile` 0 = 예측 최저
  - `monotonicity(pred: pd.DataFrame, *, n_bins: int = 10) -> dict` — 키 `spearman`, `top_minus_bottom`, `top_vs_median`, `is_monotone`. `is_monotone` 은 `spearman >= 0.6` **그리고** `top_minus_bottom > 0`
  - `concurrency_threshold(pred: pd.DataFrame, *, hold_days: int = 40, target_concurrent: int = 20) -> dict` — 키 `threshold`, `expected_per_day`, `expected_concurrent`, `n_train_days`
  - `main(argv) -> int` — `scalp-dart-layers`

**절대 임계 산식 (사전등록).** 성과가 아니라 구조에서 나온다.

```
목표 동시 보유 = 20   (스펙 §6 의 15~25 중앙값)
하루 필요 후보 = 20 / hold_days = 20 / 40 = 0.5
임계값 = 학습 구간 예측값의 상위 (0.5 × n_train_days / n_train_rows) 분위수
```

즉 "학습 구간에서 하루 평균 0.5건이 통과하는 예측값"이다. 검증 구간에서는 이 값을
그대로 적용한다. **결과를 보고 조정하지 않는다.**

- [ ] **Step 1: 실패하는 단조성 테스트 작성**

`tests/test_validate_monotonic.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.validate.monotonic import decile_table, monotonicity


def _pred(n: int, signal: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prediction = rng.normal(size=n)
    excess = prediction * signal + rng.normal(scale=0.02, size=n)
    return pd.DataFrame({
        "code": [f"{i % 50:06d}" for i in range(n)],
        "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D"),
        "excess_return": excess,
        "prediction": prediction,
        "fold": 0,
    })


def test_decile_table_shape():
    table = decile_table(_pred(2000, 0.05))
    assert list(table.columns) == ["decile", "n", "mean_excess", "median_excess"]
    assert len(table) == 10
    assert table["decile"].tolist() == list(range(10))


def test_monotone_when_prediction_carries_signal():
    got = monotonicity(_pred(2000, 0.05))
    assert got["is_monotone"] is True
    assert got["spearman"] > 0.6
    assert got["top_minus_bottom"] > 0


def test_not_monotone_when_prediction_is_noise():
    got = monotonicity(_pred(2000, 0.0))
    assert got["is_monotone"] is False


def test_inverted_signal_is_not_monotone():
    """예측이 거꾸로면 단조가 아니다 — 상위를 골라 들어갈 수 없다."""
    pred = _pred(2000, 0.05)
    pred["prediction"] = -pred["prediction"]
    assert monotonicity(pred)["is_monotone"] is False


def test_report_has_all_keys():
    got = monotonicity(_pred(500, 0.05))
    for key in ("spearman", "top_minus_bottom", "top_vs_median", "is_monotone"):
        assert key in got
```

- [ ] **Step 2: 실패하는 임계 테스트 작성**

`tests/test_report_threshold.py`:

```python
import numpy as np
import pandas as pd

from scalp_it.report.threshold import concurrency_threshold


def _pred(n_days: int, per_day: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for i in range(per_day):
            rows.append({
                "code": f"{i:06d}",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                "excess_return": rng.normal(scale=0.05),
                "prediction": rng.normal(),
                "fold": 0,
            })
    return pd.DataFrame(rows)


def test_threshold_yields_the_target_concurrency():
    pred = _pred(200, 50)
    got = concurrency_threshold(pred, hold_days=40, target_concurrent=20)
    assert np.isclose(got["expected_concurrent"], 20, rtol=0.25)


def test_threshold_is_a_prediction_quantile():
    pred = _pred(200, 50)
    got = concurrency_threshold(pred, hold_days=40, target_concurrent=20)
    passing = (pred["prediction"] >= got["threshold"]).sum()
    assert np.isclose(passing / pred["date"].nunique(), got["expected_per_day"], rtol=0.25)


def test_higher_target_lowers_the_threshold():
    pred = _pred(200, 50)
    low = concurrency_threshold(pred, target_concurrent=10)["threshold"]
    high = concurrency_threshold(pred, target_concurrent=40)["threshold"]
    assert high < low


def test_report_has_all_keys():
    got = concurrency_threshold(_pred(50, 20))
    for key in ("threshold", "expected_per_day", "expected_concurrent", "n_train_days"):
        assert key in got
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run --extra dev pytest tests/test_validate_monotonic.py tests/test_report_threshold.py -q`
Expected: FAIL — 두 모듈 `ModuleNotFoundError`

- [ ] **Step 4: 단조성 구현**

`src/scalp_it/validate/monotonic.py`:

```python
"""예측값 십분위 단조성 — "확실한 것만"이 성립하는지 (스펙 §6).

층이 랜덤 널을 이겨도 **예측값이 클수록 실제 수익이 좋지 않으면** 상위만 골라
들어가는 전략은 성립하지 않는다. 게이트와 별개로 반드시 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decile_table(pred: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """예측값 분위별 실제 초과수익. ``decile`` 0 이 예측 최저다."""
    frame = pred.dropna(subset=["prediction", "excess_return"]).copy()
    frame["decile"] = pd.qcut(
        frame["prediction"].rank(method="first"), n_bins, labels=False
    )
    table = (
        frame.groupby("decile")
        .agg(n=("excess_return", "size"),
             mean_excess=("excess_return", "mean"),
             median_excess=("excess_return", "median"))
        .reset_index()
    )
    return table.sort_values("decile").reset_index(drop=True)


def monotonicity(pred: pd.DataFrame, *, n_bins: int = 10) -> dict:
    """분위 평균이 예측 순서를 따라 오르는가.

    ``is_monotone`` 은 스피어만 상관 0.6 이상 **그리고** 최상위−최하위 > 0 이다.
    완전한 단조를 요구하지 않는 이유는 분위 평균에 잡음이 있기 때문이고,
    0.6 은 사전등록 값이다 — 결과를 보고 낮추지 않는다.
    """
    table = decile_table(pred, n_bins=n_bins)
    if len(table) < 3:
        return {"spearman": float("nan"), "top_minus_bottom": float("nan"),
                "top_vs_median": float("nan"), "is_monotone": False}

    ranks = table["decile"].to_numpy(dtype=float)
    means = table["mean_excess"].to_numpy(dtype=float)
    spearman = float(pd.Series(ranks).corr(pd.Series(means), method="spearman"))
    top_minus_bottom = float(means[-1] - means[0])
    top_vs_median = float(means[-1] - np.median(means))
    return {
        "spearman": spearman,
        "top_minus_bottom": top_minus_bottom,
        "top_vs_median": top_vs_median,
        "is_monotone": bool(spearman >= 0.6 and top_minus_bottom > 0),
    }
```

- [ ] **Step 5: 임계 구현**

`src/scalp_it/report/threshold.py`:

```python
"""절대 임계 — "확실한 것만" (스펙 §6).

날짜당 상위 N 을 억지로 뽑으면 재료가 시원찮은 날에도 후보가 나온다.
임계 미달이면 그날은 0건이어야 한다.

**임계값은 구조에서 정한다, 성과에서가 아니다.** 목표 동시 보유를 정하고
거기서 하루 통과 건수를 역산한 뒤, 학습 구간 예측값의 해당 분위수를 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency_threshold(
    pred: pd.DataFrame,
    *,
    hold_days: int = 40,
    target_concurrent: int = 20,
) -> dict:
    """목표 동시 보유를 만족하는 예측값 임계를 학습 구간에서 구한다.

    Args:
        hold_days: 보유 거래일. 동시 보유 ≈ 하루 통과 건수 × hold_days.
        target_concurrent: 목표 동시 보유 종목 수. 스펙 §6 의 15~25 중앙값이 20.
    """
    frame = pred.dropna(subset=["prediction"])
    n_days = int(frame["date"].nunique())
    if n_days == 0 or frame.empty:
        return {"threshold": float("nan"), "expected_per_day": 0.0,
                "expected_concurrent": 0.0, "n_train_days": 0}

    per_day = target_concurrent / float(hold_days)
    n_pass = max(1, int(round(per_day * n_days)))
    quantile = 1.0 - min(1.0, n_pass / len(frame))
    threshold = float(np.quantile(frame["prediction"].to_numpy(), quantile))

    passing = int((frame["prediction"] >= threshold).sum())
    return {
        "threshold": threshold,
        "expected_per_day": passing / n_days,
        "expected_concurrent": passing / n_days * hold_days,
        "n_train_days": n_days,
    }
```

- [ ] **Step 6: 판정 CLI 구현**

`src/scalp_it/cli_dart_layers.py`:

```python
"""`scalp-dart-layers` — ①수급 vs ②+공시 판정.

FAIL 이 나와도 exit 0 이다. 부정 결과는 오류가 아니라 결과다 (스펙 §11).
파라미터를 바꿔 재시도하는 경로는 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.db import connect, load_prices, load_supply_demand
from .dart.dataset import build_dart_dataset
from .dart.store import init_dart_db, load_disclosures
from .dataset import FEATURE_COLUMNS
from .features.disclosure import DISC_FEATURE_COLUMNS
from .model.layered import fold_dates, walk_forward_predict_features
from .report.threshold import concurrency_threshold
from .validate.layer_verdict import layer_verdict
from .validate.monotonic import decile_table, monotonicity

_LAYERS = {
    "supply": FEATURE_COLUMNS,
    "supply_disc": [*FEATURE_COLUMNS, *DISC_FEATURE_COLUMNS],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="공시 층 판정 (①수급 vs ②+공시)")
    parser.add_argument("--db", default=None, help="가격 DSN. 생략 시 KR_QUANT_DB")
    parser.add_argument("--dart-db", default="data/dart.db")
    parser.add_argument("--start", default="2016-09-09")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--adv-floor", type=float, default=5000.0)
    parser.add_argument("--hold-end", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--target-concurrent", type=int, default=20)
    parser.add_argument("--out", default="results/dart_layers.json")
    args = parser.parse_args(argv)

    if not Path(args.dart_db).is_file():
        print(f"\n공시 DB 가 없다: {args.dart_db}\n  → 먼저 `scalp-dart` 를 돌려야 한다.\n",
              file=sys.stderr)
        return 1

    conn = connect(args.db)
    prices = load_prices(conn, args.start, args.end)
    supply = load_supply_demand(conn, args.start, args.end)
    disc = load_disclosures(
        init_dart_db(args.dart_db),
        args.start.replace("-", ""),
        args.end.replace("-", ""),
    )
    print(f"가격 {len(prices):,} · 수급 {len(supply):,} · 공시 {len(disc):,}", file=sys.stderr)

    ds = build_dart_dataset(prices, supply, disc,
                            adv_floor=args.adv_floor, hold_end=args.hold_end)
    print(f"이벤트 {len(ds):,} · 종목 {ds['code'].nunique():,}", file=sys.stderr)
    if ds.empty:
        print("이벤트가 없다. 공시 수집 범위를 확인하라.", file=sys.stderr)
        return 1

    folds = fold_dates(ds, n_folds=args.folds)
    report: dict = {
        "n_events": len(ds),
        "n_codes": int(ds["code"].nunique()),
        "date_range": [str(ds["date"].min().date()), str(ds["date"].max().date())],
        "config": {"adv_floor": args.adv_floor, "hold_end": args.hold_end,
                   "top_n": args.top_n, "folds": args.folds,
                   "target_concurrent": args.target_concurrent},
        "layers": {},
    }

    for name, features in _LAYERS.items():
        pred = walk_forward_predict_features(ds, features, folds=folds)
        verdict = layer_verdict(pred, top_n=args.top_n)
        verdict["monotonicity"] = monotonicity(pred)
        verdict["deciles"] = decile_table(pred).round(6).to_dict("records")
        verdict["threshold"] = concurrency_threshold(
            pred, hold_days=args.hold_end, target_concurrent=args.target_concurrent
        )
        verdict["n_features"] = len(features)
        report["layers"][name] = verdict
        print(f"  {name}: {verdict['overall_verdict']} · "
              f"단조 {verdict['monotonicity']['is_monotone']}", file=sys.stderr)

    base = report["layers"]["supply"]
    disc_layer = report["layers"]["supply_disc"]
    report["disclosure_adds_value"] = bool(
        disc_layer["gate"]["strategy_mean"] > base["gate"]["strategy_mean"]
        and disc_layer["monotonicity"]["is_monotone"]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["layers"]["supply_disc"]["checks"], indent=2, ensure_ascii=False))
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**참고:** `layer_verdict` 의 반환 키 이름(`gate`·`checks`·`overall_verdict`)은 계획 4
Task 4 구현본을 따른다. 실제 키가 다르면 **CLI 를 맞춘다** — `layer_verdict` 를 고치지
않는다. 계획 4 테스트가 계속 통과해야 한다.

- [ ] **Step 7: `pyproject.toml` 에 엔트리포인트 추가**

```toml
scalp-dart-layers = "scalp_it.cli_dart_layers:main"
```

Run: `uv sync --extra dev`

- [ ] **Step 8: 전체 테스트와 lint**

Run: `uv run --extra dev pytest -q && uv run --extra dev ruff check src tests`
Expected: 전부 통과. 기존 테스트가 하나도 깨지지 않아야 한다.

- [ ] **Step 9: 커밋**

```bash
git add src/scalp_it/validate/monotonic.py src/scalp_it/report/threshold.py \
        src/scalp_it/cli_dart_layers.py pyproject.toml \
        tests/test_validate_monotonic.py tests/test_report_threshold.py
git commit -m "feat(validate): 십분위 단조성 · 절대 임계 · 공시 층 판정 CLI"
```

---

## 이 계획이 끝나면 답할 수 있는 것

> **DART 공시가 수급 위에 D+20~40 초과수익 예측력을 더하는가?**
> **그리고 "확실한 것만 골라 들어가기"가 성립하는가?**

판정은 계획 4 의 `layer_verdict`(전체 널 대비 / 모든 폴드 널 대비 / 손 안 댄 최종창 /
플라시보, 하방 회피, 선별도) 에 **십분위 단조성**을 더해서 본다.

**어느 쪽이 나오든 기록하고 끝낸다.**

- **②가 ①을 이기고 단조성도 성립하면** — 재료 축이 실재함이 무료로 증명된다.
  ③층(공시 원문 + LLM)으로 간다. 여전히 무료다.
- **②가 ①을 못 이기면** — 재료 축 자체가 이 지평에서 약하다. ③·④에 시간과 돈을
  쓰기 전에 알게 된다. **이것이 이 계획의 존재 이유다.**
- **널은 이겼는데 단조성이 없으면** — 평균으로는 우위이나 "상위만 골라 들어간다"가
  성립하지 않는다. 그 경우 운영 형태를 다시 설계해야 한다(분산 보유 등).

## 범위 밖

- **공시 원문 수집과 LLM 피처 (③층).** 원문은 무료로 받을 수 있고(`document.xml`,
  ZIP 안 XML, `<STYLE>` 블록 제거 필요) 계약금액·매출액 대비·계약기간이 텍스트에 있다.
  ②층 결과를 보고 계획 6 으로 다룬다. **여기서 미리 만들지 않는다.**
- **내부자 거래 방향(매수/매도).** 1차는 건수만 쓴다. 방향은 지분공시 종합정보 API
  또는 원문에서 2차에.
- ④층 뉴스, 포지션 예산 시뮬레이션, 실시간 모니터링, 주문 실행.
