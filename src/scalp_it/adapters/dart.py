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
