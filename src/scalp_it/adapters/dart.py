"""OpenDART 어댑터 — sync 파사드. 실제 HTTP·키 로테이션은 ``krx_news_client.DartScraper``.

무료·공식·10년+·접수일자 단위 point-in-time. 인증키는 환경변수로만 받는다.

키를 여러 개 돌리는 이유는 일 20,000건 한도 때문이다. quant-airflow 가 이미
3개를 발급받아 두었고, 10년치 수집에 약 30,000 콜이 필요하다.

2026-09-13: 이 파일이 직접 구현하던 키 로테이션·페이지네이션·상태코드 처리를
``krx_news_client``(daytrade-it도 쓰는 형제 레포)로 옮겼다 — daytrade-it의
DART 폴러와 같은 로직을 따로 유지하지 않기 위해서다. 이 파사드는 그
``asyncio`` 인터페이스를 기존 sync 호출부(``cli_dart.py``)가 쓰던 모양 그대로
감싸고, raw dict row(=``rcept_no``/``corp_cls`` 포함)를 그대로 돌려준다 —
``Disclosure`` 모델은 이 필드들을 안 들고 있어서 우리 저장 스키마와 안 맞는다.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from krx_news_client import DartAPIError, DartQuotaExceededError, DartScraper

_KEY_ENV = ("DART_API_KEY", "DART_API_KEY_2", "DART_API_KEY_3")


class DartError(Exception):
    """OpenDART 가 오류를 돌려주었거나(한도초과 포함), 인증키가 없다."""


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
    """공시검색 페이지를 읽는다. 키 로테이션과 한도 초과 폴백은 ``DartScraper`` 위임."""

    def __init__(self, keys: list[str], *, sleep: float = 0.0) -> None:
        if not keys:
            raise DartError("인증키가 비어 있다.")
        self._scraper = DartScraper(api_key=list(keys))
        # DartScraper 자체엔 요청 간 sleep 이 없다(공식 키 기반 API라 rate-limit
        # 이 아니라 일 쿼터 문제라서) -- 그래도 호출부가 명시적으로 원하면 존중한다.
        self._sleep = sleep

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
        if self._sleep:
            import time

            time.sleep(self._sleep)
        try:
            return asyncio.run(
                self._scraper.search_disclosures(
                    bgn_de=bgn_de, end_de=end_de, corp_cls=corp_cls,
                    page_no=page_no, page_count=page_count,
                )
            )
        except DartQuotaExceededError as exc:
            raise DartError(f"모든 인증키가 한도 초과다: {exc}") from exc
        except DartAPIError as exc:
            raise DartError(f"OpenDART 오류: {exc}") from exc

    def search_all(self, *, bgn_de: str, end_de: str, corp_cls: str) -> list[dict[str, Any]]:
        """구간의 모든 페이지를 모은다."""
        try:
            return asyncio.run(
                self._scraper.search_disclosures_all(bgn_de=bgn_de, end_de=end_de, corp_cls=corp_cls)
            )
        except DartQuotaExceededError as exc:
            raise DartError(f"모든 인증키가 한도 초과다: {exc}") from exc
        except DartAPIError as exc:
            raise DartError(f"OpenDART 오류: {exc}") from exc
