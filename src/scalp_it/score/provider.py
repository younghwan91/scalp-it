"""LLM provider 추상화 — Gemini / OpenAI / Anthropic 을 환경변수로 교체한다 (스펙 §8).

SDK 를 쓰지 않고 httpx 로 직접 부른다. 우리가 쓰는 기능은 "system + user →
텍스트 하나"뿐이라 provider 당 요청 조립이 20줄 남짓이고, SDK 3종을 의존성에
올리면 공개 전환 시 설치 부담이 그 이득을 넘는다.

API 키는 환경변수로만 받는다. 어떤 경로로도 로그·repr 에 실리지 않게 한다.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

# 착수 시점에 실제로 서빙되는 ID 인지 확인한다 (Task 7 Step 1).
# 기본은 최저가 등급이다 — 기사당 작업이 "제목+리드 읽고 스키마 채우기"라
# 추론 난이도가 낮다 (스펙 §8).
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash-lite",
    # 2026-10-12 부터 Haiku 5.5 만 쓴다(4.5 결과는 data/legacy/ 에 보관).
    "anthropic": "claude-haiku-5-5",
    # openai 는 기본값을 두지 않는다. 등급 이름이 자주 바뀌어 낡은 상수가
    # 조용히 비싼 모델을 부르는 사고를 막는다. SWING_LLM_MODEL 로 명시하라.
}

_ENV_KEY_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_DEFAULT_FAKE_JSON = (
    '{"material_type":"기타","persistence":"one_off","specificity":0,'
    '"freshness":"repeat","sentiment_direction":0,"sentiment_strength":0,'
    '"scale_vs_size":"unknown"}'
)

_MAX_OUTPUT_TOKENS = 300  # 7필드 JSON 이면 충분하다. 폭주 과금을 막는 상한.


class ProviderError(RuntimeError):
    """호출이 실패했다. 배치 러너가 재시도 사유로 쓴다."""


@runtime_checkable
class LLMProvider(Protocol):
    """배치 러너가 아는 유일한 인터페이스."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """모델 응답 텍스트를 그대로 돌려준다. 파싱은 호출자 책임."""
        ...


class FakeProvider:
    """테스트·드라이런용. 네트워크에 나가지 않는다.

    ``responses`` 에 예외 인스턴스를 섞으면 실패 시나리오를 재현할 수 있다.
    """

    name = "fake"

    def __init__(
        self,
        *,
        model: str = "fake-1",
        responses: list[str | Exception] | None = None,
        default: str | None = None,
    ) -> None:
        self.model = model
        self._responses: list[str | Exception] = list(responses or [])
        self._default = default if default is not None else _DEFAULT_FAKE_JSON
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return self._default


# --- provider 별 요청 조립 / 응답 추출 (순수 함수) --------------------------

_Request = tuple[str, dict[str, str], dict[str, Any]]


def _gemini_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    return url, headers, body


def _gemini_extract(payload: dict) -> str:
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def _openai_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": _MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    return url, headers, body


def _openai_extract(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


def _anthropic_request(model: str, api_key: str, system: str, user: str) -> _Request:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Haiku 5.5 기준 요청 모양 — swing-it news/judge.py 와 같다.
    # - temperature 없음: 5.5 는 기본값이 아닌 값을 400 으로 거절한다.
    # - thinking 끔: 5.5 는 기본으로 생각하고, 출력 상한(300)을 생각이 다 먹는다
    #   (swing-it 파일럿 2026-10-08: 400 토큰 상한에서 500건 중 320건 빈 응답).
    #   기본 effort(medium)에서는 disabled 가 허용된다.
    # - system 은 cache_control 블록: 텍스트는 평문과 같다. 최소 캐시 접두가
    #   512 토큰이라 그보다 짧은 system 은 캐시만 안 될 뿐 오류는 없다.
    body = {
        "model": model,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": user}],
    }
    return url, headers, body


def _anthropic_extract(payload: dict) -> str:
    # thinking 블록이 앞에 올 수 있다 — 첫 text 블록만 본다.
    for block in payload["content"]:
        if isinstance(block, dict) and block.get("type") == "text":
            return block["text"]
    raise KeyError("text 블록이 없다")


_SPECS: dict[str, tuple[Callable[..., _Request], Callable[[dict], str]]] = {
    "gemini": (_gemini_request, _gemini_extract),
    "openai": (_openai_request, _openai_extract),
    "anthropic": (_anthropic_request, _anthropic_extract),
}


class HttpProvider:
    """httpx 로 직접 부르는 provider. ``client`` 를 주입하면 테스트가 가능하다."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if name not in _SPECS:
            raise ValueError(f"모르는 provider: {name}. 가능한 값: {sorted(_SPECS)}")
        self.name = name
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def __repr__(self) -> str:  # 키가 로그·예외에 실리지 않게 한다
        return f"HttpProvider(name={self.name!r}, model={self.model!r})"

    def complete(self, system: str, user: str) -> str:
        build, extract = _SPECS[self.name]
        url, headers, body = build(self.model, self._api_key, system, user)
        try:
            response = self._client.post(
                url, headers=headers, json=body, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} 요청 실패: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            return extract(response.json())
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name} 응답 형식이 예상과 다르다: {exc}") from exc


def provider_from_env(env: Mapping[str, str] | None = None) -> LLMProvider:
    """환경변수로 provider 를 조립한다. 키는 여기서만 읽는다."""
    source: Mapping[str, str] = env if env is not None else os.environ
    name = source.get("SWING_LLM_PROVIDER", "gemini").strip().lower()

    if name == "fake":
        return FakeProvider(model=source.get("SWING_LLM_MODEL", "fake-1"))
    if name not in _SPECS:
        raise ValueError(
            f"모르는 SWING_LLM_PROVIDER: {name!r}. "
            f"가능한 값: fake, {', '.join(sorted(_SPECS))}"
        )

    model = source.get("SWING_LLM_MODEL") or DEFAULT_MODELS.get(name)
    if not model:
        raise ValueError(f"{name} 는 기본 모델이 없다. SWING_LLM_MODEL 을 설정하라.")

    key_name = _ENV_KEY_NAMES[name]
    api_key = source.get("SWING_LLM_API_KEY") or source.get(key_name)
    if not api_key:
        raise ValueError(f"API 키가 없다. SWING_LLM_API_KEY 또는 {key_name} 를 설정하라.")

    timeout = float(source.get("SWING_LLM_TIMEOUT", "60"))
    return HttpProvider(name, model, api_key, timeout=timeout)
