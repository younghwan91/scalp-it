import json

import httpx
import pytest

from scalp_it.score.provider import (
    DEFAULT_MODELS,
    FakeProvider,
    HttpProvider,
    ProviderError,
    provider_from_env,
)

_JSON = (
    '{"material_type":"수주","persistence":"one_off","specificity":2,'
    '"freshness":"new","sentiment_direction":1,"sentiment_strength":2,'
    '"scale_vs_size":"small"}'
)


# --- FakeProvider ---------------------------------------------------------

def test_fake_provider_returns_default_and_records_calls():
    provider = FakeProvider()
    out = provider.complete("sys", "user")
    assert "material_type" in out
    assert provider.calls == [("sys", "user")]


def test_fake_provider_plays_scripted_responses_in_order():
    provider = FakeProvider(responses=["첫번째", "두번째"])
    assert provider.complete("s", "u") == "첫번째"
    assert provider.complete("s", "u") == "두번째"


def test_fake_provider_raises_scripted_exception():
    provider = FakeProvider(responses=[ProviderError("일시 오류"), _JSON])
    with pytest.raises(ProviderError):
        provider.complete("s", "u")
    assert provider.complete("s", "u") == _JSON


# --- HttpProvider (MockTransport, 네트워크 없음) ---------------------------

def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": _JSON}]}}]},
        )

    provider = HttpProvider("gemini", "flash-lite-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert "flash-lite-test:generateContent" in seen["url"]
    assert seen["key"] == "KEY123"
    assert "시스템" in seen["body"] and "사용자" in seen["body"]


def test_openai_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": _JSON}}]})

    provider = HttpProvider("openai", "mini-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert seen["auth"] == "Bearer KEY123"
    assert "json_object" in seen["body"]


def test_anthropic_request_shape_and_extraction():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        seen["body"] = json.loads(request.read())
        # Haiku 5.5 는 thinking 블록을 앞에 둘 수 있다 — 첫 text 블록을 골라야 한다.
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "x"},
                    {"type": "text", "text": _JSON},
                ]
            },
        )

    provider = HttpProvider("anthropic", "haiku-test", "KEY123", client=_client(handler))
    assert provider.complete("시스템", "사용자") == _JSON
    assert seen["key"] == "KEY123"
    assert seen["version"] == "2023-06-01"
    body = seen["body"]
    assert "temperature" not in body  # Haiku 5.5 는 기본값 외 temperature 를 400 으로 거절
    assert body["thinking"] == {"type": "disabled"}
    assert body["system"] == [
        {"type": "text", "text": "시스템", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "사용자"}]


def test_anthropic_default_model_is_haiku_5_5():
    provider = provider_from_env({"SWING_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "K"})
    assert provider.model == "claude-haiku-5-5"
    assert DEFAULT_MODELS["anthropic"] == "claude-haiku-5-5"


def test_http_error_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError, match="503"):
        provider.complete("s", "u")


def test_transport_failure_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("연결 실패")

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError):
        provider.complete("s", "u")


def test_unexpected_payload_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": True})

    provider = HttpProvider("gemini", "m", "K", client=_client(handler))
    with pytest.raises(ProviderError, match="응답 형식"):
        provider.complete("s", "u")


# --- env 팩토리 -----------------------------------------------------------

def test_env_factory_builds_fake_provider():
    provider = provider_from_env({"SWING_LLM_PROVIDER": "fake"})
    assert provider.name == "fake"


def test_env_factory_defaults_to_gemini_with_default_model():
    provider = provider_from_env({"GEMINI_API_KEY": "K"})
    assert provider.name == "gemini"
    assert provider.model == DEFAULT_MODELS["gemini"]


def test_env_factory_honours_explicit_model():
    provider = provider_from_env({"GEMINI_API_KEY": "K", "SWING_LLM_MODEL": "custom-1"})
    assert provider.model == "custom-1"


def test_env_factory_requires_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        provider_from_env({"SWING_LLM_PROVIDER": "gemini"})


def test_env_factory_requires_model_for_openai():
    with pytest.raises(ValueError, match="SWING_LLM_MODEL"):
        provider_from_env({"SWING_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "K"})


def test_env_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="grok"):
        provider_from_env({"SWING_LLM_PROVIDER": "grok"})


def test_env_factory_never_returns_the_key_in_repr():
    provider = provider_from_env({"GEMINI_API_KEY": "SECRET-KEY-VALUE"})
    assert "SECRET-KEY-VALUE" not in repr(provider)
