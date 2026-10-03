"""
Offline unit tests for the synthesis-model client abstraction.

The provider is never contacted: a fake transport stands in for the HTTP layer, so
these tests cover credential resolution, missing configuration, provider failures,
malformed responses and credential masking without a network or an API key.
"""

import pytest

from backend.critique_engine.config import CritiqueConfig, CritiqueConfigurationError
from backend.critique_engine.llm_client import (
    BaseCritiqueClient,
    CritiqueAuthError,
    CritiqueClientError,
    CritiqueModelUnavailableError,
    CritiqueResponseError,
    OpenAICompatibleCritiqueClient,
    build_critique_client,
    extract_response_text,
    mask_secret,
    resolve_api_key,
)

SECRET = "sk-unit-test-secret-value"


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = FakeChat(FakeCompletions())


def client_with_fake_transport(monkeypatch, response=None, error=None, api_key=SECRET):
    monkeypatch.setenv("CRITIQUE_API_KEY", api_key)
    client = OpenAICompatibleCritiqueClient(provider="openai")
    completions = FakeCompletions(response=response, error=error)
    client._client = FakeOpenAI()
    client._client.chat.completions = completions
    return client, completions


def test_successful_call_returns_the_model_text(monkeypatch):
    client, completions = client_with_fake_transport(
        monkeypatch, response=_Response('{"sections": []}')
    )
    text = client.generate_structured(
        model="configured-model",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )
    assert text == '{"sections": []}'
    assert completions.requests[0]["model"] == "configured-model"
    assert [message["role"] for message in completions.requests[0]["messages"]] == [
        "system",
        "user",
    ]


def test_missing_api_key_raises_an_auth_error(monkeypatch):
    monkeypatch.delenv("CRITIQUE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleCritiqueClient(provider="openai")
    with pytest.raises(CritiqueAuthError) as excinfo:
        client.generate_structured("m", "", "prompt")
    assert "CRITIQUE_API_KEY" in str(excinfo.value)


def test_missing_model_is_a_configuration_error(monkeypatch):
    client, _ = client_with_fake_transport(monkeypatch, response=_Response("{}"))
    with pytest.raises(CritiqueConfigurationError):
        client.generate_structured("", "", "prompt")
    with pytest.raises(CritiqueConfigurationError):
        client.generate_structured(None, "", "prompt")


def test_empty_prompt_is_rejected_before_any_call(monkeypatch):
    client, completions = client_with_fake_transport(
        monkeypatch, response=_Response("{}")
    )
    with pytest.raises(CritiqueClientError):
        client.generate_structured("m", "", "   ")
    assert completions.requests == []


def test_authentication_failure_is_classified_and_masked(monkeypatch):
    client, _ = client_with_fake_transport(
        monkeypatch,
        error=RuntimeError(f"401 Unauthorized: invalid api key {SECRET}"),
    )
    with pytest.raises(CritiqueAuthError) as excinfo:
        client.generate_structured("m", "", "prompt")
    assert SECRET not in str(excinfo.value)


def test_unavailable_model_is_classified(monkeypatch):
    client, _ = client_with_fake_transport(
        monkeypatch, error=RuntimeError("The model 'x' does not exist")
    )
    with pytest.raises(CritiqueModelUnavailableError):
        client.generate_structured("x", "", "prompt")


def test_generic_api_failure_is_masked(monkeypatch):
    client, _ = client_with_fake_transport(
        monkeypatch, error=RuntimeError(f"upstream timeout using key {SECRET}")
    )
    with pytest.raises(CritiqueClientError) as excinfo:
        client.generate_structured("m", "", "prompt")
    assert SECRET not in str(excinfo.value)
    assert "[REDACTED_API_KEY]" in str(excinfo.value)


def test_malformed_response_is_reported(monkeypatch):
    client, _ = client_with_fake_transport(monkeypatch, response=_Response("   "))
    with pytest.raises(CritiqueResponseError):
        client.generate_structured("m", "", "prompt")


def test_extract_response_text_handles_content_parts():
    response = _Response([{"text": "{\"a\": "}, {"text": "1}"}])
    assert extract_response_text(response) == '{"a": 1}'


def test_extract_response_text_rejects_a_response_without_choices():
    with pytest.raises(CritiqueResponseError):
        extract_response_text(object())


def test_json_response_format_is_opt_in(monkeypatch):
    monkeypatch.setenv("CRITIQUE_API_KEY", SECRET)
    client = OpenAICompatibleCritiqueClient(provider="openai", json_response_format=True)
    completions = FakeCompletions(response=_Response("{}"))
    client._client = FakeOpenAI()
    client._client.chat.completions = completions
    client.generate_structured("m", "", "prompt")
    assert completions.requests[0]["response_format"] == {"type": "json_object"}


def test_api_key_resolution_prefers_the_critique_variable(monkeypatch):
    monkeypatch.setenv("CRITIQUE_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", "other")
    assert resolve_api_key("openai") == SECRET
    assert resolve_api_key("openai", api_key="explicit") == "explicit"


def test_api_key_resolution_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("CRITIQUE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert resolve_api_key("openai") is None


def test_mask_secret_leaves_unrelated_text_untouched():
    assert mask_secret("nothing to hide", SECRET) == "nothing to hide"
    assert mask_secret(f"key={SECRET}", SECRET) == "key=[REDACTED_API_KEY]"
    assert mask_secret("", SECRET) == ""


def test_build_client_rejects_an_unknown_provider():
    with pytest.raises(CritiqueClientError):
        build_critique_client(CritiqueConfig(provider="not-a-provider"))


def test_build_client_returns_the_configured_provider(monkeypatch):
    monkeypatch.setenv("CRITIQUE_API_KEY", SECRET)
    client = build_critique_client(CritiqueConfig(provider="openai"))
    assert isinstance(client, BaseCritiqueClient)
    assert client.provider_name == "openai"