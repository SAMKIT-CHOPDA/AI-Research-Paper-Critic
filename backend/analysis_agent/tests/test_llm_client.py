"""
Offline unit tests for the analysis model client abstraction.

No network is touched: the OpenAI-compatible client is exercised with an injected
fake SDK object, and credential handling is verified with environment stubs only.
"""

from types import SimpleNamespace

import pytest

from backend.analysis_agent.config import AnalysisConfig, AnalysisConfigurationError
from backend.analysis_agent.llm_client import (
    AnalysisAuthError,
    AnalysisClientError,
    AnalysisModelUnavailableError,
    AnalysisResponseError,
    BaseAnalysisClient,
    OpenAICompatibleAnalysisClient,
    build_analysis_client,
    extract_response_text,
    mask_secret,
    resolve_api_key,
)


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAI:
    """Minimal stand-in for the OpenAI SDK client object."""

    def __init__(self, completions: FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)


def make_response(content="{}"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def client_with(completions: FakeCompletions, **overrides):
    client = OpenAICompatibleAnalysisClient(api_key="test-key", **overrides)
    client._client = FakeOpenAI(completions)
    return client


def test_mock_client_implements_the_abstract_interface():
    class Dummy(BaseAnalysisClient):
        def complete(self, prompt, model):
            return "{}"

    assert Dummy().provider_name == "Dummy"


def test_resolve_api_key_prefers_the_analysis_specific_variable(monkeypatch):
    monkeypatch.setenv("ANALYSIS_API_KEY", "analysis-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")

    assert resolve_api_key("openai") == "analysis-key"
    assert resolve_api_key("typesafe") == "analysis-key"
    assert resolve_api_key("openai", api_key="explicit") == "explicit"


def test_resolve_api_key_falls_back_to_provider_specific_variables(monkeypatch):
    monkeypatch.delenv("ANALYSIS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")

    assert resolve_api_key("openai") == "openai-key"
    assert resolve_api_key("typesafe") == "typesafe-key"
    assert resolve_api_key("unknown-provider") is None


def test_resolve_api_key_returns_none_when_nothing_is_configured(monkeypatch):
    for name in ("ANALYSIS_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert resolve_api_key("openai") is None
    assert resolve_api_key("typesafe", api_key="   ") is None


def test_mask_secret_redacts_only_the_secret():
    assert mask_secret("key abc-123 failed", "abc-123") == "key [REDACTED_API_KEY] failed"
    assert mask_secret("no secret here", "abc-123") == "no secret here"
    assert mask_secret("", "abc-123") == ""
    assert mask_secret(None, "abc-123") is None
    assert mask_secret("text", None) == "text"


def test_build_analysis_client_uses_configuration_and_never_needs_import_time_keys(
    monkeypatch,
):
    for name in ("ANALYSIS_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    config = AnalysisConfig(
        provider="openai",
        base_url="https://gateway.example/v1",
        max_output_tokens=1234,
        request_timeout=9.5,
        json_response_format=True,
    )
    client = build_analysis_client(config)

    assert isinstance(client, OpenAICompatibleAnalysisClient)
    assert client.provider_name == "openai"
    assert client.base_url == "https://gateway.example/v1"
    assert client.max_output_tokens == 1234
    assert client.request_timeout == 9.5
    assert client.json_response_format is True


def test_build_analysis_client_rejects_unknown_providers():
    with pytest.raises(AnalysisClientError) as excinfo:
        build_analysis_client(AnalysisConfig(provider="not-a-provider"))
    assert "Unknown analysis provider" in str(excinfo.value)


def test_missing_credentials_fail_at_call_time_not_at_import(monkeypatch):
    for name in ("ANALYSIS_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    client = build_analysis_client(AnalysisConfig(provider="openai"))

    with pytest.raises(AnalysisAuthError) as excinfo:
        client.ensure_configured()
    assert "ANALYSIS_API_KEY" in str(excinfo.value)

    with pytest.raises(AnalysisAuthError):
        client.complete("prompt", "some-model")


def test_extract_response_text_reads_plain_and_partial_content():
    assert extract_response_text(make_response('{"a": 1}')) == '{"a": 1}'
    parts = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=[{"text": "par"}, {"text": "ts"}])
            )
        ]
    )
    assert extract_response_text(parts) == "parts"


def test_extract_response_text_rejects_malformed_responses():
    with pytest.raises(AnalysisResponseError):
        extract_response_text(SimpleNamespace(choices=[]))
    with pytest.raises(AnalysisResponseError):
        extract_response_text(
            SimpleNamespace(choices=[SimpleNamespace(message=None)])
        )
    with pytest.raises(AnalysisResponseError):
        extract_response_text(
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])
        )


def test_complete_forwards_the_request_without_structured_output_by_default():
    completions = FakeCompletions(response=make_response("{}"))
    client = client_with(completions, max_output_tokens=500)

    assert client.complete("evidence prompt", "model-x") == "{}"

    request = completions.calls[0]
    assert request["model"] == "model-x"
    assert request["max_tokens"] == 500
    assert request["messages"] == [{"role": "user", "content": "evidence prompt"}]
    assert "response_format" not in request


def test_complete_requests_json_output_only_when_configured():
    completions = FakeCompletions(response=make_response("{}"))
    client = client_with(completions, json_response_format=True)

    client.complete("evidence prompt", "model-x")

    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_requires_a_model_identifier():
    completions = FakeCompletions(response=make_response("{}"))
    client = client_with(completions)

    with pytest.raises(AnalysisConfigurationError):
        client.complete("prompt", "")
    with pytest.raises(AnalysisClientError):
        client.complete("   ", "model-x")
    assert completions.calls == []


def test_auth_failures_are_mapped_and_masked():
    completions = FakeCompletions(error=RuntimeError("401 invalid api key: test-key"))
    client = client_with(completions)

    with pytest.raises(AnalysisAuthError) as excinfo:
        client.complete("prompt", "model-x")

    message = str(excinfo.value)
    assert "rejected the credentials" in message
    assert "ANALYSIS_API_KEY" in message
    assert "test-key" not in message


def test_missing_model_failures_are_mapped():
    completions = FakeCompletions(
        error=RuntimeError("The model `model-x` does not exist")
    )
    client = client_with(completions)

    with pytest.raises(AnalysisModelUnavailableError) as excinfo:
        client.complete("prompt", "model-x")
    assert "not available" in str(excinfo.value)


def test_generic_provider_failures_are_mapped_and_masked():
    completions = FakeCompletions(error=RuntimeError("boom with test-key inside"))
    client = client_with(completions)

    with pytest.raises(AnalysisClientError) as excinfo:
        client.complete("prompt", "model-x")

    message = str(excinfo.value)
    assert "Analysis model call failed" in message
    assert "[REDACTED_API_KEY]" in message
    assert "test-key" not in message
