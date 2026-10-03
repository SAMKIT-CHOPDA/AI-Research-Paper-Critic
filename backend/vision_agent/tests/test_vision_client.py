"""
Offline unit tests for the multimodal vision client abstraction.

Verifies:
- a successful call passes the prompt, the selected model and a base64 image
- transport/API errors are wrapped and API keys are masked
- auth, model-unavailable and malformed responses map to distinct errors
- a missing model identifier or credential fails loudly (never a silent fallback)
- provider selection and credential resolution come from the environment

No real API call is made: the OpenAI-compatible transport is replaced by a fake.
"""

import io
import types

import pytest
from PIL import Image

from backend.vision_agent.config import VisionConfig, VisionConfigurationError
from backend.vision_agent.vision_client import (
    OpenAICompatibleVisionClient,
    VisionAuthError,
    VisionClientError,
    VisionModelUnavailableError,
    VisionResponseError,
    build_vision_client,
    extract_response_text,
    mask_secret,
    resolve_api_key,
)

FAKE_KEY = "sk-test-secret-1234567890"


class FakeCompletions:
    """Records the call payload and returns a scripted response or raises."""

    def __init__(self, response=None, error: Exception = None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


def make_response(text: str):
    message = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def png_bytes(size=(12, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def build_client(completions: FakeCompletions, api_key: str = FAKE_KEY):
    client = OpenAICompatibleVisionClient(api_key=api_key)
    client._client = FakeOpenAI(completions)
    return client


class TestSuccessfulCall:
    def test_prompt_model_and_image_are_sent(self):
        completions = FakeCompletions(response=make_response('{"observation": "ok"}'))
        client = build_client(completions)

        result = client.analyze_image(png_bytes(), "Describe the figure", "vision-model-1")

        assert result == '{"observation": "ok"}'
        payload = completions.calls[0]
        assert payload["model"] == "vision-model-1"
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Describe the figure"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_can_be_supplied_as_a_file_path(self, tmp_path):
        path = tmp_path / "asset.jpg"
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
        path.write_bytes(buffer.getvalue())

        completions = FakeCompletions(response=make_response("text"))
        client = build_client(completions)
        client.analyze_image(str(path), "prompt", "vision-model-1")

        data_url = completions.calls[0]["messages"][0]["content"][1]["image_url"]["url"]
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_missing_image_file_raises(self, tmp_path):
        client = build_client(FakeCompletions(response=make_response("text")))
        with pytest.raises(VisionClientError):
            client.analyze_image(str(tmp_path / "missing.png"), "prompt", "vision-model-1")


class TestErrorHandling:
    def test_api_error_is_wrapped_and_key_is_masked(self):
        completions = FakeCompletions(error=RuntimeError(f"boom with {FAKE_KEY} inside"))
        client = build_client(completions)

        with pytest.raises(VisionClientError) as excinfo:
            client.analyze_image(png_bytes(), "prompt", "vision-model-1")

        message = str(excinfo.value)
        assert FAKE_KEY not in message
        assert "[REDACTED_API_KEY]" in message

    def test_authentication_error_is_distinguished(self):
        client = build_client(FakeCompletions(error=RuntimeError("401 Unauthorized")))
        with pytest.raises(VisionAuthError):
            client.analyze_image(png_bytes(), "prompt", "vision-model-1")

    def test_unavailable_model_is_distinguished(self):
        client = build_client(FakeCompletions(error=RuntimeError("model vision-model-9 not found")))
        with pytest.raises(VisionModelUnavailableError):
            client.analyze_image(png_bytes(), "prompt", "vision-model-9")

    def test_missing_model_identifier_is_rejected_before_any_call(self):
        completions = FakeCompletions(response=make_response("text"))
        client = build_client(completions)

        with pytest.raises(VisionConfigurationError):
            client.analyze_image(png_bytes(), "prompt", "")

        assert completions.calls == []

    def test_empty_prompt_is_rejected(self):
        client = build_client(FakeCompletions(response=make_response("text")))
        with pytest.raises(VisionClientError):
            client.analyze_image(png_bytes(), "   ", "vision-model-1")

    def test_missing_credentials_raise_auth_error(self, monkeypatch):
        for name in ("VISION_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        client = OpenAICompatibleVisionClient(provider="openai")
        with pytest.raises(VisionAuthError) as excinfo:
            client.analyze_image(png_bytes(), "prompt", "vision-model-1")
        assert "VISION_API_KEY" in str(excinfo.value)


class TestResponseParsing:
    def test_malformed_response_without_choices_raises(self):
        with pytest.raises(VisionResponseError):
            extract_response_text(types.SimpleNamespace(choices=[]))

    def test_empty_content_raises(self):
        with pytest.raises(VisionResponseError):
            extract_response_text(make_response("   "))

    def test_content_parts_are_joined(self):
        message = types.SimpleNamespace(
            content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
        )
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
        assert extract_response_text(response) == "part one part two"

    def test_missing_content_is_rejected(self):
        message = types.SimpleNamespace(content=None)
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
        with pytest.raises(VisionResponseError):
            extract_response_text(response)


class TestCredentialsAndProvider:
    def test_mask_secret_replaces_key(self):
        assert mask_secret("token=abc123", "abc123") == "token=[REDACTED_API_KEY]"
        assert mask_secret("nothing", None) == "nothing"
        assert mask_secret("", "abc") == ""

    def test_explicit_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("VISION_API_KEY", "env-key")
        assert resolve_api_key("openai", "explicit-key") == "explicit-key"

    def test_vision_api_key_takes_precedence_over_provider_key(self, monkeypatch):
        monkeypatch.setenv("VISION_API_KEY", "vision-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        assert resolve_api_key("openai") == "vision-key"

    def test_provider_specific_key_is_used_as_fallback(self, monkeypatch):
        monkeypatch.delenv("VISION_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        assert resolve_api_key("openai") == "openai-key"
        monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")
        assert resolve_api_key("typesafe") == "typesafe-key"

    def test_no_credentials_resolves_to_none(self, monkeypatch):
        for name in ("VISION_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert resolve_api_key("openai") is None

    def test_build_vision_client_selects_configured_provider(self):
        client = build_vision_client()
        assert client.provider_name in {"openai", "typesafe"}

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(VisionClientError):
            build_vision_client(VisionConfig(provider="not-a-provider"))

    def test_provider_name_never_contains_the_key(self):
        client = OpenAICompatibleVisionClient(api_key=FAKE_KEY)
        assert FAKE_KEY not in client.provider_name