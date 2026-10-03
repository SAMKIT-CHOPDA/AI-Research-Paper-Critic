"""
Critique synthesis model abstraction (Phase 6).

The Critique Engine never talks to a provider directly: it calls
``BaseCritiqueClient.generate_structured(model, system_prompt, user_prompt,
response_schema)`` on an injected client. That keeps the engine provider
independent and makes the unit-test suite fully offline (tests inject a fake
client; no network is touched).

Credentials are read from environment variables only and are never logged,
printed, or embedded in error messages: provider exceptions pass through a masking
step before being re-raised as CritiqueClientError subclasses.

Model identifiers are deliberately *not* hard-coded: the caller (the engine, via
``resolve_model_for_level``) supplies the env-configured identifier, and a missing
configuration fails loudly instead of silently substituting a model.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from backend.critique_engine.config import (
    CRITIQUE_MAX_OUTPUT_TOKENS,
    CRITIQUE_REQUEST_TIMEOUT,
    DEFAULT_CRITIQUE_BASE_URL,
    DEFAULT_CRITIQUE_PROVIDER,
    CritiqueConfig,
    CritiqueConfigurationError,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = CRITIQUE_MAX_OUTPUT_TOKENS
DEFAULT_REQUEST_TIMEOUT = CRITIQUE_REQUEST_TIMEOUT

# Provider -> environment variable names holding the credential. CRITIQUE_API_KEY
# always wins so a user can point the Critique Engine at one specific credential.
_PROVIDER_KEY_ENV: Dict[str, tuple] = {
    "openai": ("CRITIQUE_API_KEY", "OPENAI_API_KEY"),
    "typesafe": ("CRITIQUE_API_KEY", "TYPESAFE_API_KEY"),
}


class CritiqueClientError(Exception):
    """Raised when a synthesis model call fails."""

    pass


class CritiqueAuthError(CritiqueClientError):
    """Raised when the configured provider rejects or lacks credentials."""

    pass


class CritiqueModelUnavailableError(CritiqueClientError):
    """Raised when the selected model cannot be reached or does not exist."""

    pass


class CritiqueResponseError(CritiqueClientError):
    """Raised when the provider returns a malformed or empty response."""

    pass


def mask_secret(text: str, secret: Optional[str]) -> str:
    """Replace any occurrence of a secret inside text with a redaction marker."""
    if not text:
        return text
    masked = str(text)
    if secret:
        masked = masked.replace(secret, "[REDACTED_API_KEY]")
    return masked


def resolve_api_key(
    provider: str = DEFAULT_CRITIQUE_PROVIDER,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the credential for a provider from the environment.

    Returns None when nothing is configured; the caller raises a clear error at
    call time rather than at import time.
    """
    if api_key and api_key.strip():
        return api_key.strip()
    key_env_names = _PROVIDER_KEY_ENV.get(
        (provider or "").strip().lower(), ("CRITIQUE_API_KEY",)
    )
    for name in key_env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


class BaseCritiqueClient(ABC):
    """Abstract synthesis client: one prompt + one model -> one text response."""

    @abstractmethod
    def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run one structured synthesis completion and return the raw model text.

        ``response_schema`` is advisory: implementations may forward it as a
        provider-side JSON schema, but the contract is always "one JSON object as
        text" so an endpoint without structured-output support still works.
        """
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier (never contains credentials)."""
        return type(self).__name__


def extract_response_text(response: object) -> str:
    """
    Pull the assistant text out of a chat-completions response.

    Kept separate from the HTTP call so malformed-response handling can be tested
    offline.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            raise CritiqueResponseError("Critique response contained no choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
    except CritiqueResponseError:
        raise
    except Exception as exc:
        raise CritiqueResponseError(f"Malformed critique response: {exc}") from exc

    if isinstance(content, list):
        # Some gateways return content parts instead of a plain string.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None or not str(content).strip():
        raise CritiqueResponseError("Critique response contained no textual content")
    return str(content)


class OpenAICompatibleCritiqueClient(BaseCritiqueClient):
    """
    Synthesis client for OpenAI-compatible chat-completions endpoints.

    Works with the OpenAI API and with any gateway implementing the same message
    shape; the endpoint is configurable via CRITIQUE_BASE_URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = DEFAULT_CRITIQUE_PROVIDER,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        json_response_format: bool = False,
    ):
        self.provider = (provider or DEFAULT_CRITIQUE_PROVIDER).strip().lower()
        self._api_key = resolve_api_key(self.provider, api_key)
        self.base_url = (base_url or DEFAULT_CRITIQUE_BASE_URL or "").strip() or None
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.request_timeout = float(request_timeout)
        self.json_response_format = bool(json_response_format)
        self._client = None

    @property
    def provider_name(self) -> str:
        return self.provider

    def _sanitize(self, message: object) -> str:
        return mask_secret(str(message), self._api_key)

    def ensure_configured(self) -> None:
        """
        Validate credentials and endpoint configuration without a model call.

        Raises CritiqueAuthError/CritiqueClientError when the client cannot be used,
        so a caller can report a clear configuration message before building any
        context.
        """
        self._get_client()

    def _get_client(self):
        """Lazy initialization so importing the module never needs a key."""
        if self._client is None:
            if not self._api_key:
                raise CritiqueAuthError(
                    f"Missing credentials for critique provider '{self.provider}'. "
                    "Set CRITIQUE_API_KEY (or the provider-specific key such as "
                    "OPENAI_API_KEY) in the environment."
                )
            try:
                from openai import OpenAI

                kwargs = {"api_key": self._api_key, "timeout": self.request_timeout}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except CritiqueClientError:
                raise
            except Exception as exc:
                raise CritiqueClientError(
                    f"Could not initialize the critique provider client: "
                    f"{self._sanitize(exc)}"
                ) from exc
        return self._client

    def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send one bounded synthesis prompt to the critique model.

        The model identifier must be supplied by the caller (environment
        configured); this method refuses to guess a fallback model.
        """
        if not model or not str(model).strip():
            raise CritiqueConfigurationError(
                "No synthesis model identifier was supplied to the critique client. "
                "Configure CRITIQUE_MODEL_BASIC / CRITIQUE_MODEL_MEDIUM / "
                "CRITIQUE_MODEL_ADVANCED (or CRITIQUE_MODEL) in the environment."
            )
        if not user_prompt or not str(user_prompt).strip():
            raise CritiqueClientError("Critique prompt cannot be empty")

        client = self._get_client()
        messages: list = []
        if system_prompt and str(system_prompt).strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})

        request: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "messages": messages,
        }
        if self.json_response_format:
            # Only requested when the deployment explicitly asks for structured
            # output; the default contract is "one JSON object as text".
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            error_text = self._sanitize(exc)
            lowered = error_text.lower()
            if (
                "401" in lowered
                or "invalid api key" in lowered
                or "unauthorized" in lowered
                or "authentication" in lowered
            ):
                raise CritiqueAuthError(
                    f"Critique provider '{self.provider}' rejected the credentials. "
                    "Verify CRITIQUE_API_KEY (or the provider-specific key)."
                ) from exc
            if "model" in lowered and (
                "not found" in lowered
                or "does not exist" in lowered
                or "unsupported" in lowered
            ):
                raise CritiqueModelUnavailableError(
                    f"Model '{model}' is not available to critique provider "
                    f"'{self.provider}'. Configure a model your account can access."
                ) from exc
            raise CritiqueClientError(
                f"Critique model call failed ({self.provider}): {error_text}"
            ) from exc

        return extract_response_text(response)


def build_critique_client(
    config: Optional[CritiqueConfig] = None,
    api_key: Optional[str] = None,
) -> BaseCritiqueClient:
    """
    Build the configured synthesis client.

    The provider is selected by CRITIQUE_PROVIDER (default: openai). Credentials
    are resolved from the environment here, but a missing credential does not fail
    until the first call, so building an engine never needs a key at all.
    """
    cfg = config or CritiqueConfig()
    provider = (cfg.provider or DEFAULT_CRITIQUE_PROVIDER).strip().lower()
    if provider not in _PROVIDER_KEY_ENV:
        raise CritiqueClientError(
            f"Unknown critique provider '{provider}'. Supported providers: "
            f"{', '.join(sorted(_PROVIDER_KEY_ENV))}."
        )
    return OpenAICompatibleCritiqueClient(
        api_key=api_key,
        base_url=cfg.base_url,
        provider=provider,
        max_output_tokens=cfg.max_output_tokens,
        request_timeout=cfg.request_timeout,
        json_response_format=cfg.json_response_format,
    )