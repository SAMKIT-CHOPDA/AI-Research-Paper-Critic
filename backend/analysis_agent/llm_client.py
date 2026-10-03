"""
Analysis model abstraction (Phase 5).

The Analysis Agent never talks to a provider directly: it calls
``BaseAnalysisClient.complete(prompt, model)`` on an injected client. That keeps
the analyzer and the agent provider independent and makes the unit-test suite
offline (tests inject a fake client; no network is touched).

Credentials are read from environment variables only and are never logged,
printed, or embedded in error messages: provider exceptions pass through a masking
step before being re-raised as AnalysisClientError subclasses.

Model identifiers are deliberately *not* hard-coded: the caller (the agent, via
``resolve_model_for_level``) supplies the env-configured identifier, and a missing
configuration fails loudly instead of silently substituting a model.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dotenv import load_dotenv

from backend.analysis_agent.config import (
    ANALYSIS_MAX_OUTPUT_TOKENS,
    ANALYSIS_REQUEST_TIMEOUT,
    DEFAULT_ANALYSIS_BASE_URL,
    DEFAULT_ANALYSIS_PROVIDER,
    AnalysisConfig,
    AnalysisConfigurationError,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = ANALYSIS_MAX_OUTPUT_TOKENS
DEFAULT_REQUEST_TIMEOUT = ANALYSIS_REQUEST_TIMEOUT

# Provider -> environment variable names holding the credential. ANALYSIS_API_KEY
# always wins so a user can point the Analysis Agent at one specific credential.
_PROVIDER_KEY_ENV: Dict[str, tuple] = {
    "openai": ("ANALYSIS_API_KEY", "OPENAI_API_KEY"),
    "typesafe": ("ANALYSIS_API_KEY", "TYPESAFE_API_KEY"),
}


class AnalysisClientError(Exception):
    """Raised when an analysis model call fails."""

    pass


class AnalysisAuthError(AnalysisClientError):
    """Raised when the configured provider rejects or lacks credentials."""

    pass


class AnalysisModelUnavailableError(AnalysisClientError):
    """Raised when the selected model cannot be reached or does not exist."""

    pass


class AnalysisResponseError(AnalysisClientError):
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
    provider: str = DEFAULT_ANALYSIS_PROVIDER,
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
        (provider or "").strip().lower(), ("ANALYSIS_API_KEY",)
    )
    for name in key_env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


class BaseAnalysisClient(ABC):
    """Abstract analysis client: one prompt + one model -> one text response."""

    @abstractmethod
    def complete(self, prompt: str, model: str) -> str:
        """Run one analysis completion and return the raw model text."""
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
            raise AnalysisResponseError("Analysis response contained no choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
    except AnalysisResponseError:
        raise
    except Exception as exc:
        raise AnalysisResponseError(f"Malformed analysis response: {exc}") from exc

    if isinstance(content, list):
        # Some gateways return content parts instead of a plain string.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None or not str(content).strip():
        raise AnalysisResponseError("Analysis response contained no textual content")
    return str(content)


class OpenAICompatibleAnalysisClient(BaseAnalysisClient):
    """
    Analysis client for OpenAI-compatible chat-completions endpoints.

    Works with the OpenAI API and with any gateway implementing the same message
    shape; the endpoint is configurable via ANALYSIS_BASE_URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = DEFAULT_ANALYSIS_PROVIDER,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        json_response_format: bool = False,
    ):
        self.provider = (provider or DEFAULT_ANALYSIS_PROVIDER).strip().lower()
        self._api_key = resolve_api_key(self.provider, api_key)
        self.base_url = (base_url or DEFAULT_ANALYSIS_BASE_URL or "").strip() or None
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

        Raises AnalysisAuthError/AnalysisClientError when the client cannot be
        used, so a caller can report a clear configuration message before doing
        any context-building work.
        """
        self._get_client()

    def _get_client(self):
        """Lazy initialization so importing the module never needs a key."""
        if self._client is None:
            if not self._api_key:
                raise AnalysisAuthError(
                    f"Missing credentials for analysis provider '{self.provider}'. "
                    "Set ANALYSIS_API_KEY (or the provider-specific key such as "
                    "OPENAI_API_KEY) in the environment."
                )
            try:
                from openai import OpenAI

                kwargs = {"api_key": self._api_key, "timeout": self.request_timeout}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except AnalysisClientError:
                raise
            except Exception as exc:
                raise AnalysisClientError(
                    f"Failed to initialize the '{self.provider}' analysis client: "
                    f"{self._sanitize(exc)}"
                ) from exc
        return self._client

    def complete(self, prompt: str, model: str) -> str:
        """
        Send one bounded evidence prompt to the analysis model.

        The model identifier must be supplied by the caller (environment
        configured); this method refuses to guess a fallback model.
        """
        if not model or not str(model).strip():
            raise AnalysisConfigurationError(
                "No analysis model identifier was supplied to the analysis client. "
                "Configure ANALYSIS_MODEL_BASIC / ANALYSIS_MODEL_MEDIUM / "
                "ANALYSIS_MODEL_ADVANCED in the environment."
            )
        if not prompt or not prompt.strip():
            raise AnalysisClientError("Analysis prompt cannot be empty")

        client = self._get_client()
        request: Dict[str, object] = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
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
                raise AnalysisAuthError(
                    f"Analysis provider '{self.provider}' rejected the credentials. "
                    "Verify ANALYSIS_API_KEY (or the provider-specific key)."
                ) from exc
            if "model" in lowered and (
                "not found" in lowered
                or "does not exist" in lowered
                or "unsupported" in lowered
            ):
                raise AnalysisModelUnavailableError(
                    f"Model '{model}' is not available to analysis provider "
                    f"'{self.provider}'. Configure a model your account can access."
                ) from exc
            raise AnalysisClientError(
                f"Analysis model call failed ({self.provider}): {error_text}"
            ) from exc

        return extract_response_text(response)


def build_analysis_client(
    config: Optional[AnalysisConfig] = None,
    api_key: Optional[str] = None,
) -> BaseAnalysisClient:
    """
    Build the configured analysis client.

    The provider is selected by ANALYSIS_PROVIDER (default: openai). Credentials
    are resolved from the environment here, but a missing credential does not fail
    until the first call, so a disabled execution never needs a key at all.
    """
    cfg = config or AnalysisConfig()
    provider = (cfg.provider or DEFAULT_ANALYSIS_PROVIDER).strip().lower()
    if provider not in _PROVIDER_KEY_ENV:
        raise AnalysisClientError(
            f"Unknown analysis provider '{provider}'. Supported providers: "
            f"{', '.join(sorted(_PROVIDER_KEY_ENV))}."
        )
    return OpenAICompatibleAnalysisClient(
        api_key=api_key,
        base_url=cfg.base_url,
        provider=provider,
        max_output_tokens=cfg.max_output_tokens,
        request_timeout=cfg.request_timeout,
        json_response_format=cfg.json_response_format,
    )
