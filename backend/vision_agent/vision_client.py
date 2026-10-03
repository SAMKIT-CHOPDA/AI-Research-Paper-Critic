"""
Multimodal vision model abstraction.

The Vision Agent never talks to a provider directly: it calls
``BaseVisionClient.analyze_image(image, prompt, model)`` on an injected client.
That keeps the analyzer and the agent provider-independent and makes the whole
unit-test suite offline (tests inject a fake client; no network is touched).

Credentials are read from environment variables only and are never logged,
printed, or embedded in error messages: provider exceptions pass through a
masking step before being re-raised as VisionClientError subclasses.

Model identifiers are deliberately *not* hard-coded: the caller (the agent, via
``resolve_model_for_level``) supplies the env-configured identifier, and a
missing configuration fails loudly instead of silently substituting a model.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

from dotenv import load_dotenv

from backend.vision_agent.config import (
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_PROVIDER,
    VisionConfig,
    VisionConfigurationError,
)
from backend.vision_agent.image_preprocessor import to_data_url

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 1200
DEFAULT_REQUEST_TIMEOUT = 120.0

# Provider -> environment variable names holding the credential. VISION_API_KEY
# always wins so a user can point the Vision Agent at one specific credential.
_PROVIDER_KEY_ENV: Dict[str, tuple] = {
    "openai": ("VISION_API_KEY", "OPENAI_API_KEY"),
    "typesafe": ("VISION_API_KEY", "TYPESAFE_API_KEY"),
}


class VisionClientError(Exception):
    """Raised when a multimodal vision call fails."""
    pass


class VisionAuthError(VisionClientError):
    """Raised when the configured provider rejects or lacks credentials."""
    pass


class VisionModelUnavailableError(VisionClientError):
    """Raised when the selected model cannot be reached or does not exist."""
    pass


class VisionResponseError(VisionClientError):
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
    provider: str = DEFAULT_VISION_PROVIDER,
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
        (provider or "").strip().lower(), ("VISION_API_KEY",)
    )
    for name in key_env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


class BaseVisionClient(ABC):
    """Abstract multimodal client: one image + one prompt -> one text response."""

    @abstractmethod
    def analyze_image(
        self,
        image: Union[str, Path, bytes, bytearray],
        prompt: str,
        model: str,
    ) -> str:
        """Analyze an image with a focused prompt and return the raw model text."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier (never contains credentials)."""
        return type(self).__name__


def extract_response_text(response: object) -> str:
    """
    Pull the assistant text out of a chat-completions response.

    Kept separate from the HTTP call so malformed-response handling can be
    tested offline.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            raise VisionResponseError("Vision response contained no choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
    except VisionResponseError:
        raise
    except Exception as exc:
        raise VisionResponseError(f"Malformed vision response: {exc}") from exc

    if isinstance(content, list):
        # Some gateways return content parts instead of a plain string.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None or not str(content).strip():
        raise VisionResponseError("Vision response contained no textual content")
    return str(content)


class OpenAICompatibleVisionClient(BaseVisionClient):
    """
    Vision client for OpenAI-compatible chat-completions endpoints.

    Works with the OpenAI API and with any gateway that implements the same
    multimodal message shape; the endpoint is configurable via VISION_BASE_URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = DEFAULT_VISION_PROVIDER,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.provider = (provider or DEFAULT_VISION_PROVIDER).strip().lower()
        self._api_key = resolve_api_key(self.provider, api_key)
        self.base_url = (base_url or DEFAULT_VISION_BASE_URL or "").strip() or None
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.request_timeout = float(request_timeout)
        self._client = None

    @property
    def provider_name(self) -> str:
        return self.provider

    def _sanitize(self, message: object) -> str:
        return mask_secret(str(message), self._api_key)

    def ensure_configured(self) -> None:
        """
        Validate credentials and endpoint configuration without a model call.

        Raises VisionAuthError/VisionClientError when the client cannot be used,
        so a caller can report a clear configuration message before doing any
        extraction work.
        """
        self._get_client()

    def _get_client(self):
        """Lazy initialization so importing the module never needs a key."""
        if self._client is None:
            if not self._api_key:
                raise VisionAuthError(
                    f"Missing credentials for vision provider '{self.provider}'. "
                    "Set VISION_API_KEY (or the provider-specific key such as "
                    "OPENAI_API_KEY) in the environment."
                )
            try:
                from openai import OpenAI

                kwargs = {"api_key": self._api_key, "timeout": self.request_timeout}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except VisionClientError:
                raise
            except Exception as exc:
                raise VisionClientError(
                    f"Failed to initialize the '{self.provider}' vision client: "
                    f"{self._sanitize(exc)}"
                ) from exc
        return self._client

    @staticmethod
    def _image_to_data_url(image: Union[str, Path, bytes, bytearray]) -> str:
        """Normalize any accepted image input into a base64 data URL."""
        if isinstance(image, (bytes, bytearray)):
            return to_data_url(bytes(image), None)
        path = Path(image)
        if not path.exists():
            raise VisionClientError(f"Image file not found: {path}")
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        try:
            return to_data_url(path.read_bytes(), mime)
        except OSError as exc:
            raise VisionClientError(f"Unable to read image file: {exc}") from exc

    def analyze_image(
        self,
        image: Union[str, Path, bytes, bytearray],
        prompt: str,
        model: str,
    ) -> str:
        """
        Send one image plus a focused prompt to the multimodal model.

        The model identifier must be supplied by the caller (environment
        configured); this method refuses to guess a fallback model.
        """
        if not model or not str(model).strip():
            raise VisionConfigurationError(
                "No multimodal model identifier was supplied to the vision client. "
                "Configure VISION_MODEL_BASIC / VISION_MODEL_MEDIUM / "
                "VISION_MODEL_ADVANCED in the environment."
            )
        if not prompt or not prompt.strip():
            raise VisionClientError("Vision prompt cannot be empty")

        client = self._get_client()
        data_url = self._image_to_data_url(image)

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
        except Exception as exc:
            error_text = self._sanitize(exc)
            lowered = error_text.lower()
            if (
                "401" in lowered
                or "invalid api key" in lowered
                or "unauthorized" in lowered
                or "authentication" in lowered
            ):
                raise VisionAuthError(
                    f"Vision provider '{self.provider}' rejected the credentials. "
                    "Verify VISION_API_KEY (or the provider-specific key)."
                ) from exc
            if "model" in lowered and (
                "not found" in lowered
                or "does not exist" in lowered
                or "unsupported" in lowered
            ):
                raise VisionModelUnavailableError(
                    f"Model '{model}' is not available to vision provider "
                    f"'{self.provider}'. Configure a model your account can access."
                ) from exc
            raise VisionClientError(
                f"Vision model call failed ({self.provider}): {error_text}"
            ) from exc

        return extract_response_text(response)


def build_vision_client(
    config: Optional[VisionConfig] = None,
    api_key: Optional[str] = None,
) -> BaseVisionClient:
    """
    Build the configured vision client.

    The provider is selected by VISION_PROVIDER (default: openai). Credentials
    are resolved from the environment here, but a missing credential does not
    fail until the first call, so a disabled or dry-run execution never needs a
    key at all.
    """
    cfg = config or VisionConfig()
    provider = (cfg.provider or DEFAULT_VISION_PROVIDER).strip().lower()
    if provider not in _PROVIDER_KEY_ENV:
        raise VisionClientError(
            f"Unknown vision provider '{provider}'. Supported providers: "
            f"{', '.join(sorted(_PROVIDER_KEY_ENV))}."
        )
    return OpenAICompatibleVisionClient(
        api_key=api_key,
        base_url=cfg.base_url,
        provider=provider,
        max_output_tokens=cfg.max_output_tokens,
        request_timeout=cfg.request_timeout,
    )