"""
HTTP client for TypeSafe System One API (https://api.typesafe.ai).

Provides:
- Model discovery (GET /v1/models)
- Structured System One evaluation (POST /v1/systemone)
- Strict error handling without secret leakage
- Clean timeout and status validation
"""

import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from backend.jev_router.config import (
    TYPESAFE_API_BASE_URL,
    DEFAULT_JEV_MODEL,
    TYPESAFE_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class TypeSafeError(Exception):
    """Base exception for TypeSafe API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TypeSafeAuthError(TypeSafeError):
    """Raised when authentication fails (HTTP 401)."""
    pass


class TypeSafeRateLimitError(TypeSafeError):
    """Raised when rate limit is exceeded (HTTP 429)."""
    pass


class TypeSafeServerError(TypeSafeError):
    """Raised on 5xx server errors (HTTP 500, 529, etc.)."""
    pass


class TypeSafeValidationError(TypeSafeError):
    """Raised on request validation errors (HTTP 422)."""
    pass


class TypeSafeTimeoutError(TypeSafeError):
    """Raised when network request times out."""
    pass


class TypeSafeClient:
    """
    Lightweight, secure HTTP client for the TypeSafe AI API.
    Guarantees API keys are never printed, logged, or exposed in exceptions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        raw_key = api_key or ""
        self._api_key = raw_key.strip()
        self.base_url = (base_url or TYPESAFE_API_BASE_URL).rstrip("/")
        self.timeout = timeout or TYPESAFE_REQUEST_TIMEOUT

    def _get_headers(self) -> Dict[str, str]:
        """Construct headers without exposing secrets."""
        if not self._api_key:
            raise TypeSafeAuthError(
                "Missing TypeSafe API key. Set TYPESAFE_API_KEY environment variable or pass api_key."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Research-Paper-Critic/1.0",
        }

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Parse and translate error responses safely without leaking sensitive information."""
        status_code = response.status_code
        try:
            body = response.json()
        except Exception:
            body = response.text

        msg = f"TypeSafe API request failed with status {status_code}"

        if status_code == 401:
            raise TypeSafeAuthError(
                "Invalid or missing API key. Verify your TYPESAFE_API_KEY.",
                status_code=401,
                response_body=body,
            )
        elif status_code == 422:
            raise TypeSafeValidationError(
                f"TypeSafe validation failed (422): {body}",
                status_code=422,
                response_body=body,
            )
        elif status_code == 429:
            raise TypeSafeRateLimitError(
                "TypeSafe rate limit exceeded (429). Please back off and retry.",
                status_code=429,
                response_body=body,
            )
        elif status_code in (500, 502, 503, 504, 529):
            raise TypeSafeServerError(
                f"TypeSafe server error ({status_code}). Service temporarily unavailable or overloaded.",
                status_code=status_code,
                response_body=body,
            )
        else:
            raise TypeSafeError(
                f"{msg}: {body}",
                status_code=status_code,
                response_body=body,
            )

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Fetch available models from GET /v1/models.

        Returns list of model metadata dicts containing 'name', 'description', etc.
        """
        headers = self._get_headers()
        url = f"{self.base_url}/v1/models"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise TypeSafeTimeoutError(f"Request to {url} timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise TypeSafeError(f"Network error connecting to TypeSafe API: {exc}") from exc

        if response.status_code != 200:
            self._handle_error_response(response)

        try:
            data = response.json()
        except Exception as exc:
            raise TypeSafeError(f"Malformed JSON response from /v1/models: {exc}") from exc

        if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
            raise TypeSafeError("Invalid response structure from /v1/models: missing 'models' list")

        return data["models"]

    def system_one(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a structured decision request to POST /v1/systemone.

        Parameters:
        - state: The compact state object or text
        - questions: Dict mapping question_id -> question definition dict
        - model: Model ID or alias (defaults to DEFAULT_JEV_MODEL)

        Returns the raw response dictionary with 'answers', 'model', and 'usage'.
        """
        if not questions:
            raise TypeSafeValidationError("questions map cannot be empty.")

        headers = self._get_headers()
        url = f"{self.base_url}/v1/systemone"
        target_model = model or DEFAULT_JEV_MODEL

        payload = {
            "model": target_model,
            "state": state,
            "questions": questions,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise TypeSafeTimeoutError(f"Request to {url} timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise TypeSafeError(f"Network error connecting to TypeSafe API: {exc}") from exc

        if response.status_code != 200:
            self._handle_error_response(response)

        try:
            data = response.json()
        except Exception as exc:
            raise TypeSafeError(f"Malformed JSON response from /v1/systemone: {exc}") from exc

        if not isinstance(data, dict) or "answers" not in data or not isinstance(data["answers"], dict):
            raise TypeSafeError("Invalid response structure from /v1/systemone: missing 'answers' map")

        return data

