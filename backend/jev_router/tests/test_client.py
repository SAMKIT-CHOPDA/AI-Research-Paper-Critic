"""
Offline unit tests for TypeSafeClient.
All network requests are mocked using pytest / unittest.mock.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import httpx

from backend.jev_router.client import (
    TypeSafeClient,
    TypeSafeError,
    TypeSafeAuthError,
    TypeSafeRateLimitError,
    TypeSafeServerError,
    TypeSafeValidationError,
    TypeSafeTimeoutError,
)


class TestTypeSafeClient:
    def test_missing_api_key_raises_auth_error(self):
        client = TypeSafeClient(api_key="")
        with pytest.raises(TypeSafeAuthError) as exc:
            client.get_models()
        assert "Missing TypeSafe API key" in str(exc.value)

    def test_api_key_never_in_str_or_repr(self):
        secret = "secret_key_12345"
        client = TypeSafeClient(api_key=secret)
        assert secret not in str(client)

    @patch("httpx.Client.get")
    def test_get_models_success(self, mock_get):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "jev-latest",
                    "description": "General-purpose system one model.",
                    "release_date": "2026-09-15",
                },
                {
                    "name": "jev-preview",
                    "description": "Preview model.",
                    "release_date": "2026-09-15",
                },
            ]
        }
        mock_get.return_value = mock_resp

        client = TypeSafeClient(api_key="valid_key")
        models = client.get_models()
        assert len(models) == 2
        assert models[0]["name"] == "jev-latest"

    @patch("httpx.Client.post")
    def test_system_one_success(self, mock_post):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "jev-1.13.0",
            "answers": {
                "rag_required": {"type": "noul", "noul": 0.95},
                "rag_level": {
                    "type": "choice",
                    "choice": "medium",
                    "probabilities": {"basic": 0.1, "medium": 0.85, "advanced": 0.05},
                    "confidence": 0.85,
                },
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        mock_post.return_value = mock_resp

        client = TypeSafeClient(api_key="valid_key")
        res = client.system_one(
            state={"test": "data"},
            questions={"rag_required": {"type": "noul", "instructions": "test"}},
        )
        assert res["model"] == "jev-1.13.0"
        assert res["answers"]["rag_required"]["noul"] == 0.95


    @patch("httpx.Client.get")
    def test_http_401_unauthorized(self, mock_get):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Invalid credentials"}
        mock_get.return_value = mock_resp

        client = TypeSafeClient(api_key="bad_key")
        with pytest.raises(TypeSafeAuthError) as exc:
            client.get_models()
        assert exc.value.status_code == 401
        assert "bad_key" not in str(exc.value)

    @patch("httpx.Client.post")
    def test_http_429_rate_limited(self, mock_post):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"detail": "Too Many Requests"}
        mock_post.return_value = mock_resp

        client = TypeSafeClient(api_key="valid_key")
        with pytest.raises(TypeSafeRateLimitError) as exc:
            client.system_one(state={}, questions={"q": {"type": "noul", "instructions": "test"}})
        assert exc.value.status_code == 429

    @patch("httpx.Client.post")
    def test_http_500_server_error(self, mock_post):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "Internal Server Error"}
        mock_post.return_value = mock_resp

        client = TypeSafeClient(api_key="valid_key")
        with pytest.raises(TypeSafeServerError) as exc:
            client.system_one(state={}, questions={"q": {"type": "noul", "instructions": "test"}})
        assert exc.value.status_code == 500

    @patch("httpx.Client.post")
    def test_timeout_handling(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        client = TypeSafeClient(api_key="valid_key", timeout=5.0)
        with pytest.raises(TypeSafeTimeoutError):
            client.system_one(state={}, questions={"q": {"type": "noul", "instructions": "test"}})

    @patch("httpx.Client.post")
    def test_malformed_json_response(self, mock_post):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        mock_post.return_value = mock_resp

        client = TypeSafeClient(api_key="valid_key")
        with pytest.raises(TypeSafeError) as exc:
            client.system_one(state={}, questions={"q": {"type": "noul", "instructions": "test"}})
        assert "Malformed JSON" in str(exc.value)

    def test_empty_questions_map_validation(self):
        client = TypeSafeClient(api_key="valid_key")
        with pytest.raises(TypeSafeValidationError):
            client.system_one(state={}, questions={})
