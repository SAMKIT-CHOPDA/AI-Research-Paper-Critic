"""
Offline unit tests for JEV Router orchestrator and response parser.
All tests run with mocked HTTP client responses or purely offline parsing.
"""

import pytest
from unittest.mock import MagicMock

from backend.jev_router.config import ModelLevel
from backend.jev_router.router import (
    RouterError,
    get_available_models,
    resolve_jev_model,
    parse_system_one_response,
    route_document,
)
from backend.jev_router.client import TypeSafeClient


def make_valid_system_one_response(
    rag_noul=0.92,
    rag_choice="medium",
    vision_noul=0.88,
    vision_choice="advanced",
    analysis_choice="advanced",
):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "rag_required": {
                "type": "noul",
                "noul": rag_noul,
            },
            "rag_level": {
                "type": "choice",
                "choice": rag_choice,
                "confidence": 0.82,
                "probabilities": {
                    "basic": 0.05,
                    "medium": 0.85,
                    "advanced": 0.10,
                },
            },
            "vision_required": {
                "type": "noul",
                "noul": vision_noul,
            },
            "vision_level": {
                "type": "choice",
                "choice": vision_choice,
                "confidence": 0.90,
                "probabilities": {
                    "basic": 0.02,
                    "medium": 0.08,
                    "advanced": 0.90,
                },
            },
            "analysis_level": {
                "type": "choice",
                "choice": analysis_choice,
                "confidence": 0.95,
                "probabilities": {
                    "basic": 0.01,
                    "medium": 0.09,
                    "advanced": 0.90,
                },
            },
        },
        "usage": {"input_tokens": 350, "output_tokens": 40},
    }


def make_benchmark_profile():
    return {
        "page_count": 11,
        "word_count": 4990,
        "sections": [{"level": 3 if i == 5 else 1} for i in range(23)],
        "figures": {"count": 2},
        "tables": {"count": 3},
        "equations": {"count": 5},
        "references": {"has_references": True, "count": 32},
        "content_characteristics": {
            "has_methodology": True,
            "has_experimental_results": True,
            "has_visual_content": True,
            "has_tables": True,
            "has_mathematical_content": True,
            "has_references": True,
            "document_nature": "born-digital",
        },
    }


class TestRouter:
    def test_successful_jev_response_all_agents_enabled(self):
        raw = make_valid_system_one_response(rag_noul=0.85, vision_noul=0.91)
        state = parse_system_one_response(raw, rag_threshold=0.70, vision_threshold=0.70)

        assert state["router_version"] == "1.0"
        assert state["analysis"]["enabled"] is True
        assert state["analysis"]["level"] == "advanced"
        assert state["analysis"]["probability"] == 0.90
        assert state["analysis"]["level_probabilities"]["advanced"] == 0.90

        assert state["rag"]["enabled"] is True
        assert state["rag"]["probability"] == 0.85
        assert state["rag"]["level"] == "medium"
        assert state["rag"]["level_probabilities"]["medium"] == 0.85

        assert state["vision"]["enabled"] is True
        assert state["vision"]["probability"] == 0.91
        assert state["vision"]["level"] == "advanced"
        assert state["vision"]["level_probabilities"]["advanced"] == 0.90

        assert state["confidence"]["rag"] == 0.82
        assert state["confidence"]["vision"] == 0.90
        assert state["confidence"]["analysis"] == 0.95

    def test_rag_disabled_under_threshold(self):
        raw = make_valid_system_one_response(rag_noul=0.45, vision_noul=0.85)
        state = parse_system_one_response(raw, rag_threshold=0.70, vision_threshold=0.70)

        assert state["rag"]["enabled"] is False
        assert state["rag"]["probability"] == 0.45
        assert state["rag"]["level"] is None
        assert state["rag"]["level_probabilities"]["medium"] == 0.85

        assert state["vision"]["enabled"] is True
        assert state["vision"]["level"] == "advanced"

    def test_vision_disabled_under_threshold(self):
        raw = make_valid_system_one_response(rag_noul=0.85, vision_noul=0.30)
        state = parse_system_one_response(raw, rag_threshold=0.70, vision_threshold=0.70)

        assert state["vision"]["enabled"] is False
        assert state["vision"]["probability"] == 0.30
        assert state["vision"]["level"] is None
        assert state["vision"]["level_probabilities"]["advanced"] == 0.90

        assert state["rag"]["enabled"] is True
        assert state["rag"]["level"] == "medium"

    def test_choice_distribution_preserved(self):
        raw = make_valid_system_one_response()
        state = parse_system_one_response(raw)

        expected_probs = {"basic": 0.05, "medium": 0.85, "advanced": 0.10}
        assert state["rag"]["level_probabilities"] == expected_probs

    def test_missing_question_fails(self):
        raw = make_valid_system_one_response()
        del raw["answers"]["vision_required"]

        with pytest.raises(RouterError) as exc:
            parse_system_one_response(raw)
        assert "vision_required" in str(exc.value)

    def test_invalid_probability_fails(self):
        raw = make_valid_system_one_response()
        raw["answers"]["rag_required"]["noul"] = 1.5

        with pytest.raises(RouterError) as exc:
            parse_system_one_response(raw)
        assert "out of range" in str(exc.value)

    def test_model_discovery_and_selection(self):
        mock_client = MagicMock(spec=TypeSafeClient)
        mock_client.get_models.return_value = [
            {"name": "jev-custom", "description": "Custom"},
            {"name": "jev-latest", "description": "Flagship"},
        ]

        selected = resolve_jev_model(client=mock_client)
        assert selected == "jev-latest"

    def test_route_document_end_to_end_mock(self):
        mock_client = MagicMock(spec=TypeSafeClient)
        mock_client.system_one.return_value = make_valid_system_one_response()

        profile = make_benchmark_profile()
        result = route_document(profile, client=mock_client)

        assert result["router_version"] == "1.0"
        assert result["analysis"]["enabled"] is True
        assert result["rag"]["enabled"] is True
        assert result["vision"]["enabled"] is True
        assert mock_client.system_one.called

