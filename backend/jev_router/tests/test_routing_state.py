"""
Unit tests for routing state validation and deterministic explanation generation.
"""

import json
import pytest

from backend.jev_router.config import ModelLevel
from backend.jev_router.routing_state import (
    validate_routing_state,
    create_routing_state,
    serialize_routing_state,
    RoutingStateValidationError,
)
from backend.jev_router.explanations import generate_routing_explanation


class TestRoutingState:
    def test_rag_disabled_with_level_none(self):
        """RAG disabled with level None is valid."""
        state = create_routing_state(
            analysis_level="advanced",
            rag_enabled=False,
            rag_level=None,
            vision_enabled=True,
            vision_level="medium",
        )
        assert state["rag"]["enabled"] is False
        assert state["rag"]["level"] is None

    def test_vision_disabled_with_level_none(self):
        """Vision disabled with level None is valid."""
        state = create_routing_state(
            analysis_level="basic",
            rag_enabled=True,
            rag_level="basic",
            vision_enabled=False,
            vision_level=None,
        )
        assert state["vision"]["enabled"] is False
        assert state["vision"]["level"] is None

    def test_analysis_disabled_must_fail_validation(self):
        """Analysis disabled must fail validation."""
        bad_state = {
            "router_version": "1.0",
            "analysis": {"enabled": False, "level": "advanced"},
            "rag": {"enabled": True, "level": "medium"},
            "vision": {"enabled": False, "level": None},
        }
        with pytest.raises(RoutingStateValidationError) as exc:
            validate_routing_state(bad_state)
        assert "analysis.enabled must always be True" in str(exc.value)

    def test_invalid_model_level_fails(self):
        """Invalid model level fails validation."""
        bad_state = {
            "router_version": "1.0",
            "analysis": {"enabled": True, "level": "ultra_smart"},
            "rag": {"enabled": False, "level": None},
            "vision": {"enabled": False, "level": None},
        }
        with pytest.raises(RoutingStateValidationError):
            validate_routing_state(bad_state)

    def test_enabled_agent_without_level_fails(self):
        """Enabled agent without level fails validation."""
        bad_state = {
            "router_version": "1.0",
            "analysis": {"enabled": True, "level": "advanced"},
            "rag": {"enabled": True, "level": None},
            "vision": {"enabled": False, "level": None},
        }
        with pytest.raises(RoutingStateValidationError):
            validate_routing_state(bad_state)

    def test_json_serialization(self):
        """JSON serialization produces valid parseable JSON."""
        state = create_routing_state(
            analysis_level=ModelLevel.ADVANCED,
            rag_enabled=True,
            rag_level=ModelLevel.MEDIUM,
            vision_enabled=True,
            vision_level=ModelLevel.ADVANCED,
            confidence={"rag": 0.91, "vision": 0.97, "analysis": 0.94},
        )
        json_str = serialize_routing_state(state)
        parsed = json.loads(json_str)

        assert parsed["router_version"] == "1.0"
        assert parsed["analysis"]["level"] == "advanced"
        assert parsed["rag"]["level"] == "medium"
        assert parsed["vision"]["level"] == "advanced"
        assert parsed["confidence"]["rag"] == 0.91

    def test_deterministic_explanation_generation(self):
        """Deterministic explanations are properly derived from input & decision."""
        doc_input = {
            "document": {"page_count": 11, "word_count": 4990, "document_nature": "born-digital"},
            "structure": {
                "section_count": 23,
                "max_section_depth": 3,
                "has_methodology": True,
                "has_experimental_results": True,
                "has_references": True,
            },
            "visual_content": {"figure_count": 2, "table_count": 3, "has_visual_content": True},
            "mathematical_content": {"equation_count": 5, "has_mathematical_content": True},
            "research_characteristics": {
                "has_methodology": True,
                "has_experimental_results": True,
                "has_tables": True,
                "has_references": True,
            },
        }

        decision = {
            "router_version": "1.0",
            "analysis": {"enabled": True, "level": "advanced"},
            "rag": {"enabled": True, "level": "medium"},
            "vision": {"enabled": True, "level": "advanced"},
        }

        explanations = generate_routing_explanation(doc_input, decision)
        assert len(explanations) == 3
        assert "Analysis Agent is enabled at 'advanced' tier" in explanations[0]
        assert "RAG is enabled for retrieval over the paper's textual content" in explanations[1]
        assert "Vision is enabled because the document contains 2 confirmed figure(s) and 3 confirmed table(s)" in explanations[2]

        # Test disabled RAG & Vision explanations
        decision_disabled = {
            "router_version": "1.0",
            "analysis": {"enabled": True, "level": "basic"},
            "rag": {"enabled": False, "level": None},
            "vision": {"enabled": False, "level": None},
        }
        explanations_disabled = generate_routing_explanation(doc_input, decision_disabled)
        assert "RAG is disabled as textual retrieval is not required" in explanations_disabled[1]
        assert "Vision is disabled" in explanations_disabled[2]
