"""
Unit tests for JEV Router schemas and validation rules.
"""

import json
import pytest
from pydantic import ValidationError

from backend.jev_router.config import ModelLevel, ROUTER_VERSION, JEV_QUESTIONS, DecisionType
from backend.jev_router.schemas import (
    JevInputState,
    AnalysisAgentState,
    OptionalAgentState,
    AgentConfidence,
    RoutingDecisionState,
)


class TestSchemas:
    def test_question_contract_definitions(self):
        """Verify the conceptual question contracts and decision types."""
        assert "rag_enabled" in JEV_QUESTIONS
        assert JEV_QUESTIONS["rag_enabled"]["decision_type"] == DecisionType.NOUL.value
        assert "retrieval over the research paper's textual content" in JEV_QUESTIONS["rag_enabled"]["conceptual_question"]

        assert "vision_enabled" in JEV_QUESTIONS
        assert JEV_QUESTIONS["vision_enabled"]["decision_type"] == DecisionType.NOUL.value
        assert "visual information" in JEV_QUESTIONS["vision_enabled"]["conceptual_question"]

        for level_key in ("rag_level", "vision_level", "analysis_level"):
            assert JEV_QUESTIONS[level_key]["decision_type"] == DecisionType.CHOICE.value
            assert JEV_QUESTIONS[level_key]["options"] == ["basic", "medium", "advanced"]

    def test_analysis_agent_always_enabled(self):
        """Analysis agent must always be enabled."""
        state = AnalysisAgentState(enabled=True, level=ModelLevel.ADVANCED)
        assert state.enabled is True
        assert state.level == ModelLevel.ADVANCED

        # Analysis disabled must fail
        with pytest.raises(ValidationError) as exc:
            AnalysisAgentState(enabled=False, level=ModelLevel.ADVANCED)
        assert "analysis.enabled must always be True" in str(exc.value)

    def test_optional_agent_state(self):
        """Optional agent validation rules."""
        # Disabled with None level is valid
        disabled = OptionalAgentState(enabled=False, level=None)
        assert disabled.enabled is False
        assert disabled.level is None

        # Enabled with valid level is valid
        enabled = OptionalAgentState(enabled=True, level=ModelLevel.MEDIUM)
        assert enabled.enabled is True
        assert enabled.level == ModelLevel.MEDIUM

        # Enabled without level must fail
        with pytest.raises(ValidationError) as exc:
            OptionalAgentState(enabled=True, level=None)
        assert "Level must be specified" in str(exc.value)

    def test_invalid_model_level(self):
        """Invalid model level string fails validation."""
        with pytest.raises(ValidationError):
            AnalysisAgentState(enabled=True, level="super_advanced")

        with pytest.raises(ValidationError):
            OptionalAgentState(enabled=True, level="Luna")

    def test_routing_decision_state_valid(self):
        """Full valid routing decision state."""
        state_dict = {
            "router_version": "1.0",
            "analysis": {"enabled": True, "level": "advanced"},
            "rag": {"enabled": True, "level": "medium"},
            "vision": {"enabled": False, "level": None},
            "confidence": {"rag": 0.91, "vision": 0.97, "analysis": 0.94},
        }
        decision = RoutingDecisionState.model_validate(state_dict)
        assert decision.router_version == "1.0"
        assert decision.analysis.level == ModelLevel.ADVANCED
        assert decision.rag.enabled is True
        assert decision.vision.enabled is False
        assert decision.confidence.rag == 0.91

    def test_routing_decision_state_missing_version(self):
        """Routing decision must have router_version."""
        state_dict = {
            "router_version": "",
            "analysis": {"enabled": True, "level": "advanced"},
            "rag": {"enabled": False, "level": None},
            "vision": {"enabled": False, "level": None},
        }
        with pytest.raises(ValidationError) as exc:
            RoutingDecisionState.model_validate(state_dict)
        assert "router_version must not be empty" in str(exc.value)
