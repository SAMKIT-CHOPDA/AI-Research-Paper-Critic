"""
Routing decision state management and validation.

Enforces JEV routing contract invariants:
- router_version must exist and match contract.
- analysis.enabled must always be True.
- analysis.level must be basic, medium, or advanced.
- If rag.enabled is False, rag.level may be None.
- If rag.enabled is True, rag.level must be basic, medium, or advanced.
- If vision.enabled is False, vision.level may be None.
- If vision.enabled is True, vision.level must be basic, medium, or advanced.
- Confidence is optional and never invented locally.
"""

import json
from typing import Any, Dict, Optional, Union
from pydantic import ValidationError

from backend.jev_router.config import ModelLevel, ROUTER_VERSION
from backend.jev_router.schemas import RoutingDecisionState


class RoutingStateValidationError(ValueError):
    """Raised when a routing decision fails contract validation."""
    pass


def validate_routing_state(data: Union[Dict[str, Any], RoutingDecisionState]) -> RoutingDecisionState:
    """
    Validate that a given dictionary or object adheres strictly to the
    JEV routing decision contract.

    Returns the validated RoutingDecisionState.
    Raises RoutingStateValidationError if any rule is violated.
    """
    if isinstance(data, RoutingDecisionState):
        return data

    if not isinstance(data, dict):
        raise RoutingStateValidationError(f"Expected dict or RoutingDecisionState, got {type(data).__name__}")

    try:
        return RoutingDecisionState.model_validate(data)
    except ValidationError as e:
        raise RoutingStateValidationError(f"Invalid routing decision state: {e}") from e


def create_routing_state(
    analysis_level: Union[str, ModelLevel],
    rag_enabled: bool,
    rag_level: Optional[Union[str, ModelLevel]] = None,
    vision_enabled: bool = False,
    vision_level: Optional[Union[str, ModelLevel]] = None,
    confidence: Optional[Dict[str, float]] = None,
    router_version: str = ROUTER_VERSION,
    analysis_probability: Optional[float] = None,
    analysis_level_probabilities: Optional[Dict[str, float]] = None,
    rag_probability: Optional[float] = None,
    rag_level_probabilities: Optional[Dict[str, float]] = None,
    vision_probability: Optional[float] = None,
    vision_level_probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Helper to construct and validate a routing decision dictionary.
    Preserves probability and level_probabilities distributions when provided.
    """
    payload = {
        "router_version": router_version,
        "analysis": {
            "enabled": True,
            "level": analysis_level,
            "probability": analysis_probability,
            "level_probabilities": analysis_level_probabilities,
        },
        "rag": {
            "enabled": rag_enabled,
            "level": rag_level,
            "probability": rag_probability,
            "level_probabilities": rag_level_probabilities,
        },
        "vision": {
            "enabled": vision_enabled,
            "level": vision_level,
            "probability": vision_probability,
            "level_probabilities": vision_level_probabilities,
        },
    }
    if confidence is not None:
        payload["confidence"] = confidence

    validated = validate_routing_state(payload)
    return validated.model_dump()


def serialize_routing_state(state: Union[Dict[str, Any], RoutingDecisionState], indent: Optional[int] = 2) -> str:
    """Serialize a validated routing state to JSON formatted string."""
    validated = validate_routing_state(state)
    return validated.model_dump_json(indent=indent)
