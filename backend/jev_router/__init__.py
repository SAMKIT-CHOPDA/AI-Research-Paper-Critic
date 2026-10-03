"""
JEV Router Subsystem.

Provides:
- build_jev_input: transforms Document Profile into compact routing input
- validate_routing_state: validates routing decisions against the contract
- create_routing_state: helper to build a valid routing decision dictionary
- serialize_routing_state: JSON serializer for routing states
- generate_routing_explanation: produces deterministic, non-hallucinated explanations
- TypeSafeClient: HTTP client for TypeSafe API (models, System One)
- route_document: orchestrator for full routing workflow
- get_available_models: discover available models
"""

from backend.jev_router.config import (
    ROUTER_VERSION,
    ModelLevel,
    ALLOWED_MODEL_LEVELS,
    DecisionType,
    JEV_QUESTIONS,
    SYSTEM_ONE_QUESTIONS,
    DEFAULT_JEV_MODEL,
    DEFAULT_RAG_ENABLE_THRESHOLD,
    DEFAULT_VISION_ENABLE_THRESHOLD,
)
from backend.jev_router.schemas import (
    JevInputState,
    DocumentNatureBlock,
    StructureBlock,
    VisualContentBlock,
    MathematicalContentBlock,
    ResearchCharacteristicsBlock,
    AnalysisAgentState,
    OptionalAgentState,
    AgentConfidence,
    RoutingDecisionState,
)
from backend.jev_router.input_builder import (
    build_jev_input,
    JevInputValidationError,
)
from backend.jev_router.routing_state import (
    validate_routing_state,
    create_routing_state,
    serialize_routing_state,
    RoutingStateValidationError,
)
from backend.jev_router.explanations import (
    generate_routing_explanation,
)
from backend.jev_router.client import (
    TypeSafeClient,
    TypeSafeError,
    TypeSafeAuthError,
    TypeSafeRateLimitError,
    TypeSafeServerError,
    TypeSafeValidationError,
    TypeSafeTimeoutError,
)
from backend.jev_router.router import (
    RouterError,
    get_available_models,
    resolve_jev_model,
    parse_system_one_response,
    route_document,
)

__all__ = [
    "ROUTER_VERSION",
    "ModelLevel",
    "ALLOWED_MODEL_LEVELS",
    "DecisionType",
    "JEV_QUESTIONS",
    "SYSTEM_ONE_QUESTIONS",
    "DEFAULT_JEV_MODEL",
    "DEFAULT_RAG_ENABLE_THRESHOLD",
    "DEFAULT_VISION_ENABLE_THRESHOLD",
    "JevInputState",
    "DocumentNatureBlock",
    "StructureBlock",
    "VisualContentBlock",
    "MathematicalContentBlock",
    "ResearchCharacteristicsBlock",
    "AnalysisAgentState",
    "OptionalAgentState",
    "AgentConfidence",
    "RoutingDecisionState",
    "build_jev_input",
    "JevInputValidationError",
    "validate_routing_state",
    "create_routing_state",
    "serialize_routing_state",
    "RoutingStateValidationError",
    "generate_routing_explanation",
    "TypeSafeClient",
    "TypeSafeError",
    "TypeSafeAuthError",
    "TypeSafeRateLimitError",
    "TypeSafeServerError",
    "TypeSafeValidationError",
    "TypeSafeTimeoutError",
    "RouterError",
    "get_available_models",
    "resolve_jev_model",
    "parse_system_one_response",
    "route_document",
]

