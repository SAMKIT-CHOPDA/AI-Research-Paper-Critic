"""
JEV Router orchestrator.

Coordinates:
1. Model discovery and selection (configurable via TYPESAFE_JEV_MODEL)
2. State preparation via build_jev_input(document_profile)
3. Structured System One evaluation with TypeSafe API
4. Normalization into the strict RoutingDecisionState contract:
   - Configurable decision thresholds for Noul questions
   - Preservation of full choice probability distributions and confidence
   - Analysis agent ALWAYS enabled
   - Selective preservation of RAG and Vision levels based on enabled status
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.jev_router.config import (
    DEFAULT_JEV_MODEL,
    DEFAULT_RAG_ENABLE_THRESHOLD,
    DEFAULT_VISION_ENABLE_THRESHOLD,
    SYSTEM_ONE_QUESTIONS,
    ModelLevel,
    ROUTER_VERSION,
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
from backend.jev_router.input_builder import build_jev_input
from backend.jev_router.schemas import RoutingDecisionState, JevInputState
from backend.jev_router.routing_state import validate_routing_state, create_routing_state

logger = logging.getLogger(__name__)


class RouterError(Exception):
    """Base exception for JEV Router orchestrator errors."""
    pass


def get_available_models(client: Optional[TypeSafeClient] = None) -> List[Dict[str, Any]]:
    """
    Fetch available models from TypeSafe API.

    Inspects GET /v1/models. Raises RouterError on failure.
    """
    api_client = client or TypeSafeClient()
    try:
        return api_client.get_models()
    except (TypeSafeError, Exception) as exc:
        raise RouterError(f"Failed to discover TypeSafe models: {exc}") from exc


def resolve_jev_model(client: Optional[TypeSafeClient] = None) -> str:
    """
    Select/configure the appropriate currently available JEV model.

    Rules:
    - If TYPESAFE_JEV_MODEL env var is provided, use it.
    - Otherwise, query available models and select the flagship alias ('jev-latest')
      or the first available model.
    """
    configured_model = os.getenv("TYPESAFE_JEV_MODEL")
    if configured_model and configured_model.strip():
        return configured_model.strip()



def parse_system_one_response(
    raw_response: Dict[str, Any],
    rag_threshold: float = DEFAULT_RAG_ENABLE_THRESHOLD,
    vision_threshold: float = DEFAULT_VISION_ENABLE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Normalize raw TypeSafe System One response into our internal Routing State.
    """
    if not isinstance(raw_response, dict):
        raise RouterError(f"Expected dict response from System One, got {type(raw_response).__name__}")

    answers = raw_response.get("answers")
    if not isinstance(answers, dict):
        raise RouterError("Missing or malformed 'answers' map in System One response")

    # 1. Evaluate RAG required (Noul)
    rag_ans = answers.get("rag_required")
    if not isinstance(rag_ans, dict) or rag_ans.get("type") != "noul":
        raise RouterError("Missing or invalid 'rag_required' Noul answer in response")

    rag_prob = rag_ans.get("noul")
    if rag_prob is None or not isinstance(rag_prob, (int, float)):
        raise RouterError("Missing or invalid 'noul' probability in 'rag_required' answer")
    rag_prob = float(rag_prob)
    if not (0.0 <= rag_prob <= 1.0):
        raise RouterError(f"'rag_required' probability out of range [0.0, 1.0]: {rag_prob}")

    rag_enabled = bool(rag_prob >= rag_threshold)

    # 2. Evaluate RAG level (Choice)
    rag_level_ans = answers.get("rag_level")
    if not isinstance(rag_level_ans, dict) or rag_level_ans.get("type") != "choice":
        raise RouterError("Missing or invalid 'rag_level' Choice answer in response")

    rag_choice = rag_level_ans.get("choice")
    rag_probs = rag_level_ans.get("probabilities")
    if not isinstance(rag_probs, dict):
        raise RouterError("Missing or invalid 'probabilities' in 'rag_level' Choice answer")

    for opt, prob in rag_probs.items():
        if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
            raise RouterError(f"Invalid probability for option '{opt}' in 'rag_level': {prob}")

    rag_level: Optional[str] = None
    if rag_enabled:
        if rag_choice not in (ModelLevel.BASIC.value, ModelLevel.MEDIUM.value, ModelLevel.ADVANCED.value):
            raise RouterError(f"Invalid model level choice returned for RAG: '{rag_choice}'")
        rag_level = rag_choice

    # 3. Evaluate Vision required (Noul)
    vision_ans = answers.get("vision_required")
    if not isinstance(vision_ans, dict) or vision_ans.get("type") != "noul":
        raise RouterError("Missing or invalid 'vision_required' Noul answer in response")

    vision_prob = vision_ans.get("noul")
    if vision_prob is None or not isinstance(vision_prob, (int, float)):
        raise RouterError("Missing or invalid 'noul' probability in 'vision_required' answer")
    vision_prob = float(vision_prob)
    if not (0.0 <= vision_prob <= 1.0):
        raise RouterError(f"'vision_required' probability out of range [0.0, 1.0]: {vision_prob}")

    vision_enabled = bool(vision_prob >= vision_threshold)

    # 4. Evaluate Vision level (Choice)
    vision_level_ans = answers.get("vision_level")
    if not isinstance(vision_level_ans, dict) or vision_level_ans.get("type") != "choice":
        raise RouterError("Missing or invalid 'vision_level' Choice answer in response")

    vision_choice = vision_level_ans.get("choice")
    vision_probs = vision_level_ans.get("probabilities")
    if not isinstance(vision_probs, dict):
        raise RouterError("Missing or invalid 'probabilities' in 'vision_level' Choice answer")

    for opt, prob in vision_probs.items():
        if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
            raise RouterError(f"Invalid probability for option '{opt}' in 'vision_level': {prob}")

    vision_level: Optional[str] = None
    if vision_enabled:
        if vision_choice not in (ModelLevel.BASIC.value, ModelLevel.MEDIUM.value, ModelLevel.ADVANCED.value):
            raise RouterError(f"Invalid model level choice returned for Vision: '{vision_choice}'")
        vision_level = vision_choice

    # 5. Evaluate Analysis level (Choice) - ALWAYS enabled
    analysis_level_ans = answers.get("analysis_level")
    if not isinstance(analysis_level_ans, dict) or analysis_level_ans.get("type") != "choice":
        raise RouterError("Missing or invalid 'analysis_level' Choice answer in response")

    analysis_choice = analysis_level_ans.get("choice")
    analysis_probs = analysis_level_ans.get("probabilities")
    if not isinstance(analysis_probs, dict):
        raise RouterError("Missing or invalid 'probabilities' in 'analysis_level' Choice answer")

    for opt, prob in analysis_probs.items():
        if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
            raise RouterError(f"Invalid probability for option '{opt}' in 'analysis_level': {prob}")

    if analysis_choice not in (ModelLevel.BASIC.value, ModelLevel.MEDIUM.value, ModelLevel.ADVANCED.value):
        raise RouterError(f"Invalid model level choice returned for Analysis: '{analysis_choice}'")

    analysis_confidence = analysis_level_ans.get("confidence")
    rag_confidence = rag_level_ans.get("confidence")
    vision_confidence = vision_level_ans.get("confidence")

    confidence_dict = None
    if any(c is not None for c in (rag_confidence, vision_confidence, analysis_confidence)):
        confidence_dict = {}
        if rag_confidence is not None:
            confidence_dict["rag"] = float(rag_confidence)
        if vision_confidence is not None:
            confidence_dict["vision"] = float(vision_confidence)
        if analysis_confidence is not None:
            confidence_dict["analysis"] = float(analysis_confidence)

    normalized_state = create_routing_state(
        analysis_level=analysis_choice,
        analysis_probability=analysis_probs.get(analysis_choice),
        analysis_level_probabilities={k: float(v) for k, v in analysis_probs.items()},
        rag_enabled=rag_enabled,
        rag_level=rag_level,
        rag_probability=rag_prob,
        rag_level_probabilities={k: float(v) for k, v in rag_probs.items()},
        vision_enabled=vision_enabled,
        vision_level=vision_level,
        vision_probability=vision_prob,
        vision_level_probabilities={k: float(v) for k, v in vision_probs.items()},
        confidence=confidence_dict,
        router_version=ROUTER_VERSION,
    )

    return normalized_state


def route_document(
    document_profile: Dict[str, Any],
    client: Optional[TypeSafeClient] = None,
    model: Optional[str] = None,
    rag_threshold: Optional[float] = None,
    vision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main entry point for JEV Router execution.

    Takes a full Document Profile, extracts compact JEV input state,
    queries TypeSafe System One API, and returns validated Routing Decision State.
    """
    # 1. Transform full profile into compact routing state
    compact_state = build_jev_input(document_profile)

    # 2. Resolve client and model
    api_client = client or TypeSafeClient()
    target_model = model or resolve_jev_model(client=api_client)

    # 3. Resolve thresholds
    rag_thresh = rag_threshold if rag_threshold is not None else DEFAULT_RAG_ENABLE_THRESHOLD
    vision_thresh = vision_threshold if vision_threshold is not None else DEFAULT_VISION_ENABLE_THRESHOLD

    # 4. Execute System One call
    try:
        raw_response = api_client.system_one(
            state=compact_state,
            questions=SYSTEM_ONE_QUESTIONS,
            model=target_model,
        )
    except (TypeSafeError, Exception) as exc:
        raise RouterError(f"JEV routing request failed: {exc}") from exc

    # 5. Parse and normalize response
    normalized_state = parse_system_one_response(
        raw_response=raw_response,
        rag_threshold=rag_thresh,
        vision_threshold=vision_thresh,
    )

    return normalized_state

    if not (0.0 <= vision_prob <= 1.0):
        raise RouterError(f"'vision_required' probability out of range [0.0, 1.0]: {vision_prob}")

    vision_enabled = bool(vision_prob >= vision_threshold)

    try:
        models = get_available_models(client=client)
        model_names = [m.get("name") for m in models if isinstance(m, dict) and "name" in m]
        if "jev-latest" in model_names:
            return "jev-latest"
        elif model_names:
            return model_names[0]
    except Exception as e:
        logger.warning(f"Could not discover models dynamically: {e}. Falling back to default '{DEFAULT_JEV_MODEL}'")

    return DEFAULT_JEV_MODEL
