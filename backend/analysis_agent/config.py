"""
Configuration for the Analysis Agent subsystem (Phase 5).

Defines:
- Analysis model provider selection (abstracted behind llm_client)
- JEV capability level -> analysis model identifier mapping (environment based)
- Evidence context budgets (sections, inventory items, retrieved chunks, visual
  assets, equations, total rendered evidence)
- Structured-output policy (a JSON response format is opt-in, never assumed)
- Call constraints (max output tokens, request timeout)

Design notes
------------
* No model identifier is hard-coded. Every level reads its model from the
  environment (ANALYSIS_MODEL_BASIC / ANALYSIS_MODEL_MEDIUM /
  ANALYSIS_MODEL_ADVANCED) because model availability cannot be verified from
  inside the codebase.
* JEV levels are model *capability* levels, not "analysis quality scores".
  A missing model for a level is a hard configuration error: the agent never
  silently downgrades a tier unless ANALYSIS_ALLOW_LEVEL_FALLBACK is explicitly
  enabled as a project policy.
* Budgets exist so the Analysis Agent sends a *bounded* evidence set: it consumes
  the frozen Document Profile plus the RAG and Vision outputs and never renders
  the whole paper into the prompt.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class AnalysisConfigurationError(Exception):
    """Raised when the Analysis Agent is configured in an unusable way."""

    pass


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# Provider abstraction (the agent never talks to a provider directly)
DEFAULT_ANALYSIS_PROVIDER: str = _env_str("ANALYSIS_PROVIDER", "openai").lower()
# Optional OpenAI-compatible endpoint override (e.g. a gateway or proxy).
DEFAULT_ANALYSIS_BASE_URL: Optional[str] = _env_str("ANALYSIS_BASE_URL") or None

# JEV capability level -> analysis model identifier (environment configured only)
ANALYSIS_LEVEL_MODELS: Dict[str, Optional[str]] = {
    "basic": _env_str("ANALYSIS_MODEL_BASIC") or None,
    "medium": _env_str("ANALYSIS_MODEL_MEDIUM") or None,
    "advanced": _env_str("ANALYSIS_MODEL_ADVANCED") or None,
}

# Explicit project policy switch. False (default) => never silently downgrade.
ANALYSIS_ALLOW_LEVEL_FALLBACK: bool = _env_bool("ANALYSIS_ALLOW_LEVEL_FALLBACK", False)

# Structured output is requested only when the deployment explicitly wants it;
# the default contract is "return one JSON object as text", which every
# OpenAI-compatible endpoint supports.
ANALYSIS_JSON_RESPONSE_FORMAT: bool = _env_bool("ANALYSIS_JSON_RESPONSE_FORMAT", False)

# Evidence context budgets (cost vs. coverage, all environment tunable)
ANALYSIS_MAX_SECTIONS: int = _env_int("ANALYSIS_MAX_SECTIONS", 40)
ANALYSIS_MAX_SECTION_TITLE_CHARS: int = _env_int("ANALYSIS_MAX_SECTION_TITLE_CHARS", 160)
ANALYSIS_MAX_INVENTORY_ITEMS: int = _env_int("ANALYSIS_MAX_INVENTORY_ITEMS", 20)
ANALYSIS_MAX_CAPTION_CHARS: int = _env_int("ANALYSIS_MAX_CAPTION_CHARS", 400)
ANALYSIS_MAX_EQUATIONS: int = _env_int("ANALYSIS_MAX_EQUATIONS", 12)
ANALYSIS_MAX_EQUATION_CHARS: int = _env_int("ANALYSIS_MAX_EQUATION_CHARS", 240)
ANALYSIS_MAX_RAG_CHUNKS: int = _env_int("ANALYSIS_MAX_RAG_CHUNKS", 12)
ANALYSIS_MAX_CHUNK_CHARS: int = _env_int("ANALYSIS_MAX_CHUNK_CHARS", 1800)
ANALYSIS_MAX_VISUAL_ASSETS: int = _env_int("ANALYSIS_MAX_VISUAL_ASSETS", 12)
ANALYSIS_MAX_ASSET_FIELD_CHARS: int = _env_int("ANALYSIS_MAX_ASSET_FIELD_CHARS", 600)
# Total evidence budget (characters) for the assembled evidence bundle
ANALYSIS_MAX_CONTEXT_CHARS: int = _env_int("ANALYSIS_MAX_CONTEXT_CHARS", 24000)

# Analysis call constraints
ANALYSIS_MAX_OUTPUT_TOKENS: int = _env_int("ANALYSIS_MAX_OUTPUT_TOKENS", 6000)
ANALYSIS_REQUEST_TIMEOUT: float = _env_float("ANALYSIS_REQUEST_TIMEOUT", 180.0)

LEVEL_ORDER = ("basic", "medium", "advanced")

_LEVEL_ENV_NAMES = {
    "basic": "ANALYSIS_MODEL_BASIC",
    "medium": "ANALYSIS_MODEL_MEDIUM",
    "advanced": "ANALYSIS_MODEL_ADVANCED",
}


@dataclass
class AnalysisConfig:
    """Runtime configuration for one Analysis Agent execution (injectable)."""

    provider: str = DEFAULT_ANALYSIS_PROVIDER
    base_url: Optional[str] = DEFAULT_ANALYSIS_BASE_URL
    level_models: Dict[str, Optional[str]] = field(
        default_factory=lambda: dict(ANALYSIS_LEVEL_MODELS)
    )
    allow_level_fallback: bool = ANALYSIS_ALLOW_LEVEL_FALLBACK
    json_response_format: bool = ANALYSIS_JSON_RESPONSE_FORMAT

    max_sections: int = ANALYSIS_MAX_SECTIONS
    max_section_title_chars: int = ANALYSIS_MAX_SECTION_TITLE_CHARS
    max_inventory_items: int = ANALYSIS_MAX_INVENTORY_ITEMS
    max_caption_chars: int = ANALYSIS_MAX_CAPTION_CHARS
    max_equations: int = ANALYSIS_MAX_EQUATIONS
    max_equation_chars: int = ANALYSIS_MAX_EQUATION_CHARS
    max_rag_chunks: int = ANALYSIS_MAX_RAG_CHUNKS
    max_chunk_chars: int = ANALYSIS_MAX_CHUNK_CHARS
    max_visual_assets: int = ANALYSIS_MAX_VISUAL_ASSETS
    max_asset_field_chars: int = ANALYSIS_MAX_ASSET_FIELD_CHARS
    max_context_chars: int = ANALYSIS_MAX_CONTEXT_CHARS

    max_output_tokens: int = ANALYSIS_MAX_OUTPUT_TOKENS
    request_timeout: float = ANALYSIS_REQUEST_TIMEOUT


def normalize_level(level: object) -> Optional[str]:
    """
    Normalize a JEV capability level into a plain lowercase string (or None).

    The frozen router emits ``ModelLevel`` enum members inside its routing state
    dict, whose ``str()`` form is ``"ModelLevel.<NAME>"``; only ``.value`` yields
    the contract token ("basic" / "medium" / "advanced").
    """
    if level is None:
        return None
    raw_level = getattr(level, "value", level)
    normalized = str(raw_level).strip().lower()
    return normalized or None


def resolve_model_for_level(
    level: object,
    level_models: Optional[Dict[str, Optional[str]]] = None,
    allow_fallback: Optional[bool] = None,
) -> str:
    """
    Resolve the analysis model identifier for a JEV capability level.

    Raises AnalysisConfigurationError when the level has no configured model.
    A downgrade to a weaker tier only happens when the fallback policy is
    explicitly enabled; otherwise the failure is surfaced, never hidden.
    """
    mapping = level_models if level_models is not None else ANALYSIS_LEVEL_MODELS
    fallback_allowed = (
        ANALYSIS_ALLOW_LEVEL_FALLBACK if allow_fallback is None else bool(allow_fallback)
    )

    normalized = normalize_level(level)
    if normalized is None:
        raise AnalysisConfigurationError(
            "Analysis level is missing from the JEV routing state, so no analysis "
            "model can be selected."
        )
    if normalized not in LEVEL_ORDER:
        raise AnalysisConfigurationError(
            f"Unsupported JEV analysis level '{normalized}'. Expected one of: "
            f"{', '.join(LEVEL_ORDER)}."
        )

    configured = {
        name: (mapping.get(name) or "").strip() or None for name in LEVEL_ORDER
    }

    candidates = [normalized]
    if fallback_allowed:
        start = LEVEL_ORDER.index(normalized)
        candidates = list(reversed(LEVEL_ORDER[: start + 1]))

    for candidate in candidates:
        if configured.get(candidate):
            return configured[candidate]

    expected = _LEVEL_ENV_NAMES[normalized]
    if fallback_allowed:
        raise AnalysisConfigurationError(
            f"No analysis model is configured for level '{normalized}' (no weaker "
            f"tier configured either). Set {expected} (or ANALYSIS_MODEL_MEDIUM / "
            "ANALYSIS_MODEL_BASIC) in the environment."
        )
    raise AnalysisConfigurationError(
        f"No analysis model is configured for level '{normalized}'. Set {expected} in "
        "the environment. The Analysis Agent does not silently downgrade to another "
        "capability tier."
    )


