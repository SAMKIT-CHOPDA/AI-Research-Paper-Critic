"""
Configuration for the Critique Engine (Phase 6).

Defines:
- Critique synthesis model provider selection (abstracted behind llm_client)
- JEV *analysis* capability level -> critique model identifier mapping
  (environment based; the Critique Engine introduces no routing decision)
- Synthesis context budgets (retrieved chunks, visual assets, findings per
  dimension, matrix items, total rendered context)
- Structured-output policy (a JSON response format is opt-in, never assumed)
- Call constraints (max output tokens, request timeout)

Design notes
------------
* No model identifier is hard-coded. Every level reads its model from the
  environment (CRITIQUE_MODEL_BASIC / CRITIQUE_MODEL_MEDIUM /
  CRITIQUE_MODEL_ADVANCED), because model availability cannot be verified from
  inside the codebase.
* The Critique Engine deliberately reuses the *analysis* level chosen by the
  frozen JEV router (``routing_state["analysis"]["level"]``). It never performs a
  second capability decision; it only maps an already-made decision onto a
  synthesis model. A single-model deployment is possible by configuring the same
  identifier for all three levels (or CRITIQUE_MODEL), which is an explicit
  configuration choice rather than a hidden fallback.
* A missing model for a level is a hard configuration error unless
  CRITIQUE_ALLOW_LEVEL_FALLBACK is explicitly enabled as project policy.
* Budgets keep the synthesis prompt bounded: the Critique Engine renders
  *structured findings*, never the raw paper.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class CritiqueConfigurationError(Exception):
    """Raised when the Critique Engine is configured in an unusable way."""

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


# Provider abstraction (the engine never talks to a provider directly)
DEFAULT_CRITIQUE_PROVIDER: str = _env_str("CRITIQUE_PROVIDER", "openai").lower()
# Optional OpenAI-compatible endpoint override (e.g. a gateway or proxy).
DEFAULT_CRITIQUE_BASE_URL: Optional[str] = _env_str("CRITIQUE_BASE_URL") or None

# Explicit single-model option. When set it wins for every JEV analysis level and
# the per-level mapping is ignored. This is a deployment decision, not a fallback.
CRITIQUE_SINGLE_MODEL: Optional[str] = _env_str("CRITIQUE_MODEL") or None

# JEV analysis capability level -> critique synthesis model (environment only)
CRITIQUE_LEVEL_MODELS: Dict[str, Optional[str]] = {
    "basic": _env_str("CRITIQUE_MODEL_BASIC") or None,
    "medium": _env_str("CRITIQUE_MODEL_MEDIUM") or None,
    "advanced": _env_str("CRITIQUE_MODEL_ADVANCED") or None,
}

# Explicit project policy switch. False (default) => never silently downgrade.
CRITIQUE_ALLOW_LEVEL_FALLBACK: bool = _env_bool(
    "CRITIQUE_ALLOW_LEVEL_FALLBACK", False
)

# Structured output is requested only when the deployment explicitly wants it;
# the default contract is "return one JSON object as text".
CRITIQUE_JSON_RESPONSE_FORMAT: bool = _env_bool(
    "CRITIQUE_JSON_RESPONSE_FORMAT", False
)

# Synthesis context budgets (cost vs. coverage, all environment tunable)
CRITIQUE_MAX_RAG_CHUNKS: int = _env_int("CRITIQUE_MAX_RAG_CHUNKS", 8)
CRITIQUE_MAX_CHUNK_CHARS: int = _env_int("CRITIQUE_MAX_CHUNK_CHARS", 900)
CRITIQUE_MAX_VISUAL_ASSETS: int = _env_int("CRITIQUE_MAX_VISUAL_ASSETS", 8)
CRITIQUE_MAX_ASSET_FIELD_CHARS: int = _env_int(
    "CRITIQUE_MAX_ASSET_FIELD_CHARS", 400
)
CRITIQUE_MAX_FINDINGS_PER_DIMENSION: int = _env_int(
    "CRITIQUE_MAX_FINDINGS_PER_DIMENSION", 6
)
CRITIQUE_MAX_STATEMENT_CHARS: int = _env_int("CRITIQUE_MAX_STATEMENT_CHARS", 700)
CRITIQUE_MAX_DIMENSIONS: int = _env_int("CRITIQUE_MAX_DIMENSIONS", 12)
CRITIQUE_MAX_CLAIMS: int = _env_int("CRITIQUE_MAX_CLAIMS", 12)
CRITIQUE_MAX_GAPS: int = _env_int("CRITIQUE_MAX_GAPS", 15)
CRITIQUE_MAX_OPEN_QUESTIONS: int = _env_int("CRITIQUE_MAX_OPEN_QUESTIONS", 10)
CRITIQUE_MAX_SECTIONS: int = _env_int("CRITIQUE_MAX_SECTIONS", 25)
CRITIQUE_MAX_SECTION_CONTENT_CHARS: int = _env_int(
    "CRITIQUE_MAX_SECTION_CONTENT_CHARS", 4000
)
CRITIQUE_MAX_CONTEXT_CHARS: int = _env_int("CRITIQUE_MAX_CONTEXT_CHARS", 60000)

# Call constraints
CRITIQUE_MAX_OUTPUT_TOKENS: int = _env_int("CRITIQUE_MAX_OUTPUT_TOKENS", 6000)
CRITIQUE_REQUEST_TIMEOUT: float = _env_float("CRITIQUE_REQUEST_TIMEOUT", 120.0)

# Validation policy switch: when True (default) a supported-but-missing section
# fails validation; when False the omission is recorded as a warning instead.
# Unsupported sections are always dropped rather than forced empty.
CRITIQUE_REQUIRE_SUPPORTED_SECTIONS: bool = _env_bool(
    "CRITIQUE_REQUIRE_SUPPORTED_SECTIONS", True
)

LEVEL_ORDER = ("basic", "medium", "advanced")

_LEVEL_ENV_NAMES = {
    "basic": "CRITIQUE_MODEL_BASIC",
    "medium": "CRITIQUE_MODEL_MEDIUM",
    "advanced": "CRITIQUE_MODEL_ADVANCED",
}


@dataclass
class CritiqueConfig:
    """Runtime configuration for one Critique Engine execution (injectable)."""

    provider: str = DEFAULT_CRITIQUE_PROVIDER
    base_url: Optional[str] = DEFAULT_CRITIQUE_BASE_URL
    single_model: Optional[str] = CRITIQUE_SINGLE_MODEL
    level_models: Dict[str, Optional[str]] = field(
        default_factory=lambda: dict(CRITIQUE_LEVEL_MODELS)
    )
    allow_level_fallback: bool = CRITIQUE_ALLOW_LEVEL_FALLBACK
    json_response_format: bool = CRITIQUE_JSON_RESPONSE_FORMAT

    max_rag_chunks: int = CRITIQUE_MAX_RAG_CHUNKS
    max_chunk_chars: int = CRITIQUE_MAX_CHUNK_CHARS
    max_visual_assets: int = CRITIQUE_MAX_VISUAL_ASSETS
    max_asset_field_chars: int = CRITIQUE_MAX_ASSET_FIELD_CHARS
    max_findings_per_dimension: int = CRITIQUE_MAX_FINDINGS_PER_DIMENSION
    max_statement_chars: int = CRITIQUE_MAX_STATEMENT_CHARS
    max_dimensions: int = CRITIQUE_MAX_DIMENSIONS
    max_claims: int = CRITIQUE_MAX_CLAIMS
    max_gaps: int = CRITIQUE_MAX_GAPS
    max_open_questions: int = CRITIQUE_MAX_OPEN_QUESTIONS
    max_sections: int = CRITIQUE_MAX_SECTIONS
    max_section_content_chars: int = CRITIQUE_MAX_SECTION_CONTENT_CHARS
    max_context_chars: int = CRITIQUE_MAX_CONTEXT_CHARS

    max_output_tokens: int = CRITIQUE_MAX_OUTPUT_TOKENS
    request_timeout: float = CRITIQUE_REQUEST_TIMEOUT

    require_supported_sections: bool = CRITIQUE_REQUIRE_SUPPORTED_SECTIONS


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
    single_model: Optional[str] = None,
) -> str:
    """
    Resolve the synthesis model identifier for a JEV *analysis* capability level.

    Order of precedence:
      1. an explicitly configured single synthesis model (deployment decision),
      2. the model configured for the exact level,
      3. a weaker level only when the fallback policy is explicitly enabled.

    Raises CritiqueConfigurationError when nothing is configured; the engine never
    silently substitutes a model it was not told to use.
    """
    explicit_single = (single_model or "").strip()
    if explicit_single:
        return explicit_single

    mapping = level_models if level_models is not None else CRITIQUE_LEVEL_MODELS
    fallback_allowed = (
        CRITIQUE_ALLOW_LEVEL_FALLBACK
        if allow_fallback is None
        else bool(allow_fallback)
    )

    normalized = normalize_level(level)
    if normalized is None:
        raise CritiqueConfigurationError(
            "No analysis level is present in the JEV routing state, so no critique "
            "synthesis model can be selected."
        )
    if normalized not in LEVEL_ORDER:
        raise CritiqueConfigurationError(
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
    raise CritiqueConfigurationError(
        f"No critique synthesis model configured for JEV analysis level "
        f"'{normalized}'. Set {expected} in the environment (or set CRITIQUE_MODEL "
        "to use one model for every level)."
    )