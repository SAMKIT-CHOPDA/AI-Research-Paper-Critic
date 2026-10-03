"""
Configuration for the Vision Agent subsystem.

Defines:
- Multimodal provider selection (provider is abstracted behind vision_client)
- JEV capability level -> multimodal model identifier mapping (environment based)
- Asset extraction constraints (render DPI, bbox padding, asset caps)
- Image preprocessing constraints (max dimension, encoding format/quality)
- Prompt context budget (surrounding text is limited, never the whole paper)

Design notes
------------
* No model identifier is hard-coded. Every level reads its model from the
  environment (VISION_MODEL_BASIC / VISION_MODEL_MEDIUM / VISION_MODEL_ADVANCED)
  because model availability cannot be verified from inside the codebase.
* JEV levels are model *capability* levels, not "analysis quality scores".
  A missing model for a level is a hard configuration error: the agent never
  silently downgrades a tier unless VISION_ALLOW_LEVEL_FALLBACK is explicitly
  enabled as a project policy.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class VisionConfigurationError(Exception):
    """Raised when the Vision Agent is configured in an unusable way."""
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


# Multimodal provider abstraction
DEFAULT_VISION_PROVIDER: str = _env_str("VISION_PROVIDER", "openai").lower()
# Optional OpenAI-compatible endpoint override (e.g. a gateway or proxy).
DEFAULT_VISION_BASE_URL: Optional[str] = _env_str("VISION_BASE_URL") or None

# JEV capability level -> multimodal model identifier (environment configured only)
VISION_LEVEL_MODELS: Dict[str, Optional[str]] = {
    "basic": _env_str("VISION_MODEL_BASIC") or None,
    "medium": _env_str("VISION_MODEL_MEDIUM") or None,
    "advanced": _env_str("VISION_MODEL_ADVANCED") or None,
}

# Explicit project policy switch. False (default) => never silently downgrade.
VISION_ALLOW_LEVEL_FALLBACK: bool = _env_bool("VISION_ALLOW_LEVEL_FALLBACK", False)

# Image constraints.
# Defaults balance API cost against research-figure legibility:
#   * 200 DPI render, so axis labels / table text stay readable when cropped
#   * 1600 px longest side, which keeps a half-page figure readable while staying
#     well inside typical multimodal input limits
VISION_RENDER_DPI: int = _env_int("VISION_RENDER_DPI", 200)
VISION_MAX_IMAGE_DIMENSION: int = _env_int("VISION_MAX_IMAGE_DIMENSION", 1600)
VISION_IMAGE_FORMAT: str = (_env_str("VISION_IMAGE_FORMAT", "PNG") or "PNG").upper()
VISION_JPEG_QUALITY: int = _env_int("VISION_JPEG_QUALITY", 90)

# Extraction constraints
VISION_BBOX_PADDING_PTS: float = _env_float("VISION_BBOX_PADDING_PTS", 4.0)
VISION_MAX_ASSETS: int = _env_int("VISION_MAX_ASSETS", 24)
# Candidate (unconfirmed) assets are never sent to the model by default: the
# frozen Document Pre-Analyzer is the source of truth for the visual inventory.
VISION_INCLUDE_CANDIDATE_ASSETS: bool = _env_bool("VISION_INCLUDE_CANDIDATE_ASSETS", False)

# Prompt context budget (a bounded neighbourhood of text, not the full paper)
VISION_CONTEXT_MAX_CHARS: int = _env_int("VISION_CONTEXT_MAX_CHARS", 700)

# Multimodal call constraints
VISION_MAX_OUTPUT_TOKENS: int = _env_int("VISION_MAX_OUTPUT_TOKENS", 1200)
VISION_REQUEST_TIMEOUT: float = _env_float("VISION_REQUEST_TIMEOUT", 120.0)

# Where extracted visual assets are written. An empty VISION_ASSET_OUTPUT_DIR
# falls back to the in-package default instead of an unusable empty path.
DEFAULT_ASSET_OUTPUT_DIR: str = _env_str("VISION_ASSET_OUTPUT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_extracted_assets"
)

# Ordered capability ladder used only for the explicitly-enabled fallback policy.
LEVEL_ORDER = ("basic", "medium", "advanced")

_LEVEL_ENV_NAMES = {
    "basic": "VISION_MODEL_BASIC",
    "medium": "VISION_MODEL_MEDIUM",
    "advanced": "VISION_MODEL_ADVANCED",
}


@dataclass
class VisionConfig:
    """Runtime configuration for a Vision Agent execution (injectable in tests)."""

    provider: str = DEFAULT_VISION_PROVIDER
    base_url: Optional[str] = DEFAULT_VISION_BASE_URL
    level_models: Dict[str, Optional[str]] = field(
        default_factory=lambda: dict(VISION_LEVEL_MODELS)
    )
    allow_level_fallback: bool = VISION_ALLOW_LEVEL_FALLBACK

    render_dpi: int = VISION_RENDER_DPI
    max_image_dimension: int = VISION_MAX_IMAGE_DIMENSION
    image_format: str = VISION_IMAGE_FORMAT
    jpeg_quality: int = VISION_JPEG_QUALITY

    bbox_padding_pts: float = VISION_BBOX_PADDING_PTS
    max_assets: int = VISION_MAX_ASSETS
    include_candidate_assets: bool = VISION_INCLUDE_CANDIDATE_ASSETS
    context_max_chars: int = VISION_CONTEXT_MAX_CHARS

    max_output_tokens: int = VISION_MAX_OUTPUT_TOKENS
    request_timeout: float = VISION_REQUEST_TIMEOUT
    output_dir: Optional[str] = None


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
    Resolve the multimodal model identifier for a JEV capability level.

    Raises VisionConfigurationError when the level has no configured model.
    A downgrade to a weaker tier only happens when the fallback policy is
    explicitly enabled; otherwise the failure is surfaced, never hidden.
    """
    mapping = level_models if level_models is not None else VISION_LEVEL_MODELS
    fallback_allowed = (
        VISION_ALLOW_LEVEL_FALLBACK if allow_fallback is None else bool(allow_fallback)
    )

    normalized = normalize_level(level)
    if normalized is None:
        raise VisionConfigurationError(
            "Vision level is missing from the JEV routing state, so no multimodal "
            "model can be selected."
        )
    if normalized not in LEVEL_ORDER:
        raise VisionConfigurationError(
            f"Unsupported JEV vision level '{normalized}'. Expected one of: "
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
        raise VisionConfigurationError(
            f"No multimodal model is configured for vision level '{normalized}' "
            f"(no weaker tier configured either). Set {expected} "
            "(or VISION_MODEL_MEDIUM / VISION_MODEL_BASIC) in the environment."
        )
    raise VisionConfigurationError(
        f"No multimodal model is configured for vision level '{normalized}'. "
        f"Set {expected} in the environment. The Vision Agent does not silently "
        "downgrade to another capability tier."
    )
