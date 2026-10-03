"""
Vision Agent subsystem for AI Research Paper Critic (Phase 4).

Exports:
- VisionAgent: high-level orchestrator (JEV-governed)
- extract_visual_assets / collect_confirmed_visual_assets: confirmed-asset extraction
- Image preprocessing helpers (prepare_image, PreparedImage, ...)
- Vision model abstraction (BaseVisionClient, OpenAICompatibleVisionClient, ...)
- Prompting / parsing helpers (build_prompt, analyze_asset, parse_visual_analysis)
- Schemas: VisualAsset, VisualAnalysisResult, VisualAnalysisFailure, VisionAgentResult
- Config: VisionConfig, VISION_LEVEL_MODELS, resolve_model_for_level, normalize_level
"""

from backend.vision_agent.config import (
    DEFAULT_ASSET_OUTPUT_DIR,
    LEVEL_ORDER,
    VISION_LEVEL_MODELS,
    VISION_MAX_IMAGE_DIMENSION,
    VISION_RENDER_DPI,
    VisionConfig,
    VisionConfigurationError,
    normalize_level,
    resolve_model_for_level,
)
from backend.vision_agent.schemas import (
    AssetExtractionFailure,
    CaptionConsistency,
    ExtractionReport,
    VisionAgentResult,
    VisualAnalysisFailure,
    VisualAnalysisResult,
    VisualAsset,
)
from backend.vision_agent.image_preprocessor import (
    ImagePreprocessingError,
    PreparedImage,
    prepare_image,
    to_data_url,
)
from backend.vision_agent.asset_extractor import (
    AssetExtractionError,
    ConfirmedAssetRef,
    collect_confirmed_visual_assets,
    extract_visual_assets,
    normalize_bbox,
)
from backend.vision_agent.vision_client import (
    BaseVisionClient,
    OpenAICompatibleVisionClient,
    VisionAuthError,
    VisionClientError,
    VisionModelUnavailableError,
    VisionResponseError,
    build_vision_client,
    mask_secret,
    resolve_api_key,
)
from backend.vision_agent.analyzer import (
    AnalysisError,
    analyze_asset,
    build_prompt,
    flag_overclaiming,
    parse_visual_analysis,
)
from backend.vision_agent.agent import (
    VisionAgent,
)

__all__ = [
    "DEFAULT_ASSET_OUTPUT_DIR",
    "LEVEL_ORDER",
    "VISION_LEVEL_MODELS",
    "VISION_MAX_IMAGE_DIMENSION",
    "VISION_RENDER_DPI",
    "VisionConfig",
    "VisionConfigurationError",
    "normalize_level",
    "resolve_model_for_level",
    "AssetExtractionFailure",
    "CaptionConsistency",
    "ExtractionReport",
    "VisionAgentResult",
    "VisualAnalysisFailure",
    "VisualAnalysisResult",
    "VisualAsset",
    "ImagePreprocessingError",
    "PreparedImage",
    "prepare_image",
    "to_data_url",
    "AssetExtractionError",
    "ConfirmedAssetRef",
    "collect_confirmed_visual_assets",
    "extract_visual_assets",
    "normalize_bbox",
    "BaseVisionClient",
    "OpenAICompatibleVisionClient",
    "VisionAuthError",
    "VisionClientError",
    "VisionModelUnavailableError",
    "VisionResponseError",
    "build_vision_client",
    "mask_secret",
    "resolve_api_key",
    "AnalysisError",
    "analyze_asset",
    "build_prompt",
    "flag_overclaiming",
    "parse_visual_analysis",
    "VisionAgent",
]