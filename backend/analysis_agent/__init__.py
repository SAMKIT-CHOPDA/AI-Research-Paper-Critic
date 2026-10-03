"""
Analysis Agent subsystem for AI Research Paper Critic (Phase 5).

Exports:
- AnalysisAgent: high-level orchestrator (JEV-governed, one model call)
- Evidence context assembly: build_evidence_bundle, ContextBuilderError
- Prompting: build_analysis_prompt, RULE_BLOCK, DIMENSION_FOCUS, OUTPUT_CONTRACT
- Analysis model abstraction: BaseAnalysisClient, OpenAICompatibleAnalysisClient
- Parsing / validation: parse_analysis, analyze_evidence, AnalysisError
- Schemas: dimension, finding, claim/evidence matrix, limitations, gaps, result
- Config: AnalysisConfig, ANALYSIS_LEVEL_MODELS, resolve_model_for_level
"""

from backend.analysis_agent.config import (
    ANALYSIS_ALLOW_LEVEL_FALLBACK,
    ANALYSIS_JSON_RESPONSE_FORMAT,
    ANALYSIS_LEVEL_MODELS,
    LEVEL_ORDER,
    AnalysisConfig,
    AnalysisConfigurationError,
    normalize_level,
    resolve_model_for_level,
)
from backend.analysis_agent.schemas import (
    AnalysisFailure,
    AnalysisPayload,
    AnalysisResult,
    AssessedPoint,
    ClaimEvidenceItem,
    ClaimEvidenceMatrix,
    DimensionAnalysis,
    EvidenceBundle,
    EvidenceGap,
    EvidenceInputs,
    EvidenceRef,
    Finding,
    InventoryItemRef,
    Limitations,
    OpenQuestion,
    OverallAssessment,
    ParsedAnalysis,
    RoutingSummary,
    SectionRef,
    TextEvidenceRef,
    VisualEvidenceRef,
)
from backend.analysis_agent.context_builder import (
    ContextBuilderError,
    build_evidence_bundle,
)
from backend.analysis_agent.prompts import (
    DIMENSION_FOCUS,
    OUTPUT_CONTRACT,
    RULE_BLOCK,
    build_analysis_prompt,
    build_context_block,
    render_gaps_block,
    render_source_index_block,
)
from backend.analysis_agent.llm_client import (
    AnalysisAuthError,
    AnalysisClientError,
    AnalysisModelUnavailableError,
    AnalysisResponseError,
    BaseAnalysisClient,
    OpenAICompatibleAnalysisClient,
    build_analysis_client,
    mask_secret,
    resolve_api_key,
)
from backend.analysis_agent.analyzer import (
    DIMENSION_KEYS,
    REQUIRED_PAYLOAD_KEYS,
    AnalysisError,
    analyze_evidence,
    count_downgraded_statuses,
    count_unverified_evidence_refs,
    detect_score_like_text,
    extract_json_payload,
    flag_overclaiming,
    parse_analysis,
)
from backend.analysis_agent.agent import AnalysisAgent

__all__ = [
    "ANALYSIS_ALLOW_LEVEL_FALLBACK",
    "ANALYSIS_JSON_RESPONSE_FORMAT",
    "ANALYSIS_LEVEL_MODELS",
    "LEVEL_ORDER",
    "AnalysisConfig",
    "AnalysisConfigurationError",
    "normalize_level",
    "resolve_model_for_level",
    "AnalysisFailure",
    "AnalysisPayload",
    "AnalysisResult",
    "AssessedPoint",
    "ClaimEvidenceItem",
    "ClaimEvidenceMatrix",
    "DimensionAnalysis",
    "EvidenceBundle",
    "EvidenceGap",
    "EvidenceInputs",
    "EvidenceRef",
    "Finding",
    "InventoryItemRef",
    "Limitations",
    "OpenQuestion",
    "OverallAssessment",
    "ParsedAnalysis",
    "RoutingSummary",
    "SectionRef",
    "TextEvidenceRef",
    "VisualEvidenceRef",
    "ContextBuilderError",
    "build_evidence_bundle",
    "DIMENSION_FOCUS",
    "OUTPUT_CONTRACT",
    "RULE_BLOCK",
    "build_analysis_prompt",
    "build_context_block",
    "render_gaps_block",
    "render_source_index_block",
    "AnalysisAuthError",
    "AnalysisClientError",
    "AnalysisModelUnavailableError",
    "AnalysisResponseError",
    "BaseAnalysisClient",
    "OpenAICompatibleAnalysisClient",
    "build_analysis_client",
    "mask_secret",
    "resolve_api_key",
    "DIMENSION_KEYS",
    "REQUIRED_PAYLOAD_KEYS",
    "AnalysisError",
    "analyze_evidence",
    "count_downgraded_statuses",
    "count_unverified_evidence_refs",
    "detect_score_like_text",
    "extract_json_payload",
    "flag_overclaiming",
    "parse_analysis",
    "AnalysisAgent",
]
