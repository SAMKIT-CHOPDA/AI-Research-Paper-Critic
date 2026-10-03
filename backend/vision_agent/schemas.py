"""
Schemas for the Vision Agent subsystem.

Pydantic models for:
- Extracted visual assets (figures / tables) with provenance metadata
- Asset extraction failures (failure isolation, never silently discarded)
- Structured visual analysis results separating observation, interpretation and
  uncertainty
- Path-level analysis failures and the final Vision Agent execution output

Accuracy contract
-----------------
Every schema keeps observations, interpretations, the paper's stated claims and
explicit uncertainties in *separate* fields so downstream stages can tell a
visible observation from a model interpretation, and can tell both apart from a
claim the paper itself asserts.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# The frozen Document Pre-Analyzer normalizes every bbox it emits into PyMuPDF
# top-left origin coordinates. The Vision Agent never reinterprets that space;
# it records the space it was handed so a mismatch stays auditable.
COORDINATE_SPACE_PYMUPDF_TOP_LEFT = "pymupdf_top_left"

CaptionStatus = Literal["consistent", "partially_consistent", "inconsistent", "unknown"]


class CaptionConsistency(BaseModel):
    """Whether the visual content agrees with its own caption/context."""

    status: CaptionStatus = Field(
        "unknown",
        description="consistent | partially_consistent | inconsistent | unknown",
    )
    explanation: str = Field("", description="Why the status was assigned")


class VisualAsset(BaseModel):
    """A visual asset extracted from a confirmed Document Profile element."""

    asset_id: str = Field(..., description="Canonical id, e.g. figure_001 / table_002")
    asset_type: str = Field(..., description="figure | table")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box in PyMuPDF top-left points, if available"
    )
    image_path: str = Field(..., description="Path of the extracted image on disk")
    source_confidence: float = Field(
        0.0, description="Confidence reported by the Document Pre-Analyzer"
    )
    source_id: Optional[str] = Field(
        None, description="Identifier inside the Document Profile (provenance)"
    )
    asset_subtype: Optional[str] = Field(
        None, description="e.g. raster_image / vector_graphic / ruling_lines"
    )
    label: Optional[str] = Field(
        None, description="Printed figure/table number, e.g. '1'"
    )
    caption: Optional[str] = Field(None, description="Caption text if detected")
    section: Optional[str] = Field(None, description="Section the asset belongs to")
    context_text: Optional[str] = Field(
        None, description="Bounded surrounding text sent to the model as context"
    )
    extraction_method: str = Field(
        "bbox_crop", description="bbox_crop | caption_band_crop | page_crop_fallback"
    )
    fallback_used: bool = Field(
        False, description="True when a precise crop was unavailable"
    )
    coordinate_space: str = Field(
        COORDINATE_SPACE_PYMUPDF_TOP_LEFT, description="Coordinate space of bbox"
    )
    image_width: int = Field(0, ge=0, description="Extracted image width in pixels")
    image_height: int = Field(0, ge=0, description="Extracted image height in pixels")
    image_format: str = Field("PNG", description="Encoded image format")
    image_bytes: int = Field(0, ge=0, description="Encoded image size in bytes")
    resized: bool = Field(
        False, description="True when preprocessing downscaled the image"
    )
    structured_table: Optional[Dict[str, Any]] = Field(
        None,
        description="Machine-readable table values read from the PDF (authoritative "
        "for exact numbers); None when unavailable or not a table",
    )


class AssetExtractionFailure(BaseModel):
    """A confirmed asset whose visual region could not be extracted."""

    asset_id: str
    asset_type: Optional[str] = None
    page_number: Optional[int] = None
    source_id: Optional[str] = None
    stage: str = Field("extraction", description="Stage where the failure occurred")
    error: str = Field(..., description="Failure reason (never contains secrets)")


class ExtractionReport(BaseModel):
    """Result of extracting confirmed visual assets from the PDF."""

    assets: List[VisualAsset] = Field(default_factory=list)
    failures: List[AssetExtractionFailure] = Field(default_factory=list)
    assets_considered: int = Field(0, ge=0)
    assets_skipped_over_limit: int = Field(0, ge=0)


class VisualAnalysisResult(BaseModel):
    """
    Structured semantic analysis of one visual asset.

    Observation / interpretation / claim separation is mandatory: the vision
    model output is never treated as proof of a claim.
    """

    asset_id: str
    asset_type: str
    page_number: int = Field(..., ge=1)
    section: Optional[str] = None
    caption: Optional[str] = None

    observation: str = Field(
        "", description="What is visibly present in the asset (no inference)"
    )
    interpretation: str = Field(
        "", description="What the visual plausibly represents (explicitly provisional)"
    )
    key_elements: List[str] = Field(
        default_factory=list, description="Axes, labels, legend, headers, variables"
    )
    reported_relationships: List[str] = Field(
        default_factory=list, description="Trends/relations visibly reported"
    )
    supports_claims: List[str] = Field(
        default_factory=list,
        description="Claims the visual appears to support (never treated as proof)",
    )
    uncertainties: List[str] = Field(
        default_factory=list, description="Ambiguities, limitations, unresolved items"
    )
    caption_consistency: CaptionConsistency = Field(default_factory=CaptionConsistency)

    model: Optional[str] = Field(None, description="Multimodal model identifier used")
    image_path: Optional[str] = None
    source_confidence: Optional[float] = None
    extraction_method: Optional[str] = None
    numeric_authority: str = Field(
        "vision_interpretation_only",
        description="vision_interpretation_only | document_profile_machine_readable",
    )
    structured_evidence: Optional[Dict[str, Any]] = Field(
        None, description="Machine-readable table values preserved as context"
    )


class VisualAnalysisFailure(BaseModel):
    """A structured failure for one asset; other assets keep being analyzed."""

    asset_id: str
    asset_type: Optional[str] = None
    page_number: Optional[int] = None
    stage: str = Field("analysis", description="Stage where the failure occurred")
    error: str = Field(..., description="Failure reason (never contains secrets)")
    model: Optional[str] = None


class VisionAgentResult(BaseModel):
    """Final JSON-serializable Vision Agent execution output."""

    agent: str = Field("vision", description="Agent name identifier")
    enabled: bool = Field(..., description="Whether vision ran per JEV routing state")
    level: Optional[str] = Field(
        None, description="JEV vision capability tier (basic, medium, advanced)"
    )
    model: Optional[str] = Field(None, description="Selected multimodal model id")
    provider: Optional[str] = Field(None, description="Vision provider identifier")

    assets_found: int = Field(0, ge=0, description="Confirmed visual assets found")
    assets_analyzed: int = Field(0, ge=0, description="Assets with a successful analysis")
    assets_failed: int = Field(0, ge=0, description="Assets that failed analysis")

    results: List[VisualAnalysisResult] = Field(default_factory=list)
    failures: List[VisualAnalysisFailure] = Field(default_factory=list)
    extraction_failures: List[AssetExtractionFailure] = Field(default_factory=list)

    equations_available: int = Field(
        0, ge=0, description="Confirmed equations kept as later-stage context only"
    )
    elapsed_time_ms: Optional[float] = Field(None, description="Elapsed milliseconds")
    message: Optional[str] = Field(None, description="Status or bypass message")