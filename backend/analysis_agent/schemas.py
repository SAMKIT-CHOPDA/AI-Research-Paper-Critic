"""
Schemas for the Analysis Agent subsystem (Phase 5).

Pydantic models for:
- Evidence provenance (source_type / source_id / pages) with an explicit
  verification flag for every reference the analysis model emits
- Findings and per-dimension analyses (research problem, contribution,
  methodology, data, baselines, metrics, results, reproducibility,
  internal consistency)
- A claim -> evidence matrix separating supporting from contradicting evidence
- Strengths, weaknesses, limitations (author stated vs. analyst identified)
- Open questions, evidence gaps and an overall assessment
- The bounded evidence context handed to the model (built from the frozen
  Document Profile plus the RAG and Vision outputs)
- The final JSON-serializable Analysis Agent execution envelope

Accuracy contract
-----------------
Every substantive statement carries an ``evidence_status`` and (ideally) concrete
``evidence_refs``. Missing provenance is never upgraded: a statement marked
``supported`` without references is downgraded by the analyzer and the downgrade
is recorded. Confidence is optional and is *never* synthesized locally: ``None``
means "the model did not state one". No overall numerical paper score exists
anywhere in this schema on purpose: that judgement belongs to the later critique
stage, and this stage must not fabricate a single number for a whole paper.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# How well the collected evidence supports a statement.
EvidenceStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "unclear",
    "insufficient_evidence",
]

EVIDENCE_STATUSES = (
    "supported",
    "partially_supported",
    "unsupported",
    "unclear",
    "insufficient_evidence",
)

# Which upstream artifact a reference points at. Only artifacts the Analysis
# Agent actually received may be cited.
EvidenceSourceType = Literal[
    "document_profile",
    "metadata",
    "section",
    "figure",
    "table",
    "equation",
    "reference",
    "text_chunk",
    "visual_asset",
]

EVIDENCE_SOURCE_TYPES = (
    "document_profile",
    "metadata",
    "section",
    "figure",
    "table",
    "equation",
    "reference",
    "text_chunk",
    "visual_asset",
)

GapCategory = Literal[
    "rag_disabled",
    "rag_unavailable",
    "vision_disabled",
    "vision_unavailable",
    "numeric_values_unavailable",
    "not_reported_in_paper",
    "model_uncertainty",
    "unverified_reference",
    "no_numeric_paper_score",
    "context_truncated",
    "other",
]

AnalysisStatus = Literal["completed", "failed", "disabled"]

CLAIM_SOURCES = ("paper", "analyst_inference")


class EvidenceRef(BaseModel):
    """
    One citable piece of evidence with provenance.

    ``verified`` is False until the analyzer has matched ``source_id`` against the
    evidence bundle it built itself, so a model-invented identifier can never
    travel downstream looking authoritative.
    """

    source_type: EvidenceSourceType = Field(
        ..., description="Kind of upstream artifact this reference points at"
    )
    source_id: str = Field(
        ..., description="Identifier inside that artifact, e.g. section_004 / chunk_002"
    )
    pages: List[int] = Field(
        default_factory=list, description="1-indexed pages this evidence spans"
    )
    verified: bool = Field(
        False,
        description="True only when the analyzer matched this id in the evidence bundle",
    )
    detail: Optional[str] = Field(
        None, description="Short human-readable label (caption, title, score, ...)"
    )


class Finding(BaseModel):
    """One atomic, evidence-grounded statement about the paper."""

    statement: str = Field(..., description="The single claim or observation made")
    evidence_status: EvidenceStatus = Field(
        "unclear", description="How well the cited evidence supports the statement"
    )
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional model-stated confidence; None means 'not stated'",
    )
    reasoning: str = Field("", description="Why the evidence leads to the statement")
    caveats: List[str] = Field(
        default_factory=list,
        description="Explicit limits, doubts or provenance problems for this finding",
    )


class DimensionAnalysis(BaseModel):
    """
    Structured analysis of one evaluation dimension.

    ``summary`` is the evidence-grounded reading of the dimension; ``findings``
    are the atomic statements behind it. Absence of evidence is expressed with
    the ``insufficient_evidence`` status, never by inventing content.
    """

    dimension: str = Field(..., description="Dimension name, e.g. methodology")
    summary: str = Field("", description="Evidence-grounded reading of the dimension")
    evidence_status: EvidenceStatus = "unclear"
    findings: List[Finding] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(
        default_factory=list, description="Dimension-level references"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)


class ClaimEvidenceItem(BaseModel):
    """One claim from the paper mapped onto the evidence that exists for it."""

    claim: str = Field(..., description="The claim being evaluated")
    claim_source: str = Field(
        "paper",
        description="paper (stated by the authors) | analyst_inference",
    )
    status: EvidenceStatus = Field(
        "unclear", description="How well the available evidence supports the claim"
    )
    supporting_evidence: List[EvidenceRef] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceRef] = Field(default_factory=list)
    assessment: str = Field("", description="Reasoning about support / contradiction")
    missing_evidence: List[str] = Field(
        default_factory=list, description="Evidence that would be needed to decide"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class ClaimEvidenceMatrix(BaseModel):
    """The claim -> evidence mapping for the paper."""

    items: List[ClaimEvidenceItem] = Field(default_factory=list)
    notes: str = Field("", description="Coverage notes about the matrix itself")


class AssessedPoint(BaseModel):
    """A strength or weakness with its evidence and rationale."""

    statement: str = Field(..., description="The strength or weakness observed")
    dimension: Optional[str] = Field(
        None, description="Dimension this point belongs to, when applicable"
    )
    evidence_status: EvidenceStatus = "unclear"
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    rationale: str = Field("", description="Why this counts as a strength/weakness")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class Limitations(BaseModel):
    """
    Limitations with an explicit split between what the authors state and what the
    analyst identifies. Blurring the two would misattribute a limitation to the
    paper, so the separation is structural rather than stylistic.
    """

    author_stated: List[Finding] = Field(
        default_factory=list, description="Limitations the paper itself states"
    )
    analyst_identified: List[Finding] = Field(
        default_factory=list, description="Limitations the analyst identified"
    )


class OpenQuestion(BaseModel):
    """A question the available evidence cannot answer."""

    question: str = Field(..., description="The open question")
    why_it_matters: str = Field("", description="Why answering it would change the reading")
    related_dimension: Optional[str] = None
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    """
    An explicitly recorded gap in the evidence base.

    Deterministic gaps (RAG disabled, Vision unavailable, no machine-readable
    numeric values, truncated context, ...) are computed by the context builder
    and are never left to the model's discretion.
    """

    gap: str = Field(..., description="What evidence is missing or unusable")
    category: GapCategory = "other"
    impact: str = Field("", description="How this gap limits the analysis")
    affected_dimensions: List[str] = Field(default_factory=list)
    source: str = Field(
        "analysis_model",
        description="deterministic_context_check | analysis_model | analyzer_validation",
    )


class OverallAssessment(BaseModel):
    """
    Evidence-grounded overall reading of the paper.

    There is deliberately no score field: the Analysis Agent reports what the
    evidence supports and where it is insufficient. Any numeric verdict belongs to
    the later critique stage.
    """

    summary: str = Field("", description="Evidence-grounded overall reading")
    evidence_status: EvidenceStatus = "unclear"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    key_basis: List[EvidenceRef] = Field(
        default_factory=list, description="The references carrying the assessment"
    )
    uncertainties: List[str] = Field(default_factory=list)


class AnalysisFailure(BaseModel):
    """A structured failure stage record (execution never fails silently)."""

    stage: str = Field(
        ..., description="context_builder | analysis (model call / parsing)"
    )
    error: str = Field(..., description="Failure reason (never contains secrets)")
    model: Optional[str] = None


class SectionRef(BaseModel):
    """A section taken from the frozen Document Profile (never re-detected here)."""

    source_id: str = Field(..., description="Stable context id, e.g. section_004")
    title: str
    normalized_title: str = ""
    number: str = ""
    page: Optional[int] = None
    level: int = Field(1, ge=1, description="Heading depth reported by the profile")
    confidence: Optional[float] = None


class InventoryItemRef(BaseModel):
    """
    A confirmed figure/table/equation from the frozen Document Profile.

    The Analysis Agent consumes this inventory as given: it does not discover
    visuals or equations of its own. Structural metadata (rows/columns, equation
    representation) is carried as context only, never as machine-read values.
    """

    source_id: str
    item_type: str = Field(..., description="figure | table | equation")
    label: Optional[str] = Field(None, description="Printed number, e.g. '2'")
    page: Optional[int] = None
    caption: Optional[str] = None
    description: Optional[str] = Field(
        None, description="Extra description (equation representation)"
    )
    structure_source: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    confidence: Optional[float] = None


class TextEvidenceRef(BaseModel):
    """A retrieved text chunk handed over by the frozen RAG Agent."""

    source_id: str = Field(..., description="Stable context id, e.g. chunk_002")
    chunk_id: str
    score: Optional[float] = None
    page_start: int = Field(1, ge=1)
    page_end: int = Field(1, ge=1)
    section: Optional[str] = None
    text: str = ""
    word_count: int = Field(0, ge=0)


class VisualEvidenceRef(BaseModel):
    """A visual analysis handed over by the frozen Vision Agent."""

    source_id: str = Field(..., description="Stable context id, e.g. visual_figure_001")
    asset_id: str
    asset_type: str = Field("figure", description="figure | table")
    page_number: Optional[int] = None
    section: Optional[str] = None
    caption: Optional[str] = None
    observation: str = ""
    interpretation: str = ""
    key_elements: List[str] = Field(default_factory=list)
    reported_relationships: List[str] = Field(default_factory=list)
    supports_claims: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    caption_consistency_status: Optional[str] = None
    numeric_authority: Optional[str] = Field(
        None, description="document_profile_machine_readable | vision_interpretation_only"
    )
    structured_values: Optional[str] = Field(
        None,
        description="Rendered machine-readable cell values when the Vision Agent had them",
    )
    source_confidence: Optional[float] = None


class EvidenceBundle(BaseModel):
    """
    The bounded evidence context the Analysis Agent reasons over.

    ``source_index`` is the authoritative registry of citable ids; the analyzer
    validates every reference the model emits against it. ``deterministic_gaps``
    records evidence the agent knows is missing before the model is even called.
    """

    filename: str = ""
    page_count: int = Field(0, ge=0)
    word_count: int = Field(0, ge=0)
    text_length: int = Field(0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_characteristics: Dict[str, Any] = Field(default_factory=dict)

    sections: List[SectionRef] = Field(default_factory=list)
    figures: List[InventoryItemRef] = Field(default_factory=list)
    tables: List[InventoryItemRef] = Field(default_factory=list)
    equations: List[InventoryItemRef] = Field(default_factory=list)
    references_summary: Dict[str, Any] = Field(default_factory=dict)

    text_evidence: List[TextEvidenceRef] = Field(default_factory=list)
    visual_evidence: List[VisualEvidenceRef] = Field(default_factory=list)

    deterministic_gaps: List[EvidenceGap] = Field(default_factory=list)
    truncations: List[str] = Field(default_factory=list)
    source_index: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)
    total_evidence_chars: int = Field(0, ge=0)


class EvidenceInputs(BaseModel):
    """Auditable summary of the evidence the analysis was actually run on."""

    sections: int = Field(0, ge=0)
    figures: int = Field(0, ge=0)
    tables: int = Field(0, ge=0)
    equations: int = Field(0, ge=0)
    references: int = Field(0, ge=0)
    rag_enabled: bool = False
    rag_chunks_used: int = Field(0, ge=0)
    vision_enabled: bool = False
    visual_assets_used: int = Field(0, ge=0)
    context_chars: int = Field(0, ge=0)
    truncations: List[str] = Field(default_factory=list)


class RoutingSummary(BaseModel):
    """
    Read-only view of the JEV routing decision this execution consumed.

    Routing confidence is passed through verbatim when the router provided it and
    is never invented here.
    """

    router_version: Optional[str] = None
    analysis_enabled: bool = True
    analysis_level: Optional[str] = None
    rag_enabled: bool = False
    rag_level: Optional[str] = None
    vision_enabled: bool = False
    vision_level: Optional[str] = None
    routing_confidence: Optional[Dict[str, float]] = None


class AnalysisPayload(BaseModel):
    """
    The structured evaluation the analysis model must produce.

    Every dimension is required by the parser; a response missing any of them is
    rejected rather than partially accepted (see ``analyzer.parse_analysis``).
    """

    research_problem: DimensionAnalysis
    contribution: DimensionAnalysis
    methodology: DimensionAnalysis
    data: DimensionAnalysis
    baselines: DimensionAnalysis
    metrics: DimensionAnalysis
    results: DimensionAnalysis
    reproducibility: DimensionAnalysis
    internal_consistency: DimensionAnalysis

    claim_evidence_matrix: ClaimEvidenceMatrix
    strengths: List[AssessedPoint] = Field(default_factory=list)
    weaknesses: List[AssessedPoint] = Field(default_factory=list)
    limitations: Limitations = Field(default_factory=Limitations)
    open_questions: List[OpenQuestion] = Field(default_factory=list)
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    overall_assessment: OverallAssessment = Field(default_factory=OverallAssessment)


class ParsedAnalysis(BaseModel):
    """
    A validated payload plus the analyzer's own honesty bookkeeping.

    ``notes`` are gap/uncertainty records the analyzer derived from the response
    (downgraded statuses, overclaiming language, score-like text, unknown
    references); they are merged with the deterministic context gaps by the agent.
    """

    payload: AnalysisPayload
    notes: List[EvidenceGap] = Field(default_factory=list)
    overclaim_flags: List[str] = Field(default_factory=list)
    unverified_evidence_refs: int = Field(0, ge=0)
    downgraded_statuses: int = Field(0, ge=0)


class AnalysisResult(BaseModel):
    """
    Final JSON-serializable Analysis Agent execution output.

    ``analysis`` is None when the execution failed or was disabled: the envelope
    still describes why, so a downstream stage never has to guess. This envelope
    contains no paper-level score.
    """

    agent: str = Field("analysis", description="Agent name identifier")
    status: AnalysisStatus = Field(..., description="completed | failed | disabled")
    enabled: bool = Field(..., description="Whether analysis ran per JEV routing state")
    level: Optional[str] = Field(
        None, description="JEV analysis capability tier (basic, medium, advanced)"
    )
    model: Optional[str] = Field(None, description="Selected analysis model id")
    provider: Optional[str] = Field(None, description="Analysis provider identifier")

    routing: RoutingSummary = Field(default_factory=RoutingSummary)
    evidence_inputs: EvidenceInputs = Field(default_factory=EvidenceInputs)
    analysis: Optional[AnalysisPayload] = Field(
        None, description="Structured evaluation, or None when it could not be produced"
    )
    evidence_gaps: List[EvidenceGap] = Field(
        default_factory=list,
        description="Deterministic context gaps merged with model-reported gaps",
    )
    unverified_evidence_refs: int = Field(
        0, ge=0, description="References that did not match the evidence bundle"
    )
    elapsed_time_ms: Optional[float] = None
    message: Optional[str] = Field(None, description="Status or failure message")
    failures: List[AnalysisFailure] = Field(default_factory=list)
