"""
Schemas for the Critique Engine (Phase 6).

Pydantic models for:
- The synthesis context the Critique Engine builds from the frozen Document
  Profile, the frozen JEV routing state and the frozen RAG / Vision / Analysis
  outputs (a *deterministic projection*, never a re-analysis)
- Evidence provenance (``SourceRef``) with an explicit verification flag, so an
  invented ``chunk_999`` / ``figure_999`` can never travel downstream as fact
- The final report sections, the claim/evidence summary, strengths, the two
  strictly separated limitation categories, evidence gaps, open questions and
  the overall assessment
- The JSON-serializable execution envelope (completed / failed /
  validation_failed)

Synthesis boundary
------------------
Phase 5 answered *"what does the evidence suggest about the research?"*; this
stage answers *"how should those structured findings be organised and
communicated as a coherent critique?"*. Two consequences are structural, not
stylistic:

* No new substantive finding can enter here. Every section carries
  ``evidence_refs`` and the validator checks them against the supplied inputs.
* There is no paper score, no quality score, no ranking and no tier anywhere in
  this schema. An unsupported overall label ("excellent"/"good"/"bad") is treated
  the same way a fabricated citation is: it fails validation.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

CRITIQUE_VERSION = "1.0"

CritiqueStatus = Literal["completed", "failed", "validation_failed"]

# Which upstream artifact a provenance reference resolves to. The Critique Engine
# adds no new source kind of its own.
SourceType = Literal["document", "rag", "vision", "analysis"]

SOURCE_TYPES = ("document", "rag", "vision", "analysis")

# Claim classifications are inherited verbatim from the frozen Analysis Agent.
# This stage never creates a new classification.
ClaimStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "unclear",
    "insufficient_evidence",
]

CLAIM_STATUSES = (
    "supported",
    "partially_supported",
    "unsupported",
    "unclear",
    "insufficient_evidence",
)

# An uncertainty marker the model may use for a statement it could not map onto a
# supplied source. Text carrying one is preserved (never deleted) but is kept out
# of the verified evidence set.
UNCERTAINTY_MARKERS = (
    "could not be established",
    "not established from the available evidence",
    "not reported in the available evidence",
    "unclear from the available evidence",
    "open question",
    "uncertain",
    "unverified",
)


class SectionSpec(BaseModel):
    """One entry of the fixed final-report section catalogue."""

    section_id: str = Field(..., description="Stable machine id of the section")
    title: str = Field(..., description="Human-readable section title")
    order: int = Field(..., ge=1, description="Display order in the report")
    group: str = Field("general", description="Coarse grouping used by renderers")
    always_required: bool = Field(
        False,
        description="True when the section must be present for every paper",
    )


# The report structure. ``always_required`` marks the two sections that carry the
# critique itself (executive summary and overall assessment); every other section
# is required only when the upstream evidence actually supports it, so an
# unsupported section is dropped instead of forced empty.
SECTION_SPECS: List[SectionSpec] = [
    SectionSpec(
        section_id="executive_summary",
        title="Executive Summary",
        order=1,
        group="front_matter",
        always_required=True,
    ),
    SectionSpec(
        section_id="research_problem",
        title="Research Problem & Motivation",
        order=2,
        group="problem",
    ),
    SectionSpec(
        section_id="contribution",
        title="Main Contribution",
        order=3,
        group="problem",
    ),
    SectionSpec(
        section_id="methodology",
        title="Methodology Assessment",
        order=4,
        group="method",
    ),
    SectionSpec(
        section_id="data_and_experimental_design",
        title="Data & Experimental Design",
        order=5,
        group="method",
    ),
    SectionSpec(
        section_id="baselines_and_metrics",
        title="Baselines & Evaluation Metrics",
        order=6,
        group="method",
    ),
    SectionSpec(
        section_id="results_and_evidence",
        title="Results & Evidence",
        order=7,
        group="evidence",
    ),
    SectionSpec(
        section_id="figures_and_tables",
        title="Figures & Tables",
        order=8,
        group="evidence",
    ),
    SectionSpec(
        section_id="claim_evidence_assessment",
        title="Claim-Evidence Assessment",
        order=9,
        group="evidence",
    ),
    SectionSpec(
        section_id="strengths",
        title="Strengths",
        order=10,
        group="assessment",
    ),
    SectionSpec(
        section_id="author_stated_limitations",
        title="Limitations - Author-Stated",
        order=11,
        group="limitations",
    ),
    SectionSpec(
        section_id="analyst_identified_limitations",
        title="Limitations - Analyst-Identified",
        order=12,
        group="limitations",
    ),
    SectionSpec(
        section_id="reproducibility",
        title="Reproducibility Assessment",
        order=13,
        group="assessment",
    ),
    SectionSpec(
        section_id="internal_consistency",
        title="Internal Consistency",
        order=14,
        group="assessment",
    ),
    SectionSpec(
        section_id="evidence_gaps",
        title="Evidence Gaps",
        order=15,
        group="closing",
    ),
    SectionSpec(
        section_id="open_questions",
        title="Open Research Questions",
        order=16,
        group="closing",
    ),
    SectionSpec(
        section_id="overall_assessment",
        title="Overall Research Assessment",
        order=17,
        group="closing",
        always_required=True,
    ),
]

SECTION_IDS = tuple(spec.section_id for spec in SECTION_SPECS)
SECTION_TITLES = {spec.section_id: spec.title for spec in SECTION_SPECS}
SECTION_ORDER = {spec.section_id: spec.order for spec in SECTION_SPECS}
ALWAYS_REQUIRED_SECTIONS = tuple(
    spec.section_id for spec in SECTION_SPECS if spec.always_required
)

# The two limitation sections must never be merged or swapped.
AUTHOR_STATED_LIMITATION_SECTION = "author_stated_limitations"
ANALYST_LIMITATION_SECTION = "analyst_identified_limitations"
LIMITATION_SECTIONS = (
    AUTHOR_STATED_LIMITATION_SECTION,
    ANALYST_LIMITATION_SECTION,
)


class SourceRef(BaseModel):
    """
    One provenance reference in the final report.

    ``verified`` is False until the validator has matched ``(source_type,
    source_id)`` against the sources the engine actually supplied, and ``pages`` is
    only ever populated from that matched metadata. An unverifiable reference is
    reported in ``unverified_evidence_refs`` instead of being silently trusted.
    """

    source_type: SourceType = Field(..., description="Which upstream artifact this is")
    source_id: str = Field(
        ..., description="Exact upstream id, e.g. chunk_012 / figure_002"
    )
    pages: List[int] = Field(
        default_factory=list, description="1-indexed pages taken from upstream metadata"
    )
    page: Optional[int] = Field(
        None, description="Single page for page-anchored sources (vision assets)"
    )
    verified: bool = Field(
        False, description="True only after the id matched a supplied source"
    )
    label: Optional[str] = Field(
        None, description="Short human-readable label (caption, section title, ...)"
    )


class CritiqueSection(BaseModel):
    """One section of the human-readable final critique."""

    section_id: str = Field(..., description="Id from the fixed section catalogue")
    title: str = Field("", description="Display title")
    content: str = Field(..., description="The readable synthesis text")
    evidence_refs: List[SourceRef] = Field(
        default_factory=list, description="Provenance for the statements in content"
    )
    uncertain: bool = Field(
        False,
        description="True when the content is explicitly marked as not established",
    )


class ClaimEvidenceSummaryItem(BaseModel):
    """One row of the claim/evidence assessment, preserving the Phase 5 status."""

    claim: str = Field(..., description="The claim being assessed")
    status: ClaimStatus = Field(
        "unclear", description="Classification inherited from the analysis stage"
    )
    assessment: str = Field("", description="How the available evidence reads it")
    evidence_refs: List[SourceRef] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)


class CritiqueStrength(BaseModel):
    """A strength, kept as prose with provenance and never as a number."""

    statement: str = Field(..., description="The strength observed")
    dimension: Optional[str] = Field(None, description="Analysis dimension it sits in")
    evidence_refs: List[SourceRef] = Field(default_factory=list)
    rationale: str = Field("")


class CritiqueLimitation(BaseModel):
    """One limitation. Its category is decided by the engine, not by the model, so
    an analyst observation can never be filed as an author one."""

    statement: str = Field(..., description="The limitation stated or identified")
    category: Literal["author_stated", "analyst_identified"]
    evidence_refs: List[SourceRef] = Field(default_factory=list)
    rationale: str = Field("")


class CritiqueLimitations(BaseModel):
    """Two strictly separate categories; merging them is a validation error."""

    author_stated: List[CritiqueLimitation] = Field(default_factory=list)
    analyst_identified: List[CritiqueLimitation] = Field(default_factory=list)


class CritiqueGap(BaseModel):
    """An explicitly reported gap in the evidence base."""

    gap: str = Field(..., description="What could not be established")
    category: str = Field(
        "other", description="Category inherited from the analysis stage"
    )
    impact: str = Field("", description="How the gap limits the critique")
    source: str = Field(
        "analysis",
        description="deterministic_context_check | analysis | validation",
    )


class CritiqueOpenQuestion(BaseModel):
    """A research question. Always a question, never a factual claim."""

    question: str = Field(..., description="The open question")
    why_it_matters: str = Field("")
    evidence_refs: List[SourceRef] = Field(default_factory=list)


class CritiqueOverallAssessment(BaseModel):
    """
    The concise whole-paper synthesis.

    There is deliberately no score, grade, rank or tier field, and no overall
    label that the evidence does not support.
    """

    content: str = Field("", description="Evidence-grounded overall reading")
    evidence_refs: List[SourceRef] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class CritiqueProvenance(BaseModel):
    """Exactly which upstream sources the final report actually leaned on."""

    rag_sources_used: List[str] = Field(default_factory=list)
    vision_sources_used: List[str] = Field(default_factory=list)
    analysis_sources_used: List[str] = Field(default_factory=list)
    document_sources_used: List[str] = Field(default_factory=list)


class CritiquePayload(BaseModel):
    """
    The validated final critique.

    ``claim_evidence_summary`` is required (the Phase 5 matrix is preserved), the
    rest may be empty when the upstream evidence supports nothing there.
    """

    sections: List[CritiqueSection] = Field(default_factory=list)
    claim_evidence_summary: List[ClaimEvidenceSummaryItem] = Field(default_factory=list)
    strengths: List[CritiqueStrength] = Field(default_factory=list)
    limitations: CritiqueLimitations = Field(default_factory=CritiqueLimitations)
    evidence_gaps: List[CritiqueGap] = Field(default_factory=list)
    open_questions: List[CritiqueOpenQuestion] = Field(default_factory=list)
    overall_assessment: CritiqueOverallAssessment = Field(
        default_factory=CritiqueOverallAssessment
    )


class ParsedCritique(BaseModel):
    """A validated payload plus the validator's honesty bookkeeping."""

    payload: CritiquePayload
    warnings: List[str] = Field(default_factory=list)
    notes: List[CritiqueGap] = Field(default_factory=list)
    unverified_evidence_refs: int = Field(0, ge=0)
    dropped_sections: List[str] = Field(default_factory=list)


class CritiqueFailure(BaseModel):
    """A structured failure record (execution never fails silently)."""

    stage: str = Field(
        ...,
        description=(
            "analysis_input | context_builder | critique_generation | validation"
        ),
    )
    error: str = Field(..., description="Failure reason (never contains secrets)")
    model: Optional[str] = None


class DocumentSummary(BaseModel):
    """The document-level facts the critique may state (from the frozen profile)."""

    title: Optional[str] = None
    filename: str = ""
    page_count: int = 0
    word_count: int = 0
    section_count: int = 0
    figure_count: int = 0
    table_count: int = 0
    equation_count: int = 0
    reference_count: int = 0


class RoutingSummary(BaseModel):
    """Read-only view of the routing decision already made by the frozen JEV."""

    router_version: Optional[str] = None
    analysis_level: Optional[str] = None
    rag_enabled: bool = False
    rag_level: Optional[str] = None
    vision_enabled: bool = False
    vision_level: Optional[str] = None
    confidence: Optional[Dict[str, float]] = None


class RetrievedEvidence(BaseModel):
    """Concise RAG provenance carried for traceability (never re-retrieved)."""

    source_id: str
    pages: List[int] = Field(default_factory=list)
    section: Optional[str] = None
    excerpt: str = ""
    score: Optional[float] = None


class VisualEvidence(BaseModel):
    """A Vision Agent observation carried verbatim into the synthesis context."""

    source_id: str
    asset_type: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    observation: str = ""
    interpretation: str = ""
    uncertainties: List[str] = Field(default_factory=list)


class AnalysisDimensionSummary(BaseModel):
    """One Phase 5 dimension, projected (not re-interpreted) for synthesis."""

    dimension: str
    summary: str = ""
    findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class AnalysisClaim(BaseModel):
    """A Phase 5 claim/evidence row projected for the critique."""

    claim: str
    status: ClaimStatus = "unclear"
    assessment: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    """The structured Phase 5 evaluation, flattened into synthesis-ready records."""

    dimensions: List[AnalysisDimensionSummary] = Field(default_factory=list)
    claims: List[AnalysisClaim] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    author_stated_limitations: List[str] = Field(default_factory=list)
    analyst_identified_limitations: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    evidence_gaps: List[CritiqueGap] = Field(default_factory=list)
    overall_assessment: str = ""


class SupportedSections(BaseModel):
    """Which report sections the supplied evidence can actually support."""

    section_ids: List[str] = Field(default_factory=list)
    reasons: Dict[str, str] = Field(default_factory=dict)


class SynthesisContext(BaseModel):
    """
    The deterministic synthesis context.

    Building it is a projection of already-produced structured output: no parsing,
    no chunking, no retrieval, no visual analysis and no new judgement happens
    here. ``source_index`` is the allow-list the validator checks every reference
    against, which is what makes "no fabricated citations" enforceable.
    """

    document: DocumentSummary = Field(default_factory=DocumentSummary)
    routing: RoutingSummary = Field(default_factory=RoutingSummary)
    retrieved_evidence: List[RetrievedEvidence] = Field(default_factory=list)
    visual_evidence: List[VisualEvidence] = Field(default_factory=list)
    analysis: AnalysisSummary = Field(default_factory=AnalysisSummary)
    supported_sections: SupportedSections = Field(default_factory=SupportedSections)
    source_index: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    deterministic_gaps: List[CritiqueGap] = Field(default_factory=list)
    truncations: List[str] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    total_context_chars: int = 0


class ValidationIssue(BaseModel):
    """One validation finding, precise enough to debug without leaking secrets."""

    code: str = Field(..., description="Stable machine-readable issue code")
    message: str = Field(..., description="Human-readable explanation")
    location: Optional[str] = Field(
        None, description="Where the issue was found, e.g. sections[2].evidence_refs"
    )
    severity: Literal["error", "warning"] = "error"


class ValidationReport(BaseModel):
    """The outcome of validating a candidate critique against the real inputs."""

    valid: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)
    unverified_evidence_refs: int = Field(0, ge=0)

    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


class CritiqueResult(BaseModel):
    """
    Final JSON-serializable Critique Engine execution output.

    On success this carries the report sections plus ``provenance``; on failure it
    carries ``failures`` and a message that never contains a credential. No
    execution path produces a paper score.
    """

    engine: str = Field("critique", description="Engine name identifier")
    version: str = Field(CRITIQUE_VERSION, description="Output contract version")
    status: CritiqueStatus = Field(...)
    enabled: bool = Field(True, description="Whether synthesis was attempted")
    level: Optional[str] = Field(
        None, description="JEV analysis level the synthesis model was resolved from"
    )
    model: Optional[str] = Field(None, description="Selected synthesis model id")
    provider: Optional[str] = Field(None, description="Synthesis provider identifier")

    routing: RoutingSummary = Field(default_factory=RoutingSummary)
    sections: List[CritiqueSection] = Field(default_factory=list)
    claim_evidence_summary: List[ClaimEvidenceSummaryItem] = Field(default_factory=list)
    strengths: List[CritiqueStrength] = Field(default_factory=list)
    limitations: CritiqueLimitations = Field(default_factory=CritiqueLimitations)
    evidence_gaps: List[CritiqueGap] = Field(default_factory=list)
    open_questions: List[CritiqueOpenQuestion] = Field(default_factory=list)
    overall_assessment: CritiqueOverallAssessment = Field(
        default_factory=CritiqueOverallAssessment
    )
    provenance: CritiqueProvenance = Field(default_factory=CritiqueProvenance)

    context_inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Auditable summary of what was synthesized"
    )
    unverified_evidence_refs: int = Field(0, ge=0)
    warnings: List[str] = Field(default_factory=list)
    elapsed_time_ms: Optional[float] = None
    message: Optional[str] = Field(None, description="Status or failure message")
    failures: List[CritiqueFailure] = Field(default_factory=list)