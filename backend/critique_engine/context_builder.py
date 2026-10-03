"""
Deterministic synthesis-context assembly for the Critique Engine (Phase 6).

This module is a *projection* step, never an analysis step. It reads the frozen
Document Profile, the frozen JEV routing state and the frozen RAG / Vision /
Analysis outputs and produces one compact ``SynthesisContext``:

    DOCUMENT  ->  title, page/word/section/figure/table/equation/reference counts
    ROUTING   ->  the levels JEV already chose plus its confidence values
    RAG       ->  concise chunk provenance (id, pages, section, short excerpt)
    VISION    ->  asset_id, type, page, section, observation, interpretation,
                 uncertainties
    ANALYSIS  ->  the structured Phase 5 findings, flattened but not reinterpreted
    ALLOWED   ->  the section ids the evidence can support
    SOURCES   ->  the allow-list every citation is validated against

What deliberately does *not* happen here: parsing the PDF, re-detecting figures
or tables, chunking, embedding, retrieval, visual analysis, re-evaluating the
research, or making any routing decision. The engine can only ever reorganise
what earlier stages produced.

Determinism
-----------
Every collection is built by iterating the upstream structures in their existing
order and dropping entries by explicit budget, so two runs on identical inputs
produce an identical context. Truncations are recorded, never silent.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.critique_engine.config import CritiqueConfig, normalize_level
from backend.critique_engine.schemas import (
    AnalysisClaim,
    AnalysisDimensionSummary,
    AnalysisSummary,
    CritiqueGap,
    DocumentSummary,
    RetrievedEvidence,
    RoutingSummary,
    SourceType,
    SupportedSections,
    SynthesisContext,
    VisualEvidence,
)

logger = logging.getLogger(__name__)

# Phase 5 dimension key -> report section it feeds. Used only to decide whether a
# section is supported; the wording is still produced by the model.
DIMENSION_TO_SECTION: Dict[str, str] = {
    "research_problem": "research_problem",
    "contribution": "contribution",
    "methodology": "methodology",
    "data": "data_and_experimental_design",
    "baselines": "baselines_and_metrics",
    "metrics": "baselines_and_metrics",
    "results": "results_and_evidence",
    "reproducibility": "reproducibility",
    "internal_consistency": "internal_consistency",
}


class ContextBuilderError(Exception):
    """Raised when the supplied upstream artifacts cannot support any synthesis."""

    pass


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _truncate(value: Any, limit: int) -> Tuple[str, bool]:
    cleaned = _text(value)
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned, False
    return cleaned[: max(1, limit - 1)].rstrip() + "...", True


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _confirmed_items(block: Any, confirmed_key: str) -> List[Dict[str, Any]]:
    """
    Confirmed inventory entries from a frozen Document Profile block.

    The profile exposes an inventory either under ``confirmed_key`` or as a
    ``{"count": n, "items": [...]}`` block whose entries carry a ``status``; both
    shapes are read here so the engine never re-detects anything itself.
    """
    if not isinstance(block, dict):
        return []
    items = block.get(confirmed_key)
    if not isinstance(items, (list, tuple)):
        items = block.get("items")
    if not isinstance(items, (list, tuple)):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("status", "confirmed")) == "confirmed"
    ]


def _page_range(start: Any, end: Any) -> List[int]:
    first = _as_int(start)
    last = _as_int(end)
    if first is None and last is None:
        return []
    first = first if first is not None else last
    last = last if last is not None else first
    first = max(1, min(first, last))
    last = max(first, last)
    return list(range(first, last + 1))


def _add_source(
    index: Dict[str, Dict[str, Any]],
    source_type: SourceType,
    source_id: Any,
    pages: Optional[List[int]] = None,
    page: Optional[int] = None,
    label: Optional[str] = None,
) -> None:
    """Register one citable source under its stable ``type:id`` key."""
    identifier = _text(source_id)
    if not identifier:
        return
    key = f"{source_type}:{identifier}"
    entry = index.get(key)
    if entry is None:
        entry = {
            "source_type": source_type,
            "source_id": identifier,
            "pages": [],
            "page": None,
            "label": None,
        }
        index[key] = entry
    for value in pages or []:
        if isinstance(value, int) and value > 0 and value not in entry["pages"]:
            entry["pages"].append(value)
    if page is not None and isinstance(page, int) and page > 0:
        entry["page"] = page
    label_text = _text(label)
    if label_text and not entry.get("label"):
        entry["label"] = label_text[:200]


def _build_document_summary(
    document_profile: Dict[str, Any]
) -> Tuple[DocumentSummary, Dict[str, Dict[str, Any]]]:
    """Document-level facts plus the profile-derived citable sources."""
    metadata = document_profile.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    title = _text(metadata.get("title")) or _text(document_profile.get("title")) or None
    filename = _text(document_profile.get("filename")) or _text(metadata.get("filename"))

    figure_items = _confirmed_items(document_profile.get("figures"), "confirmed_figures")
    table_items = _confirmed_items(document_profile.get("tables"), "confirmed_tables")
    equation_items = _confirmed_items(
        document_profile.get("equations"), "confirmed_equations"
    )

    references_block = document_profile.get("references")
    reference_count = 0
    if isinstance(references_block, dict):
        reference_count = _as_int(references_block.get("count")) or 0

    summary = DocumentSummary(
        title=title,
        filename=filename,
        page_count=_as_int(document_profile.get("page_count")) or 0,
        word_count=_as_int(document_profile.get("word_count")) or 0,
        section_count=len(_dict_list(document_profile.get("sections"))),
        figure_count=len(figure_items),
        table_count=len(table_items),
        equation_count=len(equation_items),
        reference_count=reference_count,
    )

    index: Dict[str, Dict[str, Any]] = {}
    _add_source(
        index,
        "document",
        "document_profile",
        label=title or filename or "document profile",
    )
    for position, section in enumerate(_dict_list(document_profile.get("sections")), 1):
        section_id = (
            section.get("source_id")
            or section.get("section_id")
            or f"section_{position:03d}"
        )
        page = _as_int(section.get("page") or section.get("page_number"))
        _add_source(
            index,
            "document",
            section_id,
            pages=[page] if page else [],
            label=section.get("title"),
        )
    for item in figure_items + table_items + equation_items:
        source_id = item.get("source_id") or item.get("id") or item.get("figure_number")
        page = _as_int(item.get("page") or item.get("page_number"))
        _add_source(
            index,
            "document",
            source_id,
            pages=[page] if page else [],
            label=item.get("caption") or item.get("label"),
        )
    return summary, index


def _build_routing_summary(routing_state: Any) -> RoutingSummary:
    """Read-only, verbatim view of the decision the frozen JEV already made."""
    state = routing_state if isinstance(routing_state, dict) else {}

    def block(name: str) -> Dict[str, Any]:
        value = state.get(name)
        return value if isinstance(value, dict) else {}

    analysis = block("analysis")
    rag = block("rag")
    vision = block("vision")

    confidence = state.get("confidence")
    routing_confidence: Optional[Dict[str, float]] = None
    if isinstance(confidence, dict):
        routing_confidence = {
            str(key): float(value)
            for key, value in confidence.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    router_version = state.get("router_version")
    return RoutingSummary(
        router_version=str(router_version) if router_version is not None else None,
        analysis_level=normalize_level(
            analysis.get("level", state.get("analysis_level"))
        ),
        rag_enabled=bool(rag.get("enabled", state.get("rag_enabled", False))),
        rag_level=normalize_level(rag.get("level", state.get("rag_level"))),
        vision_enabled=bool(vision.get("enabled", state.get("vision_enabled", False))),
        vision_level=normalize_level(vision.get("level", state.get("vision_level"))),
        confidence=routing_confidence,
    )


def _build_retrieved_evidence(
    rag_result: Optional[Dict[str, Any]], cfg: CritiqueConfig
) -> Tuple[List[RetrievedEvidence], List[str], Dict[str, Dict[str, Any]]]:
    """Concise RAG provenance; retrieval itself is never re-run here."""
    evidence: List[RetrievedEvidence] = []
    truncations: List[str] = []
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rag_result, dict):
        return evidence, truncations, index

    results = _dict_list(rag_result.get("results"))
    kept = results[: max(0, cfg.max_rag_chunks)]
    dropped = len(results) - len(kept)
    truncated = 0
    for chunk in kept:
        chunk_id = _text(chunk.get("chunk_id") or chunk.get("source_id"))
        if not chunk_id:
            continue
        pages = _page_range(chunk.get("page_start"), chunk.get("page_end"))
        section = _text(chunk.get("section")) or None
        excerpt, was_truncated = _truncate(chunk.get("text"), cfg.max_chunk_chars)
        if was_truncated:
            truncated += 1
        evidence.append(
            RetrievedEvidence(
                source_id=chunk_id,
                pages=pages,
                section=section,
                excerpt=excerpt,
                score=_as_float(chunk.get("score")),
            )
        )
        _add_source(index, "rag", chunk_id, pages=pages, label=section)

    if dropped > 0:
        truncations.append(
            f"retrieved evidence: {dropped} chunk(s) beyond CRITIQUE_MAX_RAG_CHUNKS "
            "were left out"
        )
    if truncated:
        truncations.append(
            f"retrieved evidence: {truncated} chunk(s) truncated to "
            "CRITIQUE_MAX_CHUNK_CHARS"
        )
    return evidence, truncations, index


def _build_visual_evidence(
    vision_result: Optional[Dict[str, Any]], cfg: CritiqueConfig
) -> Tuple[List[VisualEvidence], List[str], Dict[str, Dict[str, Any]]]:
    """Vision observations carried through with their uncertainties intact."""
    evidence: List[VisualEvidence] = []
    truncations: List[str] = []
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(vision_result, dict):
        return evidence, truncations, index

    results = _dict_list(vision_result.get("results"))
    kept = results[: max(0, cfg.max_visual_assets)]
    dropped = len(results) - len(kept)
    for asset in kept:
        asset_id = _text(asset.get("asset_id") or asset.get("source_id"))
        if not asset_id:
            continue
        page = _as_int(asset.get("page_number") or asset.get("page"))
        observation, obs_cut = _truncate(
            asset.get("observation"), cfg.max_asset_field_chars
        )
        interpretation, int_cut = _truncate(
            asset.get("interpretation"), cfg.max_asset_field_chars
        )
        uncertainties = []
        for item in asset.get("uncertainties") or []:
            cleaned, _ = _truncate(item, cfg.max_asset_field_chars)
            if cleaned:
                uncertainties.append(cleaned)
        evidence.append(
            VisualEvidence(
                source_id=asset_id,
                asset_type=_text(asset.get("asset_type")),
                page=page,
                section=_text(asset.get("section")) or None,
                observation=observation,
                interpretation=interpretation,
                uncertainties=uncertainties,
            )
        )
        _add_source(
            index,
            "vision",
            asset_id,
            page=page,
            label=_text(asset.get("caption")) or _text(asset.get("section")),
        )
        if obs_cut or int_cut:
            truncations.append(
                f"visual evidence: {asset_id} truncated to "
                "CRITIQUE_MAX_ASSET_FIELD_CHARS"
            )

    if dropped > 0:
        truncations.append(
            f"visual evidence: {dropped} analyzed asset(s) beyond "
            "CRITIQUE_MAX_VISUAL_ASSETS were left out"
        )
    return evidence, truncations, index


def _statement_list(items: Any, limit: int, max_chars: int) -> Tuple[List[str], bool]:
    """Coerce a list of statement-like objects into clean text (order preserved)."""
    out: List[str] = []
    truncated_any = False
    if not isinstance(items, (list, tuple)):
        return out, truncated_any
    for item in list(items)[: max(0, limit)]:
        if isinstance(item, dict):
            raw = item.get("statement") or item.get("question") or item.get("gap")
        else:
            raw = item
        cleaned, was_cut = _truncate(raw, max_chars)
        if cleaned:
            out.append(cleaned)
        if was_cut:
            truncated_any = True
    return out, truncated_any


def _string_list(items: Any, limit: int, max_chars: int) -> List[str]:
    out: List[str] = []
    if not isinstance(items, (list, tuple)):
        return out
    for item in list(items)[: max(0, limit)]:
        cleaned, _ = _truncate(item, max_chars)
        if cleaned:
            out.append(cleaned)
    return out


def _evidence_ids(value: Any) -> List[str]:
    """Collect the upstream source ids a Phase 5 record pointed at."""
    ids: List[str] = []
    if not isinstance(value, (list, tuple)):
        return ids
    for item in value:
        if isinstance(item, dict):
            candidate = item.get("source_id") or item.get("id")
        else:
            candidate = item
        cleaned = _text(candidate)
        if cleaned and cleaned not in ids:
            ids.append(cleaned)
    return ids


def _collect_referenced_ids(value: Any) -> List[str]:
    """
    Recursively collect every ``source_id`` referenced anywhere in the Phase 5 payload.

    The critique may legitimately cite any evidence the analysis stage itself cited,
    including ids that appear only inside a finding, a strength or a limitation.
    """
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            candidate = node.get("source_id") or node.get("id")
            cleaned = _text(candidate)
            if cleaned and cleaned not in found:
                found.append(cleaned)
            for item in node.values():
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return found


def _build_analysis_summary(
    analysis_result: Optional[Dict[str, Any]], cfg: CritiqueConfig
) -> Tuple[AnalysisSummary, List[str], Dict[str, Dict[str, Any]]]:
    """
    Project the structured Phase 5 payload into synthesis-ready records.

    Every judgement here is Phase 5's: statuses, strengths, weaknesses, limitation
    categories and gaps are copied, never re-derived or re-classified.
    """
    summary = AnalysisSummary()
    truncations: List[str] = []
    index: Dict[str, Dict[str, Any]] = {}

    if not isinstance(analysis_result, dict):
        return summary, truncations, index

    payload = analysis_result.get("analysis")
    payload = payload if isinstance(payload, dict) else {}

    for key in list(DIMENSION_TO_SECTION)[: max(0, cfg.max_dimensions)]:
        block = payload.get(key)
        if not isinstance(block, dict) or not block:
            continue
        statements, cut = _statement_list(
            block.get("findings"),
            cfg.max_findings_per_dimension,
            cfg.max_statement_chars,
        )
        summary_text, summary_cut = _truncate(
            block.get("summary"), cfg.max_statement_chars
        )
        summary.dimensions.append(
            AnalysisDimensionSummary(
                dimension=key,
                summary=summary_text,
                findings=statements,
                uncertainties=_string_list(
                    block.get("uncertainties"), 10, cfg.max_statement_chars
                ),
                evidence_ids=_evidence_ids(block.get("evidence_refs")),
            )
        )
        if cut or summary_cut:
            truncations.append(
                f"analysis dimension '{key}' truncated to the configured "
                "per-dimension finding and statement limits"
            )

    matrix = payload.get("claim_evidence_matrix")
    matrix = matrix if isinstance(matrix, dict) else {}
    for item in _dict_list(matrix.get("items"))[: max(0, cfg.max_claims)]:
        claim_text, _ = _truncate(item.get("claim"), cfg.max_statement_chars)
        assessment, _ = _truncate(item.get("assessment"), cfg.max_statement_chars)
        if not claim_text:
            continue
        evidence_ids = _evidence_ids(item.get("supporting_evidence")) + _evidence_ids(
            item.get("contradicting_evidence")
        )
        deduped: List[str] = []
        for identifier in evidence_ids:
            if identifier not in deduped:
                deduped.append(identifier)
        status = _text(item.get("status")).lower() or "unclear"
        summary.claims.append(
            AnalysisClaim(
                claim=claim_text,
                status=status,  # type: ignore[arg-type]
                assessment=assessment,
                evidence_ids=deduped,
                missing_evidence=_string_list(
                    item.get("missing_evidence"), 10, cfg.max_statement_chars
                ),
            )
        )

    summary.strengths, strengths_cut = _statement_list(
        payload.get("strengths"),
        cfg.max_findings_per_dimension,
        cfg.max_statement_chars,
    )
    summary.weaknesses, weaknesses_cut = _statement_list(
        payload.get("weaknesses"),
        cfg.max_findings_per_dimension,
        cfg.max_statement_chars,
    )
    if strengths_cut or weaknesses_cut:
        truncations.append(
            "analysis strengths/weaknesses truncated to the configured per-item limit"
        )

    limitations = payload.get("limitations")
    limitations = limitations if isinstance(limitations, dict) else {}
    summary.author_stated_limitations, _ = _statement_list(
        limitations.get("author_stated"),
        cfg.max_findings_per_dimension,
        cfg.max_statement_chars,
    )
    summary.analyst_identified_limitations, _ = _statement_list(
        limitations.get("analyst_identified"),
        cfg.max_findings_per_dimension,
        cfg.max_statement_chars,
    )

    summary.open_questions, _ = _statement_list(
        payload.get("open_questions"),
        cfg.max_open_questions,
        cfg.max_statement_chars,
    )

    overall = payload.get("overall_assessment")
    overall = overall if isinstance(overall, dict) else {}
    summary.overall_assessment, _ = _truncate(
        overall.get("summary"), cfg.max_statement_chars
    )

    gaps = analysis_result.get("evidence_gaps")
    if not isinstance(gaps, (list, tuple)):
        gaps = payload.get("evidence_gaps")
    for gap in list(gaps or [])[: max(0, cfg.max_gaps)]:
        if not isinstance(gap, dict):
            continue
        text, _ = _truncate(gap.get("gap"), cfg.max_statement_chars)
        if not text:
            continue
        impact, _ = _truncate(gap.get("impact"), cfg.max_statement_chars)
        summary.evidence_gaps.append(
            CritiqueGap(
                gap=text,
                category=_text(gap.get("category")) or "other",
                impact=impact,
                source="analysis",
            )
        )

    # Every upstream id the analysis pointed at becomes a citable "analysis"
    # source, so a citation written against Phase 5 output resolves as well.
    #
    # The walk is deliberately exhaustive rather than limited to the dimension and
    # claim records: Phase 5 also cites evidence inside individual findings,
    # strengths, weaknesses, limitations and open questions, and any of those ids
    # is real provenance the critique must be allowed to quote. Collecting only the
    # dimension-level refs would make the validator report a genuinely cited id as
    # a fabricated one.
    for identifier in _collect_referenced_ids(payload):
        _add_source(index, "analysis", identifier)

    return summary, truncations, index


def _supported_sections(
    analysis: AnalysisSummary, visuals: List[VisualEvidence]
) -> SupportedSections:
    """
    Decide which report sections the supplied evidence can honestly carry.

    A section is supported when Phase 5 actually produced content for it (or, for
    the visual section, when Vision produced an observation). This is what lets the
    engine omit an unsupported section instead of padding it with filler.
    """
    reasons: Dict[str, str] = {}
    for entry in analysis.dimensions:
        section_id = DIMENSION_TO_SECTION.get(entry.dimension)
        if not section_id or section_id in reasons:
            continue
        if entry.summary or entry.findings:
            reasons[section_id] = (
                f"the analysis stage produced findings for '{entry.dimension}'"
            )
    if analysis.claims:
        reasons["claim_evidence_assessment"] = (
            f"{len(analysis.claims)} claim/evidence row(s) were assessed in Phase 5"
        )
    if analysis.strengths:
        reasons["strengths"] = f"{len(analysis.strengths)} strength(s) were recorded"
    if analysis.author_stated_limitations:
        reasons["author_stated_limitations"] = (
            "the supplied analysis contains author-stated limitations"
        )
    if analysis.analyst_identified_limitations:
        reasons["analyst_identified_limitations"] = (
            "the analysis stage identified limitations"
        )
    if visuals:
        reasons["figures_and_tables"] = (
            f"{len(visuals)} visual asset(s) were analyzed by the vision stage"
        )
    if analysis.evidence_gaps:
        reasons["evidence_gaps"] = (
            f"{len(analysis.evidence_gaps)} evidence gap(s) were recorded"
        )
    if analysis.open_questions:
        reasons["open_questions"] = (
            f"{len(analysis.open_questions)} open question(s) were recorded"
        )
    return SupportedSections(section_ids=list(reasons.keys()), reasons=reasons)


def _deterministic_gaps(
    context: SynthesisContext, analysis_result: Optional[Dict[str, Any]]
) -> List[CritiqueGap]:
    """
    Gaps the engine can state as fact about its own inputs.

    These are never left to the model's discretion: if retrieval or vision did not
    run, the critique says so rather than quietly omitting the topic.
    """
    gaps: List[CritiqueGap] = []
    routing = context.routing

    if not context.retrieved_evidence:
        if not routing.rag_enabled:
            gaps.append(
                CritiqueGap(
                    gap="Retrieval was not enabled for this paper, so no retrieved "
                    "text passage is available as primary provenance.",
                    category="rag_disabled",
                    impact="Provenance rests on the document profile and the "
                    "analysis stage alone.",
                    source="deterministic_context_check",
                )
            )
        else:
            gaps.append(
                CritiqueGap(
                    gap="Retrieval produced no usable passage for this paper.",
                    category="rag_unavailable",
                    impact="Specific textual citations cannot be verified.",
                    source="deterministic_context_check",
                )
            )

    if not context.visual_evidence:
        if not routing.vision_enabled:
            gaps.append(
                CritiqueGap(
                    gap="Visual analysis was not enabled for this paper, so no "
                    "figure or table observation is available.",
                    category="vision_disabled",
                    impact="Figures and tables are described only as far as the "
                    "analysis stage recorded them.",
                    source="deterministic_context_check",
                )
            )
        else:
            gaps.append(
                CritiqueGap(
                    gap="Visual analysis produced no observation for this paper.",
                    category="vision_unavailable",
                    impact="Visual claims cannot be checked against an image.",
                    source="deterministic_context_check",
                )
            )

    if not context.analysis.dimensions and not context.analysis.claims:
        gaps.append(
            CritiqueGap(
                gap="The analysis stage produced no structured finding to "
                "synthesize.",
                category="model_uncertainty",
                impact="The critique is necessarily high-level and cannot make "
                "specific evidence-grounded statements.",
                source="deterministic_context_check",
            )
        )

    unverified = 0
    if isinstance(analysis_result, dict):
        unverified = _as_int(analysis_result.get("unverified_evidence_refs")) or 0
    if unverified:
        gaps.append(
            CritiqueGap(
                gap=(
                    f"The analysis stage reported {unverified} citation(s) it could "
                    "not verify against its own evidence bundle."
                ),
                category="unverified_reference",
                impact="Statements resting on those citations inherit the doubt.",
                source="deterministic_context_check",
            )
        )

    if context.truncations:
        gaps.append(
            CritiqueGap(
                gap=(
                    "Part of the available structured input was omitted to stay "
                    "within the configured synthesis budget ("
                    + "; ".join(context.truncations[:3])
                    + ")."
                ),
                category="context_truncated",
                impact="The critique is bounded by the summarized input rather "
                "than by the whole evidence base.",
                source="deterministic_context_check",
            )
        )
    return gaps


def _measure_context_chars(context: SynthesisContext) -> int:
    total = 0
    for chunk in context.retrieved_evidence:
        total += len(chunk.excerpt)
    for asset in context.visual_evidence:
        total += len(asset.observation) + len(asset.interpretation)
        total += sum(len(item) for item in asset.uncertainties)
    for dimension in context.analysis.dimensions:
        total += len(dimension.summary) + sum(len(item) for item in dimension.findings)
    for claim in context.analysis.claims:
        total += len(claim.claim) + len(claim.assessment)
    total += sum(len(item) for item in context.analysis.strengths)
    total += sum(len(item) for item in context.analysis.weaknesses)
    total += sum(len(item) for item in context.analysis.author_stated_limitations)
    total += sum(len(item) for item in context.analysis.analyst_identified_limitations)
    total += sum(len(item) for item in context.analysis.open_questions)
    total += len(context.analysis.overall_assessment)
    return total


def _enforce_char_budget(context: SynthesisContext, cfg: CritiqueConfig) -> None:
    """Drop retrieved excerpts from the tail of the ranked list until it holds."""
    if cfg.max_context_chars <= 0 or context.total_context_chars <= cfg.max_context_chars:
        return
    while (
        context.retrieved_evidence
        and context.total_context_chars > cfg.max_context_chars
    ):
        dropped = context.retrieved_evidence.pop()
        context.truncations.append(
            f"synthesis context: dropped chunk '{dropped.source_id}' to respect "
            "CRITIQUE_MAX_CONTEXT_CHARS"
        )
        context.total_context_chars = _measure_context_chars(context)


def build_synthesis_context(
    document_profile: Optional[Dict[str, Any]],
    routing_state: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    analysis_result: Optional[Dict[str, Any]] = None,
    config: Optional[CritiqueConfig] = None,
) -> SynthesisContext:
    """
    Assemble the deterministic synthesis context for one critique execution.

    Raises
    ------
    ContextBuilderError when no structured analysis is available, i.e. when there
    is nothing to synthesize; the analysis stage must run first. A missing Document
    Profile alone is not fatal: the critique can still report the analysis
    findings, with the missing document provenance recorded as a gap.
    """
    cfg = config or CritiqueConfig()
    profile = document_profile if isinstance(document_profile, dict) else {}

    has_analysis = False
    if isinstance(analysis_result, dict):
        payload = analysis_result.get("analysis")
        has_analysis = isinstance(payload, dict) and bool(payload)
    if not has_analysis:
        raise ContextBuilderError(
            "No structured analysis result was supplied, so the Critique Engine "
            "has nothing to synthesize. The analysis stage must run first."
        )

    document, document_index = _build_document_summary(profile)
    if not profile:
        logger.warning(
            "Critique Engine received no document profile; document-level "
            "provenance will be unavailable."
        )

    retrieved, rag_truncations, rag_index = _build_retrieved_evidence(rag_result, cfg)
    visuals, vision_truncations, vision_index = _build_visual_evidence(
        vision_result, cfg
    )
    analysis, analysis_truncations, analysis_index = _build_analysis_summary(
        analysis_result, cfg
    )

    context = SynthesisContext(
        document=document,
        routing=_build_routing_summary(routing_state),
        retrieved_evidence=retrieved,
        visual_evidence=visuals,
        analysis=analysis,
        supported_sections=_supported_sections(analysis, visuals),
        truncations=rag_truncations + vision_truncations + analysis_truncations,
    )

    # Later stages win when the same id exists in several artifacts: real chunk or
    # asset metadata beats the abstract analysis-level record of the same id.
    source_index: Dict[str, Dict[str, Any]] = {}
    for group in (analysis_index, document_index, rag_index, vision_index):
        for key, entry in group.items():
            source_index[key] = entry
    context.source_index = dict(sorted(source_index.items()))

    context.counts = {
        "document_sources": len(document_index),
        "rag_sources": len(retrieved),
        "vision_sources": len(visuals),
        "analysis_sources": len(analysis_index),
        "dimensions": len(analysis.dimensions),
        "claims": len(analysis.claims),
        "strengths": len(analysis.strengths),
        "weaknesses": len(analysis.weaknesses),
        "author_stated_limitations": len(analysis.author_stated_limitations),
        "analyst_identified_limitations": len(analysis.analyst_identified_limitations),
        "evidence_gaps": len(analysis.evidence_gaps),
        "open_questions": len(analysis.open_questions),
        "supported_sections": len(context.supported_sections.section_ids),
    }

    context.total_context_chars = _measure_context_chars(context)
    _enforce_char_budget(context, cfg)
    context.deterministic_gaps = _deterministic_gaps(context, analysis_result)

    if not profile:
        context.deterministic_gaps.append(
            CritiqueGap(
                gap=(
                    "The document profile was not supplied, so document-level "
                    "facts (page count, structure, inventory) are unavailable."
                ),
                category="other",
                impact="The critique cannot state document-level facts, and page "
                "numbers cannot be checked against the document.",
                source="deterministic_context_check",
            )
        )
    return context