"""
Evidence context assembly for the Analysis Agent (Phase 5).

What this module does
---------------------
It converts the artifacts the Analysis Agent receives

    frozen Document Profile  +  JEV routing state  +  RAG result  +  Vision result

into one bounded ``EvidenceBundle``: a stable, citable view of the evidence with
an explicit ``source_index`` and explicitly recorded gaps.

What this module deliberately does NOT do
-----------------------------------------
It never re-detects sections/figures/tables/equations, never re-extracts text,
never re-chunks, never embeds, never searches and never re-runs visual analysis.
Section and inventory structure comes from the frozen Document Pre-Analyzer; text
evidence comes from the frozen RAG Agent; visual evidence comes from the frozen
Vision Agent. Anything those stages did not provide becomes a recorded
``EvidenceGap`` instead of being silently recreated here.

Missing RAG / Vision output is a first-class case, not an error: the bundle simply
carries the corresponding deterministic gap so the analysis model is told what it
cannot know.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.analysis_agent.config import AnalysisConfig
from backend.analysis_agent.schemas import (
    EvidenceBundle,
    EvidenceGap,
    InventoryItemRef,
    SectionRef,
    TextEvidenceRef,
    VisualEvidenceRef,
)

logger = logging.getLogger(__name__)

DETERMINISTIC_GAP_SOURCE = "deterministic_context_check"


class ContextBuilderError(Exception):
    """Raised when the evidence context cannot be assembled at all."""

    pass


def _text(value: Any) -> str:
    """Collapse any value into clean single-line text."""
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _truncate(value: Any, limit: int) -> Tuple[str, bool]:
    """Truncate to ``limit`` characters; returns (text, was_truncated)."""
    cleaned = _text(value)
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned, False
    return cleaned[: max(0, limit - 3)].rstrip() + "...", True


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
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
    Read the confirmed inventory of a Document Profile block.

    The frozen profile exposes ``<confirmed_key>`` next to a generic ``items``
    list; the confirmed key wins because that is the audited inventory.
    """
    if not isinstance(block, dict):
        return []
    confirmed = _dict_list(block.get(confirmed_key))
    if confirmed:
        return confirmed
    return _dict_list(block.get("items"))


def _block_count(block: Any, confirmed: List[Dict[str, Any]]) -> int:
    if isinstance(block, dict):
        count = _as_int(block.get("count"))
        if count is not None and count >= 0:
            return count
    return len(confirmed)


def _gap(
    gap: str,
    category: str,
    impact: str,
    affected: Optional[List[str]] = None,
) -> EvidenceGap:
    """Build a deterministic context gap (never model generated)."""
def _gap(
    gap: str,
    category: str,
    impact: str,
    affected: Optional[List[str]] = None,
) -> EvidenceGap:
    """Build a deterministic context gap (never model generated)."""
    return EvidenceGap(
        gap=gap,
        category=category,
        impact=impact,
        affected_dimensions=affected or [],
        source=DETERMINISTIC_GAP_SOURCE,
    )


def _build_sections(profile: Dict[str, Any], cfg: AnalysisConfig) -> Tuple[List[SectionRef], List[str]]:
    """Map confirmed profile sections into stable, citable context references."""
    raw = _dict_list(profile.get("sections"))
    sections: List[SectionRef] = []
    for item in raw:
        if len(sections) >= cfg.max_sections:
            break
        title, _ = _truncate(
            item.get("title") or item.get("normalized_title"),
            cfg.max_section_title_chars,
        )
        level = _as_int(item.get("level")) or 1
        page = _as_int(item.get("page"))
        sections.append(
            SectionRef(
                source_id=f"section_{len(sections) + 1:03d}",
                title=title or "(untitled section)",
                normalized_title=_text(item.get("normalized_title")),
                number=_text(item.get("number")),
                page=page if page and page >= 1 else None,
                level=max(1, level),
                confidence=_as_float(item.get("confidence")),
            )
        )

    notes: List[str] = []
    if len(raw) > len(sections):
        notes.append(
            f"sections: {len(raw)} available, {len(sections)} included "
            "(ANALYSIS_MAX_SECTIONS)"
        )
    return sections, notes


def _build_inventory(
    items: List[Dict[str, Any]],
    item_type: str,
    cfg: AnalysisConfig,
    limit: int,
) -> Tuple[List[InventoryItemRef], int, int]:
    """
    Map a confirmed profile inventory (figures / tables / equations) to context.

    Returns (references, dropped_count, truncated_caption_count).
    """
    refs: List[InventoryItemRef] = []
    truncated_captions = 0
    for item in items:
        if len(refs) >= limit:
            break
        caption, caption_truncated = _truncate(item.get("caption"), cfg.max_caption_chars)
        truncated_captions += 1 if caption_truncated else 0
        page = _as_int(item.get("page"))
        label = _text(
            item.get("figure_number")
            or item.get("table_number")
            or item.get("equation_number")
        )
        description = None
        if item_type == "equation":
            description, _ = _truncate(
                item.get("representation"), cfg.max_equation_chars
            )
        refs.append(
            InventoryItemRef(
                source_id=_text(item.get("id")) or f"{item_type}_{len(refs) + 1}",
                item_type=item_type,
                label=label or None,
                page=page if page and page >= 1 else None,
                caption=caption or None,
                description=description or None,
                structure_source=_text(item.get("structure_source")) or None,
                rows=_as_int(item.get("rows")),
                columns=_as_int(item.get("columns")),
                confidence=_as_float(item.get("confidence")),
            )
        )
    return refs, max(0, len(items) - len(refs)), truncated_captions


def _references_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Section-level reference statistics exactly as the frozen profile reports them."""
    block = profile.get("references")
    if not isinstance(block, dict):
        return {}
    return {
        "has_references": bool(block.get("has_references", False)),
        "count": _as_int(block.get("count")) or 0,
        "start_page": _as_int(block.get("start_page")),
        "confidence": _as_float(block.get("confidence")),
        "is_sequential": bool(block.get("is_sequential", False)),
    }


def _string_list(value: Any, item_limit: int = 160) -> List[str]:
    """Coerce a model/vision string list into bounded clean strings."""
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        return []
    cleaned: List[str] = []
    for candidate in candidates:
        text, _ = _truncate(candidate, item_limit)
        if text:
            cleaned.append(text)
    return cleaned


def _build_text_evidence(
    rag_result: Optional[Dict[str, Any]], cfg: AnalysisConfig
) -> Tuple[List[TextEvidenceRef], int, int]:
    """
    Consume the frozen RAG Agent's retrieved chunks (already rank-ordered).

    Returns (references, dropped_count, truncated_text_count).
    """
    if not isinstance(rag_result, dict) or not rag_result.get("enabled"):
        return [], 0, 0

    raw = _dict_list(rag_result.get("results"))
    refs: List[TextEvidenceRef] = []
    truncated = 0
    for item in raw:
        if len(refs) >= cfg.max_rag_chunks:
            break
        text, was_truncated = _truncate(item.get("text"), cfg.max_chunk_chars)
        if not text:
            continue
        truncated += 1 if was_truncated else 0
        page_start = _as_int(item.get("page_start")) or 1
        page_end = _as_int(item.get("page_end")) or page_start
        refs.append(
            TextEvidenceRef(
                source_id=f"chunk_{len(refs) + 1:03d}",
                chunk_id=_text(item.get("chunk_id")) or f"chunk_{len(refs) + 1:03d}",
                score=_as_float(item.get("score")),
                page_start=max(1, page_start),
                page_end=max(1, page_end),
                section=_text(item.get("section")) or None,
                text=text,
                word_count=_as_int(item.get("word_count")) or 0,
            )
        )
    return refs, max(0, len(raw) - len(refs)), truncated


def _render_structured_values(structured: Any, limit: int) -> Optional[str]:
    """
    Render the Vision Agent's machine-readable cell values, when it had them.

    Exact numbers in a table are authoritative only when a PDF text layer produced
    them; the Vision Agent records that in ``authoritative_for_exact_values``. When
    it is not set, no numeric rendering is produced here, so the analysis model
    cannot mistake an interpretation for a machine-read value.
    """
    if not isinstance(structured, dict):
        return None
    if not structured.get("authoritative_for_exact_values"):
        return None
    parts: List[str] = []
    headers = structured.get("headers")
    if isinstance(headers, (list, tuple)) and headers:
        parts.append("headers: " + " | ".join(_text(h) for h in headers))
    rows = structured.get("rows")
    if isinstance(rows, (list, tuple)):
        for row in rows:
            if isinstance(row, (list, tuple)):
                rendered = " | ".join(_text(cell) for cell in row)
                if rendered.strip(" |"):
                    parts.append("row: " + rendered)
    joined = " ; ".join(part for part in parts if part)
    if not joined:
        return None
    text, _ = _truncate(joined, limit)
    return text or None


def _build_visual_evidence(
    vision_result: Optional[Dict[str, Any]], cfg: AnalysisConfig
) -> Tuple[List[VisualEvidenceRef], int]:
    """
    Consume the frozen Vision Agent's structured visual analyses.

    Observations and interpretations stay in separate fields: an interpretation
    must never be readable downstream as a visible fact.
    """
    if not isinstance(vision_result, dict) or not vision_result.get("enabled"):
        return [], 0

    raw = _dict_list(vision_result.get("results"))
    refs: List[VisualEvidenceRef] = []
    limit = cfg.max_asset_field_chars
    for item in raw:
        if len(refs) >= cfg.max_visual_assets:
            break
        asset_id = _text(item.get("asset_id")) or f"asset_{len(refs) + 1}"
        observation, _ = _truncate(item.get("observation"), limit)
        interpretation, _ = _truncate(item.get("interpretation"), limit)
        caption, _ = _truncate(item.get("caption"), limit)
        consistency = item.get("caption_consistency")
        consistency_status = (
            _text(consistency.get("status")) if isinstance(consistency, dict) else ""
        )
        page = _as_int(item.get("page_number"))
        refs.append(
            VisualEvidenceRef(
                source_id=f"visual_{asset_id}",
                asset_id=asset_id,
                asset_type=_text(item.get("asset_type")) or "figure",
                page_number=page if page and page >= 1 else None,
                section=_text(item.get("section")) or None,
                caption=caption or None,
                observation=observation,
                interpretation=interpretation,
                key_elements=_string_list(item.get("key_elements")),
                reported_relationships=_string_list(item.get("reported_relationships")),
                supports_claims=_string_list(item.get("supports_claims")),
                uncertainties=_string_list(item.get("uncertainties"), item_limit=240),
                caption_consistency_status=consistency_status or None,
                numeric_authority=_text(item.get("numeric_authority")) or None,
                structured_values=_render_structured_values(
                    item.get("structured_evidence"), limit
                ),
                source_confidence=_as_float(item.get("source_confidence")),
            )
        )
    return refs, max(0, len(raw) - len(refs))


ANALYSIS_DIMENSIONS_AFFECTED_BY_TEXT = [
    "methodology",
    "data",
    "results",
    "reproducibility",
    "internal_consistency",
]
ANALYSIS_DIMENSIONS_AFFECTED_BY_VISUALS = [
    "methodology",
    "results",
    "internal_consistency",
]
ANALYSIS_DIMENSIONS_AFFECTED_BY_NUMBERS = ["results", "metrics", "internal_consistency"]


def _read_routing_flags(routing_state: Any) -> Tuple[bool, bool]:
    """Read rag.enabled / vision.enabled from the frozen routing state."""
    state = routing_state if isinstance(routing_state, dict) else {}
    rag_state = state.get("rag")
    vision_state = state.get("vision")
    rag_enabled = (
        bool(rag_state.get("enabled", False))
        if isinstance(rag_state, dict)
        else bool(state.get("rag_enabled", False))
    )
    vision_enabled = (
        bool(vision_state.get("enabled", False))
        if isinstance(vision_state, dict)
        else bool(state.get("vision_enabled", False))
    )
    return rag_enabled, vision_enabled


def _build_rag_gaps(
    bundle: EvidenceBundle, rag_result: Optional[Dict[str, Any]]
) -> List[EvidenceGap]:
    """Deterministic gaps describing the state of the textual evidence."""
    if rag_result is None:
        return [
            _gap(
                "No RAG result was provided to the Analysis Agent.",
                "rag_unavailable",
                "Verbatim textual support is limited to section headings, captions and "
                "reference statistics; the paper's own wording cannot be quoted.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_TEXT,
            )
        ]
    if not isinstance(rag_result, dict) or not rag_result.get("enabled"):
        return [
            _gap(
                "Retrieval over the paper text was disabled by the JEV Router decision.",
                "rag_disabled",
                "The analysis rests on the confirmed document structure and captions "
                "only; no retrieved passage can be cited for the paper's wording.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_TEXT,
            )
        ]
    if not bundle.text_evidence:
        return [
            _gap(
                "The RAG result contained no usable retrieved text chunk.",
                "rag_unavailable",
                "No verbatim textual evidence is available even though retrieval was "
                "enabled, so wording-level findings cannot be grounded.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_TEXT,
            )
        ]
    return []


def _build_vision_gaps(
    bundle: EvidenceBundle, vision_result: Optional[Dict[str, Any]]
) -> List[EvidenceGap]:
    """Deterministic gaps describing the state of the visual evidence."""
    if vision_result is None:
        return [
            _gap(
                "No Vision result was provided to the Analysis Agent.",
                "vision_unavailable",
                "Figure/table content is known only through captions and structural "
                "metadata; visual claims cannot be checked.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_VISUALS,
            )
        ]
    if not isinstance(vision_result, dict) or not vision_result.get("enabled"):
        return [
            _gap(
                "Visual analysis was disabled by the JEV Router decision.",
                "vision_disabled",
                "What the figures and tables actually display is unverified; only "
                "captions and counts support the visual dimensions.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_VISUALS,
            )
        ]
    if not bundle.visual_evidence:
        return [
            _gap(
                "No visual asset was successfully analyzed even though vision was enabled.",
                "vision_unavailable",
                "Visual claims rest on captions alone.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_VISUALS,
            )
        ]
    return []


def _build_numeric_gaps(bundle: EvidenceBundle) -> List[EvidenceGap]:
    """
    Report the case where no trustworthy numeric evidence exists at all.

    Exact numbers are only safe to state when the PDF text layer produced them
    (carried by the Vision Agent's structured cell values) or when a retrieved
    chunk exposes them.
    """
    has_machine_values = any(visual.structured_values for visual in bundle.visual_evidence)
    if bundle.tables and not has_machine_values and not bundle.text_evidence:
        return [
            _gap(
                "No machine-readable values and no retrieved text were available for "
                f"the {len(bundle.tables)} confirmed table(s).",
                "numeric_values_unavailable",
                "Exact numbers must not be asserted: only table structure, captions and "
                "any visual reading can be discussed.",
                ANALYSIS_DIMENSIONS_AFFECTED_BY_NUMBERS,
            )
        ]
    return []


def _build_structure_gaps(bundle: EvidenceBundle) -> List[EvidenceGap]:
    if bundle.sections:
        return []
    return [
        _gap(
            "The Document Profile contains no confirmed section heading.",
            "other",
            "The paper's argument structure could not be reconstructed from headings, "
            "so dimension boundaries rely on counts and metadata only.",
            ["research_problem", "contribution", "methodology"],
        )
    ]


def _source_index(bundle: EvidenceBundle) -> Dict[str, Dict[str, Any]]:
    """Registry of every citable id, used to verify the model's references."""
    index: Dict[str, Dict[str, Any]] = {}

    def add(source_id: str, source_type: str, pages: List[int], detail: str) -> None:
        if not source_id or source_id in index:
            return
        text, _ = _truncate(detail, 140)
        index[source_id] = {
            "source_type": source_type,
            "pages": [page for page in pages if isinstance(page, int) and page >= 1],
            "detail": text,
        }

    add(
        "document_profile",
        "document_profile",
        [],
        f"{bundle.filename}: {bundle.page_count} page(s), {bundle.word_count} word(s)",
    )
    add(
        "paper_metadata",
        "metadata",
        [],
        _text(bundle.metadata.get("title")) or bundle.filename,
    )
    for section in bundle.sections:
        add(
            section.source_id,
            "section",
            [section.page] if section.page else [],
            section.title,
        )
    for item in [*bundle.figures, *bundle.tables, *bundle.equations]:
        label = f"{item.item_type} {item.label}" if item.label else item.item_type
        add(
            item.source_id,
            item.item_type,
            [item.page] if item.page else [],
            f"{label}: {item.caption or item.description or ''}",
        )
    if bundle.references_summary:
        start_page = bundle.references_summary.get("start_page")
        add(
            "references_summary",
            "reference",
            [start_page] if isinstance(start_page, int) else [],
            f"{bundle.references_summary.get('count', 0)} reference(s) detected",
        )
    for chunk in bundle.text_evidence:
        add(
            chunk.source_id,
            "text_chunk",
            sorted({chunk.page_start, chunk.page_end}),
            f"retrieved chunk {chunk.chunk_id}, score={chunk.score}",
        )
    for visual in bundle.visual_evidence:
        add(
            visual.source_id,
            "visual_asset",
            [visual.page_number] if visual.page_number else [],
            f"{visual.asset_type} {visual.asset_id}: {visual.caption or ''}",
        )
    return index


def _measure_evidence_chars(bundle: EvidenceBundle) -> int:
    """Character weight of the evidence actually handed to the model."""
    total = 0
    for section in bundle.sections:
        total += len(section.title) + len(section.normalized_title)
    for item in [*bundle.figures, *bundle.tables, *bundle.equations]:
        total += len(item.caption or "") + len(item.description or "")
    for chunk in bundle.text_evidence:
        total += len(chunk.text)
    for visual in bundle.visual_evidence:
        total += (
            len(visual.observation)
            + len(visual.interpretation)
            + len(visual.caption or "")
            + len(visual.structured_values or "")
        )
        total += sum(
            len(value)
            for value in (
                visual.key_elements
                + visual.reported_relationships
                + visual.supports_claims
                + visual.uncertainties
            )
        )
    return total


def _enforce_char_budget(bundle: EvidenceBundle, cfg: AnalysisConfig) -> None:
    """
    Keep the evidence bundle inside its configured character budget.

    Retrieved chunks arrive rank-ordered, so the lowest-ranked chunk is dropped
    first and the removal is recorded as both a truncation note and a
    ``context_truncated`` gap: omitted evidence must never look like absent
    evidence.
    """
    if cfg.max_context_chars <= 0:
        return

    dropped_chunks = 0
    dropped_visuals = 0
    while _measure_evidence_chars(bundle) > cfg.max_context_chars and bundle.text_evidence:
        bundle.text_evidence.pop()
        dropped_chunks += 1
    while _measure_evidence_chars(bundle) > cfg.max_context_chars and bundle.visual_evidence:
        bundle.visual_evidence.pop()
        dropped_visuals += 1

    if dropped_chunks:
        bundle.truncations.append(
            f"text evidence: {dropped_chunks} lowest-ranked chunk(s) dropped to stay "
            f"within ANALYSIS_MAX_CONTEXT_CHARS={cfg.max_context_chars}"
        )
    if dropped_visuals:
        bundle.truncations.append(
            f"visual evidence: {dropped_visuals} asset(s) dropped to stay within "
            f"ANALYSIS_MAX_CONTEXT_CHARS={cfg.max_context_chars}"
        )
    if dropped_chunks or dropped_visuals:
        bundle.deterministic_gaps.append(
            _gap(
                "Part of the evidence base was omitted to keep the analysis prompt "
                "within its configured budget.",
                "context_truncated",
                "Findings cannot rely on the omitted evidence, so the affected "
                "dimensions may appear weaker than the full evidence base would support.",
                ["results", "methodology", "data", "internal_consistency"],
            )
        )


def build_evidence_bundle(
    document_profile: Optional[Dict[str, Any]],
    rag_result: Optional[Dict[str, Any]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    routing_state: Optional[Dict[str, Any]] = None,
    config: Optional[AnalysisConfig] = None,
) -> EvidenceBundle:
    """
    Assemble the bounded evidence bundle for one Analysis Agent execution.

    Parameters
    ----------
    document_profile: frozen Document Pre-Analyzer profile (required)
    rag_result: frozen RAG Agent result, or None when retrieval did not run
    vision_result: frozen Vision Agent result, or None when vision did not run
    routing_state: read-only JEV routing decision state
    config: budget configuration (injectable in tests)

    Raises
    ------
    ContextBuilderError when the Document Profile is missing or unusable, i.e.
    when no evidence-grounded analysis is possible at all.
    """
    cfg = config or AnalysisConfig()
    if not isinstance(document_profile, dict) or not document_profile:
        raise ContextBuilderError(
            "The frozen Document Profile is missing or empty, so the Analysis Agent "
            "has no evidence base to reason over."
        )

    figures_block = document_profile.get("figures")
    tables_block = document_profile.get("tables")
    equations_block = document_profile.get("equations")

    sections, truncations = _build_sections(document_profile, cfg)

    figure_items = _confirmed_items(figures_block, "confirmed_figures")
    table_items = _confirmed_items(tables_block, "confirmed_tables")
    equation_items = _confirmed_items(equations_block, "confirmed_equations")

    figures, dropped_figures, truncated_figure_captions = _build_inventory(
        figure_items, "figure", cfg, cfg.max_inventory_items
    )
    tables, dropped_tables, truncated_table_captions = _build_inventory(
        table_items, "table", cfg, cfg.max_inventory_items
    )
    equations, dropped_equations, _ = _build_inventory(
        equation_items, "equation", cfg, cfg.max_equations
    )

    if dropped_figures:
        truncations.append(
            f"figures: {dropped_figures} confirmed figure(s) beyond "
            "ANALYSIS_MAX_INVENTORY_ITEMS were left out"
        )
    if dropped_tables:
        truncations.append(
            f"tables: {dropped_tables} confirmed table(s) beyond "
            "ANALYSIS_MAX_INVENTORY_ITEMS were left out"
        )
    if dropped_equations:
        truncations.append(
            f"equations: {dropped_equations} confirmed equation(s) beyond "
            "ANALYSIS_MAX_EQUATIONS were left out"
        )
    if truncated_figure_captions or truncated_table_captions:
        truncations.append(
            f"captions: "
            f"{truncated_figure_captions + truncated_table_captions} caption(s) "
            "truncated to ANALYSIS_MAX_CAPTION_CHARS"
        )

    text_evidence, dropped_chunks, truncated_chunks = _build_text_evidence(
        rag_result, cfg
    )
    if dropped_chunks:
        truncations.append(
            f"text evidence: {dropped_chunks} retrieved chunk(s) beyond "
            "ANALYSIS_MAX_RAG_CHUNKS were left out"
        )
    if truncated_chunks:
        truncations.append(
            f"text evidence: {truncated_chunks} chunk(s) truncated to "
            "ANALYSIS_MAX_CHUNK_CHARS"
        )

    visual_evidence, dropped_visuals = _build_visual_evidence(vision_result, cfg)
    if dropped_visuals:
        truncations.append(
            f"visual evidence: {dropped_visuals} analyzed asset(s) beyond "
            "ANALYSIS_MAX_VISUAL_ASSETS were left out"
        )

    references_summary = _references_summary(document_profile)
    metadata = document_profile.get("metadata")
    characteristics = document_profile.get("content_characteristics")

    bundle = EvidenceBundle(
        filename=_text(document_profile.get("filename")),
        page_count=_as_int(document_profile.get("page_count")) or 0,
        word_count=_as_int(document_profile.get("word_count")) or 0,
        text_length=_as_int(document_profile.get("text_length")) or 0,
        metadata=metadata if isinstance(metadata, dict) else {},
        content_characteristics=characteristics if isinstance(characteristics, dict) else {},
        sections=sections,
        figures=figures,
        tables=tables,
        equations=equations,
        references_summary=references_summary,
        text_evidence=text_evidence,
        visual_evidence=visual_evidence,
        truncations=truncations,
    )

    bundle.deterministic_gaps = (
        _build_rag_gaps(bundle, rag_result)
        + _build_vision_gaps(bundle, vision_result)
        + _build_numeric_gaps(bundle)
        + _build_structure_gaps(bundle)
    )

    _enforce_char_budget(bundle, cfg)

    bundle.counts = {
        "sections": len(bundle.sections),
        "sections_available": len(_dict_list(document_profile.get("sections"))),
        "figures": len(bundle.figures),
        "figures_available": _block_count(figures_block, figure_items),
        "tables": len(bundle.tables),
        "tables_available": _block_count(tables_block, table_items),
        "equations": len(bundle.equations),
        "equations_available": _block_count(equations_block, equation_items),
        "references": int(references_summary.get("count", 0) or 0),
        "rag_chunks": len(bundle.text_evidence),
        "visual_assets": len(bundle.visual_evidence),
    }
    bundle.source_index = _source_index(bundle)
    bundle.total_evidence_chars = _measure_evidence_chars(bundle)
    return bundle
