"""
Prompt construction for the Analysis Agent (Phase 5).

Prompting philosophy
--------------------
The model is asked for one bounded, structured, evidence-grounded evaluation of a
single paper: it receives the frozen Document Profile, the retrieved passages and
the visual analyses, and it must return one JSON object.

Three boundaries are enforced by the prompt and by the parser:

* SCOPE - this stage produces an evidence-grounded evaluation, not the final
  critique, ranking or verdict, and never an overall numeric score for the paper.
* PROVENANCE - every reference must copy a ``source_id`` from the citable source
  list; invented identifiers are detected and flagged by the analyzer.
* EVIDENCE DISCIPLINE - exact numbers may only be repeated when they appear in
  retrieved text or in machine-readable table values; anything else is reported as
  unavailable, and confidence values are never fabricated.

The prompt is built from the ``EvidenceBundle`` only: nothing is re-derived from
the PDF here, and the model never sees the whole paper.
"""

from typing import Dict, List

from backend.analysis_agent.schemas import EvidenceBundle

RULE_BLOCK = """You are the analysis stage of an automated research-paper critic. You receive a
bounded evidence bundle assembled by upstream tools and you produce a structured,
evidence-grounded evaluation of that paper.

You are NOT writing the final critique. Do not rank the paper, do not write a
review-style verdict, and do not produce an overall numeric score of any kind
(no "7/10", "4 out of 5", "82/100", no single number standing for paper quality).
Report what the available evidence supports, what it does not support and what is
missing.

Ground rules:
1. Use only the evidence in this prompt: document metadata, the section list, the
   confirmed figure/table/equation inventory, reference statistics, the retrieved
   text chunks and the visual analyses. Nothing else exists for you.
2. Never invent evidence. Every "source_id" you cite must appear verbatim in the
   "Citable source ids" list below, with the matching source_type and pages.
3. Never invent numbers. Repeat an exact number only when it appears in a retrieved
   text chunk or in provided machine-readable table values. Otherwise state that
   the value is not available in the evidence you were given.
4. Keep the registers apart: what the paper states, what the evidence shows, and
   what you infer. Word inferences as inferences ("this suggests", "appears to").
5. Never fabricate confidence. If you cannot justify a number between 0.0 and 1.0,
   omit the key or set it to null. A stated confidence must follow from the
   evidence, not from politeness.
6. "evidence_status" must be exactly one of: supported, partially_supported,
   unsupported, unclear, insufficient_evidence. Use "insufficient_evidence" when
   the provided evidence cannot decide the point, and "unsupported" when it
   contradicts the statement. Do not use "supported" without citing evidence.
7. Limitations the authors state and limitations you observe belong in separate
   fields ("author_stated" vs "analyst_identified"). Never attribute your own
   observation to the authors.
8. Every gap in the evidence base you were given (retrieval disabled or missing,
   no visual analysis, no machine-readable numbers, truncated context) must be
   recorded in "evidence_gaps" and respected in the affected dimensions. Do not
   compensate for missing evidence with speculation.
9. Return ONLY one JSON object matching the contract exactly - no markdown, no
   code fences, no commentary before or after the object.
"""

DIMENSION_FOCUS: Dict[str, str] = {
    "research_problem": (
        "1. What problem does the paper define, and which section states it?\n"
        "2. Why does the paper claim the problem matters?\n"
        "3. Is the problem statement evidenced, or only inferable from the title?"
    ),
    "contribution": (
        "1. Which contributions does the paper explicitly claim, and where?\n"
        "2. Which of them are demonstrated by evidence you were given?\n"
        "3. Which novelty claims remain unverifiable from this evidence?"
    ),
    "methodology": (
        "1. What method or architecture does the paper propose, per the evidence?\n"
        "2. Which methodological details are disclosed (equations, figures, settings) "
        "and which are absent?\n"
        "3. Which design choices are justified in the evidence, and which are not?"
    ),
    "data": (
        "1. Which datasets or corpora are described, and with what sizes or splits?\n"
        "2. Where does the data come from: public, private or unspecified?\n"
        "3. Which data details needed for replication are missing from the evidence?"
    ),
    "baselines": (
        "1. Which baselines or competing methods are compared against?\n"
        "2. Are the baselines strong and current according to the evidence?\n"
        "3. Is the comparison setup (same data, same metric, tuning effort) disclosed?"
    ),
    "metrics": (
        "1. Which metrics are reported, and how are they defined?\n"
        "2. Do the metrics correspond to the claimed goal in the evidence?\n"
        "3. Are evaluation conditions (splits, significance testing) reported?"
    ),
    "results": (
        "1. Which results are reported, and where exactly in the evidence?\n"
        "2. Do the reported results support the paper's main claims?\n"
        "3. Which results are ambiguous, missing or contradicted by other evidence?"
    ),
    "reproducibility": (
        "1. What is needed to reproduce the work (data, code, hyperparameters, "
        "hardware) and how much of it is disclosed?\n"
        "2. Are training and evaluation procedures described in enough detail?\n"
        "3. Are reference statistics consistent with claims about prior work?"
    ),
    "internal_consistency": (
        "1. Do different parts of the evidence agree (sections vs captions vs visual "
        "analyses)?\n"
        "2. Are numbers or definitions used inconsistently across the evidence?\n"
        "3. Do any claims overreach what the reported results can establish?"
    ),
    "claim_evidence_matrix": (
        "Map the central claims of the paper onto the evidence that exists for them.\n"
        "1. One entry per distinct claim; mark each as 'paper' (stated by the "
        "authors) or 'analyst_inference'.\n"
        "2. Put citing evidence in 'supporting_evidence' and anything that "
        "contradicts or weakens it in 'contradicting_evidence'.\n"
        "3. Name in 'missing_evidence' what an evaluation would need in order to "
        "accept or reject the claim."
    ),
    "strengths": (
        "Evidence-grounded strengths only: state them as neutral observations, cite "
        "the evidence and avoid praise language. A strength with no reference is not "
        "reportable."
    ),
    "weaknesses": (
        "Evidence-grounded weaknesses only: state them as neutral observations, cite "
        "the evidence and separate an actual weakness from evidence that is merely "
        "missing (missing evidence belongs in evidence_gaps)."
    ),
    "limitations": (
        "Split limitations explicitly.\n"
        "1. 'author_stated': restrictions the paper itself acknowledges (cite the "
        "section or passage).\n"
        "2. 'analyst_identified': restrictions you observe from the evidence, worded "
        "as your own assessment.\n"
        "3. Never attribute your own observation to the authors."
    ),
    "open_questions": (
        "1. Which questions matter for judging this work but cannot be answered from "
        "the evidence you were given?\n"
        "2. For each, say why the answer would change the reading and which "
        "dimension it relates to."
    ),
    "evidence_gaps": (
        "1. Record the gaps already listed in this prompt (they are facts about the "
        "run) plus any additional gap revealed by the evidence.\n"
        "2. Use the fitting category: rag_disabled, rag_unavailable, vision_disabled, "
        "vision_unavailable, numeric_values_unavailable, not_reported_in_paper, "
        "model_uncertainty, unverified_reference, no_numeric_paper_score, "
        "context_truncated or other.\n"
        "3. State the impact on the analysis, not only the missing item."
    ),
    "overall_assessment": (
        "1. Summarize what the evidence supports and where it is insufficient.\n"
        "2. Give no ranking and no numeric score.\n"
        "3. Cite the key basis references and list what you could not determine."
    ),
}

OUTPUT_CONTRACT = """Return ONLY this JSON object (every listed top-level key must be present):

{
  "research_problem": <DIMENSION>,
  "contribution": <DIMENSION>,
  "methodology": <DIMENSION>,
  "data": <DIMENSION>,
  "baselines": <DIMENSION>,
  "metrics": <DIMENSION>,
  "results": <DIMENSION>,
  "reproducibility": <DIMENSION>,
  "internal_consistency": <DIMENSION>,
  "claim_evidence_matrix": {
    "items": [
      {
        "claim": "string",
        "claim_source": "paper | analyst_inference",
        "status": <STATUS>,
        "supporting_evidence": [<REF>],
        "contradicting_evidence": [<REF>],
        "assessment": "string",
        "missing_evidence": ["string"],
        "confidence": 0.0
      }
    ],
    "notes": "string"
  },
  "strengths": [<POINT>],
  "weaknesses": [<POINT>],
  "limitations": {
    "author_stated": [<FINDING>],
    "analyst_identified": [<FINDING>]
  },
  "open_questions": [
    {
      "question": "string",
      "why_it_matters": "string",
      "related_dimension": "string or null",
      "evidence_refs": [<REF>]
    }
  ],
  "evidence_gaps": [
    {
      "gap": "string",
      "category": "string (see the evidence_gaps focus block)",
      "impact": "string",
      "affected_dimensions": ["string"]
    }
  ],
  "overall_assessment": {
    "summary": "string",
    "evidence_status": <STATUS>,
    "confidence": 0.0,
    "key_basis": [<REF>],
    "uncertainties": ["string"]
  }
}

Definitions:
  <DIMENSION> = {
    "dimension": "the dimension name",
    "summary": "required, non-empty string",
    "evidence_status": <STATUS>,
    "findings": [<FINDING>],
    "evidence_refs": [<REF>],
    "confidence": 0.0,
    "uncertainties": ["string"]
  }
  <FINDING> = {
    "statement": "required, non-empty string",
    "evidence_status": <STATUS>,
    "evidence_refs": [<REF>],
    "confidence": 0.0,
    "reasoning": "string",
    "caveats": ["string"]
  }
  <POINT> = {
    "statement": "required, non-empty string",
    "dimension": "string or null",
    "evidence_status": <STATUS>,
    "evidence_refs": [<REF>],
    "rationale": "string",
    "confidence": 0.0
  }
  <REF> = {
    "source_type": "the type shown for that id in Citable source ids",
    "source_id": "an id copied exactly from Citable source ids",
    "pages": [1],
    "detail": "short label copied from Citable source ids"
  }
  <STATUS> = "supported" | "partially_supported" | "unsupported" | "unclear" |
             "insufficient_evidence"

"confidence" is optional everywhere: omit it or use null instead of guessing. Any
statement you cannot ground must use the status "insufficient_evidence".
"""


def _clean(value: object) -> str:
    """Collapse any value into clean single-line text."""
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _pages_label(pages: List[int]) -> str:
    values = sorted({page for page in pages if isinstance(page, int) and page >= 1})
    if not values:
        return "pages unknown"
    if len(values) == 1:
        return f"p.{values[0]}"
    if values == list(range(values[0], values[-1] + 1)):
        return f"pp.{values[0]}-{values[-1]}"
    return "pp." + ", ".join(str(page) for page in values)


def render_paper_block(bundle: EvidenceBundle) -> str:
    """Render document-level facts (metadata, counts, content characteristics)."""
    title = _clean(bundle.metadata.get("title")) or "(no title in metadata)"
    authors = _clean(bundle.metadata.get("author")) or "(not stated in metadata)"
    characteristics = bundle.content_characteristics or {}
    flags = ", ".join(
        f"{key}={value}" for key, value in sorted(characteristics.items())
    ) or "(not reported)"
    counts = bundle.counts or {}
    lines = [
        "Paper facts (from the frozen Document Profile and the upstream agents):",
        f"- file: {bundle.filename or '(unknown)'}",
        f"- title (metadata): {title}",
        f"- authors (metadata): {authors}",
        f"- pages: {bundle.page_count}, words (text layer): {bundle.word_count}, "
        f"characters: {bundle.text_length}",
        f"- confirmed figures: {counts.get('figures_available', 0)}, "
        f"confirmed tables: {counts.get('tables_available', 0)}, "
        f"confirmed equations: {counts.get('equations_available', 0)}",
        f"- detected references: {counts.get('references', 0)}",
        f"- content characteristics: {flags}",
        f"- evidence actually supplied: {counts.get('sections', 0)} section(s), "
        f"{counts.get('rag_chunks', 0)} retrieved text chunk(s), "
        f"{counts.get('visual_assets', 0)} visual analysis(es)",
    ]
    return "\n".join(lines)


def render_section_block(bundle: EvidenceBundle) -> str:
    """Render the confirmed section list as structure (no re-detection)."""
    if not bundle.sections:
        return "Confirmed sections: none available."
    lines = ["Confirmed sections (order and titles exactly as the Document Profile reports):"]
    for section in bundle.sections:
        # The frozen profile keeps the printed number separately while the title
        # usually already contains it; never print it twice.
        number = (
            f"{section.number} "
            if section.number and not section.title.startswith(section.number)
            else ""
        )
        page = f"{_pages_label([section.page])}" if section.page else "page unknown"
        lines.append(
            f"- [{section.source_id}] depth {section.level} | {page} | "
            f"{number}{section.title}"
        )
    return "\n".join(lines)


def render_inventory_block(bundle: EvidenceBundle) -> str:
    """Render confirmed figures/tables/equations as context, not as re-detected items."""
    lines = ["Confirmed figures (captions as detected by the Document Pre-Analyzer):"]
    if not bundle.figures:
        lines.append("- none")
    for item in bundle.figures:
        label = f"Figure {item.label}" if item.label else "Figure"
        page = _pages_label([item.page]) if item.page else "page unknown"
        caption = _clean(item.caption) or "(no caption detected)"
        lines.append(f"- [{item.source_id}] {label} | {page} | {caption}")

    lines.append("")
    lines.append("Confirmed tables (structure is reported, cell values are only available")
    lines.append("when machine-readable values are listed under the visual evidence):")
    if not bundle.tables:
        lines.append("- none")
    for item in bundle.tables:
        label = f"Table {item.label}" if item.label else "Table"
        page = _pages_label([item.page]) if item.page else "page unknown"
        shape = (
            f" | reported shape {item.rows}x{item.columns}"
            if item.rows and item.columns
            else ""
        )
        source = f" | structure from {_clean(item.structure_source)}" if item.structure_source else ""
        caption = _clean(item.caption) or "(no caption detected)"
        lines.append(f"- [{item.source_id}] {label} | {page}{shape}{source} | {caption}")

    lines.append("")
    lines.append("Confirmed equations (representations are context; the Analysis Agent does")
    lines.append("not re-derive or re-render mathematics):")
    if not bundle.equations:
        lines.append("- none")
    for item in bundle.equations:
        label = f"Equation {item.label}" if item.label else "Equation"
        page = _pages_label([item.page]) if item.page else "page unknown"
        description = _clean(item.description) or "(no representation available)"
        lines.append(f"- [{item.source_id}] {label} | {page} | {description}")
    return "\n".join(lines)


def render_references_block(bundle: EvidenceBundle) -> str:
    """Render reference statistics only (bib entries are not in the profile)."""
    summary = bundle.references_summary
    if not summary:
        return "Reference statistics: not reported by the Document Profile."
    start_page = summary.get("start_page")
    location = f"starting {_pages_label([start_page])}" if isinstance(start_page, int) else "start page unknown"
    return (
        "Reference statistics (bib entries are not part of the profile):\n"
        f"- references detected: {summary.get('count', 0)} ({location}), sequential: "
        f"{summary.get('is_sequential')}, detection confidence: {summary.get('confidence')}\n"
        f"- cite as [references_summary] when the reference list itself supports a "
        "statement (e.g. coverage of related work)."
    )


def render_text_block(bundle: EvidenceBundle) -> str:
    """Render the retrieved passages handed over by the frozen RAG Agent."""
    header = (
        "Retrieved text evidence (from the frozen RAG Agent, rank-ordered; this is the "
        "ONLY source of the paper's verbatim wording in this prompt):"
    )
    if not bundle.text_evidence:
        return header + "\n- none available"
    lines = [header]
    for chunk in bundle.text_evidence:
        section = _clean(chunk.section) or "unlabelled section"
        lines.append(
            f"- [{chunk.source_id}] chunk {chunk.chunk_id} | "
            f"{_pages_label([chunk.page_start, chunk.page_end])} | {section} | "
            f"similarity={chunk.score} | {chunk.word_count} word(s)"
        )
        lines.append(f'  text: "{_clean(chunk.text)}"')
    return "\n".join(lines)


def render_visual_block(bundle: EvidenceBundle) -> str:
    """Render the visual analyses handed over by the frozen Vision Agent."""
    header = (
        "Visual evidence (from the frozen Vision Agent). 'observation' is what is "
        "visibly present, 'interpretation' is provisional and must never be reported as "
        "a visible fact:"
    )
    if not bundle.visual_evidence:
        return header + "\n- none available"
    lines = [header]
    for visual in bundle.visual_evidence:
        label = f"{visual.asset_type} {visual.asset_id}"
        section = _clean(visual.section) or "unlabelled section"
        lines.append(
            f"- [{visual.source_id}] {label} | "
            f"{_pages_label([visual.page_number]) if visual.page_number else 'page unknown'}"
            f" | {section} | numeric_authority={visual.numeric_authority}"
        )
        if visual.caption:
            lines.append(f"  caption: {_clean(visual.caption)}")
        lines.append(f"  observation: {_clean(visual.observation)}")
        lines.append(f"  interpretation: {_clean(visual.interpretation)}")
        if visual.key_elements:
            lines.append("  key elements: " + "; ".join(visual.key_elements))
        if visual.reported_relationships:
            lines.append(
                "  reported relationships: " + "; ".join(visual.reported_relationships)
            )
        if visual.supports_claims:
            lines.append(
                "  claims the visual appears to support (not proof): "
                + "; ".join(visual.supports_claims)
            )
        if visual.uncertainties:
            lines.append("  stated uncertainties: " + "; ".join(visual.uncertainties))
        if visual.caption_consistency_status:
            lines.append(
                f"  caption consistency: {visual.caption_consistency_status}"
            )
        if visual.structured_values:
            lines.append(
                "  machine-readable values (authoritative for exact numbers): "
                + visual.structured_values
            )
    return "\n".join(lines)


def render_gaps_block(bundle: EvidenceBundle) -> str:
    """State the deterministic gaps up front so the model cannot paper over them."""
    header = (
        "Evidence gaps already established by the pipeline (facts about this run; "
        "record them in \"evidence_gaps\" and respect them in the affected dimensions):"
    )
    if not bundle.deterministic_gaps:
        return header + "\n- none: all upstream evidence was available."
    lines = [header]
    for gap in bundle.deterministic_gaps:
        affected = (
            f" | affects: {', '.join(gap.affected_dimensions)}"
            if gap.affected_dimensions
            else ""
        )
        lines.append(f"- ({gap.category}) {gap.gap} | impact: {gap.impact}{affected}")
    return "\n".join(lines)


def render_truncations_block(bundle: EvidenceBundle) -> str:
    """Report what was left out of the context, so omissions stay visible."""
    if not bundle.truncations:
        return ""
    lines = ["Context limits applied while assembling this evidence:"]
    lines.extend(f"- {note}" for note in bundle.truncations)
    return "\n".join(lines)


def render_source_index_block(bundle: EvidenceBundle) -> str:
    """List every citable id; references outside this list are invalid."""
    lines = [
        "Citable source ids (cite these exact ids; anything else is fabricated):"
    ]
    for source_id, entry in bundle.source_index.items():
        pages = entry.get("pages") or []
        page_label = _pages_label(pages) if pages else "no page"
        detail = _clean(entry.get("detail"))
        lines.append(
            f"- {source_id} | source_type={entry.get('source_type')} | {page_label}"
            + (f" | {detail}" if detail else "")
        )
    return "\n".join(lines)


def build_dimension_focus_block() -> str:
    """Render the per-dimension instruction blocks in contract order."""
    lines = ["What to examine for each dimension:"]
    for dimension, focus in DIMENSION_FOCUS.items():
        lines.append("")
        lines.append(f"### {dimension}")
        lines.append(focus)
    return "\n".join(lines)


def build_context_block(bundle: EvidenceBundle) -> str:
    """Assemble the complete evidence context handed to the model."""
    parts = [
        render_paper_block(bundle),
        render_section_block(bundle),
        render_inventory_block(bundle),
        render_references_block(bundle),
        render_text_block(bundle),
        render_visual_block(bundle),
        render_gaps_block(bundle),
        render_source_index_block(bundle),
    ]
    truncations = render_truncations_block(bundle)
    if truncations:
        parts.insert(6, truncations)
    return "\n\n".join(part for part in parts if part)


def build_analysis_prompt(bundle: EvidenceBundle) -> str:
    """
    Build the single analysis prompt for one paper.

    One prompt, one response: there is no repair round trip in this stage. A
    malformed or incomplete response is rejected and reported as a structured
    failure instead of being silently patched, because a second call would hide
    the fact that the first answer was unusable.
    """
    sections = [
        RULE_BLOCK,
        "",
        "=" * 72,
        "EVIDENCE BUNDLE",
        "=" * 72,
        build_context_block(bundle),
        "",
        "=" * 72,
        "ANALYSIS TASK",
        "=" * 72,
        build_dimension_focus_block(),
        "",
        "=" * 72,
        "OUTPUT CONTRACT",
        "=" * 72,
        OUTPUT_CONTRACT,
    ]
    return "\n".join(sections)
