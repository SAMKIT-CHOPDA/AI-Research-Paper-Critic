"""
Prompt construction for the Critique Engine (Phase 6).

Prompting philosophy
--------------------
The model is asked for one bounded, structured, human-readable synthesis of an
already-completed structured analysis. It receives a rendered summary of the
frozen Document Profile, the JEV routing decision, concise RAG/Vision provenance
and the Phase 5 findings, and it must return one JSON object.

Four boundaries are enforced by the prompt and re-checked by the validator:

* SYNTHESIS - this stage organizes and communicates findings. It does not
  rediscover figures, rerun retrieval, re-analyze the paper or make any routing
  decision, and it introduces no substantive finding that is not already present
  in the supplied structured sources.
* PROVENANCE - every ``source_id`` must be copied from the "Allowed citation ids"
  list. Invented identifiers are detected and rejected by the validator.
* NO SCORE - no overall paper score, quality score, rank, tier or unsupported
  overall label ("excellent"/"good"/"bad") may appear anywhere.
* REGISTER - "the paper reports...", "the authors state...", "the available
  evidence indicates..." and "this could not be established..." are kept apart,
  and author-stated limitations are never merged with analyst-identified ones.

The prompt is built from the ``SynthesisContext`` only: nothing is re-derived from
the PDF here, and the model never sees the whole paper.
"""

from typing import List

from backend.critique_engine.schemas import (
    ALWAYS_REQUIRED_SECTIONS,
    SECTION_IDS,
    SECTION_TITLES,
    SynthesisContext,
)

RULE_BLOCK = """You are the final synthesis stage of an automated research-paper critic.
You receive the structured output of an earlier analysis stage, which already
evaluated this paper against the available evidence, and you turn it into a
coherent, readable research critique.

You are NOT re-analyzing the paper. You must not: rediscover figures or tables,
run retrieval, read the raw document, invent evidence, invent citations, invent
numbers, re-decide any routing level, or introduce a substantive finding that is
not already present in the supplied analysis findings. Your job is to organise,
connect and communicate what the analysis established - and to say plainly what
it could not establish.

Ground rules:
1. Use only the content in this prompt. Nothing else exists for you.
2. Never invent evidence. Every "source_id" you cite must appear verbatim in the
   "Allowed citation ids" list below. Identifiers such as chunk_999, figure_999
   or table_999 that do not appear there are fabrications and will be rejected.
3. Never invent page numbers. Cite a page only as "p. N" when that page came from
   the pages given for a cited source in this prompt.
4. Never invent numbers. Repeat an exact experimental number only when the
   supplied analysis explicitly reports it. If a value is not reported, say that
   it could not be established from the available evidence.
5. Keep the registers apart. Use "the paper reports..." for author statements,
   "the authors state..." for explicit author claims, "the analysis indicates..."
   for structured analytical reading, and "this could not be established from the
   available evidence..." for gaps. Never present your own reading as an author
   claim.
6. Never produce a score. No paper score, quality score, rank, tier, grade or
   overall verdict label (such as "excellent", "good", "bad", "weak", "strong
   paper"). Also no "7/10", "4 out of 5" or "82/100". A single overall number for
   a whole paper is forbidden everywhere in your output.
7. Never fabricate confidence or statistics. Do not state significance,
   variance or effect sizes that the supplied analysis does not report. Phrase
   uncertainty honestly instead ("the supplied evidence does not establish...").
8. Limitations the authors state and limitations the analysis identified must
   stay in separate sections and separate lists. Never file your own observation
   as something the authors acknowledged, and never move an author-stated
   limitation into the analyst list.
9. You are a synthesis layer, not an analyst. You may reorganise, combine,
   connect, explain more clearly and summarise the supplied findings. You may
   rewrite a supplied finding in clearer words, but you MUST NOT introduce any
   strength, weakness, limitation, claim, result or conclusion that is not
   listed under "SUPPLIED FINDINGS" below. Do not convert general knowledge or
   your own interpretation into a new research finding, however plausible or
   reasonable it sounds. If something is not in the supplied findings, it is not
   in your critique.
10. Only include strengths that are directly supported by the supplied Analysis
    result. Do not infer or introduce additional strengths.
11. Only include limitations that are directly present in the author-stated or
    analyst-identified lists of the supplied Analysis result. Do not introduce
    new limitations - for example, do not add a statistical, reproducibility or
    generalisation concern that the analysis did not record.
12. Produce EVERY section marked as "supported - you MUST produce this section" in
    the report-section guidance. A supported section that you omit is a failure.
    Only "executive_summary" and "overall_assessment" are required even when they
    are not otherwise supported.
13. Omit any section listed as "NOT supported" in the report-section guidance
    rather than inventing content to fill it. Do not force unsupported content
    into a section.
14. Present open research questions as questions, not as factual claims.
15. Use cautious wording for internal consistency ("the supplied evidence suggests
    a possible discrepancy between..."), not categorical verdicts.
16. Write in a professional, academic, concise and neutral register. No
    sensational language, no filler, no repetition.
17. Return ONLY one JSON object matching the contract exactly - no markdown, no
    code fences, no commentary before or after the object.
"""

OUTPUT_CONTRACT_HEADER = """Return one JSON object with exactly these top-level keys:

{
  "sections": [                       // the report sections, in any order
    {
      "section_id": "<one of the allowed section ids>",
      "title": "<human readable title>",
      "content": "<the readable synthesis text for this section>",
      "evidence_refs": [
        {"source_type": "document|rag|vision|analysis",
         "source_id": "<id copied from the allowed citation ids>"}
      ]
    }
  ],
  "claim_evidence_summary": [
    {
      "claim": "<the claim>",
      "status": "supported|partially_supported|unsupported|unclear|insufficient_evidence",
      "assessment": "<how the available evidence reads it>",
      "evidence_refs": [ ... same reference objects ... ],
      "uncertainty_notes": ["..."]
    }
  ],
  "strengths": [
    {"statement": "...", "dimension": "...", "evidence_refs": [ ... ], "rationale": "..."}
  ],
  "limitations": {
    "author_stated": [
      {"statement": "...", "evidence_refs": [ ... ], "rationale": "..."}
    ],
    "analyst_identified": [
      {"statement": "...", "evidence_refs": [ ... ], "rationale": "..."}
    ]
  },
  "evidence_gaps": [
    {"gap": "...", "category": "...", "impact": "..."}
  ],
  "open_questions": [
    {"question": "...", "why_it_matters": "...", "evidence_refs": [ ... ]}
  ],
  "overall_assessment": {
    "content": "<evidence-grounded synthesis; no score, no label>",
    "evidence_refs": [ ... ],
    "uncertainties": ["..."]
  }
}

Rules on the object above:
- "claim_evidence_summary" must preserve every claim from the supplied analysis
  matrix, with its classification unchanged. Never create a new classification.
- "strengths" must contain only the strengths listed under "SUPPLIED FINDINGS".
  Never add one, however plausible it seems.
- "limitations.author_stated" may only contain limitations the supplied analysis
  lists as author-stated; "limitations.analyst_identified" only those it lists as
  analyst-identified. Never move an item between the two lists and never add one
  that is not in either supplied list.
- Never add keys such as "score", "rating", "rank", "tier", "grade" or
  "overall_verdict": the critique has no numeric or categorical paper verdict.
- "evidence_gaps" must include the deterministic gaps listed in this prompt, even
  when you did not write a separate section about them.
"""

SECTION_EXPECTATIONS = [
    "Section expectations:",
    "- executive_summary: what the paper studies, its stated contribution, the "
    "primary methodology, the main reported findings and the most important "
    "analytical observations. Concise; no claim absent from the analysis.",
    "- research_problem: the problem, its motivation and scope, keeping author "
    "statement and analytical reading apart.",
    "- contribution: the authors' stated contribution as 'the paper presents X as "
    "its primary contribution', plus whether the supplied evidence supports, "
    "partially supports or does not establish it. Never declare novelty on the "
    "engine's own authority.",
    "- methodology / data_and_experimental_design / baselines_and_metrics: "
    "synthesize what the analysis found. Where the analysis established nothing, "
    "say it could not be established from the available evidence; never infer "
    "missing experimental detail.",
    "- results_and_evidence: describe each important result with its supporting "
    "evidence, the claim it connects to and its uncertainty. Never fabricate "
    "exact numbers.",
    "- figures_and_tables: use the vision observations, keeping each asset's "
    "uncertainties. Do not reproduce large tables or invent values.",
    "- claim_evidence_assessment: reproduce the supplied matrix classifications "
    "unchanged.",
    "- strengths: the supplied strengths, as prose with provenance, never as a "
    "number.",
    "- author_stated_limitations / analyst_identified_limitations: the two "
    "categories, kept strictly separate.",
    "- reproducibility: data availability, preprocessing, architecture, training, "
    "hyperparameters, evaluation procedure and baseline implementation, but only "
    "where the analysis identified them.",
    "- internal_consistency: cautious language about possible discrepancies "
    "between text, figures, tables, methodology and results.",
    "- evidence_gaps / open_questions: preserve the supplied gaps and questions.",
    "- overall_assessment: a concise synthesis of major contribution, strongest "
    "evidence, principal methodological concern, major evidence gap and most "
    "important open question. No score, no rank, no tier, no overall label.",
]


def _render_document(context: SynthesisContext) -> str:
    document = context.document
    lines = ["DOCUMENT", "--------"]
    lines.append(f"- title: {document.title or 'not established from the profile'}")
    if document.filename:
        lines.append(f"- filename: {document.filename}")
    lines.append(f"- page_count: {document.page_count}")
    lines.append(f"- word_count: {document.word_count}")
    lines.append(f"- section_count: {document.section_count}")
    lines.append(f"- figure_count: {document.figure_count}")
    lines.append(f"- table_count: {document.table_count}")
    lines.append(f"- equation_count: {document.equation_count}")
    lines.append(f"- reference_count: {document.reference_count}")
    return "\n".join(lines)


def _render_routing(context: SynthesisContext) -> str:
    routing = context.routing
    lines = ["ROUTING (decided upstream; you make no routing decision)", "-------"]
    lines.append(
        "- analysis_level: "
        f"{routing.analysis_level or 'not reported'} (your synthesis reuses this level)"
    )
    lines.append(f"- rag_enabled: {routing.rag_enabled}")
    lines.append(f"- rag_level: {routing.rag_level or 'n/a'}")
    lines.append(f"- vision_enabled: {routing.vision_enabled}")
    lines.append(f"- vision_level: {routing.vision_level or 'n/a'}")
    if routing.confidence:
        rendered = ", ".join(
            f"{key}={value}" for key, value in sorted(routing.confidence.items())
        )
        lines.append(f"- router_confidence (upstream, informational only): {rendered}")
    return "\n".join(lines)


def _render_retrieved(context: SynthesisContext) -> str:
    lines = ["RAG (retrieved text evidence, for provenance only)", "---"]
    if not context.retrieved_evidence:
        lines.append("- no retrieved passage is available for this paper")
        return "\n".join(lines)
    for chunk in context.retrieved_evidence:
        pages = ", ".join(str(page) for page in chunk.pages) or "not reported"
        header = f"- {chunk.source_id} (pages: {pages}"
        if chunk.section:
            header += f"; section: {chunk.section}"
        header += ")"
        lines.append(header)
        if chunk.excerpt:
            lines.append(f"  excerpt: {chunk.excerpt}")
    return "\n".join(lines)


def _render_visual(context: SynthesisContext) -> str:
    lines = ["VISION (visual asset observations)", "---"]
    if not context.visual_evidence:
        lines.append("- no visual observation is available for this paper")
        return "\n".join(lines)
    for asset in context.visual_evidence:
        location = f"page {asset.page}" if asset.page else "page not reported"
        header = f"- {asset.source_id} (type: {asset.asset_type or 'unknown'}; {location}"
        if asset.section:
            header += f"; section: {asset.section}"
        header += ")"
        lines.append(header)
        if asset.observation:
            lines.append(f"  observation: {asset.observation}")
        if asset.interpretation:
            lines.append(f"  interpretation: {asset.interpretation}")
        for item in asset.uncertainties:
            lines.append(f"  uncertainty: {item}")
    return "\n".join(lines)


def _render_analysis(context: SynthesisContext) -> str:
    analysis = context.analysis
    lines = ["ANALYSIS (structured findings from the previous stage)", "--------"]
    if not analysis.dimensions and not analysis.claims:
        lines.append("- no structured finding is available for this paper")
        return "\n".join(lines)

    for dimension in analysis.dimensions:
        lines.append(f"- {dimension.dimension}:")
        if dimension.summary:
            lines.append(f"  summary: {dimension.summary}")
        for finding in dimension.findings:
            lines.append(f"  finding: {finding}")
        for item in dimension.uncertainties:
            lines.append(f"  uncertainty: {item}")

    if analysis.claims:
        lines.append("- claim_evidence_matrix:")
        for claim in analysis.claims:
            lines.append(f"  claim [{claim.status}]: {claim.claim}")
            if claim.assessment:
                lines.append(f"    assessment: {claim.assessment}")
            for item in claim.missing_evidence:
                lines.append(f"    missing_evidence: {item}")

    for label, values in (
        ("strengths", analysis.strengths),
        ("weaknesses", analysis.weaknesses),
        ("author_stated_limitations", analysis.author_stated_limitations),
        ("analyst_identified_limitations", analysis.analyst_identified_limitations),
    ):
        if values:
            lines.append(f"- {label}:")
            for value in values:
                lines.append(f"  - {value}")

    if analysis.open_questions:
        lines.append("- open_questions:")
        for question in analysis.open_questions:
            lines.append(f"  - {question}")
    if analysis.overall_assessment:
        lines.append(f"- analysis overall reading: {analysis.overall_assessment}")
    return "\n".join(lines)


def render_supplied_findings_block(context: SynthesisContext) -> str:
    """
    Present the analysis findings as an authoritative, enumerated allow-list.

    The critique may reword these findings, but the enumeration is what makes the
    boundary explicit: anything not listed here is not available to the synthesis
    stage, which is exactly what the validator enforces afterwards.
    """
    analysis = context.analysis
    lines = [
        "SUPPLIED FINDINGS (authoritative)",
        "-----------------------------",
        "Your output MUST remain grounded in these supplied findings. You may "
        "reword them into clearer language, but you may not add to them.",
        "",
    ]

    def listed(label, values):
        lines.append(f"{label}:")
        if values:
            for value in values:
                lines.append(f"- {value}")
        else:
            lines.append("- (none supplied)")
        lines.append("")

    listed("STRENGTHS SUPPLIED BY ANALYSIS", analysis.strengths)
    listed("WEAKNESSES SUPPLIED BY ANALYSIS", analysis.weaknesses)
    listed("AUTHOR-STATED LIMITATIONS SUPPLIED BY ANALYSIS",
           analysis.author_stated_limitations)
    listed("ANALYST-IDENTIFIED LIMITATIONS SUPPLIED BY ANALYSIS",
           analysis.analyst_identified_limitations)
    listed("OPEN QUESTIONS SUPPLIED BY ANALYSIS", analysis.open_questions)

    lines.append("CLAIMS SUPPLIED BY ANALYSIS (classification must not change):")
    if analysis.claims:
        for claim in analysis.claims:
            lines.append(f"- [{claim.status}] {claim.claim}")
    else:
        lines.append("- (none supplied)")
    lines.append("")
    return "\n".join(lines)


def render_gaps_block(context: SynthesisContext) -> str:
    """Render the gaps the engine knows about, both inherited and its own."""
    lines = ["EVIDENCE GAPS (must be preserved in your output)", "---"]
    if not context.analysis.evidence_gaps:
        lines.append("- none reported by the analysis stage")
    for gap in context.analysis.evidence_gaps:
        rendered = f"- ({gap.category}) {gap.gap}"
        if gap.impact:
            rendered += f" Impact: {gap.impact}"
        lines.append(rendered)

    if context.deterministic_gaps:
        lines.append("")
        lines.append("DETERMINISTIC GAPS (facts about this run; also preserve these)")
        for gap in context.deterministic_gaps:
            rendered = f"- ({gap.category}) {gap.gap}"
            if gap.impact:
                rendered += f" Impact: {gap.impact}"
            lines.append(rendered)
    return "\n".join(lines)


def render_source_index_block(context: SynthesisContext) -> str:
    """
    Render the citation allow-list.

    This list is the single source of truth for provenance: anything outside it is
    a fabrication and will be rejected by the validator.
    """
    lines = ["Allowed citation ids (copy these source_id values exactly)", "---"]
    if not context.source_index:
        lines.append("- none available")
        return "\n".join(lines)
    for entry in context.source_index.values():
        rendered = f"- {entry['source_type']}: {entry['source_id']}"
        if entry.get("pages"):
            rendered += f" (pages: {', '.join(str(p) for p in entry['pages'])})"
        if entry.get("page"):
            rendered += f" (page: {entry['page']})"
        if entry.get("label"):
            rendered += f" - {entry['label']}"
        lines.append(rendered)
    return "\n".join(lines)


def render_section_guidance(context: SynthesisContext) -> str:
    """State which sections are supported, and how each must be treated."""
    supported = set(context.supported_sections.section_ids)
    lines = ["REPORT SECTIONS", "---"]
    for section_id in SECTION_IDS:
        if section_id in supported:
            reason = context.supported_sections.reasons.get(section_id)
            detail = f": {reason}" if reason else ""
            suffix = f"  [SUPPORTED - you MUST produce this section{detail}]"
        elif section_id in ALWAYS_REQUIRED_SECTIONS:
            suffix = "  [SUPPORTED - you MUST produce this section: always required]"
        else:
            suffix = "  [NOT supported by the supplied evidence: omit this section]"
        lines.append(f"- {section_id}: {SECTION_TITLES[section_id]}{suffix}")
    lines.append("")
    lines.extend(SECTION_EXPECTATIONS)
    return "\n".join(lines)


def build_context_block(context: SynthesisContext) -> str:
    """Render the whole structured synthesis context deterministically."""
    blocks: List[str] = [
        _render_document(context),
        _render_routing(context),
        render_section_guidance(context),
        render_supplied_findings_block(context),
        _render_retrieved(context),
        _render_visual(context),
        _render_analysis(context),
        render_gaps_block(context),
        render_source_index_block(context),
    ]
    return "\n\n".join(blocks)


def build_critique_prompt(context: SynthesisContext) -> str:
    """
    Build the single synthesis prompt handed to the critique model.

    The prompt contains only what the engine assembled itself: the rules, the
    contract, and the deterministic rendering of the structured inputs.
    """
    parts = [
        RULE_BLOCK,
        OUTPUT_CONTRACT_HEADER,
        "STRUCTURED SYNTHESIS CONTEXT",
        "===========================",
        build_context_block(context),
    ]
    return "\n\n".join(parts)