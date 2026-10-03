"""
Visual semantics: prompting and structured response parsing.

Prompting philosophy
--------------------
The model is never asked to "critique the entire paper". Each asset gets one
focused prompt describing exactly what to inspect, plus a bounded slice of
context (caption, page, section, neighbouring text). The prompt forces the model
to keep three registers apart:

* OBSERVATION   - what is visibly present in the pixels
* INTERPRETATION- what the visual plausibly represents (provisional wording)
* UNCERTAINTY   - what cannot be determined from the image alone

Confirmatory language ("proves", "demonstrates conclusively") is not accepted as
evidence: any such phrasing the model still returns is detected and surfaced as
an explicit uncertainty, because the causal/evaluative verdict belongs to the
later Analysis/Critique stage, not to the Vision Agent.

Table numeric authority
-----------------------
When the PDF exposes machine-readable cells, those values are the only
authoritative source for exact numbers. The vision model may interpret trends,
comparisons and visual emphasis, but it is explicitly told not to invent or
correct numeric values.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from backend.vision_agent.schemas import (
    CaptionConsistency,
    VisualAnalysisResult,
    VisualAsset,
)
from backend.vision_agent.vision_client import BaseVisionClient

logger = logging.getLogger(__name__)

# Phrases that turn an observation into an unsupported claim of proof.
OVERCLAIM_PHRASES = (
    "proves",
    "proven",
    "proof that",
    "demonstrates conclusively",
    "shows conclusively",
    "conclusively demonstrates",
    "establishes that",
    "guarantees",
    "definitively shows",
)

NOTE_NO_UNCERTAINTY = (
    "No explicit uncertainty was reported by the vision model; the interpretation "
    "is therefore provisional."
)

FIGURE_FOCUS_QUESTIONS = (
    "1. What type of visual is this (line plot, bar chart, heatmap, schematic, "
    "diagram, training curve, architecture figure, ...)?",
    "2. What does it appear to represent?",
    "3. What variables, axes, units, legends and labels are visible?",
    "4. What trends or relationships are visually observable?",
    "5. What comparison does the figure communicate?",
    "6. What result does the figure appear to support?",
    "7. What ambiguities or limitations are visible (missing legends, clipped "
    "ranges, unreadable labels, absent error bars, insufficient resolution)?",
    "8. Is the visual consistent with its caption and surrounding context?",
)

TABLE_FOCUS_QUESTIONS = (
    "1. What is the table comparing?",
    "2. What are the rows and columns?",
    "3. What metrics (and units) are presented?",
    "4. Which values appear strongest and weakest (by visual position/emphasis)?",
    "5. What comparison does the table communicate?",
    "6. What result does the table appear to support?",
    "7. Are units, headers or values ambiguous, missing or visually unclear?",
    "8. Does the table appear consistent with its caption and surrounding context?",
)

OUTPUT_CONTRACT = """Return ONLY a single JSON object (no markdown, no code fences) with exactly these keys:

{
  "observation": "string - only what is visibly present in the image",
  "interpretation": "string - provisional reading of what the visual represents; use hedging verbs such as 'appears to', 'suggests'",
  "key_elements": ["axes/labels/legend/headers/variables actually visible"],
  "reported_relationships": ["trends or relations the visual itself displays"],
  "supports_claims": ["claims this visual appears to support (never 'proves')"],
  "uncertainties": ["anything you cannot determine from the image alone"],
  "caption_consistency": {
    "status": "consistent | partially_consistent | inconsistent | unknown",
    "explanation": "why that status was assigned"
  }
}

Rules:
- Do not claim the figure or table "proves" anything; it can only appear to support a claim.
- Never invent values, labels, units or error bars that are not visible.
- Put every unresolved item in "uncertainties" instead of guessing.
- If a field cannot be determined, return an empty string or an empty list rather than fabricating content."""

RULE_BLOCK = """You are performing SEMANTIC VISUAL ANALYSIS of one visual element from a research paper.
Separate your answer strictly into:
- OBSERVATION: what is visibly present in the image (no inference).
- INTERPRETATION: what it plausibly represents (explicitly provisional).
- UNCERTAINTY: what cannot be determined from the image alone.

Use hedged language ("appears to", "suggests", "in the shown interval"). Never state or
imply that the visual proves, establishes or guarantees a claim: a single figure or table can
only appear to support a claim, and the evaluative judgement belongs to a later stage."""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def build_context_block(asset: VisualAsset) -> str:
    """Assemble the bounded textual context attached to one asset."""
    lines = [
        f"- asset type: {asset.asset_type}",
        f"- page: {asset.page_number}",
    ]
    if asset.label:
        lines.append(f"- printed number: {asset.asset_type} {asset.label}")
    if asset.section:
        lines.append(f"- section: {asset.section}")
    if asset.caption:
        lines.append(f"- caption: {_clean_text(asset.caption)}")
    if asset.context_text:
        lines.append(f"- surrounding text: {_clean_text(asset.context_text)}")
    if asset.asset_subtype:
        lines.append(f"- detected visual subtype: {asset.asset_subtype}")
    return "\n".join(lines)


class AnalysisError(Exception):
    """Raised when a vision response cannot be turned into a valid analysis."""
    pass


def _structured_table_lines(asset: VisualAsset) -> List[str]:
    """Render machine-readable table values as authoritative prompt context."""
    table = asset.structured_table
    if not table:
        return []
    source = table.get("source")
    lines = [f"- machine-readable table values (source: {source}):"]

    counts = []
    for key, label in (
        ("cell_row_count", "row count"),
        ("cell_column_count", "column count"),
        ("row_count", "reported rows"),
        ("column_count", "reported columns"),
    ):
        if isinstance(table.get(key), int):
            counts.append(f"{label}: {table[key]}")
    if counts:
        lines.append("  " + ", ".join(counts))
    if table.get("structure_source"):
        lines.append(f"  structure source: {table['structure_source']}")

    headers = table.get("headers") or []
    if isinstance(headers, (list, tuple)) and headers:
        lines.append("  headers: " + " | ".join(_clean_text(h) for h in headers))

    rows = table.get("rows")
    if isinstance(rows, (list, tuple)):
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            cells = " | ".join(_clean_text(c) for c in row)
            if cells.strip(" |"):
                lines.append("  row: " + cells)

    if table.get("authoritative_for_exact_values"):
        lines.append(
            "  These extracted cell values are authoritative for exact numbers. "
            "Do not invent, round or correct any number that is not present above."
        )
    else:
        lines.append(
            "  Only structural metadata is available (no machine-read cell values). "
            "Do not assert exact numbers from the image; report them as approximate "
            "or as uncertain."
        )
    return lines


def build_figure_prompt(asset: VisualAsset) -> str:
    """Focused prompt for a figure/plot/diagram asset."""
    sections = [
        RULE_BLOCK,
        "",
        "Visual element context:",
        build_context_block(asset),
        "",
        "Answer these questions about the figure (do not answer anything else, and do "
        "not critique the paper):",
        "\n".join(FIGURE_FOCUS_QUESTIONS),
        "",
        OUTPUT_CONTRACT,
    ]
    return "\n".join(sections)


def build_table_prompt(asset: VisualAsset) -> str:
    """Focused prompt for a table asset, including numeric-authority guidance."""
    sections = [
        RULE_BLOCK,
        "",
        "Visual element context:",
        build_context_block(asset),
    ]
    table_lines = _structured_table_lines(asset)
    if table_lines:
        sections.append("")
        sections.extend(table_lines)
    sections.extend([
        "",
        "Answer these questions about the table (do not answer anything else, and do "
        "not critique the paper):",
        "\n".join(TABLE_FOCUS_QUESTIONS),
        "",
        "Exact numeric values are NOT your responsibility: the PDF text layer above is "
        "the authoritative source for numbers, and your role is to read the table's "
        "trends, comparisons and visual emphasis.",
        "",
        OUTPUT_CONTRACT,
    ])
    return "\n".join(sections)


def build_prompt(asset: VisualAsset) -> str:
    """Select the focused prompt for an asset by type."""
    if asset.asset_type == "table":
        return build_table_prompt(asset)
    return build_figure_prompt(asset)


_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Extract the JSON object from a model response.

    Accepts fenced blocks and preamble text, but raises AnalysisError when no
    parseable object exists, so a malformed response becomes a recorded failure
    instead of a silently empty analysis.
    """
    if not raw_text or not str(raw_text).strip():
        raise AnalysisError("Vision model returned an empty response")

    text = str(raw_text).strip()
    candidates: List[str] = []

    fenced = _FENCE_PATTERN.findall(text)
    candidates.extend(block.strip() for block in fenced if block.strip())
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = candidate[start: end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise AnalysisError(
        "Vision model response did not contain a parseable JSON object"
    )


_CAPTION_STATUSES = {"consistent", "partially_consistent", "inconsistent", "unknown"}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _clean_text(json.dumps(value))
    return _clean_text(value)


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            cleaned = _coerce_text(item)
            if cleaned:
                result.append(cleaned)
        return result
    cleaned = _coerce_text(value)
    return [cleaned] if cleaned else []


def flag_overclaiming(*texts: Any) -> List[str]:
    """
    Detect confirmatory phrasing that an image alone cannot support.

    Returns the phrases found (deduplicated, sorted). The caller records them as
    explicit uncertainties so an unsupported claim can never travel downstream as
    a visual fact.
    """
    flagged = set()
    for text in texts:
        lowered = _coerce_text(text).lower()
        if not lowered:
            continue
        for phrase in OVERCLAIM_PHRASES:
            if phrase in lowered:
                flagged.add(phrase)
    return sorted(flagged)


def parse_caption_consistency(value: Any) -> CaptionConsistency:
    """Normalize the model's caption-consistency field, defaulting to unknown."""
    explanation = ""
    raw_status: Any = value
    if isinstance(value, dict):
        raw_status = value.get("status")
        explanation = _coerce_text(value.get("explanation"))
    elif isinstance(value, str):
        raw_status = value
    else:
        raw_status = None

    normalized = _coerce_text(raw_status).lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    if normalized not in _CAPTION_STATUSES:
        if raw_status is None:
            return CaptionConsistency(
                status="unknown",
                explanation=explanation
                or "The vision model did not report caption consistency.",
            )
        return CaptionConsistency(
            status="unknown",
            explanation=explanation
            or f"Unrecognized caption-consistency status '{_coerce_text(raw_status)}'.",
        )
    return CaptionConsistency(status=normalized, explanation=explanation)


def parse_visual_analysis(
    raw_text: str,
    asset: VisualAsset,
    model: Optional[str] = None,
) -> VisualAnalysisResult:
    """
    Turn a raw model response into a validated VisualAnalysisResult.

    Field-level leniency is deliberate (a missing key becomes an explicit gap
    rather than an error), but a response with no observation and no
    interpretation at all is treated as a failure.
    """
    payload = extract_json_payload(raw_text)

    observation = _coerce_text(payload.get("observation"))
    interpretation = _coerce_text(payload.get("interpretation"))
    if not observation and not interpretation:
        raise AnalysisError(
            "Vision model response contained neither an observation nor an interpretation"
        )

    key_elements = _coerce_string_list(payload.get("key_elements"))
    reported_relationships = _coerce_string_list(payload.get("reported_relationships"))
    supports_claims = _coerce_string_list(payload.get("supports_claims"))
    uncertainties = _coerce_string_list(payload.get("uncertainties"))

    for phrase in flag_overclaiming(observation, interpretation, supports_claims):
        uncertainties.append(
            f"Confirmatory language detected ('{phrase}'): an image alone cannot "
            "establish this, so the statement is treated as an interpretation."
        )
    if not uncertainties:
        uncertainties = [NOTE_NO_UNCERTAINTY]

    structured_table = asset.structured_table or {}
    numeric_authority = (
        "document_profile_machine_readable"
        if asset.asset_type == "table"
        and structured_table.get("authoritative_for_exact_values")
        else "vision_interpretation_only"
    )

    return VisualAnalysisResult(
        asset_id=asset.asset_id,
        asset_type=asset.asset_type,
        page_number=asset.page_number,
        section=asset.section,
        caption=asset.caption,
        observation=observation,
        interpretation=interpretation,
        key_elements=key_elements,
        reported_relationships=reported_relationships,
        supports_claims=supports_claims,
        uncertainties=uncertainties,
        caption_consistency=parse_caption_consistency(payload.get("caption_consistency")),
        model=model,
        image_path=asset.image_path,
        source_confidence=asset.source_confidence,
        extraction_method=asset.extraction_method,
        numeric_authority=numeric_authority,
        structured_evidence=asset.structured_table if asset.asset_type == "table" else None,
    )


def analyze_asset(
    asset: VisualAsset,
    client: BaseVisionClient,
    model: str,
    prompt: Optional[str] = None,
) -> VisualAnalysisResult:
    """
    Analyze one asset end to end: focused prompt -> model call -> parsed result.

    Client failures (auth, unavailable model, transport) propagate as
    VisionClientError; parsing failures raise AnalysisError. The agent isolates
    both per asset.
    """
    if not asset.image_path:
        raise AnalysisError(f"No extracted image path for asset {asset.asset_id}")

    resolved_prompt = prompt or build_prompt(asset)
    raw_response = client.analyze_image(asset.image_path, resolved_prompt, model)
    return parse_visual_analysis(raw_response, asset, model)