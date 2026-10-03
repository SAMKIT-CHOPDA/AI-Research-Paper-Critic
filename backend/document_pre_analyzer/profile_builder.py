"""
Profile builder assembling the complete JSON-serializable Document Profile.

Aggregates:
- metadata, text & word statistics
- sections
- figures (confirmed + candidates + rejection audit trail)
- tables (confirmed + candidates + rejection audit trail)
- equations (confirmed + candidates + rejection audit trail)
- references
- content characteristics
- page-by-page breakdown
"""

import os
from typing import Dict, Any, List


def _object_block(payload: Dict[str, Any], keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Build one profile sub-object exposing confirmed items plus the full candidate
    and rejection audit trail, so every detector decision is inspectable.
    """
    confirmed_key = keys["confirmed"]
    candidates_key = keys["candidates"]
    return {
        "count": payload.get("count", 0),
        "items": payload.get("items", []),
        confirmed_key: payload.get(confirmed_key, payload.get("items", [])),
        candidates_key: payload.get(candidates_key, []),
        "candidate_count": payload.get("candidate_count", 0),
        "rejected_candidates": payload.get("rejected_candidates", []),
        "rejected_candidate_count": payload.get("rejected_candidate_count", 0),
        keys["mentions"]: payload.get(keys["mentions"], []),
        keys["mention_count"]: payload.get(keys["mention_count"], 0),
    }


def build_document_profile(
    parsed_pdf_info: Dict[str, Any],
    sections: List[Dict[str, Any]],
    figures_data: Dict[str, Any],
    tables_data: Dict[str, Any],
    equations_data: Dict[str, Any],
    references_data: Dict[str, Any],
    ocr_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct the final standardized Document Profile according to requirements.
    All fields are JSON serializable.
    """
    pdf_path = parsed_pdf_info.get("pdf_path", "")
    filename = os.path.basename(pdf_path) if pdf_path else "unknown.pdf"
    page_count = parsed_pdf_info.get("page_count", 0)
    text_length = parsed_pdf_info.get("text_length", 0)
    word_count = parsed_pdf_info.get("word_count", 0)

    # Content characteristics derivation
    section_normalized_titles = [s.get("normalized_title", "") for s in sections]

    has_methodology = any(
        norm in [
            "methodology", "methods", "approach", "proposed method",
            "proposed approach", "model architecture", "architecture",
        ]
        for norm in section_normalized_titles
    )

    has_experimental_results = any(
        norm in [
            "experiments", "experimental setup", "experimental results",
            "results", "results and discussion", "evaluation", "ablation study",
        ]
        for norm in section_normalized_titles
    )

    has_visual = figures_data.get("count", 0) > 0 or parsed_pdf_info.get("total_images", 0) > 0
    has_tbls = tables_data.get("count", 0) > 0
    has_math = equations_data.get("count", 0) > 0
    has_refs = references_data.get("has_references", False)

    content_characteristics = {
        "has_methodology": has_methodology,
        "has_experimental_results": has_experimental_results,
        "has_visual_content": has_visual,
        "has_tables": has_tbls,
        "has_mathematical_content": has_math,
        "has_references": has_refs,
        "document_nature": ocr_assessment.get("nature", "born-digital"),
    }

    # Per-page summaries preserving block/drawing counts and item presences
    pages_summary = []
    fig_pages = {f["page"] for f in figures_data.get("items", [])}
    tbl_pages = {t["page"] for t in tables_data.get("items", [])}
    eq_pages = {e["page"] for e in equations_data.get("items", [])}
    fig_candidate_pages = {f["page"] for f in figures_data.get("figure_candidates", [])}
    tbl_candidate_pages = {t["page"] for t in tables_data.get("table_candidates", [])}

    for p in parsed_pdf_info.get("pages", []):
        pnum = p["page_number"]
        pages_summary.append({
            "page_number": pnum,
            "width": p.get("width"),
            "height": p.get("height"),
            "text_length": p.get("text_length", 0),
            "word_count": p.get("word_count", 0),
            "image_count": p.get("image_count", 0),
            "drawing_count": p.get("drawing_count", 0),
            "has_figures": pnum in fig_pages,
            "has_tables": pnum in tbl_pages,
            "has_equations": pnum in eq_pages,
            "has_figure_candidates": pnum in fig_candidate_pages,
            "has_table_candidates": pnum in tbl_candidate_pages,
        })

    profile = {
        "filename": filename,
        "page_count": page_count,
        "text_length": text_length,
        "word_count": word_count,
        "metadata": parsed_pdf_info.get("metadata", {}),
        "sections": sections,
        "figures": _object_block(figures_data, {
            "confirmed": "confirmed_figures",
            "candidates": "figure_candidates",
            "mentions": "rejected_figure_mentions",
            "mention_count": "rejected_figure_mention_count",
        }),
        "tables": _object_block(tables_data, {
            "confirmed": "confirmed_tables",
            "candidates": "table_candidates",
            "mentions": "rejected_table_mentions",
            "mention_count": "rejected_table_mention_count",
        }),
        "equations": _object_block(equations_data, {
            "confirmed": "confirmed_equations",
            "candidates": "equation_candidates",
            "mentions": "rejected_equation_mentions",
            "mention_count": "rejected_equation_mention_count",
        }),
        "references": {
            "has_references": references_data.get("has_references", False),
            "count": references_data.get("count", 0),
            "start_page": references_data.get("start_page"),
            "confidence": references_data.get("confidence", 0.0),
            "is_sequential": references_data.get("is_sequential", False),
        },
        "content_characteristics": content_characteristics,
        "pages": pages_summary,
    }

    return profile
