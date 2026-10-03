"""
Section detector identifying research paper headings and hierarchy.

Combines:
- PyMuPDF font information (bold flags, relative size > dominant font)
- Section numbering regexes (e.g., '1', '1.1', 'I.', 'A.')
- Numbering hierarchy depth (subsections such as '3.1' / '3.2.1')
- Standard research section dictionary matching
- Vertical whitespace isolation relative to neighbouring paragraphs
- Layout analysis / Docling section_header evidence
- Confidence calculation with a transparent, non-inflated evidence budget
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .confidence import calculate_section_confidence
from .config import PreAnalyzerConfig

STANDARD_SECTIONS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "model architecture", "methodology", "methods",
    "approach", "proposed approach", "proposed method", "architecture",
    "system architecture", "experiments", "experimental setup",
    "experimental results", "results", "results and discussion",
    "discussion", "evaluation", "ablation study", "model variations",
    "conclusion", "conclusions", "future work", "limitations",
    "acknowledgements", "acknowledgments", "references", "bibliography",
    "appendix",
]

NUMBERING_PATTERN = re.compile(
    r"^(?:(\d+(?:\.\d+)*)|([IVXLCDM]+))[\s\.\-]+", re.IGNORECASE
)

# Blocks made only of digits/punctuation are table data rows or year lines,
# never headings (e.g. "1 512 512 5.29 24.9 ..." inside Table 3).
DATA_ONLY_PATTERN = re.compile(r"^[\d\s\.\-–—%/,()]+$")


def _flatten_block_text(block: Dict[str, Any]) -> str:
    """Flatten a PyMuPDF text block into a single normalized string."""
    return " ".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    ).strip()


def _text_blocks(blocks: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """(original_index, block) pairs for every block that actually holds text."""
    collected = []
    for block_index, block in enumerate(blocks or []):
        if "lines" not in block:
            continue
        if not _flatten_block_text(block):
            continue
        collected.append((block_index, block))
    return collected


def _vertical_spacing(
    text_blocks: List[Tuple[int, Dict[str, Any]]],
    position: int,
    font_size: float,
    line_multiple: float,
) -> Dict[str, Any]:
    """
    Measure the whitespace around a candidate heading.

    A heading is normally separated from the surrounding paragraphs by more than
    one line height, which is what distinguishes it from an in-paragraph sentence
    that merely happens to start with a number.
    """
    _, block = text_blocks[position]
    top = float(block["bbox"][1])
    bottom = float(block["bbox"][3])

    gap_above: Optional[float] = None
    gap_below: Optional[float] = None
    if position > 0:
        gap_above = round(top - float(text_blocks[position - 1][1]["bbox"][3]), 1)
    if position + 1 < len(text_blocks):
        gap_below = round(float(text_blocks[position + 1][1]["bbox"][1]) - bottom, 1)

    line_height = max(font_size, 1.0) * line_multiple
    has_spacing = (
        (gap_above is not None and gap_above >= line_height)
        or (gap_below is not None and gap_below >= line_height)
    )

    return {
        "gap_above_pts": gap_above,
        "gap_below_pts": gap_below,
        "line_height_pts": round(line_height, 1),
        "has_vertical_spacing": has_spacing,
    }


def _is_subsection_numbered(number_str: str) -> bool:
    """True for hierarchical numbering such as '3.1' or '3.2.1'."""
    return bool(number_str) and "." in number_str


def _normalize_title(title: str) -> str:
    cleaned = NUMBERING_PATTERN.sub("", title.strip()).strip()
    return cleaned.lower()


def _determine_heading_level(number_str: str, default_level: int = 1) -> int:
    if not number_str:
        return default_level
    if "." in number_str:
        return number_str.count(".") + 1
    return 1


def detect_sections(
    parsed_pdf_info: Dict[str, Any],
    layout_data: Dict[str, Any] = None,
    config: Optional[PreAnalyzerConfig] = None,
) -> List[Dict[str, Any]]:
    """Detect structured sections across all pages using font hierarchy and layout signals."""
    cfg = config or PreAnalyzerConfig()
    dominant_font_size = parsed_pdf_info.get("dominant_font_size", 10.0)
    docling_headers = (layout_data or {}).get("headers", [])

    detected_sections: List[Dict[str, Any]] = []
    seen_titles = set()

    for page in parsed_pdf_info["pages"]:
        page_num = page["page_number"]
        page_height = float(page.get("height") or 0.0) or 792.0
        text_blocks = _text_blocks(page.get("blocks", []))

        for position, (_, block) in enumerate(text_blocks):
            lines = block["lines"]
            if not lines or len(lines) > 4:
                continue

            block_lines_text = []
            max_size = 0.0
            is_bold = False

            for line in lines:
                spans = line.get("spans", [])
                line_str = "".join(s.get("text", "") for s in spans).strip()
                if line_str:
                    block_lines_text.append(line_str)
                for s in spans:
                    sz = s.get("size", 0.0)
                    if sz > max_size:
                        max_size = sz
                    flags = s.get("flags", 0)
                    font_name = s.get("font", "").lower()
                    if (flags & 16) or "bold" in font_name or "medi" in font_name:
                        is_bold = True

            candidate_text = " ".join(block_lines_text).strip()
            if not candidate_text or len(candidate_text) > 120:
                continue
            # Table data rows and numeric-only lines are never headings.
            if DATA_ONLY_PATTERN.match(candidate_text):
                continue

            num_match = NUMBERING_PATTERN.match(candidate_text)
            has_numbering = num_match is not None
            number_str = num_match.group(1) or num_match.group(2) if num_match else ""
            is_subsection = _is_subsection_numbered(number_str)

            norm_title = _normalize_title(candidate_text)
            matches_standard = any(
                std == norm_title or norm_title.startswith(std + " ") or norm_title.endswith(" " + std)
                for std in STANDARD_SECTIONS
            )

            has_distinct_font_size = max_size >= dominant_font_size * 1.05

            has_docling = any(
                h.get("page") == page_num and _normalize_title(h.get("text", "")) == norm_title
                for h in docling_headers
            )

            spacing = _vertical_spacing(
                text_blocks, position, max_size, cfg.heading_spacing_line_multiple
            )


            is_valid_header = False
            if matches_standard and (is_bold or has_distinct_font_size or has_docling):
                is_valid_header = True
            elif has_numbering and (is_bold or has_distinct_font_size) \
                    and len(candidate_text.split()) <= 15:
                is_valid_header = True
            elif has_docling and (is_bold or has_distinct_font_size) \
                    and spacing["has_vertical_spacing"]:
                is_valid_header = True
            elif is_subsection and (is_bold or has_docling) \
                    and spacing["has_vertical_spacing"] \
                    and len(candidate_text.split()) <= 15:
                is_valid_header = True

            # Exclude the paper title on page 1
            if page_num == 1 and max_size >= dominant_font_size * 1.35 \
                    and not has_numbering and not matches_standard:
                is_valid_header = False

            if not is_valid_header:
                continue

            key = (page_num, norm_title)
            if key in seen_titles:
                continue
            seen_titles.add(key)

            level = _determine_heading_level(number_str, default_level=1)
            conf = calculate_section_confidence(
                matches_standard_title=matches_standard,
                has_numbering=has_numbering,
                has_distinct_font_size=has_distinct_font_size,
                is_bold=is_bold,
                has_docling_header=has_docling,
                has_vertical_spacing=spacing["has_vertical_spacing"],
                is_subsection_numbered=is_subsection,
            )

            detected_sections.append({
                "title": candidate_text,
                "normalized_title": norm_title,
                "number": number_str,
                "page": page_num,
                "level": level,
                "font_size": round(max_size, 1),
                "is_bold": is_bold,
                "bbox": list(block.get("bbox", [])),
                "confidence": conf,
                "evidence": {
                    "matches_standard": matches_standard,
                    "has_numbering": has_numbering,
                    "is_subsection_numbered": is_subsection,
                    "has_distinct_font_size": has_distinct_font_size,
                    "docling_verified": has_docling,
                    "has_vertical_spacing": spacing["has_vertical_spacing"],
                    "gap_above_pts": spacing["gap_above_pts"],
                    "gap_below_pts": spacing["gap_below_pts"],
                    "line_height_pts": spacing["line_height_pts"],
                    "page_position_ratio": round(
                        float(block["bbox"][1]) / page_height, 3
                    ),
                },
            })

    return detected_sections

