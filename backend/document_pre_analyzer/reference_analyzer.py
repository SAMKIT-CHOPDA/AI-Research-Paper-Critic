"""
Reference analyzer identifying bibliography sections and individual citation entries.

Supports:
- IEEE bracketed style: [1] Author, "Title", ...
- Numbered period style: 1. Author, Title, ...
- Docling list_item verification
- Start page tracking, entry counting, and confidence scoring
"""

import re
from typing import Dict, Any, List
from .confidence import calculate_reference_confidence

REF_HEADER_REGEX = re.compile(
    r"^(references|bibliography|works\s+cited)$", re.IGNORECASE
)
BRACKET_ENTRY_REGEX = re.compile(r"^\[(\d{1,3})\]\s*(.*)", re.DOTALL)
NUMBER_PERIOD_ENTRY_REGEX = re.compile(r"^(\d{1,3})\.\s+([A-Z].*)", re.DOTALL)


def detect_references(
    parsed_pdf_info: Dict[str, Any],
    layout_data: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Detect reference section, parse complete entries, and compute confidence."""
    start_page: int = None
    ref_lines_raw: List[str] = []
    has_header = False

    # 1. Identify start page of References section
    for page in parsed_pdf_info["pages"]:
        p_num = page["page_number"]
        blocks = page.get("blocks", [])

        page_in_ref = start_page is not None

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = " ".join(
                "".join(s.get("text", "") for s in line.get("spans", []))
                for line in block["lines"]
            ).strip()

            if not page_in_ref:
                # Check for reference header
                first_line = block_text.split("\n")[0].strip()
                cleaned_title = re.sub(r"^(?:\d+[\.\s]*)", "", first_line).strip()
                if REF_HEADER_REGEX.match(cleaned_title):
                    has_header = True
                    start_page = p_num
                    page_in_ref = True
                    # Lines following the header in this block
                    rem = block_text[len(first_line):].strip()
                    if rem:
                        ref_lines_raw.append(rem)
            else:
                ref_lines_raw.append(block_text)

    # 2. Parse individual reference entries
    entries: List[Dict[str, Any]] = []

    # Try bracket style first e.g. [1] ... [2] ...
    full_ref_text = "\n".join(ref_lines_raw)
    bracket_splits = re.split(r"(?:^|\n)\s*\[(\d{1,3})\]", full_ref_text)

    if len(bracket_splits) > 3:
        # Bracketed style confirmed
        # format of bracket_splits: [preamble, "1", text1, "2", text2, ...]
        i = 1
        while i < len(bracket_splits) - 1:
            ref_idx = bracket_splits[i]
            ref_text = bracket_splits[i + 1].strip()
            # Clean up newlines within an entry
            clean_entry = re.sub(r"\s+", " ", ref_text)
            entries.append({
                "index": int(ref_idx) if ref_idx.isdigit() else len(entries) + 1,
                "text": clean_entry,
                "format": "bracketed",
            })
            i += 2

    # If bracket style didn't find entries, try Docling list_items on ref pages
    if not entries and start_page and layout_data:
        docling_items = layout_data.get("list_items", [])
        for item in docling_items:
            if item.get("page") and item["page"] >= start_page:
                entries.append({
                    "index": len(entries) + 1,
                    "text": item.get("text", ""),
                    "format": "docling_list_item",
                })

    # If still not found, try number dot format (1. Author...)
    if not entries and start_page:
        dot_splits = re.split(r"(?:^|\n)\s*(\d{1,3})\.\s+", full_ref_text)
        if len(dot_splits) > 3:
            i = 1
            while i < len(dot_splits) - 1:
                ref_idx = dot_splits[i]
                ref_text = dot_splits[i + 1].strip()
                clean_entry = re.sub(r"\s+", " ", ref_text)
                entries.append({
                    "index": int(ref_idx) if ref_idx.isdigit() else len(entries) + 1,
                    "text": clean_entry,
                    "format": "number_dot",
                })
                i += 2

    count = len(entries)
    has_refs = count > 0 or has_header

    # Check if entry indices are sequential
    is_sequential = False
    if count >= 3:
        indices = [e["index"] for e in entries if isinstance(e.get("index"), int)]
        is_sequential = indices == list(range(1, count + 1))

    conf = calculate_reference_confidence(
        has_section_header=has_header,
        entry_count=count,
        is_sequential=is_sequential,
        has_standard_citation_format=count > 0,
    )

    return {
        "has_references": has_refs,
        "count": count,
        "start_page": start_page,
        "is_sequential": is_sequential,
        "confidence": conf,
        "items": entries,
    }
