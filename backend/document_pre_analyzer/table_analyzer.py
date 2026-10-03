"""
Table analyzer detecting tables, captions, structural grids, and dimensions.

Every table goes through an explicit two-stage decision, and both stages are
preserved in the returned payload so every decision can be audited:

1. Candidate discovery (text based, PyMuPDF only)
   A candidate exists only when a *text block starts* with a caption of the
   form ``Table <n>:`` / ``Table <n>.``.  Anchoring the pattern to the start of
   a block means prose such as
   "listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs"
   can never be promoted to a table.  Such in-text mentions are recorded
   separately under ``rejected_table_mentions``.

2. Structural confirmation (geometry based, cross-validated)
   A candidate is only confirmed when independent structural evidence exists in
   the region below the caption (fallback: directly above it, for papers that
   place captions underneath tables):
   - vector ruling lines spanning the caption width (PyMuPDF drawings)
   - a PyMuPDF ``page.find_tables()`` region with >= 2 rows and >= 2 columns
   - a Docling recognized ``table`` item

Returned keys: ``count`` / ``items`` / ``confirmed_tables`` (the same confirmed
list), ``table_candidates`` (all caption-anchored candidates, confirmed or not),
``rejected_candidates`` and ``rejected_table_mentions``.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple

import pymupdf

from .confidence import calculate_table_confidence
from .config import PreAnalyzerConfig

# A table caption must begin the text block (no mid-sentence matches).
TABLE_CAPTION_REGEX = re.compile(
    r"^Table\s+(\d+)\s*[:.\-—]\s*(.*)", re.IGNORECASE
)
# Any in-text mention of a table, used only to build the rejection audit trail.
TABLE_MENTION_REGEX = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)

MIN_RULING_LINE_LENGTH = 80.0      # points - shorter strokes are underlines/artifacts
MAX_RULING_LINE_THICKNESS = 2.5    # points
MIN_RULING_LINE_SPAN_RATIO = 0.40  # fraction of the caption width a rule must span
MIN_RULING_LINES_BELOW = 2         # rules required when searching below the caption
MIN_RULING_LINES_ABOVE = 3         # stricter fallback for captions placed below tables

AUTHOR_METADATA_TERMS = [
    "@google.com", "@gmail.com", "university", "department",
    "equal contribution", "equal contributions",
]


def _extract_block_text(block: Dict[str, Any]) -> str:
    """Flatten a PyMuPDF text block into a single normalized string."""
    return " ".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    ).strip()


def _is_author_metadata_block(text: str) -> bool:
    """Detect author affiliation headers masquerading as tabular content."""
    lowered = text.lower()
    return any(term in lowered for term in AUTHOR_METADATA_TERMS)


def _collect_ruling_lines(
    drawings: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Split vector drawings into long horizontal and vertical ruling lines."""
    horizontal: List[Dict[str, float]] = []
    vertical: List[Dict[str, float]] = []
    for drawing in drawings or []:
        rect = drawing.get("rect") or []
        if len(rect) != 4:
            continue
        x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
        width = x1 - x0
        height = y1 - y0
        if height <= MAX_RULING_LINE_THICKNESS and width >= MIN_RULING_LINE_LENGTH:
            horizontal.append({"y": (y0 + y1) / 2.0, "x0": x0, "x1": x1})
        elif width <= MAX_RULING_LINE_THICKNESS and height >= 20.0:
            vertical.append({"x": (x0 + x1) / 2.0, "y0": y0, "y1": y1})
    return horizontal, vertical


def _horizontal_overlap_ratio(
    range_a: Tuple[float, float], range_b: Tuple[float, float]
) -> float:
    """Intersection width of two x-ranges divided by the narrower range width."""
    low = max(range_a[0], range_b[0])
    high = min(range_a[1], range_b[1])
    narrower = min(range_a[1] - range_a[0], range_b[1] - range_b[0])
    if high <= low or narrower <= 0:
        return 0.0
    return (high - low) / narrower


def _union_bbox(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
    """Smallest box containing both inputs (either input may be empty)."""
    if not bbox_a:
        return list(bbox_b)
    if not bbox_b:
        return list(bbox_a)
    return [
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    ]

def _collect_native_table_regions(
    pdf_path: Optional[str],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Run PyMuPDF's own table finder once and index the results by page number.

    Returns ``{page_number: [{"bbox", "rows", "cols"}, ...]}``. Failures are
    swallowed so the analyzer degrades gracefully on unusual PDFs.
    """
    regions: Dict[int, List[Dict[str, Any]]] = {}
    if not pdf_path or not os.path.exists(pdf_path):
        return regions
    try:
        document = pymupdf.open(pdf_path)
    except Exception:
        return regions
    try:
        for page_index, page in enumerate(document):
            try:
                found = page.find_tables()
            except Exception:
                continue
            page_regions: List[Dict[str, Any]] = []
            for table in getattr(found, "tables", []) or []:
                page_regions.append({
                    "bbox": [float(v) for v in table.bbox],
                    "rows": int(getattr(table, "row_count", 0) or 0),
                    "cols": int(getattr(table, "col_count", 0) or 0),
                })
            if page_regions:
                regions[page_index + 1] = page_regions
    finally:
        document.close()
    return regions


def _docling_grid_size(docling_table: Dict[str, Any]) -> Tuple[int, int]:
    """Derive (rows, columns) from a Docling table's markdown rendering."""
    markdown = (docling_table.get("markdown") or "").strip()
    if not markdown:
        return 0, 0
    data_rows = [
        line for line in markdown.splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ")
    ]
    columns = 0
    for line in data_rows:
        columns = max(columns, line.count("|") - 1)
    return len(data_rows), max(columns, 0)


def _match_regions(
    regions: List[Dict[str, Any]],
    band_top: float,
    band_bottom: float,
    caption_x: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """Keep regions that start inside the band and overlap the caption width."""
    matched = []
    for region in regions or []:
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        top = float(bbox[1])
        if not (band_top - 2.0 <= top <= band_bottom):
            continue
        overlap = _horizontal_overlap_ratio(caption_x, (float(bbox[0]), float(bbox[2])))
        if overlap >= MIN_RULING_LINE_SPAN_RATIO:
            matched.append(region)
    return matched

def _find_structural_evidence(
    caption_bbox: List[float],
    page_height: float,
    band_pts: float,
    h_lines: List[Dict[str, float]],
    v_lines: List[Dict[str, float]],
    native_regions: List[Dict[str, Any]],
    docling_regions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Search for table structures below the caption, then above it as a fallback.

    The below-caption search needs >= 2 rules spanning the caption width; the
    above-caption fallback (caption underneath the table) is stricter and needs
    >= 3 rules, so a single paragraph underline can never confirm a table.
    """
    caption_top, caption_bottom = float(caption_bbox[1]), float(caption_bbox[3])
    caption_x = (float(caption_bbox[0]), float(caption_bbox[2]))

    below = (caption_top, min(page_height, caption_bottom + band_pts))
    above = (max(0.0, caption_top - band_pts), caption_bottom)

    for direction, band, min_rules in (
        ("below", below, MIN_RULING_LINES_BELOW),
        ("above", above, MIN_RULING_LINES_ABOVE),
    ):
        band_top, band_bottom = band
        spanning = [
            line for line in h_lines
            if band_top - 2.0 <= line["y"] <= band_bottom
            and _horizontal_overlap_ratio(caption_x, (line["x0"], line["x1"]))
            >= MIN_RULING_LINE_SPAN_RATIO
        ]
        band_verticals = [
            line for line in v_lines
            if line["y0"] <= band_bottom + 2.0 and line["y1"] >= band_top - 2.0
        ]
        native_matches = _match_regions(native_regions, band_top, band_bottom, caption_x)
        docling_matches = _match_regions(docling_regions, band_top, band_bottom, caption_x)

        has_structure = (
            len(spanning) >= min_rules
            or bool(native_matches)
            or bool(docling_matches)
        )
        if not has_structure:
            continue

        line_bbox = None
        if spanning:
            line_bbox = [
                min(line["x0"] for line in spanning),
                min(line["y"] for line in spanning),
                max(line["x1"] for line in spanning),
                max(line["y"] for line in spanning),
            ]

        return {
            "direction": direction,
            "ruling_line_count": len(spanning),
            "ruling_line_bbox": line_bbox,
            "vertical_line_count": len(band_verticals),
            "native_matches": native_matches,
            "docling_matches": docling_matches,
            "required_ruling_lines": min_rules,
            "has_structure": True,
        }

    return {
        "direction": None,
        "ruling_line_count": 0,
        "ruling_line_bbox": None,
        "vertical_line_count": 0,
        "native_matches": [],
        "docling_matches": [],
        "required_ruling_lines": MIN_RULING_LINES_BELOW,
        "has_structure": False,
    }

def _estimate_grid(evidence: Dict[str, Any]) -> Tuple[int, int, str]:
    """
    Estimate (rows, columns, source) from the strongest available structure.

    Priority: PyMuPDF find_tables dimensions > Docling markdown dimensions >
    geometry inferred from ruling lines.
    """
    native_matches = evidence.get("native_matches") or []
    if native_matches:
        best = max(
            native_matches,
            key=lambda region: region.get("rows", 0) * max(region.get("cols", 1), 1),
        )
        return (
            max(int(best.get("rows", 0)), 1),
            max(int(best.get("cols", 0)), 1),
            "pymupdf_find_tables",
        )

    docling_matches = evidence.get("docling_matches") or []
    if docling_matches:
        rows = 0
        columns = 0
        for region in docling_matches:
            grid = region.get("grid") or (0, 0)
            rows = max(rows, grid[0])
            columns = max(columns, grid[1])
        if rows and columns:
            return rows, columns, "docling"

    ruling_line_count = evidence.get("ruling_line_count", 0)
    if ruling_line_count:
        return (
            max(ruling_line_count - 1, 1),
            max(evidence.get("vertical_line_count", 0) + 1, 1),
            "ruling_lines",
        )

    return 0, 0, "none"




def detect_tables(
    parsed_pdf_info: Dict[str, Any],
    layout_data: Dict[str, Any] = None,
    config: Optional[PreAnalyzerConfig] = None,
) -> Dict[str, Any]:
    """
    Detect tables and return both candidates and structurally confirmed tables.

    ``items``/``count`` remain the confirmed tables (backward compatible), while
    ``table_candidates`` holds every caption-anchored candidate with its
    structural evidence, and ``rejected_table_mentions`` documents in-text
    references that were deliberately not treated as captions.
    """
    cfg = config or PreAnalyzerConfig()
    min_confidence = cfg.min_table_confidence
    band_pts = cfg.table_caption_search_band_pts

    # Docling recognized tables, indexed by page and normalized to the same shape
    # as the PyMuPDF find_tables regions so both can be matched identically.
    docling_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for item in (layout_data or {}).get("tables", []) or []:
        page_no = item.get("page")
        bbox = item.get("bbox")
        if not page_no or not bbox or len(bbox) != 4:
            continue
        docling_by_page.setdefault(int(page_no), []).append({
            "bbox": [float(v) for v in bbox],
            "grid": _docling_grid_size(item),
            "caption": item.get("caption", ""),
        })

    native_by_page = _collect_native_table_regions(parsed_pdf_info.get("pdf_path"))

    candidates: List[Dict[str, Any]] = []
    rejected_mentions: List[Dict[str, Any]] = []

    for page in parsed_pdf_info.get("pages", []):
        page_num = page.get("page_number")
        page_height = float(page.get("height") or 0.0) or 792.0
        h_lines, v_lines = _collect_ruling_lines(page.get("drawings", []))
        page_native = native_by_page.get(page_num, [])
        page_docling = docling_by_page.get(page_num, [])

        for block_index, block in enumerate(page.get("blocks", [])):
            if "lines" not in block:
                continue
            block_text = _extract_block_text(block)
            if not block_text:
                continue

            caption_match = TABLE_CAPTION_REGEX.match(block_text)
            if not caption_match:
                # Record every prose reference so rejections stay auditable.
                for mention in TABLE_MENTION_REGEX.finditer(block_text):
                    rejected_mentions.append({
                        "page": page_num,
                        "mentioned_table": mention.group(1),
                        "context": block_text[
                            max(0, mention.start() - 40): mention.end() + 120
                        ],
                        "text": block_text[:300],
                        "reason": "in_text_reference_not_caption_at_block_start",
                    })
                continue

            caption_bbox = [float(v) for v in block.get("bbox", [])]
            if len(caption_bbox) != 4:
                continue

            table_number = caption_match.group(1)
            evidence = _find_structural_evidence(
                caption_bbox=caption_bbox,
                page_height=page_height,
                band_pts=band_pts,
                h_lines=h_lines,
                v_lines=v_lines,
                native_regions=page_native,
                docling_regions=page_docling,
            )

            rows, columns, grid_source = _estimate_grid(evidence)
            has_native = bool(evidence["native_matches"])
            has_docling = bool(evidence["docling_matches"])
            ruling_ok = evidence["ruling_line_count"] >= evidence["required_ruling_lines"]
            has_structure = bool(evidence["has_structure"])
            is_meta = _is_author_metadata_block(block_text)

            confidence = calculate_table_confidence(
                has_caption=True,
                caption_matches_table_num=True,
                has_grid_or_drawings=has_structure,
                has_docling_match=has_docling,
                has_pymupdf_table=has_native,
                is_author_or_meta_block=is_meta,
            )

            confirmed = (
                has_structure
                and not is_meta
                and confidence >= min_confidence
            )
            if cfg.require_structural_evidence and not has_structure:
                confirmed = False

            # Table region = caption plus whatever structure was matched.
            region_bbox = list(caption_bbox)
            if evidence["ruling_line_bbox"]:
                region_bbox = _union_bbox(region_bbox, evidence["ruling_line_bbox"])
            for region in evidence["native_matches"] + evidence["docling_matches"]:
                region_bbox = _union_bbox(region_bbox, region["bbox"])

            candidates.append({
                "id": f"table_{page_num}_{table_number}_{block_index}",
                "table_number": table_number,
                "page": page_num,
                "caption": block_text,
                "bbox": region_bbox,
                "caption_bbox": caption_bbox,
                "rows": rows,
                "columns": columns,
                "status": "confirmed" if confirmed else "candidate",
                "structure_source": grid_source if confirmed else None,
                "evidence": {
                    "caption_at_block_start": True,
                    "structural_evidence_required": cfg.require_structural_evidence,
                    "ruling_lines_in_region": evidence["ruling_line_count"],
                    "ruling_lines_required": evidence["required_ruling_lines"],
                    "ruling_lines_spanning_caption": ruling_ok,
                    "vertical_lines_in_region": evidence["vertical_line_count"],
                    "structure_direction": evidence["direction"],
                    "pymupdf_find_tables_matched": has_native,
                    "docling_table_matched": has_docling,
                    "author_metadata_block": is_meta,
                    "vector_drawings_on_page": len(page.get("drawings", []) or []),
                },
                "confidence": confidence,
            })


    # Deduplicate confirmed tables by table number, keeping the highest confidence
    # (tie broken by the earliest page) so a stray duplicate caption cannot shadow
    # a real table.
    confirmed_by_number: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["status"] != "confirmed":
            continue
        number = candidate["table_number"]
        current = confirmed_by_number.get(number)
        if current is None or (
            candidate["confidence"],
            -candidate["page"],
        ) > (
            current["confidence"],
            -current["page"],
        ):
            confirmed_by_number[number] = candidate

    confirmed_tables = sorted(
        confirmed_by_number.values(),
        key=lambda t: (t["page"], int(t["table_number"]) if t["table_number"].isdigit() else 999),
    )

    for table in confirmed_tables:
        table["id"] = f"table_{table['table_number']}"

    rejected_candidates = [c for c in candidates if c["status"] != "confirmed"]

    return {
        "count": len(confirmed_tables),
        "items": confirmed_tables,
        "confirmed_tables": confirmed_tables,
        "table_candidates": candidates,
        "candidate_count": len(candidates),
        "rejected_candidates": rejected_candidates,
        "rejected_candidate_count": len(rejected_candidates),
        "rejected_table_mentions": rejected_mentions,
        "rejected_table_mention_count": len(rejected_mentions),
    }

