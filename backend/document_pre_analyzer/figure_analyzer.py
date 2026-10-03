"""
Figure analyzer detecting figures via captions, raster images, and vector layout.

Separates figure_candidates (caption-anchored, always reported) from
confirmed_figures (candidates with independent visual evidence), so a caption is
never on its own enough to declare a figure:

1. Candidate discovery - a text block must *start* with a figure caption
   (``Figure 1:``, ``Fig. 1:``). Mid-sentence prose such as
   "as shown in Figure 2. The Transformer ..." is recorded under
   ``rejected_figure_mentions`` instead of being treated as a caption.
2. Confirmation - a raster image (XObject), a Docling picture region, or a
   cluster of vector drawings must be located next to the caption (above it,
   the dominant convention, or below it as a fallback).

Distinguishes raster diagrams, vector graphics, and composite figures, and never
equates raw image counts to figures.
"""

import re
from typing import Dict, Any, List, Tuple

from .confidence import calculate_figure_confidence
from .config import PreAnalyzerConfig

# Anchored: a figure caption must begin the block.
FIGURE_CAPTION_REGEX = re.compile(
    r"^Fig(?:ure)?\.?\s+(\d+)\s*[:.\-—]\s*(.*)", re.IGNORECASE
)
# Any in-text mention of a figure, used only to build the rejection audit trail.
FIGURE_MENTION_REGEX = re.compile(r"\bFig(?:ure)?\.?\s+(\d+)\b", re.IGNORECASE)

MIN_REGION_OVERLAP_RATIO = 0.20   # fraction of caption width a visual must span
MIN_VECTOR_STROKES_IN_REGION = 3  # strokes needed to call a region "vector art"


def _bbox_union(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
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


def _x_overlap_ratio(
    range_a: Tuple[float, float], range_b: Tuple[float, float]
) -> float:
    """Intersection width of two x-ranges divided by the narrower range width."""
    low = max(range_a[0], range_b[0])
    high = min(range_a[1], range_b[1])
    narrower = min(range_a[1] - range_a[0], range_b[1] - range_b[0])
    if high <= low or narrower <= 0:
        return 0.0
    return (high - low) / narrower


def _collect_block_text(block: Dict[str, Any]) -> str:
    """Flatten a PyMuPDF text block into a single normalized string."""
    return " ".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    ).strip()


def _visuals_in_band(
    caption_bbox: List[float],
    band: Tuple[float, float],
    images: List[Dict[str, Any]],
    docling_pictures: List[Dict[str, Any]],
    drawings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Locate visual elements inside a vertical band next to a caption.

    Only elements whose vertical extent falls inside the band and whose x-range
    overlaps the caption horizontally are considered, so page-wide decorations
    cannot masquerade as figure artwork.
    """
    band_top, band_bottom = band
    caption_x = (caption_bbox[0], caption_bbox[2])

    def _in_band(bbox: List[float]) -> bool:
        if not bbox or len(bbox) != 4:
            return False
        overlaps_vertically = bbox[3] > band_top and bbox[1] < band_bottom
        return overlaps_vertically and \
            _x_overlap_ratio(caption_x, (bbox[0], bbox[2])) >= MIN_REGION_OVERLAP_RATIO

    raster = [img for img in images or [] if _in_band(img.get("bbox") or [])]
    docling = [pic for pic in docling_pictures or [] if _in_band(pic.get("bbox") or [])]
    vector_strokes = [
        drawing for drawing in drawings or []
        if _in_band(drawing.get("rect") or [])
    ]

    return {
        "raster_images": raster,
        "docling_pictures": docling,
        "vector_strokes": len(vector_strokes),
    }

def detect_figures(
    parsed_pdf_info: Dict[str, Any],
    layout_data: Dict[str, Any] = None,
    config: PreAnalyzerConfig = None,
) -> Dict[str, Any]:
    """
    Detect figures and return both candidates and visually confirmed figures.

    ``items``/``count`` remain the confirmed figures (backward compatible);
    ``figure_candidates`` holds every caption-anchored candidate with its visual
    evidence, and ``rejected_figure_mentions`` documents in-text references that
    were deliberately not treated as captions.
    """
    cfg = config or PreAnalyzerConfig()
    min_confidence = cfg.min_figure_confidence
    band_pts = cfg.table_caption_search_band_pts

    docling_pictures = (layout_data or {}).get("pictures", []) or []
    pictures_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for picture in docling_pictures:
        page_no = picture.get("page")
        if page_no:
            pictures_by_page.setdefault(int(page_no), []).append(picture)

    candidates: List[Dict[str, Any]] = []
    rejected_mentions: List[Dict[str, Any]] = []

    for page in parsed_pdf_info.get("pages", []):
        page_num = page["page_number"]
        page_images = page.get("images", []) or []
        page_drawings = page.get("drawings", []) or []
        page_pictures = pictures_by_page.get(page_num, [])
        page_height = float(page.get("height") or 0.0) or 792.0

        for block_index, block in enumerate(page.get("blocks", [])):
            if "lines" not in block:
                continue
            block_text = _collect_block_text(block)
            if not block_text:
                continue

            caption_match = FIGURE_CAPTION_REGEX.match(block_text)
            if not caption_match:
                for mention in FIGURE_MENTION_REGEX.finditer(block_text):
                    rejected_mentions.append({
                        "page": page_num,
                        "mentioned_figure": mention.group(1),
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

            figure_number = caption_match.group(1)
            caption_top = caption_bbox[1]
            caption_bottom = caption_bbox[3]

            # Figures normally sit above their caption; search there first and
            # fall back to a short band below the caption.
            search_order = (
                ("above", (max(0.0, caption_top - band_pts), caption_bottom)),
                ("below", (caption_top, min(page_height, caption_bottom + 90.0))),
            )

            direction = None
            visuals: Dict[str, Any] = {
                "raster_images": [],
                "docling_pictures": [],
                "vector_strokes": 0,
            }
            for candidate_direction, band in search_order:
                found = _visuals_in_band(
                    caption_bbox, band, page_images, page_pictures, page_drawings
                )
                if found["raster_images"] or found["docling_pictures"] \
                        or found["vector_strokes"] >= MIN_VECTOR_STROKES_IN_REGION:
                    direction, visuals = candidate_direction, found
                    break
                if direction is None:
                    direction, visuals = candidate_direction, found

            has_raster = bool(visuals["raster_images"])
            has_docling_pic = bool(visuals["docling_pictures"])
            has_vector = visuals["vector_strokes"] >= MIN_VECTOR_STROKES_IN_REGION
            has_visual_evidence = has_raster or has_docling_pic or has_vector

            figure_bbox = list(caption_bbox)
            if has_raster:
                for image in visuals["raster_images"]:
                    figure_bbox = _bbox_union(figure_bbox, image.get("bbox") or [])
            if has_docling_pic:
                for picture in visuals["docling_pictures"]:
                    figure_bbox = _bbox_union(figure_bbox, picture.get("bbox") or [])

            if has_raster:
                figure_type = "raster_image"
            elif has_vector:
                figure_type = "vector_graphic"
            elif has_docling_pic:
                figure_type = "diagram"
            else:
                figure_type = "unverified_caption"

            confidence = calculate_figure_confidence(
                has_caption=True,
                has_raster_image=has_raster,
                has_vector_drawings=has_vector,
                has_docling_picture=has_docling_pic,
            )
            confirmed = has_visual_evidence and confidence >= min_confidence

            candidates.append({
                "id": f"figure_{figure_number}",
                "figure_number": figure_number,
                "page": page_num,
                "caption": block_text,
                "caption_bbox": caption_bbox,
                "bbox": figure_bbox,
                "type": figure_type,
                "status": "confirmed" if confirmed else "candidate",
                "evidence": {
                    "caption_at_block_start": True,
                    "visual_search_direction": direction,
                    "raster_image_matched": has_raster,
                    "raster_image_bboxes": [
                        img.get("bbox") for img in visuals["raster_images"]
                    ],
                    "docling_picture_matched": has_docling_pic,
                    "vector_strokes_in_region": visuals["vector_strokes"],
                    "vector_strokes_required": MIN_VECTOR_STROKES_IN_REGION,
                    "vector_drawings_on_page": len(page_drawings),
                    "raster_images_on_page": len(page_images),
                },
                "confidence": confidence,
            })


    # Docling pictures with no caption at all are kept as confirmed but unlabeled
    # graphics; they never reuse a real figure number.
    for picture_index, picture in enumerate(docling_pictures):
        page_no = picture.get("page")
        bbox = picture.get("bbox")
        if not page_no or not bbox:
            continue
        if any(c["page"] == page_no and c.get("bbox") == bbox for c in candidates):
            continue
        if any(c["page"] == page_no for c in candidates):
            continue
        candidates.append({
            "id": f"figure_unlabeled_{page_no}_{picture_index}",
            "figure_number": None,
            "page": int(page_no),
            "caption": picture.get("caption", ""),
            "caption_bbox": None,
            "bbox": [float(v) for v in bbox],
            "type": "unlabeled_graphic",
            "status": "confirmed",
            "evidence": {
                "caption_at_block_start": False,
                "visual_search_direction": None,
                "raster_image_matched": False,
                "raster_image_bboxes": [],
                "docling_picture_matched": True,
                "vector_strokes_in_region": 0,
                "vector_strokes_required": MIN_VECTOR_STROKES_IN_REGION,
                "vector_drawings_on_page": 0,
                "raster_images_on_page": 0,
            },
            "confidence": 0.55,
        })

    # Deduplicate confirmed figures by figure number (highest confidence wins, so a
    # stray duplicate caption cannot shadow a real figure).
    confirmed_by_number: Dict[str, Dict[str, Any]] = {}
    confirmed_unlabeled: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["status"] != "confirmed":
            continue
        number = candidate["figure_number"]
        if number is None:
            confirmed_unlabeled.append(candidate)
            continue
        current = confirmed_by_number.get(number)
        if current is None or (
            candidate["confidence"], -candidate["page"]
        ) > (current["confidence"], -current["page"]):
            confirmed_by_number[number] = candidate

    def _sort_key(figure: Dict[str, Any]) -> Tuple[int, int]:
        number = figure.get("figure_number") or ""
        return (
            figure["page"],
            int(number) if str(number).isdigit() else 999,
        )

    confirmed_figures = sorted(
        list(confirmed_by_number.values()) + confirmed_unlabeled,
        key=_sort_key,
    )
    rejected_candidates = [c for c in candidates if c["status"] != "confirmed"]

    return {
        "count": len(confirmed_figures),
        "items": confirmed_figures,
        "confirmed_figures": confirmed_figures,
        "figure_candidates": candidates,
        "candidate_count": len(candidates),
        "rejected_candidates": rejected_candidates,
        "rejected_candidate_count": len(rejected_candidates),
        "rejected_figure_mentions": rejected_mentions,
        "rejected_figure_mention_count": len(rejected_mentions),
    }

