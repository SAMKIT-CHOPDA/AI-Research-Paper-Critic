"""
Visual asset extraction from a confirmed Document Profile.

The frozen Document Pre-Analyzer owns figure/table *discovery*. This module does
not run any detector of its own: it consumes the profile's confirmed visual
inventory (``figures.confirmed_figures`` / ``tables.confirmed_tables``) and
renders the corresponding region of the original PDF.

Coordinate contract
-------------------
The Document Pre-Analyzer normalizes every bounding box it emits into PyMuPDF
top-left origin coordinates (points). This module never reinterprets that space:
boxes are validated, clamped to the page rectangle and passed to
``page.get_pixmap(clip=...)``, which uses the same top-left space. Each extracted
asset records ``coordinate_space`` so a mismatch would be visible, not silent.

Extraction strategy
-------------------
* figure  -> precise bbox crop (preferred)
* table   -> precise bbox crop (preferred); caption-band or whole-page crop only
             when the profile has no usable box, and the asset is then flagged
             with ``fallback_used=True`` plus a non-exact ``extraction_method``.
* equation -> never a separate vision subsystem; equations are only carried as
             later-stage context by the agent.

Failure isolation: one unrenderable asset never aborts the run. The failure is
recorded in the report and the remaining assets are still processed.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pymupdf

from backend.vision_agent.config import (
    DEFAULT_ASSET_OUTPUT_DIR,
    VisionConfig,
)
from backend.vision_agent.image_preprocessor import (
    ImagePreprocessingError,
    prepare_image,
)
from backend.vision_agent.schemas import (
    COORDINATE_SPACE_PYMUPDF_TOP_LEFT,
    AssetExtractionFailure,
    ExtractionReport,
    VisualAsset,
)

logger = logging.getLogger(__name__)

# Vertical band (points) used when a confirmed element has a caption box but no
# usable content box: enough to cover a typical figure/table above or below it.
FALLBACK_BAND_PTS: float = 300.0

# A box thinner than this in either axis is treated as unusable (a hairline box
# would produce an unreadable strip and must not be sent to the model silently).
MIN_USABLE_BOX_PTS: float = 12.0

# Machine-readable table extraction is capped so one huge table cannot dominate
# the prompt payload.
MAX_STRUCTURED_TABLE_ROWS: int = 30
MAX_STRUCTURED_TABLE_COLUMNS: int = 12


class AssetExtractionError(Exception):
    """Raised when a visual asset region cannot be extracted at all."""
    pass


@dataclass
class ConfirmedAssetRef:
    """
    A confirmed visual element taken from the Document Profile.

    ``asset_id`` is the Vision Agent's canonical, deterministic identifier
    (figure_001 / table_001, assigned in page order). ``source_id`` preserves
    the Document Profile identifier so every result stays traceable upstream.
    """

    asset_id: str
    asset_type: str
    page_number: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    label: Optional[str] = None
    source_id: Optional[str] = None
    source_confidence: float = 0.0
    asset_subtype: Optional[str] = None
    caption_bbox: Optional[List[float]] = None
    profile_table_info: Dict[str, Any] = field(default_factory=dict)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bbox(bbox: Any, page_rect: Any = None) -> Optional[List[float]]:
    """
    Validate and normalize a bbox expressed in PyMuPDF top-left coordinates.

    Returns ``[x0, y0, x1, y1]`` clamped to the page rectangle, or None when the
    box is malformed or too small to be useful. Inverted coordinates are
    normalized rather than rejected, because an inverted box is a recoverable
    ordering issue, not a different coordinate space.
    """
    if bbox is None:
        return None
    try:
        values = list(bbox)
    except TypeError:
        return None
    if len(values) != 4:
        return None

    numbers = [_as_float(v) for v in values]
    if any(v is None for v in numbers):
        return None

    x0, y0, x1, y1 = numbers
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    if page_rect is not None:
        x0 = max(float(page_rect.x0), x0)
        y0 = max(float(page_rect.y0), y0)
        x1 = min(float(page_rect.x1), x1)
        y1 = min(float(page_rect.y1), y1)

    if (x1 - x0) < MIN_USABLE_BOX_PTS or (y1 - y0) < MIN_USABLE_BOX_PTS:
        return None

    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def section_for_page(document_profile: Optional[Dict[str, Any]], page_number: int) -> Optional[str]:
    """Return the section that owns a page, using the profile's section list."""
    if not document_profile:
        return None
    sections = document_profile.get("sections") or []
    candidates = [
        s for s in sections
        if isinstance(s, dict) and isinstance(s.get("page"), int) and s["page"] <= page_number
    ]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda s: s["page"])
    return chosen.get("title") or chosen.get("normalized_title") or None


def _confirmed_items(document_profile: Dict[str, Any], key: str, confirmed_key: str) -> List[Dict[str, Any]]:
    block = (document_profile or {}).get(key) or {}
    items = block.get(confirmed_key)
    if items is None:
        items = block.get("items")
    return [item for item in (items or []) if isinstance(item, dict)]


def collect_confirmed_visual_assets(
    document_profile: Dict[str, Any],
    include_candidates: bool = False,
    max_assets: Optional[int] = None,
) -> List[ConfirmedAssetRef]:
    """
    Build the ordered visual inventory from the Document Profile.

    Confirmed elements are the default input. Candidates (elements the frozen
    Pre-Analyzer deliberately did not confirm) are only included when the caller
    explicitly opts in, so a rejected prose mention can never become a visual
    analysis by accident.

    Returns refs sorted by (page, asset type, source id) with deterministic
    canonical ids: figure_001, figure_002, ... and table_001, table_002, ...
    """
    if not document_profile:
        return []

    figure_keys = ["confirmed_figures"]
    table_keys = ["confirmed_tables"]
    if include_candidates:
        figure_keys.append("figure_candidates")
        table_keys.append("table_candidates")

    raw_entries: List[Dict[str, Any]] = []
    for key in figure_keys:
        for item in _confirmed_items(document_profile, "figures", key):
            if key != "confirmed_figures" and item.get("status") == "confirmed":
                continue  # already taken from the confirmed list
            raw_entries.append({"asset_type": "figure", "item": item})
    for key in table_keys:
        for item in _confirmed_items(document_profile, "tables", key):
            if key != "confirmed_tables" and item.get("status") == "confirmed":
                continue  # already taken from the confirmed list
            raw_entries.append({"asset_type": "table", "item": item})

    def sort_key(entry: Dict[str, Any]):
        item = entry["item"]
        page = item.get("page")
        page_value = page if isinstance(page, int) else 10**6
        return (page_value, entry["asset_type"], str(item.get("id") or ""))

    raw_entries.sort(key=sort_key)

    refs: List[ConfirmedAssetRef] = []
    counters = {"figure": 0, "table": 0}
    for entry in raw_entries:
        item = entry["item"]
        asset_type = entry["asset_type"]
        page = item.get("page")
        if not isinstance(page, int) or page < 1:
            # Without a page the asset cannot be located in the PDF at all.
            logger.warning("Skipping %s without a usable page number: %s", asset_type, item.get("id"))
            continue

        counters[asset_type] += 1
        label = item.get("figure_number") if asset_type == "figure" else item.get("table_number")
        profile_table_info: Dict[str, Any] = {}
        if asset_type == "table":
            for key in ("rows", "columns", "structure_source"):
                if item.get(key) is not None:
                    profile_table_info[key] = item.get(key)

        refs.append(
            ConfirmedAssetRef(
                asset_id=f"{asset_type}_{counters[asset_type]:03d}",
                asset_type=asset_type,
                page_number=int(page),
                bbox=item.get("bbox"),
                caption=(item.get("caption") or None),
                label=(str(label) if label is not None else None),
                source_id=item.get("id"),
                source_confidence=float(item.get("confidence") or 0.0),
                asset_subtype=item.get("type") or item.get("structure_source"),
                caption_bbox=item.get("caption_bbox"),
                profile_table_info=profile_table_info,
            )
        )

    if max_assets is not None and max_assets > 0:
        refs = refs[:max_assets]
    return refs


def _to_rect(bbox: Any, page_rect: Any) -> Optional["pymupdf.Rect"]:
    """
    Convert a profile bbox into a clamped PyMuPDF rect without a minimum-size
    test (used for caption boxes, which are legitimately thin).
    """
    if bbox is None:
        return None
    try:
        values = list(bbox)
    except TypeError:
        return None
    if len(values) != 4:
        return None
    numbers = [_as_float(v) for v in values]
    if any(v is None for v in numbers):
        return None

    x0, y0, x1, y1 = numbers
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    rect = pymupdf.Rect(x0, y0, x1, y1)
    if page_rect is not None:
        rect = rect & page_rect
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _padded_rect(rect: "pymupdf.Rect", padding: float, page_rect: "pymupdf.Rect") -> "pymupdf.Rect":
    """Expand a crop by a small padding so labels are not shaved off, then clamp."""
    if padding <= 0:
        return rect & page_rect
    padded = pymupdf.Rect(
        rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding
    )
    return padded & page_rect


def _render_region(page: "pymupdf.Page", rect: "pymupdf.Rect", dpi: int) -> bytes:
    """Render a page region to PNG bytes at the configured DPI."""
    try:
        pixmap = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
        data = pixmap.tobytes("png")
    except Exception as exc:
        raise AssetExtractionError(f"Failed to render page region: {exc}") from exc
    if not data:
        raise AssetExtractionError("Rendering produced an empty image")
    return data


def _caption_band_rect(
    asset_type: str, caption_rect: "pymupdf.Rect", page_rect: "pymupdf.Rect"
) -> "pymupdf.Rect":
    """
    Fallback region for an element with a caption but no trustworthy content box.

    Figures normally sit above their caption and tables below it, which mirrors
    the search direction the frozen Pre-Analyzer uses during detection.
    """
    if asset_type == "figure":
        band = pymupdf.Rect(
            caption_rect.x0,
            max(page_rect.y0, caption_rect.y0 - FALLBACK_BAND_PTS),
            caption_rect.x1,
            caption_rect.y1,
        )
    else:
        band = pymupdf.Rect(
            caption_rect.x0,
            caption_rect.y0,
            caption_rect.x1,
            min(page_rect.y1, caption_rect.y1 + FALLBACK_BAND_PTS),
        )
    return band & page_rect


def _neighbouring_text(
    page: "pymupdf.Page", rect: "pymupdf.Rect", max_chars: int
) -> Optional[str]:
    """
    Bounded surrounding text: the nearest text blocks above and below the asset.

    Only a small neighbourhood is returned so the prompt stays focused; the full
    paper text is never attached to an image.
    """
    if max_chars <= 0:
        return None
    try:
        blocks = page.get_text("blocks") or []
    except Exception as exc:  # pragma: no cover - PyMuPDF failure path
        logger.warning("Unable to read page text for asset context: %s", exc)
        return None

    above: List[Any] = []
    below: List[Any] = []
    for block in blocks:
        if len(block) < 5 or not isinstance(block[4], str):
            continue
        if len(block) > 6 and block[6] != 0:
            continue  # image block, not text
        text = " ".join(block[4].split())
        if not text:
            continue
        y0, y1 = float(block[1]), float(block[3])
        if y1 <= rect.y0:
            above.append((rect.y0 - y1, text))
        elif y0 >= rect.y1:
            below.append((y0 - rect.y1, text))

    ordered = [text for _, text in sorted(above)[:2]] + [
        text for _, text in sorted(below)[:2]
    ]
    joined = " ".join(ordered).strip()
    if not joined:
        return None
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "..."
    return joined


def _structured_table_from_page(
    page: "pymupdf.Page", rect: "pymupdf.Rect"
) -> Optional[Dict[str, Any]]:
    """
    Read machine-readable table cells for a confirmed table region.

    Exact numeric values in a research table must not be sourced from a vision
    model when the PDF itself exposes structure, so the text layer is read
    directly and preserved as authoritative context.
    """
    try:
        finder = page.find_tables(clip=rect)
        tables = list(getattr(finder, "tables", None) or [])
    except Exception as exc:
        logger.warning("Machine-readable table read failed: %s", exc)
        return None
    if not tables:
        return None

    table = tables[0]
    try:
        raw_rows = table.extract() or []
    except Exception as exc:
        logger.warning("Table cell extraction failed: %s", exc)
        return None
    if not raw_rows:
        return None

    def _clean(cell: Any) -> str:
        return " ".join(str(cell).split()) if cell is not None else ""

    cleaned_rows = [[_clean(cell) for cell in (row or [])] for row in raw_rows]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    if not cleaned_rows:
        return None

    column_count = max(len(row) for row in cleaned_rows)
    headers = cleaned_rows[0][:MAX_STRUCTURED_TABLE_COLUMNS]
    body = [
        row[:MAX_STRUCTURED_TABLE_COLUMNS] for row in cleaned_rows[1: MAX_STRUCTURED_TABLE_ROWS + 1]
    ]

    return {
        "source": "pymupdf_find_tables",
        "cell_row_count": len(cleaned_rows),
        "cell_column_count": column_count,
        "headers": headers,
        "rows": body,
        "truncated": len(cleaned_rows) > (MAX_STRUCTURED_TABLE_ROWS + 1)
        or column_count > MAX_STRUCTURED_TABLE_COLUMNS,
        "authoritative_for_exact_values": True,
    }


def _extract_one(
    doc: "pymupdf.Document",
    ref: ConfirmedAssetRef,
    config: VisionConfig,
    output_dir: str,
    document_profile: Optional[Dict[str, Any]],
) -> VisualAsset:
    """Extract a single confirmed asset, raising AssetExtractionError on failure."""
    if ref.page_number > doc.page_count:
        raise AssetExtractionError(
            f"Page {ref.page_number} is not present in the PDF "
            f"({doc.page_count} pages available)"
        )

    page = doc[ref.page_number - 1]
    page_rect = page.rect

    usable_bbox = normalize_bbox(ref.bbox, page_rect)
    if usable_bbox is not None:
        crop_rect = pymupdf.Rect(*usable_bbox)
        extraction_method = "bbox_crop"
    else:
        caption_rect = _to_rect(ref.caption_bbox, page_rect)
        if caption_rect is not None and caption_rect.height >= 2.0:
            crop_rect = _caption_band_rect(ref.asset_type, caption_rect, page_rect)
            extraction_method = "caption_band_crop"
        else:
            crop_rect = pymupdf.Rect(page_rect)
            extraction_method = "page_crop_fallback"

    crop_rect = _padded_rect(crop_rect, config.bbox_padding_pts, page_rect)
    if crop_rect.is_empty or crop_rect.width <= 0 or crop_rect.height <= 0:
        raise AssetExtractionError(
            f"Resolved crop region is empty for {ref.asset_id}"
        )

    raw_image = _render_region(page, crop_rect, config.render_dpi)
    try:
        prepared = prepare_image(
            raw_image,
            max_dimension=config.max_image_dimension,
            image_format=config.image_format,
            jpeg_quality=config.jpeg_quality,
        )
    except ImagePreprocessingError as exc:
        raise AssetExtractionError(f"Image preprocessing failed: {exc}") from exc

    extension = "jpg" if prepared.image_format == "JPEG" else "png"
    image_path = os.path.join(output_dir, f"{ref.asset_id}.{extension}")
    try:
        with open(image_path, "wb") as handle:
            handle.write(prepared.data)
    except OSError as exc:
        raise AssetExtractionError(f"Unable to write asset image: {exc}") from exc

    structured_table: Optional[Dict[str, Any]] = None
    if ref.asset_type == "table":
        structured_table = _structured_table_from_page(page, crop_rect)
        if structured_table is None and ref.profile_table_info:
            # Preserve the Pre-Analyzer's structure metadata as context, while
            # stating clearly that exact cell values were not machine-read.
            # Counts use *_count keys so they can never be confused with the
            # machine-readable "rows" list of cell values.
            structured_table = {
                "source": "document_profile",
                "row_count": ref.profile_table_info.get("rows"),
                "column_count": ref.profile_table_info.get("columns"),
                "structure_source": ref.profile_table_info.get("structure_source"),
                "authoritative_for_exact_values": False,
            }

    return VisualAsset(
        asset_id=ref.asset_id,
        asset_type=ref.asset_type,
        page_number=ref.page_number,
        bbox=[round(crop_rect.x0, 2), round(crop_rect.y0, 2), round(crop_rect.x1, 2), round(crop_rect.y1, 2)],
        image_path=os.path.abspath(image_path),
        source_confidence=ref.source_confidence,
        source_id=ref.source_id,
        asset_subtype=ref.asset_subtype,
        label=ref.label,
        caption=ref.caption,
        section=section_for_page(document_profile, ref.page_number),
        context_text=_neighbouring_text(page, crop_rect, config.context_max_chars),
        extraction_method=extraction_method,
        fallback_used=extraction_method != "bbox_crop",
        coordinate_space=COORDINATE_SPACE_PYMUPDF_TOP_LEFT,
        image_width=prepared.width,
        image_height=prepared.height,
        image_format=prepared.image_format,
        image_bytes=prepared.size_bytes,
        resized=prepared.resized,
        structured_table=structured_table,
    )


def extract_visual_assets(
    pdf_path: str,
    document_profile: Optional[Dict[str, Any]] = None,
    config: Optional[VisionConfig] = None,
    output_dir: Optional[str] = None,
    assets: Optional[Iterable[ConfirmedAssetRef]] = None,
) -> ExtractionReport:
    """
    Extract every confirmed visual asset from the PDF.

    Parameters
    - pdf_path: the original paper PDF
    - document_profile: the frozen Document Pre-Analyzer's profile (visual inventory)
    - config: VisionConfig (defaults from environment)
    - output_dir: where extracted images are written
    - assets: optional pre-collected refs (used by tests and callers that already
      hold the inventory)

    Failures are isolated per asset: the report lists them instead of raising, so
    one unrenderable figure cannot destroy the paper analysis.
    """
    cfg = config or VisionConfig()

    if assets is not None:
        refs = list(assets)
        considered = len(refs)
        skipped = 0
    else:
        all_refs = collect_confirmed_visual_assets(
            document_profile or {},
            include_candidates=cfg.include_candidate_assets,
            max_assets=None,
        )
        limit = cfg.max_assets if cfg.max_assets and cfg.max_assets > 0 else None
        refs = all_refs[:limit] if limit else all_refs
        considered = len(refs)
        skipped = max(0, len(all_refs) - len(refs))

    report = ExtractionReport(assets_considered=considered, assets_skipped_over_limit=skipped)
    if not refs:
        return report

    target_dir = output_dir or cfg.output_dir or DEFAULT_ASSET_OUTPUT_DIR
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        raise AssetExtractionError(f"Unable to create asset output directory: {exc}") from exc

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise AssetExtractionError(f"Unable to open PDF '{pdf_path}': {exc}") from exc

    try:
        if doc.page_count == 0:
            raise AssetExtractionError(f"PDF '{pdf_path}' contains no pages")
        for ref in refs:
            try:
                report.assets.append(
                    _extract_one(doc, ref, cfg, target_dir, document_profile)
                )
            except Exception as exc:
                logger.warning("Asset extraction failed for %s: %s", ref.asset_id, exc)
                report.failures.append(
                    AssetExtractionFailure(
                        asset_id=ref.asset_id,
                        asset_type=ref.asset_type,
                        page_number=ref.page_number,
                        source_id=ref.source_id,
                        stage="extraction",
                        error=str(exc),
                    )
                )
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover - defensive
            pass

    return report