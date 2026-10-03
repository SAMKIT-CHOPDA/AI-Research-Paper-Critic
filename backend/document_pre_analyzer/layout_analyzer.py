"""
Layout analyzer bridging PyMuPDF coordinates and Docling layout analysis.

Uses Docling (with OCR turned off for speed and VRAM efficiency) to extract:
- section_header items
- formula items
- table items
- picture items
- list items / bibliography

Coordinate space
----------------
Docling reports page bounding boxes in PDF user space (origin bottom-left,
``CoordOrigin.BOTTOMLEFT``), while PyMuPDF reports them with a top-left origin.
Every bbox handed out by this module is therefore normalized to PyMuPDF's
top-left space (and the original is preserved as ``bbox_raw``) so the two
sources can actually be cross-validated geometrically.
"""

from typing import Dict, Any, List, Optional


def _normalize_bbox(bbox_obj: Any, page_height: Optional[float]) -> Optional[List[float]]:
    """
    Convert a Docling bbox to PyMuPDF top-left coordinates.

    Falls back to a vertical flip when ``coord_origin`` is unavailable, which is
    Docling's default ``BOTTOMLEFT`` behaviour.
    """
    if bbox_obj is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox_obj.as_tuple())
    except Exception:
        return None

    origin = str(getattr(bbox_obj, "coord_origin", "")).upper()
    if "TOP" in origin:
        return [x0, y0, x1, y1]
    if not page_height:
        return [x0, y0, x1, y1]
    return [x0, float(page_height) - y1, x1, float(page_height) - y0]


class LayoutAnalyzer:
    """Wrapper around Docling layout converter with safe fallbacks."""

    def __init__(self, use_docling: bool = True):
        self.use_docling = use_docling
        self._converter = None
        self._cached_results: Dict[str, Any] = {}

    def _get_converter(self):
        if not self.use_docling:
            return None
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat

                opts = PdfPipelineOptions()
                opts.do_ocr = False
                opts.do_table_structure = True

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
                    }
                )
            except Exception:
                self._converter = None
                self.use_docling = False
        return self._converter

    def analyze_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF layout using Docling."""
        if pdf_path in self._cached_results:
            return self._cached_results[pdf_path]

        empty_result: Dict[str, Any] = {
            "success": False,
            "engine": "none",
            "bbox_coordinate_space": "pymupdf_top_left",
            "headers": [],
            "tables": [],
            "pictures": [],
            "formulas": [],
            "list_items": [],
        }

        converter = self._get_converter()
        if converter is None:
            return empty_result

        try:
            res = converter.convert(pdf_path)
            doc = res.document

            headers = []
            tables = []
            pictures = []
            formulas = []
            list_items = []

            # Page heights are needed to flip Docling's bottom-left origin bboxes
            # into the top-left origin used by PyMuPDF.
            page_heights: Dict[int, float] = {}
            try:
                for page_no, page_item in (doc.pages or {}).items():
                    page_heights[int(page_no)] = float(page_item.size.height)
            except Exception:
                page_heights = {}

            for item, level in doc.iterate_items():
                label = str(getattr(item, "label", type(item).__name__))
                prov = getattr(item, "prov", [])
                pno = prov[0].page_no if prov else None
                bbox_raw = list(prov[0].bbox.as_tuple()) if prov else None
                bbox = _normalize_bbox(prov[0].bbox, page_heights.get(pno)) if prov else None
                text = getattr(item, "text", "") or ""

                if "section" in label or "header" in label or "title" in label:
                    headers.append({
                        "page": pno,
                        "bbox": bbox,
                        "bbox_raw": bbox_raw,
                        "text": text.strip(),
                        "level": level,
                    })

                elif "table" in label:
                    caption_text = ""
                    captions = getattr(item, "captions", [])
                    if captions:
                        try:
                            cap_item = captions[0].resolve(doc)
                            caption_text = getattr(cap_item, "text", "")
                        except Exception:
                            caption_text = ""

                    data_md = ""
                    try:
                        data_md = item.export_to_markdown(doc=doc)
                    except Exception:
                        pass

                    tables.append({
                        "page": pno,
                        "bbox": bbox,
                        "bbox_raw": bbox_raw,
                        "caption": caption_text.strip(),
                        "markdown": data_md,
                    })

                elif "picture" in label:
                    caption_text = ""
                    captions = getattr(item, "captions", [])
                    if captions:
                        try:
                            cap_item = captions[0].resolve(doc)
                            caption_text = getattr(cap_item, "text", "")
                        except Exception:
                            caption_text = ""

                    pictures.append({
                        "page": pno,
                        "bbox": bbox,
                        "bbox_raw": bbox_raw,
                        "caption": caption_text.strip(),
                    })

                elif "formula" in label or "math" in label:
                    orig_text = getattr(item, "orig", "") or ""
                    formulas.append({
                        "page": pno,
                        "bbox": bbox,
                        "bbox_raw": bbox_raw,
                        "text": orig_text or text,
                    })

                elif "list_item" in label:
                    list_items.append({
                        "page": pno,
                        "bbox": bbox,
                        "bbox_raw": bbox_raw,
                        "text": text.strip(),
                    })

            result = {
                "success": True,
                "engine": "docling",
                "bbox_coordinate_space": "pymupdf_top_left",
                "headers": headers,
                "tables": tables,
                "pictures": pictures,
                "formulas": formulas,
                "list_items": list_items,
            }
            self._cached_results[pdf_path] = result
            return result

        except Exception as e:
            err_res = dict(empty_result)
            err_res["error"] = str(e)
            return err_res
