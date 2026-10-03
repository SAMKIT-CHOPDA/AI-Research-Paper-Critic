"""
PDF parser extracting low-level PDF objects using modern PyMuPDF.

Extracts:
- Document metadata
- Page dimensions
- Text blocks, lines, spans with font properties (size, font name, flags)
- Raster images (XObjects) with page coordinates
- Vector drawings and ruling lines with coordinates
"""

from typing import Dict, Any, List
import pymupdf


def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Parse a PDF file using modern PyMuPDF API (pymupdf).

    Extracts detailed raw information from every page without destructive summaries.
    Preserves raw evidence for specialized detectors.
    """
    doc = pymupdf.open(pdf_path)

    pages: List[Dict[str, Any]] = []
    total_images_count = 0
    total_drawings_count = 0
    total_text_length = 0
    total_word_count = 0

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_rect = page.rect
        page_text = page.get_text("text", sort=True)
        page_dict = page.get_text("dict", sort=True)

        # Extract raster image rects and metadata
        image_list = page.get_images(full=True)
        parsed_images = []
        for img_info in image_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            bbox = list(rects[0]) if rects else None
            parsed_images.append({
                "xref": xref,
                "bbox": bbox,
                "width": img_info[2],
                "height": img_info[3],
                "colorspace": img_info[5],
                "name": img_info[7],
            })

        # Extract vector drawings
        drawings = page.get_drawings()
        parsed_drawings = []
        for d in drawings:
            parsed_drawings.append({
                "rect": list(d.get("rect", [])),
                "fill": d.get("fill"),
                "color": d.get("color"),
                "width": d.get("width"),
            })

        # Calculate font size distribution for body font detection
        font_sizes: List[float] = []
        for block in page_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        sz = span.get("size", 0.0)
                        if sz > 0:
                            font_sizes.append(round(sz, 1))

        # Basic page statistics
        p_len = len(page_text)
        p_words = len(page_text.split())
        total_text_length += p_len
        total_word_count += p_words
        total_images_count += len(parsed_images)
        total_drawings_count += len(parsed_drawings)

        pages.append({
            "page_number": page_num,
            "width": round(page_rect.width, 2),
            "height": round(page_rect.height, 2),
            "text": page_text,
            "text_length": p_len,
            "word_count": p_words,
            "blocks": page_dict.get("blocks", []),
            "images": parsed_images,
            "image_count": len(parsed_images),
            "drawings": parsed_drawings,
            "drawing_count": len(parsed_drawings),
            "font_sizes": font_sizes,
        })

    # Estimate dominant body font size across the document
    all_sizes: Dict[float, int] = {}
    for p in pages:
        for s in p["font_sizes"]:
            all_sizes[s] = all_sizes.get(s, 0) + 1
    dominant_font_size = (
        max(all_sizes.items(), key=lambda kv: kv[1])[0] if all_sizes else 10.0
    )

    metadata = {
        "title": doc.metadata.get("title") or "",
        "author": doc.metadata.get("author") or "",
        "subject": doc.metadata.get("subject") or "",
        "keywords": doc.metadata.get("keywords") or "",
        "creator": doc.metadata.get("creator") or "",
        "producer": doc.metadata.get("producer") or "",
        "creationDate": doc.metadata.get("creationDate") or "",
        "modDate": doc.metadata.get("modDate") or "",
    }

    doc_info = {
        "pdf_path": pdf_path,
        "page_count": len(doc),
        "dominant_font_size": dominant_font_size,
        "text_length": total_text_length,
        "word_count": total_word_count,
        "total_images": total_images_count,
        "total_drawings": total_drawings_count,
        "metadata": metadata,
        "pages": pages,
    }

    doc.close()
    return doc_info
