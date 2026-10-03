"""
OCR Analyzer for scanned and image-heavy PDF pages.

Classifies documents into:
- 'born-digital'
- 'scanned'
- 'mixed'

Invokes PyMuPDF OCR (powered by Tesseract engine in PyMuPDF) only when text
is absent or insufficient on a page. Avoids running OCR unnecessarily on normal digital PDFs.
"""

from typing import Dict, Any, List
import pymupdf


def assess_document_digital_nature(parsed_pdf_info: Dict[str, Any], min_words_per_page: int = 20) -> Dict[str, Any]:
    """
    Evaluate whether the document is born-digital, scanned, or mixed.

    A page is considered 'sparse' or scanned if it has very few words (< min_words_per_page)
    while having images or drawings.
    """
    total_pages = parsed_pdf_info["page_count"]
    pages_with_sufficient_text = 0
    pages_scanned_or_empty = []

    for page in parsed_pdf_info["pages"]:
        p_num = page["page_number"]
        words = page["word_count"]
        imgs = page["image_count"]

        if words >= min_words_per_page:
            pages_with_sufficient_text += 1
        else:
            pages_scanned_or_empty.append(p_num)

    if total_pages == 0:
        nature = "unknown"
    elif pages_with_sufficient_text == total_pages:
        nature = "born-digital"
    elif pages_with_sufficient_text == 0:
        nature = "scanned"
    else:
        nature = "mixed"

    return {
        "nature": nature,
        "total_pages": total_pages,
        "digital_pages_count": pages_with_sufficient_text,
        "scanned_or_empty_pages": pages_scanned_or_empty,
        "requires_ocr": nature in ["scanned", "mixed"],
    }


def ocr_page_if_needed(doc: pymupdf.Document, page_number: int, min_words: int = 20) -> Dict[str, Any]:
    """
    Perform OCR on a single page if text content is insufficient.

    Returns dict with OCR text, flag whether OCR was triggered, and confidence estimate.
    """
    page_idx = page_number - 1
    page = doc[page_idx]

    current_text = page.get_text()
    if len(current_text.split()) >= min_words:
        return {
            "page_number": page_number,
            "ocr_triggered": False,
            "text": current_text,
            "confidence": 1.0,
            "method": "digital_text",
        }

    # Text is insufficient, invoke PyMuPDF OCR
    try:
        textpage_ocr = page.get_textpage_ocr(dpi=150, full=True)
        ocr_text = textpage_ocr.extractText()
        word_count = len(ocr_text.split())

        return {
            "page_number": page_number,
            "ocr_triggered": True,
            "text": ocr_text,
            "word_count": word_count,
            "confidence": 0.85 if word_count > 10 else 0.40,
            "method": "pymupdf_ocr",
        }
    except Exception as e:
        return {
            "page_number": page_number,
            "ocr_triggered": True,
            "error": str(e),
            "text": current_text,
            "confidence": 0.20,
            "method": "ocr_failed",
        }
