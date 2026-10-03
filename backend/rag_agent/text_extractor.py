"""
Text extractor for PDF research papers.

Extracts page-aware text preserving:
- 1-indexed page number
- text content
- word and character counts

Gracefully handles:
- Missing file
- Invalid / corrupted PDF
- Empty PDF
- Pages with zero text
"""

import os
from typing import List
import pymupdf

from backend.rag_agent.schemas import PageText


class TextExtractionError(Exception):
    """Raised when PDF textual extraction fails."""
    pass


def extract_paper_text(pdf_path: str) -> List[PageText]:
    """
    Extract structured, page-aware text from a PDF file.

    Parameters:
    - pdf_path: Absolute or relative path to the PDF document

    Returns:
    - List of PageText objects, one per page (1-indexed)

    Raises:
    - TextExtractionError if file does not exist, is empty, or is unreadable
    """
    if not pdf_path or not os.path.exists(pdf_path):
        raise TextExtractionError(f"PDF file not found at path: {pdf_path}")

    file_size = os.path.getsize(pdf_path)
    if file_size == 0:
        raise TextExtractionError(f"PDF file is empty (0 bytes): {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise TextExtractionError(f"Failed to open PDF document: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise TextExtractionError("PDF document contains 0 pages")

        pages: List[PageText] = []
        for pno in range(len(doc)):
            page = doc[pno]
            raw_text = page.get_text() or ""
            # Clean and normalize line endings
            cleaned_text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
            words = cleaned_text.split() if cleaned_text else []
            pages.append(
                PageText(
                    page_number=pno + 1,
                    text=cleaned_text,
                    word_count=len(words),
                    char_count=len(cleaned_text),
                )
            )
        return pages
    finally:
        doc.close()
