"""
Unit tests for text_extractor module.
"""

import os
import pytest
import pymupdf

from backend.rag_agent.text_extractor import extract_paper_text, TextExtractionError

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestTextExtractor:
    def test_extract_valid_benchmark_pdf(self):
        """Extract text from the valid benchmark Attention Is All You Need paper."""
        assert os.path.exists(BENCHMARK_PDF)
        pages = extract_paper_text(BENCHMARK_PDF)

        assert len(pages) == 11
        for i, page in enumerate(pages):
            assert page.page_number == i + 1
            assert page.word_count > 0
            assert page.char_count > 0
            assert len(page.text) > 50

        # Check title text present on page 1
        assert "Attention Is All You Need" in pages[0].text or "Attention is All you Need" in pages[0].text

    def test_missing_file_raises_error(self):
        """Missing file raises clear TextExtractionError."""
        with pytest.raises(TextExtractionError) as exc:
            extract_paper_text("non_existent_file.pdf")
        assert "not found" in str(exc.value)

    def test_empty_file_raises_error(self, tmp_path):
        """0-byte file raises TextExtractionError."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with pytest.raises(TextExtractionError) as exc:
            extract_paper_text(str(empty_pdf))
        assert "empty" in str(exc.value)

    def test_corrupt_file_raises_error(self, tmp_path):
        """Corrupted PDF raises TextExtractionError."""
        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"%PDF-1.4\ncorrupted content that cannot parse")
        with pytest.raises(TextExtractionError) as exc:
            extract_paper_text(str(corrupt_pdf))
        assert "Failed to open PDF document" in str(exc.value)

    def test_empty_page_handling(self, tmp_path):
        """PDF with a page having no text returns PageText with word_count 0."""
        pdf_path = str(tmp_path / "blank_page.pdf")
        doc = pymupdf.open()
        # Add a page with no text
        doc.new_page(width=595, height=842)
        doc.save(pdf_path)
        doc.close()

        pages = extract_paper_text(pdf_path)
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].text == ""
        assert pages[0].word_count == 0
