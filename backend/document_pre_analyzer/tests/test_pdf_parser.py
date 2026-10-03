"""Tests for PDF parser module."""
import os
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestPdfParser(unittest.TestCase):
    def test_parse_pdf_structure(self):
        self.assertTrue(os.path.exists(TEST_PDF), f"Test PDF missing at {TEST_PDF}")
        result = parse_pdf(TEST_PDF)

        self.assertEqual(result["page_count"], 11)
        self.assertGreater(result["text_length"], 10000)
        self.assertGreater(result["word_count"], 2000)
        self.assertEqual(len(result["pages"]), 11)
        self.assertGreater(result["dominant_font_size"], 8.0)

        meta = result["metadata"]
        self.assertIn("title", meta)
        self.assertIn("creator", meta)

        p1 = result["pages"][0]
        self.assertEqual(p1["page_number"], 1)
        self.assertGreater(p1["width"], 0)
        self.assertGreater(p1["height"], 0)
        self.assertGreater(len(p1["blocks"]), 0)
        self.assertIsInstance(p1["images"], list)
        self.assertIsInstance(p1["drawings"], list)


if __name__ == "__main__":
    unittest.main()

