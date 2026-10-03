"""Tests for Reference Analyzer."""
import os
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf
from backend.document_pre_analyzer.reference_analyzer import detect_references

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestReferenceAnalyzer(unittest.TestCase):
    def test_reference_detection(self):
        parsed = parse_pdf(TEST_PDF)
        refs_data = detect_references(parsed)

        self.assertTrue(refs_data["has_references"])
        self.assertEqual(refs_data["count"], 32, f"Expected exactly 32 references, got {refs_data['count']}")
        self.assertEqual(refs_data["start_page"], 10)
        self.assertTrue(refs_data["is_sequential"])
        self.assertGreaterEqual(refs_data["confidence"], 0.90)
        self.assertEqual(len(refs_data["items"]), 32)


if __name__ == "__main__":
    unittest.main()

