"""Tests for Section Analyzer."""
import os
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf
from backend.document_pre_analyzer.section_analyzer import detect_sections

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestSectionAnalyzer(unittest.TestCase):
    def test_section_detection(self):
        parsed = parse_pdf(TEST_PDF)
        sections = detect_sections(parsed)

        self.assertGreaterEqual(len(sections), 5, f"Expected at least 5 sections, got {len(sections)}")

        norm_titles = [s["normalized_title"] for s in sections]
        self.assertTrue(any("abstract" in t for t in norm_titles), "Abstract section not detected")
        self.assertTrue(any("introduction" in t for t in norm_titles), "Introduction section not detected")
        self.assertTrue(any("background" in t for t in norm_titles), "Background section not detected")
        self.assertTrue(any("conclusion" in t for t in norm_titles), "Conclusion section not detected")
        self.assertTrue(any("references" in t for t in norm_titles), "References section not detected")

        for s in sections:
            self.assertTrue(0.0 <= s["confidence"] <= 1.0)
            self.assertGreaterEqual(s["page"], 1)
            self.assertIn("evidence", s)

    def test_subsections_and_hierarchy_are_detected(self):
        parsed = parse_pdf(TEST_PDF)
        sections = detect_sections(parsed)

        numbers = {s["number"] for s in sections}
        self.assertIn("3.1", numbers, "Subsection 3.1 was not detected")
        self.assertIn("3.2.1", numbers, "Sub-subsection 3.2.1 was not detected")

        levels = {s["number"]: s["level"] for s in sections if s["number"]}
        self.assertEqual(levels.get("3"), 1)
        self.assertEqual(levels.get("3.1"), 2)
        self.assertEqual(levels.get("3.2.1"), 3)

        subsections = [s for s in sections if s["evidence"]["is_subsection_numbered"]]
        self.assertGreaterEqual(len(subsections), 5)

    def test_spacing_and_position_evidence_is_recorded(self):
        parsed = parse_pdf(TEST_PDF)
        sections = detect_sections(parsed)

        for s in sections:
            evidence = s["evidence"]
            self.assertIn("has_vertical_spacing", evidence)
            self.assertIn("gap_above_pts", evidence)
            self.assertIn("gap_below_pts", evidence)
            self.assertIn("page_position_ratio", evidence)
            self.assertGreaterEqual(evidence["page_position_ratio"], 0.0)
            self.assertLessEqual(evidence["page_position_ratio"], 1.0)

        # Real headings are isolated by surrounding whitespace.
        headings_with_spacing = [s for s in sections if s["evidence"]["has_vertical_spacing"]]
        self.assertGreaterEqual(len(headings_with_spacing), len(sections) - 1)

    def test_table_data_rows_are_not_headings(self):
        """Regression: the '1 512 512 5.29 24.9 ...' row inside Table 3 on page 9."""
        parsed = parse_pdf(TEST_PDF)
        sections = detect_sections(parsed)

        page_nine_titles = [s["title"] for s in sections if s["page"] == 9]
        for title in page_nine_titles:
            self.assertFalse(
                title.replace(" ", "").replace(".", "").isdigit(),
                f"Numeric table row detected as a heading: {title}",
            )


if __name__ == "__main__":
    unittest.main()
