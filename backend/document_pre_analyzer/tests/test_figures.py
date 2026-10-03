"""Tests for Figure Analyzer."""
import os
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf
from backend.document_pre_analyzer.figure_analyzer import detect_figures

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")
MIN_VECTOR_STROKES = 3


class TestFigureAnalyzer(unittest.TestCase):
    def test_figure_detection(self):
        parsed = parse_pdf(TEST_PDF)
        figures_data = detect_figures(parsed)

        self.assertGreaterEqual(figures_data["count"], 2, f"Expected at least 2 figures, got {figures_data['count']}")
        items = figures_data["items"]

        fig_numbers = [f["figure_number"] for f in items]
        self.assertIn("1", fig_numbers, "Figure 1 was not detected")
        self.assertIn("2", fig_numbers, "Figure 2 was not detected")

        for f in items:
            self.assertTrue(0.0 <= f["confidence"] <= 1.0)
            self.assertIn(f["page"], [3, 4])
            self.assertIn("evidence", f)
            self.assertTrue("Figure" in f["caption"] or "Fig" in f["caption"])

    def test_confirmed_figures_are_backward_compatible(self):
        parsed = parse_pdf(TEST_PDF)
        figures_data = detect_figures(parsed)

        self.assertEqual(figures_data["count"], len(figures_data["confirmed_figures"]))
        self.assertEqual(figures_data["items"], figures_data["confirmed_figures"])
        for f in figures_data["confirmed_figures"]:
            self.assertEqual(f["status"], "confirmed")
            self.assertIsNotNone(f["type"])

    def test_every_confirmed_figure_has_visual_evidence(self):
        parsed = parse_pdf(TEST_PDF)
        figures_data = detect_figures(parsed)

        for f in figures_data["confirmed_figures"]:
            evidence = f["evidence"]
            self.assertTrue(
                evidence["raster_image_matched"]
                or evidence["docling_picture_matched"]
                or evidence["vector_strokes_in_region"] >= MIN_VECTOR_STROKES,
                f"Figure {f['figure_number']} was confirmed without visual evidence",
            )
            self.assertTrue(evidence["caption_at_block_start"])
            self.assertIsNotNone(evidence["visual_search_direction"])

    def test_body_text_mention_is_not_a_figure(self):
        parsed = parse_pdf(TEST_PDF)
        figures_data = detect_figures(parsed)

        confirmed_pages = {f["page"] for f in figures_data["confirmed_figures"]}
        self.assertEqual(confirmed_pages, {3, 4})
        for mention in figures_data["rejected_figure_mentions"]:
            self.assertEqual(mention["reason"], "in_text_reference_not_caption_at_block_start")
            self.assertNotRegex(mention["context"], r"^Fig")

    def test_candidates_expose_audit_trail(self):
        parsed = parse_pdf(TEST_PDF)
        figures_data = detect_figures(parsed)

        self.assertEqual(figures_data["candidate_count"], len(figures_data["figure_candidates"]))
        self.assertEqual(
            figures_data["rejected_candidate_count"] + len(figures_data["confirmed_figures"]),
            figures_data["candidate_count"],
        )


if __name__ == "__main__":
    unittest.main()
