"""Tests for Equation Analyzer."""
import os
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf
from backend.document_pre_analyzer.equation_analyzer import detect_equations

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestEquationAnalyzer(unittest.TestCase):
    def test_equation_detection(self):
        parsed = parse_pdf(TEST_PDF)
        eq_data = detect_equations(parsed)

        self.assertGreaterEqual(eq_data["count"], 3, f"Expected at least 3 equations, got {eq_data['count']}")
        items = eq_data["items"]

        eq_nums = [e.get("equation_number") for e in items if e.get("equation_number")]
        self.assertIn("1", eq_nums, "Equation (1) was not detected")
        self.assertIn("2", eq_nums, "Equation (2) was not detected")
        self.assertIn("3", eq_nums, "Equation (3) was not detected")

        for e in items:
            self.assertTrue(0.0 <= e["confidence"] <= 1.0)
            self.assertIn("evidence", e)

    def test_confirmed_equations_are_backward_compatible(self):
        parsed = parse_pdf(TEST_PDF)
        eq_data = detect_equations(parsed)

        self.assertEqual(eq_data["count"], len(eq_data["confirmed_equations"]))
        self.assertEqual(eq_data["items"], eq_data["confirmed_equations"])

    def test_every_confirmed_equation_has_math_evidence(self):
        parsed = parse_pdf(TEST_PDF)
        eq_data = detect_equations(parsed)

        for e in eq_data["confirmed_equations"]:
            self.assertEqual(e["status"], "confirmed")
            evidence = e["evidence"]
            self.assertTrue(
                evidence["math_fonts_present"] or evidence["docling_formula_detected"],
                f"Equation {e['id']} was confirmed without math typography or Docling recognition",
            )
            self.assertIsNotNone(e["representation"])

    def test_candidates_expose_audit_trail(self):
        parsed = parse_pdf(TEST_PDF)
        eq_data = detect_equations(parsed)

        self.assertEqual(eq_data["candidate_count"], len(eq_data["equation_candidates"]))
        self.assertEqual(
            eq_data["rejected_candidate_count"] + len(eq_data["confirmed_equations"]),
            eq_data["candidate_count"],
        )
        self.assertEqual(
            eq_data["rejected_equation_mention_count"],
            len(eq_data["rejected_equation_mentions"]),
        )


if __name__ == "__main__":
    unittest.main()
