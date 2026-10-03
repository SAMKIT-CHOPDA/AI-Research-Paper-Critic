"""Tests for Table Analyzer."""
import os
import re
import unittest
from backend.document_pre_analyzer.pdf_parser import parse_pdf
from backend.document_pre_analyzer.table_analyzer import detect_tables

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")

# A caption must begin the block, so mid-sentence prose can never match.
CAPTION_AT_BLOCK_START = re.compile(r"^Table\s+\d+\s*[:.\-—]", re.IGNORECASE)


class TestTableAnalyzer(unittest.TestCase):
    def test_table_detection(self):
        parsed = parse_pdf(TEST_PDF)
        tables_data = detect_tables(parsed)

        self.assertGreaterEqual(tables_data["count"], 3, f"Expected at least 3 tables, got {tables_data['count']}")
        items = tables_data["items"]

        table_numbers = [t["table_number"] for t in items]
        self.assertIn("1", table_numbers, "Table 1 was not detected")
        self.assertIn("2", table_numbers, "Table 2 was not detected")
        self.assertIn("3", table_numbers, "Table 3 was not detected")

        for t in items:
            self.assertTrue(0.0 <= t["confidence"] <= 1.0)
            self.assertIn(t["page"], [6, 8, 9])
            self.assertIn("evidence", t)
            self.assertTrue(t["caption"].startswith("Table"))

    def test_confirmed_tables_are_backward_compatible(self):
        parsed = parse_pdf(TEST_PDF)
        tables_data = detect_tables(parsed)

        self.assertEqual(tables_data["count"], len(tables_data["confirmed_tables"]))
        self.assertEqual(tables_data["items"], tables_data["confirmed_tables"])
        for t in tables_data["confirmed_tables"]:
            self.assertEqual(t["status"], "confirmed")
            self.assertTrue(t["caption"].startswith("Table"))

    def test_every_confirmed_table_has_structural_evidence(self):
        parsed = parse_pdf(TEST_PDF)
        tables_data = detect_tables(parsed)

        for t in tables_data["confirmed_tables"]:
            evidence = t["evidence"]
            self.assertTrue(
                evidence["ruling_lines_spanning_caption"]
                or evidence["pymupdf_find_tables_matched"]
                or evidence["docling_table_matched"],
                f"Table {t['table_number']} was confirmed without structural evidence",
            )
            self.assertFalse(evidence["author_metadata_block"])
            self.assertGreater(t["rows"], 0)
            self.assertGreater(t["columns"], 0)

    def test_body_text_mention_is_not_a_table(self):
        """Regression: 'listed in the bottom line of Table 3. Training took 3.5 days'."""
        parsed = parse_pdf(TEST_PDF)
        tables_data = detect_tables(parsed)

        for t in tables_data["confirmed_tables"]:
            self.assertNotIn("Training took", t["caption"])
            self.assertRegex(
                t["caption"], CAPTION_AT_BLOCK_START,
                f"Caption for table {t['table_number']} does not start the block",
            )
            self.assertTrue(t["evidence"]["caption_at_block_start"])

        mentions = tables_data["rejected_table_mentions"]
        self.assertGreaterEqual(len(mentions), 1)
        self.assertTrue(
            any("Training took" in m["context"] for m in mentions),
            "In-text prose mention of 'Table 3. Training took ...' was not recorded as rejected",
        )
        for mention in mentions:
            self.assertEqual(mention["reason"], "in_text_reference_not_caption_at_block_start")
            self.assertNotRegex(mention["context"], CAPTION_AT_BLOCK_START)

    def test_candidates_expose_audit_trail(self):
        parsed = parse_pdf(TEST_PDF)
        tables_data = detect_tables(parsed)

        self.assertEqual(tables_data["candidate_count"], len(tables_data["table_candidates"]))
        self.assertEqual(
            tables_data["rejected_candidate_count"],
            len(tables_data["rejected_candidates"]),
        )
        self.assertEqual(
            tables_data["rejected_candidate_count"] + len(tables_data["confirmed_tables"]),
            tables_data["candidate_count"],
        )


if __name__ == "__main__":
    unittest.main()
