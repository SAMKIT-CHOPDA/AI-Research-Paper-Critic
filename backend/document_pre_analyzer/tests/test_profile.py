"""Integration tests for Document Profile generation and JSON serialization."""
import os
import json
import unittest
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.document_pre_analyzer.config import PreAnalyzerConfig

TEST_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class TestDocumentProfile(unittest.TestCase):
    def test_complete_profile_generation(self):
        self.assertTrue(os.path.exists(TEST_PDF))

        # Run fast profile generation
        cfg_fast = PreAnalyzerConfig(use_docling=False)
        profile = analyze_document(TEST_PDF, config=cfg_fast)

        # Check required top-level keys
        required_keys = [
            "filename", "page_count", "text_length", "word_count",
            "metadata", "sections", "figures", "tables", "equations",
            "references", "content_characteristics", "pages",
        ]
        for k in required_keys:
            self.assertIn(k, profile, f"Missing key '{k}' in profile")

        self.assertEqual(profile["page_count"], 11)
        self.assertGreater(profile["word_count"], 2000)
        self.assertGreaterEqual(profile["figures"]["count"], 2)
        self.assertGreaterEqual(profile["tables"]["count"], 3)
        self.assertGreaterEqual(profile["equations"]["count"], 3)
        self.assertTrue(profile["references"]["has_references"])
        self.assertEqual(profile["references"]["count"], 32)

        # Check content characteristics
        cc = profile["content_characteristics"]
        self.assertTrue(cc["has_methodology"])
        self.assertTrue(cc["has_experimental_results"])
        self.assertTrue(cc["has_visual_content"])
        self.assertTrue(cc["has_tables"])
        self.assertTrue(cc["has_mathematical_content"])
        self.assertTrue(cc["has_references"])

        # Candidate vs confirmed separation is exposed for every object class
        for key, confirmed_key in (
            ("figures", "confirmed_figures"),
            ("tables", "confirmed_tables"),
            ("equations", "confirmed_equations"),
        ):
            block = profile[key]
            for field in (
                "count", "items", confirmed_key, "candidate_count",
                "rejected_candidates", "rejected_candidate_count",
            ):
                self.assertIn(field, block, f"Missing '{field}' in profile['{key}']")
            self.assertEqual(block["items"], block[confirmed_key])
            self.assertGreaterEqual(block["candidate_count"], block["count"])

        self.assertEqual(profile["tables"]["count"], len(profile["tables"]["confirmed_tables"]))
        self.assertIn("rejected_table_mentions", profile["tables"])
        self.assertIn("rejected_figure_mentions", profile["figures"])
        self.assertIn("rejected_equation_mentions", profile["equations"])

        # No prose sentence may ever be confirmed as a table
        for table in profile["tables"]["confirmed_tables"]:
            self.assertTrue(table["caption"].startswith("Table"))
            self.assertNotIn("Training took", table["caption"])

        # Test JSON serialization
        serialized = json.dumps(profile, indent=2)
        self.assertGreater(len(serialized), 1000)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["filename"], "NIPS-2017-attention-is-all-you-need-Paper.pdf")


if __name__ == "__main__":
    unittest.main()

