"""
Unit tests for build_jev_input contract and validation.
"""

import json
import pytest

from backend.jev_router.input_builder import build_jev_input, JevInputValidationError


def make_valid_profile():
    """Helper returning a complete minimal valid Document Profile."""
    return {
        "filename": "minimal.pdf",
        "page_count": 5,
        "text_length": 8000,
        "word_count": 1500,
        "sections": [
            {"title": "1 Introduction", "page": 1, "level": 1},
            {"title": "2 Methodology", "page": 2, "level": 2},
        ],
        "figures": {"count": 1, "items": [{"caption": "Fig 1"}]},
        "tables": {"count": 2, "items": [{"caption": "Table 1"}]},
        "equations": {"count": 3, "items": [{"caption": "Eq 1"}]},
        "references": {"has_references": True, "count": 10},
        "content_characteristics": {
            "has_methodology": True,
            "has_experimental_results": False,
            "has_visual_content": True,
            "has_tables": True,
            "has_mathematical_content": True,
            "has_references": True,
            "document_nature": "born-digital",
        },
    }


class TestInputBuilder:
    def test_minimal_valid_profile(self):
        """Test with a minimal valid document profile."""
        profile = make_valid_profile()
        jev_input = build_jev_input(profile)

        assert jev_input["document"]["page_count"] == 5
        assert jev_input["document"]["word_count"] == 1500
        assert jev_input["document"]["document_nature"] == "born-digital"

        assert jev_input["structure"]["section_count"] == 2
        assert jev_input["structure"]["max_section_depth"] == 2
        assert jev_input["structure"]["has_methodology"] is True
        assert jev_input["structure"]["has_experimental_results"] is False

        assert jev_input["visual_content"]["figure_count"] == 1
        assert jev_input["visual_content"]["table_count"] == 2
        assert jev_input["visual_content"]["has_visual_content"] is True
    def test_attention_is_all_you_need_profile(self):
        """Test with Attention Is All You Need exact profile values."""
        profile = {
            "page_count": 11,
            "word_count": 4990,
            "sections": [{"level": 1 if i % 2 == 0 else 2} for i in range(23)],
            "figures": {"count": 2},
            "tables": {"count": 3},
            "equations": {"count": 5},
            "references": {"has_references": True, "count": 32},
            "content_characteristics": {
                "has_methodology": True,
                "has_experimental_results": True,
                "has_visual_content": True,
                "has_tables": True,
                "has_mathematical_content": True,
                "has_references": True,
                "document_nature": "born-digital",
            },
        }
        profile["sections"][5]["level"] = 3

        jev_input = build_jev_input(profile)

        assert jev_input["document"]["page_count"] == 11
        assert jev_input["document"]["word_count"] == 4990
        assert jev_input["structure"]["section_count"] == 23
        assert jev_input["structure"]["max_section_depth"] == 3
        assert jev_input["visual_content"]["figure_count"] == 2
        assert jev_input["visual_content"]["table_count"] == 3
        assert jev_input["mathematical_content"]["equation_count"] == 5
        assert jev_input["structure"]["has_references"] is True
        assert jev_input["research_characteristics"]["has_references"] is True
        assert jev_input["research_characteristics"]["has_experimental_results"] is True

    def test_text_heavy_paper(self):
        """Test with a text-heavy paper (0 figures, 0 tables, 0 equations)."""
        profile = {
            "page_count": 15,
            "word_count": 8000,
            "sections": [{"level": 1}, {"level": 1}],
            "figures": {"count": 0},
            "tables": {"count": 0},
            "equations": {"count": 0},
            "references": {"has_references": True, "count": 40},
            "content_characteristics": {
                "has_methodology": False,
                "has_experimental_results": False,
                "has_visual_content": False,
                "has_tables": False,
                "has_mathematical_content": False,
                "has_references": True,
                "document_nature": "born-digital",
            },
        }
        jev_input = build_jev_input(profile)

        assert jev_input["visual_content"]["figure_count"] == 0
        assert jev_input["visual_content"]["table_count"] == 0
        assert jev_input["visual_content"]["has_visual_content"] is False
        assert jev_input["mathematical_content"]["equation_count"] == 0
        assert jev_input["mathematical_content"]["has_mathematical_content"] is False

    def test_invalid_missing_fields(self):
        """Builder must raise a clear validation error identifying missing fields."""
        profile = make_valid_profile()
        del profile["page_count"]

        with pytest.raises(JevInputValidationError) as exc:
            build_jev_input(profile)
        assert "page_count" in str(exc.value)

        profile2 = make_valid_profile()
        del profile2["content_characteristics"]["document_nature"]
        with pytest.raises(JevInputValidationError) as exc2:
            build_jev_input(profile2)
        assert "document_nature" in str(exc2.value)

        profile3 = make_valid_profile()
        del profile3["figures"]["count"]
        with pytest.raises(JevInputValidationError) as exc3:
            build_jev_input(profile3)
        assert "figures.count" in str(exc3.value)

    def test_invalid_field_types(self):
        """Builder must raise clear validation error on invalid field types."""
        profile = make_valid_profile()
        profile["page_count"] = "eleven"
        with pytest.raises(JevInputValidationError):
            build_jev_input(profile)

        profile2 = make_valid_profile()
        profile2["content_characteristics"]["has_methodology"] = "yes"
        with pytest.raises(JevInputValidationError):
            build_jev_input(profile2)

    def test_no_forbidden_fields_in_output(self):
        """Ensure forbidden fields are excluded from compact routing state."""
        profile = make_valid_profile()
        profile["full_text"] = "This is the full paper text..."
        profile["blocks"] = [{"bbox": [0, 0, 100, 100]}]

        jev_input = build_jev_input(profile)
        assert "full_text" not in jev_input
        assert "blocks" not in jev_input
        expected_keys = {
            "document", "structure", "visual_content",
            "mathematical_content", "research_characteristics"
        }
        assert set(jev_input.keys()) == expected_keys


        assert jev_input["mathematical_content"]["equation_count"] == 3
        assert jev_input["mathematical_content"]["has_mathematical_content"] is True

        assert jev_input["research_characteristics"]["has_tables"] is True
