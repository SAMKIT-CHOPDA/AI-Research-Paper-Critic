"""
Offline unit tests for vision prompting and structured-response parsing.

Verifies:
- figure and table prompts are focused, contextual and bounded
- table prompts carry the numeric-authority rule (machine-readable values win)
- structured responses are parsed into the schema
- uncertainty is preserved and missing fields become explicit gaps
- confirmatory language ("proves") is flagged as an unsupported claim
- malformed responses fail loudly instead of producing an empty analysis
"""

import json

import pytest

from backend.vision_agent.analyzer import (
    AnalysisError,
    NOTE_NO_UNCERTAINTY,
    analyze_asset,
    build_figure_prompt,
    build_prompt,
    build_table_prompt,
    extract_json_payload,
    flag_overclaiming,
    parse_caption_consistency,
    parse_visual_analysis,
)
from backend.vision_agent.schemas import VisualAsset
from backend.vision_agent.vision_client import BaseVisionClient


def make_asset(asset_type="figure", **overrides) -> VisualAsset:
    payload = {
        "asset_id": "figure_001" if asset_type == "figure" else "table_001",
        "asset_type": asset_type,
        "page_number": 7,
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "image_path": "extracted/figure_001.png",
        "source_confidence": 0.7,
        "source_id": "figure_1",
        "asset_subtype": "vector_graphic",
        "label": "1",
        "caption": "Figure 1: Validation loss across training epochs.",
        "section": "Experiments",
        "context_text": "We report validation loss for the base model.",
        "extraction_method": "bbox_crop",
    }
    payload.update(overrides)
    return VisualAsset(**payload)


def valid_payload(**overrides) -> str:
    payload = {
        "observation": "Two curves are plotted against training epochs on the x axis.",
        "interpretation": "The validation curve appears to decrease during the shown interval.",
        "key_elements": ["x axis: epochs", "y axis: loss", "legend: training, validation"],
        "reported_relationships": ["loss decreases as epochs increase"],
        "supports_claims": ["the model appears to converge within the shown epochs"],
        "uncertainties": ["no error bars are visible"],
        "caption_consistency": {"status": "consistent", "explanation": "matches the caption"},
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeVisionClient(BaseVisionClient):
    """Scripted offline client that records prompts."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else valid_payload()
        self.error = error
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def analyze_image(self, image, prompt, model):
        self.calls.append({"image": image, "prompt": prompt, "model": model})
        if self.error is not None:
            raise self.error
        return self.response


class TestPromptBuilding:
    def test_figure_prompt_contains_context_and_focus_areas(self):
        prompt = build_figure_prompt(make_asset())
        assert "Figure 1: Validation loss across training epochs." in prompt
        assert "- page: 7" in prompt
        assert "- section: Experiments" in prompt
        assert "- surrounding text: We report validation loss for the base model." in prompt
        assert "What type of visual is this" in prompt
        assert "trends or relationships" in prompt
        assert "ambiguities or limitations" in prompt
        assert "OBSERVATION" in prompt and "INTERPRETATION" in prompt and "UNCERTAINTY" in prompt
        assert "do not critique the paper" in prompt

    def test_table_prompt_contains_table_focus_areas(self):
        prompt = build_table_prompt(make_asset("table"))
        assert "What is the table comparing?" in prompt
        assert "What are the rows and columns?" in prompt
        assert "Which values appear strongest and weakest" in prompt

    def test_absent_context_fields_are_omitted(self):
        prompt = build_figure_prompt(make_asset(caption=None, section=None, context_text=None))
        assert "- caption:" not in prompt
        assert "- section:" not in prompt
        assert "- surrounding text:" not in prompt

    def test_prompt_dispatcher_selects_by_asset_type(self):
        assert "What type of visual is this" in build_prompt(make_asset("figure"))
        assert "What is the table comparing?" in build_prompt(make_asset("table"))

    def test_machine_readable_table_values_are_authoritative(self):
        asset = make_asset(
            "table",
            structured_table={
                "source": "pymupdf_find_tables",
                "headers": ["Model", "BLEU"],
                "rows": [["Transformer (base)", "27.3"]],
                "authoritative_for_exact_values": True,
            },
        )
        prompt = build_table_prompt(asset)
        assert "27.3" in prompt
        assert "authoritative for exact numbers" in prompt
        assert "Do not invent, round or correct any number" in prompt

    def test_profile_only_table_metadata_forbids_exact_numbers(self):
        asset = make_asset(
            "table",
            structured_table={
                "row_count": 4,
                "column_count": 3,
                "structure_source": "ruling_lines",
                "source": "document_profile",
                "authoritative_for_exact_values": False,
            },
        )
        prompt = build_table_prompt(asset)
        assert "reported rows: 4" in prompt
        assert "Do not assert exact numbers from the image" in prompt


class TestStructuredResponseParsing:
    def test_valid_response_is_parsed_into_the_schema(self):
        result = parse_visual_analysis(valid_payload(), make_asset(), "vision-model-1")
        assert result.asset_id == "figure_001"
        assert result.asset_type == "figure"
        assert result.page_number == 7
        assert result.section == "Experiments"
        assert "Two curves" in result.observation
        assert "appears to decrease" in result.interpretation
        assert len(result.key_elements) == 3
        assert result.reported_relationships == ["loss decreases as epochs increase"]
        assert result.caption_consistency.status == "consistent"
        assert result.caption_consistency.explanation == "matches the caption"
        assert result.model == "vision-model-1"
        assert result.image_path == "extracted/figure_001.png"
        assert result.numeric_authority == "vision_interpretation_only"

    def test_fenced_json_is_accepted(self):
        raw = "Here is the analysis:\n```json\n" + valid_payload() + "\n```\n"
        result = parse_visual_analysis(raw, make_asset(), "vision-model-1")
        assert result.observation.startswith("Two curves")

    def test_json_with_preamble_is_accepted(self):
        raw = "Sure! " + valid_payload() + " I hope that helps."
        assert parse_visual_analysis(raw, make_asset()).observation

    def test_uncertainties_are_preserved(self):
        result = parse_visual_analysis(valid_payload(), make_asset())
        assert "no error bars are visible" in result.uncertainties

    def test_missing_uncertainty_becomes_an_explicit_gap(self):
        raw = valid_payload(uncertainties=[])
        result = parse_visual_analysis(raw, make_asset())
        assert result.uncertainties == [NOTE_NO_UNCERTAINTY]

    def test_missing_optional_fields_default_safely(self):
        raw = json.dumps({"observation": "A bar chart with four bars."})
        result = parse_visual_analysis(raw, make_asset())
        assert result.key_elements == []
        assert result.reported_relationships == []
        assert result.supports_claims == []
        assert result.uncertainties == [NOTE_NO_UNCERTAINTY]
        assert result.caption_consistency.status == "unknown"
        assert "did not report caption consistency" in result.caption_consistency.explanation

    def test_string_instead_of_list_is_coerced(self):
        raw = valid_payload(key_elements="x axis: epochs")
        result = parse_visual_analysis(raw, make_asset())
        assert result.key_elements == ["x axis: epochs"]

    def test_caption_status_variants_are_normalized(self):
        assert parse_caption_consistency("Partially Consistent").status == "partially_consistent"
        assert parse_caption_consistency({"status": "INCONSISTENT"}).status == "inconsistent"
        assert parse_caption_consistency({"status": "nonsense"}).status == "unknown"
        assert parse_caption_consistency(None).status == "unknown"

    def test_table_numeric_authority_is_recorded(self):
        asset = make_asset(
            "table",
            structured_table={
                "source": "pymupdf_find_tables",
                "rows": [["a", "1"]],
                "authoritative_for_exact_values": True,
            },
        )
        result = parse_visual_analysis(valid_payload(), asset, "vision-model-1")
        assert result.numeric_authority == "document_profile_machine_readable"
        assert result.structured_evidence["source"] == "pymupdf_find_tables"

    def test_figures_have_no_structured_evidence(self):
        result = parse_visual_analysis(valid_payload(), make_asset("figure"))
        assert result.structured_evidence is None


class TestOverclaimHandling:
    def test_flag_overclaiming_detects_confirmatory_phrases(self):
        assert flag_overclaiming("This figure proves the model is better.") == ["proves"]
        assert flag_overclaiming("Nothing objectionable here.") == []

    def test_overclaiming_becomes_an_explicit_uncertainty(self):
        raw = valid_payload(
            interpretation="The figure proves that the Transformer generalizes better.",
            uncertainties=[],
        )
        result = parse_visual_analysis(raw, make_asset())
        assert any("Confirmatory language detected" in item for item in result.uncertainties)
        assert any("proves" in item for item in result.uncertainties)

    def test_overclaiming_in_supports_claims_is_also_flagged(self):
        raw = valid_payload(supports_claims=["This table guarantees state-of-the-art results."])
        result = parse_visual_analysis(raw, make_asset("table"))
        assert any("guarantees" in item for item in result.uncertainties)


class TestMalformedResponses:
    def test_non_json_response_raises(self):
        with pytest.raises(AnalysisError):
            parse_visual_analysis("I could not analyse the image.", make_asset())

    def test_empty_response_raises(self):
        with pytest.raises(AnalysisError):
            parse_visual_analysis("   ", make_asset())

    def test_json_without_content_raises(self):
        with pytest.raises(AnalysisError):
            parse_visual_analysis(json.dumps({"key_elements": ["x axis"]}), make_asset())

    def test_json_array_is_rejected(self):
        with pytest.raises(AnalysisError):
            extract_json_payload("[1, 2, 3]")


class TestAnalyzeAsset:
    def test_client_receives_the_selected_prompt_and_model(self):
        client = FakeVisionClient()
        asset = make_asset()
        result = analyze_asset(asset, client, "vision-model-1")
        assert len(client.calls) == 1
        assert client.calls[0]["image"] == asset.image_path
        assert client.calls[0]["model"] == "vision-model-1"
        assert "What type of visual is this" in client.calls[0]["prompt"]
        assert result.model == "vision-model-1"

    def test_explicit_prompt_override_is_used(self):
        client = FakeVisionClient()
        analyze_asset(make_asset(), client, "vision-model-1", prompt="custom prompt")
        assert client.calls[0]["prompt"] == "custom prompt"

    def test_client_failure_propagates_for_agent_level_isolation(self):
        client = FakeVisionClient(error=RuntimeError("provider exploded"))
        with pytest.raises(RuntimeError):
            analyze_asset(make_asset(), client, "vision-model-1")

    def test_asset_without_image_path_fails_cleanly(self):
        asset = make_asset()
        asset.image_path = ""
        with pytest.raises(AnalysisError):
            analyze_asset(asset, FakeVisionClient(), "vision-model-1")