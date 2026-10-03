"""
Offline unit tests for the VisionAgent orchestrator.

Verifies:
- vision disabled -> ZERO model calls and no asset extraction at all
- vision enabled -> basic / medium / advanced map to the configured models
- a level without a configured model fails clearly (never a silent downgrade)
- multiple assets produce one independent result each
- partial failures are isolated and recorded, never silently dropped
- output is JSON-serializable and never contains credentials
- JEV enum-typed levels are consumed correctly

The vision model is always mocked; no network call is made.
"""

import json
import os

import pymupdf
import pytest

from backend.jev_router.config import ModelLevel
from backend.jev_router.routing_state import create_routing_state
from backend.vision_agent.agent import VisionAgent
from backend.vision_agent.config import VisionConfig, VisionConfigurationError
from backend.vision_agent.vision_client import BaseVisionClient

VALID_RESPONSE = json.dumps(
    {
        "observation": "A plotted curve is visible in the cropped region.",
        "interpretation": "The curve appears to decrease across the shown interval.",
        "key_elements": ["x axis", "y axis"],
        "reported_relationships": ["value decreases"],
        "supports_claims": ["the trend appears monotonic"],
        "uncertainties": ["labels are small in the crop"],
        "caption_consistency": {"status": "partially_consistent", "explanation": "partly matches"},
    }
)


def make_pdf(path: str, pages: int = 3) -> str:
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 60), f"Page {index + 1} body text", fontsize=10)
        page.draw_rect(pymupdf.Rect(60, 100, 300, 220), color=(0, 0, 0), width=1)
        page.insert_text((60, 240), "Figure 1: caption", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def make_profile() -> dict:
    """Two confirmed figures + three confirmed tables + five equations."""
    return {
        "filename": "synthetic.pdf",
        "sections": [{"title": "Results", "page": 2, "level": 1}],
        "figures": {
            "count": 2,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 1,
                    "caption": "Figure 1: caption one",
                    "caption_bbox": [60.0, 230.0, 300.0, 245.0],
                    "bbox": [60.0, 100.0, 300.0, 220.0],
                    "type": "vector_graphic",
                    "confidence": 0.8,
                },
                {
                    "id": "figure_2",
                    "figure_number": "2",
                    "page": 2,
                    "caption": "Figure 2: caption two",
                    "caption_bbox": [60.0, 230.0, 300.0, 245.0],
                    "bbox": [60.0, 100.0, 300.0, 220.0],
                    "type": "vector_graphic",
                    "confidence": 0.7,
                },
            ],
            "figure_candidates": [],
        },
        "tables": {
            "count": 3,
            "confirmed_tables": [
                {
                    "id": f"table_{index}_{index}_1",
                    "table_number": str(index),
                    "page": index,
                    "caption": f"Table {index}: caption",
                    "caption_bbox": [60.0, 230.0, 300.0, 245.0],
                    "bbox": [60.0, 100.0, 500.0, 220.0],
                    "rows": 3,
                    "columns": 2,
                    "structure_source": "ruling_lines",
                    "confidence": 0.6,
                }
                for index in (1, 2, 3)
            ],
            "table_candidates": [],
        },
        "equations": {
            "count": 5,
            "confirmed_equations": [
                {"id": f"eq_{i}", "page": i, "confidence": 0.4} for i in range(1, 6)
            ],
        },
    }


class RecordingVisionClient(BaseVisionClient):
    """Mock multimodal client: returns a fixed structured response."""

    def __init__(self, response: str = VALID_RESPONSE, fail_on_calls=()):
        self.response = response
        self.fail_on_calls = set(fail_on_calls)
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "recording"

    def analyze_image(self, image, prompt, model):
        self.calls.append({"image": str(image), "prompt": prompt, "model": model})
        if len(self.calls) in self.fail_on_calls:
            raise RuntimeError(f"synthetic provider failure # {len(self.calls)}")
        return self.response


def level_config(**models) -> VisionConfig:
    mapping = {"basic": None, "medium": None, "advanced": None}
    mapping.update(models)
    return VisionConfig(level_models=mapping, max_assets=0)


@pytest.fixture()
def pdf_path(tmp_path) -> str:
    return make_pdf(str(tmp_path / "paper.pdf"))


@pytest.fixture()
def agent_output_dir(tmp_path) -> str:
    target = tmp_path / "assets"
    target.mkdir()
    return str(target)


class TestVisionDisabled:
    def test_disabled_returns_structured_bypass_with_zero_model_calls(self, agent_output_dir):
        client = RecordingVisionClient()
        agent = VisionAgent(vision_client=client, config=level_config(advanced="vision-model-a"))
        result = agent.run(
            pdf_path="this_pdf_does_not_exist.pdf",  # proves no extraction is attempted
            document_profile=make_profile(),
            routing_state=create_routing_state(
                analysis_level="advanced", rag_enabled=False, vision_enabled=False
            ),
            output_dir=agent_output_dir,
        )
        assert result["agent"] == "vision"
        assert result["enabled"] is False
        assert result["results"] == []
        assert result["failures"] == []
        assert result["extraction_failures"] == []
        assert result["assets_found"] == 0
        assert result["model"] is None
        assert "disabled" in result["message"].lower()
        assert client.calls == []

    def test_disabled_still_reports_equation_context(self, agent_output_dir):
        agent = VisionAgent(vision_client=RecordingVisionClient(), config=level_config())
        result = agent.run(
            pdf_path="unused.pdf",
            document_profile=make_profile(),
            routing_state=create_routing_state(
                analysis_level="basic", rag_enabled=False, vision_enabled=False
            ),
            output_dir=agent_output_dir,
        )
        assert result["equations_available"] == 5  # context only, never analyzed

    def test_legacy_flat_routing_state_is_supported(self, agent_output_dir):
        agent = VisionAgent(vision_client=RecordingVisionClient(), config=level_config())
        result = agent.run(
            pdf_path="unused.pdf",
            document_profile=make_profile(),
            routing_state={"vision_enabled": False, "vision_level": None},
            output_dir=agent_output_dir,
        )
        assert result["enabled"] is False


class TestLevelModelMapping:
    @pytest.mark.parametrize("level", ["basic", "medium", "advanced"])
    def test_each_level_uses_its_configured_model(self, pdf_path, agent_output_dir, level):
        client = RecordingVisionClient()
        agent = VisionAgent(
            vision_client=client,
            config=level_config(basic="vision-basic", medium="vision-medium", advanced="vision-advanced"),
        )
        result = agent.run(
            pdf_path=pdf_path,
            document_profile=make_profile(),
            routing_state=create_routing_state(
                analysis_level="advanced", rag_enabled=False,
                vision_enabled=True, vision_level=level,
            ),
            output_dir=agent_output_dir,
        )
        assert result["level"] == level
        assert result["model"] == f"vision-{level}"
        assert {call["model"] for call in client.calls} == {f"vision-{level}"}

    def test_enum_typed_level_from_frozen_router_is_consumed(self, pdf_path, agent_output_dir):
        client = RecordingVisionClient()
        agent = VisionAgent(vision_client=client, config=level_config(advanced="vision-advanced"))
        routing_state = create_routing_state(
            analysis_level=ModelLevel.ADVANCED,
            rag_enabled=False,
            vision_enabled=True,
            vision_level=ModelLevel.ADVANCED,
        )
        result = agent.run(
            pdf_path=pdf_path,
            document_profile=make_profile(),
            routing_state=routing_state,
            output_dir=agent_output_dir,
        )
        assert result["level"] == "advanced"  # not "ModelLevel.ADVANCED"
        assert result["model"] == "vision-advanced"
        assert client.calls

    def test_missing_model_for_level_fails_clearly_without_model_calls(self, pdf_path, agent_output_dir):
        client = RecordingVisionClient()
        agent = VisionAgent(vision_client=client, config=level_config(basic="vision-basic"))
        with pytest.raises(VisionConfigurationError) as excinfo:
            agent.run(
                pdf_path=pdf_path,
                document_profile=make_profile(),
                routing_state=create_routing_state(
                    analysis_level="advanced", rag_enabled=False,
                    vision_enabled=True, vision_level="advanced",
                ),
                output_dir=agent_output_dir,
            )
        assert "VISION_MODEL_ADVANCED" in str(excinfo.value)
        assert client.calls == []

    def test_no_silent_downgrade_even_when_a_weaker_model_exists(self, pdf_path, agent_output_dir):
        agent = VisionAgent(
            vision_client=RecordingVisionClient(), config=level_config(basic="vision-basic")
        )
        with pytest.raises(VisionConfigurationError):
            agent.run(
                pdf_path=pdf_path,
                document_profile=make_profile(),
                routing_state=create_routing_state(
                    analysis_level="advanced", rag_enabled=False,
                    vision_enabled=True, vision_level="medium",
                ),
                output_dir=agent_output_dir,
            )

    def test_explicit_fallback_policy_is_honoured_when_enabled(self, pdf_path, agent_output_dir):
        config = level_config(basic="vision-basic")
        config.allow_level_fallback = True
        agent = VisionAgent(vision_client=RecordingVisionClient(), config=config)
        result = agent.run(
            pdf_path=pdf_path,
            document_profile=make_profile(),
            routing_state=create_routing_state(
                analysis_level="advanced", rag_enabled=False,
                vision_enabled=True, vision_level="advanced",
            ),
            output_dir=agent_output_dir,
        )
        assert result["model"] == "vision-basic"  # explicit project policy, not silent


def run_agent(pdf_path, agent_output_dir, client=None, config=None, profile=None, level="advanced"):
    client = client or RecordingVisionClient()
    config = config or level_config(advanced="vision-advanced")
    agent = VisionAgent(vision_client=client, config=config)
    result = agent.run(
        pdf_path=pdf_path,
        document_profile=profile if profile is not None else make_profile(),
        routing_state=create_routing_state(
            analysis_level="advanced", rag_enabled=False,
            vision_enabled=True, vision_level=level,
        ),
        output_dir=agent_output_dir,
    )
    return result, client


class TestMultipleAssets:
    def test_five_confirmed_assets_produce_five_independent_results(self, pdf_path, agent_output_dir):
        result, client = run_agent(pdf_path, agent_output_dir)
        assert result["assets_found"] == 5
        assert result["assets_analyzed"] == 5
        assert result["assets_failed"] == 0
        assert len(result["results"]) == 5
        assert len({item["asset_id"] for item in result["results"]}) == 5
        assert len(client.calls) == 5
        # Every asset keeps its own identity, page and type.
        by_id = {item["asset_id"]: item for item in result["results"]}
        assert by_id["figure_001"]["page_number"] == 1
        assert by_id["figure_002"]["page_number"] == 2
        assert by_id["table_003"]["asset_type"] == "table"
        assert by_id["table_003"]["page_number"] == 3

    def test_each_asset_gets_its_own_prompt(self, pdf_path, agent_output_dir):
        result, client = run_agent(pdf_path, agent_output_dir)
        prompts = [call["prompt"] for call in client.calls]
        assert len(set(prompts)) == 5
        assert any("What is the table comparing?" in prompt for prompt in prompts)
        assert any("What type of visual is this" in prompt for prompt in prompts)

    def test_metadata_is_preserved_in_results(self, pdf_path, agent_output_dir):
        result, _ = run_agent(pdf_path, agent_output_dir)
        figure = next(item for item in result["results"] if item["asset_id"] == "figure_001")
        assert figure["source_confidence"] == pytest.approx(0.8)
        assert figure["extraction_method"] == "bbox_crop"
        assert figure["caption"] == "Figure 1: caption one"
        assert figure["section"] is None  # the only section starts on page 2
        assert figure["image_path"].endswith("figure_001.png")
        on_second_page = next(
            item for item in result["results"] if item["asset_id"] == "figure_002"
        )
        assert on_second_page["section"] == "Results"

    def test_equations_are_reported_as_context_only(self, pdf_path, agent_output_dir):
        result, _ = run_agent(pdf_path, agent_output_dir)
        assert result["equations_available"] == 5
        assert all(item["asset_type"] in {"figure", "table"} for item in result["results"])


class TestFailureIsolation:
    def test_one_analysis_failure_does_not_stop_the_others(self, pdf_path, agent_output_dir):
        client = RecordingVisionClient(fail_on_calls={3})
        result, _ = run_agent(pdf_path, agent_output_dir, client=client)
        assert result["assets_found"] == 5
        assert result["assets_analyzed"] == 4
        assert result["assets_failed"] == 1
        assert len(result["failures"]) == 1
        failure = result["failures"][0]
        assert failure["stage"] == "analysis"
        assert failure["model"] == "vision-advanced"
        assert "synthetic provider failure" in failure["error"]
        assert failure["asset_id"] not in {item["asset_id"] for item in result["results"]}

    def test_extraction_failure_is_isolated_and_reported(self, pdf_path, agent_output_dir):
        profile = make_profile()
        profile["figures"]["confirmed_figures"][1]["page"] = 42  # page not in the PDF
        result, client = run_agent(pdf_path, agent_output_dir, profile=profile)
        assert len(result["extraction_failures"]) == 1
        assert result["extraction_failures"][0]["asset_id"] == "figure_002"
        assert result["extraction_failures"][0]["stage"] == "extraction"
        assert result["assets_found"] == 4
        assert len(client.calls) == 4

    def test_unreadable_pdf_returns_structured_failure(self, agent_output_dir):
        result, client = run_agent("missing_paper.pdf", agent_output_dir)
        assert result["enabled"] is True
        assert result["assets_found"] == 0
        assert len(result["extraction_failures"]) == 1
        assert "failed" in result["message"].lower()
        assert client.calls == []

    def test_empty_visual_inventory_returns_no_results_without_model_calls(
        self, pdf_path, agent_output_dir
    ):
        result, client = run_agent(
            pdf_path, agent_output_dir, profile={"figures": {}, "tables": {}}
        )
        assert result["assets_found"] == 0
        assert result["results"] == []
        assert client.calls == []


class TestOutputContract:
    def test_result_is_json_serializable(self, pdf_path, agent_output_dir):
        result, _ = run_agent(pdf_path, agent_output_dir)
        assert json.loads(json.dumps(result))["agent"] == "vision"

    def test_api_key_never_appears_in_the_output(self, pdf_path, agent_output_dir, monkeypatch):
        monkeypatch.setenv("VISION_API_KEY", "sk-super-secret-value")
        result, _ = run_agent(
            pdf_path, agent_output_dir, client=RecordingVisionClient(fail_on_calls={2})
        )
        dumped = json.dumps(result)
        assert "sk-super-secret-value" not in dumped

    def test_provider_error_containing_a_key_is_redacted(self, pdf_path, agent_output_dir, monkeypatch):
        monkeypatch.setenv("VISION_API_KEY", "sk-super-secret-value")
        client = RecordingVisionClient()

        def leaky(image, prompt, model):
            client.calls.append({"image": str(image), "prompt": prompt, "model": model})
            raise RuntimeError("auth failed for sk-super-secret-value")

        client.analyze_image = leaky
        result, _ = run_agent(pdf_path, agent_output_dir, client=client)
        dumped = json.dumps(result)
        assert "sk-super-secret-value" not in dumped
        assert "[REDACTED_API_KEY]" in dumped

    def test_elapsed_time_and_message_are_reported(self, pdf_path, agent_output_dir):
        result, _ = run_agent(pdf_path, agent_output_dir)
        assert isinstance(result["elapsed_time_ms"], float)
        assert "5" in result["message"]
        assert "vision-advanced" in result["message"]

    def test_configured_output_dir_is_used_when_run_passes_none(
        self, pdf_path, tmp_path
    ):
        configured = str(tmp_path / "agent-configured")
        config = level_config(advanced="vision-advanced")
        config.output_dir = configured
        result = VisionAgent(
            vision_client=RecordingVisionClient(), config=config
        ).run(
            pdf_path=pdf_path,
            document_profile=make_profile(),
            routing_state=create_routing_state(
                analysis_level="advanced", rag_enabled=False,
                vision_enabled=True, vision_level="advanced",
            ),
            output_dir=None,
        )
        assert result["assets_analyzed"] == 5
        for item in result["results"]:
            assert os.path.dirname(item["image_path"]) == configured
            assert os.path.exists(item["image_path"])