"""
Offline end-to-end integration tests for the Vision Agent pipeline.

Two independent paths are exercised with a mocked multimodal model:

1. Synthetic frozen Document Profile (2 figures + 3 tables + 5 equations)
   -> 5 independent analysis results, equations never analyzed.
2. The real benchmark paper, where the visual inventory is produced by the frozen
   Document Pre-Analyzer itself. This proves the Vision Agent consumes the
   existing confirmed inventory instead of running a second figure/table detector,
   and that the Pre-Analyzer's bounding boxes are the ones actually cropped.

No network access and no vision API key are required.
"""

import copy
import json
import os

import pytest

from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.routing_state import create_routing_state
from backend.vision_agent.agent import VisionAgent
from backend.vision_agent.config import VisionConfig
from backend.vision_agent.vision_client import BaseVisionClient

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")

MODEL_RESPONSE = json.dumps(
    {
        "observation": "A visual element with labelled regions is visible.",
        "interpretation": "It appears to illustrate the reported comparison.",
        "key_elements": ["labels", "legend"],
        "reported_relationships": ["one series appears larger than the other"],
        "supports_claims": ["the reported comparison appears supported visually"],
        "uncertainties": ["small text in the crop limits precision"],
        "caption_consistency": {"status": "consistent", "explanation": "caption matches"},
    }
)


class MockVisionClient(BaseVisionClient):
    """Deterministic offline model standing in for a multimodal provider."""

    def __init__(self):
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def analyze_image(self, image, prompt, model):
        self.calls.append({"image": str(image), "prompt": prompt, "model": model})
        return MODEL_RESPONSE


def frozen_style_profile() -> dict:
    """A frozen-Document-Pre-Analyzer-shaped profile: 2 figures, 3 tables, 5 equations."""
    return {
        "filename": "synthetic_frozen.pdf",
        "page_count": 4,
        "sections": [
            {"title": "3 Model Architecture", "page": 1, "level": 1},
            {"title": "4 Experiments", "page": 3, "level": 1},
        ],
        "figures": {
            "count": 2,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 1,
                    "caption": "Figure 1: Frozen inventory figure one.",
                    "caption_bbox": [60.0, 230.0, 300.0, 245.0],
                    "bbox": [60.0, 100.0, 300.0, 220.0],
                    "type": "vector_graphic",
                    "status": "confirmed",
                    "confidence": 0.9,
                },
                {
                    "id": "figure_2",
                    "figure_number": "2",
                    "page": 3,
                    "caption": "Figure 2: Frozen inventory figure two.",
                    "caption_bbox": [60.0, 230.0, 300.0, 245.0],
                    "bbox": [60.0, 100.0, 300.0, 220.0],
                    "type": "raster_image",
                    "status": "confirmed",
                    "confidence": 0.65,
                },
            ],
            "figure_candidates": [],
        },
        "tables": {
            "count": 3,
            "confirmed_tables": [
                {
                    "id": f"table_{index}_{index}_3",
                    "table_number": str(index),
                    "page": index,
                    "caption": f"Table {index}: Frozen inventory table {index}.",
                    "caption_bbox": [60.0, 500.0, 300.0, 515.0],
                    "bbox": [60.0, 330.0, 500.0, 470.0],
                    "rows": 4,
                    "columns": 3,
                    "status": "confirmed",
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
                {"id": f"eq_{i}", "page": i, "status": "confirmed", "confidence": 0.4}
                for i in range(1, 6)
            ],
        },
    }


@pytest.fixture(scope="module")
def frozen_profile() -> dict:
    return frozen_style_profile()


@pytest.fixture(scope="module")
def benchmark_profile():
    """Real Document Profile produced by the frozen Document Pre-Analyzer."""
    if not os.path.exists(BENCHMARK_PDF):
        pytest.skip("Benchmark PDF is not available")
    return analyze_document(BENCHMARK_PDF)


@pytest.fixture()
def synthetic_pdf(tmp_path) -> str:
    """A 4-page PDF with drawable figure and table regions matching the profile."""
    import pymupdf

    path = str(tmp_path / "synthetic_frozen.pdf")
    doc = pymupdf.open()
    for index in range(4):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 60), f"Section body text on page {index + 1}", fontsize=10)
        page.draw_rect(pymupdf.Rect(60, 100, 300, 220), color=(0, 0, 0), width=1)
        page.insert_text((60, 240), "Figure 1: caption text", fontsize=8)
        for offset in (0, 35, 70, 105, 140):
            y = 330 + offset
            page.draw_line(pymupdf.Point(60, y), pymupdf.Point(500, y))
        page.insert_text((60, 490), "Table 1: caption text", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def run_vision_agent(pdf_path, profile, output_dir, level="advanced"):
    client = MockVisionClient()
    config = VisionConfig(
        level_models={"basic": "mock-basic", "medium": "mock-medium", "advanced": "mock-advanced"},
        max_assets=0,
    )
    agent = VisionAgent(vision_client=client, config=config)
    result = agent.run(
        pdf_path=pdf_path,
        document_profile=profile,
        routing_state=create_routing_state(
            analysis_level="advanced", rag_enabled=False,
            vision_enabled=True, vision_level=level,
        ),
        output_dir=output_dir,
    )
    return result, client


class TestSyntheticFrozenProfileIntegration:
    def test_five_visual_assets_produce_five_independent_analyses(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        result, client = run_vision_agent(synthetic_pdf, frozen_profile, str(tmp_path / "assets"))

        assert result["agent"] == "vision"
        assert result["enabled"] is True
        assert result["level"] == "advanced"
        assert result["model"] == "mock-advanced"

        # 2 figures + 3 tables = 5 visual assets, each analyzed separately.
        assert result["assets_found"] == 5
        assert result["assets_analyzed"] == 5
        assert result["assets_failed"] == 0
        assert len(result["results"]) == 5
        assert len(client.calls) == 5

        asset_types = [item["asset_type"] for item in result["results"]]
        assert asset_types.count("figure") == 2
        assert asset_types.count("table") == 3
        assert len({item["asset_id"] for item in result["results"]}) == 5
        assert sorted(item["asset_id"] for item in result["results"]) == [
            "figure_001", "figure_002", "table_001", "table_002", "table_003",
        ]

    def test_equations_are_context_only_and_never_analyzed(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        result, _ = run_vision_agent(synthetic_pdf, frozen_profile, str(tmp_path / "assets"))
        assert result["equations_available"] == 5
        assert all(
            item["asset_id"].startswith(("figure_", "table_"))
            for item in result["results"]
        )

    def test_results_preserve_uncertainty_and_claim_separation(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        result, _ = run_vision_agent(synthetic_pdf, frozen_profile, str(tmp_path / "assets"))
        for item in result["results"]:
            assert item["observation"]
            assert item["interpretation"]
            assert item["uncertainties"]  # never empty
            assert item["caption_consistency"]["status"] == "consistent"
            assert "proves" not in json.dumps(item).lower()

    def test_table_numbers_are_flagged_as_non_authoritative_without_machine_read_cells(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        result, _ = run_vision_agent(synthetic_pdf, frozen_profile, str(tmp_path / "assets"))
        for item in result["results"]:
            if item["asset_type"] != "table":
                continue
            assert item["numeric_authority"] == "vision_interpretation_only"
            assert item["structured_evidence"]["authoritative_for_exact_values"] is False

    def test_visual_inventory_is_consumed_read_only(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        before = copy.deepcopy(frozen_profile)
        run_vision_agent(synthetic_pdf, frozen_profile, str(tmp_path / "assets"))
        assert frozen_profile == before

    def test_vision_disabled_makes_no_model_calls_on_the_same_profile(
        self, synthetic_pdf, frozen_profile, tmp_path
    ):
        client = MockVisionClient()
        agent = VisionAgent(
            vision_client=client,
            config=VisionConfig(
                level_models={"basic": None, "medium": None, "advanced": "mock-advanced"},
                max_assets=0,
            ),
        )
        result = agent.run(
            pdf_path=synthetic_pdf,
            document_profile=frozen_profile,
            routing_state=create_routing_state(
                analysis_level="advanced", rag_enabled=False, vision_enabled=False
            ),
            output_dir=str(tmp_path / "assets"),
        )
        assert result["enabled"] is False
        assert result["results"] == []
        assert client.calls == []


class TestBenchmarkPaperIntegration:
    """Real paper: the visual inventory is produced by the frozen Pre-Analyzer."""

    def test_visual_inventory_matches_the_frozen_pre_analyzer(
        self, benchmark_profile, tmp_path
    ):
        figures = benchmark_profile["figures"]["confirmed_figures"]
        tables = benchmark_profile["tables"]["confirmed_tables"]
        assert len(figures) == 2
        assert len(tables) == 3

        result, client = run_vision_agent(
            BENCHMARK_PDF, benchmark_profile, str(tmp_path / "assets")
        )

        # 2 confirmed figures + 3 confirmed tables = 5 visual assets.
        assert result["assets_found"] == 5
        assert result["assets_analyzed"] == 5
        assert len(client.calls) == 5

        by_id = {item["asset_id"]: item for item in result["results"]}
        assert sorted(item["page_number"] for item in result["results"]) == [3, 4, 6, 8, 9]
        assert by_id["figure_001"]["page_number"] == figures[0]["page"]
        assert by_id["figure_002"]["page_number"] == figures[1]["page"]
        assert by_id["table_003"]["page_number"] == tables[2]["page"]

    def test_confirmed_bboxes_are_the_regions_actually_cropped(
        self, benchmark_profile, tmp_path
    ):
        from backend.vision_agent.asset_extractor import extract_visual_assets
        from backend.vision_agent.config import VisionConfig as AssetConfig

        report = extract_visual_assets(
            BENCHMARK_PDF,
            benchmark_profile,
            AssetConfig(max_assets=0),
            output_dir=str(tmp_path / "assets"),
        )
        confirmed = sorted(
            benchmark_profile["figures"]["confirmed_figures"], key=lambda f: f["page"]
        )
        figure_assets = sorted(
            [asset for asset in report.assets if asset.asset_type == "figure"],
            key=lambda asset: asset.page_number,
        )
        assert len(figure_assets) == len(confirmed) == 2

        for asset, source in zip(figure_assets, confirmed):
            # The crop is the confirmed box expanded by the small padding and
            # clamped to the page, so it must contain the profile box.
            assert asset.page_number == source["page"]
            assert asset.extraction_method == "bbox_crop"
            assert asset.fallback_used is False
            assert asset.bbox[0] <= source["bbox"][0] and asset.bbox[1] <= source["bbox"][1]
            assert asset.bbox[2] >= source["bbox"][2] and asset.bbox[3] >= source["bbox"][3]

    def test_machine_readable_table_values_are_trusted_over_vision(
        self, benchmark_profile, tmp_path
    ):
        result, _ = run_vision_agent(
            BENCHMARK_PDF, benchmark_profile, str(tmp_path / "assets")
        )
        tables = {t["id"]: t for t in benchmark_profile["tables"]["confirmed_tables"]}
        assert any(
            t.get("structure_source") == "pymupdf_find_tables" for t in tables.values()
        )

        for item in result["results"]:
            if item["asset_type"] != "table":
                continue
            evidence = item["structured_evidence"]
            assert evidence is not None
            if evidence["authoritative_for_exact_values"]:
                # Machine-readable cells exist: vision output is NOT the source of
                # exact numbers, and the extracted values travel alongside.
                assert item["numeric_authority"] == "document_profile_machine_readable"
                assert evidence["source"] == "pymupdf_find_tables"
            else:
                assert item["numeric_authority"] == "vision_interpretation_only"

    def test_full_result_payload_is_json_serializable(self, benchmark_profile, tmp_path):
        result, _ = run_vision_agent(
            BENCHMARK_PDF, benchmark_profile, str(tmp_path / "assets")
        )
        payload = json.dumps(result)
        assert len(payload) > 1000
        assert "sk-" not in payload

    def test_assets_are_written_as_readable_images(self, benchmark_profile, tmp_path):
        output_dir = tmp_path / "assets"
        result, _ = run_vision_agent(BENCHMARK_PDF, benchmark_profile, str(output_dir))
        from PIL import Image

        files = sorted(os.listdir(output_dir))
        assert files, "expected extracted asset images"
        for item in result["results"]:
            assert os.path.exists(item["image_path"])
            with Image.open(item["image_path"]) as image:
                assert image.mode == "RGB"
                assert max(image.size) <= 1600