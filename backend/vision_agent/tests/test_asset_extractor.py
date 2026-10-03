"""
Offline unit tests for visual asset extraction.

Verifies:
- confirmed figures/tables are consumed from the frozen Document Profile shape
- precise bbox cropping in PyMuPDF top-left coordinates
- caption-band and whole-page fallbacks are flagged, never silently claimed as exact
- per-asset failure isolation (a missing page does not abort the run)
- deterministic canonical asset ids and provenance metadata

A small synthetic PDF is generated with PyMuPDF; no real paper or network is used.
"""

import os

import pymupdf
import pytest

from backend.vision_agent import asset_extractor
from backend.vision_agent.asset_extractor import (
    AssetExtractionError,
    collect_confirmed_visual_assets,
    extract_visual_assets,
    normalize_bbox,
)
from backend.vision_agent.config import DEFAULT_ASSET_OUTPUT_DIR, VisionConfig

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0


def make_synthetic_pdf(path: str, page_count: int = 3) -> str:
    """Create a small PDF with drawable figure/table regions and captions."""
    doc = pymupdf.open()
    for index in range(page_count):
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text((60, 60), f"Synthetic paper page {index + 1}", fontsize=12)
        # A paragraph above the visual region (used as surrounding context).
        page.insert_text((60, 90), "Body text describing the experiment setup.", fontsize=9)
        # A drawn "figure" region (vector strokes, no raster image needed).
        rect = pymupdf.Rect(60, 120, 300, 260)
        page.draw_rect(rect, color=(0, 0, 0), width=1)
        page.draw_line(pymupdf.Point(70, 240), pymupdf.Point(290, 150))
        page.insert_text((60, 280), "Figure 1: Synthetic accuracy curve.", fontsize=9)
        # A drawn "table" region with ruling lines.
        table_rect = pymupdf.Rect(60, 330, 500, 430)
        for offset in (0, 25, 50, 75, 100):
            y = table_rect.y0 + offset
            page.draw_line(pymupdf.Point(table_rect.x0, y), pymupdf.Point(table_rect.x1, y))
        page.draw_line(
            pymupdf.Point(table_rect.x0, table_rect.y0),
            pymupdf.Point(table_rect.x0, table_rect.y1),
        )
        page.insert_text((60, 450), "Table 1: Synthetic results.", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def build_profile() -> dict:
    """Profile shaped exactly like the frozen Document Pre-Analyzer output."""
    return {
        "filename": "synthetic.pdf",
        "page_count": 3,
        "sections": [
            {"title": "1 Introduction", "normalized_title": "introduction", "page": 1, "level": 1},
            {"title": "4 Experiments", "normalized_title": "experiments", "page": 2, "level": 1},
        ],
        "figures": {
            "count": 2,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 1,
                    "caption": "Figure 1: Synthetic accuracy curve.",
                    "caption_bbox": [60.0, 270.0, 300.0, 285.0],
                    "bbox": [60.0, 120.0, 300.0, 260.0],
                    "type": "vector_graphic",
                    "status": "confirmed",
                    "confidence": 0.75,
                },
                {
                    "id": "figure_2",
                    "figure_number": "2",
                    "page": 9,  # deliberately beyond the synthetic page count
                    "caption": "Figure 2: Missing page figure.",
                    "caption_bbox": None,
                    "bbox": [60.0, 120.0, 300.0, 260.0],
                    "type": "raster_image",
                    "status": "confirmed",
                    "confidence": 0.6,
                },
            ],
            "figure_candidates": [
                {
                    "id": "figure_3",
                    "figure_number": "3",
                    "page": 2,
                    "caption": "Figure 3: candidate only",
                    "bbox": [60.0, 120.0, 300.0, 260.0],
                    "status": "candidate",
                    "confidence": 0.3,
                }
            ],
        },
        "tables": {
            "count": 3,
            "confirmed_tables": [
                {
                    "id": "table_1_1_3",
                    "table_number": "1",
                    "page": 2,
                    "caption": "Table 1: Synthetic results.",
                    "caption_bbox": [60.0, 440.0, 300.0, 455.0],
                    "bbox": [60.0, 330.0, 500.0, 430.0],
                    "rows": 4,
                    "columns": 3,
                    "status": "confirmed",
                    "structure_source": "ruling_lines",
                    "confidence": 0.6,
                },
                {
                    "id": "table_2_2_7",
                    "table_number": "2",
                    "page": 2,
                    "caption": "Table 2: Fallback table.",
                    "caption_bbox": [60.0, 500.0, 300.0, 515.0],
                    "bbox": None,  # no usable box -> caption-band fallback
                    "rows": 3,
                    "columns": 2,
                    "status": "confirmed",
                    "structure_source": "ruling_lines",
                    "confidence": 0.5,
                },
                {
                    "id": "table_3_3_11",
                    "table_number": "3",
                    "page": 1,
                    "caption": "Table 3: Out-of-page box.",
                    "caption_bbox": None,
                    "bbox": [-50.0, -20.0, 900.0, 900.0],  # must be clamped to the page
                    "rows": 2,
                    "columns": 2,
                    "status": "confirmed",
                    "structure_source": "pymupdf_find_tables",
                    "confidence": 0.7,
                },
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


@pytest.fixture()
def synthetic_pdf(tmp_path) -> str:
    return make_synthetic_pdf(str(tmp_path / "synthetic.pdf"))


@pytest.fixture()
def output_dir(tmp_path) -> str:
    target = tmp_path / "assets"
    target.mkdir()
    return str(target)


def _extract(pdf_path: str, output_dir: str, profile: dict = None, config: VisionConfig = None):
    return extract_visual_assets(
        pdf_path=pdf_path,
        document_profile=profile if profile is not None else build_profile(),
        config=config or VisionConfig(max_assets=0),
        output_dir=output_dir,
    )


def _by_source(report, source_id: str):
    """Look an extracted asset up by its Document Profile provenance id."""
    return next(a for a in report.assets if a.source_id == source_id)


class TestConfirmedInventory:
    def test_confirmed_assets_only_and_sorted_with_canonical_ids(self):
        refs = collect_confirmed_visual_assets(build_profile())
        ids = [ref.asset_id for ref in refs]
        assert ids == ["figure_001", "table_001", "table_002", "table_003", "figure_002"]
        # Sorted by page: figure_2 lives on page 9, so it comes last.
        assert [ref.page_number for ref in refs] == [1, 1, 2, 2, 9]

    def test_candidate_assets_excluded_by_default(self):
        refs = collect_confirmed_visual_assets(build_profile())
        assert all("candidate" not in (ref.caption or "") for ref in refs)
        assert len(refs) == 5

    def test_candidate_assets_included_on_explicit_opt_in(self):
        refs = collect_confirmed_visual_assets(build_profile(), include_candidates=True)
        assert any("candidate" in (ref.caption or "") for ref in refs)
        assert len(refs) == 6

    def test_provenance_fields_preserved(self):
        ref = collect_confirmed_visual_assets(build_profile())[0]
        assert ref.source_id == "figure_1"
        assert ref.label == "1"
        assert ref.source_confidence == pytest.approx(0.75)
        assert ref.caption.startswith("Figure 1:")
        assert ref.asset_subtype == "vector_graphic"

    def test_entries_without_page_are_skipped(self):
        profile = build_profile()
        profile["figures"]["confirmed_figures"].append(
            {"id": "figure_x", "figure_number": "x", "page": None, "bbox": [1, 2, 3, 4]}
        )
        refs = collect_confirmed_visual_assets(profile)
        assert all(ref.source_id != "figure_x" for ref in refs)

    def test_equations_are_never_extracted_as_assets(self):
        refs = collect_confirmed_visual_assets(build_profile())
        assert all(ref.asset_type in {"figure", "table"} for ref in refs)
        assert len(refs) == 5  # the 5 confirmed equations produce no visual asset

    def test_max_assets_cap_is_applied(self):
        refs = collect_confirmed_visual_assets(build_profile(), max_assets=2)
        assert len(refs) == 2


class TestBboxCoordinateHandling:
    def test_inverted_bbox_is_normalized_not_reinterpreted(self):
        # Ordering issues are recovered; the coordinate space is not reinterpreted.
        ordered = normalize_bbox([300.0, 260.0, 60.0, 120.0])
        assert ordered == [60.0, 120.0, 300.0, 260.0]

    def test_bbox_is_clamped_to_the_page(self):
        page = pymupdf.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT)
        clamped = normalize_bbox([-50.0, -20.0, 900.0, 900.0], page)
        assert clamped == [0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT]

    def test_malformed_or_tiny_boxes_are_rejected(self):
        assert normalize_bbox(None) is None
        assert normalize_bbox([1, 2, 3]) is None
        assert normalize_bbox(["a", "b", "c", "d"]) is None
        assert normalize_bbox([10, 10, 15, 15]) is None  # below the usable minimum

    def test_out_of_page_box_is_clamped_during_extraction(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        table = _by_source(report, "table_3_3_11")
        assert table.bbox[0] >= 0 and table.bbox[1] >= 0
        assert table.bbox[2] <= PAGE_WIDTH and table.bbox[3] <= PAGE_HEIGHT
        assert table.extraction_method == "bbox_crop"
        assert table.coordinate_space == "pymupdf_top_left"

    def test_padding_never_escapes_the_page(self, synthetic_pdf, output_dir):
        config = VisionConfig(max_assets=0, bbox_padding_pts=40.0)
        report = _extract(synthetic_pdf, output_dir, config=config)
        for asset in report.assets:
            assert asset.bbox[0] >= 0 and asset.bbox[1] >= 0
            assert asset.bbox[2] <= PAGE_WIDTH and asset.bbox[3] <= PAGE_HEIGHT


class TestExtraction:
    def test_confirmed_figure_is_cropped_by_bbox(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        figure = next(a for a in report.assets if a.asset_id == "figure_001")
        assert figure.asset_type == "figure"
        assert figure.page_number == 1
        assert figure.extraction_method == "bbox_crop"
        assert figure.fallback_used is False
        assert figure.label == "1"
        assert figure.source_id == "figure_1"
        assert figure.source_confidence == pytest.approx(0.75)
        assert figure.image_width > 0 and figure.image_height > 0
        assert figure.image_bytes > 0
        assert os.path.exists(figure.image_path)
        assert figure.section == "1 Introduction"

    def test_confirmed_table_is_cropped_by_bbox(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        table = _by_source(report, "table_1_1_3")
        assert table.asset_type == "table"
        assert table.page_number == 2
        assert table.extraction_method == "bbox_crop"
        assert table.fallback_used is False
        assert os.path.exists(table.image_path)

    def test_table_without_bbox_uses_caption_band_and_is_flagged(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        table = _by_source(report, "table_2_2_7")
        assert table.asset_id == "table_003"  # canonical id, deterministic by page order
        assert table.extraction_method == "caption_band_crop"
        assert table.fallback_used is True  # never claimed as an exact crop
        assert os.path.exists(table.image_path)

    def test_table_without_any_box_falls_back_to_page_crop(self, synthetic_pdf, output_dir, tmp_path):
        profile = build_profile()
        second = profile["tables"]["confirmed_tables"][1]
        second["caption_bbox"] = None
        report = _extract(synthetic_pdf, output_dir, profile=profile)
        table = _by_source(report, "table_2_2_7")
        assert table.extraction_method == "page_crop_fallback"
        assert table.fallback_used is True

    def test_table_structure_metadata_is_preserved(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        table = _by_source(report, "table_1_1_3")
        assert table.structured_table is not None
        assert table.structured_table["source"] == "document_profile"
        assert table.structured_table["row_count"] == 4
        assert table.structured_table["column_count"] == 3
        # No machine-readable cells were read, so exact values are not authoritative.
        assert table.structured_table["authoritative_for_exact_values"] is False

    def test_figures_carry_no_structured_table_payload(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir)
        figure = _by_source(report, "figure_1")
        assert figure.structured_table is None

    def test_surrounding_text_context_is_bounded(self, synthetic_pdf, output_dir):
        config = VisionConfig(max_assets=0, context_max_chars=40)
        report = _extract(synthetic_pdf, output_dir, config=config)
        figure = _by_source(report, "figure_1")
        assert figure.context_text is not None
        assert len(figure.context_text) <= 43

    def test_max_assets_cap_records_skipped_assets(self, synthetic_pdf, output_dir):
        config = VisionConfig(max_assets=2)
        report = _extract(synthetic_pdf, output_dir, config=config)
        assert report.assets_considered == 2
        assert report.assets_skipped_over_limit == 3
        assert len(report.assets) == 2

    def test_empty_profile_yields_no_assets(self, synthetic_pdf, output_dir):
        report = _extract(synthetic_pdf, output_dir, profile={"figures": {}, "tables": {}})
        assert report.assets == []
        assert report.failures == []


class TestFailureIsolation:
    def test_missing_page_records_failure_and_other_assets_still_extract(
        self, synthetic_pdf, output_dir
    ):
        report = _extract(synthetic_pdf, output_dir)
        failed_ids = [failure.asset_id for failure in report.failures]
        assert failed_ids == ["figure_002"]
        failure = report.failures[0]
        assert failure.stage == "extraction"
        assert failure.page_number == 9
        assert "not present" in failure.error
        # The four other confirmed assets were extracted successfully.
        assert len(report.assets) == 4
        assert report.assets_considered == 5

    def test_unreadable_pdf_raises_extraction_error(self, output_dir):
        with pytest.raises(AssetExtractionError):
            _extract("does_not_exist.pdf", output_dir)

    def test_empty_pdf_raises_extraction_error(self, output_dir, tmp_path):
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        with pytest.raises(AssetExtractionError):
            _extract(str(empty), output_dir)


class TestOutputDirectoryResolution:
    """Asset files must land in the resolved output directory, always."""

    def test_configured_output_dir_is_used_when_caller_passes_none(
        self, synthetic_pdf, tmp_path
    ):
        configured = str(tmp_path / "configured")
        report = extract_visual_assets(
            pdf_path=synthetic_pdf,
            document_profile=build_profile(),
            config=VisionConfig(max_assets=0, output_dir=configured),
            output_dir=None,
        )
        assert report.assets
        for asset in report.assets:
            assert os.path.dirname(asset.image_path) == configured
            assert os.path.exists(asset.image_path)

    def test_package_default_is_used_when_nothing_is_configured(
        self, synthetic_pdf, tmp_path, monkeypatch
    ):
        # Regression guard: the default directory constant must actually be
        # resolvable inside the extractor module.
        default = str(tmp_path / "package-default")
        monkeypatch.setattr(asset_extractor, "DEFAULT_ASSET_OUTPUT_DIR", default)
        report = extract_visual_assets(
            pdf_path=synthetic_pdf,
            document_profile=build_profile(),
            config=VisionConfig(max_assets=0, output_dir=None),
            output_dir=None,
        )
        assert report.assets
        for asset in report.assets:
            assert os.path.dirname(asset.image_path) == default
            assert os.path.exists(asset.image_path)

    def test_empty_string_config_value_still_resolves_to_the_default(
        self, synthetic_pdf, tmp_path, monkeypatch
    ):
        # An empty VISION_ASSET_OUTPUT_DIR (e.g. "VISION_ASSET_OUTPUT_DIR=" in
        # .env) must never produce an empty target path for os.makedirs.
        default = str(tmp_path / "empty-safe")
        monkeypatch.setattr(asset_extractor, "DEFAULT_ASSET_OUTPUT_DIR", default)
        report = extract_visual_assets(
            pdf_path=synthetic_pdf,
            document_profile=build_profile(),
            config=VisionConfig(max_assets=0, output_dir=""),
            output_dir="",
        )
        assert report.assets
        for asset in report.assets:
            assert os.path.dirname(asset.image_path) == default

    def test_shipped_default_directory_is_never_empty(self):
        assert DEFAULT_ASSET_OUTPUT_DIR.strip()
        assert os.path.isabs(DEFAULT_ASSET_OUTPUT_DIR)