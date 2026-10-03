"""
Offline unit tests for evidence context assembly.

Verifies:
- the bundle is built from the frozen Document Profile without re-detecting
  anything (candidate-only inventories stay empty)
- section/inventory/equation/reference context and the citable source index
- RAG and Vision absence produces explicit deterministic gaps instead of errors
- machine-readable table values are only carried when the Vision Agent marked them
  authoritative
- context budgets truncate evidence and record both the truncation and the gap
"""

import pytest

from backend.analysis_agent.config import AnalysisConfig
from backend.analysis_agent.context_builder import (
    ContextBuilderError,
    build_evidence_bundle,
)


def make_profile(**overrides) -> dict:
    profile = {
        "filename": "synthetic.pdf",
        "page_count": 11,
        "word_count": 4990,
        "text_length": 36309,
        "metadata": {"title": "Synthetic Paper", "author": "A. Author"},
        "content_characteristics": {"has_methodology": True, "has_tables": True},
        "sections": [
            {
                "title": "Abstract",
                "normalized_title": "abstract",
                "number": "",
                "page": 1,
                "level": 1,
                "confidence": 0.7,
            },
            {
                "title": "6 Results",
                "normalized_title": "results",
                "number": "6",
                "page": 8,
                "level": 1,
                "confidence": 0.8,
            },
        ],
        "figures": {
            "count": 2,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 3,
                    "caption": "Figure 1: architecture.",
                    "type": "raster_image",
                    "status": "confirmed",
                    "confidence": 0.9,
                },
                {
                    "id": "figure_2",
                    "figure_number": "2",
                    "page": 4,
                    "caption": "Figure 2: attention.",
                    "type": "raster_image",
                    "status": "confirmed",
                    "confidence": 0.9,
                },
            ],
            "figure_candidates": [{"id": "figure_candidate_1", "page": 5}],
            "candidate_count": 1,
        },
        "tables": {
            "count": 3,
            "confirmed_tables": [
                {
                    "id": "table_1",
                    "table_number": "1",
                    "page": 6,
                    "caption": "Table 1: complexity.",
                    "rows": 5,
                    "columns": 4,
                    "status": "confirmed",
                    "structure_source": "docling",
                    "confidence": 0.9,
                },
                {
                    "id": "table_2",
                    "table_number": "2",
                    "page": 8,
                    "caption": "Table 2: BLEU.",
                    "rows": 11,
                    "columns": 5,
                    "status": "confirmed",
                    "structure_source": "docling",
                    "confidence": 0.9,
                },
                {
                    "id": "table_3",
                    "table_number": "3",
                    "page": 9,
                    "caption": "Table 3: ablations.",
                    "rows": 7,
                    "columns": 4,
                    "status": "confirmed",
                    "structure_source": "docling",
                    "confidence": 0.9,
                },
            ],
        },
        "equations": {
            "count": 5,
            "confirmed_equations": [
                {
                    "id": f"eq_{index}",
                    "page": 4,
                    "equation_number": str(index),
                    "representation": f"x_{index} = y_{index}",
                    "status": "confirmed",
                    "structure_source": "docling_formula",
                    "confidence": 1.0,
                }
                for index in range(1, 6)
            ],
        },
        "references": {
            "has_references": True,
            "count": 32,
            "start_page": 10,
            "confidence": 1.0,
            "is_sequential": True,
        },
        "pages": [{"page_number": index} for index in range(1, 12)],
    }
    profile.update(overrides)
    return profile


def make_rag(enabled: bool = True, chunks: int = 5) -> dict:
    return {
        "agent": "rag",
        "enabled": enabled,
        "level": "medium",
        "query": "results",
        "chunks_indexed": 40,
        "top_k": chunks,
        "embedding_model": "text-embedding-3-small",
        "results": [
            {
                "chunk_id": f"chunk_{index:03d}",
                "score": round(0.8 - index * 0.05, 4),
                "page_start": index,
                "page_end": index + 1,
                "section": "6 Results",
                "text": f"Retrieved passage number {index} describing the reported results.",
                "word_count": 9,
            }
            for index in range(1, chunks + 1)
        ],
    }


def make_vision(enabled: bool = True, authoritative: bool = False) -> dict:
    results = []
    for asset_id, asset_type, page in (
        ("figure_001", "figure", 3),
        ("figure_002", "figure", 4),
        ("table_001", "table", 6),
        ("table_002", "table", 8),
        ("table_003", "table", 9),
    ):
        results.append(
            {
                "asset_id": asset_id,
                "asset_type": asset_type,
                "page_number": page,
                "section": "6 Results",
                "caption": f"{asset_type.title()} on page {page}.",
                "observation": "A visual element with labelled regions is visible.",
                "interpretation": "It appears to illustrate the reported comparison.",
                "key_elements": ["legend", "axis labels"],
                "reported_relationships": ["one series appears higher"],
                "supports_claims": ["the reported comparison appears visually supported"],
                "uncertainties": ["small text limits precision"],
                "caption_consistency": {"status": "consistent", "explanation": "matches"},
                "model": "mock-vision",
                "numeric_authority": (
                    "document_profile_machine_readable"
                    if asset_type == "table" and authoritative
                    else "vision_interpretation_only"
                ),
                "structured_evidence": {
                    "authoritative_for_exact_values": authoritative
                    and asset_type == "table",
                    "headers": ["Model", "BLEU"],
                    "rows": [["Transformer", "28.4"], ["ByteNet", "23.75"]],
                },
            }
        )
    return {
        "agent": "vision",
        "enabled": enabled,
        "level": "medium",
        "model": "mock-vision",
        "provider": "mock",
        "assets_found": 5,
        "assets_analyzed": 5,
        "assets_failed": 0,
        "results": results,
        "failures": [],
        "extraction_failures": [],
        "equations_available": 5,
        "message": "ok",
    }


def gap_categories(bundle) -> list:
    return [gap.category for gap in bundle.deterministic_gaps]


def test_bundle_is_built_from_the_frozen_profile():
    bundle = build_evidence_bundle(make_profile(), make_rag(), make_vision())

    assert bundle.filename == "synthetic.pdf"
    assert bundle.page_count == 11
    assert bundle.word_count == 4990
    assert bundle.metadata["title"] == "Synthetic Paper"

    # Sections keep profile order and get stable, sequential ids.
    assert [section.source_id for section in bundle.sections] == [
        "section_001",
        "section_002",
    ]
    assert bundle.sections[1].title == "6 Results"
    assert bundle.sections[1].page == 8

    # Inventories come from the confirmed lists only (ids preserved verbatim).
    assert [item.source_id for item in bundle.figures] == ["figure_1", "figure_2"]
    assert [item.source_id for item in bundle.tables] == [
        "table_1",
        "table_2",
        "table_3",
    ]
    assert [item.source_id for item in bundle.equations] == [
        f"eq_{index}" for index in range(1, 6)
    ]
    assert bundle.equations[0].description == "x_1 = y_1"
    assert bundle.tables[1].rows == 11

    # Reference statistics are passed through, not re-derived.
    assert bundle.references_summary["count"] == 32
    assert bundle.references_summary["start_page"] == 10

    # Retrieved chunks and visual analyses are consumed as given.
    assert [chunk.chunk_id for chunk in bundle.text_evidence] == [
        f"chunk_{index:03d}" for index in range(1, 6)
    ]
    assert bundle.text_evidence[0].score == 0.75
    assert [visual.asset_id for visual in bundle.visual_evidence][0] == "figure_001"
    assert bundle.visual_evidence[0].observation.startswith("A visual element")

    assert bundle.counts["sections"] == 2
    assert bundle.counts["sections_available"] == 2
    assert bundle.counts["figures_available"] == 2
    assert bundle.counts["tables_available"] == 3
    assert bundle.counts["equations_available"] == 5
    assert bundle.counts["references"] == 32
    assert bundle.counts["rag_chunks"] == 5
    assert bundle.counts["visual_assets"] == 5


def test_candidate_items_are_never_promoted_into_the_context():
    profile = make_profile(
        figures={"count": 0, "confirmed_figures": [], "figure_candidates": [{"id": "c1"}]},
        tables={"count": 0, "confirmed_tables": [], "table_candidates": [{"id": "t1"}]},
        equations={
            "count": 0,
            "confirmed_equations": [],
            "equation_candidates": [{"id": "e1"}],
        },
    )
    bundle = build_evidence_bundle(profile)

    assert bundle.figures == []
    assert bundle.tables == []
    assert bundle.equations == []
    assert bundle.counts["figures_available"] == 0


def test_source_index_lists_every_citable_id_with_type_and_pages():
    bundle = build_evidence_bundle(make_profile(), make_rag(), make_vision())

    assert bundle.source_index["document_profile"]["source_type"] == "document_profile"
    assert bundle.source_index["paper_metadata"]["detail"] == "Synthetic Paper"
    assert bundle.source_index["section_001"]["source_type"] == "section"
    assert bundle.source_index["section_002"]["pages"] == [8]
    assert bundle.source_index["figure_1"]["source_type"] == "figure"
    assert bundle.source_index["figure_1"]["pages"] == [3]
    assert bundle.source_index["table_2"]["source_type"] == "table"
    assert bundle.source_index["eq_3"]["source_type"] == "equation"
    assert bundle.source_index["references_summary"]["pages"] == [10]
    assert bundle.source_index["chunk_001"]["source_type"] == "text_chunk"
    assert bundle.source_index["chunk_001"]["pages"] == [1, 2]
    assert bundle.source_index["visual_table_003"]["source_type"] == "visual_asset"
    assert bundle.source_index["visual_table_003"]["pages"] == [9]


def test_missing_rag_and_vision_are_recorded_as_deterministic_gaps():
    bundle = build_evidence_bundle(make_profile())
    categories = gap_categories(bundle)

    assert "rag_unavailable" in categories
    assert "vision_unavailable" in categories
    assert "numeric_values_unavailable" in categories

    rag_gap = next(gap for gap in bundle.deterministic_gaps if gap.category == "rag_unavailable")
    assert rag_gap.source == "deterministic_context_check"
    assert "methodology" in rag_gap.affected_dimensions
    assert bundle.text_evidence == []
    assert bundle.visual_evidence == []


def test_disabled_upstream_agents_are_reported_as_routing_decisions():
    bundle = build_evidence_bundle(
        make_profile(), make_rag(enabled=False), make_vision(enabled=False)
    )
    categories = gap_categories(bundle)

    assert "rag_disabled" in categories
    assert "vision_disabled" in categories
    assert "rag_unavailable" not in categories
    assert "vision_unavailable" not in categories


def test_enabled_but_empty_upstream_results_are_gaps_not_errors():
    bundle = build_evidence_bundle(
        make_profile(),
        dict(make_rag(), results=[]),
        dict(make_vision(), results=[]),
    )
    categories = gap_categories(bundle)

    assert "rag_unavailable" in categories
    assert "vision_unavailable" in categories
    assert bundle.text_evidence == []
    assert bundle.visual_evidence == []


def test_machine_readable_values_are_only_carried_when_authoritative():
    not_authoritative = build_evidence_bundle(
        make_profile(), make_rag(), make_vision(authoritative=False)
    )
    assert all(
        visual.structured_values is None for visual in not_authoritative.visual_evidence
    )

    authoritative = build_evidence_bundle(
        make_profile(), make_rag(), make_vision(authoritative=True)
    )
    table_assets = [v for v in authoritative.visual_evidence if v.asset_type == "table"]
    assert table_assets
    assert table_assets[0].structured_values.startswith("headers: Model | BLEU")
    assert "Transformer | 28.4" in table_assets[0].structured_values
    assert "numeric_values_unavailable" not in gap_categories(authoritative)


def test_context_budget_truncates_evidence_and_records_the_omission():
    config = AnalysisConfig(max_context_chars=900)
    bundle = build_evidence_bundle(
        make_profile(), make_rag(), make_vision(), config=config
    )

    assert bundle.total_evidence_chars <= 900
    assert len(bundle.text_evidence) < 5
    assert any(
        "ANALYSIS_MAX_CONTEXT_CHARS" in note for note in bundle.truncations
    )
    assert "context_truncated" in gap_categories(bundle)
    # Dropped evidence must also leave the citable index.
    assert "chunk_005" not in bundle.source_index


def test_section_and_inventory_limits_are_recorded():
    config = AnalysisConfig(max_sections=1, max_inventory_items=1, max_equations=2)
    bundle = build_evidence_bundle(make_profile(), config=config)

    assert len(bundle.sections) == 1
    assert len(bundle.figures) == 1
    assert len(bundle.tables) == 1
    assert len(bundle.equations) == 2

    notes = " ".join(bundle.truncations)
    assert "sections: 2 available, 1 included" in notes
    assert "figures: 1 confirmed figure(s)" in notes
    assert "tables: 2 confirmed table(s)" in notes
    assert "equations: 3 confirmed equation(s)" in notes
    # The available counts stay truthful even when context is capped.
    assert bundle.counts["figures_available"] == 2
    assert bundle.counts["equations_available"] == 5


def test_long_captions_and_chunks_are_truncated_with_notes():
    long_caption = "C" * 900
    profile = make_profile(
        figures={
            "count": 1,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 3,
                    "caption": long_caption,
                    "confidence": 0.9,
                }
            ],
        }
    )
    rag = make_rag(chunks=1)
    rag["results"][0]["text"] = "T" * 5000
    bundle = build_evidence_bundle(
        profile, rag, config=AnalysisConfig(max_caption_chars=50, max_chunk_chars=100)
    )

    assert len(bundle.figures[0].caption) == 50
    assert bundle.figures[0].caption.endswith("...")
    assert len(bundle.text_evidence[0].text) == 100
    notes = " ".join(bundle.truncations)
    assert "caption(s) truncated" in notes
    assert "chunk(s) truncated" in notes


def test_headerless_or_empty_profile_is_rejected():
    with pytest.raises(ContextBuilderError):
        build_evidence_bundle(None)
    with pytest.raises(ContextBuilderError):
        build_evidence_bundle({})
    with pytest.raises(ContextBuilderError):
        build_evidence_bundle(["not", "a", "profile"])


def test_profile_without_sections_records_a_structure_gap():
    bundle = build_evidence_bundle(make_profile(sections=[]))

    assert "other" in gap_categories(bundle)
    gap = next(gap for gap in bundle.deterministic_gaps if gap.category == "other")
    assert "section" in gap.gap.lower()

