"""
Offline unit tests for the deterministic synthesis-context builder.

Verifies that the Critique Engine *projects* upstream structured output and nothing
more: document facts, routing, RAG/Vision provenance, the Phase 5 findings, the
supported-section set and the citation allow-list - deterministically, and within
the configured budgets.
"""

import json

import pytest

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import (
    DIMENSION_TO_SECTION,
    ContextBuilderError,
    build_synthesis_context,
)
from backend.critique_engine.tests import (
    routing_state,
    synthetic_analysis_result,
    synthetic_profile,
    synthetic_rag_result,
    synthetic_vision_result,
)


def config(**overrides):
    payload = {}
    payload.update(overrides)
    return CritiqueConfig(**payload)


def build(**overrides):
    payload = {
        "document_profile": synthetic_profile(),
        "routing_state": routing_state(),
        "rag_result": synthetic_rag_result(),
        "vision_result": synthetic_vision_result(),
        "analysis_result": synthetic_analysis_result(),
    }
    payload.update(overrides)
    return build_synthesis_context(config=payload.pop("config", config()), **payload)


def test_document_summary_is_projected_from_the_frozen_profile():
    context = build()
    document = context.document
    assert document.title == "A Synthetic Study Of Structured Evidence"
    assert document.page_count == 11
    assert document.word_count == 4200
    assert document.section_count == 3
    assert document.figure_count == 1
    assert document.table_count == 1
    assert document.equation_count == 1
    assert document.reference_count == 30


def test_routing_state_is_read_verbatim_without_a_new_decision():
    context = build(routing_state=routing_state(analysis_level="medium"))
    assert context.routing.analysis_level == "medium"
    assert context.routing.rag_enabled is True
    assert context.routing.vision_enabled is True
    assert context.routing.rag_level == "medium"
    assert context.routing.confidence == {"analysis": 0.93, "rag": 0.71, "vision": 0.68}


def test_routing_state_accepts_the_flat_contract_shape():
    context = build(
        routing_state={
            "analysis_level": "basic",
            "rag_enabled": False,
            "rag_level": None,
            "vision_enabled": False,
        }
    )
    assert context.routing.analysis_level == "basic"
    assert context.routing.rag_enabled is False
    assert context.routing.rag_level is None


def test_retrieved_evidence_keeps_chunk_ids_pages_and_section():
    context = build()
    ids = [chunk.source_id for chunk in context.retrieved_evidence]
    assert ids == ["chunk_001", "chunk_002"]
    assert context.retrieved_evidence[0].pages == [2]
    assert context.retrieved_evidence[1].pages == [8, 9]
    assert context.retrieved_evidence[0].section == "Introduction"
    assert "sequence transduction" in context.retrieved_evidence[0].excerpt


def test_visual_evidence_preserves_asset_metadata_and_uncertainties():
    context = build()
    asset = context.visual_evidence[0]
    assert asset.source_id == "figure_001"
    assert asset.asset_type == "figure"
    assert asset.page == 6
    assert asset.section == "Method"
    assert "stacked blocks" in asset.observation
    assert "encoder-decoder" in asset.interpretation
    assert asset.uncertainties == ["Small labels limit certainty about the wiring."]


def test_analysis_findings_are_projected_not_reinterpreted():
    context = build()
    dimensions = {entry.dimension: entry for entry in context.analysis.dimensions}
    assert set(dimensions) == set(DIMENSION_TO_SECTION)
    assert "sequence transduction" in dimensions["research_problem"].summary
    assert dimensions["results"].findings
    assert context.analysis.claims[0].status == "partially_supported"
    assert context.analysis.author_stated_limitations
    assert context.analysis.analyst_identified_limitations
    assert context.analysis.open_questions
    assert context.analysis.evidence_gaps[0].source == "analysis"


def test_source_index_covers_every_upstream_artifact():
    context = build()
    keys = set(context.source_index)
    assert "rag:chunk_001" in keys
    assert "vision:figure_001" in keys
    assert "document:document_profile" in keys
    assert "document:table_001" in keys
    # The same id reached through the analysis stage is citable as its own kind.
    assert "analysis:chunk_001" in keys
    assert context.source_index["rag:chunk_001"]["pages"] == [2]
    assert context.source_index["vision:figure_001"]["page"] == 6


def test_source_index_is_sorted_for_determinism():
    context = build()
    assert list(context.source_index) == sorted(context.source_index)


def test_ids_cited_only_inside_findings_are_still_citable():
    """
    Phase 5 may cite an id inside a finding (not only at dimension level).

    Such an id is real provenance, so the critique must be allowed to cite it
    instead of the validator reporting it as fabricated.
    """
    analysis = synthetic_analysis_result()
    analysis["analysis"]["methodology"]["findings"].append(
        {
            "statement": "The architecture detail appears in a specific section.",
            "evidence_status": "supported",
            "evidence_refs": [
                {
                    "source_type": "section",
                    "source_id": "section_009",
                    "pages": [9],
                    "verified": True,
                }
            ],
            "confidence": None,
            "reasoning": "Cited inside the finding only.",
            "caveats": [],
        }
    )
    context = build(analysis_result=analysis)
    assert "analysis:section_009" in context.source_index


def test_ids_cited_inside_limitations_are_still_citable():
    analysis = synthetic_analysis_result()
    analysis["analysis"]["limitations"]["analyst_identified"][0]["evidence_refs"] = [
        {
            "source_type": "visual_asset",
            "source_id": "figure_001",
            "pages": [6],
            "verified": True,
        }
    ]
    context = build(analysis_result=analysis)
    assert "analysis:figure_001" in context.source_index


def test_context_is_deterministic_for_identical_inputs():
    first = build().model_dump()
    second = build().model_dump()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_supported_sections_follow_the_available_evidence():
    context = build()
    supported = set(context.supported_sections.section_ids)
    assert "methodology" in supported
    assert "figures_and_tables" in supported
    assert "claim_evidence_assessment" in supported
    assert context.supported_sections.reasons["methodology"]


def test_visual_section_is_unsupported_when_vision_produced_nothing():
    result = synthetic_vision_result()
    result["results"] = []
    context = build(vision_result=result)
    assert "figures_and_tables" not in context.supported_sections.section_ids


def test_disabled_upstream_stages_produce_honest_deterministic_gaps():
    context = build(
        rag_result=None,
        vision_result=None,
        routing_state=routing_state(rag_enabled=False, vision_enabled=False),
    )
    categories = [gap.category for gap in context.deterministic_gaps]
    assert "rag_disabled" in categories
    assert "vision_disabled" in categories
    assert context.retrieved_evidence == []
    assert context.visual_evidence == []


def test_budget_limits_are_applied_and_recorded():
    context = build(config=config(max_rag_chunks=1, max_visual_assets=0))
    assert len(context.retrieved_evidence) == 1
    assert context.visual_evidence == []
    assert any("CRITIQUE_MAX_RAG_CHUNKS" in item for item in context.truncations)
    assert any("CRITIQUE_MAX_VISUAL_ASSETS" in item for item in context.truncations)


def test_char_budget_drops_excerpts_and_records_the_truncation():
    context = build(config=config(max_context_chars=200))
    assert context.total_context_chars <= 200 or not context.retrieved_evidence
    assert any(
        "CRITIQUE_MAX_CONTEXT_CHARS" in item for item in context.truncations
    )


def test_missing_analysis_result_is_a_context_builder_error():
    with pytest.raises(ContextBuilderError):
        build(analysis_result=None)
    with pytest.raises(ContextBuilderError):
        build(analysis_result={"status": "failed", "analysis": None})


def test_missing_profile_degrades_instead_of_failing_and_records_a_gap():
    context = build(document_profile=None)
    assert context.document.page_count == 0
    assert context.source_index.get("rag:chunk_001") is not None
    assert any(
        "document profile was not supplied" in gap.gap
        for gap in context.deterministic_gaps
    )


def test_unverified_analysis_references_are_carried_forward_as_a_gap():
    analysis = synthetic_analysis_result()
    analysis["unverified_evidence_refs"] = 3
    context = build(analysis_result=analysis)
    assert any(
        gap.category == "unverified_reference" for gap in context.deterministic_gaps
    )


def test_counts_summarize_what_was_synthesized():
    context = build()
    assert context.counts["rag_sources"] == 2
    assert context.counts["vision_sources"] == 1
    assert context.counts["dimensions"] == 9
    assert context.counts["claims"] == 1