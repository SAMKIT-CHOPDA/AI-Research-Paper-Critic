"""
Offline unit tests for the synthesis prompt.

Verifies that the prompt states the synthesis boundary, the provenance and
no-fabrication rules, the no-score rule, the author-vs-analyst separation and the
uncertainty vocabulary - and that it renders only the structured inputs.
"""

import json

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import build_synthesis_context
from backend.critique_engine.prompts import (
    OUTPUT_CONTRACT_HEADER,
    RULE_BLOCK,
    SECTION_EXPECTATIONS,
    build_context_block,
    build_critique_prompt,
    render_gaps_block,
    render_section_guidance,
    render_source_index_block,
)
from backend.critique_engine.schemas import SECTION_IDS
from backend.critique_engine.tests import (
    routing_state,
    synthetic_analysis_result,
    synthetic_profile,
    synthetic_rag_result,
    synthetic_vision_result,
)


def context(**overrides):
    payload = {
        "document_profile": synthetic_profile(),
        "routing_state": routing_state(),
        "rag_result": synthetic_rag_result(),
        "vision_result": synthetic_vision_result(),
        "analysis_result": synthetic_analysis_result(),
    }
    payload.update(overrides)
    return build_synthesis_context(
        config=CritiqueConfig(), **payload
    )


def test_prompt_combines_rules_contract_and_context():
    prompt = build_critique_prompt(context())
    assert RULE_BLOCK in prompt
    assert OUTPUT_CONTRACT_HEADER in prompt
    assert "STRUCTURED SYNTHESIS CONTEXT" in prompt
    assert "DOCUMENT" in prompt


def test_prompt_declares_the_synthesis_boundary():
    assert "NOT re-analyzing the paper" in RULE_BLOCK
    assert "run retrieval" in RULE_BLOCK
    assert "re-decide any routing level" in RULE_BLOCK


def test_prompt_forbids_fabricated_identifiers_and_pages():
    assert "chunk_999" in RULE_BLOCK
    assert "figure_999" in RULE_BLOCK
    assert "Never invent page numbers" in RULE_BLOCK


def test_prompt_forbids_any_numeric_paper_score():
    assert "Never produce a score" in RULE_BLOCK
    assert "7/10" in RULE_BLOCK
    assert "82/100" in RULE_BLOCK


def test_prompt_keeps_author_statements_and_analyst_reading_apart():
    assert "the authors state" in RULE_BLOCK
    assert "the analysis indicates" in RULE_BLOCK
    assert "separate sections and separate lists" in RULE_BLOCK


def test_prompt_requires_honest_uncertainty_and_no_fabricated_confidence():
    assert "could not be established from the available evidence" in RULE_BLOCK
    assert "Never fabricate confidence or statistics" in RULE_BLOCK
    assert "possible discrepancy between" in RULE_BLOCK


def test_prompt_requires_omitting_unsupported_sections():
    assert 'Omit any section listed as "NOT supported"' in RULE_BLOCK
    guidance = render_section_guidance(
        context(
            vision_result=None,
            routing_state=routing_state(vision_enabled=False),
        )
    )
    assert "NOT supported" in guidance
    for section_id in SECTION_IDS:
        assert section_id in guidance


def test_contract_header_declares_the_separated_limitation_lists():
    assert '"author_stated"' in OUTPUT_CONTRACT_HEADER
    assert '"analyst_identified"' in OUTPUT_CONTRACT_HEADER
    assert "Never move an item between the two lists" in OUTPUT_CONTRACT_HEADER


def test_contract_header_forbids_verdict_fields():
    assert '"score"' in OUTPUT_CONTRACT_HEADER
    assert '"rank"' in OUTPUT_CONTRACT_HEADER
    assert '"tier"' in OUTPUT_CONTRACT_HEADER


def test_section_expectations_cover_the_required_wording_guidance():
    joined = "\n".join(SECTION_EXPECTATIONS)
    assert "the paper presents X as its primary contribution" in joined
    assert "never fabricate exact numbers" in joined.lower()
    assert "No score, no rank, no tier, no overall label" in joined


def test_context_block_renders_document_routing_and_evidence():
    block = build_context_block(context())
    assert "- page_count: 11" in block
    assert "- analysis_level: advanced" in block
    assert "chunk_001" in block
    assert "figure_001" in block
    assert "claim_evidence_matrix" in block


def test_context_block_states_that_routing_is_upstream():
    block = build_context_block(context())
    assert "you make no routing decision" in block
    assert "router_confidence (upstream, informational only)" in block


def test_source_index_block_is_the_citation_allow_list():
    block = render_source_index_block(context())
    assert "Allowed citation ids" in block
    assert "rag: chunk_001 (pages: 2)" in block
    assert "vision: figure_001 (page: 6)" in block
    assert "analysis: chunk_001" in block


def test_gaps_block_preserves_inherited_and_deterministic_gaps():
    block = render_gaps_block(
        context(
            rag_result=None,
            routing_state=routing_state(rag_enabled=False),
        )
    )
    assert "Per-run variance is not present" in block
    assert "DETERMINISTIC GAPS" in block


def test_disabled_stages_are_declared_in_the_prompt():
    prompt = build_critique_prompt(
        context(
            rag_result=None,
            vision_result=None,
            routing_state=routing_state(rag_enabled=False, vision_enabled=False),
        )
    )
    assert "Retrieval was not enabled" in prompt
    assert "Visual analysis was not enabled" in prompt
    assert "no retrieved passage is available" in prompt


def test_prompt_is_deterministic_for_identical_inputs():
    first = build_critique_prompt(context())
    second = build_critique_prompt(context())
    assert json.dumps(first) == json.dumps(second)


def test_prompt_contains_no_credential_and_no_hard_coded_model():
    prompt = build_critique_prompt(context())
    assert "sk-" not in prompt
    assert "gpt-" not in prompt.lower()