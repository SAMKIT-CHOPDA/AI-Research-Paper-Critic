"""
Regression tests for the critique grounding contract (Phase 7 correctness fix).

These cover the real production failure: the synthesis model added strengths and
limitations that the Analysis stage had never recorded, and omitted sections that
the analysis did support. The tests pin down the intended behaviour:

* an extra strength or limitation is rejected,
* a supported section is required by the generation contract,
* a faithful reword of an existing finding is accepted,
* a critique containing only grounded findings passes.
"""

import pytest

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import build_synthesis_context
from backend.critique_engine.prompts import (
    RULE_BLOCK,
    build_context_block,
    render_section_guidance,
    render_supplied_findings_block,
)
from backend.critique_engine.tests import (
    build_compliant_response,
    routing_state,
    synthetic_analysis_result,
    synthetic_profile,
    synthetic_rag_result,
    synthetic_vision_result,
)
from backend.critique_engine.validator import validate_critique


def context(**overrides):
    payload = {
        "document_profile": synthetic_profile(),
        "routing_state": routing_state(),
        "rag_result": synthetic_rag_result(),
        "vision_result": synthetic_vision_result(),
        "analysis_result": synthetic_analysis_result(),
    }
    payload.update(overrides)
    return build_synthesis_context(config=CritiqueConfig(), **payload)


def codes(response, ctx):
    _, report = validate_critique(response, ctx)
    return [issue.code for issue in report.issues]


def test_an_extra_strength_is_rejected():
    """Test 1: an invented strength must be rejected."""
    ctx = context()
    response = build_compliant_response(ctx)
    response["strengths"].append(
        {
            "statement": (
                "The paper demonstrates that the method generalises across every "
                "domain."
            ),
            "evidence_refs": [],
        }
    )
    assert "unsupported_new_strength" in codes(response, ctx)


def test_an_extra_limitation_is_rejected_in_both_categories():
    """Test 2: an invented limitation must be rejected wherever it is filed."""
    ctx = context()
    invented = "No statistical significance testing or confidence intervals are reported."

    response = build_compliant_response(ctx)
    response["limitations"]["analyst_identified"].append(
        {"statement": invented, "evidence_refs": []}
    )
    assert "unsupported_limitation" in codes(response, ctx)

    response = build_compliant_response(ctx)
    response["limitations"]["author_stated"].append(
        {"statement": invented, "evidence_refs": []}
    )
    assert "unsupported_limitation" in codes(response, ctx)


def test_supported_internal_consistency_section_is_required():
    """Test 3: when analysis supports internal_consistency, it must be produced."""
    ctx = context()
    assert "internal_consistency" in ctx.supported_sections.section_ids

    response = build_compliant_response(ctx)
    response["sections"] = [
        item
        for item in response["sections"]
        if item["section_id"] != "internal_consistency"
    ]
    _, report = validate_critique(response, ctx, require_supported_sections=True)
    assert "missing_supported_section" in [issue.code for issue in report.errors()]


def test_supported_reproducibility_section_is_required():
    """Test 4: when analysis supports reproducibility, it must be produced."""
    ctx = context()
    assert "reproducibility" in ctx.supported_sections.section_ids

    response = build_compliant_response(ctx)
    response["sections"] = [
        item for item in response["sections"] if item["section_id"] != "reproducibility"
    ]
    _, report = validate_critique(response, ctx, require_supported_sections=True)
    assert "missing_supported_section" in [issue.code for issue in report.errors()]


def test_prompt_marks_supported_sections_as_must_produce():
    ctx = context()
    guidance = render_section_guidance(ctx)
    assert "you MUST produce this section" in guidance
    for section_id in ("internal_consistency", "reproducibility"):
        line = next(
            line for line in guidance.split("\n") if line.startswith(f"- {section_id}:")
        )
        assert "MUST produce" in line


def test_a_legitimate_paraphrase_of_a_strength_passes():
    """Test 5: rewording an existing strength keeps its meaning, so it passes."""
    ctx = context()
    response = build_compliant_response(ctx)
    response["strengths"][0]["statement"] = (
        "The approach is described in detail and illustrated by a figure the document "
        "profile confirms exists."
    )
    assert "unsupported_new_strength" not in codes(response, ctx)


def test_a_legitimate_paraphrase_of_a_limitation_passes():
    """Test 6: rewording an existing limitation keeps its meaning, so it passes."""
    ctx = context()
    response = build_compliant_response(ctx)
    response["limitations"]["analyst_identified"][0]["statement"] = (
        "The analysis indicates that training details are not established from the "
        "available evidence for this paper."
    )
    assert "unsupported_limitation" not in codes(response, ctx)


def test_a_grounded_critique_with_every_supported_section_passes():
    """Test 7: only grounded findings, every supported section, no issues."""
    ctx = context()
    response = build_compliant_response(ctx)
    found = codes(response, ctx)
    assert "unsupported_new_strength" not in found
    assert "unsupported_limitation" not in found
    produced = [item["section_id"] for item in response["sections"]]
    for section_id in ctx.supported_sections.section_ids:
        assert section_id in produced


def test_supplied_findings_block_enumerates_the_allow_list():
    ctx = context()
    block = render_supplied_findings_block(ctx)
    assert "authoritative" in block
    assert "STRENGTHS SUPPLIED BY ANALYSIS" in block
    assert "AUTHOR-STATED LIMITATIONS SUPPLIED BY ANALYSIS" in block
    assert "ANALYST-IDENTIFIED LIMITATIONS SUPPLIED BY ANALYSIS" in block
    for strength in ctx.analysis.strengths:
        assert strength in block
    for limitation in ctx.analysis.author_stated_limitations:
        assert limitation in block


def test_rules_forbid_inventing_findings():
    # The rules are line-wrapped, so compare on collapsed whitespace.
    rules = " ".join(RULE_BLOCK.split())
    assert "must not introduce any" in rules.lower()
    assert "Do not convert general knowledge" in rules
    assert "Only include strengths that are directly supported" in rules
    assert "Do not introduce new limitations" in rules


def test_context_block_contains_the_supplied_findings_block():
    ctx = context()
    assert "SUPPLIED FINDINGS (authoritative)" in build_context_block(ctx)