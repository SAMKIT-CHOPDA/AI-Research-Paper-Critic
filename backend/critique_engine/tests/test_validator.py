"""
Offline unit tests for critique validation.

These are the tests that matter most for the project's honesty contract: a
fabricated citation, an impossible page, a merged limitation category, a dropped or
re-classified claim, a numeric score, a ranking field or a leaked credential must
all be rejected, and a clean response must pass untouched.
"""

import json

import pytest

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import build_synthesis_context
from backend.critique_engine.validator import (
    CritiqueValidationError,
    detect_score_like_text,
    detect_secrets,
    detect_unsupported_verdicts,
    extract_json_payload,
    extract_page_citations,
    find_forbidden_keys,
    is_uncertain_text,
    validate_critique,
    validate_model_response,
)
from backend.critique_engine.tests import (
    build_compliant_response,
    compliant_response_json,
    routing_state,
    synthetic_analysis_result,
    synthetic_profile,
    synthetic_rag_result,
    synthetic_vision_result,
)

SECRET = "sk-validator-test-secret-123456"


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


def candidate(ctx):
    return build_compliant_response(ctx)


def codes(response, ctx):
    _, report = validate_critique(response, ctx)
    return [issue.code for issue in report.issues]


def section(response, section_id):
    return next(item for item in response["sections"] if item["section_id"] == section_id)


# --- helpers -----------------------------------------------------------------


def test_extract_json_payload_accepts_fenced_objects():
    assert extract_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_payload('Here you go: {"a": 1} done') == {"a": 1}


def test_extract_json_payload_rejects_unparseable_text():
    with pytest.raises(CritiqueValidationError):
        extract_json_payload("no json at all")
    with pytest.raises(CritiqueValidationError):
        extract_json_payload("   ")


def test_score_detection_catches_verdict_numbers():
    assert detect_score_like_text("Overall this is 7/10.")
    assert detect_score_like_text("score: 82")
    assert detect_score_like_text("grade of B")
    assert not detect_score_like_text("The reported BLEU value improved.")


def test_verdict_label_detection():
    assert detect_unsupported_verdicts("This is an excellent paper.")
    assert detect_unsupported_verdicts("a top-tier paper")
    assert not detect_unsupported_verdicts("The evidence supports the design.")


def test_secret_detection():
    assert detect_secrets(f"the key is {SECRET}")
    assert detect_secrets("Authorization: Bearer abcdefghijklmnop")
    assert not detect_secrets("The paper reports a comparison.")


def test_forbidden_key_detection_walks_nested_objects():
    found = find_forbidden_keys({"overall": {"score": 8, "label": "good"}, "tier": "A"})
    keys = {key for _, key in found}
    assert keys == {"score", "tier"}


def test_page_citation_extraction():
    assert extract_page_citations("as shown (p. 8)") == [8]
    assert extract_page_citations("pages 3-5") == [3, 5]


def test_uncertainty_marker_detection():
    assert is_uncertain_text("This could not be established from the evidence.")
    assert not is_uncertain_text("The paper reports two datasets.")


# --- happy path --------------------------------------------------------------


def test_a_compliant_response_validates_without_issues():
    ctx = context()
    parsed, report = validate_critique(candidate(ctx), ctx)
    assert report.valid
    assert report.issues == []
    assert parsed.unverified_evidence_refs == 0
    assert parsed.payload.sections
    assert parsed.payload.overall_assessment.content


def test_valid_source_references_are_verified_and_real():
    ctx = context()
    parsed, report = validate_critique(candidate(ctx), ctx)
    assert report.valid
    known = {entry["source_id"] for entry in ctx.source_index.values()}
    refs = parsed.payload.sections[0].evidence_refs
    assert refs
    assert all(ref.verified for ref in refs)
    assert all(ref.source_id in known for ref in refs)


def test_a_real_page_citation_matching_the_source_is_accepted():
    ctx = context()
    response = candidate(ctx)
    target = section(response, "methodology")
    target["evidence_refs"] = [{"source_type": "vision", "source_id": "figure_001"}]
    target["content"] = (
        "The figure illustrates the described design (p. 6); the observation notes "
        "limited label detail."
    )
    _, report = validate_critique(response, ctx)
    assert report.valid, [issue.code for issue in report.issues]


# --- structure ---------------------------------------------------------------


def test_missing_required_section_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"] = [
        item
        for item in response["sections"]
        if item["section_id"] != "executive_summary"
    ]
    assert "missing_required_section" in codes(response, ctx)


def test_missing_supported_section_is_rejected_by_default():
    ctx = context()
    response = candidate(ctx)
    response["sections"] = [
        item for item in response["sections"] if item["section_id"] != "methodology"
    ]
    assert "missing_supported_section" in codes(response, ctx)


def test_missing_supported_section_can_be_a_warning_by_policy():
    ctx = context()
    response = candidate(ctx)
    response["sections"] = [
        item for item in response["sections"] if item["section_id"] != "methodology"
    ]
    _, report = validate_critique(response, ctx, require_supported_sections=False)
    assert report.valid
    assert any(
        issue.code == "missing_supported_section" for issue in report.warnings()
    )


def test_invalid_section_id_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"].append(
        {"section_id": "peer_rebuttal", "content": "text", "evidence_refs": []}
    )
    assert "invalid_section_id" in codes(response, ctx)


def test_empty_section_content_is_rejected():
    ctx = context()
    response = candidate(ctx)
    section(response, "strengths")["content"] = "   "
    assert "empty_section" in codes(response, ctx)


def test_duplicate_section_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"].append(
        json.loads(json.dumps(section(response, "executive_summary")))
    )
    assert "duplicate_section" in codes(response, ctx)


# --- provenance --------------------------------------------------------------


def test_fabricated_chunk_id_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "rag", "source_id": "chunk_999"}
    ]
    assert "unresolved_source_id" in codes(response, ctx)


def test_fabricated_asset_id_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "vision", "source_id": "figure_999"}
    ]
    assert "unresolved_source_id" in codes(response, ctx)


def test_fabricated_table_id_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["claim_evidence_summary"][0]["evidence_refs"] = [
        {"source_type": "document", "source_id": "table_999"}
    ]
    assert "unresolved_source_id" in codes(response, ctx)


def test_unknown_source_type_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "internet", "source_id": "chunk_001"}
    ]
    assert "unresolved_source_id" in codes(response, ctx)


def test_unverified_references_are_counted_and_never_trusted():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "rag", "source_id": "chunk_999"}
    ]
    parsed, report = validate_critique(response, ctx)
    assert report.unverified_evidence_refs == 1
    assert parsed.unverified_evidence_refs == 1
    assert any(gap.category == "unverified_reference" for gap in parsed.notes)


def test_a_bare_source_id_is_accepted_only_when_unambiguous():
    ctx = context()
    response = candidate(ctx)
    # chunk_001 exists both as a rag chunk and as an analysis-level reference, so
    # the bare form is ambiguous and must not be silently resolved.
    response["sections"][0]["evidence_refs"] = ["chunk_001"]
    assert "unresolved_source_id" in codes(response, ctx)


def test_a_real_id_with_a_mistyped_source_type_resolves_from_real_metadata():
    """
    Live models pair real ids with the wrong type ("analysis:chunk_005").

    The id is what carries the fabrication risk, so the reference resolves and the
    stored type comes from the allow-list, never from the model.
    """
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "analysis", "source_id": "chunk_001"}
    ]
    parsed, report = validate_critique(response, ctx)
    assert "unresolved_source_id" not in [issue.code for issue in report.issues]
    ref = parsed.payload.sections[0].evidence_refs[0]
    assert ref.source_id == "chunk_001"
    # The id exists under two real types; the claimed one is honoured.
    assert ref.source_type in {"rag", "analysis"}
    assert ref.verified is True


def test_a_mistyped_type_falls_back_to_the_single_real_type():
    """table_001 exists only as a document source, so a 'vision' claim resolves."""
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "vision", "source_id": "table_001"}
    ]
    parsed, report = validate_critique(response, ctx)
    assert "unresolved_source_id" not in [issue.code for issue in report.issues]
    ref = parsed.payload.sections[0].evidence_refs[0]
    assert ref.source_type == "document"
    assert ref.verified is True


def test_an_unknown_id_is_still_rejected_even_with_a_valid_type():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "rag", "source_id": "chunk_404"}
    ]
    assert "unresolved_source_id" in codes(response, ctx)


def test_page_citation_beyond_the_document_is_rejected():
    ctx = context()
    response = candidate(ctx)
    section(response, "methodology")["content"] = (
        "The architecture is described on p. 99."
    )
    assert "fabricated_page" in codes(response, ctx)


def test_page_citation_outside_the_cited_evidence_is_rejected():
    ctx = context()
    response = candidate(ctx)
    target = section(response, "methodology")
    target["evidence_refs"] = [{"source_type": "rag", "source_id": "chunk_001"}]
    target["content"] = "The architecture is described (p. 8) in the retrieved text."
    assert "unverified_page_citation" in codes(response, ctx)


# --- scores, ranks, secrets --------------------------------------------------


def test_numeric_score_in_section_content_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["content"] += " Overall this paper is 8/10."
    assert "numeric_paper_score" in codes(response, ctx)


def test_numeric_score_in_the_overall_assessment_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["overall_assessment"]["content"] += " We rate this work a score of 9."
    assert "numeric_paper_score" in codes(response, ctx)


def test_unsupported_overall_label_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["overall_assessment"]["content"] += " This is an excellent paper."
    assert "unsupported_verdict_label" in codes(response, ctx)


def test_a_reported_metric_value_is_not_treated_as_a_paper_score():
    """
    "BLEU score of 28.4" is a result the evidence supports, not a verdict.

    Rejecting it would contradict the rule that upstream numbers are preserved
    rather than fabricated.
    """
    ctx = context()
    response = candidate(ctx)
    target = section(response, "results_and_evidence")
    target["evidence_refs"] = [{"source_type": "rag", "source_id": "chunk_002"}]
    target["content"] = (
        "The paper reports a BLEU score of 28.4 on the benchmark task (p. 8)."
    )
    assert "numeric_paper_score" not in codes(response, ctx)


def test_an_overall_verdict_number_is_still_rejected_in_any_context():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["content"] += " Overall this paper is 8/10."
    assert "numeric_paper_score" in codes(response, ctx)


def test_ranking_and_tiering_fields_are_rejected():
    ctx = context()
    response = candidate(ctx)
    response["paper_score"] = 8
    response["overall_assessment"]["tier"] = "A"
    assert codes(response, ctx).count("forbidden_field") >= 2


def test_model_invented_confidence_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["confidence"] = 0.97
    assert "forbidden_field" in codes(response, ctx)


def test_credential_in_the_output_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["sections"][0]["content"] += f" The key is {SECRET}."
    assert "credential_leak" in codes(response, ctx)


# --- claims, limitations, questions -----------------------------------------


def test_dropping_a_claim_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["claim_evidence_summary"] = []
    assert "claim_dropped" in codes(response, ctx)


def test_changing_a_claim_classification_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["claim_evidence_summary"][0]["status"] = "supported"
    assert "claim_status_changed" in codes(response, ctx)


def test_adding_a_new_claim_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["claim_evidence_summary"].append(
        {
            "claim": "The method generalizes to medical imaging.",
            "status": "supported",
            "assessment": "Assumed without evidence.",
            "evidence_refs": [],
        }
    )
    assert "unsupported_new_claim" in codes(response, ctx)


def test_invalid_claim_status_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["claim_evidence_summary"][0]["status"] = "mostly_true"
    assert "invalid_claim_status" in codes(response, ctx)


def test_analyst_limitation_cannot_be_filed_as_author_stated():
    """An analyst observation is relocated back, never left as an author claim."""
    ctx = context()
    response = candidate(ctx)
    analyst = response["limitations"]["analyst_identified"][0]["statement"]
    response["limitations"]["author_stated"].append(
        {"statement": analyst, "evidence_refs": []}
    )
    parsed, report = validate_critique(response, ctx)

    assert report.valid
    assert all(
        entry.statement != analyst
        for entry in parsed.payload.limitations.author_stated
    )
    assert any(
        item.statement == analyst
        for item in parsed.payload.limitations.analyst_identified
    )


def test_an_invented_limitation_is_still_rejected():
    """A limitation the analysis never recorded is still a fabrication."""
    ctx = context()
    response = candidate(ctx)
    response["limitations"]["analyst_identified"].append(
        {
            "statement": "The authors concealed a failed replication.",
            "evidence_refs": [],
        }
    )


def test_a_paraphrased_strength_is_still_traceable_to_the_analysis():
    """The synthesis stage rewrites wording; a faithful paraphrase is accepted."""
    ctx = context()
    response = candidate(ctx)
    response["strengths"][0]["statement"] = (
        "The architecture is both described in the retrieved text and shown in a "
        "figure the profile confirms."
    )
    assert "unsupported_new_strength" not in codes(response, ctx)


def test_an_overall_assessment_written_only_as_a_section_is_recovered():
    ctx = context()
    response = candidate(ctx)
    section_text = next(
        item["content"]
        for item in response["sections"]
        if item["section_id"] == "overall_assessment"
    )
    response["overall_assessment"] = {"content": "  "}
    parsed, report = validate_critique(response, ctx)
    assert report.valid
    assert parsed.payload.overall_assessment.content == section_text


def test_author_limitation_cannot_be_filed_as_analyst_identified():
    """An author-stated limitation is relocated back to the author category."""
    ctx = context()
    response = candidate(ctx)
    author = response["limitations"]["author_stated"][0]["statement"]
    response["limitations"]["analyst_identified"].append(
        {"statement": author, "evidence_refs": []}
    )
    parsed, report = validate_critique(response, ctx)

    assert report.valid
    assert all(
        entry.statement != author
        for entry in parsed.payload.limitations.analyst_identified
    )
    assert any(
        entry.statement == author
        for entry in parsed.payload.limitations.author_stated
    )


def test_invented_strength_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["strengths"].append(
        {
            "statement": "The paper is the first to solve this problem.",
            "evidence_refs": [],
        }
    )
    assert "unsupported_new_strength" in codes(response, ctx)


def test_open_question_must_be_phrased_as_a_question():
    ctx = context()
    response = candidate(ctx)
    response["open_questions"][0]["question"] = (
        "The method is insensitive to random seeds."
    )
    assert "open_question_not_a_question" in codes(response, ctx)


def test_missing_overall_assessment_is_rejected():
    ctx = context()
    response = candidate(ctx)
    response["overall_assessment"] = {"content": "  "}
    response["sections"] = [
        item
        for item in response["sections"]
        if item["section_id"] != "overall_assessment"
    ]
    assert "missing_overall_assessment" in codes(response, ctx)


def test_deterministic_gaps_are_reinjected_when_the_model_omits_them():
    ctx = context(rag_result=None, routing_state=routing_state(rag_enabled=False))
    response = candidate(ctx)
    response["evidence_gaps"] = []
    parsed, report = validate_critique(response, ctx)
    assert report.valid
    assert any(gap.category == "rag_disabled" for gap in parsed.payload.evidence_gaps)


# --- entry points ------------------------------------------------------------


def test_validate_model_response_returns_the_parsed_payload():
    ctx = context()
    parsed, report = validate_model_response(compliant_response_json(ctx), ctx)
    assert report.valid
    assert parsed.payload.sections


def test_validate_model_response_raises_with_a_report_on_failure():
    ctx = context()
    with pytest.raises(CritiqueValidationError) as excinfo:
        validate_model_response('{"sections": []}', ctx)
    assert excinfo.value.report is not None
    assert excinfo.value.report.valid is False


def test_validate_model_response_rejects_unparseable_text():
    ctx = context()
    with pytest.raises(CritiqueValidationError) as excinfo:
        validate_model_response("I cannot comply with that request.", ctx)
    assert excinfo.value.report.issues[0].code == "unparseable_response"