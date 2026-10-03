"""
Offline unit tests for analysis parsing, validation and provenance verification.

Verifies:
- a contract-compliant response becomes a fully populated ParsedAnalysis
- malformed JSON, missing contract keys, empty required fields, invalid enums and
  invalid confidence values are rejected (no partial acceptance)
- references are verified against the evidence bundle, unknown ids stay visible but
  unverified and a wrong source_type is corrected from the bundle
- 'supported' without provenance is downgraded and the downgrade is recorded
- confirmatory language and score-like text are recorded rather than propagated
- a malformed response triggers no repair call (the client is called exactly once)
"""

import json

import pytest

from backend.analysis_agent.analyzer import (
    DIMENSION_KEYS,
    REQUIRED_PAYLOAD_KEYS,
    AnalysisError,
    analyze_evidence,
    count_downgraded_statuses,
    count_unverified_evidence_refs,
    detect_score_like_text,
    extract_json_payload,
    flag_overclaiming,
    parse_analysis,
)
from backend.analysis_agent.context_builder import build_evidence_bundle
from backend.analysis_agent.llm_client import AnalysisClientError, BaseAnalysisClient
from backend.analysis_agent.schemas import AnalysisPayload, AnalysisResult

MODEL = "mock-model"


def profile() -> dict:
    return {
        "filename": "synthetic.pdf",
        "page_count": 11,
        "word_count": 4990,
        "text_length": 36309,
        "metadata": {"title": "Synthetic Paper"},
        "content_characteristics": {"has_methodology": True},
        "sections": [
            {"title": "Abstract", "page": 1, "level": 1},
            {"title": "6 Results", "number": "6", "page": 8, "level": 1},
        ],
        "figures": {
            "count": 1,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 3,
                    "caption": "Figure 1.",
                    "confidence": 0.9,
                }
            ],
        },
        "tables": {
            "count": 1,
            "confirmed_tables": [
                {
                    "id": "table_1",
                    "table_number": "1",
                    "page": 8,
                    "caption": "Table 1.",
                    "rows": 11,
                    "columns": 5,
                    "confidence": 0.9,
                }
            ],
        },
        "equations": {
            "count": 1,
            "confirmed_equations": [
                {
                    "id": "eq_1",
                    "page": 4,
                    "equation_number": "1",
                    "representation": "x = y",
                    "confidence": 1.0,
                }
            ],
        },
        "references": {
            "has_references": True,
            "count": 32,
            "start_page": 10,
            "confidence": 1.0,
            "is_sequential": True,
        },
    }


def rag() -> dict:
    return {
        "agent": "rag",
        "enabled": True,
        "results": [
            {
                "chunk_id": "chunk_001",
                "score": 0.81,
                "page_start": 8,
                "page_end": 9,
                "section": "6 Results",
                "text": "The model reaches a reported score on the benchmark.",
                "word_count": 9,
            }
        ],
    }


def vision() -> dict:
    return {
        "agent": "vision",
        "enabled": True,
        "results": [
            {
                "asset_id": "figure_001",
                "asset_type": "figure",
                "page_number": 3,
                "section": "6 Results",
                "caption": "Figure 1.",
                "observation": "Two curves are visible.",
                "interpretation": "The curves appear to converge.",
                "uncertainties": ["no error bars"],
                "caption_consistency": {"status": "consistent", "explanation": "ok"},
                "numeric_authority": "vision_interpretation_only",
            }
        ],
    }


def bundle(with_upstream: bool = True):
    if not with_upstream:
        return build_evidence_bundle(profile())
    return build_evidence_bundle(profile(), rag(), vision())


def ref(source_type="text_chunk", source_id="chunk_001", pages=None) -> dict:
    payload = {"source_type": source_type, "source_id": source_id}
    if pages is not None:
        payload["pages"] = pages
    return payload


def dimension(name: str, **overrides) -> dict:
    payload = {
        "dimension": name,
        "summary": f"Evidence-grounded reading of {name}.",
        "evidence_status": "supported",
        "findings": [
            {
                "statement": f"A finding about {name}.",
                "evidence_status": "supported",
                "evidence_refs": [ref()],
                "confidence": 0.6,
                "reasoning": "The retrieved passage states it directly.",
                "caveats": [],
            }
        ],
        "evidence_refs": [ref(source_type="section", source_id="section_002", pages=[8])],
        "confidence": 0.7,
        "uncertainties": [],
    }
    payload.update(overrides)
    return payload


def payload_dict(**overrides) -> dict:
    payload = {key: dimension(key) for key in DIMENSION_KEYS}
    payload["claim_evidence_matrix"] = {
        "items": [
            {
                "claim": "The method improves the benchmark metric.",
                "claim_source": "paper",
                "status": "partially_supported",
                "supporting_evidence": [ref()],
                "contradicting_evidence": [],
                "assessment": "Supported for one setting only.",
                "missing_evidence": ["Full ablation table"],
                "confidence": 0.5,
            }
        ],
        "notes": "One claim assessed.",
    }
    payload["strengths"] = [
        {
            "statement": "The architecture is described in a figure.",
            "dimension": "methodology",
            "evidence_status": "supported",
            "evidence_refs": [ref(source_type="figure", source_id="figure_1", pages=[3])],
            "rationale": "A confirmed figure exists.",
            "confidence": 0.5,
        }
    ]
    payload["weaknesses"] = [
        {
            "statement": "Numeric details are not in the supplied evidence.",
            "dimension": "results",
            "evidence_status": "insufficient_evidence",
            "evidence_refs": [],
            "rationale": "No machine-readable values were provided.",
            "confidence": None,
        }
    ]
    payload["limitations"] = {
        "author_stated": [
            {
                "statement": "The paper states a scope restriction.",
                "evidence_status": "supported",
                "evidence_refs": [ref()],
                "confidence": 0.4,
                "reasoning": "Stated in the retrieved passage.",
                "caveats": [],
            }
        ],
        "analyst_identified": [
            {
                "statement": "The evaluation scope appears narrow.",
                "evidence_status": "unclear",
                "evidence_refs": [],
                "confidence": None,
                "reasoning": "Inferred from the section list.",
                "caveats": [],
            }
        ],
    }
    payload["open_questions"] = [
        {
            "question": "How stable are the gains across seeds?",
            "why_it_matters": "It decides whether the gains are robust.",
            "related_dimension": "results",
            "evidence_refs": [ref(source_type="table", source_id="table_1", pages=[8])],
        }
    ]
    payload["evidence_gaps"] = [
        {
            "gap": "No machine-readable numeric values were supplied.",
            "category": "numeric_values_unavailable",
            "impact": "Exact numbers cannot be verified.",
            "affected_dimensions": ["results", "metrics"],
        }
    ]
    payload["overall_assessment"] = {
        "summary": "The evidence supports the structural claims; numeric verification is missing.",
        "evidence_status": "partially_supported",
        "confidence": 0.5,
        "key_basis": [ref(source_type="document_profile", source_id="document_profile")],
        "uncertainties": ["No machine-readable numbers were available."],
    }
    payload.update(overrides)
    return payload


def render(**overrides) -> str:
    return json.dumps(payload_dict(**overrides))


class MockAnalysisClient(BaseAnalysisClient):
    """Deterministic offline model standing in for an analysis provider."""

    def __init__(self, response: str = "{}", error: Exception = None):
        self.response = response
        self.error = error
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if self.error is not None:
            raise self.error
        return self.response


def test_valid_response_is_fully_parsed():
    parsed = parse_analysis(render(), bundle())

    payload = parsed.payload
    for key in DIMENSION_KEYS:
        dimension_analysis = getattr(payload, key)
        assert dimension_analysis.dimension == key
        assert dimension_analysis.summary
    assert len(payload.claim_evidence_matrix.items) == 1
    assert payload.claim_evidence_matrix.items[0].claim_source == "paper"
    assert len(payload.strengths) == 1
    assert len(payload.weaknesses) == 1
    assert len(payload.limitations.author_stated) == 1
    assert len(payload.limitations.analyst_identified) == 1
    assert payload.open_questions[0].related_dimension == "results"
    assert payload.evidence_gaps[0].category == "numeric_values_unavailable"
    assert payload.overall_assessment.confidence == 0.5

    assert parsed.notes == []
    assert parsed.unverified_evidence_refs == 0
    assert parsed.downgraded_statuses == 0
    assert parsed.overclaim_flags == []


def test_references_are_verified_against_the_evidence_bundle():
    parsed = parse_analysis(render(), bundle())

    dimension_analysis = parsed.payload.methodology
    section_ref = dimension_analysis.evidence_refs[0]
    assert section_ref.source_id == "section_002"
    assert section_ref.verified is True
    assert section_ref.pages == [8]
    assert section_ref.source_type == "section"

    # page-less reference: pages are filled in from the bundle index
    finding_ref = dimension_analysis.findings[0].evidence_refs[0]
    assert finding_ref.source_id == "chunk_001"
    assert finding_ref.verified is True
    assert finding_ref.pages == [8, 9]


def test_wrong_source_type_is_corrected_from_the_bundle():
    parsed = parse_analysis(
        render(methodology=dimension("methodology", evidence_refs=[ref(source_type="section")])),
        bundle(),
    )

    assert parsed.payload.methodology.evidence_refs[0].source_type == "text_chunk"


def test_unknown_reference_is_kept_but_marked_unverified_and_recorded():
    parsed = parse_analysis(
        render(
            methodology=dimension(
                "methodology", evidence_refs=[ref(source_id="chunk_999")]
            )
        ),
        bundle(),
    )

    unknown = parsed.payload.methodology.evidence_refs[0]
    assert unknown.source_id == "chunk_999"
    assert unknown.verified is False
    assert parsed.unverified_evidence_refs == 1
    note = next(n for n in parsed.notes if n.category == "unverified_reference")
    assert note.source == "analyzer_validation"
    assert "1 cited reference(s)" in note.gap


def test_unknown_reference_without_source_type_is_rejected():
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(
                methodology=dimension(
                    "methodology",
                    evidence_refs=[{"source_id": "chunk_999"}],
                )
            ),
            bundle(),
        )
    assert "source_type" in str(excinfo.value)


def test_a_mistyped_type_on_a_real_id_is_corrected_from_the_bundle():
    """
    Production models abbreviate or improvise the type token.

    A real id keeps the bundle's own type, so the reference stays verified and can
    never be attributed to the wrong kind of evidence. An invalid type is only
    fatal when the id itself is unknown (see the test above).
    """
    for improvised in ("twitter_post", "all", "entire_document", "chunk"):
        parsed = parse_analysis(
            render(
                methodology=dimension(
                    "methodology",
                    evidence_refs=[ref(source_type=improvised)],
                )
            ),
            bundle(),
        )
        record = parsed.payload.methodology.evidence_refs[0]
        assert record.source_type == "text_chunk"
        assert record.verified is True


def test_invalid_reference_source_type_is_rejected():
    """
    An invalid type is fatal when the id is not a known source.

    There the model is the only source of the type, so nothing can be verified;
    when the id does exist the type comes from the bundle instead.
    """
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(
                methodology=dimension(
                    "methodology",
                    evidence_refs=[
                        {"source_id": "chunk_404", "source_type": "twitter_post"}
                    ],
                )
            ),
            bundle(),
        )
    assert "Invalid source_type" in str(excinfo.value)


def test_missing_contract_key_is_rejected():
    payload = payload_dict()
    del payload["results"]

    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(json.dumps(payload), bundle())
    assert "results" in str(excinfo.value)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"summary": ""}, "summary"),
        ({"summary": "   "}, "summary"),
    ],
)
def test_dimension_without_summary_is_rejected(override, message):
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(methodology=dimension("methodology", **override)), bundle()
        )
    assert message in str(excinfo.value)


def test_dimension_without_evidence_status_is_rejected():
    payload = dimension("methodology")
    del payload["evidence_status"]

    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(render(methodology=payload), bundle())
    assert "evidence_status" in str(excinfo.value)


def test_invalid_evidence_status_is_rejected():
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(methodology=dimension("methodology", evidence_status="probably")),
            bundle(),
        )
    assert "Invalid evidence_status" in str(excinfo.value)


@pytest.mark.parametrize("confidence", [1.5, -0.2, "high", True, [0.5]])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(methodology=dimension("methodology", confidence=confidence)),
            bundle(),
        )
    assert "confidence" in str(excinfo.value)


def test_missing_confidence_is_never_invented():
    parsed = parse_analysis(
        render(methodology=dimension("methodology", confidence=None)), bundle()
    )
    assert parsed.payload.methodology.confidence is None
    assert parsed.payload.weaknesses[0].confidence is None


def test_supported_status_without_references_is_downgraded_and_recorded():
    payload = payload_dict()
    payload["methodology"]["evidence_refs"] = []
    payload["methodology"]["findings"][0]["evidence_refs"] = []
    payload["strengths"][0]["evidence_refs"] = []
    payload["claim_evidence_matrix"]["items"][0]["status"] = "supported"
    payload["claim_evidence_matrix"]["items"][0]["supporting_evidence"] = []
    payload["overall_assessment"]["evidence_status"] = "supported"
    payload["overall_assessment"]["key_basis"] = []

    parsed = parse_analysis(json.dumps(payload), bundle())

    assert parsed.payload.methodology.evidence_status == "unclear"
    assert parsed.payload.methodology.findings[0].evidence_status == "unclear"
    assert parsed.payload.strengths[0].evidence_status == "unclear"
    assert parsed.payload.claim_evidence_matrix.items[0].status == "unclear"
    assert parsed.payload.overall_assessment.evidence_status == "unclear"
    assert parsed.downgraded_statuses == 5

    note = next(
        n
        for n in parsed.notes
        if n.category == "unverified_reference" and "downgraded" in n.gap
    )
    assert "5 statement(s)" in note.gap
    assert "downgraded to 'unclear'" in parsed.payload.methodology.uncertainties[0]


def test_findings_without_a_statement_are_rejected():
    payload = dimension("methodology")
    del payload["findings"][0]["statement"]

    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(render(methodology=payload), bundle())
    assert "statement" in str(excinfo.value)


def test_confirmatory_language_is_flagged_as_a_caveat():
    payload = dimension("methodology")
    payload["findings"][0]["statement"] = "The experiment proves that the method wins."
    parsed = parse_analysis(render(methodology=payload), bundle())

    assert parsed.overclaim_flags == ["proves"]
    caveats = parsed.payload.methodology.findings[0].caveats
    assert any(
        "Confirmatory language detected ('proves')" in caveat for caveat in caveats
    )
    note = next(n for n in parsed.notes if n.category == "model_uncertainty")
    assert "proves" in note.gap


def test_score_like_text_is_recorded_and_no_score_field_exists():
    payload = payload_dict()
    payload["overall_assessment"]["summary"] = "Overall the paper deserves a score of 8/10."

    parsed = parse_analysis(json.dumps(payload), bundle())

    note = next(n for n in parsed.notes if n.category == "no_numeric_paper_score")
    assert "8/10" in note.gap
    assert note.source == "analyzer_validation"
    assert "score" not in AnalysisPayload.model_fields
    assert "score" not in AnalysisResult.model_fields


def test_invalid_gap_category_is_rejected():
    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(
            render(
                evidence_gaps=[
                    {"gap": "something missing", "category": "vibes", "impact": "x"}
                ]
            ),
            bundle(),
        )
    assert "Invalid gap category" in str(excinfo.value)


def test_invalid_claim_source_is_rejected():
    payload = payload_dict()
    payload["claim_evidence_matrix"]["items"][0]["claim_source"] = "trust me"

    with pytest.raises(AnalysisError) as excinfo:
        parse_analysis(json.dumps(payload), bundle())
    assert "Invalid claim_source" in str(excinfo.value)


def test_matrix_and_limitations_must_be_objects():
    with pytest.raises(AnalysisError):
        parse_analysis(render(claim_evidence_matrix=["not", "an", "object"]), bundle())
    with pytest.raises(AnalysisError):
        parse_analysis(render(limitations="not an object"), bundle())


def test_limitations_split_is_preserved():
    parsed = parse_analysis(render(), bundle())

    assert parsed.payload.limitations.author_stated[0].statement.startswith("The paper")
    assert parsed.payload.limitations.analyst_identified[0].statement.startswith(
        "The evaluation"
    )


def test_extract_json_payload_accepts_fences_and_preamble():
    fenced = 'Here is the analysis:\n```json\n{"a": 1}\n```\n'
    assert extract_json_payload(fenced) == {"a": 1}
    assert extract_json_payload('prefix {"b": 2} suffix') == {"b": 2}

    with pytest.raises(AnalysisError):
        extract_json_payload("")
    with pytest.raises(AnalysisError):
        extract_json_payload("no json object at all")


def test_helpers_detect_overclaiming_and_score_like_text():
    assert flag_overclaiming("The result proves the hypothesis.") == ["proves"]
    assert flag_overclaiming("This is proof that the method works.") == ["proof that"]
    # Word-boundary matching: ordinary words containing a phrase are not flagged.
    assert flag_overclaiming("The method improves accuracy.") == []
    assert flag_overclaiming("nothing to see here") == []
    assert detect_score_like_text("score of 7", "rating: 9", "8/10", "no score") == [
        "score of 7",
        "rating: 9",
        "8/10",
    ]
    assert detect_score_like_text("accuracy of 28.4 BLEU") == []


def test_analyze_evidence_makes_exactly_one_model_call():
    client = MockAnalysisClient(response=render())

    parsed = analyze_evidence(bundle(), client, MODEL)

    assert len(client.calls) == 1
    assert client.calls[0]["model"] == MODEL
    assert "EVIDENCE BUNDLE" in client.calls[0]["prompt"]
    assert "Citable source ids" in client.calls[0]["prompt"]
    assert parsed.payload.results.dimension == "results"


def test_analyze_evidence_does_not_repair_a_malformed_response():
    client = MockAnalysisClient(response="I could not comply.")

    with pytest.raises(AnalysisError):
        analyze_evidence(bundle(), client, MODEL)

    assert len(client.calls) == 1


def test_analyze_evidence_propagates_client_failures_without_retrying():
    client = MockAnalysisClient(error=AnalysisClientError("provider exploded"))

    with pytest.raises(AnalysisClientError):
        analyze_evidence(bundle(), client, MODEL)

    assert len(client.calls) == 1


def test_near_miss_source_types_are_normalized_to_contract_tokens():
    """
    Production models abbreviate the contract tokens ("chunk", "asset").

    Those near misses are normalized; anything not in the alias table is still
    rejected, so the fail-closed validation is preserved.
    """
    from backend.analysis_agent.analyzer import normalize_source_type

    assert normalize_source_type("chunk") == "text_chunk"
    assert normalize_source_type("Chunk") == "text_chunk"
    assert normalize_source_type("text-chunk") == "text_chunk"
    assert normalize_source_type("asset") == "visual_asset"
    assert normalize_source_type("image") == "visual_asset"
    assert normalize_source_type("text_chunk") == "text_chunk"
    assert normalize_source_type("chunk_999") == "chunk_999"
    assert normalize_source_type("hallucinated") == "hallucinated"


def test_abbreviated_source_type_in_a_response_is_accepted_and_verified():
    """A response using 'chunk' parses, and the reference is still verified."""
    abbreviated = ref(source_type="chunk", source_id="chunk_001", pages=[2])
    response = render(results=dimension("results", evidence_refs=[abbreviated]))

    parsed = analyze_evidence(bundle(), MockAnalysisClient(response), MODEL)

    refs = parsed.payload.results.evidence_refs
    assert refs[0].source_type == "text_chunk"
    assert refs[0].verified is True


def test_reference_without_a_source_id_is_dropped_instead_of_failing():
    """
    A reference the model could not identify carries no provenance.

    It is dropped so it can never support a statement; the surrounding evaluation
    survives, and any statement that relied on it is downgraded and recorded.
    """
    unusable = {"source_type": "text_chunk", "pages": [3]}
    response = render(
        methodology=dimension("methodology", evidence_refs=[unusable, ref()])
    )

    parsed = parse_analysis(response, bundle())

    refs = parsed.payload.methodology.evidence_refs
    assert [item.source_id for item in refs] == ["chunk_001"]
    assert all(item.verified for item in refs)
