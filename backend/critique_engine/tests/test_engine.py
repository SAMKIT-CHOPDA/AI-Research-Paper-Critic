"""
Offline unit tests for the Critique Engine orchestrator.

Verifies the contract of one execution: the model comes from the level JEV already
chose, exactly one call is made, RAG/Vision may be disabled without breaking the
report, every failure mode becomes a structured result, and no output path can
produce a paper score or leak a credential.
"""

import json

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import build_synthesis_context
from backend.critique_engine.engine import CritiqueEngine
from backend.critique_engine.llm_client import CritiqueClientError
from backend.critique_engine.tests import (
    MODEL_MAP,
    MockCritiqueClient,
    compliant_response_json,
    routing_state,
    synthetic_analysis_result,
    synthetic_profile,
    synthetic_rag_result,
    synthetic_vision_result,
)
from backend.jev_router.config import ModelLevel

SECRET = "sk-engine-test-secret-987654"


def engine_config(**overrides):
    payload = {"level_models": dict(MODEL_MAP)}
    payload.update(overrides)
    return CritiqueConfig(**payload)


def build_context(**overrides):
    """Build the context the engine will assemble for the same inputs."""
    payload = {
        "document_profile": synthetic_profile(),
        "routing_state": routing_state(),
        "rag_result": synthetic_rag_result(),
        "vision_result": synthetic_vision_result(),
        "analysis_result": synthetic_analysis_result(),
    }
    payload.update(overrides)
    return build_synthesis_context(config=engine_config(), **payload)


def run(client, config=None, **overrides):
    payload = {
        "document_profile": synthetic_profile(),
        "routing_state": routing_state(),
        "rag_result": synthetic_rag_result(),
        "vision_result": synthetic_vision_result(),
        "analysis_result": synthetic_analysis_result(),
    }
    payload.update(overrides)
    engine = CritiqueEngine(
        critique_client=client, config=config or engine_config()
    )
    return engine.generate_critique(**payload)


def compliant_run(client, **overrides):
    """Run with a response built from the context the engine will assemble."""
    client.response = compliant_response_json(build_context(**overrides))
    return run(client, **overrides)


def test_successful_synthesis_returns_the_full_report():
    client = MockCritiqueClient()
    result = compliant_run(client)
    assert result["status"] == "completed"
    assert result["engine"] == "critique"
    assert result["version"] == "1.0"
    assert len(client.calls) == 1
    assert result["model"] == "mock-critique-advanced"
    assert result["level"] == "advanced"
    assert result["provider"] == "mock-critique"
    section_ids = {item["section_id"] for item in result["sections"]}
    assert {"executive_summary", "overall_assessment"} <= section_ids
    assert result["overall_assessment"]["content"]
    assert result["claim_evidence_summary"]


def test_model_is_resolved_from_the_jev_analysis_level():
    client = MockCritiqueClient()
    compliant_run(client, routing_state=routing_state(analysis_level="medium"))
    assert client.calls[0]["model"] == "mock-critique-medium"


def test_jev_level_enum_members_are_accepted():
    engine = CritiqueEngine(
        critique_client=MockCritiqueClient(), config=engine_config()
    )
    assert engine.resolve_model(ModelLevel.MEDIUM) == "mock-critique-medium"
    assert engine.normalize_level(ModelLevel.ADVANCED) == "advanced"


def test_no_new_routing_decision_is_taken():
    """The synthesis model always follows the analysis level chosen upstream."""
    for level in ("basic", "medium", "advanced"):
        client = MockCritiqueClient()
        compliant_run(client, routing_state=routing_state(analysis_level=level))
        assert client.calls[0]["model"] == MODEL_MAP[level]


def test_single_model_configuration_is_explicit_and_wins():
    engine = CritiqueEngine(
        critique_client=MockCritiqueClient(),
        config=engine_config(single_model="one-model-for-everything"),
    )
    assert engine.resolve_model("advanced") == "one-model-for-everything"


def test_unconfigured_level_fails_loudly_before_any_model_call():
    client = MockCritiqueClient()
    result = run(
        client,
        config=CritiqueConfig(level_models={}, allow_level_fallback=False),
    )
    assert result["status"] == "failed"
    assert client.calls == []
    assert "CRITIQUE_MODEL" in result["failures"][0]["error"]


def test_level_fallback_only_happens_when_explicitly_enabled():
    engine = CritiqueEngine(
        critique_client=MockCritiqueClient(),
        config=CritiqueConfig(
            level_models={"basic": "only-basic"}, allow_level_fallback=True
        ),
    )
    assert engine.resolve_model("advanced") == "only-basic"


def test_missing_analysis_result_is_a_structured_failure():
    client = MockCritiqueClient()
    result = run(client, analysis_result=None)
    assert result["status"] == "failed"
    assert result["failures"][0]["stage"] == "analysis_input"
    assert client.calls == []


def test_failed_analysis_upstream_is_reported_not_synthesized():
    client = MockCritiqueClient()
    failed = synthetic_analysis_result()
    failed["status"] = "failed"
    failed["analysis"] = None
    result = run(client, analysis_result=failed)
    assert result["status"] == "failed"
    assert result["failures"][0]["stage"] == "analysis_input"


def test_rag_disabled_still_produces_a_report_with_an_honest_gap():
    client = MockCritiqueClient()
    result = compliant_run(
        client, rag_result=None, routing_state=routing_state(rag_enabled=False)
    )
    assert result["status"] == "completed"
    assert any(gap["category"] == "rag_disabled" for gap in result["evidence_gaps"])
    assert result["provenance"]["rag_sources_used"] == []
    assert result["context_inputs"]["rag_enabled"] is False


def test_vision_disabled_still_produces_a_report_with_an_honest_gap():
    client = MockCritiqueClient()
    result = compliant_run(
        client, vision_result=None, routing_state=routing_state(vision_enabled=False)
    )
    assert result["status"] == "completed"
    assert any(gap["category"] == "vision_disabled" for gap in result["evidence_gaps"])
    assert result["provenance"]["vision_sources_used"] == []


def test_both_stages_disabled_still_produce_a_report():
    client = MockCritiqueClient()
    result = compliant_run(
        client,
        rag_result=None,
        vision_result=None,
        routing_state=routing_state(rag_enabled=False, vision_enabled=False),
    )
    assert result["status"] == "completed"
    categories = {gap["category"] for gap in result["evidence_gaps"]}
    assert {"rag_disabled", "vision_disabled"} <= categories


def test_model_failure_becomes_a_structured_failure():
    client = MockCritiqueClient(error=CritiqueClientError("upstream 500"))
    result = run(client)
    assert result["status"] == "failed"
    assert result["failures"][0]["stage"] == "critique_generation"
    assert result["sections"] == []


def test_model_failure_never_leaks_the_credential(monkeypatch):
    monkeypatch.setenv("CRITIQUE_API_KEY", SECRET)
    client = MockCritiqueClient(error=CritiqueClientError(f"rejected key {SECRET}"))
    result = run(client)
    assert SECRET not in json.dumps(result)
    assert "[REDACTED_API_KEY]" in result["failures"][0]["error"]


def test_malformed_model_output_becomes_a_validation_failure():
    client = MockCritiqueClient(response="I am unable to produce a critique.")
    result = run(client)
    assert result["status"] == "validation_failed"
    assert result["failures"][0]["stage"] == "validation"
    assert result["sections"] == []
    assert result["warnings"]


def test_fabricated_citation_is_kept_unverified_and_counted():
    """
    A model-invented identifier must never travel as fact.

    It stays visible but unverified, is counted in unverified_evidence_refs, is
    excluded from provenance and is reported as an evidence gap.
    """
    client = MockCritiqueClient()
    response = json.loads(compliant_response_json(build_context()))
    response["sections"][0]["evidence_refs"] = [
        {"source_type": "rag", "source_id": "chunk_999"}
    ]
    client.response = json.dumps(response)
    result = run(client)
    assert result["unverified_evidence_refs"] == 1
    assert any("unresolved_source_id" in item for item in result["warnings"])
    fabricated = [
        ref
        for section in result["sections"]
        for ref in section["evidence_refs"]
        if ref["source_id"] == "chunk_999"
    ]
    assert fabricated and all(ref["verified"] is False for ref in fabricated)
    assert "chunk_999" not in result["provenance"]["rag_sources_used"]
    assert any(
        gap["category"] == "unverified_reference" for gap in result["evidence_gaps"]
    )


def test_numeric_score_output_fails_validation():
    client = MockCritiqueClient()
    response = json.loads(compliant_response_json(build_context()))
    response["overall_assessment"]["content"] += " Overall: 9/10."
    client.response = json.dumps(response)
    result = run(client)
    assert result["status"] == "validation_failed"
    assert any("numeric_paper_score" in item for item in result["warnings"])


def test_result_is_json_serializable_and_free_of_verdict_fields():
    client = MockCritiqueClient()
    result = compliant_run(client)
    serialized = json.dumps(result)
    assert SECRET not in serialized
    for forbidden in ("paper_score", "overall_score", '"tier"', '"rank"'):
        assert forbidden not in serialized


def test_provenance_records_the_sources_that_were_actually_used():
    client = MockCritiqueClient()
    result = compliant_run(client)
    used = result["provenance"]["rag_sources_used"]
    assert used
    assert all(source_id.startswith("chunk_") for source_id in used)


def test_context_inputs_are_auditable():
    client = MockCritiqueClient()
    result = compliant_run(client)
    inputs = result["context_inputs"]
    assert inputs["analysis_available"] is True
    assert inputs["page_count"] == 11
    assert inputs["sources_available"] >= 10
    assert inputs["context_chars"] > 0
    assert "truncations" in inputs


def test_limitations_stay_separate_in_the_result():
    client = MockCritiqueClient()
    result = compliant_run(client)
    author = result["limitations"]["author_stated"]
    analyst = result["limitations"]["analyst_identified"]
    assert author and analyst
    assert all(item["category"] == "author_stated" for item in author)
    assert all(item["category"] == "analyst_identified" for item in analyst)
    assert {item["statement"] for item in author}.isdisjoint(
        {item["statement"] for item in analyst}
    )


def test_no_model_identifier_is_hard_coded_in_the_package():
    """
    Model availability cannot be verified from inside the codebase, so every
    identifier must come from the environment.
    """
    import pathlib
    import re

    package = pathlib.Path(__file__).resolve().parent.parent
    forbidden = re.compile(
        r"(gpt-|o1-|claude-|gemini-|llama-|mistral-|qwen-)", re.IGNORECASE
    )
    offenders = []
    for path in package.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []