"""
Offline tests for the orchestration layer.

The pipeline itself must stay free of agent logic, so these tests inject fake
clients and assert only what the orchestrator is responsible for: the stage order,
the pass-through of structured outputs, the propagation of a JEV decision that
disables a stage, structured failures, credential masking, and the fact that no
network call can happen when every client is injected.
"""

import json

import pytest

import backend.pipeline as pipeline
from backend.pipeline import STAGES, run_pipeline

SECRET = "sk-pipeline-test-secret-424242"


class Boom(RuntimeError):
    pass


@pytest.fixture
def fake_modules(monkeypatch):
    """Replace every stage with a recorder, so no agent logic runs for real."""
    calls = []

    def fake_analyze(path):
        calls.append(("document_pre_analyzer", path))
        return {
            "filename": "synthetic.pdf",
            "page_count": 4,
            "word_count": 900,
            "metadata": {"title": "Synthetic"},
            "sections": [{"title": "Intro", "page": 1}],
            "figures": {"count": 1, "items": []},
            "tables": {"count": 0, "items": []},
            "equations": {"count": 0, "items": []},
            "references": {"count": 5},
        }

    def fake_route(profile, client=None, **kwargs):
        calls.append(("jev_router", profile.get("filename")))
        return {
            "router_version": "test",
            "analysis": {"enabled": True, "level": "medium"},
            "rag": {"enabled": True, "level": "basic"},
            "vision": {"enabled": False, "level": None},
            "confidence": {"analysis": 0.9},
        }

    class FakeRag:
        def __init__(self, embedding_provider=None):
            self.embedding_provider = embedding_provider

        def run(self, pdf_path, routing_state, query, top_k=None):
            calls.append(("rag", routing_state["rag"]["enabled"], query))
            return {
                "agent": "rag",
                "enabled": routing_state["rag"]["enabled"],
                "level": "basic",
                "results": [{"chunk_id": "chunk_001", "score": 0.5}],
            }

    class FakeVision:
        def __init__(self, vision_client=None):
            self.vision_client = vision_client

        def run(self, pdf_path, document_profile, routing_state, output_dir=None):
            calls.append(("vision", routing_state["vision"]["enabled"]))
            return {"agent": "vision", "enabled": False, "results": []}

    class FakeAnalysis:
        def __init__(self, analysis_client=None):
            self.analysis_client = analysis_client

        def run(self, document_profile, routing_state, rag_result, vision_result):
            calls.append(("analysis", rag_result["agent"], vision_result["agent"]))
            return {
                "status": "completed",
                "level": "medium",
                "model": "fake-analysis",
                "analysis": {"research_problem": {"summary": "s", "findings": []}},
                "evidence_gaps": [],
            }

    class FakeCritique:
        def __init__(self, critique_client=None):
            self.critique_client = critique_client

        def generate_critique(self, **kwargs):
            calls.append(("critique", kwargs["analysis_result"]["model"]))
            return {
                "status": "completed",
                "model": "fake-critique",
                "sections": [{"section_id": "executive_summary", "content": "x"}],
                "provenance": {"rag_sources_used": []},
            }

    monkeypatch.setattr(pipeline, "analyze_document", fake_analyze)
    monkeypatch.setattr(pipeline, "route_document", fake_route)
    monkeypatch.setattr(pipeline, "RAGAgent", FakeRag)
    monkeypatch.setattr(pipeline, "VisionAgent", FakeVision)
    monkeypatch.setattr(pipeline, "AnalysisAgent", FakeAnalysis)
    monkeypatch.setattr(pipeline, "CritiqueEngine", FakeCritique)
    return calls


def test_pipeline_runs_every_stage_in_order_and_returns_results(fake_modules):
    result = run_pipeline("synthetic.pdf")

    assert result["status"] == "completed"
    assert result["failed_stage"] is None
    assert [name for name, *_ in fake_modules] == list(STAGES)
    for stage in STAGES:
        assert result["stages"][stage]["status"] == "completed"
        assert result["stages"][stage]["result"] is not None
    assert result["elapsed_ms"] >= 0


def test_the_query_is_forwarded_to_the_rag_agent(fake_modules):
    run_pipeline("synthetic.pdf", query="where is the result measured?")
    rag_call = next(item for item in fake_modules if item[0] == "rag")
    assert rag_call[2] == "where is the result measured?"


def test_the_jev_decision_is_passed_through_untouched(fake_modules):
    result = run_pipeline("synthetic.pdf")
    assert result["stages"]["jev_router"]["result"]["vision"]["enabled"] is False
    vision_call = next(item for item in fake_modules if item[0] == "vision")
    assert vision_call[1] is False


def test_structured_outputs_are_passed_between_stages(fake_modules):
    run_pipeline("synthetic.pdf")
    analysis_call = next(item for item in fake_modules if item[0] == "analysis")
    assert analysis_call[1] == "rag"
    assert analysis_call[2] == "vision"
    critique_call = next(item for item in fake_modules if item[0] == "critique")
    assert critique_call[1] == "fake-analysis"


def test_the_result_is_json_serializable(fake_modules):
    result = run_pipeline("synthetic.pdf")
    assert json.loads(json.dumps(result))["status"] == "completed"


def test_progress_callback_receives_every_stage(fake_modules):
    seen = []
    run_pipeline(
        "synthetic.pdf",
        progress=lambda stage, status: seen.append((stage, status)),
    )
    assert [stage for stage, status in seen if status == "completed"] == list(STAGES)


def test_a_failing_stage_stops_the_pipeline_and_is_reported(monkeypatch, fake_modules):
    def boom(*args, **kwargs):
        raise Boom("upstream unavailable")

    monkeypatch.setattr(pipeline, "route_document", boom)
    result = run_pipeline("synthetic.pdf")

    assert result["status"] == "failed"
    assert result["failed_stage"] == "jev_router"
    assert result["stages"]["jev_router"]["error_type"] == "Boom"
    assert "upstream unavailable" in result["stages"]["jev_router"]["error"]
    assert "analysis" not in result["stages"]


def test_a_failed_critique_is_reported_without_faking_a_report(
    monkeypatch, fake_modules
):
    class FailingCritique:
        def __init__(self, critique_client=None):
            pass

        def generate_critique(self, **kwargs):
            return {
                "status": "failed",
                "failures": [{"stage": "critique_generation", "error": "provider"}],
            }

    monkeypatch.setattr(pipeline, "CritiqueEngine", FailingCritique)
    result = run_pipeline("synthetic.pdf")

    assert result["status"] == "failed"
    assert result["failed_stage"] == "critique"
    assert "sections" not in (result["stages"]["critique"]["result"] or {})


def test_provider_errors_are_credential_masked(monkeypatch, fake_modules):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    def leaky(*args, **kwargs):
        raise Boom(f"request rejected for key {SECRET}")

    monkeypatch.setattr(pipeline, "route_document", leaky)
    result = run_pipeline("synthetic.pdf")

    assert SECRET not in json.dumps(result)
    assert "[REDACTED_API_KEY]" in result["stages"]["jev_router"]["error"]


def test_jev_client_is_built_with_the_environment_credential(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    client = pipeline._build_jev_client()
    assert client._api_key == SECRET