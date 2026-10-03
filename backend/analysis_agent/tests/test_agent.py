"""
Offline unit tests for the Analysis Agent orchestrator.

Verifies:
- the disabled path performs zero model calls and assembles no evidence
- the enabled path resolves the model from the JEV level (never hard-coded) and
  reports level, model, provider and auditable evidence inputs
- deterministic gaps, model-reported gaps and analyzer notes are merged once
- misconfiguration fails loudly, while a failed model call or context build is
  reported as a structured failure with a credential-masked error
- no model identifier is hard-coded anywhere in the package
"""

import json
import os
import re

import pytest

from backend.analysis_agent.agent import AnalysisAgent
from backend.analysis_agent.config import AnalysisConfig, AnalysisConfigurationError
from backend.analysis_agent.context_builder import build_evidence_bundle
from backend.analysis_agent.llm_client import AnalysisClientError, BaseAnalysisClient
from backend.jev_router.config import ModelLevel
from backend.jev_router.routing_state import create_routing_state

MODEL_MAP = {
    "basic": "basic-model",
    "medium": "medium-model",
    "advanced": "advanced-model",
}

DIMENSION_KEYS = (
    "research_problem",
    "contribution",
    "methodology",
    "data",
    "baselines",
    "metrics",
    "results",
    "reproducibility",
    "internal_consistency",
)


class MockAnalysisClient(BaseAnalysisClient):
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


class ExplodingClient(BaseAnalysisClient):
    """Fails the test if it is ever called (used for the disabled path)."""

    @property
    def provider_name(self) -> str:
        return "exploding"

    def complete(self, prompt: str, model: str) -> str:  # pragma: no cover - guard
        raise AssertionError("The analysis model must not be called on this path")


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


def dimension(name: str, **overrides) -> dict:
    payload = {
        "dimension": name,
        "summary": f"Evidence-grounded reading of {name}.",
        "evidence_status": "supported",
        "findings": [],
        "evidence_refs": [
            {"source_type": "section", "source_id": "section_002", "pages": [8]}
        ],
        "confidence": 0.6,
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
                "supporting_evidence": [
                    {"source_type": "text_chunk", "source_id": "chunk_001", "pages": [8]}
                ],
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
            "evidence_refs": [
                {"source_type": "figure", "source_id": "figure_1", "pages": [3]}
            ],
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
        "author_stated": [],
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
    payload["open_questions"] = []
    payload["evidence_gaps"] = [
        {
            "gap": "No machine-readable numeric values were supplied.",
            "category": "numeric_values_unavailable",
            "impact": "Exact numbers cannot be verified.",
            "affected_dimensions": ["results", "metrics"],
        }
    ]
    payload["overall_assessment"] = {
        "summary": "The evidence supports the structural claims.",
        "evidence_status": "partially_supported",
        "confidence": 0.5,
        "key_basis": [
            {"source_type": "document_profile", "source_id": "document_profile"}
        ],
        "uncertainties": [],
    }
    payload.update(overrides)
    return payload


def render(**overrides) -> str:
    return json.dumps(payload_dict(**overrides))


def agent_config() -> AnalysisConfig:
    return AnalysisConfig(level_models=dict(MODEL_MAP))


def enabled_routing(**overrides) -> dict:
    payload = {
        "analysis_level": "advanced",
        "rag_enabled": True,
        "rag_level": "medium",
        "vision_enabled": True,
        "vision_level": "medium",
    }
    payload.update(overrides)
    return create_routing_state(**payload)


def test_disabled_path_makes_no_calls_and_assembles_no_evidence(tmp_path):
    agent = AnalysisAgent(analysis_client=ExplodingClient(), config=agent_config())
    routing = {
        "router_version": "1.0",
        "analysis": {"enabled": False, "level": None},
        "rag": {"enabled": False, "level": None},
        "vision": {"enabled": False, "level": None},
    }

    result = agent.run(profile(), routing, rag_result=rag(), vision_result=vision())

    assert result["status"] == "disabled"
    assert result["enabled"] is False
    assert result["agent"] == "analysis"
    assert result["analysis"] is None
    assert result["model"] is None
    assert result["provider"] is None
    assert result["failures"] == []
    assert result["evidence_gaps"] == []
    assert result["evidence_inputs"]["sections"] == 0
    assert result["evidence_inputs"]["context_chars"] == 0
    assert "no model calls were made" in result["message"]


def test_completed_run_reports_level_model_provider_and_inputs():
    client = MockAnalysisClient(response=render())
    agent = AnalysisAgent(analysis_client=client, config=agent_config())

    result = agent.run(profile(), enabled_routing(), rag_result=rag(), vision_result=vision())

    assert result["status"] == "completed"
    assert result["enabled"] is True
    assert result["level"] == "advanced"
    assert result["model"] == MODEL_MAP["advanced"]
    assert result["provider"] == "mock"
    assert result["failures"] == []

    inputs = result["evidence_inputs"]
    assert inputs["sections"] == 2
    assert inputs["figures"] == 1
    assert inputs["tables"] == 1
    assert inputs["equations"] == 1
    assert inputs["references"] == 32
    assert inputs["rag_chunks_used"] == 1
    assert inputs["visual_assets_used"] == 1
    assert inputs["context_chars"] > 0
    assert inputs["truncations"] == []

    routing = result["routing"]
    assert routing["router_version"] == "1.0"
    assert routing["analysis_enabled"] is True
    assert routing["analysis_level"] == "advanced"
    assert routing["rag_enabled"] is True
    assert routing["rag_level"] == "medium"
    assert routing["vision_enabled"] is True
    assert routing["vision_level"] == "medium"
    assert routing["routing_confidence"] is None  # never invented locally

    assert "1 claim(s) mapped to evidence" in result["message"]
    assert "1 strength(s)" in result["message"]
    assert "1 weakness(s)" in result["message"]
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == MODEL_MAP["advanced"]


def test_model_level_enum_from_the_frozen_router_is_normalized():
    agent = AnalysisAgent(analysis_client=MockAnalysisClient(response=render()), config=agent_config())
    routing = enabled_routing(analysis_level=ModelLevel.MEDIUM)

    result = agent.run(profile(), routing, rag_result=rag())

    assert result["level"] == "medium"
    assert result["model"] == MODEL_MAP["medium"]
    assert result["routing"]["analysis_level"] == "medium"


def test_routing_confidence_is_passed_through_verbatim():
    agent = AnalysisAgent(analysis_client=MockAnalysisClient(response=render()), config=agent_config())
    routing = enabled_routing(
        confidence={"analysis": 0.91, "rag": 0.55, "vision": 0.5}
    )

    result = agent.run(profile(), routing, rag_result=rag())

    assert result["routing"]["routing_confidence"] == {
        "analysis": 0.91,
        "rag": 0.55,
        "vision": 0.5,
    }


def test_deterministic_model_and_analyzer_gaps_are_merged_once():
    config = agent_config()
    # Reuse the deterministic wording so the model repeating it can be deduplicated.
    deterministic = build_evidence_bundle(
        profile(), routing_state=enabled_routing(), config=config
    ).deterministic_gaps
    rag_gap = next(gap for gap in deterministic if gap.category == "rag_unavailable")

    payload = payload_dict()
    payload["overall_assessment"]["summary"] = "This work proves the hypothesis."
    # RAG is absent in this run, so cite evidence that exists without it.
    payload["claim_evidence_matrix"]["items"][0]["supporting_evidence"] = [
        {"source_type": "section", "source_id": "section_002", "pages": [8]}
    ]
    payload["evidence_gaps"] = [
        {
            "gap": rag_gap.gap,  # repeats a gap the pipeline already proved
            "category": rag_gap.category,
            "impact": rag_gap.impact,
            "affected_dimensions": ["results"],
        },
        {
            "gap": "The paper does not report a per-seed variance for the result.",
            "category": "not_reported_in_paper",
            "impact": "Stability of the reported gain cannot be assessed.",
            "affected_dimensions": ["methodology"],
        },
    ]
    agent = AnalysisAgent(
        analysis_client=MockAnalysisClient(response=json.dumps(payload)), config=config
    )

    result = agent.run(profile(), enabled_routing())

    gaps = result["evidence_gaps"]
    categories = [gap["category"] for gap in gaps]

    assert categories.count("rag_unavailable") == 1
    # The deterministic record wins: it is the one the pipeline can vouch for.
    assert next(gap for gap in gaps if gap["category"] == "rag_unavailable")["source"] == (
        "deterministic_context_check"
    )
    assert "vision_unavailable" in categories  # deterministic only
    assert "not_reported_in_paper" in categories  # model only
    assert "model_uncertainty" in categories  # analyzer validation note
    assert {gap["source"] for gap in gaps} == {
        "deterministic_context_check",
        "analysis_model",
        "analyzer_validation",
    }
    assert result["unverified_evidence_refs"] == 0
    assert result["analysis"]["overall_assessment"]["uncertainties"]


def test_distinctly_worded_gaps_in_the_same_category_are_both_kept():
    agent = AnalysisAgent(
        analysis_client=MockAnalysisClient(response=render()), config=agent_config()
    )

    result = agent.run(profile(), enabled_routing())

    numeric = [
        gap
        for gap in result["evidence_gaps"]
        if gap["category"] == "numeric_values_unavailable"
    ]
    assert len(numeric) == 2
    assert {gap["source"] for gap in numeric} == {
        "deterministic_context_check",
        "analysis_model",
    }


def test_missing_configured_model_fails_loudly():
    agent = AnalysisAgent(
        analysis_client=MockAnalysisClient(response=render()),
        config=AnalysisConfig(level_models={"basic": None, "medium": None, "advanced": None}),
    )

    with pytest.raises(AnalysisConfigurationError) as excinfo:
        agent.run(profile(), enabled_routing())

    assert "ANALYSIS_MODEL_ADVANCED" in str(excinfo.value)


def test_missing_level_in_routing_state_fails_loudly():
    agent = AnalysisAgent(analysis_client=MockAnalysisClient(response=render()), config=agent_config())

    with pytest.raises(AnalysisConfigurationError) as excinfo:
        agent.run(profile(), {"analysis": {"enabled": True, "level": None}})

    assert "level is missing" in str(excinfo.value)


def test_unsupported_level_fails_loudly():
    agent = AnalysisAgent(analysis_client=MockAnalysisClient(response=render()), config=agent_config())

    with pytest.raises(AnalysisConfigurationError) as excinfo:
        agent.run(profile(), {"analysis": {"enabled": True, "level": "cosmic"}})

    assert "Unsupported JEV analysis level" in str(excinfo.value)


def test_missing_document_profile_is_a_structured_failure():
    client = MockAnalysisClient(response=render())
    agent = AnalysisAgent(analysis_client=client, config=agent_config())

    result = agent.run({}, enabled_routing())

    assert result["status"] == "failed"
    assert result["analysis"] is None
    assert result["failures"][0]["stage"] == "evidence_bundle"
    assert "Document Profile" in result["failures"][0]["error"]
    assert "could not be completed" in result["message"]
    assert result["level"] == "advanced"
    assert result["model"] == MODEL_MAP["advanced"]
    assert client.calls == []


def test_failed_model_call_is_a_structured_failure_with_auditable_inputs():
    client = MockAnalysisClient(error=AnalysisClientError("the analysis endpoint timed out"))
    agent = AnalysisAgent(analysis_client=client, config=agent_config())

    result = agent.run(profile(), enabled_routing(), rag_result=rag())

    assert result["status"] == "failed"
    assert result["enabled"] is True
    assert result["analysis"] is None
    failure = result["failures"][0]
    assert failure["stage"] == "analysis"
    assert failure["model"] == MODEL_MAP["advanced"]
    assert "timed out" in failure["error"]
    # Evidence that *was* assembled stays auditable, and its gaps stay recorded.
    assert result["evidence_inputs"]["rag_chunks_used"] == 1
    assert result["evidence_inputs"]["context_chars"] > 0
    categories = [gap["category"] for gap in result["evidence_gaps"]]
    assert "vision_unavailable" in categories


def test_malformed_model_response_is_a_structured_failure():
    agent = AnalysisAgent(
        analysis_client=MockAnalysisClient(response="not json at all"),
        config=agent_config(),
    )

    result = agent.run(profile(), enabled_routing(), rag_result=rag(), vision_result=vision())

    assert result["status"] == "failed"
    assert result["failures"][0]["stage"] == "analysis"
    assert result["analysis"] is None


def test_failure_message_masks_the_configured_api_key(monkeypatch):
    secret = "sk-analysis-secret-token"
    monkeypatch.setenv("ANALYSIS_API_KEY", secret)
    client = MockAnalysisClient(
        error=AnalysisClientError(f"401 unauthorized for key {secret}")
    )
    agent = AnalysisAgent(analysis_client=client, config=agent_config())

    result = agent.run(profile(), enabled_routing(), rag_result=rag())

    text = json.dumps(result)
    assert secret not in text
    assert "[REDACTED_API_KEY]" in result["failures"][0]["error"]


def test_provider_falls_back_to_config_when_client_is_unnamed():
    class UnnamedClient:
        def __init__(self):
            self.models = []

        def complete(self, prompt: str, model: str) -> str:
            self.models.append(model)
            return render()

    client = UnnamedClient()
    agent = AnalysisAgent(
        analysis_client=client,
        config=AnalysisConfig(provider="configured-provider", level_models=dict(MODEL_MAP)),
    )

    result = agent.run(profile(), enabled_routing(), rag_result=rag())

    assert result["provider"] == "configured-provider"
    assert client.models == [MODEL_MAP["advanced"]]


def test_unverifiable_refs_and_downgrades_are_reported_in_the_message():
    payload = payload_dict()
    payload["strengths"][0]["evidence_refs"] = [
        {"source_type": "figure", "source_id": "figure_999", "pages": [99]}
    ]
    agent = AnalysisAgent(
        analysis_client=MockAnalysisClient(response=json.dumps(payload)),
        config=agent_config(),
    )

    result = agent.run(profile(), enabled_routing(), rag_result=rag(), vision_result=vision())

    assert result["status"] == "completed"
    assert result["unverified_evidence_refs"] >= 1
    assert "could not be verified" in result["message"]


def test_no_model_identifier_is_hard_coded_in_the_package():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(package_dir)
    forbidden = re.compile(
        r"\b(?:gpt-[0-9o][\w.\-]*|o[134]-[\w.\-]*|chatgpt[\w.\-]*|claude[\w.\-]*|"
        r"gemini[\w.\-]*|llama[\w.\-]*|mistral[\w.\-]*|deepseek[\w.\-]*)\b",
        re.IGNORECASE,
    )

    offenders = []
    for name in sorted(os.listdir(parent)):
        if not name.endswith(".py") or name.startswith("_phase5_smoke"):
            continue
        path = os.path.join(parent, name)
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                match = forbidden.search(line)
                if match:
                    offenders.append(f"{name}:{line_number}: {match.group(0)}")

    assert offenders == []
