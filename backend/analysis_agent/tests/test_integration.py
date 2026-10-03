"""
Offline end-to-end integration tests for the Analysis Agent (Phase 5).

The whole pipeline is exercised on the real benchmark paper with no network and
no API key:

    Document Pre-Analyzer (frozen) -> JEV routing state (frozen contract shape)
    -> RAG Agent (frozen, deterministic offline embedder)
    -> Vision Agent (frozen, mocked multimodal model)
    -> Analysis Agent (mocked analysis model)

These tests exist to prove the *seams*: the Analysis Agent consumes the real
upstream artifacts (no re-parsing, no re-detection, no second retrieval), the
ids it cites are the ids the upstream agents actually produced, and nothing
invented by a model can travel downstream as a verified fact.
"""

import hashlib
import json
import math
import os
import re

import pytest

from backend.analysis_agent.agent import AnalysisAgent
from backend.analysis_agent.config import AnalysisConfig
from backend.analysis_agent.context_builder import build_evidence_bundle
from backend.analysis_agent.llm_client import BaseAnalysisClient
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.routing_state import create_routing_state
from backend.rag_agent.agent import RAGAgent
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.vision_agent.agent import VisionAgent
from backend.vision_agent.config import VisionConfig
from backend.vision_agent.vision_client import BaseVisionClient

BENCHMARK_PDF = os.path.join(
    "uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

MODEL_MAP = {
    "basic": "mock-analysis-basic",
    "medium": "mock-analysis-medium",
    "advanced": "mock-analysis-advanced",
}

VISION_MODEL_MAP = {
    "basic": "mock-vision-basic",
    "medium": "mock-vision-medium",
    "advanced": "mock-vision-advanced",
}

QUERY = "How is the reported result measured and on which data?"


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic offline embedder: hashed bag-of-words, L2-normalized."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    @property
    def model_name(self) -> str:
        return "hashing-bow-256"

    def _embed(self, text: str):
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
            vec[digest % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, query):
        return self._embed(query)


class MockVisionClient(BaseVisionClient):
    """Deterministic offline multimodal model standing in for a provider."""

    def __init__(self):
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock-vision"

    def analyze_image(self, image, prompt, model):
        self.calls.append({"model": model, "prompt": prompt})
        return json.dumps(
            {
                "observation": "A visual element with labelled regions is visible.",
                "interpretation": "It appears to illustrate the reported comparison.",
                "key_elements": ["labels", "legend"],
                "reported_relationships": ["one series appears larger"],
                "supports_claims": ["the reported comparison appears supported"],
                "uncertainties": ["small text in the crop limits precision"],
                "caption_consistency": {
                    "status": "consistent",
                    "explanation": "caption matches the visible content",
                },
            }
        )


class MockAnalysisClient(BaseAnalysisClient):
    """Returns a caller-supplied contract payload and records every prompt."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock-analysis"

    def complete(self, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        return self.response


def routing_state(**overrides) -> dict:
    payload = {
        "analysis_level": "advanced",
        "rag_enabled": True,
        "rag_level": "medium",
        "vision_enabled": True,
        "vision_level": "medium",
        "confidence": {"analysis": 0.93, "rag": 0.71, "vision": 0.68},
    }
    payload.update(overrides)
    return create_routing_state(**payload)


@pytest.fixture(scope="module")
def benchmark_profile():
    """Real Document Profile from the frozen Document Pre-Analyzer."""
    if not os.path.exists(BENCHMARK_PDF):
        pytest.skip("Benchmark PDF is not available")
    return analyze_document(BENCHMARK_PDF)


@pytest.fixture(scope="module")
def benchmark_rag_result(benchmark_profile):
    """Real RAG Agent output using a deterministic offline embedder."""
    agent = RAGAgent(embedding_provider=HashingEmbeddingProvider())
    return agent.run(
        pdf_path=BENCHMARK_PDF,
        routing_state=routing_state(),
        query=QUERY,
    )


@pytest.fixture(scope="module")
def benchmark_vision_result(benchmark_profile, tmp_path_factory):
    """Real Vision Agent output with a mocked multimodal model."""
    agent = VisionAgent(
        vision_client=MockVisionClient(),
        config=VisionConfig(level_models=dict(VISION_MODEL_MAP), max_assets=0),
    )
    return agent.run(
        pdf_path=BENCHMARK_PDF,
        document_profile=benchmark_profile,
        routing_state=routing_state(),
        output_dir=str(tmp_path_factory.mktemp("phase5-assets")),
    )


def analysis_config() -> AnalysisConfig:
    return AnalysisConfig(level_models=dict(MODEL_MAP))


def run_pipeline(
    benchmark_profile,
    rag_result,
    vision_result,
    response_builder,
    level="advanced",
    rag_enabled=True,
    vision_enabled=True,
):
    """Run the Analysis Agent on real upstream artifacts."""
    state = routing_state(
        analysis_level=level,
        rag_enabled=rag_enabled,
        vision_enabled=vision_enabled,
    )

    bundle = build_evidence_bundle(
        benchmark_profile,
        rag_result=rag_result,
        vision_result=vision_result,
        routing_state=state,
        config=analysis_config(),
    )
    client = MockAnalysisClient(response=response_builder(bundle))
    agent = AnalysisAgent(analysis_client=client, config=analysis_config())
    result = agent.run(
        document_profile=benchmark_profile,
        routing_state=state,
        rag_result=rag_result,
        vision_result=vision_result,
    )
    return result, client, state


def ref(source_id: str, bundle) -> dict:
    entry = bundle.source_index.get(source_id) or {}
    pages = entry.get("pages") or []
    payload = {"source_type": entry.get("source_type", "section"), "source_id": source_id}
    if pages:
        payload["pages"] = list(pages)
    return payload


def contract_payload(bundle) -> str:
    """
    Build a contract-compliant analysis response that only cites real ids.

    Everything cited here exists in ``bundle.source_index``, so any reference the
    Analysis Agent reports as unverifiable is a bug in the integration, not in
    the mock.
    """
    section_ids = [section.source_id for section in bundle.sections]
    chunk_ids = [chunk.source_id for chunk in bundle.text_evidence]
    visual_ids = [visual.source_id for visual in bundle.visual_evidence]
    figure_ids = [figure.source_id for figure in bundle.figures]
    table_ids = [table.source_id for table in bundle.tables]

    def refs(ids):
        return [ref(identifier, bundle) for identifier in ids[:2]] if ids else []

    dimension_names = (
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
    dimensions = {}
    for index, name in enumerate(dimension_names):
        cited = refs(section_ids[index % max(1, len(section_ids)) :][:1]) or refs(chunk_ids)
        dimensions[name] = {
            "dimension": name,
            "summary": f"The confirmed structure supports a bounded reading of {name}.",
            "evidence_status": "supported" if cited else "unclear",
            "findings": [
                {
                    "statement": f"Observed evidence for {name} is limited to what "
                    "the supplied artifacts state.",
                    "evidence_status": "supported" if cited else "unclear",
                    "evidence_refs": cited,
                    "confidence": 0.5 if cited else None,
                    "reasoning": "Cited directly from artifacts in this bundle.",
                    "caveats": [],
                }
            ],
            "evidence_refs": cited,
            "confidence": 0.5 if cited else None,
            "uncertainties": [] if cited else ["No citable artifact was available."],
        }

    return json.dumps(
        {
            **dimensions,
            "claim_evidence_matrix": {
                "items": [
                    {
                        "claim": "The paper reports a measured comparison against baselines.",
                        "claim_source": "paper",
                        "status": "partially_supported",
                        "supporting_evidence": refs(chunk_ids) or refs(section_ids),
                        "contradicting_evidence": [],
                        "assessment": "Only the supplied evidence is assessed.",
                        "missing_evidence": ["Machine-readable values for every cell"],
                        "confidence": 0.4,
                    }
                ],
                "notes": "One claim assessed against the supplied evidence.",
            },

            "strengths": [
                {
                    "statement": "The paper contains a confirmed reference section.",
                    "dimension": "contribution",
                    "evidence_status": "supported",
                    "evidence_refs": refs(section_ids[-1:]) or refs(figure_ids),
                    "rationale": "The Document Profile confirms reference structure.",
                    "confidence": 0.5,
                }
            ],
            "weaknesses": [
                {
                    "statement": "Exact numeric values are only as reliable as the "
                    "evidence that carries them.",
                    "dimension": "results",
                    "evidence_status": "insufficient_evidence",
                    "evidence_refs": refs(table_ids),
                    "rationale": "No authoritative machine-readable values were assumed.",
                    "confidence": None,
                }
            ],
            "limitations": {
                "author_stated": [],
                "analyst_identified": [
                    {
                        "statement": "The reading is bounded by the supplied evidence.",
                        "evidence_status": "unclear",
                        "evidence_refs": [],
                        "confidence": None,
                        "reasoning": "Only artifacts handed to this agent were used.",
                        "caveats": ["Full-text reasoning beyond the bundle is excluded."],
                    }
                ],
            },
            "open_questions": [
                {
                    "question": "Are the reported gains stable across runs?",
                    "why_it_matters": "Stability determines how far the result generalizes.",
                    "related_dimension": "results",
                    "evidence_refs": refs(chunk_ids[:1]),
                }
            ],
            "evidence_gaps": [
                {
                    "gap": "Per-run variance is not present in the supplied evidence.",
                    "category": "not_reported_in_paper",
                    "impact": "Result stability cannot be assessed.",
                    "affected_dimensions": ["results", "reproducibility"],
                }
            ],
            "overall_assessment": {
                "summary": "The supplied evidence supports structural findings while "
                "numeric verification stays explicitly open.",
                "evidence_status": "partially_supported",
                "confidence": 0.45,
                "key_basis": refs(section_ids[:1] + chunk_ids[:1]),
                "uncertainties": ["Numeric values were not asserted as verified."],
            },
        }
    )

class TestBenchmarkPipeline:
    """The real Pre-Analyzer + RAG + Vision outputs feed the Analysis Agent."""

    def test_frozen_upstream_artifacts_drive_the_evidence_inputs(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        result, client, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            contract_payload,
        )

        assert result["status"] == "completed"
        assert result["model"] == MODEL_MAP["advanced"]
        assert result["provider"] == "mock-analysis"
        assert len(client.calls) == 1

        inputs = result["evidence_inputs"]
        assert inputs["sections"] == len(benchmark_profile["sections"])
        assert inputs["figures"] == benchmark_profile["figures"]["count"]
        assert inputs["tables"] == benchmark_profile["tables"]["count"]
        assert inputs["equations"] == benchmark_profile["equations"]["count"]
        assert inputs["references"] == benchmark_profile["references"]["count"]
        assert inputs["rag_enabled"] is True
        assert inputs["rag_chunks_used"] == len(benchmark_rag_result["results"])
        assert inputs["vision_enabled"] is True
        assert inputs["visual_assets_used"] == len(benchmark_vision_result["results"])
        assert inputs["context_chars"] > 0

    def test_upstream_ids_reach_the_prompt_and_all_citations_verify(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        result, client, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            contract_payload,
        )
        prompt = client.calls[0]["prompt"]

        for chunk in benchmark_rag_result["results"]:
            assert chunk["chunk_id"] in prompt
        for item in benchmark_vision_result["results"]:
            assert item["asset_id"] in prompt

        assert result["unverified_evidence_refs"] == 0
        assert result["analysis"]["claim_evidence_matrix"]["items"]

    def test_model_comes_from_the_routed_level_and_prompt_states_the_boundary(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        result, client, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            contract_payload,
            level="medium",
        )

        assert result["level"] == "medium"
        assert result["model"] == MODEL_MAP["medium"]
        assert client.calls[0]["model"] == MODEL_MAP["medium"]
        assert "numeric score" in client.calls[0]["prompt"].lower()

    def test_router_confidence_is_carried_verbatim(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        result = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            contract_payload,
        )[0]

        assert result["routing"]["routing_confidence"] == {
            "analysis": 0.93,
            "rag": 0.71,
            "vision": 0.68,
        }
        assert result["routing"]["analysis_enabled"] is True
        assert result["routing"]["rag_level"] == "medium"
        assert result["routing"]["vision_level"] == "medium"

    def test_disabled_upstream_agents_leave_honest_gaps(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        disabled_rag = RAGAgent(embedding_provider=HashingEmbeddingProvider()).run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state(rag_enabled=False),
            query=QUERY,
        )
        assert disabled_rag["enabled"] is False
        assert disabled_rag["results"] == []

        result, client, _ = run_pipeline(
            benchmark_profile,
            disabled_rag,
            benchmark_vision_result,
            contract_payload,
            rag_enabled=False,
        )

        categories = [gap["category"] for gap in result["evidence_gaps"]]
        assert "rag_disabled" in categories
        assert result["evidence_inputs"]["rag_enabled"] is False
        assert result["evidence_inputs"]["rag_chunks_used"] == 0
        assert "rag" in client.calls[0]["prompt"].lower()

    def test_analysis_disabled_by_router_makes_no_calls_on_real_profile(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        # The frozen contract always enables analysis, so this defensive path is
        # only reachable with a hand-built (non-conforming) state.
        state = routing_state()
        state["analysis"]["enabled"] = False

        client = MockAnalysisClient(response=contract_payload(build_evidence_bundle(
            benchmark_profile,
            rag_result=benchmark_rag_result,
            vision_result=benchmark_vision_result,
            routing_state=state,
            config=analysis_config(),
        )))
        agent = AnalysisAgent(analysis_client=client, config=analysis_config())
        result = agent.run(
            document_profile=benchmark_profile,
            routing_state=state,
            rag_result=benchmark_rag_result,
            vision_result=benchmark_vision_result,
        )

        assert result["status"] == "disabled"
        assert result["model"] is None
        assert result["analysis"] is None
        assert client.calls == []

    def test_output_is_serializable_credential_free_and_score_free(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        result = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            contract_payload,
        )[0]

        text = json.dumps(result)
        assert "sk-" not in text
        assert "api_key" not in text.lower()
        # The Analysis Agent never emits an overall numeric paper score.
        assert not [key for key in result["analysis"] if "score" in key]
        assert not [
            key for key in result["analysis"]["overall_assessment"] if "score" in key
        ]

    def test_context_respects_the_configured_char_budget(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result
    ):
        budget = 6000
        state = routing_state()
        bundle = build_evidence_bundle(
            benchmark_profile,
            rag_result=benchmark_rag_result,
            vision_result=benchmark_vision_result,
            routing_state=state,
            config=AnalysisConfig(
                level_models=dict(MODEL_MAP), max_context_chars=budget
            ),
        )

        assert bundle.total_evidence_chars <= budget
        assert bundle.truncations

