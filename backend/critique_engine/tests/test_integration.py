"""
Offline end-to-end integration tests for the Critique Engine (Phase 6).

The whole pipeline is exercised on the real benchmark paper with no network and no
API key:

    Document Pre-Analyzer (frozen) -> JEV routing state (frozen contract shape)
    -> RAG Agent (frozen, deterministic offline embedder)
    -> Vision Agent (frozen, mocked multimodal model)
    -> Analysis Agent (mocked analysis model)
    -> Critique Engine (mocked synthesis model)

These tests prove the seams that matter for Phase 6: the critique cites ids the
upstream agents really produced, its page numbers exist in the real document, the
Phase 5 claim matrix survives unchanged, the two limitation categories stay apart,
no numeric score appears anywhere, and provenance is preserved in the
machine-readable output.
"""

import json
import os

import pytest

from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.context_builder import build_synthesis_context
from backend.critique_engine.engine import CritiqueEngine
from backend.critique_engine.tests import (
    MODEL_MAP as CRITIQUE_MODEL_MAP,
    MockCritiqueClient,
    build_compliant_response,
    compliant_response_json,
)
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.routing_state import create_routing_state
from backend.rag_agent.agent import RAGAgent
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.vision_agent.agent import VisionAgent
from backend.vision_agent.config import VisionConfig
from backend.vision_agent.vision_client import BaseVisionClient
from backend.analysis_agent.agent import AnalysisAgent
from backend.analysis_agent.config import AnalysisConfig
from backend.analysis_agent.context_builder import build_evidence_bundle
from backend.analysis_agent.llm_client import BaseAnalysisClient

BENCHMARK_PDF = os.path.join(
    "uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

ANALYSIS_MODEL_MAP = {
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
        import hashlib
        import math
        import re

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
                "observation": "A labelled visual element is visible in the crop.",
                "interpretation": "It appears to illustrate the described design.",
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
    """Returns a caller-supplied Phase 5 payload and records every prompt."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock-analysis"

    def complete(self, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        return self.response


def routing_state(**overrides):
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
    return agent.run(pdf_path=BENCHMARK_PDF, routing_state=routing_state(), query=QUERY)


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
        output_dir=str(tmp_path_factory.mktemp("phase6-assets")),
    )


def _ref(source_id: str, bundle) -> dict:
    entry = bundle.source_index.get(source_id) or {}
    payload = {
        "source_type": entry.get("source_type", "section"),
        "source_id": source_id,
    }
    pages = entry.get("pages") or []
    if pages:
        payload["pages"] = list(pages)
    return payload


def _refs(source_ids, bundle):
    return [_ref(identifier, bundle) for identifier in source_ids[:2]] if source_ids else []


def _dimension(name, cited, statement):
    return {
        "dimension": name,
        "summary": f"The supplied evidence supports a bounded reading of {name}.",
        "evidence_status": "supported" if cited else "unclear",
        "findings": [
            {
                "statement": statement,
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


def analysis_payload(bundle) -> str:
    """
    A Phase 5 response that only cites ids the real evidence bundle contains.

    Every citation here exists in ``bundle.source_index``, so any reference the
    Analysis Agent reports as unverifiable would be an integration bug rather than
    a mock defect.
    """
    section_ids = [item.source_id for item in bundle.sections]
    chunk_ids = [item.source_id for item in bundle.text_evidence]
    visual_ids = [item.source_id for item in bundle.visual_evidence]
    figure_ids = [item.source_id for item in bundle.figures]
    table_ids = [item.source_id for item in bundle.tables]

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
        pool = chunk_ids or section_ids
        chosen = pool[index % len(pool) :][:1] if pool else []
        cited = _refs(chosen, bundle) or _refs(section_ids[:1], bundle)
        dimensions[name] = _dimension(
            name,
            cited,
            (
                f"Observed evidence for {name} is limited to what the supplied "
                "artifacts state."
            ),
        )

    payload = {
        "claim_evidence_matrix": {
            "items": [
                {
                    "claim": (
                        "The paper reports a measured comparison against baselines."
                    ),
                    "claim_source": "paper",
                    "status": "partially_supported",
                    "supporting_evidence": _refs(chunk_ids or section_ids, bundle),
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
                "evidence_refs": _refs(section_ids[-1:] or figure_ids, bundle),
                "rationale": "The Document Profile confirms reference structure.",
                "confidence": 0.5,
            }
        ],
        "weaknesses": [
            {
                "statement": (
                    "Exact numeric values are only as reliable as the evidence that "
                    "carries them."
                ),
                "dimension": "results",
                "evidence_status": "insufficient_evidence",
                "evidence_refs": _refs(table_ids, bundle),
                "rationale": "No authoritative machine-readable values were assumed.",
                "confidence": None,
            }
        ],
        "limitations": _limitations(bundle),
        "open_questions": [
            {
                "question": "Are the reported gains stable across runs?",
                "why_it_matters": (
                    "Stability determines how far the result generalizes."
                ),
                "related_dimension": "results",
                "evidence_refs": _refs(chunk_ids or visual_ids, bundle),
            }
        ],
        "evidence_gaps": [
            {
                "gap": "Per-run variance is not present in the supplied evidence.",
                "category": "not_reported_in_paper",
                "impact": "Result stability cannot be assessed.",
                "affected_dimensions": ["results"],
            }
        ],
        "overall_assessment": {
            "summary": (
                "The supplied evidence supports structural findings while numeric "
                "verification stays explicitly open."
            ),
            "evidence_status": "partially_supported",
            "confidence": None,
            "key_basis": _refs((section_ids[:1] + chunk_ids[:1]), bundle),
            "uncertainties": ["Numeric values were not asserted as verified."],
        },
    }
    payload.update(dimensions)
    return json.dumps(payload)


def _limitations(bundle):
    chunk_ids = [item.source_id for item in bundle.text_evidence]
    section_ids = [item.source_id for item in bundle.sections]
    return {
        "author_stated": [
            {
                "statement": (
                    "The authors state that the evaluation covers a limited set of "
                    "settings."
                ),
                "evidence_status": "supported",
                "evidence_refs": _refs(chunk_ids[:1] or section_ids[:1], bundle),
                "confidence": None,
                "reasoning": "Cited from the retrieved text.",
                "caveats": [],
            }
        ],
        "analyst_identified": [
            {
                "statement": (
                    "The analysis indicates that training details are not established "
                    "from the available evidence."
                ),
                "evidence_status": "unclear",
                "evidence_refs": _refs(chunk_ids[:1] or section_ids[:1], bundle),
                "confidence": None,
                "reasoning": "No training detail was available in the bundle.",
                "caveats": [],
            }
        ],
    }


def critique_config(**overrides):
    payload = {"level_models": dict(CRITIQUE_MODEL_MAP)}
    payload.update(overrides)
    return CritiqueConfig(**payload)


@pytest.fixture(scope="module")
def benchmark_analysis_result(benchmark_profile, benchmark_rag_result, benchmark_vision_result):
    """Real Phase 5 output produced from the real upstream artifacts."""
    bundle = build_evidence_bundle(
        benchmark_profile,
        rag_result=benchmark_rag_result,
        vision_result=benchmark_vision_result,
        routing_state=routing_state(),
        config=AnalysisConfig(level_models=dict(ANALYSIS_MODEL_MAP)),
    )
    client = MockAnalysisClient(response=analysis_payload(bundle))
    agent = AnalysisAgent(
        analysis_client=client,
        config=AnalysisConfig(level_models=dict(ANALYSIS_MODEL_MAP)),
    )
    result = agent.run(
        document_profile=benchmark_profile,
        routing_state=routing_state(),
        rag_result=benchmark_rag_result,
        vision_result=benchmark_vision_result,
    )
    assert result["status"] == "completed"
    return result


def run_pipeline(
    profile,
    rag_result,
    vision_result,
    analysis_result,
    level="advanced",
    rag_enabled=True,
    vision_enabled=True,
):
    """Run the Critique Engine on the real upstream artifacts."""
    state = routing_state(
        analysis_level=level,
        rag_enabled=rag_enabled,
        vision_enabled=vision_enabled,
    )
    effective_rag = rag_result if rag_enabled else None
    effective_vision = vision_result if vision_enabled else None
    context = build_synthesis_context(
        profile,
        routing_state=state,
        rag_result=effective_rag,
        vision_result=effective_vision,
        analysis_result=analysis_result,
        config=critique_config(),
    )
    client = MockCritiqueClient(compliant_response_json(context))
    engine = CritiqueEngine(critique_client=client, config=critique_config())
    result = engine.generate_critique(
        document_profile=profile,
        routing_state=state,
        rag_result=effective_rag,
        vision_result=effective_vision,
        analysis_result=analysis_result,
    )
    return result, client, context


class TestBenchmarkPipeline:
    """The real frozen artifacts drive the final critique end to end."""

    def test_real_upstream_outputs_produce_a_completed_critique(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, client, context = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        assert result["status"] == "completed"
        assert len(client.calls) == 1
        assert result["model"] == CRITIQUE_MODEL_MAP["advanced"]
        assert result["level"] == "advanced"
        assert result["sections"]

    def test_every_cited_source_id_really_exists_upstream(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        assert result["unverified_evidence_refs"] == 0

        chunk_ids = {item["chunk_id"] for item in benchmark_rag_result["results"]}
        asset_ids = {item["asset_id"] for item in benchmark_vision_result["results"]}
        analysis_ids = set()
        for dimension in benchmark_analysis_result["analysis"].values():
            if isinstance(dimension, dict):
                for ref in dimension.get("evidence_refs") or []:
                    analysis_ids.add(ref["source_id"])

        known = chunk_ids | asset_ids | analysis_ids | {"document_profile"}
        for record in result["sections"]:
            for ref in record["evidence_refs"]:
                assert ref["verified"] is True
                assert ref["source_id"] in known
                assert ref["source_id"] not in {"chunk_999", "figure_999", "table_999"}

    def test_prompt_carries_the_real_ids_and_no_fabricated_pages(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, client, context = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        prompt = client.calls[0]["prompt"]
        for chunk in benchmark_rag_result["results"]:
            assert chunk["chunk_id"] in prompt
        for asset in benchmark_vision_result["results"]:
            assert asset["asset_id"] in prompt

        page_count = benchmark_profile["page_count"]
        for record in result["sections"]:
            for ref in record["evidence_refs"]:
                for page in ref.get("pages") or []:
                    assert 1 <= page <= page_count

    def test_model_follows_the_routed_analysis_level(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, client, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
            level="medium",
        )
        assert result["level"] == "medium"
        assert result["model"] == CRITIQUE_MODEL_MAP["medium"]
        assert client.calls[0]["model"] == CRITIQUE_MODEL_MAP["medium"]

    def test_router_confidence_is_carried_verbatim(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        assert result["routing"]["confidence"] == {
            "analysis": 0.93,
            "rag": 0.71,
            "vision": 0.68,
        }

    def test_author_and_analyst_limitations_stay_separate(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        source = benchmark_analysis_result["analysis"]["limitations"]
        author = result["limitations"]["author_stated"]
        analyst = result["limitations"]["analyst_identified"]
        assert [item["statement"] for item in author] == [
            item["statement"] for item in source["author_stated"]
        ]
        assert [item["statement"] for item in analyst] == [
            item["statement"] for item in source["analyst_identified"]
        ]
        assert {item["statement"] for item in author}.isdisjoint(
            {item["statement"] for item in analyst}
        )

    def test_claim_evidence_matrix_is_preserved_unchanged(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        source_items = benchmark_analysis_result["analysis"]["claim_evidence_matrix"][
            "items"
        ]
        assert len(result["claim_evidence_summary"]) == len(source_items)
        for produced, original in zip(result["claim_evidence_summary"], source_items):
            assert produced["status"] == original["status"]
            assert produced["claim"] == original["claim"]

    def test_no_numeric_score_appears_anywhere_in_the_output(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        serialized = json.dumps(result).lower()
        assert "paper_score" not in serialized
        assert "overall_score" not in serialized
        assert '"tier"' not in serialized
        assert '"rank"' not in serialized
        from backend.critique_engine.validator import detect_score_like_text

        texts = [record["content"] for record in result["sections"]]
        texts.append(result["overall_assessment"]["content"])
        for text in texts:
            assert detect_score_like_text(text) == []

    def test_provenance_and_serializability_are_preserved(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, context = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
        )
        assert json.loads(json.dumps(result))["status"] == "completed"
        used = set(result["provenance"]["rag_sources_used"])
        assert used
        known = {entry["source_id"] for entry in context.source_index.values()}
        assert used <= known

    def test_disabled_upstream_stages_still_yield_an_honest_report(
        self, benchmark_profile, benchmark_rag_result, benchmark_vision_result,
        benchmark_analysis_result,
    ):
        result, _, _ = run_pipeline(
            benchmark_profile,
            benchmark_rag_result,
            benchmark_vision_result,
            benchmark_analysis_result,
            rag_enabled=False,
            vision_enabled=False,
        )
        assert result["status"] == "completed"
        categories = {gap["category"] for gap in result["evidence_gaps"]}
        assert "rag_disabled" in categories
        assert "vision_disabled" in categories