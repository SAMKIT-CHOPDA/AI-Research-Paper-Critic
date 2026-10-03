"""
Offline end-to-end integration tests for the RAG Agent pipeline.

Covers the full chain on the real benchmark paper with a deterministic,
network-free embedding provider:
    text_extractor -> chunker -> embeddings -> vector_store -> retriever -> agent

No OpenAI API calls are performed.
"""

import os
import re
import math
import json
import hashlib

import pytest

from backend.rag_agent.agent import RAGAgent
from backend.rag_agent.chunker import chunk_paper
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.rag_agent.text_extractor import extract_paper_text

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")
BENCHMARK_PAGE_COUNT = 11


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
        return [self._embed(t) for t in texts]

    def embed_query(self, query):
        return self._embed(query)


@pytest.fixture(scope="module")
def benchmark_pages():
    assert os.path.exists(BENCHMARK_PDF)
    return extract_paper_text(BENCHMARK_PDF)


class TestRAGPipelineIntegration:
    def test_extraction_and_chunking_cover_whole_paper(self, benchmark_pages):
        """Every page of the benchmark paper must survive extraction and chunking."""
        assert len(benchmark_pages) == BENCHMARK_PAGE_COUNT

        chunks = chunk_paper(benchmark_pages)
        assert len(chunks) >= 5

        # Sequential, unique chunk identifiers
        assert [c.chunk_id for c in chunks] == [
            f"chunk_{i:03d}" for i in range(1, len(chunks) + 1)
        ]

        covered_pages = set()
        for chunk in chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= BENCHMARK_PAGE_COUNT
            assert chunk.word_count > 0
            assert chunk.text.strip()
            covered_pages.update(range(chunk.page_start, chunk.page_end + 1))

        assert covered_pages == set(range(1, BENCHMARK_PAGE_COUNT + 1))

    def test_retrieval_returns_relevant_cited_evidence(self, benchmark_pages):
        """A results-oriented query must surface the BLEU results chunk with citations."""
        agent = RAGAgent(embedding_provider=HashingEmbeddingProvider())
        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state={"rag": {"enabled": True, "level": "medium"}},
            query=(
                "What BLEU score does the Transformer achieve on the WMT 2014 "
                "English-to-German translation task?"
            ),
        )

        assert result["enabled"] is True
        assert result["level"] == "medium"
        assert result["top_k"] == 5
        assert result["chunks_indexed"] == len(chunk_paper(benchmark_pages))
        assert len(result["results"]) == 5

        # Scores must be bounded and in descending order
        scores = [r["score"] for r in result["results"]]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)

        # Top-ranked evidence must be the paper's translation-results passage
        assert "BLEU" in result["results"][0]["text"]

        # Citation metadata must be intact and machine-readable
        for item in result["results"]:
            assert 1 <= item["page_start"] <= item["page_end"] <= BENCHMARK_PAGE_COUNT
            assert item["chunk_id"].startswith("chunk_")
            assert item["word_count"] > 0
        assert json.loads(json.dumps(result))["results"][0]["chunk_id"] == result["results"][0]["chunk_id"]

    def test_rag_disabled_skips_all_work_on_real_paper(self, benchmark_pages):
        """Disabled routing state must bypass extraction, embedding and retrieval entirely."""
        agent = RAGAgent(embedding_provider=HashingEmbeddingProvider())
        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state={"rag": {"enabled": False, "level": None}},
            query="What BLEU score does the Transformer achieve?",
        )

        assert result["enabled"] is False
        assert result["chunks_indexed"] == 0
        assert result["results"] == []
        assert result["embedding_model"] is None


class TestRAGDemoScript:
    """Guards the standalone demo runner's safe, non-crashing behaviour."""

    def test_demo_exits_gracefully_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import backend.rag_agent.run_rag_demo as demo

        # Must NOT attempt any API call and must exit successfully
        assert demo.run_rag_demo() == 0

    def test_demo_safe_helper_never_raises_on_unicode_math_symbols(self):
        import backend.rag_agent.run_rag_demo as demo

        # U+2217 appears throughout the benchmark paper's author list
        assert isinstance(demo._safe("attention \u2217 scaled \ufb01nal"), str)
