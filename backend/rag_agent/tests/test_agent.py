"""
Unit tests for RAGAgent orchestrator.
Verifies:
- RAG disabled: zero API calls, returns structured rejection
- RAG enabled: basic (top_k=3), medium (top_k=5), advanced (top_k=8)
- JSON serializability of returned result
- Correct source provenance
"""

import json
import os
import pytest
from unittest.mock import MagicMock

from backend.rag_agent.agent import RAGAgent
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.jev_router.config import ModelLevel
from backend.jev_router.routing_state import create_routing_state

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


class DummyEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic offline mock provider without network calls."""

    @property
    def model_name(self) -> str:
        return "mock-embedding-3-small"

    def embed_texts(self, texts):
        # Generate deterministic synthetic vectors
        dim = 8
        vectors = []
        for i, t in enumerate(texts):
            vec = [0.0] * dim
            vec[i % dim] = 1.0
            vectors.append(vec)
        return vectors

    def embed_query(self, query):
        # Always return unit vector along dim 0
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestRAGAgent:
    def test_rag_disabled_makes_zero_calls(self):
        """When routing_state specifies RAG disabled, returns immediately without embedding work."""
        mock_provider = MagicMock(spec=BaseEmbeddingProvider)
        agent = RAGAgent(embedding_provider=mock_provider)

        routing_state = {
            "rag": {
                "enabled": False,
                "level": None,
            }
        }

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="What is the attention mechanism?",
        )

        assert result["agent"] == "rag"
        assert result["enabled"] is False
        assert result["chunks_indexed"] == 0
        assert result["results"] == []
        assert "disabled by JEV" in result["message"]

        # Ensure no embedding API calls occurred
        assert not mock_provider.embed_texts.called
        assert not mock_provider.embed_query.called

        # JSON serializable
        assert json.loads(json.dumps(result))["enabled"] is False

    def test_rag_enabled_basic_tier(self):
        """Basic tier defaults to top_k=3."""
        provider = DummyEmbeddingProvider()
        agent = RAGAgent(embedding_provider=provider)

        routing_state = {
            "rag": {
                "enabled": True,
                "level": "basic",
            }
        }

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="What datasets are used?",
        )

        assert result["enabled"] is True
        assert result["level"] == "basic"
        assert result["top_k"] == 3
        assert len(result["results"]) == 3
        assert result["chunks_indexed"] > 0
        assert result["embedding_model"] == "mock-embedding-3-small"

        # Check citation metadata
        first_chunk = result["results"][0]
        assert "chunk_id" in first_chunk
        assert "page_start" in first_chunk
        assert "page_end" in first_chunk
        assert "score" in first_chunk
        assert first_chunk["page_start"] >= 1

    def test_rag_enabled_medium_tier(self):
        """Medium tier defaults to top_k=5."""
        provider = DummyEmbeddingProvider()
        agent = RAGAgent(embedding_provider=provider)

        routing_state = {
            "rag": {
                "enabled": True,
                "level": "medium",
            }
        }

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="What is the model architecture?",
        )

        assert result["enabled"] is True
        assert result["level"] == "medium"
        assert result["top_k"] == 5
        assert len(result["results"]) == 5

    def test_rag_enabled_advanced_tier(self):
        """Advanced tier defaults to top_k=8."""
        provider = DummyEmbeddingProvider()
        agent = RAGAgent(embedding_provider=provider)

        routing_state = {
            "rag": {
                "enabled": True,
                "level": "advanced",
            }
        }

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="What empirical evidence supports the main contribution?",
        )

        assert result["enabled"] is True
        assert result["level"] == "advanced"
        assert result["top_k"] == 8
        assert len(result["results"]) == 8

        # JSON serializable
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert len(deserialized["results"]) == 8


class TestRAGAgentRouterContract:
    """
    Verifies read-only consumption of the frozen JEV Router contract output.

    The frozen ``create_routing_state`` helper preserves ``ModelLevel`` enum
    members inside its returned dict, so these tests guard against the enum
    being stringified into an unusable value (e.g. "ModelLevel.ADVANCED").
    """

    def test_resolve_top_k_mapping(self):
        agent = RAGAgent(embedding_provider=DummyEmbeddingProvider())

        assert agent.resolve_top_k("basic") == 3
        assert agent.resolve_top_k("medium") == 5
        assert agent.resolve_top_k("advanced") == 8
        # Enum members from the frozen router resolve to the same tiers
        assert agent.resolve_top_k(ModelLevel.BASIC) == 3
        assert agent.resolve_top_k(ModelLevel.ADVANCED) == 8
        # Unknown / missing levels fall back to the documented default
        assert agent.resolve_top_k(None) == 5
        assert agent.resolve_top_k("unmapped-tier") == 5

    @pytest.mark.parametrize(
        "level, expected_top_k",
        [("basic", 3), ("medium", 5), ("advanced", 8)],
    )
    def test_consumes_real_frozen_routing_state(self, level, expected_top_k):
        agent = RAGAgent(embedding_provider=DummyEmbeddingProvider())

        # Produced by the FROZEN jev_router contract layer - never modified here
        routing_state = create_routing_state(
            analysis_level="advanced",
            rag_enabled=True,
            rag_level=level,
        )
        assert routing_state["rag"]["enabled"] is True

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="What is the encoder-decoder architecture?",
        )

        assert result["enabled"] is True
        assert result["level"] == level
        assert isinstance(result["level"], str)
        assert result["top_k"] == expected_top_k
        assert len(result["results"]) == expected_top_k

    def test_consumes_real_frozen_routing_state_when_disabled(self):
        mock_provider = MagicMock(spec=BaseEmbeddingProvider)
        agent = RAGAgent(embedding_provider=mock_provider)

        routing_state = create_routing_state(
            analysis_level="basic",
            rag_enabled=False,
            rag_level=None,
        )

        result = agent.run(
            pdf_path=BENCHMARK_PDF,
            routing_state=routing_state,
            query="Any query at all",
        )

        assert result["enabled"] is False
        assert result["level"] is None
        assert result["top_k"] == 0
        # The frozen decision short-circuits all embedding work
        assert not mock_provider.embed_texts.called
        assert not mock_provider.embed_query.called
