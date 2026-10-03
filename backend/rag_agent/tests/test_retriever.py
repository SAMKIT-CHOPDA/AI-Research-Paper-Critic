"""
Unit tests for retriever module.
Verifies:
- Top-k retrieval coordination
- Preservation of similarity scores and source metadata
- Empty query handling
"""

import pytest
from unittest.mock import MagicMock

from backend.rag_agent.schemas import DocumentChunk
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.rag_agent.vector_store import LocalVectorStore
from backend.rag_agent.retriever import DocumentRetriever, RetrieverError


class TestRetriever:
    def test_retrieve_ranked_evidence(self):
        chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                text="Methodology section describing encoder-decoder.",
                page_start=2,
                page_end=3,
                word_count=5,
                section="2 Model",
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                text="Results table showing BLEU score of 28.4.",
                page_start=8,
                page_end=9,
                word_count=7,
                section="5 Results",
            ),
        ]
        embeddings = [
            [0.9, 0.1],
            [0.1, 0.9],
        ]

        store = LocalVectorStore()
        store.add(chunks, embeddings)

        mock_provider = MagicMock(spec=BaseEmbeddingProvider)
        # Mock query aligned with chunk 2
        mock_provider.embed_query.return_value = [0.15, 0.85]

        retriever = DocumentRetriever(
            vector_store=store,
            embedding_provider=mock_provider,
            default_top_k=2,
        )

        results = retriever.retrieve("What BLEU scores were reported?")
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_002"
        assert results[0].section == "5 Results"
        assert results[0].page_start == 8
        assert results[0].page_end == 9
        assert results[0].score > results[1].score

    def test_empty_query_raises_error(self):
        store = LocalVectorStore()
        mock_provider = MagicMock(spec=BaseEmbeddingProvider)
        retriever = DocumentRetriever(vector_store=store, embedding_provider=mock_provider)

        with pytest.raises(RetrieverError):
            retriever.retrieve("")
