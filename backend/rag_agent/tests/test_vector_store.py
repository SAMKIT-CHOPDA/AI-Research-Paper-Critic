"""
Unit tests for vector_store module.
Verifies:
- Adding chunks and embeddings
- Normalized cosine similarity search
- Top-k ordering descending
- Metadata preservation
- Save / Load disk round-trip
"""

import os
import pytest
from backend.rag_agent.schemas import DocumentChunk
from backend.rag_agent.vector_store import LocalVectorStore, VectorStoreError


def make_chunks_and_embeddings():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_001",
            text="Attention mechanism and Transformer architecture.",
            page_start=1,
            page_end=2,
            word_count=5,
            section="1 Introduction",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            text="Experimental results on WMT 2014 English-to-German dataset.",
            page_start=7,
            page_end=8,
            word_count=8,
            section="5 Results",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            text="Positional encoding formulas using sine and cosine functions.",
            page_start=5,
            page_end=6,
            word_count=8,
            section="3.5 Positional Encoding",
        ),
    ]
    # Synthetic 4-dimensional embeddings
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],  # Aligned with [1, 0, 0, 0]
        [0.0, 1.0, 0.0, 0.0],  # Aligned with [0, 1, 0, 0]
        [0.0, 0.0, 1.0, 0.0],  # Aligned with [0, 0, 1, 0]
    ]
    return chunks, embeddings


class TestVectorStore:
    def test_add_and_search(self):
        chunks, embeddings = make_chunks_and_embeddings()
        store = LocalVectorStore()
        store.add(chunks, embeddings)

        assert len(store) == 3

        # Query strongly matching chunk 1
        query_vec = [0.95, 0.05, 0.0, 0.0]
        results = store.search(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score > 0.90
        assert results[0].page_start == 1
        assert results[0].section == "1 Introduction"

    def test_top_k_ordering_descending(self):
        chunks, embeddings = make_chunks_and_embeddings()
        store = LocalVectorStore()
        store.add(chunks, embeddings)

        query_vec = [0.2, 0.8, 0.1, 0.0]
        results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        # Should be sorted descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
        assert results[0].chunk_id == "chunk_002"

    def test_save_and_load_persistence(self, tmp_path):
        chunks, embeddings = make_chunks_and_embeddings()
        store = LocalVectorStore()
        store.add(chunks, embeddings)

        save_dir = str(tmp_path / "vec_store")
        store.save(save_dir)

        assert os.path.exists(os.path.join(save_dir, "metadata.json"))
        assert os.path.exists(os.path.join(save_dir, "vectors.npy"))

        loaded_store = LocalVectorStore()
        loaded_store.load(save_dir)

        assert len(loaded_store) == 3
        assert loaded_store.dimension == 4

        # Verify search works identically on loaded store
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = loaded_store.search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"
        assert results[0].section == "1 Introduction"

    def test_mismatched_dimensions_raise(self):
        chunks, embeddings = make_chunks_and_embeddings()
        store = LocalVectorStore()
        store.add(chunks, embeddings)

        # Query with 3 dims instead of 4
        with pytest.raises(VectorStoreError):
            store.search([1.0, 0.0, 0.0], top_k=1)
