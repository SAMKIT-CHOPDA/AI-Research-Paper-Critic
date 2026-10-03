"""
Lightweight local vector store utilizing NumPy cosine similarity.

Stores embeddings alongside research paper chunk metadata:
- chunk_id
- text
- page_start, page_end
- section
- word_count
"""

import json
import os
from typing import List, Optional
import numpy as np

from backend.rag_agent.schemas import DocumentChunk, RetrievedChunk


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class LocalVectorStore:
    """
    In-memory local vector store with NumPy-based normalized cosine search
    and disk serialization.
    """

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add document chunks and their corresponding embeddings into the store."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: {len(chunks)} chunks provided with {len(embeddings)} embeddings."
            )

        if not chunks:
            return

        new_mat = np.array(embeddings, dtype=np.float32)
        if new_mat.ndim != 2:
            raise VectorStoreError(f"Embeddings matrix must be 2D, got shape {new_mat.shape}")

        if self.dimension is None:
            self.dimension = new_mat.shape[1]
        elif new_mat.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {new_mat.shape[1]}"
            )

        # Normalize vectors to unit length so the dot product equals cosine similarity
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_new = new_mat / norms

        if self.embeddings is None:
            self.embeddings = normalized_new
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_new])
            self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        """Execute cosine similarity search and return top-k ranked chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32).flatten()
        if q_vec.shape[0] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension {q_vec.shape[0]} does not match store dimension {self.dimension}"
            )

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.embeddings, q_vec)
        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            raw_score = float(scores[idx])
            bounded_score = round(max(0.0, min(1.0, raw_score)), 4)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    score=bounded_score,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    text=chunk.text,
                    word_count=chunk.word_count,
                )
            )

        return results

    def save(self, directory_path: str) -> None:
        """Persist vector store index and chunk metadata to a local directory."""
        try:
            os.makedirs(directory_path, exist_ok=True)
            meta_path = os.path.join(directory_path, "metadata.json")
            vec_path = os.path.join(directory_path, "vectors.npy")

            metadata_payload = {
                "dimension": self.dimension,
                "chunks": [c.model_dump() for c in self.chunks],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2)

            if self.embeddings is not None:
                np.save(vec_path, self.embeddings)
            else:
                np.save(vec_path, np.empty((0, self.dimension or 0), dtype=np.float32))

        except Exception as exc:
            raise VectorStoreError(f"Failed to save vector store to {directory_path}: {exc}") from exc

    def load(self, directory_path: str) -> None:
        """Load persisted vector store from a local directory."""
        meta_path = os.path.join(directory_path, "metadata.json")
        vec_path = os.path.join(directory_path, "vectors.npy")

        if not os.path.exists(meta_path) or not os.path.exists(vec_path):
            raise VectorStoreError(f"Vector store files not found in {directory_path}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.dimension = meta.get("dimension")
            raw_chunks = meta.get("chunks", [])
            self.chunks = [DocumentChunk.model_validate(c) for c in raw_chunks]

            loaded_embeddings = np.load(vec_path)
            if loaded_embeddings.size > 0:
                self.embeddings = loaded_embeddings.astype(np.float32)
            else:
                self.embeddings = None

        except Exception as exc:
            raise VectorStoreError(f"Failed to load vector store from {directory_path}: {exc}") from exc
