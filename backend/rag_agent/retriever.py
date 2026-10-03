"""
Retriever component for RAG Agent.

Performs:
1. Natural-language query embedding
2. Vector store search for top-k nearest chunks
3. Similarity score sorting and metadata preservation
"""

from typing import List, Optional
from backend.rag_agent.schemas import RetrievedChunk
from backend.rag_agent.embeddings import BaseEmbeddingProvider
from backend.rag_agent.vector_store import LocalVectorStore


class RetrieverError(Exception):
    """Raised when query retrieval fails."""
    pass


class DocumentRetriever:
    """
    Coordinates query embedding and vector store similarity ranking.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        default_top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks matching the query.

        Parameters:
        - query: Natural language research query
        - top_k: Number of chunks to retrieve (defaults to default_top_k)

        Returns:
        - List of RetrievedChunk objects ordered by descending cosine similarity.
        """
        if not query or not query.strip():
            raise RetrieverError("Query string cannot be empty")

        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            return []

        try:
            # 1. Embed query
            query_vector = self.embedding_provider.embed_query(query)
            # 2. Search vector store
            results = self.vector_store.search(query_vector, top_k=k)
            return results
        except Exception as exc:
            raise RetrieverError(f"Evidence retrieval failed: {exc}") from exc
