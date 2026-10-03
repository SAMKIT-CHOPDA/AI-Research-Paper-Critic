"""
RAG Agent entry point and orchestrator.

Coordinates:
1. Routing state inspection (rag.enabled, rag.level)
2. If RAG disabled: zero API calls, returns structured disabled result
3. If RAG enabled:
   - Text extraction (text_extractor)
   - Semantic chunking (chunker)
   - Vector indexing (embeddings + vector_store)
   - Top-k derivation from JEV capability tier
   - Evidence retrieval (retriever)
   - Citation-ready structured JSON-serializable output
"""

import time
import logging
from typing import Any, Dict, Optional

from backend.rag_agent.config import (
    LEVEL_TOP_K_MAPPING,
    DEFAULT_FALLBACK_TOP_K,
)
from backend.rag_agent.schemas import RAGResult
from backend.rag_agent.text_extractor import extract_paper_text
from backend.rag_agent.chunker import chunk_paper
from backend.rag_agent.embeddings import BaseEmbeddingProvider, OpenAIEmbeddingProvider
from backend.rag_agent.vector_store import LocalVectorStore
from backend.rag_agent.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent orchestrating textual extraction, semantic chunking,
    vector indexing, and JEV-governed evidence retrieval.
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        top_k_mapping: Optional[Dict[str, int]] = None,
    ):
        self._embedding_provider = embedding_provider
        self.top_k_mapping = top_k_mapping or LEVEL_TOP_K_MAPPING

    def _get_embedding_provider(self) -> BaseEmbeddingProvider:
        """Lazy initialization of embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = OpenAIEmbeddingProvider()
        return self._embedding_provider

    @staticmethod
    def normalize_level(level: Any) -> Optional[str]:
        """
        Normalize a JEV capability level to a plain lowercase string (or None).

        The frozen router emits ``ModelLevel`` enum members, whose
        ``str()`` form is ``"ModelLevel.<NAME>"``; only ``.value`` yields the
        contract token ("basic" / "medium" / "advanced").
        """
        if level is None:
            return None
        raw_level = getattr(level, "value", level)
        normalized = str(raw_level).strip().lower()
        return normalized or None

    def resolve_top_k(self, level: Optional[str]) -> int:
        """
        Map JEV capability level (basic, medium, advanced) to retrieval top_k.

        Accepts either a plain string or the frozen router's ``ModelLevel`` enum
        member (``create_routing_state`` preserves enum values inside its dict).

        Documented as an engineering configuration choice, not an empirically
        validated absolute.
        """
        normalized_level = self.normalize_level(level)
        if normalized_level is None:
            return DEFAULT_FALLBACK_TOP_K
        return self.top_k_mapping.get(normalized_level, DEFAULT_FALLBACK_TOP_K)

    def run(
        self,
        pdf_path: str,
        routing_state: Dict[str, Any],
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute RAG agent retrieval workflow governed by JEV routing state.

        Parameters:
        - pdf_path: Path to the paper PDF
        - routing_state: Read-only JEV routing decision state. The frozen router's
          canonical nested shape (``{"rag": {"enabled": bool, "level": ...}}``,
          as produced by ``create_routing_state``) is consumed directly; flat
          ``rag_enabled`` / ``rag_level`` keys are also tolerated. This agent never
          validates, re-writes, or otherwise mutates the router's decision.
        - query: Natural language research query
        - top_k: Optional override for retrieval top_k

        Returns:
        - Standardized JSON-serializable dictionary matching RAGResult
        """
        start_time = time.time()

        # 1. Inspect JEV routing state
        rag_state = routing_state.get("rag") if isinstance(routing_state, dict) else None
        if rag_state is None:
            # Fallback if top-level has rag_enabled directly
            rag_enabled = routing_state.get("rag_enabled", False)
            rag_level = routing_state.get("rag_level")
        else:
            rag_enabled = rag_state.get("enabled", False)
            rag_level = rag_state.get("level")

        # Normalize enum-typed levels from the frozen router into plain strings
        rag_level = self.normalize_level(rag_level)

        # 2. Strict policy: If RAG is disabled, perform ZERO embeddings or API calls
        if not rag_enabled:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            result = RAGResult(
                agent="rag",
                enabled=False,
                level=rag_level,
                query=query,
                chunks_indexed=0,
                top_k=0,
                embedding_model=None,
                results=[],
                elapsed_time_ms=elapsed_ms,
                message="RAG Agent disabled by JEV Router decision. Retrieval was bypassed.",
            )
            return result.model_dump()

        # 3. Resolve top_k from JEV level
        effective_top_k = top_k if top_k is not None else self.resolve_top_k(rag_level)

        # 4. Text extraction (page-aware)
        pages = extract_paper_text(pdf_path)

        # 5. Semantic research chunking
        chunks = chunk_paper(pages)
        if not chunks:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            result = RAGResult(
                agent="rag",
                enabled=True,
                level=rag_level,
                query=query,
                chunks_indexed=0,
                top_k=effective_top_k,
                embedding_model=None,
                results=[],
                elapsed_time_ms=elapsed_ms,
                message="Document contained no extractable text chunks.",
            )
            return result.model_dump()

        # 6. Generate embeddings
        provider = self._get_embedding_provider()
        chunk_texts = [c.text for c in chunks]
        embeddings = provider.embed_texts(chunk_texts)

        # 7. Index in local vector store
        store = LocalVectorStore()
        store.add(chunks, embeddings)

        # 8. Retrieve evidence
        retriever = DocumentRetriever(
            vector_store=store,
            embedding_provider=provider,
            default_top_k=effective_top_k,
        )
        retrieved_chunks = retriever.retrieve(query=query, top_k=effective_top_k)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        result = RAGResult(
            agent="rag",
            enabled=True,
            level=rag_level,
            query=query,
            chunks_indexed=len(chunks),
            top_k=effective_top_k,
            embedding_model=provider.model_name,
            results=retrieved_chunks,
            retrieval_method="cosine_similarity",
            elapsed_time_ms=elapsed_ms,
            message="Evidence retrieval completed successfully.",
        )
        return result.model_dump()
