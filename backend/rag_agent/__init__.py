"""
RAG Agent subsystem for AI Research Paper Critic.

Exports:
- RAGAgent: High-level orchestrator
- extract_paper_text: Page-aware textual extractor
- chunk_paper: Word-aware semantic chunker
- BaseEmbeddingProvider, OpenAIEmbeddingProvider: Embeddings abstraction
- LocalVectorStore: NumPy cosine similarity vector store
- DocumentRetriever: Query retrieval coordinator
- Schemas: PageText, DocumentChunk, RetrievedChunk, RAGResult
"""

from backend.rag_agent.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TARGET_CHUNK_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    LEVEL_TOP_K_MAPPING,
)
from backend.rag_agent.schemas import (
    PageText,
    DocumentChunk,
    RetrievedChunk,
    RAGResult,
)
from backend.rag_agent.text_extractor import (
    extract_paper_text,
    TextExtractionError,
)
from backend.rag_agent.chunker import (
    chunk_paper,
)
from backend.rag_agent.embeddings import (
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    EmbeddingError,
)
from backend.rag_agent.vector_store import (
    LocalVectorStore,
    VectorStoreError,
)
from backend.rag_agent.retriever import (
    DocumentRetriever,
    RetrieverError,
)
from backend.rag_agent.agent import (
    RAGAgent,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_TARGET_CHUNK_WORDS",
    "DEFAULT_CHUNK_OVERLAP_WORDS",
    "LEVEL_TOP_K_MAPPING",
    "PageText",
    "DocumentChunk",
    "RetrievedChunk",
    "RAGResult",
    "extract_paper_text",
    "TextExtractionError",
    "chunk_paper",
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingError",
    "LocalVectorStore",
    "VectorStoreError",
    "DocumentRetriever",
    "RetrieverError",
    "RAGAgent",
]
