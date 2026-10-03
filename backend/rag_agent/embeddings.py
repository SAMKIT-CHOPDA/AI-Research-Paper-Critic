"""
Embeddings abstraction and OpenAI implementation.

Defines:
- BaseEmbeddingProvider: Clean abstract base class for embedding providers
- OpenAIEmbeddingProvider: Concrete implementation utilizing OpenAI embeddings API
- Batching support, API error handling, and key masking
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from backend.rag_agent.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when text embedding generation fails."""
    pass


class BaseEmbeddingProvider(ABC):
    """Abstract interface for generating text vector embeddings."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model in use."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embeddings implementation supporting batching and clean error handling.
    Reads OPENAI_API_KEY from environment. Never logs or prints the API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        raw_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._api_key = raw_key.strip()
        self._model = model or DEFAULT_EMBEDDING_MODEL
        self.batch_size = max(1, batch_size)
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingError(
                    "Missing OPENAI_API_KEY. Set OPENAI_API_KEY environment variable or pass api_key."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except Exception as exc:
                raise EmbeddingError(f"Failed to initialize OpenAI client: {exc}") from exc
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches.
        Handles empty texts, API errors, and rate limits gracefully.
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Replace empty strings with a single space to prevent OpenAI API 400 errors
            cleaned_batch = [t.strip() if t.strip() else " " for t in batch]

            try:
                response = client.embeddings.create(
                    input=cleaned_batch,
                    model=self._model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as exc:
                # Sanitize error message to prevent accidental key exposure
                err_msg = str(exc)
                if self._api_key and self._api_key in err_msg:
                    err_msg = err_msg.replace(self._api_key, "[REDACTED_API_KEY]")
                raise EmbeddingError(f"OpenAI embedding generation failed: {err_msg}") from exc

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single search query."""
        if not query or not query.strip():
            raise EmbeddingError("Query text cannot be empty")
        embeddings = self.embed_texts([query])
        if not embeddings:
            raise EmbeddingError("Failed to generate embedding for query")
        return embeddings[0]
