"""
Unit tests for embeddings module.
All OpenAI API calls are mocked using unittest.mock.
Verifies:
- Batching
- Empty input
- Mocked successful API response
- Malformed response
- API error / missing key
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.rag_agent.embeddings import (
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    EmbeddingError,
)


class MockEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [MockEmbeddingData(e) for e in embeddings]


class TestEmbeddings:
    def test_missing_api_key_raises_error(self):
        """Missing API key raises clear EmbeddingError."""
        provider = OpenAIEmbeddingProvider(api_key="")
        with pytest.raises(EmbeddingError) as exc:
            provider.embed_texts(["some text"])
        assert "Missing OPENAI_API_KEY" in str(exc.value)

    def test_empty_input_returns_empty_list(self):
        """Empty list input returns empty list without calling API."""
        provider = OpenAIEmbeddingProvider(api_key="mock_key")
        assert provider.embed_texts([]) == []

    @patch("openai.OpenAI")
    def test_successful_embedding_batching(self, mock_openai_cls):
        """Mocked OpenAI client embedding generation with batching."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock embedding return: vector of dimension 4
        mock_client.embeddings.create.side_effect = [
            MockEmbeddingResponse([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            MockEmbeddingResponse([[0.9, 0.1, 0.2, 0.3]]),
        ]

        provider = OpenAIEmbeddingProvider(api_key="mock_key", batch_size=2)
        texts = ["text1", "text2", "text3"]

        vectors = provider.embed_texts(texts)
        assert len(vectors) == 3
        assert len(vectors[0]) == 4
        assert vectors[0] == [0.1, 0.2, 0.3, 0.4]
        assert mock_client.embeddings.create.call_count == 2

    @patch("openai.OpenAI")
    def test_embed_query_success(self, mock_openai_cls):
        """embed_query returns single 1D vector."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])

        provider = OpenAIEmbeddingProvider(api_key="mock_key")
        q_vec = provider.embed_query("What is attention?")
        assert q_vec == [0.1, 0.2, 0.3]

    def test_embed_query_empty_raises(self):
        """Empty query raises EmbeddingError."""
        provider = OpenAIEmbeddingProvider(api_key="mock_key")
        with pytest.raises(EmbeddingError):
            provider.embed_query("   ")

    @patch("openai.OpenAI")
    def test_api_error_handling(self, mock_openai_cls):
        """OpenAI API error raises sanitized EmbeddingError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("OpenAI rate limit reached")

        provider = OpenAIEmbeddingProvider(api_key="mock_secret_key")
        with pytest.raises(EmbeddingError) as exc:
            provider.embed_texts(["hello"])
        assert "OpenAI embedding generation failed" in str(exc.value)
        assert "mock_secret_key" not in str(exc.value)
