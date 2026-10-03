"""
Configuration for the RAG Agent subsystem.

Defines:
- Embedding model configuration and defaults
- Chunking parameters (word counts, overlaps)
- JEV capability level to retrieval top_k mapping
- Storage defaults
"""

import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# OpenAI Embedding configuration
# Default to text-embedding-3-small (cost-effective, 1536 dims, supported across OpenAI accounts)
DEFAULT_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))

# Chunking configuration (Word-based natural boundary chunking)
# Target: 500-800 words per chunk, 80-120 word overlap
DEFAULT_TARGET_CHUNK_WORDS: int = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "600"))
DEFAULT_CHUNK_OVERLAP_WORDS: int = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "100"))
MIN_CHUNK_WORDS: int = int(os.getenv("RAG_MIN_CHUNK_WORDS", "50"))

# JEV Router Level -> Retrieval Configuration mapping
# Engineering configuration mapping abstract JEV model capability levels to retrieval depth
# Note: These are engineering choices, not scientifically calibrated absolutes.
LEVEL_TOP_K_MAPPING: Dict[str, int] = {
    "basic": 3,
    "medium": 5,
    "advanced": 8,
}
DEFAULT_FALLBACK_TOP_K: int = 5
