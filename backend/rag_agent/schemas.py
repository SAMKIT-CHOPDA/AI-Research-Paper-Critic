"""
Schemas for RAG Agent subsystem.

Pydantic models for:
- Extracted page text
- Document chunks with source metadata
- Retrieved evidence results with similarity score
- RAG Agent execution output
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Structured text extracted from a single PDF page."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Raw textual content of the page")
    word_count: int = Field(..., ge=0, description="Word count of the page text")
    char_count: int = Field(..., ge=0, description="Character count of the page text")


class DocumentChunk(BaseModel):
    """
    Research paper text chunk preserving source provenance.
    Never splits words in the middle.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. chunk_001")
    text: str = Field(..., description="Clean textual content of the chunk")
    page_start: int = Field(..., ge=1, description="Page where chunk starts (1-indexed)")
    page_end: int = Field(..., ge=1, description="Page where chunk ends (1-indexed)")
    word_count: int = Field(..., ge=0, description="Word count of the chunk")
    section: Optional[str] = Field(None, description="Section heading if detected, else None")


class RetrievedChunk(BaseModel):
    """
    Retrieved chunk evidence with similarity ranking.
    Citation-ready metadata preserved.
    """
    chunk_id: str = Field(..., description="Chunk identifier")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    page_start: int = Field(..., ge=1, description="Starting page")
    page_end: int = Field(..., ge=1, description="Ending page")
    section: Optional[str] = Field(None, description="Section heading if available")
    text: str = Field(..., description="Chunk text content")
    word_count: int = Field(..., ge=0, description="Word count of chunk")


class RAGResult(BaseModel):
    """
    Final JSON-serializable RAG agent execution result.
    """
    agent: str = Field("rag", description="Agent name identifier")
    enabled: bool = Field(..., description="Whether RAG was executed based on JEV router")
    level: Optional[str] = Field(None, description="JEV RAG capability tier (basic, medium, advanced)")
    query: Optional[str] = Field(None, description="Query evaluated if RAG was enabled")
    chunks_indexed: int = Field(0, ge=0, description="Total number of chunks indexed")
    top_k: int = Field(0, ge=0, description="Target top-k retrieved chunks")
    embedding_model: Optional[str] = Field(None, description="Embedding model identifier used")
    results: List[RetrievedChunk] = Field(default_factory=list, description="Ranked evidence chunks")
    retrieval_method: str = Field("cosine_similarity", description="Vector similarity method")
    elapsed_time_ms: Optional[float] = Field(None, description="Execution elapsed time in milliseconds")
    message: Optional[str] = Field(None, description="Status or rejection message if disabled")
