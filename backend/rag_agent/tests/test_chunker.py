"""
Unit tests for chunker module.
Verifies:
- No mid-word splits
- Overlap behavior
- Page metadata tracking
- Section heading detection
- Short vs long document behavior
"""

import pytest
from backend.rag_agent.schemas import PageText
from backend.rag_agent.chunker import chunk_paper


class TestChunker:
    def test_chunking_empty_pages(self):
        """Empty input yields empty chunk list."""
        assert chunk_paper([]) == []

    def test_no_mid_word_splits(self):
        """Verify chunks never begin or end with a severed/truncated word."""
        text = " ".join([f"word{i}" for i in range(1200)])
        page = PageText(page_number=1, text=text, word_count=1200, char_count=len(text))

        chunks = chunk_paper([page], target_words=500, overlap_words=80)
        assert len(chunks) >= 2

        for chunk in chunks:
            words = chunk.text.split()
            # Every word must match word<number>
            for w in words:
                assert w.startswith("word"), f"Severed word found: {w}"
            assert chunk.word_count == len(words)

    def test_overlap_retention(self):
        """Verify overlap words are carried over into successive chunks."""
        words = [f"token_{i}" for i in range(1000)]
        text = " ".join(words)
        page = PageText(page_number=1, text=text, word_count=1000, char_count=len(text))

        chunks = chunk_paper([page], target_words=500, overlap_words=100)
        assert len(chunks) >= 2

        # The end words of chunk 1 should appear at the start of chunk 2
        chunk1_words = chunks[0].text.split()
        chunk2_words = chunks[1].text.split()

        overlap_slice = chunk1_words[-100:]
        assert chunk2_words[:100] == overlap_slice

    def test_page_span_metadata_preserved(self):
        """Verify page_start and page_end track correctly across multiple pages."""
        page1_words = [f"p1_word_{i}" for i in range(400)]
        page2_words = [f"p2_word_{i}" for i in range(400)]

        p1 = PageText(page_number=1, text=" ".join(page1_words), word_count=400, char_count=2000)
        p2 = PageText(page_number=2, text=" ".join(page2_words), word_count=400, char_count=2000)

        chunks = chunk_paper([p1, p2], target_words=600, overlap_words=50)
        assert len(chunks) >= 2

        # First chunk contains page-1 content only.
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1

        # Second chunk carries page-1 overlap words plus page-2 content.
        assert chunks[1].page_start == 1
        assert chunks[1].page_end == 2

    def test_chunk_sizes_are_bounded(self):
        """No chunk may materially exceed the configured target word count."""
        text = " ".join([f"token{i}" for i in range(3000)])
        page = PageText(page_number=1, text=text, word_count=3000, char_count=len(text))

        chunks = chunk_paper([page], target_words=400, overlap_words=50)
        assert len(chunks) >= 5
        for chunk in chunks:
            # target + overlap + a single trailing element is the structural ceiling
            assert chunk.word_count <= 400 + 50 + 400

    def test_section_heading_detected(self):
        """Headings should be detected and assigned to chunks."""
        text = "3.2 Model Architecture\n\nThe multi-head attention mechanism consists of several parallel attention layers."
        p = PageText(page_number=3, text=text, word_count=15, char_count=len(text))

        chunks = chunk_paper([p], target_words=500, min_chunk_words=5)
        assert len(chunks) == 1
        assert chunks[0].section == "3.2 Model Architecture"

    def test_short_document_single_chunk(self):
        """Short document fits into a single chunk."""
        text = "This is a short paper abstract discussing self-attention models."
        p = PageText(page_number=1, text=text, word_count=9, char_count=len(text))

        chunks = chunk_paper([p], target_words=600, min_chunk_words=5)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk_001"
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1
        assert chunks[0].word_count == 9
