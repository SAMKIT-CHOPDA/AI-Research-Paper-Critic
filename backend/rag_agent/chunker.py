"""
Research-paper chunker.

Implements word/token-aware natural boundary chunking:
- Target: 500-800 words per chunk (default: 600)
- Overlap: 80-120 words (default: 100)
- Preserves paragraph/sentence/word boundaries
- NEVER splits in the middle of a word
- Tracks accurate page_start, page_end, and section headings
"""

import re
from typing import List, Optional, Tuple
from backend.rag_agent.config import (
    DEFAULT_TARGET_CHUNK_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    MIN_CHUNK_WORDS,
)
from backend.rag_agent.schemas import PageText, DocumentChunk

HEADING_REGEX = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[\s\.\-]+)?([A-Z][A-Za-z0-9\s,\-–—]{2,60})$"
)


def _split_into_paragraphs_with_page(pages: List[PageText]) -> List[Tuple[str, int]]:
    """Split pages into paragraphs while preserving 1-indexed page provenance."""
    paragraphs: List[Tuple[str, int]] = []
    for page in pages:
        if not page.text:
            continue
        raw_paras = re.split(r"\n\s*\n+", page.text)
        for p in raw_paras:
            clean_p = p.strip()
            if clean_p:
                paragraphs.append((clean_p, page.page_number))
    return paragraphs


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly without splitting common abbreviations."""
    protected = text
    subs = {
        "et al.": "__ET_AL__",
        "e.g.": "__E_G__",
        "i.e.": "__I_E__",
        "Fig.": "__FIG__",
        "Eq.": "__EQ__",
        "Ref.": "__REF__",
        "vs.": "__VS__",
    }
    for orig, rep in subs.items():
        protected = protected.replace(orig, rep)

    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for s in raw_sentences:
        restored = s
        for orig, rep in subs.items():
            restored = restored.replace(rep, orig)
        clean_s = restored.strip()
        if clean_s:
            sentences.append(clean_s)
    return sentences or [text]


def _detect_heading(para_text: str) -> Optional[str]:
    """Detect if a paragraph looks like a section heading."""
    first_line = para_text.split("\n")[0].strip()
    words = first_line.split()
    if 1 <= len(words) <= 8 and len(first_line) < 80:
        if HEADING_REGEX.match(first_line):
            return first_line
    return None


def chunk_paper(
    pages: List[PageText],
    target_words: int = DEFAULT_TARGET_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[DocumentChunk]:
    """
    Chunk extracted pages into semantic research chunks.

    Parameters:
    - pages: List of PageText objects from text extractor
    - target_words: Target word size per chunk (default 600)
    - overlap_words: Number of words to carry over into the next chunk (default 100)
    - min_chunk_words: Minimum words to form a standalone chunk (default 50)

    Returns:
    - List of DocumentChunk instances with chunk_id, page_start, page_end, word_count, and section.
    """
    if not pages:
        return []

    # Defensive parameter normalization
    if target_words <= 0:
        target_words = DEFAULT_TARGET_CHUNK_WORDS
    overlap_words = max(0, min(overlap_words, target_words - 1))
    min_chunk_words = max(0, min(min_chunk_words, target_words))

    paras_with_page = _split_into_paragraphs_with_page(pages)
    if not paras_with_page:
        return []

    chunks: List[DocumentChunk] = []
    chunk_index = 1

    current_words: List[str] = []
    current_pages: List[int] = []
    current_section: Optional[str] = None
    active_section: Optional[str] = None

    for para_text, page_num in paras_with_page:
        heading = _detect_heading(para_text)
        if heading:
            active_section = heading

        para_words = para_text.split()
        if not para_words:
            continue

        # Oversized paragraphs are first split on sentence boundaries ...
        if len(para_words) > target_words:
            sub_texts = _split_into_sentences(para_text)
        else:
            sub_texts = [para_text]

        # ... then flattened into word slices bounded by target_words.
        # Slicing occurs strictly on whitespace boundaries, so tokens are never severed.
        sub_elements: List[Tuple[List[str], int]] = []
        for sub_text in sub_texts:
            sub_words = sub_text.split()
            for start in range(0, len(sub_words), target_words):
                sub_elements.append((sub_words[start : start + target_words], page_num))

        for element_words, element_page in sub_elements:
            if len(current_words) + len(element_words) > target_words and len(current_words) >= min_chunk_words:
                chunk_text = " ".join(current_words)
                p_start = min(current_pages) if current_pages else element_page
                p_end = max(current_pages) if current_pages else element_page

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{chunk_index:03d}",
                        text=chunk_text,
                        page_start=p_start,
                        page_end=p_end,
                        word_count=len(current_words),
                        section=current_section or active_section,
                    )
                )
                chunk_index += 1

                # The emitted chunk consumed the pending section label; the next chunk
                # is re-labelled from whatever heading is active at that point.
                current_section = None

                if overlap_words > 0 and len(current_words) > overlap_words:
                    overlap_slice = current_words[-overlap_words:]
                    overlap_pages = [current_pages[-1]] if current_pages else [element_page]
                    current_words = list(overlap_slice)
                    current_pages = list(overlap_pages)
                else:
                    current_words = []
                    current_pages = []

            current_words.extend(element_words)
            current_pages.append(element_page)
            if not current_section and active_section:
                current_section = active_section

    if current_words:
        chunk_text = " ".join(current_words)
        p_start = min(current_pages) if current_pages else 1
        p_end = max(current_pages) if current_pages else p_start

        if len(current_words) < min_chunk_words and chunks:
            prev = chunks[-1]
            merged_text = prev.text + "\n\n" + chunk_text
            chunks[-1] = DocumentChunk(
                chunk_id=prev.chunk_id,
                text=merged_text,
                page_start=prev.page_start,
                page_end=max(prev.page_end, p_end),
                word_count=len(merged_text.split()),
                section=prev.section,
            )
        else:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{chunk_index:03d}",
                    text=chunk_text,
                    page_start=p_start,
                    page_end=p_end,
                    word_count=len(current_words),
                    section=current_section or active_section,
                )
            )

    return chunks
