"""
Configuration settings for Document Pre-Analyzer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreAnalyzerConfig:
    """Configuration parameters for the Document Pre-Analyzer pipeline."""

    # Docling layout engine usage
    use_docling: bool = True
    docling_do_ocr: bool = False
    docling_do_table_structure: bool = True

    # OCR fallback thresholds
    ocr_min_page_words: int = 15  # Pages with fewer words than this may be considered scanned/sparse
    ocr_min_char_count: int = 50
    ocr_force_enabled: bool = False

    # Heading detection thresholds
    heading_font_size_ratio_min: float = 1.05  # Ratio above body font size
    heading_max_words: int = 25

    # Table detection thresholds
    min_table_confidence: float = 0.40
    # A caption alone is never enough: tables must also show structural evidence
    # (ruling lines in the region, a PyMuPDF find_tables region, or a Docling
    # recognized table). Set to False only for degraded diagnostics runs.
    require_structural_evidence: bool = True
    # Vertical distance (points) below a caption in which table structures are searched
    table_caption_search_band_pts: float = 450.0

    # Figure detection thresholds
    min_figure_confidence: float = 0.40

    # Equation detection thresholds
    min_equation_confidence: float = 0.35

    # Section detection: relative line-height multiple used to decide whether a
    # heading is isolated by vertical whitespace from surrounding paragraphs
    heading_spacing_line_multiple: float = 1.15
