"""
Main entry point for Document Pre-Analyzer subsystem.

Provides:
analyze_document(pdf_path, config=None) -> dict
"""

import os
from typing import Dict, Any, Optional

from .config import PreAnalyzerConfig
from .pdf_parser import parse_pdf
from .ocr_analyzer import assess_document_digital_nature
from .layout_analyzer import LayoutAnalyzer
from .section_analyzer import detect_sections
from .figure_analyzer import detect_figures
from .table_analyzer import detect_tables
from .equation_analyzer import detect_equations
from .reference_analyzer import detect_references
from .profile_builder import build_document_profile


# Shared layout analyzer instance to avoid repeated Docling model loadings
_SHARED_LAYOUT_ANALYZER = None


def get_layout_analyzer(use_docling: bool = True) -> LayoutAnalyzer:
    global _SHARED_LAYOUT_ANALYZER
    if _SHARED_LAYOUT_ANALYZER is None:
        _SHARED_LAYOUT_ANALYZER = LayoutAnalyzer(use_docling=use_docling)
    return _SHARED_LAYOUT_ANALYZER


def analyze_document(
    pdf_path: str,
    config: Optional[PreAnalyzerConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze a research paper PDF and produce a structured Document Profile.

    Pipeline:
    1. PyMuPDF raw extraction (page geometry, blocks, spans, images, drawings)
    2. OCR assessment (determine born-digital, scanned, or mixed)
    3. Layout & Document structure extraction (Docling layout engine)
    4. Specialized detectors:
       - Sections & hierarchy
       - Figures & visual elements
       - Tables & structural grids
       - Mathematical equations & numbering
       - References & individual citations
       Each detector separates *candidates* (text-level evidence) from
       *confirmed* objects (independent structural/visual evidence) and keeps a
       rejection audit trail, so prose mentions such as
       "Table 3. Training took 3.5 days ..." can never be reported as an object.
    5. Evidence aggregation, cross-validation & confidence scoring
    6. Document Profile assembly (JSON-serializable)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    cfg = config or PreAnalyzerConfig()

    # Step 1: Raw extraction with PyMuPDF
    parsed_pdf_info = parse_pdf(pdf_path)

    # Step 2: Digital nature and OCR requirement assessment
    ocr_assessment = assess_document_digital_nature(
        parsed_pdf_info,
        min_words_per_page=cfg.ocr_min_page_words,
    )

    # Step 3: Layout analysis
    layout_data = {}
    if cfg.use_docling:
        layout_engine = get_layout_analyzer(use_docling=True)
        layout_data = layout_engine.analyze_layout(pdf_path)

    # Step 4: Specialized detectors with cross-validation
    sections = detect_sections(parsed_pdf_info, layout_data=layout_data, config=cfg)
    figures_data = detect_figures(parsed_pdf_info, layout_data=layout_data, config=cfg)
    tables_data = detect_tables(parsed_pdf_info, layout_data=layout_data, config=cfg)
    equations_data = detect_equations(parsed_pdf_info, layout_data=layout_data, config=cfg)
    references_data = detect_references(parsed_pdf_info, layout_data=layout_data)

    # Step 5 & 6: Profile compilation
    document_profile = build_document_profile(
        parsed_pdf_info=parsed_pdf_info,
        sections=sections,
        figures_data=figures_data,
        tables_data=tables_data,
        equations_data=equations_data,
        references_data=references_data,
        ocr_assessment=ocr_assessment,
    )

    return document_profile
