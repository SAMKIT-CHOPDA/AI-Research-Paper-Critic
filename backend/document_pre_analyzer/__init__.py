"""
Document Pre-Analyzer Subsystem.

Provides accurate structural and document analysis for research papers.
Produces a structured Document Profile for downstream processing.
"""

from .analyzer import analyze_document
from .config import PreAnalyzerConfig
from .pdf_parser import parse_pdf
from .section_analyzer import detect_sections
from .figure_analyzer import detect_figures
from .table_analyzer import detect_tables
from .equation_analyzer import detect_equations
from .reference_analyzer import detect_references

__all__ = [
    "analyze_document",
    "PreAnalyzerConfig",
    "parse_pdf",
    "detect_sections",
    "detect_figures",
    "detect_tables",
    "detect_equations",
    "detect_references",
]

