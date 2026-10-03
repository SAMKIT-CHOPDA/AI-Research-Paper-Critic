"""
Confidence calculation models and scoring formulas.

Every detected object (table, figure, equation, section, reference) receives a
confidence score between 0.0 and 1.0 calculated from explicit signals and evidence.
"""

from typing import Dict, Any, List


def calculate_table_confidence(
    has_caption: bool,
    caption_matches_table_num: bool,
    has_grid_or_drawings: bool,
    has_docling_match: bool,
    has_pymupdf_table: bool,
    is_author_or_meta_block: bool = False,
    is_body_text_reference: bool = False,
) -> float:
    """
    Calculate confidence score for a table detection candidate.

    Evidence weights:
    - Caption present at start of block: +0.25
    - Caption format standard ('Table X:'): +0.10
    - Ruling lines / vector drawings in table region: +0.25
    - Docling structure detection match: +0.25
    - PyMuPDF find_tables match: +0.10
    - Negative indicator: Author/meta block falsely flagged: -0.60
    - Negative indicator: In-text body citation / reference: -0.50

    Note: a caption-only candidate (no structural evidence) tops out at 0.35,
    which is deliberately below PreAnalyzerConfig.min_table_confidence (0.40)
    so that prose mentions such as "Table 3. Training took 3.5 days..." can
    never be promoted to a confirmed table.
    """
    if is_author_or_meta_block:
        return 0.10

    score = 0.0
    if has_caption:
        score += 0.25
    if caption_matches_table_num:
        score += 0.10
    if has_grid_or_drawings:
        score += 0.25
    if has_docling_match:
        score += 0.25
    if has_pymupdf_table:
        score += 0.10
    if is_body_text_reference:
        score -= 0.50

    return min(1.0, max(0.0, round(score, 2)))


def calculate_figure_confidence(
    has_caption: bool,
    has_raster_image: bool,
    has_vector_drawings: bool,
    has_docling_picture: bool,
    is_body_text_reference: bool = False,
) -> float:
    """
    Calculate confidence score for a figure detection candidate.

    Evidence weights:
    - Caption present at start of block: +0.35
    - Raster image (XObject) present in region: +0.30
    - Docling picture detector identified region: +0.25
    - Vector drawings present in region: +0.15
    - Negative indicator: In-text body citation / reference: -0.50
    """
    score = 0.0
    if has_caption:
        score += 0.35
    if has_raster_image:
        score += 0.30
    if has_docling_picture:
        score += 0.25
    if has_vector_drawings and not has_raster_image:
        score += 0.15
    if is_body_text_reference:
        score -= 0.50

    return min(1.0, max(0.0, round(score, 2)))


def calculate_equation_confidence(
    has_numbering: bool,
    has_math_font: bool,
    has_docling_formula: bool,
    has_math_operators: bool,
) -> float:
    """
    Calculate confidence score for an equation candidate.

    Evidence weights:
    - Numbering tag e.g. '(1)', '(2)': +0.35
    - Mathematical font (CMR, CMMI, CMSY, etc.): +0.30
    - Docling formula recognition: +0.25
    - Mathematical symbols/operators: +0.15
    """
    score = 0.0
    if has_numbering:
        score += 0.35
    if has_math_font:
        score += 0.30
    if has_docling_formula:
        score += 0.25
    if has_math_operators:
        score += 0.15

    return min(1.0, max(0.0, round(score, 2)))


def calculate_section_confidence(
    matches_standard_title: bool,
    has_numbering: bool,
    has_distinct_font_size: bool,
    is_bold: bool,
    has_docling_header: bool,
    has_vertical_spacing: bool = False,
    is_subsection_numbered: bool = False,
) -> float:
    """
    Calculate confidence score for a detected section heading.

    The weights are additive and deliberately sum to exactly 1.00 so the score
    is a transparent evidence fraction rather than an inflated heuristic:

    - Matches standard research paper section name: +0.20
    - Numbered section e.g. '1 Introduction': +0.15
    - Subsection numbering hierarchy e.g. '3.1', '3.2.1': +0.15
    - Font size larger than body text: +0.15
    - Bold font flags or bold font name: +0.15
    - Docling section_header classification: +0.15
    - Vertical spacing / margin isolation around the heading: +0.05
    """
    score = 0.0
    if matches_standard_title:
        score += 0.20
    if has_numbering:
        score += 0.15
    if is_subsection_numbered:
        score += 0.15
    if has_distinct_font_size:
        score += 0.15
    if is_bold:
        score += 0.15
    if has_docling_header:
        score += 0.15
    if has_vertical_spacing:
        score += 0.05

    return min(1.0, max(0.0, round(score, 2)))


def calculate_reference_confidence(
    has_section_header: bool,
    entry_count: int,
    is_sequential: bool,
    has_standard_citation_format: bool,
) -> float:
    """
    Calculate confidence for document-level reference detection.

    Evidence weights:
    - Dedicated 'References' / 'Bibliography' header found: +0.40
    - Substantial reference entries (> 3 entries): +0.25
    - Sequential numbering verification (e.g. [1] .. [N]): +0.20
    - Recognized citation format: +0.15
    """
    score = 0.0
    if has_section_header:
        score += 0.40
    if entry_count >= 5:
        score += 0.25
    elif entry_count > 0:
        score += 0.15
    if is_sequential:
        score += 0.20
    if has_standard_citation_format:
        score += 0.15

    return min(1.0, max(0.0, round(score, 2)))

