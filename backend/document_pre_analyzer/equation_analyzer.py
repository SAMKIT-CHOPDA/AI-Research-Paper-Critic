"""
Equation analyzer detecting numbered and display equations.

Separates equation_candidates from confirmed_equations so a numbering tag alone
never declares an equation:

1. Candidates
   - Docling ``formula`` items
   - text blocks ending in an equation tag such as ``(1)``
     (``Table 2 (2014)`` styled trailing parentheses are filtered out because a
     math font or math operator must also be present)
2. Confirmation requires independent structure:
   - mathematical font families (CMMI, CMSY, CMBX, CMR, Math, Symbol, ...), or
   - Docling formula recognition for the same equation number
   Candidates with a numbering tag but no math typography are kept as rejected
   candidates; trailing tags without any math evidence at all are recorded under
   ``rejected_equation_mentions``.

LaTeX/text representations are preserved for downstream critique.
"""

import re
from typing import Dict, Any, List, Optional

from .confidence import calculate_equation_confidence
from .config import PreAnalyzerConfig

EQUATION_TAG_REGEX = re.compile(r"\s*\((\d{1,2})\)\s*$")
MATH_OPERATORS = [
    "=", "+", "×", "·", "/", "√", "∑", "∫", "≤", "≥", "∈",
    "softmax", "argmin", "argmax",
]
MATH_FONT_PREFIXES = [
    "cmmi", "cmsy", "cmbx", "cmr", "math", "symbol", "msam", "msbm",
]


def _collect_block_text(block: Dict[str, Any]) -> str:
    """Flatten a PyMuPDF text block into a single normalized string."""
    return " ".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    ).strip()


def _block_has_math_font(block: Dict[str, Any]) -> bool:
    """True when any span uses a mathematical font family."""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            font_name = (span.get("font") or "").lower()
            if any(prefix in font_name for prefix in MATH_FONT_PREFIXES):
                return True
    return False


def _block_fonts(block: Dict[str, Any]) -> List[str]:
    """Sorted list of distinct font names used inside a block."""
    fonts = set()
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            name = span.get("font")
            if name:
                fonts.add(name)
    return sorted(fonts)



def detect_equations(
    parsed_pdf_info: Dict[str, Any],
    layout_data: Dict[str, Any] = None,
    config: Optional[PreAnalyzerConfig] = None,
) -> Dict[str, Any]:
    """
    Detect numbered and prominent display equations with representations.

    ``items``/``count`` remain the confirmed equations (backward compatible);
    ``equation_candidates`` holds every candidate with its evidence and
    ``rejected_equation_mentions`` documents numbering tags that had no math
    evidence at all.
    """
    cfg = config or PreAnalyzerConfig()
    min_confidence = cfg.min_equation_confidence

    docling_formulas = (layout_data or {}).get("formulas", []) or []
    candidates: List[Dict[str, Any]] = []
    rejected_mentions: List[Dict[str, Any]] = []
    seen_eq_numbers = set()
    docling_numbers = set()

    # 1. Docling formula recognition is structural evidence on its own.
    for formula in docling_formulas:
        formula_text = (formula.get("text") or "").strip()
        formula_page = formula.get("page")
        formula_bbox = formula.get("bbox")

        tag_match = EQUATION_TAG_REGEX.search(formula_text)
        equation_number = tag_match.group(1) if tag_match else None
        has_math_operators = any(op in formula_text for op in MATH_OPERATORS)

        if equation_number:
            seen_eq_numbers.add(equation_number)
            docling_numbers.add(equation_number)

        confidence = calculate_equation_confidence(
            has_numbering=equation_number is not None,
            has_math_font=True,
            has_docling_formula=True,
            has_math_operators=has_math_operators,
        )

        candidates.append({
            "id": f"eq_{equation_number}" if equation_number
                  else f"eq_p{formula_page}_{len(candidates) + 1}",
            "page": formula_page,
            "equation_number": equation_number,
            "representation": formula_text,
            "bbox": formula_bbox,
            "status": "confirmed",
            "structure_source": "docling_formula",
            "evidence": {
                "has_numbering": equation_number is not None,
                "math_fonts_present": True,
                "math_operators_present": has_math_operators,
                "docling_formula_detected": True,
            },
            "confidence": confidence,
        })

    # 2. PyMuPDF text blocks: a trailing tag plus math typography is required.
    for page_data in parsed_pdf_info.get("pages", []):
        page_num = page_data["page_number"]
        for block_index, block in enumerate(page_data.get("blocks", [])):
            if "lines" not in block:
                continue
            block_text = _collect_block_text(block)
            if not block_text:
                continue

            tag_match = EQUATION_TAG_REGEX.search(block_text)
            if not tag_match:
                continue

            equation_number = tag_match.group(1)
            if equation_number in seen_eq_numbers:
                continue

            has_math_font = _block_has_math_font(block)
            has_math_operators = any(op in block_text for op in MATH_OPERATORS)

            if not has_math_font and not has_math_operators:
                rejected_mentions.append({
                    "page": page_num,
                    "numbering_tag": equation_number,
                    "text": block_text[:300],
                    "reason": "numbering_tag_without_math_font_or_operators",
                })
                continue

            confidence = calculate_equation_confidence(
                has_numbering=True,
                has_math_font=has_math_font,
                has_docling_formula=False,
                has_math_operators=has_math_operators,
            )
            # Math fonts are typographic proof; operators such as "=" alone are not.
            confirmed = has_math_font and confidence >= min_confidence
            if confirmed:
                seen_eq_numbers.add(equation_number)

            candidates.append({
                "id": f"eq_{equation_number}" if confirmed
                      else f"eq_p{page_num}_{block_index}",
                "page": page_num,
                "equation_number": equation_number,
                "representation": block_text,
                "bbox": list(block.get("bbox", [])),
                "status": "confirmed" if confirmed else "candidate",
                "structure_source": "math_fonts" if confirmed else None,
                "evidence": {
                    "has_numbering": True,
                    "math_fonts_present": has_math_font,
                    "math_operators_present": has_math_operators,
                    "docling_formula_detected": equation_number in docling_numbers,
                    "fonts": _block_fonts(block),
                },
                "confidence": confidence,
            })


    # Deduplicate confirmed equations by equation number (highest confidence wins).
    confirmed_by_number: Dict[str, Dict[str, Any]] = {}
    confirmed_unnumbered: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["status"] != "confirmed":
            continue
        number = candidate["equation_number"]
        if not number:
            confirmed_unnumbered.append(candidate)
            continue
        current = confirmed_by_number.get(number)
        if current is None or candidate["confidence"] > current["confidence"]:
            confirmed_by_number[number] = candidate

    def _sort_key(equation: Dict[str, Any]) -> Any:
        page = equation.get("page") or 0
        number = equation.get("equation_number") or ""
        return (page, int(number) if str(number).isdigit() else 999)

    confirmed_equations = sorted(
        list(confirmed_by_number.values()) + confirmed_unnumbered,
        key=_sort_key,
    )
    rejected_candidates = [c for c in candidates if c["status"] != "confirmed"]

    return {
        "count": len(confirmed_equations),
        "items": confirmed_equations,
        "confirmed_equations": confirmed_equations,
        "equation_candidates": candidates,
        "candidate_count": len(candidates),
        "rejected_candidates": rejected_candidates,
        "rejected_candidate_count": len(rejected_candidates),
        "rejected_equation_mentions": rejected_mentions,
        "rejected_equation_mention_count": len(rejected_mentions),
    }
