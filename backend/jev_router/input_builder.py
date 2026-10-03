"""
Input builder transforming a full Document Profile into the JEV routing input contract.

Rules:
- Validates all required fields strictly.
- Never silently converts missing/invalid values into defaults.
- Only includes routing-relevant information.
- Excludes full text, raw page blocks, Docling internals, drawings, and candidate details.
"""

from typing import Any, Dict, List, Set, Union
from pydantic import ValidationError

from backend.jev_router.schemas import (
    JevInputState,
    DocumentNatureBlock,
    StructureBlock,
    VisualContentBlock,
    MathematicalContentBlock,
    ResearchCharacteristicsBlock,
)


class JevInputValidationError(ValueError):
    """Raised when document profile fails validation for JEV input construction."""
    pass


def _get_required_field(data: Dict[str, Any], field_path: str) -> Any:
    """Traverse and fetch field, raising JevInputValidationError if missing or None."""
    keys = field_path.split(".")
    curr = data
    for idx, k in enumerate(keys):
        if not isinstance(curr, dict) or k not in curr:
            raise JevInputValidationError(f"Missing required field in document profile: '{field_path}'")
        curr = curr[k]
        if curr is None:
            raise JevInputValidationError(f"Required field in document profile cannot be None: '{field_path}'")
    return curr


def build_jev_input(document_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a Document Profile into the compact JEV input contract.

    Returns the validated dictionary representation matching JevInputState schema.
    Raises JevInputValidationError if required fields are missing or invalid.
    """
    if not isinstance(document_profile, dict):
        raise JevInputValidationError(
            f"Expected document_profile to be a dict, got {type(document_profile).__name__}"
        )

    # 1. Document nature block
    page_count = _get_required_field(document_profile, "page_count")
    word_count = _get_required_field(document_profile, "word_count")
    document_nature = _get_required_field(
        document_profile, "content_characteristics.document_nature"
    )

    # 2. Structure block
    sections = document_profile.get("sections")
    if sections is None or not isinstance(sections, list):
        raise JevInputValidationError("Missing or invalid required field: 'sections' must be a list")

    section_count = len(sections)
    # Calculate max hierarchy depth across sections.
    # If no sections, max depth is 0.
    max_depth = 0
    for idx, s in enumerate(sections):
        if not isinstance(s, dict):
            raise JevInputValidationError(f"Section at index {idx} is not a dictionary")
        # Ensure level exists and is valid int
        s_level = s.get("level")
        if s_level is None or not isinstance(s_level, int):
            raise JevInputValidationError(f"Section at index {idx} missing valid 'level' integer")
        if s_level > max_depth:
            max_depth = s_level

    # Content characteristics extraction
    cc = document_profile.get("content_characteristics")
    if cc is None or not isinstance(cc, dict):
        raise JevInputValidationError("Missing or invalid required field: 'content_characteristics' must be a dict")

    has_methodology = _get_required_field(document_profile, "content_characteristics.has_methodology")
    has_experimental_results = _get_required_field(
        document_profile, "content_characteristics.has_experimental_results"
    )
    has_references = _get_required_field(document_profile, "content_characteristics.has_references")
    has_tables = _get_required_field(document_profile, "content_characteristics.has_tables")
    has_visual_content = _get_required_field(
        document_profile, "content_characteristics.has_visual_content"
    )
    has_mathematical_content = _get_required_field(
        document_profile, "content_characteristics.has_mathematical_content"
    )

    # 3. Visual content block
    figure_count = _get_required_field(document_profile, "figures.count")
    table_count = _get_required_field(document_profile, "tables.count")

    # 4. Mathematical content block
    equation_count = _get_required_field(document_profile, "equations.count")

    # Assemble raw dictionary
    raw_jev_input = {
        "document": {
            "page_count": page_count,
            "word_count": word_count,
            "document_nature": document_nature,
        },
        "structure": {
            "section_count": section_count,
            "max_section_depth": max_depth,
            "has_methodology": has_methodology,
            "has_experimental_results": has_experimental_results,
            "has_references": has_references,
        },
        "visual_content": {
            "figure_count": figure_count,
            "table_count": table_count,
            "has_visual_content": has_visual_content,
        },
        "mathematical_content": {
            "equation_count": equation_count,
            "has_mathematical_content": has_mathematical_content,
        },
        "research_characteristics": {
            "has_methodology": has_methodology,
            "has_experimental_results": has_experimental_results,
            "has_tables": has_tables,
            "has_references": has_references,
        },
    }

    # Strict validation via Pydantic model
    try:
        validated_state = JevInputState.model_validate(raw_jev_input)
    except ValidationError as e:
        raise JevInputValidationError(f"JEV input validation failed: {e}") from e

    return validated_state.model_dump()
