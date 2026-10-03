"""
Offline unit tests for analysis prompt construction.

Verifies:
- the scope boundaries are stated (no final critique, no paper-level score)
- provenance and evidence-discipline rules are present
- every dimension of the contract has a focus block and appears in the contract
- the evidence bundle, the citable source ids, the deterministic gaps and the
  applied context limits are rendered into the prompt
- a machine-readable table value is rendered only when the Vision Agent marked it
  authoritative
"""

from backend.analysis_agent.analyzer import DIMENSION_KEYS, REQUIRED_PAYLOAD_KEYS
from backend.analysis_agent.config import AnalysisConfig
from backend.analysis_agent.context_builder import build_evidence_bundle
from backend.analysis_agent.prompts import (
    DIMENSION_FOCUS,
    OUTPUT_CONTRACT,
    RULE_BLOCK,
    build_analysis_prompt,
    build_context_block,
    render_gaps_block,
    render_section_block,
    render_source_index_block,
    render_truncations_block,
)


def profile() -> dict:
    return {
        "filename": "synthetic.pdf",
        "page_count": 11,
        "word_count": 4990,
        "text_length": 36309,
        "metadata": {"title": "Synthetic Paper", "author": "A. Author"},
        "content_characteristics": {"has_methodology": True},
        "sections": [
            {"title": "Abstract", "normalized_title": "abstract", "page": 1, "level": 1},
            {"title": "6 Results", "normalized_title": "results", "number": "6", "page": 8, "level": 1},
        ],
        "figures": {
            "count": 1,
            "confirmed_figures": [
                {
                    "id": "figure_1",
                    "figure_number": "1",
                    "page": 3,
                    "caption": "Figure 1: the model architecture.",
                    "confidence": 0.9,
                }
            ],
        },
        "tables": {
            "count": 1,
            "confirmed_tables": [
                {
                    "id": "table_2",
                    "table_number": "2",
                    "page": 8,
                    "caption": "Table 2: reported BLEU scores.",
                    "rows": 11,
                    "columns": 5,
                    "structure_source": "docling",
                    "confidence": 0.9,
                }
            ],
        },
        "equations": {
            "count": 1,
            "confirmed_equations": [
                {
                    "id": "eq_1",
                    "page": 4,
                    "equation_number": "1",
                    "representation": "Attention(Q, K, V ) = softmax(QKT )V",
                    "structure_source": "docling_formula",
                    "confidence": 1.0,
                }
            ],
        },
        "references": {
            "has_references": True,
            "count": 32,
            "start_page": 10,
            "confidence": 1.0,
            "is_sequential": True,
        },
    }


def rag() -> dict:
    return {
        "agent": "rag",
        "enabled": True,
        "results": [
            {
                "chunk_id": "chunk_001",
                "score": 0.81,
                "page_start": 8,
                "page_end": 9,
                "section": "6 Results",
                "text": "The Transformer achieves 28.4 BLEU on the English-to-German task.",
                "word_count": 11,
            }
        ],
    }


def vision(authoritative: bool = False) -> dict:
    return {
        "agent": "vision",
        "enabled": True,
        "results": [
            {
                "asset_id": "table_002",
                "asset_type": "table",
                "page_number": 8,
                "section": "6 Results",
                "caption": "Table 2: reported BLEU scores.",
                "observation": "Five columns and eleven rows are visible.",
                "interpretation": "The first row appears to report the highest score.",
                "key_elements": ["columns: Model, BLEU"],
                "reported_relationships": ["row 1 appears largest"],
                "supports_claims": ["the model appears competitive"],
                "uncertainties": ["small text limits precision"],
                "caption_consistency": {"status": "consistent", "explanation": "matches"},
                "numeric_authority": (
                    "document_profile_machine_readable"
                    if authoritative
                    else "vision_interpretation_only"
                ),
                "structured_evidence": {
                    "authoritative_for_exact_values": authoritative,
                    "headers": ["Model", "BLEU"],
                    "rows": [["Transformer", "28.4"]],
                },
            }
        ],
    }


def full_bundle(authoritative: bool = False):
    return build_evidence_bundle(
        profile(), rag(), vision(authoritative=authoritative)
    )


def test_scope_boundaries_are_stated():
    assert "NOT writing the final critique" in RULE_BLOCK
    assert "overall numeric score" in RULE_BLOCK
    assert "Do not rank the paper" in RULE_BLOCK
    assert "Return ONLY one JSON object" in RULE_BLOCK


def test_provenance_and_evidence_discipline_rules_are_stated():
    assert "Never invent evidence" in RULE_BLOCK
    assert "Citable source ids" in RULE_BLOCK
    assert "Never invent numbers" in RULE_BLOCK
    assert "Never fabricate confidence" in RULE_BLOCK
    for status in (
        "supported",
        "partially_supported",
        "unsupported",
        "unclear",
        "insufficient_evidence",
    ):
        assert status in RULE_BLOCK


def test_limitations_must_stay_separated():
    assert "author_stated" in RULE_BLOCK
    assert "analyst_identified" in RULE_BLOCK
    assert "Never attribute your own" in RULE_BLOCK


def test_contract_lists_every_required_key():
    for key in REQUIRED_PAYLOAD_KEYS:
        assert f'"{key}"' in OUTPUT_CONTRACT


def test_every_dimension_has_a_focus_block():
    assert set(DIMENSION_FOCUS) == set(REQUIRED_PAYLOAD_KEYS)
    prompt = build_analysis_prompt(full_bundle())
    for dimension in DIMENSION_KEYS:
        assert f"### {dimension}" in prompt
    assert "### claim_evidence_matrix" in prompt
    assert "### limitations" in prompt
    assert "### evidence_gaps" in prompt


def test_prompt_sections_and_evidence_are_rendered():
    prompt = build_analysis_prompt(full_bundle())

    assert "EVIDENCE BUNDLE" in prompt
    assert "ANALYSIS TASK" in prompt
    assert "OUTPUT CONTRACT" in prompt

    # Paper facts, structure, inventory and references.
    assert "title (metadata): Synthetic Paper" in prompt
    assert "pages: 11" in prompt
    assert "- [section_002] depth 1 | p.8 | 6 Results" in prompt
    assert "- [figure_1] Figure 1 | p.3 | Figure 1: the model architecture." in prompt
    assert "- [table_2] Table 2 | p.8 | reported shape 11x5" in prompt
    assert "- [eq_1] Equation 1 | p.4 | Attention(Q, K, V ) = softmax(QKT )V" in prompt
    assert "references detected: 32" in prompt

    # Retrieved text becomes the only verbatim evidence.
    assert "chunk_001" in prompt
    assert "28.4 BLEU" in prompt
    assert "ONLY source of the paper's verbatim wording" in prompt

    # Visual evidence keeps observation and interpretation apart.
    assert "observation: Five columns and eleven rows are visible." in prompt
    assert "interpretation: The first row appears to report the highest score." in prompt
    assert "must never be reported as a visible fact" in prompt
    assert "stated uncertainties: small text limits precision" in prompt


def test_section_numbers_are_not_printed_twice():
    custom = profile()
    custom["sections"] = [
        {"title": "3 Method", "number": "3", "page": 5, "level": 1},
        {"title": "Method", "number": "3", "page": 5, "level": 1},
    ]
    block = render_section_block(build_evidence_bundle(custom))

    assert "- [section_001] depth 1 | p.5 | 3 Method" in block
    assert "- [section_002] depth 1 | p.5 | 3 Method" in block
    assert "3 3 Method" not in block


def test_source_index_block_lists_ids_with_types_and_pages():
    block = render_source_index_block(full_bundle())

    assert "Citable source ids" in block
    assert "section_001 | source_type=section | p.1 | Abstract" in block
    assert "chunk_001 | source_type=text_chunk | pp.8-9" in block
    assert "visual_table_002 | source_type=visual_asset | p.8" in block
    assert "references_summary | source_type=reference | p.10" in block


def test_deterministic_gaps_are_stated_in_the_prompt():
    bundle = build_evidence_bundle(profile())  # no RAG, no Vision
    block = render_gaps_block(bundle)
    prompt = build_analysis_prompt(bundle)

    assert "Evidence gaps already established by the pipeline" in block
    assert "(rag_unavailable)" in block
    assert "(vision_unavailable)" in block
    assert "affects: methodology" in block
    assert "record them in \"evidence_gaps\"" in prompt


def test_no_gap_block_still_tells_the_model_gaps_are_complete():
    block = render_gaps_block(full_bundle())
    assert "all upstream evidence was available" in block


def test_context_limits_are_rendered_when_present():
    bundle = build_evidence_bundle(profile(), config=AnalysisConfig(max_sections=1))
    block = render_truncations_block(bundle)

    assert "Context limits applied while assembling this evidence" in block
    assert "sections: 2 available, 1 included" in block
    assert "ANALYSIS_MAX_SECTIONS" in block


def test_empty_truncation_block_when_nothing_was_dropped():
    assert render_truncations_block(full_bundle()) == ""


def test_machine_readable_values_only_appear_when_authoritative():
    marker = "machine-readable values (authoritative for exact numbers)"
    not_authoritative = build_context_block(full_bundle(authoritative=False))
    assert marker not in not_authoritative

    authoritative = build_context_block(full_bundle(authoritative=True))
    assert marker in authoritative
    assert "Transformer | 28.4" in authoritative


def test_prompt_is_built_only_from_supplied_evidence():
    bundle = full_bundle()
    context = build_context_block(bundle)

    # Nothing outside the bundle is invented: no raw-LaTeX dump, no full paper text.
    assert bundle.filename in context
    assert "None" not in context.split("Citable source ids")[1]
