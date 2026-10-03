"""
Offline test support for the Critique Engine (Phase 6).

Provides a fully synthetic but internally consistent upstream pipeline
(document profile, JEV routing state, RAG result, Vision result, Analysis result)
plus a builder for a contract-compliant model response that only cites ids that
really exist. Tests use these instead of a network or an API key.
"""

import json
from typing import Any, Dict, List, Optional

from backend.critique_engine.llm_client import BaseCritiqueClient
from backend.critique_engine.schemas import (
    ALWAYS_REQUIRED_SECTIONS,
    SECTION_TITLES,
    SynthesisContext,
)

MODEL_MAP = {
    "basic": "mock-critique-basic",
    "medium": "mock-critique-medium",
    "advanced": "mock-critique-advanced",
}


class MockCritiqueClient(BaseCritiqueClient):
    """Returns a caller-supplied response and records every call."""

    def __init__(self, response: str = "{}", error: Optional[Exception] = None):
        self.response = response
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock-critique"

    def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response_schema": response_schema,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def synthetic_profile() -> Dict[str, Any]:
    """A frozen-style document profile with real, quotable ids."""
    return {
        "filename": "synthetic-study.pdf",
        "page_count": 11,
        "word_count": 4200,
        "metadata": {"title": "A Synthetic Study Of Structured Evidence"},
        "sections": [
            {"source_id": "section_001", "title": "Introduction", "page": 1},
            {"source_id": "section_002", "title": "Method", "page": 4},
            {"source_id": "section_003", "title": "Results", "page": 8},
        ],
        "figures": {
            "confirmed_figures": [
                {
                    "source_id": "figure_001",
                    "page": 6,
                    "caption": "Encoder-decoder architecture",
                }
            ]
        },
        "tables": {
            "confirmed_tables": [
                {"source_id": "table_001", "page": 8, "caption": "Main results"}
            ]
        },
        "equations": {
            "confirmed_equations": [
                {"source_id": "equation_001", "page": 5, "caption": "Loss"}
            ]
        },
        "references": {"count": 30},
    }


def synthetic_rag_result() -> Dict[str, Any]:
    return {
        "agent": "rag",
        "enabled": True,
        "level": "medium",
        "results": [
            {
                "chunk_id": "chunk_001",
                "score": 0.82,
                "page_start": 2,
                "page_end": 2,
                "section": "Introduction",
                "text": "The paper states that it addresses sequence transduction.",
            },
            {
                "chunk_id": "chunk_002",
                "score": 0.77,
                "page_start": 8,
                "page_end": 9,
                "section": "Results",
                "text": "The paper reports a measured comparison in Table 1.",
            },
        ],
    }


def synthetic_vision_result() -> Dict[str, Any]:
    return {
        "agent": "vision",
        "enabled": True,
        "level": "medium",
        "results": [
            {
                "asset_id": "figure_001",
                "asset_type": "figure",
                "page_number": 6,
                "section": "Method",
                "observation": "Two stacked blocks are shown with connecting arrows.",
                "interpretation": (
                    "This appears to illustrate an encoder-decoder design."
                ),
                "uncertainties": ["Small labels limit certainty about the wiring."],
            }
        ],
    }


CHUNK_ONE_REF = {
    "source_type": "text_chunk",
    "source_id": "chunk_001",
    "pages": [2],
    "verified": True,
}
CHUNK_TWO_REF = {
    "source_type": "text_chunk",
    "source_id": "chunk_002",
    "pages": [8, 9],
    "verified": True,
}
FIGURE_REF = {
    "source_type": "visual_asset",
    "source_id": "figure_001",
    "pages": [6],
    "verified": True,
}


def _finding(statement: str, refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "statement": statement,
        "evidence_status": "supported",
        "evidence_refs": refs,
        "confidence": 0.5,
        "reasoning": "Taken from the supplied artifacts.",
        "caveats": [],
    }


def _dimension(name: str, summary: str, statement: str) -> Dict[str, Any]:
    refs = [CHUNK_TWO_REF] if name == "results" else [CHUNK_ONE_REF]
    return {
        "dimension": name,
        "summary": summary,
        "evidence_status": "supported",
        "findings": [_finding(statement, refs)],
        "evidence_refs": refs,
        "confidence": 0.5,
        "uncertainties": [],
    }


def synthetic_analysis_result() -> Dict[str, Any]:
    """A Phase 5 result whose findings all trace to the synthetic sources above."""
    limitations_author = {
        "statement": (
            "The authors state that the evaluation covers a limited set of tasks."
        ),
        "evidence_status": "supported",
        "evidence_refs": [CHUNK_ONE_REF],
        "confidence": None,
        "reasoning": "Recorded by the analysis stage.",
        "caveats": [],
    }
    limitations_analyst = {
        "statement": (
            "The analysis indicates that training details are not established from "
            "the available evidence."
        ),
        "evidence_status": "unclear",
        "evidence_refs": [CHUNK_ONE_REF],
        "confidence": None,
        "reasoning": "No training detail was supplied.",
        "caveats": [],
    }
    gap = {
        "gap": "Per-run variance is not present in the supplied evidence.",
        "category": "not_reported_in_paper",
        "impact": "Result stability cannot be assessed.",
        "affected_dimensions": ["results"],
    }

    return {
        "agent": "analysis",
        "status": "completed",
        "enabled": True,
        "level": "advanced",
        "model": "mock-analysis-advanced",
        "provider": "mock-analysis",
        "routing": {},
        "evidence_inputs": {},
        "analysis": {
            "research_problem": _dimension(
                "research_problem",
                "The paper addresses sequence transduction without recurrence.",
                "The authors state that recurrence is a bottleneck for sequence "
                "transduction.",
            ),
            "contribution": _dimension(
                "contribution",
                "The paper presents an encoder-decoder attention architecture as its "
                "primary contribution.",
                "The supplied evidence supports the architecture description but not "
                "a novelty claim.",
            ),
            "methodology": _dimension(
                "methodology",
                "The approach is described as an encoder-decoder with attention.",
                "The architecture is described in the retrieved text and illustrated "
                "by a figure.",
            ),
            "data": _dimension(
                "data",
                "Two translation tasks are named in the supplied analysis.",
                "Dataset splits are not specified in the available evidence.",
            ),
            "baselines": _dimension(
                "baselines",
                "Comparison baselines are named in the analysis.",
                "Baseline selection fairness could not be assessed from the evidence.",
            ),
            "metrics": _dimension(
                "metrics",
                "A translation metric is named in the analysis.",
                "The metric definition is not described in the supplied evidence.",
            ),
            "results": _dimension(
                "results",
                "The paper reports a measured comparison on a benchmark task.",
                "The analysis records a reported improvement whose exact value is "
                "not established from the available evidence.",
            ),
            "reproducibility": _dimension(
                "reproducibility",
                "Architecture detail is described; training details are not.",
                "Hyperparameters are not reported in the supplied analysis.",
            ),
            "internal_consistency": _dimension(
                "internal_consistency",
                "No clear contradiction between text and figures was established.",
                "The analysis found no established inconsistency in the supplied "
                "evidence.",
            ),
            "claim_evidence_matrix": {
                "items": [
                    {
                        "claim": (
                            "The paper reports a measured comparison against "
                            "baselines."
                        ),
                        "claim_source": "paper",
                        "status": "partially_supported",
                        "supporting_evidence": [CHUNK_TWO_REF],
                        "contradicting_evidence": [],
                        "assessment": (
                            "The comparison is described, but not every value is "
                            "available."
                        ),
                        "missing_evidence": [
                            "Machine-readable values for every reported cell"
                        ],
                        "confidence": 0.4,
                    }
                ],
                "notes": "One claim assessed against the supplied evidence.",
            },
            "strengths": [
                {
                    "statement": (
                        "The architecture is described and illustrated by a confirmed "
                        "figure."
                    ),
                    "dimension": "methodology",
                    "evidence_status": "supported",
                    "evidence_refs": [FIGURE_REF],
                    "rationale": "The document profile confirms the figure exists.",
                    "confidence": 0.5,
                }
            ],
            "weaknesses": [
                {
                    "statement": (
                        "Exact numeric values are not established from the available "
                        "evidence."
                    ),
                    "dimension": "results",
                    "evidence_status": "insufficient_evidence",
                    "evidence_refs": [CHUNK_TWO_REF],
                    "rationale": "No machine-readable values were supplied.",
                    "confidence": None,
                }
            ],
            "limitations": {
                "author_stated": [limitations_author],
                "analyst_identified": [limitations_analyst],
            },
            "open_questions": [
                {
                    "question": "How sensitive is the method to hyperparameters?",
                    "why_it_matters": "Sensitivity bounds how general the result is.",
                    "related_dimension": "results",
                    "evidence_refs": [CHUNK_TWO_REF],
                }
            ],
            "evidence_gaps": [gap],
            "overall_assessment": {
                "summary": (
                    "The evidence supports the described architecture but leaves the "
                    "reported magnitude unestablished."
                ),
                "evidence_status": "partially_supported",
                "confidence": None,
                "key_basis": [CHUNK_ONE_REF],
                "uncertainties": ["Exact values are unavailable."],
            },
        },
        "evidence_gaps": [dict(gap, source="deterministic_context_check")],
        "unverified_evidence_refs": 0,
        "elapsed_time_ms": 12.0,
        "message": "analysis completed",
        "failures": [],
    }


def primary_ref(context: SynthesisContext) -> Dict[str, str]:
    """A reference object for a source that really exists in this context."""
    preferred = ("rag", "vision", "document", "analysis")
    for source_type in preferred:
        for entry in context.source_index.values():
            if entry["source_type"] == source_type:
                return {
                    "source_type": entry["source_type"],
                    "source_id": entry["source_id"],
                }
    return {}


SECTION_PROSE: Dict[str, str] = {
    "executive_summary": (
        "The paper studies sequence transduction and presents an encoder-decoder "
        "architecture with attention as its stated contribution. The available "
        "evidence supports the architecture description; the exact reported "
        "magnitude is not established from the available evidence."
    ),
    "research_problem": (
        "The authors state that recurrence is a bottleneck for sequence "
        "transduction. The analysis indicates that the motivation is the cost of "
        "sequential computation."
    ),
    "contribution": (
        "The paper presents an encoder-decoder attention architecture as its "
        "primary contribution. The supplied evidence partially supports that "
        "description; the analysis indicates that a novelty claim is not "
        "established from the available evidence."
    ),
    "methodology": (
        "The analysis indicates that the approach is an encoder-decoder with "
        "attention, described in the retrieved text and illustrated by a confirmed "
        "figure."
    ),
    "data_and_experimental_design": (
        "The analysis names two translation tasks. Dataset splits could not be "
        "established from the available evidence."
    ),
    "baselines_and_metrics": (
        "The analysis names comparison baselines and a translation metric; the "
        "metric definition could not be established from the available evidence."
    ),
    "results_and_evidence": (
        "The paper reports a measured comparison on a benchmark task. The analysis "
        "indicates that the exact value is not established from the available "
        "evidence."
    ),
    "figures_and_tables": (
        "Figure 1 illustrates the described encoder-decoder design. The visual "
        "observation notes that small labels limit certainty about the wiring."
    ),
    "claim_evidence_assessment": (
        "One reported claim is assessed against the supplied evidence and is only "
        "partially supported."
    ),
    "strengths": (
        "The architecture is described in text and illustrated by a confirmed "
        "figure."
    ),
    "author_stated_limitations": (
        "The authors state that the evaluation covers a limited set of tasks."
    ),
    "analyst_identified_limitations": (
        "The analysis indicates that training details are not established from the "
        "available evidence."
    ),
    "reproducibility": (
        "The analysis describes the architecture but reports that hyperparameters "
        "are not established from the available evidence."
    ),
    "internal_consistency": (
        "The supplied evidence suggests no established discrepancy between the "
        "text and the confirmed figure; the comparison could not be checked "
        "directly."
    ),
    "evidence_gaps": (
        "Per-run variance is not present in the supplied evidence, and the exact "
        "reported value could not be established from the available evidence."
    ),
    "open_questions": (
        "The analysis raises the question of how sensitive the method is to "
        "hyperparameters."
    ),
    "overall_assessment": (
        "The evidence supports the described architecture while the reported "
        "magnitude and the training details remain unestablished. The principal "
        "concern is that the exact values behind the reported comparison are not "
        "available in the supplied evidence."
    ),
}


def build_compliant_response(context: SynthesisContext) -> Dict[str, Any]:
    """
    Build a contract-compliant model response for the given context.

    Every section that the context declares supported is emitted with real ids, so
    a response built here must pass validation; tests that need a specific defect
    mutate a copy of this response.
    """
    ref = primary_ref(context)
    sections = []
    for section_id in list(ALWAYS_REQUIRED_SECTIONS) + [
        item
        for item in context.supported_sections.section_ids
        if item not in ALWAYS_REQUIRED_SECTIONS
    ]:
        sections.append(
            {
                "section_id": section_id,
                "title": SECTION_TITLES.get(section_id, section_id),
                "content": SECTION_PROSE.get(section_id, f"Synthesis of {section_id}."),
                "evidence_refs": [ref] if ref else [],
            }
        )

    limitations = context.analysis
    return {
        "sections": sections,
        "claim_evidence_summary": [
            {
                "claim": claim.claim,
                "status": claim.status,
                "assessment": claim.assessment or "Assessed against the supplied evidence.",
                "evidence_refs": [ref] if ref else [],
                "uncertainty_notes": claim.missing_evidence[:1],
            }
            for claim in limitations.claims
        ],
        "strengths": [
            {
                "statement": statement,
                "dimension": "methodology",
                "evidence_refs": [ref] if ref else [],
                "rationale": "Recorded by the analysis stage.",
            }
            for statement in limitations.strengths
        ],
        "limitations": {
            "author_stated": [
                {"statement": statement, "evidence_refs": [ref] if ref else []}
                for statement in limitations.author_stated_limitations
            ],
            "analyst_identified": [
                {"statement": statement, "evidence_refs": [ref] if ref else []}
                for statement in limitations.analyst_identified_limitations
            ],
        },
        "evidence_gaps": [
            {"gap": gap.gap, "category": gap.category, "impact": gap.impact}
            for gap in context.deterministic_gaps + context.analysis.evidence_gaps
        ],
        "open_questions": [
            {
                "question": question
                if question.rstrip().endswith("?")
                else f"{question.rstrip('.')} - is this answerable?",
                "why_it_matters": "It bounds how far the reading generalizes.",
                "evidence_refs": [ref] if ref else [],
            }
            for question in limitations.open_questions
        ],
        "overall_assessment": {
            "content": SECTION_PROSE["overall_assessment"],
            "evidence_refs": [ref] if ref else [],
            "uncertainties": ["Exact reported values are unavailable."],
        },
    }


def compliant_response_json(context: SynthesisContext) -> str:
    """The compliant response serialized as the model's JSON text."""
    return json.dumps(build_compliant_response(context))


def routing_state(**overrides: Any) -> Dict[str, Any]:
    """A routing state built through the frozen JEV contract layer."""
    from backend.jev_router.routing_state import create_routing_state

    payload: Dict[str, Any] = {
        "analysis_level": "advanced",
        "rag_enabled": True,
        "rag_level": "medium",
        "vision_enabled": True,
        "vision_level": "medium",
        "confidence": {"analysis": 0.93, "rag": 0.71, "vision": 0.68},
    }
    payload.update(overrides)
    return create_routing_state(**payload)