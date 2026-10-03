"""
Deterministic explanation generator for routing decisions.

Generates human-readable, deterministic justifications without asking
JEV to generate free-form prose.

The explanations are derived purely from:
1. Compact document routing state (JevInputState / build_jev_input output)
2. Validated routing decision (RoutingDecisionState)
"""

from typing import Any, Dict, List, Union

from backend.jev_router.routing_state import validate_routing_state
from backend.jev_router.schemas import JevInputState, RoutingDecisionState


def generate_routing_explanation(
    document_input: Union[Dict[str, Any], JevInputState],
    routing_decision: Union[Dict[str, Any], RoutingDecisionState],
) -> List[str]:
    """
    Produce deterministic explanations for each agent's routing decision.

    Derived purely from the input characteristics and routing choices.
    Never invents or fabricates LLM reasoning.
    """
    if isinstance(document_input, dict):
        doc = JevInputState.model_validate(document_input)
    elif isinstance(document_input, JevInputState):
        doc = document_input
    else:
        raise ValueError(f"Expected dict or JevInputState, got {type(document_input).__name__}")

    decision = validate_routing_state(routing_decision)

    explanations: List[str] = []

    # 1. Analysis agent explanation
    explanations.append(
        f"Analysis Agent is enabled at '{decision.analysis.level.value}' tier "
        f"to synthesize the primary research critique."
    )

    # 2. RAG agent explanation
    if decision.rag.enabled:
        rag_level_str = decision.rag.level.value if decision.rag.level else "unspecified"
        reason = "RAG is enabled for retrieval over the paper's textual content."
        if doc.document.page_count > 10 or doc.document.word_count > 4000:
            reason += f" Document is comprehensive ({doc.document.page_count} pages, {doc.document.word_count} words)."
        explanations.append(f"{reason} Assigned tier: '{rag_level_str}'.")
    else:
        explanations.append(
            "RAG is disabled as textual retrieval is not required for this document."
        )

    # 3. Vision agent explanation
    if decision.vision.enabled:
        vis_level_str = decision.vision.level.value if decision.vision.level else "unspecified"
        vis_reasons = []
        if doc.visual_content.figure_count > 0:
            vis_reasons.append(f"{doc.visual_content.figure_count} confirmed figure(s)")
        if doc.visual_content.table_count > 0:
            vis_reasons.append(f"{doc.visual_content.table_count} confirmed table(s)")

        if vis_reasons:
            joined = " and ".join(vis_reasons)
            explanations.append(
                f"Vision is enabled because the document contains {joined}. "
                f"Assigned tier: '{vis_level_str}'."
            )
        else:
            explanations.append(
                f"Vision is enabled for visual inspection of document elements. "
                f"Assigned tier: '{vis_level_str}'."
            )
    else:
        explanations.append(
            "Vision is disabled because the document contains no visual figures or tables requiring inspection."
        )

    return explanations
