"""
Standalone Critique Engine demo / smoke runner (Phase 6).

Instructions:
1. Loads the synthesis credentials and model identifiers from environment / .env
   (CRITIQUE_API_KEY or the provider-specific key; CRITIQUE_MODEL_* per JEV
   analysis level, or a single CRITIQUE_MODEL).
2. If no credential or no model is configured for the routed level, prints a
   clear message and exits gracefully (code 0). No network call is attempted.
3. Otherwise:
    - runs the frozen Document Pre-Analyzer on "Attention Is All You Need" to
      obtain the real document profile
    - builds a routing state through the frozen JEV contract layer (the Critique
      Engine adds no routing decision of its own)
    - constructs representative RAG / Vision / Analysis artifacts *derived from
      that profile* (this demo does not rerun retrieval or visual analysis; the
      real agents produce those in the pipeline)
    - runs the Critique Engine and prints the selected model, the sections, the
      executive summary, strengths, both limitation categories, the claim-evidence
      assessment, the evidence gaps, the open questions, the overall assessment
      and the provenance summary
    - NEVER logs, prints, or exposes any API key.

Note: this script is standalone and is NOT executed automatically by pytest.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Extracted text can contain Unicode that a legacy Windows console cannot encode;
# keep each stream's encoding but never raise on print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        pass

from dotenv import load_dotenv

load_dotenv()

from backend.critique_engine.config import (
    CritiqueConfig,
    CritiqueConfigurationError,
    normalize_level,
    resolve_model_for_level,
)
from backend.critique_engine.engine import CritiqueEngine
from backend.critique_engine.llm_client import build_critique_client, resolve_api_key
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.routing_state import create_routing_state

BENCHMARK_PDF = os.path.join(
    "uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

QUERY = "How is the reported result measured and on which data?"


def _safe(text: object) -> str:
    """Return text printable on the active console encoding, never raising."""
    value = "" if text is None else str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except (LookupError, UnicodeError):  # pragma: no cover - platform dependent
        return value.encode("ascii", errors="replace").decode("ascii")


def _confirmed_items(block: object) -> list:
    """Confirmed entries of a Document Profile inventory block."""
    if not isinstance(block, dict):
        return []
    items = block.get("items") or []
    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("status", "confirmed")) == "confirmed"
    ]


def build_representative_rag_result(profile: dict) -> dict:
    """
    Build a representative RAG result from the frozen profile.

    This does not run retrieval: it stands in for what the frozen RAG Agent
    produces, so the demo can exercise the synthesis stage in isolation. Every id
    here (``chunk_001``...) is real provenance the Critique Engine can validate.
    """
    sections = [
        item for item in (profile.get("sections") or []) if isinstance(item, dict)
    ]
    results = []
    for index, section in enumerate(sections[:6], 1):
        page = section.get("page") or 1
        results.append(
            {
                "chunk_id": f"chunk_{index:03d}",
                "score": round(0.9 - index * 0.05, 2),
                "page_start": page,
                "page_end": page,
                "section": section.get("title") or None,
                "text": (
                    f"Representative excerpt for the '{section.get('title')}' part "
                    "of the paper, taken from the frozen document structure."
                ),
                "word_count": 24,
            }
        )
    return {
        "agent": "rag",
        "enabled": True,
        "level": "medium",
        "query": QUERY,
        "chunks_indexed": len(results),
        "top_k": len(results),
        "results": results,
        "message": "representative RAG artifact constructed from the frozen profile",
    }


def build_representative_vision_result(profile: dict) -> dict:
    """Build a representative Vision result from the confirmed inventory."""
    results = []
    figures = _confirmed_items(profile.get("figures"))[:4]
    tables = _confirmed_items(profile.get("tables"))[:4]
    for kind, items in (("figure", figures), ("table", tables)):
        for index, item in enumerate(items, 1):
            results.append(
                {
                    "asset_id": str(item.get("id") or f"{kind}_{index:03d}"),
                    "asset_type": kind,
                    "page_number": item.get("page") or 1,
                    "section": None,
                    "caption": item.get("caption"),
                    "observation": (
                        "A representative visual observation was recorded for this "
                        "asset by the vision stage."
                    ),
                    "interpretation": (
                        "It appears to illustrate the element the surrounding text "
                        "describes."
                    ),
                    "uncertainties": [
                        "This stand-in observation is representative, not an actual "
                        "multimodal reading."
                    ],
                }
            )
    return {
        "agent": "vision",
        "enabled": True,
        "level": "medium",
        "assets_found": len(results),
        "assets_analyzed": len(results),
        "results": results,
        "message": "representative vision artifact constructed from the frozen profile",
    }


def _finding(statement: str, refs: list) -> dict:
    return {
        "statement": statement,
        "evidence_status": "supported" if refs else "unclear",
        "evidence_refs": refs,
        "confidence": None,
        "reasoning": "Taken from the representative upstream artifacts.",
        "caveats": [],
    }


def _dimension(name: str, statement: str, refs: list) -> dict:
    return {
        "dimension": name,
        "summary": statement,
        "evidence_status": "supported" if refs else "unclear",
        "findings": [_finding(statement, refs)],
        "evidence_refs": refs,
        "confidence": None,
        "uncertainties": [],
    }


def build_representative_analysis_result(
    profile: dict, rag_result: dict, vision_result: dict
) -> dict:
    """
    Build a representative Phase 5 result from the artifacts above.

    In the real pipeline the frozen Analysis Agent produces this; the demo only
    needs a structured analysis input that references the same upstream ids.
    """
    chunk_refs = [
        {
            "source_type": "text_chunk",
            "source_id": item["chunk_id"],
            "pages": [item["page_start"]],
            "verified": True,
        }
        for item in rag_result["results"][:3]
    ]
    visual_refs = [
        {
            "source_type": "visual_asset",
            "source_id": item["asset_id"],
            "pages": [item["page_number"]],
            "verified": True,
        }
        for item in vision_result["results"][:3]
    ]

    def pick(position: int) -> list:
        if not chunk_refs:
            return []
        return [chunk_refs[position % len(chunk_refs)]]

    dimensions = {
        "research_problem": _dimension(
            "research_problem",
            "The paper addresses sequential transduction without recurrence.",
            pick(0),
        ),
        "contribution": _dimension(
            "contribution",
            "The paper presents a Transformer encoder-decoder as its contribution.",
            pick(0),
        ),
        "methodology": _dimension(
            "methodology",
            "The approach is described as an attention-based encoder-decoder.",
            pick(1),
        ),
        "data": _dimension(
            "data",
            "Translation tasks are named; splits are not established from the "
            "available evidence.",
            pick(0),
        ),
        "baselines": _dimension(
            "baselines",
            "Comparison baselines are named in the available evidence.",
            pick(2),
        ),
        "metrics": _dimension(
            "metrics",
            "An evaluation metric is named in the available evidence.",
            pick(0),
        ),
        "results": _dimension(
            "results",
            "The paper reports a measured comparison whose exact values are not "
            "established from the available evidence.",
            pick(2),
        ),
        "reproducibility": _dimension(
            "reproducibility",
            "Hyperparameters are not established from the available evidence.",
            pick(0),
        ),
        "internal_consistency": _dimension(
            "internal_consistency",
            "No established discrepancy was found between the text and the recorded "
            "visual evidence.",
            visual_refs[:1],
        ),
    }

    return {
        "agent": "analysis",
        "status": "completed",
        "enabled": True,
        "level": "advanced",
        "model": "representative-analysis",
        "provider": "representative",
        "analysis": {
            **dimensions,
            "claim_evidence_matrix": {
                "items": [
                    {
                        "claim": (
                            "The paper reports a measured comparison against "
                            "baselines."
                        ),
                        "claim_source": "paper",
                        "status": "partially_supported",
                        "supporting_evidence": pick(0),
                        "contradicting_evidence": [],
                        "assessment": (
                            "The comparison is described, but the exact values are "
                            "not established."
                        ),
                        "missing_evidence": ["Machine-readable values for every cell"],
                        "confidence": None,
                    }
                ],
                "notes": "One claim assessed.",
            },
            "strengths": [
                {
                    "statement": (
                        "The architecture is described in text and illustrated by a "
                        "confirmed figure."
                    ),
                    "dimension": "methodology",
                    "evidence_status": "supported",
                    "evidence_refs": visual_refs[:1],
                    "rationale": "The document profile confirms the figure exists.",
                    "confidence": None,
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
                    "evidence_refs": pick(0),
                    "rationale": "No machine-readable values were supplied.",
                    "confidence": None,
                }
            ],
            "limitations": {
                "author_stated": [
                    {
                        "statement": (
                            "The authors state that the evaluation covers a limited "
                            "set of settings."
                        ),
                        "evidence_status": "supported",
                        "evidence_refs": pick(0),
                        "confidence": None,
                        "reasoning": "Cited from the retrieved text.",
                        "caveats": [],
                    }
                ],
                "analyst_identified": [
                    {
                        "statement": (
                            "Training details are not established from the available "
                            "evidence."
                        ),
                        "evidence_status": "unclear",
                        "evidence_refs": pick(0),
                        "confidence": None,
                        "reasoning": "No training detail was available.",
                        "caveats": [],
                    }
                ],
            },
            "open_questions": [
                {
                    "question": "How sensitive is the method to hyperparameters?",
                    "why_it_matters": "Sensitivity bounds how general the result is.",
                    "related_dimension": "results",
                    "evidence_refs": pick(0),
                }
            ],
            "evidence_gaps": [
                {
                    "gap": (
                        "Per-run variance is not present in the available evidence."
                    ),
                    "category": "not_reported_in_paper",
                    "impact": "Result stability cannot be assessed.",
                    "affected_dimensions": ["results"],
                }
            ],
            "overall_assessment": {
                "summary": (
                    "The evidence supports the described architecture while the "
                    "reported magnitude stays unestablished."
                ),
                "evidence_status": "partially_supported",
                "confidence": None,
                "key_basis": pick(0),
                "uncertainties": ["Exact values are unavailable."],
            },
        },
        "evidence_gaps": [
            {
                "gap": "Per-run variance is not present in the available evidence.",
                "category": "not_reported_in_paper",
                "impact": "Result stability cannot be assessed.",
                "affected_dimensions": ["results"],
                "source": "analysis",
            }
        ],
        "unverified_evidence_refs": 0,
        "message": "representative analysis artifact constructed for the demo",
        "failures": [],
    }


def _print_report(result: dict) -> None:
    print("\n--- Final critique ---")
    print(
        f"status={result['status']} engine={result['engine']} version={result['version']} "
        f"model={result.get('model')} level={result.get('level')} "
        f"provider={result.get('provider')}"
    )
    print(f"sections generated: {len(result['sections'])}")
    for record in result["sections"]:
        print(f"  - {record['section_id']}: {record['title']}")

    def section(section_id: str) -> str:
        for record in result["sections"]:
            if record["section_id"] == section_id:
                return record["content"]
        return "(section omitted: not supported by the available evidence)"

    print("\n--- Executive summary ---")
    print(_safe(section("executive_summary")))

    print("\n--- Major strengths ---")
    if result["strengths"]:
        for item in result["strengths"]:
            print(f"  - {_safe(item['statement'])}")
    else:
        print("  (none recorded by the analysis stage)")

    print("\n--- Limitations ---")
    print("  Author-stated:")
    for item in result["limitations"]["author_stated"] or []:
        print(f"    - {_safe(item['statement'])}")
    print("  Analyst-identified:")
    for item in result["limitations"]["analyst_identified"] or []:
        print(f"    - {_safe(item['statement'])}")

    print("\n--- Claim-evidence assessment ---")
    for item in result["claim_evidence_summary"] or []:
        print(f"  [{item['status']}] {_safe(item['claim'])}")
        if item.get("assessment"):
            print(f"      {_safe(item['assessment'])}")

    print("\n--- Evidence gaps ---")
    for gap in result["evidence_gaps"] or []:
        print(f"  ({gap['category']}) {_safe(gap['gap'])}")

    print("\n--- Open research questions ---")
    for question in result["open_questions"] or []:
        print(f"  - {_safe(question['question'])}")

    print("\n--- Overall research assessment ---")
    print(_safe(result["overall_assessment"]["content"]))

    print("\n--- Provenance ---")
    provenance = result["provenance"]
    for key in sorted(provenance):
        print(f"  {key}: {provenance[key]}")
    print(f"  unverified_evidence_refs: {result['unverified_evidence_refs']}")
    if result.get("warnings"):
        print("  warnings:")
        for warning in result["warnings"]:
            print(f"    - {_safe(warning)}")


def _print_failure(result: dict) -> None:
    print("\n--- The critique could not be produced ---")
    print(f"status={result['status']}")
    print(f"message: {_safe(result.get('message'))}")
    for failure in result.get("failures") or []:
        print(f"stage={failure['stage']} error: {_safe(failure['error'])}")
    for warning in result.get("warnings") or []:
        print(f"warning: {_safe(warning)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Critique Engine demo.")
    parser.add_argument("--pdf", default=BENCHMARK_PDF, help="Path to the benchmark PDF.")
    parser.add_argument(
        "--analysis-level",
        default="advanced",
        help="JEV analysis level whose critique model is selected.",
    )
    parser.add_argument(
        "--no-rag", action="store_true", help="Simulate a disabled RAG stage."
    )
    parser.add_argument(
        "--no-vision", action="store_true", help="Simulate a disabled Vision stage."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Synthesis model identifier to use for this run (overrides the "
            "CRITIQUE_MODEL_* / CRITIQUE_MODEL configuration)."
        ),
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.pdf):
        print(f"Benchmark PDF not found at '{args.pdf}'. Nothing to do.")
        return 0

    # 1. The real frozen Document Profile.
    print("Running the frozen Document Pre-Analyzer ...")
    profile = analyze_document(args.pdf)
    print(
        f"pages={profile['page_count']} words={profile['word_count']} "
        f"sections={len(profile.get('sections') or [])}"
    )

    # 2. A routing state from the frozen JEV contract layer. The Critique Engine
    #    adds no routing decision of its own; it reuses this analysis level.
    state = create_routing_state(
        analysis_level=args.analysis_level,
        rag_enabled=not args.no_rag,
        rag_level="medium" if not args.no_rag else None,
        vision_enabled=not args.no_vision,
        vision_level="medium" if not args.no_vision else None,
        confidence={"analysis": 0.93, "rag": 0.71, "vision": 0.68},
    )
    level = normalize_level(state["analysis"]["level"])

    # 3. Representative upstream artifacts derived from that profile.
    rag_result = build_representative_rag_result(profile)
    vision_result = build_representative_vision_result(profile)
    analysis_result = build_representative_analysis_result(
        profile, rag_result, vision_result
    )

    # 4. Configuration gate: exit gracefully, without printing any credential.
    config = CritiqueConfig(single_model=args.model) if args.model else CritiqueConfig()
    print(f"\nJEV analysis level: {level}")
    try:
        model = resolve_model_for_level(
            level,
            level_models=config.level_models,
            allow_fallback=config.allow_level_fallback,
            single_model=config.single_model,
        )
    except CritiqueConfigurationError as exc:
        print(f"No critique synthesis model is configured: {exc}")
        print(
            "Set CRITIQUE_MODEL_ADVANCED (or the level you selected) / CRITIQUE_MODEL "
            "in .env and try again."
        )
        return 0
    print(f"Selected synthesis model: {model}")

    if not resolve_api_key(config.provider):
        print(
            f"No credential found for critique provider '{config.provider}'.\n"
            "Set CRITIQUE_API_KEY (or the provider-specific key such as "
            "OPENAI_API_KEY) in .env and try again.\n"
            "The demo exits gracefully; no API key was printed."
        )
        return 0

    # 5. The real synthesis call.
    client = build_critique_client(config)
    engine = CritiqueEngine(critique_client=client, config=config)
    result = engine.generate_critique(
        document_profile=profile,
        routing_state=state,
        rag_result=rag_result if not args.no_rag else None,
        vision_result=vision_result if not args.no_vision else None,
        analysis_result=analysis_result,
    )

    if result["status"] == "completed":
        _print_report(result)
        print(f"\nelapsed_time_ms={result.get('elapsed_time_ms')}")
        return 0

    _print_failure(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())