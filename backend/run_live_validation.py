"""
One controlled real end-to-end validation run (Phase 7).

Runs the complete pipeline once against the benchmark paper with the production
clients built from the environment configuration (real JEV, real embeddings, real
multimodal vision model, real analysis model, real synthesis model) and writes a
secret-free validation artifact.

Usage:
    python -m backend.run_live_validation
    python -m backend.run_live_validation --pdf <path> --output artifacts/live_e2e_validation.json

Safety:
* Credentials are never printed; the credential report only says PRESENT/ABSENT.
* Provider errors are reported by type and sanitized message only.
* Nothing is fabricated: a failed stage is recorded as a failure and the run
  continues only where it legitimately can.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        pass

from dotenv import load_dotenv

load_dotenv()

from backend.pipeline import DEFAULT_QUERY, run_pipeline

BENCHMARK_PDF = os.path.join(
    "uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf"
)
DEFAULT_ARTIFACT = os.path.join("artifacts", "live_e2e_validation.json")

SECRET_VARS = {
    "TYPESAFE_API_KEY": "jev_router",
    "OPENAI_API_KEY": "rag_embeddings",
    "VISION_API_KEY": "vision",
    "ANALYSIS_API_KEY": "analysis",
    "CRITIQUE_API_KEY": "critique",
}


def credential_report() -> Dict[str, str]:
    """Report only whether each credential is PRESENT or ABSENT (never a value)."""
    return {
        stage: ("PRESENT" if (os.getenv(name) or "").strip() else "ABSENT")
        for name, stage in SECRET_VARS.items()
    }


def model_report() -> Dict[str, Any]:
    """Report the configured provider/model identifiers for each component."""
    from backend.analysis_agent.config import ANALYSIS_LEVEL_MODELS, DEFAULT_ANALYSIS_PROVIDER
    from backend.critique_engine.config import CRITIQUE_LEVEL_MODELS, DEFAULT_CRITIQUE_PROVIDER
    from backend.jev_router.config import DEFAULT_JEV_MODEL
    from backend.rag_agent.config import DEFAULT_EMBEDDING_MODEL
    from backend.vision_agent.config import DEFAULT_VISION_PROVIDER, VISION_LEVEL_MODELS

    return {
        "jev": {"model": DEFAULT_JEV_MODEL},
        "rag": {"embedding_model": DEFAULT_EMBEDDING_MODEL},
        "vision": {
            "provider": DEFAULT_VISION_PROVIDER,
            "level_models": dict(VISION_LEVEL_MODELS),
        },
        "analysis": {
            "provider": DEFAULT_ANALYSIS_PROVIDER,
            "level_models": dict(ANALYSIS_LEVEL_MODELS),
        },
        "critique": {
            "provider": DEFAULT_CRITIQUE_PROVIDER,
            "level_models": dict(CRITIQUE_LEVEL_MODELS),
        },
    }


def summarize(stages: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, secret-free summary of the run for the artifact and the report.

    Only counts, identifiers and statuses are extracted; no prompt text, no page
    body content and no credential ever reaches the artifact.
    """
    profile = (stages.get("document_pre_analyzer") or {}).get("result") or {}
    routing = (stages.get("jev_router") or {}).get("result") or {}
    rag = (stages.get("rag") or {}).get("result") or {}
    vision = (stages.get("vision") or {}).get("result") or {}
    analysis = (stages.get("analysis") or {}).get("result") or {}
    critique = (stages.get("critique") or {}).get("result") or {}

    analysis_payload = analysis.get("analysis") or {}
    matrix_items = (analysis_payload.get("claim_evidence_matrix") or {}).get(
        "items"
    ) or []
    findings = 0
    for value in analysis_payload.values():
        if isinstance(value, dict) and isinstance(value.get("findings"), list):
            findings += len(value["findings"])

    def count_refs(node: Any) -> int:
        if isinstance(node, dict):
            if isinstance(node.get("evidence_refs"), list):
                return len(node["evidence_refs"])
            return sum(count_refs(item) for item in node.values())
        if isinstance(node, list):
            return sum(count_refs(item) for item in node)
        return 0

    figures = [
        item
        for item in (vision.get("results") or [])
        if str(item.get("asset_type")) == "figure"
    ]
    tables = [
        item
        for item in (vision.get("results") or [])
        if str(item.get("asset_type")) == "table"
    ]

    return {
        "document": {
            "filename": profile.get("filename"),
            "title": (profile.get("metadata") or {}).get("title"),
            "page_count": profile.get("page_count"),
            "word_count": profile.get("word_count"),
            "sections": len(profile.get("sections") or []),
            "figures_confirmed": (profile.get("figures") or {}).get("count"),
            "tables_confirmed": (profile.get("tables") or {}).get("count"),
            "equations_confirmed": (profile.get("equations") or {}).get("count"),
            "references": (profile.get("references") or {}).get("count"),
            "document_nature": (profile.get("content_characteristics") or {}).get(
                "document_nature"
            ),
        },
        "routing": {
            "router_version": routing.get("router_version"),
            "analysis": routing.get("analysis"),
            "rag": routing.get("rag"),
            "vision": routing.get("vision"),
            "confidence": routing.get("confidence"),
        },
        "rag": {
            "enabled": rag.get("enabled"),
            "level": rag.get("level"),
            "embedding_model": rag.get("embedding_model"),
            "chunks_indexed": rag.get("chunks_indexed"),
            "top_k": rag.get("top_k"),
            "retrieved_chunks": len(rag.get("results") or []),
        },
        "vision": {
            "enabled": vision.get("enabled"),
            "level": vision.get("level"),
            "model": vision.get("model"),
            "provider": vision.get("provider"),
            "assets_found": vision.get("assets_found"),
            "assets_analyzed": vision.get("assets_analyzed"),
            "assets_failed": vision.get("assets_failed"),
            "figures_analyzed": len(figures),
            "tables_analyzed": len(tables),
            "extraction_failures": len(vision.get("extraction_failures") or []),
        },
        "analysis": {
            "status": analysis.get("status"),
            "level": analysis.get("level"),
            "model": analysis.get("model"),
            "provider": analysis.get("provider"),
            "findings": findings,
            "evidence_refs": count_refs(analysis_payload),
            "claim_evidence_records": len(matrix_items),
            "strengths": len(analysis_payload.get("strengths") or []),
            "weaknesses": len(analysis_payload.get("weaknesses") or []),
            "author_stated_limitations": len(
                (analysis_payload.get("limitations") or {}).get("author_stated") or []
            ),
            "analyst_identified_limitations": len(
                (analysis_payload.get("limitations") or {}).get("analyst_identified")
                or []
            ),
            "open_questions": len(analysis_payload.get("open_questions") or []),
            "evidence_gaps": len(analysis.get("evidence_gaps") or []),
            "unverified_evidence_refs": analysis.get("unverified_evidence_refs"),
        },
        "critique": {
            "status": critique.get("status"),
            "engine": critique.get("engine"),
            "version": critique.get("version"),
            "level": critique.get("level"),
            "model": critique.get("model"),
            "provider": critique.get("provider"),
            "sections_generated": len(critique.get("sections") or []),
            "claim_evidence_summary": len(critique.get("claim_evidence_summary") or []),
            "strengths": len(critique.get("strengths") or []),
            "author_stated_limitations": len(
                (critique.get("limitations") or {}).get("author_stated") or []
            ),
            "analyst_identified_limitations": len(
                (critique.get("limitations") or {}).get("analyst_identified") or []
            ),
            "evidence_gaps": len(critique.get("evidence_gaps") or []),
            "open_questions": len(critique.get("open_questions") or []),
            "unverified_evidence_refs": critique.get("unverified_evidence_refs"),
            "provenance": critique.get("provenance"),
            "warnings": len(critique.get("warnings") or []),
            "failures": critique.get("failures") or [],
        },
    }


def _sanitize_text(value: object, limit: int = 400) -> str:
    """Truncate and strip control characters from text stored in the artifact."""
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text[:limit]


STAGE_NAMES = (
    "document_pre_analyzer",
    "jev_router",
    "rag",
    "vision",
    "analysis",
    "critique",
)


def build_artifact(pdf_path: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the validation artifact.

    The artifact holds stage statuses, a summary, the routing state, the analysis
    and critique results and timings. It deliberately excludes raw prompts, page
    text and every credential.
    """
    stages = outcome.get("stages") or {}
    return {
        "artifact": "live_e2e_validation",
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
        },
        "credentials": credential_report(),
        "models": model_report(),
        "overall_status": outcome.get("status"),
        "failed_stage": outcome.get("failed_stage"),
        "critique_status": outcome.get("critique_status"),
        "elapsed_ms": outcome.get("elapsed_ms"),
        "timings": {
            name: (stages.get(name) or {}).get("elapsed_ms") for name in STAGE_NAMES
        },
        "stage_status": {
            name: (stages.get(name) or {}).get("status") for name in STAGE_NAMES
        },
        "stage_errors": {
            name: {
                "error_type": (stages.get(name) or {}).get("error_type"),
                "error": _sanitize_text((stages.get(name) or {}).get("error")),
            }
            for name in STAGE_NAMES
            if (stages.get(name) or {}).get("status") == "failed"
        },
        "agent_failures": {
            "vision_analysis_failures": (
                ((stages.get("vision") or {}).get("result") or {}).get("failures") or []
            ),
            "vision_extraction_failures": (
                ((stages.get("vision") or {}).get("result") or {}).get(
                    "extraction_failures"
                )
                or []
            ),
            "analysis_failures": (
                ((stages.get("analysis") or {}).get("result") or {}).get("failures")
                or []
            ),
        },
        "summary": summarize(stages),
        "routing_state": (stages.get("jev_router") or {}).get("result"),
        "analysis_result": (stages.get("analysis") or {}).get("result"),
        "critique_result": (stages.get("critique") or {}).get("result"),
    }


def _print_report(artifact: Dict[str, Any]) -> None:
    summary = artifact["summary"]
    document = summary["document"]
    routing = summary["routing"]
    rag = summary["rag"]
    vision = summary["vision"]
    analysis = summary["analysis"]
    critique = summary["critique"]

    print("\n----------------------------------------")
    print("LIVE END-TO-END VALIDATION")
    print("----------------------------------------")
    print(f"Overall: {artifact['overall_status']}")
    if artifact.get("failed_stage"):
        print(f"Failed stage: {artifact['failed_stage']}")

    print("\nDocument:", document["title"] or document["filename"])
    print(
        f"  pages={document['page_count']} words={document['word_count']} "
        f"figures={document['figures_confirmed']} "
        f"tables={document['tables_confirmed']} "
        f"equations={document['equations_confirmed']} "
        f"references={document['references']} nature={document['document_nature']}"
    )

    print(
        f"\nJEV: model={artifact['models']['jev']['model']} "
        f"status={artifact['stage_status']['jev_router']}"
    )
    print(f"  analysis={routing['analysis']}")
    print(f"  rag={routing['rag']}")
    print(f"  vision={routing['vision']}")
    print(f"  confidence={routing['confidence']}")

    print(
        f"\nRAG: status={artifact['stage_status']['rag']} "
        f"embedding={rag['embedding_model']}"
    )
    print(
        f"  chunks_indexed={rag['chunks_indexed']} "
        f"retrieved={rag['retrieved_chunks']} level={rag['level']}"
    )

    print(
        f"\nVision: status={artifact['stage_status']['vision']} "
        f"model={vision['model']}"
    )
    print(
        f"  figures={vision['figures_analyzed']} tables={vision['tables_analyzed']} "
        f"failed={vision['assets_failed']} "
        f"extraction_failures={vision['extraction_failures']}"
    )

    print(
        f"\nAnalysis: status={analysis['status']} model={analysis['model']} "
        f"level={analysis['level']}"
    )
    print(
        f"  findings={analysis['findings']} refs={analysis['evidence_refs']} "
        f"claim_records={analysis['claim_evidence_records']} "
        f"unverified_refs={analysis['unverified_evidence_refs']}"
    )

    print(f"\nCritique: status={critique['status']} model={critique['model']}")
    print(
        f"  sections={critique['sections_generated']} "
        f"claims={critique['claim_evidence_summary']} "
        f"author_limitations={critique['author_stated_limitations']} "
        f"analyst_limitations={critique['analyst_identified_limitations']} "
        f"gaps={critique['evidence_gaps']} "
        f"unverified_refs={critique['unverified_evidence_refs']}"
    )

    print("\nTimings (ms):")
    for stage, value in artifact["timings"].items():
        print(f"  {stage}: {value}")
    print(f"  total: {artifact['elapsed_ms']}")

    if artifact["stage_errors"]:
        print("\nStage errors (sanitized):")
        for stage, info in artifact["stage_errors"].items():
            print(f"  {stage}: {info['error_type']}: {info['error']}")
    print("----------------------------------------")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one live end-to-end validation.")
    parser.add_argument("--pdf", default=BENCHMARK_PDF)
    parser.add_argument("--output", default=DEFAULT_ARTIFACT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args(argv)

    if not os.path.exists(args.pdf):
        print(
            f"Benchmark PDF not found at '{args.pdf}'. Stopping: the live validation "
            "must run on the project's benchmark paper, not on a substitute."
        )
        return 1

    print("Credentials:", credential_report())
    print(f"Running the full pipeline on {args.pdf} ...")

    def progress(stage: str, status: str) -> None:
        print(f"  [{status}] {stage}", flush=True)

    started = time.time()
    outcome = run_pipeline(args.pdf, query=args.query, progress=progress)
    print(f"Pipeline finished in {round(time.time() - started, 2)}s")

    artifact = build_artifact(args.pdf, outcome)
    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, default=str)
    print(f"Validation artifact written to {args.output}")

    _print_report(artifact)
    return 0 if artifact["overall_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())