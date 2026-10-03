"""
Thin end-to-end orchestration for the AI Research Paper Critic.

This module contains **no agent logic**. It only calls the frozen components in
the fixed order and passes their structured outputs to the next stage:

    PDF
      -> Document Pre-Analyzer (frozen)
      -> JEV Router (frozen, real routing decision)
      -> RAG Agent (frozen, governed by the JEV decision)
      -> Vision Agent (frozen, governed by the JEV decision)
      -> Analysis Agent (frozen, governed by the JEV decision)
      -> Critique Engine (frozen, reuses the analysis level)
      -> final structured research critique

Design rules
------------
* Every stage result is the agent's own JSON-serializable dict; nothing is
  re-derived, summarised away or fabricated here.
* The JEV routing decision is produced by the router and is never overridden,
  hard-coded or post-edited. If JEV disables RAG or Vision, those stages are not
  forced to run.
* A failing stage stops the dependent stages and is reported as a structured
  failure (stage name, sanitized error type/message). No stage is faked, and no
  model fallback is attempted.
* Client injection points exist for offline tests only. In production every
  client is built from the environment configuration.

The Streamlit frontend calls :func:`run_pipeline`; it never imports agents.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from backend.analysis_agent.agent import AnalysisAgent
from backend.critique_engine.config import CritiqueConfig
from backend.critique_engine.engine import CritiqueEngine
from backend.critique_engine.llm_client import resolve_api_key as critique_api_key
from backend.analysis_agent.llm_client import mask_secret, resolve_api_key
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.router import route_document
from backend.rag_agent.agent import RAGAgent
from backend.vision_agent.agent import VisionAgent

logger = logging.getLogger(__name__)

# The default retrieval question used by the project demos. It is a retrieval
# query only; it never influences routing or any verdict.
DEFAULT_QUERY = "How is the reported result measured and on which data?"

STAGES = (
    "document_pre_analyzer",
    "jev_router",
    "rag",
    "vision",
    "analysis",
    "critique",
)


def _sanitize(message: object) -> str:
    """Redact any known credential from a provider error message."""
    text = "" if message is None else str(message)
    for resolver in (resolve_api_key, critique_api_key):
        try:
            text = mask_secret(text, resolver())
        except Exception:  # pragma: no cover - masking must never raise
            continue
    return text


def _failure(stage: str, exc: BaseException, elapsed_ms: float) -> Dict[str, Any]:
    """Build one structured stage failure without leaking a credential."""
    return {
        "stage": stage,
        "status": "failed",
        "elapsed_ms": elapsed_ms,
        "error_type": type(exc).__name__,
        "error": _sanitize(exc),
        "result": None,
    }


def _completed(stage: str, elapsed_ms: float, result: Any) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "completed",
        "elapsed_ms": elapsed_ms,
        "error_type": None,
        "error": None,
        "result": result,
    }


def _build_jev_client() -> Any:
    """
    Build the frozen TypeSafe client with its credential.

    ``TypeSafeClient`` deliberately takes the key as a constructor argument (its
    own live smoke test reads ``TYPESAFE_API_KEY`` and passes it in), so the
    environment lookup belongs here in the orchestration layer rather than in the
    frozen router. The key is never logged.
    """
    from backend.jev_router.client import TypeSafeClient

    return TypeSafeClient(api_key=(os.getenv("TYPESAFE_API_KEY") or "").strip())


def _finalize(
    stages: Dict[str, Dict[str, Any]], started: float, failed_stage: Optional[str]
) -> Dict[str, Any]:
    """Attach the overall status and total runtime to the stage map."""
    payload: Dict[str, Any] = {
        "pipeline": "ai-research-paper-critic",
        "status": "failed" if failed_stage else "completed",
        "failed_stage": failed_stage,
        "elapsed_ms": round((time.time() - started) * 1000, 2),
        "stages": stages,
    }
    return payload


def run_pipeline(
    pdf_path: str,
    query: str = DEFAULT_QUERY,
    *,
    vision_output_dir: Optional[str] = None,
    jev_client: Optional[Any] = None,
    embedding_provider: Optional[Any] = None,
    vision_client: Optional[Any] = None,
    analysis_client: Optional[Any] = None,
    critique_client: Optional[Any] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete pipeline once and return every structured stage result.

    Parameters
    ----------
    pdf_path: path to the uploaded PDF
    query: retrieval query handed to the RAG Agent
    vision_output_dir: optional directory for extracted visual assets
    jev_client / embedding_provider / vision_client / analysis_client /
    critique_client: injection points used by the offline tests; when omitted the
        production client is built from the environment configuration.
    progress: optional callback invoked as ``progress(stage, status)``.

    Returns
    -------
    A JSON-serializable dict with one entry per stage plus timings and an overall
    status. A failed stage yields ``status="failed"`` and stops the stages that
    depend on it.
    """
    started = time.time()
    stages: Dict[str, Dict[str, Any]] = {}

    def notify(stage: str, status: str) -> None:
        if progress is not None:
            try:
                progress(stage, status)
            except Exception:  # pragma: no cover - progress must never break a run
                logger.debug("progress callback failed for stage %s", stage)

    # 1. Document Pre-Analyzer (frozen, local, no API).
    notify("document_pre_analyzer", "running")
    tick = time.time()
    try:
        profile = analyze_document(pdf_path)
    except Exception as exc:
        stages["document_pre_analyzer"] = _failure(
            "document_pre_analyzer", exc, round((time.time() - tick) * 1000, 2)
        )
        return _finalize(stages, started, "document_pre_analyzer")
    stages["document_pre_analyzer"] = _completed(
        "document_pre_analyzer", round((time.time() - tick) * 1000, 2), profile
    )
    notify("document_pre_analyzer", "completed")

    # 2. JEV Router (frozen, real TypeSafe call, real decision).
    notify("jev_router", "running")
    tick = time.time()
    try:
        routing_state = route_document(
            profile, client=jev_client or _build_jev_client()
        )
    except Exception as exc:
        stages["jev_router"] = _failure(
            "jev_router", exc, round((time.time() - tick) * 1000, 2)
        )
        return _finalize(stages, started, "jev_router")
    stages["jev_router"] = _completed(
        "jev_router", round((time.time() - tick) * 1000, 2), routing_state
    )
    notify("jev_router", "completed")

    # 3. RAG Agent (governed by the routing decision; never forced on).
    notify("rag", "running")
    tick = time.time()
    try:
        rag_result = RAGAgent(embedding_provider=embedding_provider).run(
            pdf_path=pdf_path, routing_state=routing_state, query=query
        )
    except Exception as exc:
        stages["rag"] = _failure("rag", exc, round((time.time() - tick) * 1000, 2))
        return _finalize(stages, started, "rag")
    stages["rag"] = _completed(
        "rag", round((time.time() - tick) * 1000, 2), rag_result
    )
    notify("rag", "completed")

    # 4. Vision Agent (governed by the routing decision; never forced on).
    notify("vision", "running")
    tick = time.time()
    try:
        vision_result = VisionAgent(vision_client=vision_client).run(
            pdf_path=pdf_path,
            document_profile=profile,
            routing_state=routing_state,
            output_dir=vision_output_dir,
        )
    except Exception as exc:
        stages["vision"] = _failure(
            "vision", exc, round((time.time() - tick) * 1000, 2)
        )
        return _finalize(stages, started, "vision")
    stages["vision"] = _completed(
        "vision", round((time.time() - tick) * 1000, 2), vision_result
    )
    notify("vision", "completed")

    # 5. Analysis Agent (frozen, governed by the routing decision).
    notify("analysis", "running")
    tick = time.time()
    try:
        analysis_result = AnalysisAgent(analysis_client=analysis_client).run(
            document_profile=profile,
            routing_state=routing_state,
            rag_result=rag_result,
            vision_result=vision_result,
        )
    except Exception as exc:
        stages["analysis"] = _failure(
            "analysis", exc, round((time.time() - tick) * 1000, 2)
        )
        return _finalize(stages, started, "analysis")
    stages["analysis"] = _completed(
        "analysis", round((time.time() - tick) * 1000, 2), analysis_result
    )
    notify("analysis", "completed")

    # 6. Critique Engine (frozen, reuses the analysis level chosen by JEV).
    notify("critique", "running")
    tick = time.time()
    try:
        critique_result = CritiqueEngine(critique_client=critique_client).generate_critique(
            document_profile=profile,
            routing_state=routing_state,
            rag_result=rag_result,
            vision_result=vision_result,
            analysis_result=analysis_result,
        )
    except Exception as exc:
        stages["critique"] = _failure(
            "critique", exc, round((time.time() - tick) * 1000, 2)
        )
        return _finalize(stages, started, "critique")
    stages["critique"] = _completed(
        "critique", round((time.time() - tick) * 1000, 2), critique_result
    )
    notify("critique", "completed")

    # 7. The critique engine already validated its own output; a structured
    #    validation_failed status is reported as such and never hidden.
    overall = (
        "completed"
        if critique_result.get("status") == "completed"
        else "failed"
    )
    payload = _finalize(stages, started, None if overall == "completed" else "critique")
    payload["status"] = overall
    payload["critique_status"] = critique_result.get("status")
    return payload