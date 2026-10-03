"""
Critique Engine entry point and orchestrator (Phase 6).

Coordinates:
1. Model selection from the *already-made* JEV analysis level (no new routing
   decision is ever taken here)
2. Deterministic synthesis-context assembly from the frozen Document Profile, the
   frozen JEV routing state and the frozen RAG / Vision / Analysis outputs
3. ONE synthesis model call producing the structured final critique
4. Validation against the supplied artifacts, so an invented citation, an
   impossible page number, a numeric score or a leaked credential can never be
   returned as a finished report

Scope boundaries
-----------------
This engine is a synthesis layer. It does not rediscover figures or tables, run
retrieval, perform visual analysis, repeat the research evaluation, or make any
routing decision. It also never emits a paper score, a rank or a tier.

Failure isolation
-----------------
A missing analysis result, a failed context build, a failed model call, a
malformed response and a failed validation each produce a structured result
(``status="failed"`` or ``status="validation_failed"``) instead of an exception,
and every error text passes through a credential-masking step.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.critique_engine.config import (
    CritiqueConfig,
    CritiqueConfigurationError,
    normalize_level,
    resolve_model_for_level,
)
from backend.critique_engine.context_builder import (
    ContextBuilderError,
    build_synthesis_context,
)
from backend.critique_engine.llm_client import (
    BaseCritiqueClient,
    build_critique_client,
    mask_secret,
    resolve_api_key,
)
from backend.critique_engine.prompts import build_critique_prompt
from backend.critique_engine.schemas import (
    CritiqueFailure,
    CritiqueGap,
    CritiquePayload,
    CritiqueResult,
    RoutingSummary,
    SynthesisContext,
)
from backend.critique_engine.validator import (
    CritiqueValidationError,
    build_provenance,
    validate_model_response,
)

logger = logging.getLogger(__name__)

STAGE_ANALYSIS_INPUT = "analysis_input"
STAGE_CONTEXT_BUILDER = "context_builder"
STAGE_CRITIQUE_GENERATION = "critique_generation"
STAGE_VALIDATION = "validation"


def _response_schema() -> Dict[str, Any]:
    """
    Best-effort JSON schema for the payload, forwarded to providers that support it.

    The contract never depends on this: an endpoint that ignores it still receives
    the same textual contract and returns one JSON object.
    """
    try:
        return CritiquePayload.model_json_schema()
    except Exception:  # pragma: no cover - defensive, the schema shape is stable
        return {}


class CritiqueEngine:
    """
    Critique Engine orchestrating bounded synthesis of the structured findings.

    The engine consumes the frozen upstream artifacts and produces the final
    human-readable critique. It never re-does any upstream work.
    """

    def __init__(
        self,
        critique_client: Optional[BaseCritiqueClient] = None,
        config: Optional[CritiqueConfig] = None,
    ):
        self._client = critique_client
        self.config = config or CritiqueConfig()

    def _get_client(self) -> BaseCritiqueClient:
        """Lazy client construction (no credential is needed until first call)."""
        if self._client is None:
            self._client = build_critique_client(self.config)
        return self._client

    def _provider_name(self) -> str:
        if self._client is None:
            return self.config.provider
        return getattr(self._client, "provider_name", None) or self.config.provider

    def _safe_error(self, exc: BaseException) -> str:
        """Error text with any credential redacted (defensive second layer)."""
        return mask_secret(str(exc), resolve_api_key(self.config.provider))

    @staticmethod
    def normalize_level(level: Any) -> Optional[str]:
        """Normalize a JEV capability level to a plain lowercase string (or None)."""
        return normalize_level(level)

    def resolve_model(self, level: Any) -> str:
        """
        Map the JEV analysis capability level to a configured synthesis model.

        The Critique Engine introduces no capability decision of its own: it reuses
        ``routing_state["analysis"]["level"]`` so the architecture keeps a single
        routing decision. A single-model deployment is an explicit configuration
        choice (``CRITIQUE_MODEL``), never a hidden fallback.
        """
        return resolve_model_for_level(
            level,
            level_models=self.config.level_models,
            allow_fallback=self.config.allow_level_fallback,
            single_model=self.config.single_model,
        )

    @staticmethod
    def _read_analysis_level(routing_state: Any) -> Tuple[Optional[str], Any]:
        """Read the analysis level from the frozen routing state (read-only)."""
        if not isinstance(routing_state, dict):
            return None, None
        state = routing_state.get("analysis")
        if isinstance(state, dict):
            return normalize_level(state.get("level")), state.get("level")
        return (
            normalize_level(routing_state.get("analysis_level")),
            routing_state.get("analysis_level"),
        )

    @staticmethod
    def _context_inputs(
        context: SynthesisContext, analysis_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Auditable summary of exactly what was synthesized."""
        analysis_payload = None
        if isinstance(analysis_result, dict):
            analysis_payload = analysis_result.get("analysis")
        summary: Dict[str, Any] = {
            "title": context.document.title,
            "page_count": context.document.page_count,
            "rag_enabled": context.routing.rag_enabled,
            "vision_enabled": context.routing.vision_enabled,
            "analysis_available": isinstance(analysis_payload, dict),
            "analysis_level": context.routing.analysis_level,
            "context_chars": context.total_context_chars,
            "sources_available": len(context.source_index),
            "truncations": list(context.truncations),
        }
        summary.update(context.counts)
        return summary

    @staticmethod
    def _merge_gaps(*collections: List[CritiqueGap]) -> List[CritiqueGap]:
        """
        Merge gap records into one list, deduplicated by (category, text).

        The deterministic source is listed first because the pipeline can vouch for
        it without trusting any model.
        """
        merged: List[CritiqueGap] = []
        seen = set()
        for collection in collections:
            for gap in collection or []:
                key = (gap.category, gap.gap.strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                merged.append(gap)
        return merged

    def _structured_failure(
        self,
        *,
        status: str,
        stage: str,
        error: str,
        level: Optional[str],
        model: Optional[str],
        routing: RoutingSummary,
        elapsed_time_ms: float,
        context_inputs: Optional[Dict[str, Any]] = None,
        evidence_gaps: Optional[List[CritiqueGap]] = None,
        warnings: Optional[List[str]] = None,
        unverified_evidence_refs: int = 0,
    ) -> Dict[str, Any]:
        """Build a structured failure result (never an exception, never a secret)."""
        return CritiqueResult(
            status=status,  # type: ignore[arg-type]
            level=level,
            model=model,
            provider=self._provider_name(),
            routing=routing,
            context_inputs=context_inputs or {},
            evidence_gaps=evidence_gaps or [],
            unverified_evidence_refs=unverified_evidence_refs,
            warnings=warnings or [],
            elapsed_time_ms=elapsed_time_ms,
            message=f"The critique could not be produced ({stage}): {error}",
            failures=[CritiqueFailure(stage=stage, error=error, model=model)],
        ).model_dump()

    def generate_critique(
        self,
        document_profile: Optional[Dict[str, Any]],
        routing_state: Optional[Dict[str, Any]] = None,
        rag_result: Optional[Dict[str, Any]] = None,
        vision_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Critique Engine workflow.

        Parameters
        ----------
        document_profile: frozen Document Pre-Analyzer profile (may be absent)
        routing_state: read-only JEV routing decision state
        rag_result: optional frozen RAG Agent result
        vision_result: optional frozen Vision Agent result
        analysis_result: frozen Analysis Agent result (required: it carries the
            structured findings this stage synthesizes)

        Returns
        -------
        Standardized JSON-serializable dictionary matching CritiqueResult.
        """
        start_time = time.time()

        # 1. The analysis result is what there is to synthesize. Without it the
        #    engine has nothing to do, and it says so instead of inventing content.
        analysis_payload = None
        if isinstance(analysis_result, dict):
            analysis_payload = analysis_result.get("analysis")
        if not isinstance(analysis_payload, dict) or not analysis_payload:
            error = (
                "No structured analysis result was supplied. The Critique Engine "
                "synthesizes the analysis stage's findings and cannot produce a "
                "critique without them."
            )
            logger.warning(error)
            return self._structured_failure(
                status="failed",
                stage=STAGE_ANALYSIS_INPUT,
                error=error,
                level=None,
                model=None,
                routing=RoutingSummary(),
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        level, raw_level = self._read_analysis_level(routing_state)

        # 2. Resolve the synthesis model from the level JEV already chose.
        try:
            model = self.resolve_model(raw_level)
        except CritiqueConfigurationError as exc:
            return self._structured_failure(
                status="failed",
                stage=STAGE_CRITIQUE_GENERATION,
                error=self._safe_error(exc),
                level=level,
                model=None,
                routing=RoutingSummary(),
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        # 3. Assemble the deterministic synthesis context.
        try:
            context = build_synthesis_context(
                document_profile,
                routing_state=routing_state,
                rag_result=rag_result,
                vision_result=vision_result,
                analysis_result=analysis_result,
                config=self.config,
            )
        except ContextBuilderError as exc:
            return self._structured_failure(
                status="failed",
                stage=STAGE_CONTEXT_BUILDER,
                error=self._safe_error(exc),
                level=level,
                model=model,
                routing=RoutingSummary(),
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        context_inputs = self._context_inputs(context, analysis_result)
        prompt = build_critique_prompt(context)

        # 4. One synthesis call, then hard validation. No repair round trip.
        try:
            raw_text = self._get_client().generate_structured(
                model=model,
                system_prompt="",
                user_prompt=prompt,
                response_schema=_response_schema(),
            )
        except Exception as exc:
            logger.warning("Critique model call failed: %s", exc)
            return self._structured_failure(
                status="failed",
                stage=STAGE_CRITIQUE_GENERATION,
                error=self._safe_error(exc),
                level=level,
                model=model,
                routing=context.routing,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                context_inputs=context_inputs,
                evidence_gaps=self._merge_gaps(context.deterministic_gaps),
            )

        try:
            parsed, report = validate_model_response(
                raw_text,
                context,
                require_supported_sections=self.config.require_supported_sections,
            )
        except CritiqueValidationError as exc:
            validation_report = exc.report
            issues = validation_report.issues if validation_report else []
            warnings = [
                f"{issue.code} at {issue.location or 'payload'}: {issue.message}"
                for issue in issues[:10]
            ]
            logger.warning("Critique validation failed: %s", exc)
            return self._structured_failure(
                status="validation_failed",
                stage=STAGE_VALIDATION,
                error=self._safe_error(exc),
                level=level,
                model=model,
                routing=context.routing,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                context_inputs=context_inputs,
                evidence_gaps=self._merge_gaps(context.deterministic_gaps),
                warnings=warnings,
                unverified_evidence_refs=(
                    validation_report.unverified_evidence_refs
                    if validation_report
                    else 0
                ),
            )

        payload = parsed.payload
        gaps = self._merge_gaps(
            context.deterministic_gaps,
            context.analysis.evidence_gaps,
            payload.evidence_gaps,
            parsed.notes,
        )
        payload = payload.model_copy(update={"evidence_gaps": gaps})

        message = (
            f"Synthesized a final critique with '{model}' from the level-"
            f"{level or 'unspecified'} analysis findings: {len(payload.sections)} "
            f"section(s), {len(payload.claim_evidence_summary)} claim(s) assessed, "
            f"{len(payload.strengths)} strength(s), "
            f"{len(payload.limitations.author_stated)} author-stated and "
            f"{len(payload.limitations.analyst_identified)} analyst-identified "
            f"limitation(s), {len(gaps)} evidence gap(s)."
        )
        if parsed.unverified_evidence_refs:
            message += (
                f" {parsed.unverified_evidence_refs} citation(s) could not be matched "
                "to the supplied upstream artifacts."
            )

        result = CritiqueResult(
            status="completed",
            enabled=True,
            level=level,
            model=model,
            provider=self._provider_name(),
            routing=context.routing,
            sections=payload.sections,
            claim_evidence_summary=payload.claim_evidence_summary,
            strengths=payload.strengths,
            limitations=payload.limitations,
            evidence_gaps=gaps,
            open_questions=payload.open_questions,
            overall_assessment=payload.overall_assessment,
            provenance=build_provenance(payload),
            context_inputs=context_inputs,
            unverified_evidence_refs=parsed.unverified_evidence_refs,
            warnings=list(parsed.warnings)
            + [
                f"{issue.code} at {issue.location or 'payload'}: {issue.message}"
                for issue in report.warnings()
            ],
            elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            message=message,
            failures=[],
        )
        return result.model_dump()
