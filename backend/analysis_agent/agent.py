"""
Analysis Agent entry point and orchestrator (Phase 5).

Coordinates:
1. Routing state inspection (analysis.enabled, analysis.level)
2. If analysis is disabled (defensive path - the JEV contract always enables it):
   ZERO model calls and no context assembly at all
3. If analysis is enabled:
   - the bounded evidence bundle is assembled from the frozen Document Profile plus
     whatever the frozen RAG and Vision Agents produced (context_builder)
   - the analysis model is selected from the JEV capability level (env config)
   - ONE model call produces the structured evaluation (analyzer)
   - deterministic gaps, model-reported gaps and analyzer validation notes are
     merged into a single auditable ``evidence_gaps`` list

Scope boundaries
----------------
This agent performs evidence-grounded evaluation only. It does not rebuild any
upstream work (no re-parsing, no re-chunking, no embedding, no retrieval, no
re-detection of figures/tables/equations, no visual analysis), it does not write
the final critique, and it never emits an overall numeric paper score.

Failure isolation
-----------------
A failed context build or model call produces a structured result
(``status="failed"`` with a recorded failure) instead of an exception, and the
error text is passed through a credential-masking step.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.analysis_agent.analyzer import analyze_evidence
from backend.analysis_agent.config import (
    AnalysisConfig,
    normalize_level,
    resolve_model_for_level,
)
from backend.analysis_agent.context_builder import (
    ContextBuilderError,
    build_evidence_bundle,
)
from backend.analysis_agent.llm_client import (
    BaseAnalysisClient,
    build_analysis_client,
    mask_secret,
    resolve_api_key,
)
from backend.analysis_agent.schemas import (
    AnalysisFailure,
    AnalysisResult,
    EvidenceBundle,
    EvidenceGap,
    EvidenceInputs,
    RoutingSummary,
)

logger = logging.getLogger(__name__)

STAGE_EVIDENCE_BUNDLE = "evidence_bundle"
STAGE_ANALYSIS = "analysis"


class AnalysisAgent:
    """
    Analysis Agent orchestrating bounded evidence assembly and JEV-governed
    structured evaluation.
    """

    def __init__(
        self,
        analysis_client: Optional[BaseAnalysisClient] = None,
        config: Optional[AnalysisConfig] = None,
    ):
        self._analysis_client = analysis_client
        self.config = config or AnalysisConfig()

    def _get_client(self) -> BaseAnalysisClient:
        """Lazy client construction (no credential is needed until first call)."""
        if self._analysis_client is None:
            self._analysis_client = build_analysis_client(self.config)
        return self._analysis_client

    def _provider_name(self) -> str:
        if self._analysis_client is None:
            return self.config.provider
        return (
            getattr(self._analysis_client, "provider_name", None) or self.config.provider
        )

    def _safe_error(self, exc: BaseException) -> str:
        """Error text with any credential redacted (defensive second layer)."""
        return mask_secret(str(exc), resolve_api_key(self.config.provider))

    @staticmethod
    def normalize_level(level: Any) -> Optional[str]:
        """
        Normalize a JEV capability level to a plain lowercase string (or None).

        The frozen router exposes ``ModelLevel`` enum members inside its routing
        state dict, whose ``str()`` form is ``"ModelLevel.<NAME>"``.
        """
        return normalize_level(level)

    def resolve_model(self, level: Any) -> str:
        """
        Map a JEV analysis capability level to a configured model.

        Raises AnalysisConfigurationError when the level has no configured model;
        the agent never silently downgrades to a different capability tier.
        """
        return resolve_model_for_level(
            level,
            level_models=self.config.level_models,
            allow_fallback=self.config.allow_level_fallback,
        )

    @staticmethod
    def _read_analysis_state(routing_state: Any) -> Tuple[bool, Any]:
        """
        Read analysis.enabled / analysis.level from the frozen routing state.

        The JEV contract requires ``analysis.enabled`` to be True, so the fallback
        for a partially specified state is True: analysis is the core stage and a
        missing flag must not silently skip it.
        """
        if not isinstance(routing_state, dict):
            return True, None
        state = routing_state.get("analysis")
        if not isinstance(state, dict):
            return bool(routing_state.get("analysis_enabled", True)), routing_state.get(
                "analysis_level"
            )
        return bool(state.get("enabled", True)), state.get("level")

    @staticmethod
    def _read_routing_summary(routing_state: Any) -> RoutingSummary:
        """Read-only, verbatim view of the routing decision consumed here."""
        state = routing_state if isinstance(routing_state, dict) else {}

        def block(name: str) -> Dict[str, Any]:
            value = state.get(name)
            return value if isinstance(value, dict) else {}

        analysis = block("analysis")
        rag = block("rag")
        vision = block("vision")

        confidence = state.get("confidence")
        routing_confidence: Optional[Dict[str, float]] = None
        if isinstance(confidence, dict):
            routing_confidence = {
                str(key): float(value)
                for key, value in confidence.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }

        router_version = state.get("router_version")
        return RoutingSummary(
            router_version=str(router_version) if router_version is not None else None,
            analysis_enabled=bool(
                analysis.get("enabled", state.get("analysis_enabled", True))
            ),
            analysis_level=normalize_level(
                analysis.get("level", state.get("analysis_level"))
            ),
            rag_enabled=bool(rag.get("enabled", state.get("rag_enabled", False))),
            rag_level=normalize_level(rag.get("level", state.get("rag_level"))),
            vision_enabled=bool(
                vision.get("enabled", state.get("vision_enabled", False))
            ),
            vision_level=normalize_level(vision.get("level", state.get("vision_level"))),
            routing_confidence=routing_confidence,
        )

    @staticmethod
    def _evidence_inputs(bundle: EvidenceBundle) -> EvidenceInputs:
        """Summarize the evidence the analysis actually ran on."""
        counts = bundle.counts or {}
        return EvidenceInputs(
            sections=int(counts.get("sections", 0) or 0),
            figures=int(counts.get("figures_available", 0) or 0),
            tables=int(counts.get("tables_available", 0) or 0),
            equations=int(counts.get("equations_available", 0) or 0),
            references=int(counts.get("references", 0) or 0),
            rag_enabled=bool(counts.get("rag_chunks", 0)),
            rag_chunks_used=int(counts.get("rag_chunks", 0) or 0),
            vision_enabled=bool(counts.get("visual_assets", 0)),
            visual_assets_used=int(counts.get("visual_assets", 0) or 0),
            context_chars=int(bundle.total_evidence_chars or 0),
            truncations=list(bundle.truncations),
        )

    @staticmethod
    def _merge_gaps(*collections: List[EvidenceGap]) -> List[EvidenceGap]:
        """
        Merge deterministic, model-reported and analyzer notes into one list.

        Deduplication is by (category, gap text) so the same missing-evidence fact
        reported by two sources appears once, while the deterministic source is
        listed first (it is the one the pipeline can vouch for).
        """
        merged: List[EvidenceGap] = []
        seen = set()
        for collection in collections:
            for gap in collection or []:
                key = (gap.category, gap.gap.strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                merged.append(gap)
        return merged

    def _failure_result(
        self,
        *,
        stage: str,
        error: str,
        level: Optional[str],
        model: Optional[str],
        routing: RoutingSummary,
        elapsed_time_ms: float,
        evidence_inputs: Optional[EvidenceInputs] = None,
        evidence_gaps: Optional[List[EvidenceGap]] = None,
    ) -> Dict[str, Any]:
        """Build a structured failure result (never an exception, never a secret)."""
        return AnalysisResult(
            agent="analysis",
            status="failed",
            enabled=True,
            level=level,
            model=model,
            provider=self._provider_name(),
            routing=routing,
            evidence_inputs=evidence_inputs or EvidenceInputs(),
            analysis=None,
            evidence_gaps=evidence_gaps or [],
            unverified_evidence_refs=0,
            elapsed_time_ms=elapsed_time_ms,
            message=f"Analysis could not be completed ({stage}): {error}",
            failures=[AnalysisFailure(stage=stage, error=error, model=model)],
        ).model_dump()

    def run(
        self,
        document_profile: Dict[str, Any],
        routing_state: Dict[str, Any],
        rag_result: Optional[Dict[str, Any]] = None,
        vision_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Analysis Agent workflow governed by the JEV routing state.

        Parameters
        ----------
        document_profile: frozen Document Pre-Analyzer profile (evidence structure)
        routing_state: read-only JEV routing decision state
        rag_result: optional frozen RAG Agent result (retrieved text evidence)
        vision_result: optional frozen Vision Agent result (visual evidence)

        Returns
        -------
        Standardized JSON-serializable dictionary matching AnalysisResult.
        """
        start_time = time.time()
        routing = self._read_routing_summary(routing_state)
        enabled, raw_level = self._read_analysis_state(routing_state)
        level = normalize_level(raw_level)

        # 1. Bypass path: no evidence assembly, no client construction, no model call.
        if not enabled:
            return AnalysisResult(
                agent="analysis",
                status="disabled",
                enabled=False,
                level=level,
                model=None,
                provider=None,
                routing=routing,
                evidence_inputs=EvidenceInputs(),
                analysis=None,
                evidence_gaps=[],
                unverified_evidence_refs=0,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                message=(
                    "Analysis Agent disabled by the JEV routing state. No evidence was "
                    "assembled and no model calls were made."
                ),
            ).model_dump()

        # 2. Fail fast on a misconfigured capability level (before any model work).
        model = self.resolve_model(raw_level)

        # 3. Assemble the bounded evidence bundle from upstream artifacts only.
        try:
            bundle = build_evidence_bundle(
                document_profile,
                rag_result=rag_result,
                vision_result=vision_result,
                routing_state=routing_state,
                config=self.config,
            )
        except ContextBuilderError as exc:
            return self._failure_result(
                stage=STAGE_EVIDENCE_BUNDLE,
                error=self._safe_error(exc),
                level=level,
                model=model,
                routing=routing,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        evidence_inputs = self._evidence_inputs(bundle)

        # 4. One model call, one validated payload. No repair round trip.
        try:
            parsed = analyze_evidence(bundle, self._get_client(), model)
        except Exception as exc:
            logger.warning("Analysis model call or parsing failed: %s", exc)
            return self._failure_result(
                stage=STAGE_ANALYSIS,
                error=self._safe_error(exc),
                level=level,
                model=model,
                routing=routing,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                evidence_inputs=evidence_inputs,
                evidence_gaps=list(bundle.deterministic_gaps),
            )

        # 5. Merge deterministic gaps, model gaps and analyzer validation notes.
        gaps = self._merge_gaps(
            bundle.deterministic_gaps, parsed.payload.evidence_gaps, parsed.notes
        )

        payload = parsed.payload
        message = (
            f"Produced a structured evidence-grounded evaluation using '{model}': "
            f"{len(payload.claim_evidence_matrix.items)} claim(s) mapped to evidence, "
            f"{len(payload.strengths)} strength(s), {len(payload.weaknesses)} "
            f"weakness(s), {len(gaps)} recorded evidence gap(s)."
        )
        if parsed.overclaim_flags:
            message += (
                f" Confirmatory language ({', '.join(parsed.overclaim_flags)}) was "
                "recorded as a caveat."
            )
        if parsed.unverified_evidence_refs:
            message += (
                f" {parsed.unverified_evidence_refs} reference(s) could not be verified "
                "against the evidence bundle."
            )
        if parsed.downgraded_statuses:
            message += (
                f" {parsed.downgraded_statuses} statement(s) were downgraded to "
                "'unclear' for missing provenance."
            )

        return AnalysisResult(
            agent="analysis",
            status="completed",
            enabled=True,
            level=level,
            model=model,
            provider=self._provider_name(),
            routing=routing,
            evidence_inputs=evidence_inputs,
            analysis=payload,
            evidence_gaps=gaps,
            unverified_evidence_refs=parsed.unverified_evidence_refs,
            elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            message=message,
            failures=[],
        ).model_dump()
