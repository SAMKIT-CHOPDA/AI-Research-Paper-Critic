"""
Vision Agent entry point and orchestrator (Phase 4).

Coordinates:
1. Routing state inspection (vision.enabled, vision.level)
2. If vision is disabled: ZERO model calls and no asset extraction at all
3. If vision is enabled:
   - confirmed visual inventory comes from the frozen Document Pre-Analyzer
   - visual assets are extracted from the original PDF (asset_extractor)
   - images are normalized for the model (image_preprocessor)
   - the multimodal model is selected from the JEV capability level (env config)
   - each asset is analyzed independently (analyzer)
   - structured, JSON-serializable visual evidence is returned

Scope boundaries
----------------
This agent performs semantic visual analysis only. It does not rediscover
figures or tables, does not make routing decisions, does not retrieve text, does
not write the critique and does not perform final multimodal synthesis.

Failure isolation
-----------------
A failed asset (extraction or model call) is recorded in ``failures`` /
``extraction_failures`` and the remaining assets are still analyzed. Nothing is
silently dropped.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from backend.vision_agent.analyzer import analyze_asset
from backend.vision_agent.asset_extractor import (
    AssetExtractionError,
    extract_visual_assets,
)
from backend.vision_agent.config import (
    VisionConfig,
    normalize_level,
    resolve_model_for_level,
)
from backend.vision_agent.schemas import (
    AssetExtractionFailure,
    VisionAgentResult,
    VisualAnalysisFailure,
    VisualAnalysisResult,
)
from backend.vision_agent.vision_client import (
    BaseVisionClient,
    build_vision_client,
    mask_secret,
    resolve_api_key,
)

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Vision Agent orchestrating asset extraction, preprocessing and JEV-governed
    multimodal visual analysis.
    """

    def __init__(
        self,
        vision_client: Optional[BaseVisionClient] = None,
        config: Optional[VisionConfig] = None,
    ):
        self._vision_client = vision_client
        self.config = config or VisionConfig()

    def _get_client(self) -> BaseVisionClient:
        """Lazy client construction (no credential is needed until first call)."""
        if self._vision_client is None:
            self._vision_client = build_vision_client(self.config)
        return self._vision_client

    def _provider_name(self) -> str:
        if self._vision_client is None:
            return self.config.provider
        return getattr(self._vision_client, "provider_name", None) or self.config.provider

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
        Map a JEV vision capability level to a configured multimodal model.

        Raises VisionConfigurationError when the level has no configured model;
        the agent never silently downgrades to a different capability tier.
        """
        return resolve_model_for_level(
            level,
            level_models=self.config.level_models,
            allow_fallback=self.config.allow_level_fallback,
        )

    @staticmethod
    def _read_vision_state(routing_state: Dict[str, Any]) -> tuple:
        """Read vision.enabled / vision.level from the frozen routing state."""
        vision_state = routing_state.get("vision") if isinstance(routing_state, dict) else None
        if vision_state is None:
            enabled = bool(routing_state.get("vision_enabled", False)) if isinstance(routing_state, dict) else False
            level = routing_state.get("vision_level") if isinstance(routing_state, dict) else None
            return enabled, level
        return bool(vision_state.get("enabled", False)), vision_state.get("level")

    @staticmethod
    def _count_equations(document_profile: Optional[Dict[str, Any]]) -> int:
        """Count confirmed equations (context for later stages, never analyzed here)."""
        if not document_profile:
            return 0
        block = document_profile.get("equations") or {}
        count = block.get("count")
        if isinstance(count, int) and count >= 0:
            return count
        confirmed = block.get("confirmed_equations") or block.get("items") or []
        return len(confirmed) if isinstance(confirmed, list) else 0


    def run(
        self,
        pdf_path: str,
        document_profile: Dict[str, Any],
        routing_state: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Vision Agent workflow governed by the JEV routing state.

        Parameters
        - pdf_path: path to the paper PDF
        - document_profile: frozen Document Pre-Analyzer profile (visual inventory)
        - routing_state: read-only JEV routing decision state
        - output_dir: optional override for where extracted assets are written

        Returns
        - Standardized JSON-serializable dictionary matching VisionAgentResult.
        """
        start_time = time.time()
        vision_enabled, vision_level = self._read_vision_state(routing_state)
        level = normalize_level(vision_level)
        equations_available = self._count_equations(document_profile)

        # 1. Bypass path: no extraction, no client construction, no model calls.
        if not vision_enabled:
            return VisionAgentResult(
                agent="vision",
                enabled=False,
                level=level,
                model=None,
                provider=None,
                assets_found=0,
                assets_analyzed=0,
                assets_failed=0,
                results=[],
                failures=[],
                extraction_failures=[],
                equations_available=equations_available,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                message=(
                    "Vision Agent disabled by JEV Router decision. No assets were "
                    "extracted and no model calls were made."
                ),
            ).model_dump()

        # 2. Fail fast on a misconfigured capability level (before any API work).
        model = self.resolve_model(vision_level)

        # 3. Extract the confirmed visual inventory from the PDF.
        try:
            report = extract_visual_assets(
                pdf_path=pdf_path,
                document_profile=document_profile,
                config=self.config,
                output_dir=output_dir,
            )
        except AssetExtractionError as exc:
            return VisionAgentResult(
                agent="vision",
                enabled=True,
                level=level,
                model=model,
                provider=self._provider_name(),
                assets_found=0,
                assets_analyzed=0,
                assets_failed=0,
                results=[],
                failures=[],
                extraction_failures=[
                    AssetExtractionFailure(
                        asset_id="document",
                        asset_type=None,
                        page_number=None,
                        source_id=None,
                        stage="extraction",
                        error=self._safe_error(exc),
                    )
                ],
                equations_available=equations_available,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                message=f"Visual asset extraction failed: {self._safe_error(exc)}",
            ).model_dump()

        assets_found = len(report.assets)

        if assets_found == 0:
            message = "No confirmed visual assets were available for analysis."
            if report.failures:
                message = "No visual asset could be extracted from the confirmed inventory."
            return VisionAgentResult(
                agent="vision",
                enabled=True,
                level=level,
                model=model,
                provider=self._provider_name(),
                assets_found=0,
                assets_analyzed=0,
                assets_failed=0,
                results=[],
                failures=[],
                extraction_failures=list(report.failures),
                equations_available=equations_available,
                elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
                message=message,
            ).model_dump()

        # 4. Analyze each asset independently; isolate every failure.
        client = self._get_client()
        results: List[VisualAnalysisResult] = []
        failures: List[VisualAnalysisFailure] = []

        for asset in report.assets:
            try:
                results.append(analyze_asset(asset, client, model))
            except Exception as exc:
                logger.warning("Vision analysis failed for %s: %s", asset.asset_id, exc)
                failures.append(
                    VisualAnalysisFailure(
                        asset_id=asset.asset_id,
                        asset_type=asset.asset_type,
                        page_number=asset.page_number,
                        stage="analysis",
                        error=self._safe_error(exc),
                        model=model,
                    )
                )

        message = (
            f"Analyzed {len(results)} of {assets_found} confirmed visual asset(s) "
            f"using '{model}'."
        )
        if failures:
            message += f" {len(failures)} asset(s) failed and were recorded."

        return VisionAgentResult(
            agent="vision",
            enabled=True,
            level=level,
            model=model,
            provider=self._provider_name(),
            assets_found=assets_found,
            assets_analyzed=len(results),
            assets_failed=len(failures),
            results=results,
            failures=failures,
            extraction_failures=list(report.failures),
            equations_available=equations_available,
            elapsed_time_ms=round((time.time() - start_time) * 1000, 2),
            message=message,
        ).model_dump()