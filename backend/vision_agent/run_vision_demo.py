"""
Standalone Vision Agent demo / smoke runner (Phase 4).

Instructions:
1. Loads the multimodal credentials and model identifiers from environment / .env
   (VISION_API_KEY or the provider-specific key; VISION_MODEL_* per JEV level).
2. If no model is configured for the requested level, or no credential exists,
   prints a clear message and exits gracefully (code 0).
3. Otherwise:
   - runs the frozen Document Pre-Analyzer on "Attention Is All You Need" (or
     loads a cached profile JSON) to obtain the confirmed visual inventory
   - builds a routing state through the frozen JEV contract layer
   - runs the Vision Agent with vision disabled (bypass path) and then enabled
   - prints assets found, asset types/pages, the selected model, the structured
     analysis of each asset, explicit uncertainties and every failure
   - NEVER logs, prints, or exposes any API key.

Note: this script is standalone and is NOT executed automatically by pytest.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Extracted captions/context can contain Unicode that a legacy Windows cp1252
# console cannot encode; keep each stream's encoding but never raise on print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - platform dependent
        pass

from dotenv import load_dotenv

load_dotenv()

from backend.document_pre_analyzer.analyzer import analyze_document
from backend.jev_router.routing_state import create_routing_state
from backend.vision_agent.agent import VisionAgent
from backend.vision_agent.config import (
    VisionConfig,
    VisionConfigurationError,
    resolve_model_for_level,
)
from backend.vision_agent.vision_client import (
    VisionAuthError,
    build_vision_client,
    resolve_api_key,
)

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")


def _safe(text: object) -> str:
    """Return text printable on the active console encoding, never raising."""
    value = "" if text is None else str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except (LookupError, UnicodeError):  # pragma: no cover - platform dependent
        return value.encode("ascii", errors="replace").decode("ascii")


def _print_bypass(result: dict) -> None:
    print("\n--- Vision disabled (bypass path) ---")
    print(f"agent={result['agent']} enabled={result['enabled']} level={result['level']}")
    print(f"assets_found={result['assets_found']} results={len(result['results'])}")
    print(f"equations_available (context only)={result['equations_available']}")
    print(f"message: {_safe(result['message'])}")


def _print_analysis(result: dict) -> None:
    print("\n--- Vision enabled (multimodal analysis) ---")
    print(
        f"enabled={result['enabled']} level={result['level']} "
        f"provider={result['provider']} model={result['model']}"
    )
    print(
        f"assets_found={result['assets_found']} "
        f"assets_analyzed={result['assets_analyzed']} "
        f"assets_failed={result['assets_failed']} "
        f"elapsed={result['elapsed_time_ms']}ms"
    )
    print(f"message: {_safe(result['message'])}")

    for item in result["results"]:
        label = f"{item['asset_type']} | {item.get('caption') or ''}".strip()
        print(f"\n  [{item['asset_id']}] page {item['page_number']} | {_safe(label[:90])}")
        print(f"      section: {_safe(item['section'])}")
        print(f"      observation: {_safe(item['observation'])}")
        print(f"      interpretation: {_safe(item['interpretation'])}")
        print(f"      caption_consistency: {item['caption_consistency']['status']}")
        print(f"      numeric_authority: {item['numeric_authority']}")
        for element in item["key_elements"][:4]:
            print(f"      key element: {_safe(element)}")
        for uncertainty in item["uncertainties"][:4]:
            print(f"      uncertainty: {_safe(uncertainty)}")

    if result["extraction_failures"]:
        print("\n  Extraction failures:")
        for failure in result["extraction_failures"]:
            print(
                f"    {failure['asset_id']} page={failure['page_number']} "
                f"[{failure['stage']}]: {_safe(failure['error'])}"
            )

    if result["failures"]:
        print("\n  Analysis failures:")
        for failure in result["failures"]:
            print(
                f"    {failure['asset_id']} page={failure['page_number']} "
                f"[{failure['stage']}]: {_safe(failure['error'])}"
            )


def _load_profile(pdf_path: str, profile_path: str = None) -> dict:
    if profile_path:
        print(f"[STATUS] Loading cached Document Profile: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    print("[STATUS] Running the frozen Document Pre-Analyzer on the benchmark paper...")
    return analyze_document(pdf_path)


def _warm_up_client(client) -> None:
    """Validate credentials/configuration without issuing a model call."""
    ensure_configured = getattr(client, "ensure_configured", None)
    if callable(ensure_configured):
        ensure_configured()
        return
    if not resolve_api_key(getattr(client, "provider_name", None)):
        raise VisionAuthError(
            "No multimodal credential is configured. Set VISION_API_KEY (or the "
            "provider-specific key such as OPENAI_API_KEY) in the environment."
        )


def run_vision_demo(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vision Agent demonstration")
    parser.add_argument("pdf", nargs="?", default=BENCHMARK_PDF, help="paper PDF path")
    parser.add_argument("--level", default="advanced", choices=["basic", "medium", "advanced"])
    parser.add_argument("--profile", default=None, help="cached document profile JSON")
    parser.add_argument("--output-dir", default=None, help="where to write extracted assets")
    parser.add_argument("--max-assets", type=int, default=None)
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Vision Agent Demonstration (Phase 4)")
    print("=" * 60)

    if not os.path.exists(args.pdf):
        print(f"[ERROR] PDF not found at: {args.pdf}")
        return 1

    config = VisionConfig()
    if args.max_assets is not None:
        config.max_assets = args.max_assets

    # Model configuration is environment based and never hard-coded.
    try:
        model = resolve_model_for_level(
            args.level, config.level_models, config.allow_level_fallback
        )
    except VisionConfigurationError as exc:
        print("[INFO] No multimodal model is configured for this capability level.")
        print(f"[INFO] {exc}")
        print("[INFO] Set VISION_MODEL_BASIC / VISION_MODEL_MEDIUM / VISION_MODEL_ADVANCED")
        print("[INFO] (and VISION_API_KEY) in your .env file to run this demo.")
        print("[INFO] Exiting safely without failure.")
        return 0

    client = build_vision_client(config)
    print(f"[STATUS] Provider: {client.provider_name}")
    print(f"[STATUS] JEV vision level '{args.level}' -> model '{model}'")

    # Credentials are validated without ever printing the key itself.
    try:
        _warm_up_client(client)
    except VisionAuthError as exc:
        print("[INFO] Multimodal credentials are not configured.")
        print(f"[INFO] {exc}")
        print("[INFO] Exiting safely without failure.")
        return 0
    except Exception as exc:  # pragma: no cover - provider specific
        print(f"[ERROR] Unable to initialize the vision client: {_safe(exc)}")
        return 1

    profile = _load_profile(args.pdf, args.profile)
    inventory = profile.get("figures", {}), profile.get("tables", {}), profile.get("equations", {})
    print(
        "[STATUS] Confirmed inventory from the Document Pre-Analyzer: "
        f"{len(inventory[0].get('confirmed_figures', []))} figure(s), "
        f"{len(inventory[1].get('confirmed_tables', []))} table(s), "
        f"{len(inventory[2].get('confirmed_equations', []))} equation(s)"
    )

    agent = VisionAgent(vision_client=client, config=config)

    print("\n--- Step 1: Routing state, vision disabled (frozen JEV contract) ---")
    disabled_state = create_routing_state(
        analysis_level="advanced", rag_enabled=False, vision_enabled=False
    )
    print(json.dumps(disabled_state, indent=2, default=str))
    _print_bypass(
        agent.run(
            pdf_path=args.pdf,
            document_profile=profile,
            routing_state=disabled_state,
            output_dir=args.output_dir,
        )
    )

    print(f"\n--- Step 2: Enabling vision at level '{args.level}' ---")
    routing_state = create_routing_state(
        analysis_level="advanced",
        rag_enabled=False,
        vision_enabled=True,
        vision_level=args.level,
    )
    print(json.dumps(routing_state, indent=2, default=str))

    final_result = agent.run(
        pdf_path=args.pdf,
        document_profile=profile,
        routing_state=routing_state,
        output_dir=args.output_dir,
    )
    _print_analysis(final_result)

    print("\n--- Step 3: Machine-readable payload (truncated) ---")
    print(_safe(json.dumps(final_result, indent=2)[:2000]))

    print("\n" + "=" * 60)
    print("Vision Agent Demo: COMPLETED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_vision_demo())