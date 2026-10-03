"""
Live API Smoke Test for TypeSafe JEV System One Integration.

Instructions:
1. Loads TYPESAFE_API_KEY from environment or .env
2. If no API key is found, prints a clear message and exits gracefully (code 0).
3. If API key exists:
   - Discovers available models via GET /v1/models
   - Builds compact JEV input for Attention Is All You Need benchmark profile
   - Executes live POST /v1/systemone request
   - Safely prints raw structured response
   - Normalizes and prints Routing Decision State
   - NEVER logs, prints, or exposes the API key or authorization headers.

Note: This script is standalone and NOT run automatically as part of pytest.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from dotenv import load_dotenv

# Load .env
load_dotenv()

from backend.jev_router.client import TypeSafeClient, TypeSafeError
from backend.jev_router.config import DEFAULT_JEV_MODEL, SYSTEM_ONE_QUESTIONS
from backend.jev_router.router import (
    get_available_models,
    resolve_jev_model,
    parse_system_one_response,
    route_document,
)
from backend.jev_router.routing_state import serialize_routing_state
from backend.jev_router.explanations import generate_routing_explanation


def run_live_smoke_test():
    print("=" * 60)
    print("TypeSafe JEV Live Smoke Test")
    print("=" * 60)

    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        print("[INFO] No TYPESAFE_API_KEY detected in environment or .env.")
        print("[INFO] To run live API smoke test, set TYPESAFE_API_KEY in your .env file.")
        print("[INFO] Exiting safely without failure.")
        return 0

    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"[STATUS] TYPESAFE_API_KEY detected ({masked_key}). Connecting to TypeSafe API...")

    client = TypeSafeClient(api_key=api_key)

    # Step 1: Model Discovery
    print("\n--- Step 1: Model Discovery (GET /v1/models) ---")
    try:
        models = get_available_models(client=client)
        print(f"Successfully discovered {len(models)} model(s):")
        for m in models:
            print(f"  - Name: {m.get('name')} | Release: {m.get('release_date')} | {m.get('description')}")
    except TypeSafeError as e:
        print(f"[ERROR] Failed to discover models: {e}")
        return 1

    selected_model = resolve_jev_model(client=client)
    print(f"\nSelected Model for System One: '{selected_model}'")

    # Step 2: Attention Is All You Need Benchmark Profile
    print("\n--- Step 2: Benchmark Paper Profile ---")
    benchmark_profile = {
        "filename": "NIPS-2017-attention-is-all-you-need-Paper.pdf",
        "page_count": 11,
        "text_length": 38600,
        "word_count": 4990,
        "sections": [{"title": f"Section {i}", "page": 1, "level": 3 if i == 5 else 1} for i in range(23)],
        "figures": {"count": 2, "items": [{"caption": "Figure 1"}, {"caption": "Figure 2"}]},
        "tables": {"count": 3, "items": [{"caption": "Table 1"}, {"caption": "Table 2"}, {"caption": "Table 3"}]},
        "equations": {"count": 5, "items": [{"caption": "Eq 1"}, {"caption": "Eq 2"}]},
        "references": {"has_references": True, "count": 32},
        "content_characteristics": {
            "has_methodology": True,
            "has_experimental_results": True,
            "has_visual_content": True,
            "has_tables": True,
            "has_mathematical_content": True,
            "has_references": True,
            "document_nature": "born-digital",
        },
    }
    print("Benchmark profile initialized: 11 pages, 4990 words, 2 figures, 3 tables, 5 equations, 32 refs.")

    # Step 3: Execute Live System One Request
    print("\n--- Step 3: Live System One Evaluation (POST /v1/systemone) ---")
    try:
        routing_state = route_document(
            document_profile=benchmark_profile,
            client=client,
            model=selected_model,
        )
    except Exception as e:
        print(f"[ERROR] Live routing request failed: {e}")
        return 1

    # Step 4: Display Results
    print("\n--- Step 4: Normalized Routing State ---")
    print(serialize_routing_state(routing_state, indent=2))

    # Step 5: Deterministic Explanations
    print("\n--- Step 5: Deterministic Explanations ---")
    from backend.jev_router.input_builder import build_jev_input
    compact_input = build_jev_input(benchmark_profile)
    explanations = generate_routing_explanation(compact_input, routing_state)
    for expl in explanations:
        print(f"  * {expl}")

    print("\n" + "=" * 60)
    print("Live API Smoke Test: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_live_smoke_test())
