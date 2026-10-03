"""
Standalone RAG Agent demo / smoke runner.

Instructions:
1. Loads OPENAI_API_KEY from environment or .env
2. If no API key is found, prints a clear message and exits gracefully (code 0).
3. If an API key exists:
   - Extracts the Attention Is All You Need benchmark paper
   - Consumes the frozen JEV Router's routing state (built via create_routing_state)
   - Runs the RAG agent under each JEV capability tier (basic / medium / advanced)
   - Runs the RAG agent with RAG disabled to demonstrate the zero-API-call path
   - Prints citation-ready evidence (page spans, sections, similarity scores)
   - NEVER logs, prints, or exposes the API key.

Note: This script is standalone and NOT executed automatically by pytest.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Extracted paper text contains Unicode math symbols (e.g. U+2217) that a legacy
# Windows cp1252 console cannot encode. Keep each stream's own encoding (to avoid
# mojibake in legacy consoles) but never let printing raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - platform dependent
        pass

import json
from dotenv import load_dotenv

# Load .env
load_dotenv()

from backend.rag_agent.agent import RAGAgent
from backend.jev_router.routing_state import create_routing_state

BENCHMARK_PDF = os.path.join("uploaded_files", "NIPS-2017-attention-is-all-you-need-Paper.pdf")
DEMO_QUERY = (
    "What empirical results support the claim that the Transformer "
    "outperforms recurrent and convolutional sequence models?"
)


def _safe(text: str) -> str:
    """Return text printable on the active console encoding, never raising."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except (LookupError, UnicodeError):  # pragma: no cover - platform dependent
        return text.encode("ascii", errors="replace").decode("ascii")


def _print_result(label, result):
    print(f"\n--- {label} ---")
    print(
        f"enabled={result['enabled']} level={result['level']} "
        f"top_k={result['top_k']} chunks_indexed={result['chunks_indexed']} "
        f"model={result['embedding_model']} elapsed={result['elapsed_time_ms']}ms"
    )
    print(f"message: {result['message']}")
    for i, item in enumerate(result["results"], start=1):
        section = item.get("section") or "unlabelled"
        pages = (
            f"p.{item['page_start']}"
            if item["page_start"] == item["page_end"]
            else f"pp.{item['page_start']}-{item['page_end']}"
        )
        snippet = " ".join(item["text"].split()[:35])
        print(f"  [{i}] {item['chunk_id']} | {pages} | {_safe(section)} | score={item['score']}")
        print(f"      {_safe(snippet)}...")


def run_rag_demo():
    print("=" * 60)
    print("RAG Agent Demonstration")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[INFO] No OPENAI_API_KEY detected in environment or .env.")
        print("[INFO] Embeddings require OPENAI_API_KEY; the RAG agent cannot index the paper.")
        print("[INFO] Set OPENAI_API_KEY in your .env file to run this demo.")
        print("[INFO] Exiting safely without failure.")
        return 0

    if not os.path.exists(BENCHMARK_PDF):
        print(f"[ERROR] Benchmark PDF not found at: {BENCHMARK_PDF}")
        return 1

    print(f"[STATUS] Benchmark paper: {BENCHMARK_PDF}")
    print(f"[STATUS] Query: {DEMO_QUERY}")

    # Step 1: Demonstrates the RAG-disabled path (zero embedding / API work)
    print("\n--- Step 1: JEV decision is RAG disabled (bypass path) ---")
    disabled_agent = RAGAgent()
    disabled_result = disabled_agent.run(
        pdf_path=BENCHMARK_PDF,
        routing_state=create_routing_state(
            analysis_level="advanced",
            rag_enabled=False,
            rag_level=None,
        ),
        query=DEMO_QUERY,
    )
    _print_result("RAG disabled", disabled_result)

    # Step 2: Runs retrieval for each JEV capability tier
    agent = RAGAgent()
    for level in ("basic", "medium", "advanced"):
        # Routing state built by the FROZEN jev_router contract layer (read-only here)
        routing_state = create_routing_state(
            analysis_level="advanced",
            rag_enabled=True,
            rag_level=level,
        )
        try:
            result = agent.run(
                pdf_path=BENCHMARK_PDF,
                routing_state=routing_state,
                query=DEMO_QUERY,
            )
        except Exception as exc:
            print(f"[ERROR] RAG retrieval failed for level '{level}': {exc}")
            return 1
        _print_result(f"RAG enabled - level '{level}'", result)

    # Step 3: Machine-readable payload for downstream agents
    print("\n--- Step 3: Final JSON payload (advanced tier) ---")
    print(json.dumps(result, indent=2)[:2000])

    print("\n" + "=" * 60)
    print("RAG Agent Demo: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_rag_demo())
