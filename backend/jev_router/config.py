"""
Configuration and constants for the JEV Router subsystem.

Defines:
- Supported model levels (basic, medium, advanced)
- Current router schema version
- JEV conceptual question definitions and decision types (Noul / Choice)
- TypeSafe API settings, environment variables, and configurable thresholds
"""

import os
from enum import Enum
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

ROUTER_VERSION: str = "1.0"

# TypeSafe API Endpoints & Defaults
TYPESAFE_API_BASE_URL: str = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
DEFAULT_JEV_MODEL: str = os.getenv("TYPESAFE_JEV_MODEL", "jev-latest")

# Configurable decision thresholds (defaults to 0.70; can be tuned/calibrated)
DEFAULT_RAG_ENABLE_THRESHOLD: float = float(os.getenv("JEV_RAG_ENABLE_THRESHOLD", "0.70"))
DEFAULT_VISION_ENABLE_THRESHOLD: float = float(os.getenv("JEV_VISION_ENABLE_THRESHOLD", "0.70"))

# Request timeout in seconds
TYPESAFE_REQUEST_TIMEOUT: float = float(os.getenv("TYPESAFE_REQUEST_TIMEOUT", "30.0"))


class ModelLevel(str, Enum):
    """
    Abstract model capability tiers.

    Only abstract tiers are permitted at the contract layer:
    - basic
    - medium
    - advanced

    Actual model mapping (e.g. Luna, Terra, Sol) is strictly deferred.
    """
    BASIC = "basic"
    MEDIUM = "medium"
    ADVANCED = "advanced"


ALLOWED_MODEL_LEVELS: List[str] = [level.value for level in ModelLevel]


class DecisionType(str, Enum):
    """JEV structured decision types."""
    NOUL = "Noul"       # Yes/No decision
    CHOICE = "Choice"   # Select one option from a defined set


# JEV Question Contracts (definitions only - no TypeSafe API calls)
JEV_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "rag_enabled": {
        "decision_type": DecisionType.NOUL.value,
        "conceptual_question": (
            "Is retrieval over the research paper's textual content necessary "
            "for producing a grounded research critique?"
        ),
        "target": "rag.enabled",
    },
    "vision_enabled": {
        "decision_type": DecisionType.NOUL.value,
        "conceptual_question": (
            "Does the research paper contain visual information that should be "
            "examined to produce a complete research critique?"
        ),
        "target": "vision.enabled",
    },
    "rag_level": {
        "decision_type": DecisionType.CHOICE.value,
        "conceptual_question": "Select the capability tier for the RAG agent.",
        "options": ALLOWED_MODEL_LEVELS,
        "target": "rag.level",
    },
    "vision_level": {
        "decision_type": DecisionType.CHOICE.value,
        "conceptual_question": "Select the capability tier for the Vision agent.",
        "options": ALLOWED_MODEL_LEVELS,
        "target": "vision.level",
    },
    "analysis_level": {
        "decision_type": DecisionType.CHOICE.value,
        "conceptual_question": "Select the capability tier for the Analysis agent.",
        "options": ALLOWED_MODEL_LEVELS,
        "target": "analysis.level",
    },
}

# Concrete questions payload definitions for TypeSafe POST /v1/systemone
# Following official TypeSafe API specification:
# Noul: {"type": "noul", "instructions": str}
# Choice: {"type": "choice", "instructions": str, "criteria": {opt: desc}}
SYSTEM_ONE_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "rag_required": {
        "type": "noul",
        "instructions": (
            "Is retrieval over the research paper's textual content necessary "
            "for producing a grounded research critique?"
        ),
    },
    "rag_level": {
        "type": "choice",
        "instructions": (
            "What model capability level is appropriate for the RAG agent "
            "for this research paper?"
        ),
        "criteria": {
            "basic": "Basic retrieval and summary over straightforward or short papers.",
            "medium": "Standard multi-section retrieval with moderate domain complexity.",
            "advanced": "Deep cross-referencing and technical precision over complex methodologies.",
        },
    },
    "vision_required": {
        "type": "noul",
        "instructions": (
            "Does the research paper contain visual information that should be "
            "examined to produce a complete research critique?"
        ),
    },
    "vision_level": {
        "type": "choice",
        "instructions": (
            "What model capability level is appropriate for the Vision agent "
            "for this research paper?"
        ),
        "criteria": {
            "basic": "Simple inspection of standard graphs or diagrams.",
            "medium": "Inspection of detailed technical figures, plots, or standard tables.",
            "advanced": "Complex architectural diagrams, dense formulas, or multi-panel figures.",
        },
    },
    "analysis_level": {
        "type": "choice",
        "instructions": (
            "What model capability level is appropriate for the core Analysis agent "
            "for this research paper?"
        ),
        "criteria": {
            "basic": "General critique of straightforward or incremental papers.",
            "medium": "Comprehensive critique evaluating methodology and empirical claims.",
            "advanced": "Rigorous critique of novel architectures, theoretical math, or major breakthroughs.",
        },
    },
}

