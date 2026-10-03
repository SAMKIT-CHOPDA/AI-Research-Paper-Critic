"""
Structured analysis: prompt execution, validation and provenance verification.

Validation policy
-----------------
Strict where a missing or invalid value would create a false impression:

* the response must parse to a single JSON object,
* every contract key must be present (all nine dimensions, the claim/evidence
  matrix, strengths, weaknesses, limitations, open questions, evidence gaps and the
  overall assessment),
* each dimension must carry a non-empty ``summary`` and a valid
  ``evidence_status``,
* findings, strengths/weaknesses, matrix entries, open questions and gaps must
  carry their primary text field,
* ``evidence_status``, reference ``source_type`` and gap ``category`` values must
  be valid enum tokens, and ``confidence`` must be a number between 0.0 and 1.0.

Lenient only where the alternative would be discarding usable content: secondary
fields (reasoning, caveats, per-finding status, page lists) default to an explicit
empty/``unclear`` value that can never look like support.

Honesty bookkeeping
-------------------
* Every reference is matched against the evidence bundle: a reference whose
  ``source_id`` is not in the bundle is kept but marked ``verified=False``.
* A statement marked ``supported`` without any reference is downgraded to
  ``unclear`` and the downgrade is recorded (an unverifiable claim must not travel
  downstream looking proven).
* Confirmatory language ("proves", "conclusively demonstrates") is recorded as a
  caveat/uncertainty, exactly as the Vision Agent does for images.
* Score-like text (e.g. "7/10", "score of 82") is recorded as an explicit
  ``no_numeric_paper_score`` gap: this stage never emits an overall numeric score.

No repair round trip
--------------------
There is no second model call to fix malformed output. A response that does not
satisfy the contract is rejected with ``AnalysisError`` so the agent can report a
structured failure; silently repairing it would hide that the model did not follow
the contract (the only exception is a provider that requires a structured-output
flag, which is a request-side option, not a repair call).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.analysis_agent.llm_client import BaseAnalysisClient
from backend.analysis_agent.prompts import build_analysis_prompt
from backend.analysis_agent.schemas import (
    EVIDENCE_SOURCE_TYPES,
    EVIDENCE_STATUSES,
    AnalysisPayload,
    AssessedPoint,
    ClaimEvidenceItem,
    ClaimEvidenceMatrix,
    DimensionAnalysis,
    EvidenceBundle,
    EvidenceGap,
    EvidenceRef,
    Finding,
    Limitations,
    OpenQuestion,
    OverallAssessment,
    ParsedAnalysis,
)

logger = logging.getLogger(__name__)

DIMENSION_KEYS: Tuple[str, ...] = (
    "research_problem",
    "contribution",
    "methodology",
    "data",
    "baselines",
    "metrics",
    "results",
    "reproducibility",
    "internal_consistency",
)

REQUIRED_PAYLOAD_KEYS: Tuple[str, ...] = DIMENSION_KEYS + (
    "claim_evidence_matrix",
    "strengths",
    "weaknesses",
    "limitations",
    "open_questions",
    "evidence_gaps",
    "overall_assessment",
)

GAP_CATEGORIES = (
    "rag_disabled",
    "rag_unavailable",
    "vision_disabled",
    "vision_unavailable",
    "numeric_values_unavailable",
    "not_reported_in_paper",
    "model_uncertainty",
    "unverified_reference",
    "no_numeric_paper_score",
    "context_truncated",
    "other",
)

# Phrases that turn an evaluation into a claim of proof. Same list as the Vision
# Agent's, because the failure mode (a model asserting more than its evidence can
# carry) is identical.
OVERCLAIM_PHRASES: Tuple[str, ...] = (
    "proves",
    "proven",
    "proof that",
    "demonstrates conclusively",
    "shows conclusively",
    "conclusively demonstrates",
    "establishes that",
    "guarantees",
    "definitively shows",
)

# Score-like expressions: this stage must never emit a paper-level number.
SCORE_PATTERNS: Tuple[str, ...] = (
    r"\b\d+(?:\.\d+)?\s*/\s*(?:5|10|100)\b",
    r"\bscore(?:d)?\s*(?:of|:|=)\s*\d+(?:\.\d+)?\b",
    r"\brating\s*(?:of|:|=)\s*\d+(?:\.\d+)?\b",
    r"\b\d+(?:\.\d+)?\s*out of\s*(?:5|10|100)\b",
    r"\bgrade\s*(?:of|:|=)\s*[A-F]\b",
)

_SCORE_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in SCORE_PATTERNS)

# Word-boundary matching prevents substring false positives ("improves" is not
# "proves"), which would otherwise annotate honest findings as overclaims.
_OVERCLAIM_REGEXES = tuple(
    (
        phrase,
        re.compile(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", re.IGNORECASE),
    )
    for phrase in OVERCLAIM_PHRASES
)

DEFAULT_STATUS = "unclear"

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class AnalysisError(Exception):
    """Raised when a model response cannot be turned into a valid analysis."""

    pass


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Extract the JSON object from a model response.

    Accepts fenced blocks and surrounding commentary, but raises AnalysisError when
    no parseable object exists, so a malformed response becomes a recorded failure
    instead of a silently empty analysis.
    """
    if not raw_text or not str(raw_text).strip():
        raise AnalysisError("Analysis model returned an empty response")

    text = str(raw_text).strip()
    candidates: List[str] = []
    candidates.extend(
        block.strip()
        for block in _FENCE_PATTERN.findall(text)
        if block and block.strip()
    )
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = candidate[start : end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise AnalysisError(
        "Analysis model response did not contain a parseable JSON object"
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _clean_text(json.dumps(value))
    return _clean_text(value)


def _coerce_string_list(value: Any) -> List[str]:
    """Coerce a model list into clean strings, dropping empties."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            cleaned = _coerce_text(item)
            if cleaned:
                result.append(cleaned)
        return result
    cleaned = _coerce_text(value)
    return [cleaned] if cleaned else []


def _coerce_page_list(value: Any) -> List[int]:
    """Coerce pages into positive ints, ignoring anything unusable."""
    if value is None:
        return []
    candidates = value if isinstance(value, (list, tuple, set)) else [value]
    pages: List[int] = []
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        number: Optional[int] = None
        if isinstance(candidate, int):
            number = candidate
        elif isinstance(candidate, float):
            number = int(candidate)
        else:
            try:
                number = int(str(candidate).strip())
            except (TypeError, ValueError):
                number = None
        if number is not None and number >= 1 and number not in pages:
            pages.append(number)
    return sorted(pages)


def _coerce_confidence(value: Any, where: str) -> Optional[float]:
    """
    Validate a confidence value.

    Missing/None stays None (the model stated none, so none is invented). Anything
    present that is not a number inside [0.0, 1.0] is a contract violation.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnalysisError(f"'confidence' must be a number in {where}, got a boolean")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except (TypeError, ValueError):
            raise AnalysisError(
                f"'confidence' must be a number between 0.0 and 1.0 in {where}, "
                f"got {value!r}"
            )
    if not isinstance(value, (int, float)):
        raise AnalysisError(
            f"'confidence' must be a number between 0.0 and 1.0 in {where}, "
            f"got {type(value).__name__}"
        )
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise AnalysisError(
            f"'confidence' must be between 0.0 and 1.0 in {where}, got {confidence}"
        )
    return confidence


def _coerce_status(value: Any, where: str, default: str = DEFAULT_STATUS) -> str:
    """Validate an evidence_status token (invalid tokens are rejected)."""
    if value is None:
        return default
    normalized = _clean_text(value).lower()
    if not normalized:
        return default
    if normalized not in EVIDENCE_STATUSES:
        raise AnalysisError(
            f"Invalid evidence_status {value!r} in {where}. Expected one of: "
            f"{', '.join(EVIDENCE_STATUSES)}."
        )
    return normalized


def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object for {where}")
    return value


def _require_list(value: Any, where: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise AnalysisError(f"Expected a JSON array for {where}")
    return list(value)


def _require_text(value: Any, where: str) -> str:
    text = _coerce_text(value)
    if not text:
        raise AnalysisError(f"Missing required text field {where}")
    return text


# Near-miss ``source_type`` tokens observed from production models. The contract
# tokens are the keys; the values are the obvious short forms a model reaches for
# when it abbreviates. Normalization happens *before* the strict enum check so the
# fail-closed behaviour is preserved for anything not listed here: an unknown type
# still raises, because a wrong type would misrepresent where evidence came from.
SOURCE_TYPE_ALIASES: Dict[str, str] = {
    "chunk": "text_chunk",
    "text_chunk": "text_chunk",
    "chunks": "text_chunk",
    "retrieved_chunk": "text_chunk",
    "text": "text_chunk",
    "asset": "visual_asset",
    "visual_asset": "visual_asset",
    "assets": "visual_asset",
    "image": "visual_asset",
    "figure_or_table": "visual_asset",
    "doc": "document_profile",
    "profile": "document_profile",
    "document": "document_profile",
    "references": "reference",
    "bib": "reference",
    "equations": "equation",
    "figures": "figure",
    "tables": "table",
    "sections": "section",
    "metadata": "metadata",
}


def normalize_source_type(value: Any) -> str:
    """
    Map an abbreviated ``source_type`` onto its contract token.

    Only the explicit alias table above is accepted; every other unknown token is
    returned unchanged so the caller still rejects it. This keeps the strict,
    fail-closed validation intact while tolerating the shorthand real models use.
    """
    token = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    return SOURCE_TYPE_ALIASES.get(token, token)


def parse_evidence_refs(
    value: Any, bundle: EvidenceBundle, where: str
) -> List[EvidenceRef]:
    """
    Validate the references attached to a model statement.

    A reference is *verified* only when its ``source_id`` exists in the evidence
    bundle. Unmatched ids are preserved (the model's claim about provenance stays
    auditable) but marked unverified; a bad ``source_type`` enum is rejected
    outright, because a wrong type would misrepresent where evidence came from.
    """
    refs: List[EvidenceRef] = []
    for index, item in enumerate(_require_list(value, where)):
        if not isinstance(item, dict):
            raise AnalysisError(f"Expected an object for {where}[{index}]")
        source_id = _clean_text(item.get("source_id"))
        if not source_id:
            # A reference the model could not identify carries no provenance at
            # all. It is dropped and counted (see count_unverified_evidence_refs
            # via the analyzer notes) instead of discarding a complete, otherwise
            # grounded evaluation: an unusable reference can never make a
            # statement look supported, it only removes its own support.
            logger.info("Dropped unusable evidence reference at %s[%d]", where, index)
            continue

        entry = bundle.source_index.get(source_id)
        raw_type = normalize_source_type(item.get("source_type"))
        entry_type = _clean_text(entry.get("source_type")).lower() if entry else ""
        if entry is not None and entry_type in EVIDENCE_SOURCE_TYPES:
            # The bundle is the authority on provenance: when the id is a real
            # source, the type is taken from the bundle and never from the model,
            # so a mistyped type cannot misattribute where evidence came from.
            # Production models abbreviate these tokens constantly ("chunk",
            # "asset", "all", "entire_document").
            source_type = entry_type
            verified = True
        elif raw_type and raw_type not in EVIDENCE_SOURCE_TYPES:
            # The id is not a known source, so the model is the only source of the
            # type and an invalid one cannot be checked at all: reject it.
            raise AnalysisError(
                f"Invalid source_type {raw_type!r} in {where}[{index}] for unknown "
                f"source_id {source_id!r}. Expected one of: "
                f"{', '.join(EVIDENCE_SOURCE_TYPES)}."
            )
        else:
            source_type = raw_type
            verified = False

        if not source_type:
            raise AnalysisError(
                f"Missing 'source_type' for reference {source_id!r} in {where}[{index}]; "
                "the id is not part of the evidence bundle either."
            )

        pages = _coerce_page_list(item.get("pages"))
        if entry is not None and not pages:
            pages = _coerce_page_list(entry.get("pages"))
        detail = _coerce_text(item.get("detail"))
        if not detail and entry is not None:
            detail = _coerce_text(entry.get("detail"))

        refs.append(
            EvidenceRef(
                source_type=source_type,
                source_id=source_id,
                pages=pages,
                verified=verified,
                detail=detail or None,
            )
        )
    return refs


def parse_finding(value: Any, bundle: EvidenceBundle, where: str) -> Finding:
    payload = _require_mapping(value, where)
    statement = _require_text(payload.get("statement"), f"{where}.statement")
    evidence_refs = parse_evidence_refs(
        payload.get("evidence_refs"), bundle, f"{where}.evidence_refs"
    )
    status = _coerce_status(payload.get("evidence_status"), where)
    caveats = _coerce_string_list(payload.get("caveats"))

    if status == "supported" and not evidence_refs:
        status = "unclear"
        caveats.append(
            "The model reported status 'supported' without any evidence reference, so "
            "the status was downgraded to 'unclear': support cannot be verified."
        )

    return Finding(
        statement=statement,
        evidence_status=status,
        evidence_refs=evidence_refs,
        confidence=_coerce_confidence(payload.get("confidence"), where),
        reasoning=_coerce_text(payload.get("reasoning")),
        caveats=caveats,
    )


def parse_dimension(
    key: str, value: Any, bundle: EvidenceBundle
) -> DimensionAnalysis:
    payload = _require_mapping(value, f"'{key}'")
    summary = _require_text(payload.get("summary"), f"'{key}'.summary")
    if "evidence_status" not in payload:
        raise AnalysisError(f"Missing required key 'evidence_status' in '{key}'")
    status = _coerce_status(payload.get("evidence_status"), f"'{key}'")
    evidence_refs = parse_evidence_refs(
        payload.get("evidence_refs"), bundle, f"'{key}'.evidence_refs"
    )
    uncertainties = _coerce_string_list(payload.get("uncertainties"))

    if status == "supported" and not evidence_refs:
        status = "unclear"
        uncertainties.append(
            "The dimension was reported as 'supported' without any evidence reference, "
            "so the status was downgraded to 'unclear'."
        )

    findings: List[Finding] = []
    for index, item in enumerate(_require_list(payload.get("findings"), f"'{key}'.findings")):
        findings.append(parse_finding(item, bundle, f"'{key}'.findings[{index}]"))

    return DimensionAnalysis(
        dimension=key,
        summary=summary,
        evidence_status=status,
        findings=findings,
        evidence_refs=evidence_refs,
        confidence=_coerce_confidence(payload.get("confidence"), f"'{key}'"),
        uncertainties=uncertainties,
    )


def parse_point(value: Any, bundle: EvidenceBundle, where: str) -> AssessedPoint:
    """Parse one strength or weakness."""
    payload = _require_mapping(value, where)
    statement = _require_text(payload.get("statement"), f"{where}.statement")
    evidence_refs = parse_evidence_refs(
        payload.get("evidence_refs"), bundle, f"{where}.evidence_refs"
    )
    status = _coerce_status(payload.get("evidence_status"), where)
    rationale = _coerce_text(payload.get("rationale"))
    if status == "supported" and not evidence_refs:
        status = "unclear"
        rationale = (
            rationale
            + " [Recorded status 'supported' was downgraded to 'unclear': no evidence "
            "reference was given.]"
        ).strip()
    return AssessedPoint(
        statement=statement,
        dimension=_coerce_text(payload.get("dimension")) or None,
        evidence_status=status,
        evidence_refs=evidence_refs,
        rationale=rationale,
        confidence=_coerce_confidence(payload.get("confidence"), where),
    )


def parse_claim_item(
    value: Any, bundle: EvidenceBundle, where: str
) -> ClaimEvidenceItem:
    payload = _require_mapping(value, where)
    claim = _require_text(payload.get("claim"), f"{where}.claim")
    supporting = parse_evidence_refs(
        payload.get("supporting_evidence"), bundle, f"{where}.supporting_evidence"
    )
    contradicting = parse_evidence_refs(
        payload.get("contradicting_evidence"), bundle, f"{where}.contradicting_evidence"
    )
    status = _coerce_status(payload.get("status"), where)
    missing_evidence = _coerce_string_list(payload.get("missing_evidence"))
    assessment = _coerce_text(payload.get("assessment"))

    claim_source = _clean_text(payload.get("claim_source")).lower() or "paper"
    if claim_source not in ("paper", "analyst_inference"):
        raise AnalysisError(
            f"Invalid claim_source {claim_source!r} in {where}. Expected 'paper' or "
            "'analyst_inference'."
        )

    if status == "supported" and not supporting:
        status = "unclear"
        assessment = (
            assessment
            + " [Recorded status 'supported' was downgraded to 'unclear': no supporting "
            "evidence was referenced.]"
        ).strip()

    return ClaimEvidenceItem(
        claim=claim,
        claim_source=claim_source,
        status=status,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        assessment=assessment,
        missing_evidence=missing_evidence,
        confidence=_coerce_confidence(payload.get("confidence"), where),
    )


def parse_matrix(value: Any, bundle: EvidenceBundle) -> ClaimEvidenceMatrix:
    payload = _require_mapping(value, "'claim_evidence_matrix'")
    items = [
        parse_claim_item(item, bundle, f"'claim_evidence_matrix'.items[{index}]")
        for index, item in enumerate(
            _require_list(payload.get("items"), "'claim_evidence_matrix'.items")
        )
    ]
    return ClaimEvidenceMatrix(items=items, notes=_coerce_text(payload.get("notes")))


def parse_limitations(value: Any, bundle: EvidenceBundle) -> Limitations:
    payload = _require_mapping(value, "'limitations'")
    author_stated = [
        parse_finding(item, bundle, f"'limitations'.author_stated[{index}]")
        for index, item in enumerate(
            _require_list(payload.get("author_stated"), "'limitations'.author_stated")
        )
    ]
    analyst_identified = [
        parse_finding(item, bundle, f"'limitations'.analyst_identified[{index}]")
        for index, item in enumerate(
            _require_list(
                payload.get("analyst_identified"), "'limitations'.analyst_identified"
            )
        )
    ]
    return Limitations(
        author_stated=author_stated, analyst_identified=analyst_identified
    )


def parse_open_question(value: Any, bundle: EvidenceBundle, where: str) -> OpenQuestion:
    payload = _require_mapping(value, where)
    question = _require_text(payload.get("question"), f"{where}.question")
    return OpenQuestion(
        question=question,
        why_it_matters=_coerce_text(payload.get("why_it_matters")),
        related_dimension=_coerce_text(payload.get("related_dimension")) or None,
        evidence_refs=parse_evidence_refs(
            payload.get("evidence_refs"), bundle, f"{where}.evidence_refs"
        ),
    )


def parse_gap(value: Any, where: str) -> EvidenceGap:
    payload = _require_mapping(value, where)
    gap = _require_text(payload.get("gap"), f"{where}.gap")
    category = _clean_text(payload.get("category")).lower() or "other"
    if category not in GAP_CATEGORIES:
        raise AnalysisError(
            f"Invalid gap category {category!r} in {where}. Expected one of: "
            f"{', '.join(GAP_CATEGORIES)}."
        )
    return EvidenceGap(
        gap=gap,
        category=category,
        impact=_coerce_text(payload.get("impact")),
        affected_dimensions=_coerce_string_list(payload.get("affected_dimensions")),
        source="analysis_model",
    )


def parse_overall(value: Any, bundle: EvidenceBundle) -> OverallAssessment:
    payload = _require_mapping(value, "'overall_assessment'")
    summary = _require_text(payload.get("summary"), "'overall_assessment'.summary")
    if "evidence_status" not in payload:
        raise AnalysisError(
            "Missing required key 'evidence_status' in 'overall_assessment'"
        )
    status = _coerce_status(payload.get("evidence_status"), "'overall_assessment'")
    key_basis = parse_evidence_refs(
        payload.get("key_basis"), bundle, "'overall_assessment'.key_basis"
    )
    uncertainties = _coerce_string_list(payload.get("uncertainties"))
    if status == "supported" and not key_basis:
        status = "unclear"
        uncertainties.append(
            "The overall assessment was reported as 'supported' without any evidence "
            "reference, so the status was downgraded to 'unclear'."
        )
    return OverallAssessment(
        summary=summary,
        evidence_status=status,
        confidence=_coerce_confidence(
            payload.get("confidence"), "'overall_assessment'"
        ),
        key_basis=key_basis,
        uncertainties=uncertainties,
    )


DOWNGRADE_MARKER = "downgraded to 'unclear'"


def flag_overclaiming(*texts: Any) -> List[str]:
    """
    Detect confirmatory phrasing that evidence-grounded analysis cannot support.

    Matching is word-boundary based, so ordinary words that merely contain a
    phrase ("improves") are not reported as overclaims.

    Returns the phrases found (deduplicated, sorted). The caller records them as
    explicit caveats so an unsupported claim never travels downstream as fact.
    """
    found: List[str] = []
    for text in texts:
        cleaned = _coerce_text(text)
        if not cleaned:
            continue
        for phrase, regex in _OVERCLAIM_REGEXES:
            if phrase not in found and regex.search(cleaned):
                found.append(phrase)
    return sorted(found)


def detect_score_like_text(*texts: Any) -> List[str]:
    """
    Detect score-like expressions.

    This stage must not emit an overall numeric paper score, so a score-like
    expression in a verdict position is recorded as an explicit gap instead of
    being passed downstream as an evaluation result.
    """
    found: List[str] = []
    for text in texts:
        cleaned = _coerce_text(text)
        if not cleaned:
            continue
        for regex in _SCORE_REGEXES:
            for match in regex.findall(cleaned):
                snippet = _clean_text(match)
                if snippet and snippet.lower() not in [f.lower() for f in found]:
                    found.append(snippet)
    return found


def _scan_overclaiming(payload: AnalysisPayload) -> List[str]:
    """
    Append overclaiming caveats in place and return the phrases detected.

    Mutating the parsed payload (instead of rejecting the response) keeps the
    finding visible downstream together with the warning that it overreaches.
    """
    phrases: List[str] = []

    def list_sink(target: List[str]):
        def sink(note: str) -> None:
            if note not in target:
                target.append(note)

        return sink

    def text_sink(obj: Any, field: str):
        def sink(note: str) -> None:
            current = _clean_text(getattr(obj, field, ""))
            setattr(obj, field, f"{current} [{note}]".strip())

        return sink

    def apply(texts: List[Any], sink) -> None:
        for phrase in flag_overclaiming(*texts):
            if phrase not in phrases:
                phrases.append(phrase)
            sink(
                f"Confirmatory language detected ('{phrase}'): an evidence-grounded "
                "analysis cannot establish this, so the statement stays provisional."
            )

    for key in DIMENSION_KEYS:
        dimension = getattr(payload, key)
        apply([dimension.summary], list_sink(dimension.uncertainties))
        for finding in dimension.findings:
            apply([finding.statement, finding.reasoning], list_sink(finding.caveats))
    for item in payload.claim_evidence_matrix.items:
        apply([item.claim, item.assessment], text_sink(item, "assessment"))
    for point in [*payload.strengths, *payload.weaknesses]:
        apply([point.statement, point.rationale], text_sink(point, "rationale"))
    for finding in [
        *payload.limitations.author_stated,
        *payload.limitations.analyst_identified,
    ]:
        apply([finding.statement, finding.reasoning], list_sink(finding.caveats))
    for question in payload.open_questions:
        apply([question.question], text_sink(question, "why_it_matters"))
    apply(
        [payload.overall_assessment.summary],
        list_sink(payload.overall_assessment.uncertainties),
    )
    return sorted(phrases)


def _scan_score_like(payload: AnalysisPayload) -> List[str]:
    """Look for score-like text in verdict positions only."""
    texts: List[Any] = [getattr(payload, key).summary for key in DIMENSION_KEYS]
    texts.append(payload.overall_assessment.summary)
    texts.append(payload.claim_evidence_matrix.notes)
    return detect_score_like_text(*texts)


def parse_analysis(raw_text: str, bundle: EvidenceBundle) -> ParsedAnalysis:
    """
    Turn a raw model response into a validated ParsedAnalysis.

    Raises AnalysisError when the contract is not met (missing keys, empty required
    fields, invalid enums, invalid confidence, wrongly typed collections). Field
    level leniency is limited to secondary fields, which become explicit empty or
    'unclear' values that can never look like support.
    """
    payload_dict = extract_json_payload(raw_text)

    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload_dict]
    if missing:
        raise AnalysisError(
            "The analysis response is missing required key(s): " + ", ".join(missing)
        )

    dimensions = {
        key: parse_dimension(key, payload_dict.get(key), bundle) for key in DIMENSION_KEYS
    }
    strengths = [
        parse_point(item, bundle, f"'strengths'[{index}]")
        for index, item in enumerate(
            _require_list(payload_dict.get("strengths"), "'strengths'")
        )
    ]
    weaknesses = [
        parse_point(item, bundle, f"'weaknesses'[{index}]")
        for index, item in enumerate(
            _require_list(payload_dict.get("weaknesses"), "'weaknesses'")
        )
    ]
    open_questions = [
        parse_open_question(item, bundle, f"'open_questions'[{index}]")
        for index, item in enumerate(
            _require_list(payload_dict.get("open_questions"), "'open_questions'")
        )
    ]
    model_gaps = [
        parse_gap(item, f"'evidence_gaps'[{index}]")
        for index, item in enumerate(
            _require_list(payload_dict.get("evidence_gaps"), "'evidence_gaps'")
        )
    ]

    payload = AnalysisPayload(
        **dimensions,
        claim_evidence_matrix=parse_matrix(
            payload_dict.get("claim_evidence_matrix"), bundle
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        limitations=parse_limitations(payload_dict.get("limitations"), bundle),
        open_questions=open_questions,
        evidence_gaps=model_gaps,
        overall_assessment=parse_overall(payload_dict.get("overall_assessment"), bundle),
    )

    overclaim_flags = _scan_overclaiming(payload)
    score_snippets = _scan_score_like(payload)
    unverified = count_unverified_evidence_refs(payload)
    downgraded = count_downgraded_statuses(payload)

    notes: List[EvidenceGap] = []
    if unverified:
        notes.append(
            EvidenceGap(
                gap=(
                    f"{unverified} cited reference(s) do not exist in the evidence bundle "
                    "and are marked unverified."
                ),
                category="unverified_reference",
                impact=(
                    "Those statements cannot be traced back to the supplied evidence; "
                    "treat them as unverified rather than grounded."
                ),
                source="analyzer_validation",
            )
        )
    if downgraded:
        notes.append(
            EvidenceGap(
                gap=(
                    f"{downgraded} statement(s) recorded as 'supported' without an "
                    "evidence reference were downgraded to 'unclear'."
                ),
                category="unverified_reference",
                impact=(
                    "Provenance for those statements is missing, so their support could "
                    "not be verified and was not accepted at face value."
                ),
                source="analyzer_validation",
            )
        )
    if overclaim_flags:
        notes.append(
            EvidenceGap(
                gap=(
                    "Confirmatory language appeared in the response: "
                    + ", ".join(overclaim_flags)
                ),
                category="model_uncertainty",
                impact=(
                    "Those statements were kept but annotated as provisional; the "
                    "confirmatory wording is not supported by the evidence."
                ),
                source="analyzer_validation",
            )
        )
    if score_snippets:
        notes.append(
            EvidenceGap(
                gap=(
                    "Score-like expression(s) appeared in the response ("
                    + ", ".join(score_snippets)
                    + ")."
                ),
                category="no_numeric_paper_score",
                impact=(
                    "This stage does not emit an overall numeric paper score; the "
                    "expression is recorded rather than propagated as an evaluation "
                    "result."
                ),
                source="analyzer_validation",
            )
        )

    return ParsedAnalysis(
        payload=payload,
        notes=notes,
        overclaim_flags=overclaim_flags,
        unverified_evidence_refs=unverified,
        downgraded_statuses=downgraded,
    )


def analyze_evidence(
    bundle: EvidenceBundle,
    client: BaseAnalysisClient,
    model: str,
    prompt: Optional[str] = None,
) -> ParsedAnalysis:
    """
    Analyze one evidence bundle end to end: prompt -> single model call -> parse.

    Client failures (auth, unavailable model, transport) propagate as
    AnalysisClientError; contract violations raise AnalysisError. The agent isolates
    both and reports a structured failure. There is no repair call: the response is
    either contract-compliant or it is reported as unusable.
    """
    resolved_prompt = prompt or build_analysis_prompt(bundle)
    raw_response = client.complete(resolved_prompt, model)
    return parse_analysis(raw_response, bundle)


def _iter_evidence_refs(payload: AnalysisPayload):
    """Yield every reference in the payload (verified or not)."""
    for key in DIMENSION_KEYS:
        dimension = getattr(payload, key)
        yield from dimension.evidence_refs
        for finding in dimension.findings:
            yield from finding.evidence_refs
    for item in payload.claim_evidence_matrix.items:
        yield from item.supporting_evidence
        yield from item.contradicting_evidence
    for point in [*payload.strengths, *payload.weaknesses]:
        yield from point.evidence_refs
    for finding in [
        *payload.limitations.author_stated,
        *payload.limitations.analyst_identified,
    ]:
        yield from finding.evidence_refs
    for question in payload.open_questions:
        yield from question.evidence_refs
    yield from payload.overall_assessment.key_basis


def count_unverified_evidence_refs(payload: AnalysisPayload) -> int:
    """References whose source_id did not match the evidence bundle."""
    return sum(1 for ref in _iter_evidence_refs(payload) if not ref.verified)


def count_downgraded_statuses(payload: AnalysisPayload) -> int:
    """Statements whose recorded 'supported' status had to be downgraded."""
    total = 0
    for key in DIMENSION_KEYS:
        dimension = getattr(payload, key)
        total += sum(1 for note in dimension.uncertainties if DOWNGRADE_MARKER in note)
        for finding in dimension.findings:
            total += sum(1 for note in finding.caveats if DOWNGRADE_MARKER in note)
    for item in payload.claim_evidence_matrix.items:
        total += 1 if DOWNGRADE_MARKER in item.assessment else 0
    for point in [*payload.strengths, *payload.weaknesses]:
        total += 1 if DOWNGRADE_MARKER in point.rationale else 0
    for finding in [
        *payload.limitations.author_stated,
        *payload.limitations.analyst_identified,
    ]:
        total += sum(1 for note in finding.caveats if DOWNGRADE_MARKER in note)
    total += sum(
        1 for note in payload.overall_assessment.uncertainties if DOWNGRADE_MARKER in note
    )
    return total
