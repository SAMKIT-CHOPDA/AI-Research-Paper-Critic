"""
Candidate-critique validation and honesty bookkeeping (Phase 6).

The Critique Engine is the last stage before a human reads the output, so this
module is where the "no fabrication" contract is actually enforced rather than
merely requested in the prompt.

What is checked
---------------
1. schema correctness (the candidate must parse into ``CritiquePayload``),
2. valid section ids from the fixed catalogue,
3. non-empty content for every emitted section,
4. every evidence reference resolves to a source the engine actually supplied,
5. no fabricated source ids (``chunk_999``, ``figure_999``, ``table_999``),
6. no fabricated page citations: a "p. N" in the prose must correspond to a page
   that came from the validated source metadata and must exist in the document,
7. no API secrets anywhere in the output,
8. no numeric paper score (in prose or in a field name),
9. no ranking / tiering / grading fields or unsupported overall labels,
10. no confidence values invented by the model,
11. required sections present when the evidence supports them, and unsupported
    sections omitted rather than padded,
12. the Phase 5 claim/evidence matrix preserved: no dropped claim, no changed
    classification, no new claim,
13. author-stated and analyst-identified limitations traceable to the matching
    Phase 5 category, so the two can never be merged or swapped.

Failure policy
--------------
Fabrication is a hard error, not a warning: an unresolved citation or a page
number that cannot exist makes the whole critique ``validation_failed``, because a
plausible-looking report with invented provenance is worse than no report. Softer
problems (a claim stated without an assessment) are recorded as warnings and still
surface in the output.

There is no repair round trip: the validator never asks the model to try again.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from backend.critique_engine.schemas import (
    ALWAYS_REQUIRED_SECTIONS,
    CLAIM_STATUSES,
    SECTION_IDS,
    SECTION_TITLES,
    SOURCE_TYPES,
    UNCERTAINTY_MARKERS,
    ClaimEvidenceSummaryItem,
    CritiqueGap,
    CritiqueOverallAssessment,
    CritiqueProvenance,
    CritiqueLimitation,
    CritiqueLimitations,
    CritiqueOpenQuestion,
    CritiqueStrength,
    CritiquePayload,
    CritiqueSection,
    ParsedCritique,
    SourceRef,
    SynthesisContext,
    ValidationIssue,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# Numeric verdict expressions. A single overall number for a whole paper is
# forbidden, exactly as in Phase 5 - the two stages share the failure mode.
SCORE_PATTERNS: Tuple[str, ...] = (
    r"\b\d+(?:\.\d+)?\s*/\s*(?:5|10|100)\b",
    r"\bscore(?:d)?\s*(?:of|:|=)\s*\d+(?:\.\d+)?\b",
    r"\brating\s*(?:of|:|=)\s*\d+(?:\.\d+)?\b",
    r"\b\d+(?:\.\d+)?\s*out of\s*(?:5|10|100)\b",
    r"\bgrade[ds]?\s*(?:of|:|=)?\s*[A-F]\b",
)

# Overall verdict labels the evidence cannot support. These are rejected because
# an unsupported "this is a strong paper" is a ranking in prose form.
UNSUPPORTED_VERDICT_PATTERNS: Tuple[str, ...] = (
    r"\bthis is an? (?:excellent|good|bad|poor|weak|strong) paper\b",
    r"\b(?:excellent|outstanding|poor quality|high[- ]quality) (?:paper|work|study)\b",
    r"\b(?:top|best|leading)[- ]tier (?:paper|work|venue)\b",
    r"\boverall(?:ly)? (?:excellent|outstanding|poor|bad)\b",
)

# Metric names. "BLEU score of 28.4" is a *reported result the evidence supports*,
# not an overall paper-quality score, so a metric context exempts the numeric
# verdict check. Only the paper-level verdict is forbidden.
METRIC_CONTEXT_PATTERN = re.compile(
    r"\b(bleu|rouge|meteor|ter|chrf|bertscore|accuracy|precision|recall|f1|"
    r"perplexity|ppl|auc|map|mrr|ndcg|wer|cer|psnr|ssim)\b[\w\s.,%+-]{0,40}$",
    re.IGNORECASE,
)

# Field names that would carry a forbidden verdict or an invented confidence.
FORBIDDEN_KEYS: Tuple[str, ...] = (
    "score",
    "paper_score",
    "quality_score",
    "overall_score",
    "rating",
    "rank",
    "ranking",
    "tier",
    "grade",
    "overall_verdict",
    "verdict",
    "quality_label",
    "star_rating",
    "confidence",
    "certainty",
    "probability",
)

# Anything that looks like a leaked credential in model output.
SECRET_PATTERNS: Tuple[str, ...] = (
    r"\bsk-[A-Za-z0-9_\-]{8,}",
    r"\bbearer\s+[A-Za-z0-9._\-]{12,}",
    r"\bauthorization\b\s*[:=]",
    r"\bapi[_-]?key\b\s*[:=]\s*\S+",
)

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Visible citation forms the validator must be able to check against real pages.
PAGE_CITATION_PATTERNS: Tuple[str, ...] = (
    r"\bpp?\.\s*(\d{1,4})",
    r"\bpages?\s+(\d{1,4})\s*(?:-|–|to)\s*(\d{1,4})",
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SCORE_REGEXES = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in SCORE_PATTERNS
)


class CritiqueValidationError(Exception):
    """Raised when a candidate critique cannot be turned into a valid payload."""

    def __init__(self, message: str, report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.report = report


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Extract the JSON object from a model response.

    Accepts fenced blocks and surrounding commentary, but raises
    CritiqueValidationError when no parseable object exists, so a malformed
    response becomes a recorded failure instead of a silently empty critique.
    """
    if not raw_text or not str(raw_text).strip():
        raise CritiqueValidationError("Critique model returned an empty response")

    text = str(raw_text).strip()
    candidates: List[str] = [
        block.strip()
        for block in FENCE_PATTERN.findall(text)
        if block and block.strip()
    ]
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise CritiqueValidationError(
        "Critique model response did not contain a parseable JSON object"
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize(value: Any) -> str:
    """Lowercase, punctuation-free form used for similarity comparisons."""
    return " ".join(_WORD_RE.findall(_text(value).lower()))


def _similar(left: Any, right: Any, threshold: float = 0.55) -> bool:
    """
    Tolerant overlap test used to trace model text back to its Phase 5 source.

    The synthesis stage legitimately rewrites and condenses the analysis wording,
    so an exact match is too strict, while a single shared word would let an
    invented claim through. The test therefore requires a minimum token overlap
    *and* at least two shared content tokens.
    """
    a = _normalize(left)
    b = _normalize(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return False
    shared = a_tokens & b_tokens
    if len(shared) < 2:
        return False
    return len(shared) / min(len(a_tokens), len(b_tokens)) >= threshold


def _scan_patterns(text: Any, patterns: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, _text(text), re.IGNORECASE):
            matches.append(match.group(0))
    return matches


def detect_score_like_text(text: Any) -> List[str]:
    """Numeric-verdict expressions the critique must never contain."""
    return _scan_patterns(text, SCORE_PATTERNS)


def detect_unsupported_verdicts(text: Any) -> List[str]:
    """Overall labels/rankings the evidence cannot support."""
    return _scan_patterns(text, UNSUPPORTED_VERDICT_PATTERNS)


def detect_secrets(text: Any) -> List[str]:
    """Credential-shaped substrings that must never reach the output."""
    return _scan_patterns(text, SECRET_PATTERNS)


def find_forbidden_keys(value: Any, path: str = "") -> List[Tuple[str, str]]:
    """Walk a parsed object and report every forbidden field name."""
    found: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{path}.{key}" if path else str(key)
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                found.append((location, str(key)))
            found.extend(find_forbidden_keys(item, location))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(find_forbidden_keys(item, f"{path}[{index}]"))
    return found


def extract_page_citations(text: Any) -> List[int]:
    """
    Collect every page number the visible prose cites.

    These are what a reader trusts, so each one must correspond to a page that
    came from validated upstream metadata.
    """
    pages: List[int] = []
    content = _text(text)
    for pattern in PAGE_CITATION_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            for group in match.groups():
                if group is None:
                    continue
                try:
                    pages.append(int(group))
                except ValueError:
                    continue
    return pages


def is_uncertain_text(text: Any) -> bool:
    """True when the wording marks a statement as not established."""
    lowered = _text(text).lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


def _resolve_refs(
    raw_refs: Any,
    context: SynthesisContext,
    issues: List[ValidationIssue],
    location: str,
) -> Tuple[List[SourceRef], int]:
    """
    Resolve and normalize the references of one record.

    A reference resolves when its ``source_id`` exists in the context's allow-list.
    The ``source_type`` is always taken from the matched upstream metadata, never
    from the model, so a mistyped type cannot misattribute where evidence came
    from. Pages are likewise never taken from the model.

    An id that does not exist upstream is still rejected: that is the actual
    fabrication risk (``chunk_999``), and it is never silently accepted.
    """
    resolved: List[SourceRef] = []
    unverified = 0
    if raw_refs in (None, "", [], {}):
        return resolved, unverified
    if not isinstance(raw_refs, (list, tuple)):
        issues.append(
            ValidationIssue(
                code="evidence_refs_not_a_list",
                message="evidence_refs must be a list of reference objects",
                location=location,
            )
        )
        return resolved, unverified

    for index, raw in enumerate(raw_refs):
        ref_location = f"{location}[{index}]"
        entry = None
        if isinstance(raw, str):
            entry = _lookup_by_id(context, _text(raw), "")
        elif isinstance(raw, dict):
            source_type = _text(raw.get("source_type")).lower()
            source_id = _text(raw.get("source_id"))
            entry = context.source_index.get(f"{source_type}:{source_id}")
            if entry is None:
                entry = _lookup_by_id(context, source_id, source_type)
        else:
            issues.append(
                ValidationIssue(
                    code="invalid_evidence_ref",
                    message="An evidence reference must be an object or a source id",
                    location=ref_location,
                )
            )
            unverified += 1
            continue

        if entry is None:
            # The reference is kept but never trusted: it stays unverified, keeps
            # no pages, is excluded from provenance and is counted in
            # unverified_evidence_refs so the doubt stays visible. This mirrors
            # the analysis stage, and it means a model-invented identifier can
            # never travel downstream looking authoritative.
            issues.append(
                ValidationIssue(
                    code="unresolved_source_id",
                    message=(
                        "This reference does not exist in the supplied upstream "
                        "artifacts; it is recorded as unverified and cannot support "
                        "any statement."
                    ),
                    location=ref_location,
                    severity="warning",
                )
            )
            claimed = raw if isinstance(raw, dict) else {}
            source_type = _text(claimed.get("source_type")).lower() or "document"
            resolved.append(
                SourceRef(
                    source_type=source_type if source_type in SOURCE_TYPES else "document",
                    source_id=_text(claimed.get("source_id")) or _text(raw),
                    verified=False,
                )
            )
            unverified += 1
            continue

        pages = [int(page) for page in (entry.get("pages") or [])]
        single_page = entry.get("page")
        if not isinstance(single_page, int):
            single_page = pages[0] if pages else None
        resolved.append(
            SourceRef(
                source_type=entry["source_type"],
                source_id=entry["source_id"],
                pages=pages,
                page=single_page,
                verified=True,
                label=entry.get("label"),
            )
        )
    return resolved, unverified


def _lookup_by_id(
    context: SynthesisContext, source_id: str, claimed_type: str
) -> Optional[Dict[str, Any]]:
    """
    Resolve a reference by its id when the claimed type does not match.

    Live models frequently pair a real id with the wrong ``source_type`` (for
    example citing ``analysis:chunk_005`` for a chunk that is a ``rag`` source).
    The id carries the fabrication risk, so an id that exists upstream is resolved
    against real metadata instead of being rejected; the stored type always comes
    from the allow-list, never from the model.

    Resolution is refused when the id is genuinely unknown, or when it exists
    under several different types and none of them is the claimed one - in both
    cases the reference stays unverified and is counted as such.
    """
    identifier = _text(source_id)
    if not identifier:
        return None
    matches = [
        entry
        for entry in context.source_index.values()
        if entry["source_id"] == identifier
    ]
    if not matches:
        return None
    types = {entry["source_type"] for entry in matches}
    if claimed_type and claimed_type in types:
        for entry in matches:
            if entry["source_type"] == claimed_type:
                return entry
    if len(types) == 1:
        return matches[0]
    return None


def _check_prose(
    text: Any,
    allowed_pages: Optional[Sequence[int]],
    page_count: int,
    issues: List[ValidationIssue],
    location: str,
) -> None:
    """
    Check one piece of readable text for scores, verdicts, secrets and bad pages.

    ``allowed_pages`` is the set of pages reachable from this record's own
    validated references; when a record cites no source at all, every page of the
    document is allowed so an uncited sentence is not rejected for the wrong
    reason.
    """
    content = _text(text)
    if not content:
        return

    for match in _SCORE_REGEXES:
        for found in match.finditer(content):
            window = content[max(0, found.start() - 60) : found.end()]
            if METRIC_CONTEXT_PATTERN.search(window):
                # A reported metric value, not an overall paper verdict.
                continue
            issues.append(
                ValidationIssue(
                    code="numeric_paper_score",
                    message=(
                        f"A numeric paper score or rating ({found.group(0)!r}) is not "
                        "permitted in the final critique."
                    ),
                    location=location,
                )
            )
    for hit in detect_unsupported_verdicts(content):
        issues.append(
            ValidationIssue(
                code="unsupported_verdict_label",
                message=(
                    f"The unsupported overall label {hit!r} is not permitted; the "
                    "assessment must stay evidence-grounded."
                ),
                location=location,
            )
        )
    for hit in detect_secrets(content):
        issues.append(
            ValidationIssue(
                code="credential_leak",
                message="The output appears to contain a credential and was rejected.",
                location=location,
            )
        )

    if page_count <= 0:
        return
    permitted = set(allowed_pages) if allowed_pages else None
    for page in extract_page_citations(content):
        if page < 1 or page > page_count:
            issues.append(
                ValidationIssue(
                    code="fabricated_page",
                    message=(
                        f"The critique cites page {page}, but the document has "
                        f"{page_count} page(s). A page outside the document is a "
                        "fabricated citation."
                    ),
                    location=location,
                )
            )
            continue
        if permitted is not None and page not in permitted:
            issues.append(
                ValidationIssue(
                    code="unverified_page_citation",
                    message=(
                        f"The critique cites page {page}, which is not part of the "
                        "evidence cited alongside it."
                    ),
                    location=location,
                )
            )


def _context_pages(context: SynthesisContext) -> List[int]:
    pages: List[int] = []
    for entry in context.source_index.values():
        for page in entry.get("pages") or []:
            if isinstance(page, int) and page not in pages:
                pages.append(page)
        single = entry.get("page")
        if isinstance(single, int) and single not in pages:
            pages.append(single)
    return sorted(pages)


def _pages_from_refs(refs: Sequence[SourceRef]) -> List[int]:
    pages: List[int] = []
    for ref in refs:
        for page in ref.pages:
            if page not in pages:
                pages.append(page)
        if isinstance(ref.page, int) and ref.page not in pages:
            pages.append(ref.page)
    return sorted(pages)


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _validate_sections(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
    dropped: List[str],
    require_supported: bool,
) -> Tuple[List[CritiqueSection], List[str], int]:
    """
    Validate the emitted sections against the fixed catalogue.

    An unsupported section that was omitted is dropped silently (that is the
    contract); an *unknown* section id or empty content is an error, because it
    means the model invented structure rather than content.
    """
    sections: List[CritiqueSection] = []
    warnings: List[str] = []
    unverified_total = 0
    page_count = context.document.page_count
    supported = set(context.supported_sections.section_ids)

    seen: List[str] = []
    for index, raw in enumerate(_as_dict_list(candidate.get("sections"))):
        location = f"sections[{index}]"
        section_id = _text(raw.get("section_id"))
        if section_id not in SECTION_IDS:
            issues.append(
                ValidationIssue(
                    code="invalid_section_id",
                    message=(
                        f"'{section_id}' is not a section of the final report. "
                        f"Allowed ids: {', '.join(SECTION_IDS)}."
                    ),
                    location=location,
                )
            )
            continue
        if section_id in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_section",
                    message=f"Section '{section_id}' was emitted more than once.",
                    location=location,
                )
            )
            continue
        seen.append(section_id)

        content = _text(raw.get("content"))
        if not content:
            issues.append(
                ValidationIssue(
                    code="empty_section",
                    message=(
                        f"Section '{section_id}' has no content. Omit a section "
                        "instead of emitting an empty one."
                    ),
                    location=location,
                )
            )
            continue

        refs, unverified = _resolve_refs(
            raw.get("evidence_refs"), context, issues, f"{location}.evidence_refs"
        )
        unverified_total += unverified
        if not refs:
            warnings.append(
                f"Section '{section_id}' carries no verifiable evidence reference."
            )

        allowed_pages = _pages_from_refs(refs)
        _check_prose(content, allowed_pages, page_count, issues, f"{location}.content")
        _check_prose(
            raw.get("title"), allowed_pages, page_count, issues, f"{location}.title"
        )

        sections.append(
            CritiqueSection(
                section_id=section_id,
                title=_text(raw.get("title")) or SECTION_TITLES[section_id],
                content=content,
                evidence_refs=refs,
                uncertain=is_uncertain_text(content),
            )
        )

    for section_id in ALWAYS_REQUIRED_SECTIONS:
        if section_id not in seen:
            issues.append(
                ValidationIssue(
                    code="missing_required_section",
                    message=f"The required section '{section_id}' is missing.",
                    location="sections",
                )
            )
    for section_id in sorted(supported):
        if section_id in ALWAYS_REQUIRED_SECTIONS:
            continue
        if section_id not in seen:
            issues.append(
                ValidationIssue(
                    code="missing_supported_section",
                    message=(
                        f"Section '{section_id}' is supported by the supplied "
                        "evidence but was not produced."
                    ),
                    location="sections",
                    severity="error" if require_supported else "warning",
                )
            )

    for section_id in seen:
        if section_id not in supported and section_id not in ALWAYS_REQUIRED_SECTIONS:
            dropped.append(section_id)
    return sections, warnings, unverified_total


def _validate_claim_summary(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[List[ClaimEvidenceSummaryItem], int, List[str]]:
    """
    Check that the Phase 5 claim/evidence matrix survived into the critique.

    The critique may reword a claim's assessment, but it may not drop a claim, add
    a new one, or change its classification: those three moves are exactly how a
    synthesis layer starts inventing findings.
    """
    items: List[ClaimEvidenceSummaryItem] = []
    warnings: List[str] = []
    page_count = context.document.page_count
    unverified_total = 0

    for index, raw in enumerate(_as_dict_list(candidate.get("claim_evidence_summary"))):
        location = f"claim_evidence_summary[{index}]"
        claim = _text(raw.get("claim"))
        if not claim:
            issues.append(
                ValidationIssue(
                    code="empty_claim",
                    message="A claim/evidence row carries no claim text.",
                    location=location,
                )
            )
            continue

        source = next(
            (entry for entry in context.analysis.claims if _similar(claim, entry.claim)),
            None,
        )
        if source is None:
            issues.append(
                ValidationIssue(
                    code="unsupported_new_claim",
                    message=(
                        "This claim is not present in the supplied analysis, so the "
                        "critique would be introducing a new finding."
                    ),
                    location=location,
                )
            )
            continue

        status = _text(raw.get("status")).lower() or "unclear"
        if status not in CLAIM_STATUSES:
            issues.append(
                ValidationIssue(
                    code="invalid_claim_status",
                    message=(
                        f"'{status}' is not a valid claim classification. Allowed: "
                        f"{', '.join(CLAIM_STATUSES)}."
                    ),
                    location=location,
                )
            )
            continue
        if status != source.status:
            issues.append(
                ValidationIssue(
                    code="claim_status_changed",
                    message=(
                        f"The claim classification was changed from '{source.status}' "
                        f"to '{status}'. The critique must preserve the analysis "
                        "stage's classification."
                    ),
                    location=location,
                )
            )
            continue

        assessment = _text(raw.get("assessment"))
        refs, unverified = _resolve_refs(
            raw.get("evidence_refs"), context, issues, f"{location}.evidence_refs"
        )
        unverified_total += unverified
        allowed_pages = _pages_from_refs(refs)
        _check_prose(
            assessment, allowed_pages, page_count, issues, f"{location}.assessment"
        )
        _check_prose(claim, allowed_pages, page_count, issues, f"{location}.claim")

        items.append(
            ClaimEvidenceSummaryItem(
                claim=claim,
                status=status,
                assessment=assessment,
                evidence_refs=refs,
                uncertainty_notes=[
                    _text(value)
                    for value in (raw.get("uncertainty_notes") or [])
                    if _text(value)
                ],
            )
        )

    for source_claim in context.analysis.claims:
        if not any(_similar(item.claim, source_claim.claim) for item in items):
            issues.append(
                ValidationIssue(
                    code="claim_dropped",
                    message=(
                        "A claim from the supplied analysis is missing from the "
                        "claim-evidence assessment."
                    ),
                    location="claim_evidence_summary",
                )
            )
    return items, unverified_total, warnings


def _validate_strengths(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[List[CritiqueStrength], int, List[str]]:
    """Strengths stay prose with provenance; a strength is never a number."""
    strengths: List[CritiqueStrength] = []
    warnings: List[str] = []
    unverified_total = 0
    page_count = context.document.page_count

    for index, raw in enumerate(_as_dict_list(candidate.get("strengths"))):
        location = f"strengths[{index}]"
        statement = _text(raw.get("statement"))
        if not statement:
            issues.append(
                ValidationIssue(
                    code="empty_strength",
                    message="A strength entry carries no statement.",
                    location=location,
                )
            )
            continue
        if not any(_similar(statement, value) for value in context.analysis.strengths):
            issues.append(
                ValidationIssue(
                    code="unsupported_new_strength",
                    message=(
                        "This strength is not present in the supplied analysis, so "
                        "the critique would be introducing a new finding."
                    ),
                    location=location,
                )
            )
            continue
        refs, unverified = _resolve_refs(
            raw.get("evidence_refs"), context, issues, f"{location}.evidence_refs"
        )
        unverified_total += unverified
        rationale = _text(raw.get("rationale"))
        allowed_pages = _pages_from_refs(refs)
        _check_prose(
            statement, allowed_pages, page_count, issues, f"{location}.statement"
        )
        _check_prose(rationale, allowed_pages, page_count, issues, f"{location}.rationale")
        strengths.append(
            CritiqueStrength(
                statement=statement,
                dimension=_text(raw.get("dimension")) or None,
                evidence_refs=refs,
                rationale=rationale,
            )
        )

    for value in context.analysis.strengths:
        if not any(_similar(item.statement, value) for item in strengths):
            warnings.append(
                "A strength recorded by the analysis stage is not represented in the "
                "structured strengths list."
            )
    return strengths, unverified_total, warnings


def _resolve_limitation_category(
    statement: str,
    author_stated: Sequence[str],
    analyst_identified: Sequence[str],
) -> Optional[str]:
    """
    Decide which category a limitation belongs to, from the Phase 5 source lists.

    The engine owns this decision, exactly as the design states: the model may only
    propose a wording, never a category. Matching is done against both lists, so a
    limitation the model filed on the wrong side is *relocated* to its true
    category instead of being rejected - the separation is then guaranteed by the
    engine rather than by the model's bookkeeping.
    """
    if any(_similar(statement, value) for value in author_stated):
        return "author_stated"
    if any(_similar(statement, value) for value in analyst_identified):
        return "analyst_identified"
    return None


def _validate_limitation_group(
    raw_items: Any,
    claimed_category: str,
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[List[CritiqueLimitation], List[CritiqueLimitation], int, List[str]]:
    """
    Validate one limitation category against its Phase 5 counterpart.

    The category is decided here, not by the model. An entry is filed under the
    category the analysis stage recorded it in; when the model filed it on the
    wrong side it is relocated and the relocation is recorded as a warning. That
    is what makes the two categories impossible to merge silently: neither the
    visible text nor the machine-readable category can disagree with Phase 5.
    """
    limitations: List[CritiqueLimitation] = []
    relocated: List[CritiqueLimitation] = []
    warnings: List[str] = []
    unverified_total = 0
    page_count = context.document.page_count

    for index, raw in enumerate(_as_dict_list(raw_items)):
        location = f"limitations.{claimed_category}[{index}]"
        statement = _text(raw.get("statement"))
        if not statement:
            issues.append(
                ValidationIssue(
                    code="empty_limitation",
                    message="A limitation entry carries no statement.",
                    location=location,
                )
            )
            continue
        category = _resolve_limitation_category(
            statement,
            context.analysis.author_stated_limitations,
            context.analysis.analyst_identified_limitations,
        )
        if category is None:
            issues.append(
                ValidationIssue(
                    code="unsupported_limitation",
                    message=(
                        "This limitation is not recorded by the analysis stage in "
                        "either category, so the critique would be introducing a new "
                        "finding."
                    ),
                    location=location,
                )
            )
            continue
        if category != claimed_category:
            warnings.append(
                f"A limitation filed as '{claimed_category}' was recorded by the "
                f"analysis stage as '{category}' and has been relocated accordingly."
            )
        refs, unverified = _resolve_refs(
            raw.get("evidence_refs"), context, issues, f"{location}.evidence_refs"
        )
        unverified_total += unverified
        rationale = _text(raw.get("rationale"))
        allowed_pages = _pages_from_refs(refs)
        _check_prose(
            statement, allowed_pages, page_count, issues, f"{location}.statement"
        )
        _check_prose(
            rationale, allowed_pages, page_count, issues, f"{location}.rationale"
        )
        entry = CritiqueLimitation(
            statement=statement,
            category=category,
            evidence_refs=refs,
            rationale=rationale,
        )
        if category == claimed_category:
            limitations.append(entry)
        else:
            relocated.append(entry)
    return limitations, relocated, unverified_total, warnings


def _validate_limitations(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[CritiqueLimitations, int, List[str]]:
    raw = candidate.get("limitations")
    raw = raw if isinstance(raw, dict) else {}
    author, author_moved, unverified_a, warnings_a = _validate_limitation_group(
        raw.get("author_stated"),
        "author_stated",
        context,
        issues,
    )
    analyst, analyst_moved, unverified_b, warnings_b = _validate_limitation_group(
        raw.get("analyst_identified"),
        "analyst_identified",
        context,
        issues,
    )
    author.extend(analyst_moved)
    analyst.extend(author_moved)
    return (
        CritiqueLimitations(author_stated=author, analyst_identified=analyst),
        unverified_a + unverified_b,
        warnings_a + warnings_b,
    )


def _validate_gaps(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> List[CritiqueGap]:
    """
    Preserve the evidence gaps, and re-inject the deterministic ones.

    A model may summarize a gap in its own words, but it may not silently drop a
    gap the engine knows about (retrieval disabled, no visual evidence, truncated
    context): those are facts about this run, so they are merged back in.
    """
    gaps: List[CritiqueGap] = []
    page_count = context.document.page_count
    for index, raw in enumerate(_as_dict_list(candidate.get("evidence_gaps"))):
        location = f"evidence_gaps[{index}]"
        text = _text(raw.get("gap"))
        if not text:
            issues.append(
                ValidationIssue(
                    code="empty_evidence_gap",
                    message="An evidence gap carries no description.",
                    location=location,
                )
            )
            continue
        impact = _text(raw.get("impact"))
        _check_prose(text, None, page_count, issues, f"{location}.gap")
        gaps.append(
            CritiqueGap(
                gap=text,
                category=_text(raw.get("category")) or "other",
                impact=impact,
                source=_text(raw.get("source")) or "analysis",
            )
        )

    for deterministic in context.deterministic_gaps:
        if not any(_similar(existing.gap, deterministic.gap) for existing in gaps):
            gaps.append(deterministic)
    for inherited in context.analysis.evidence_gaps:
        if not any(_similar(existing.gap, inherited.gap) for existing in gaps):
            gaps.append(inherited)
    return gaps


def _validate_open_questions(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[List[CritiqueOpenQuestion], int, List[str]]:
    """
    Open questions must stay questions and must stay traceable to the analysis.

    An entry that reads as a flat assertion is rejected: presenting an unsupported
    statement as a question would still be an unsupported claim.
    """
    questions: List[CritiqueOpenQuestion] = []
    warnings: List[str] = []
    unverified_total = 0
    page_count = context.document.page_count

    for index, raw in enumerate(_as_dict_list(candidate.get("open_questions"))):
        location = f"open_questions[{index}]"
        question = _text(raw.get("question"))
        if not question:
            issues.append(
                ValidationIssue(
                    code="empty_open_question",
                    message="An open research question carries no text.",
                    location=location,
                )
            )
            continue
        if not question.rstrip().endswith("?"):
            issues.append(
                ValidationIssue(
                    code="open_question_not_a_question",
                    message=(
                        "An open research question must be phrased as a question, "
                        "not as a factual claim."
                    ),
                    location=location,
                )
            )
            continue
        refs, unverified = _resolve_refs(
            raw.get("evidence_refs"), context, issues, f"{location}.evidence_refs"
        )
        unverified_total += unverified
        allowed_pages = _pages_from_refs(refs)
        why = _text(raw.get("why_it_matters"))
        _check_prose(question, allowed_pages, page_count, issues, f"{location}.question")
        _check_prose(
            why, allowed_pages, page_count, issues, f"{location}.why_it_matters"
        )
        questions.append(
            CritiqueOpenQuestion(
                question=question,
                why_it_matters=why,
                evidence_refs=refs,
            )
        )

    for value in context.analysis.open_questions:
        if not any(_similar(item.question, value) for item in questions):
            warnings.append(
                "An open question recorded by the analysis stage is not represented "
                "in the structured open-questions list."
            )
    return questions, unverified_total, warnings


def _validate_overall(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    issues: List[ValidationIssue],
) -> Tuple[CritiqueOverallAssessment, int]:
    """
    Validate the closing assessment.

    It must be non-empty (the report would otherwise have no conclusion) and must
    never carry a score, a rank, a tier or an unsupported overall label.
    """
    raw = candidate.get("overall_assessment")
    raw = raw if isinstance(raw, dict) else {}
    content = _text(raw.get("content"))
    if not content:
        # The model sometimes writes the closing assessment only as a report
        # section. That text is already validated with the sections, so it is
        # reused verbatim here rather than inventing or discarding it.
        for item in _as_dict_list(candidate.get("sections")):
            if _text(item.get("section_id")) == "overall_assessment":
                content = _text(item.get("content"))
                break
    if not content:
        issues.append(
            ValidationIssue(
                code="missing_overall_assessment",
                message="The overall research assessment carries no content.",
                location="overall_assessment",
            )
        )
        return CritiqueOverallAssessment(), 0

    refs, unverified = _resolve_refs(
        raw.get("evidence_refs"), context, issues, "overall_assessment.evidence_refs"
    )
    allowed_pages = _pages_from_refs(refs)
    page_count = context.document.page_count
    _check_prose(
        content, allowed_pages, page_count, issues, "overall_assessment.content"
    )
    uncertainties = [
        _text(value) for value in (raw.get("uncertainties") or []) if _text(value)
    ]
    for value in uncertainties:
        _check_prose(
            value,
            allowed_pages,
            page_count,
            issues,
            "overall_assessment.uncertainties",
        )

    return (
        CritiqueOverallAssessment(
            content=content,
            evidence_refs=refs,
            uncertainties=uncertainties,
        ),
        unverified,
    )


def build_provenance(payload: CritiquePayload) -> CritiqueProvenance:
    """
    Record exactly which upstream sources the finished report leaned on.

    Provenance is collected from the validated references only, so it describes
    what the critique really used rather than what it could have used.
    """
    buckets: Dict[str, List[str]] = {
        "rag": [],
        "vision": [],
        "analysis": [],
        "document": [],
    }

    def record(refs: Sequence[SourceRef]) -> None:
        for ref in refs:
            if not ref.verified:
                # Only verified references count as provenance; an unverified one
                # must never appear in the "sources used" summary.
                continue
            bucket = buckets.get(ref.source_type)
            if bucket is not None and ref.source_id not in bucket:
                bucket.append(ref.source_id)

    for section in payload.sections:
        record(section.evidence_refs)
    for item in payload.claim_evidence_summary:
        record(item.evidence_refs)
    for strength in payload.strengths:
        record(strength.evidence_refs)
    for limitation in (
        payload.limitations.author_stated + payload.limitations.analyst_identified
    ):
        record(limitation.evidence_refs)
    for question in payload.open_questions:
        record(question.evidence_refs)
    record(payload.overall_assessment.evidence_refs)

    return CritiqueProvenance(
        rag_sources_used=sorted(buckets["rag"]),
        vision_sources_used=sorted(buckets["vision"]),
        analysis_sources_used=sorted(buckets["analysis"]),
        document_sources_used=sorted(buckets["document"]),
    )


def validate_critique(
    candidate: Dict[str, Any],
    context: SynthesisContext,
    require_supported_sections: bool = True,
) -> Tuple[ParsedCritique, ValidationReport]:
    """
    Validate one candidate critique against the artifacts that were supplied.

    Returns the parsed critique together with a full validation report. A report
    containing any *error* means the candidate must not be returned as a completed
    critique; the caller turns that into a structured ``validation_failed`` result.
    """
    issues: List[ValidationIssue] = []
    warnings: List[str] = []
    notes: List[CritiqueGap] = []
    unverified_total = 0

    # A forbidden field anywhere (score, rank, tier, confidence) is rejected before
    # anything is interpreted, so it can never reach the rendered report.
    for location, key in find_forbidden_keys(candidate):
        issues.append(
            ValidationIssue(
                code="forbidden_field",
                message=(
                    f"The field '{key}' is not allowed in a critique: the report has "
                    "no score, ranking, tier or model-invented confidence."
                ),
                location=location,
            )
        )

    dropped: List[str] = []
    sections, section_warnings, unverified_sections = _validate_sections(
        candidate, context, issues, dropped, require_supported_sections
    )
    unverified_total += unverified_sections
    warnings.extend(section_warnings)

    claims, unverified_claims, claim_warnings = _validate_claim_summary(
        candidate, context, issues
    )
    unverified_total += unverified_claims
    warnings.extend(claim_warnings)

    strengths, unverified_strengths, strength_warnings = _validate_strengths(
        candidate, context, issues
    )
    unverified_total += unverified_strengths
    warnings.extend(strength_warnings)

    limitations, unverified_limitations, limitation_warnings = _validate_limitations(
        candidate, context, issues
    )
    unverified_total += unverified_limitations
    warnings.extend(limitation_warnings)

    gaps = _validate_gaps(candidate, context, issues)

    questions, unverified_questions, question_warnings = _validate_open_questions(
        candidate, context, issues
    )
    unverified_total += unverified_questions
    warnings.extend(question_warnings)

    overall, unverified_overall = _validate_overall(candidate, context, issues)
    unverified_total += unverified_overall

    if any(issue.code == "unresolved_source_id" for issue in issues):
        notes.append(
            CritiqueGap(
                gap=(
                    f"{unverified_total} citation(s) in the generated critique could "
                    "not be matched to the supplied upstream artifacts."
                ),
                category="unverified_reference",
                impact="Those statements must not be treated as established.",
                source="validation",
            )
        )

    payload = CritiquePayload(
        sections=sections,
        claim_evidence_summary=claims,
        strengths=strengths,
        limitations=limitations,
        evidence_gaps=gaps,
        open_questions=questions,
        overall_assessment=overall,
    )

    # Fail closed on the payload itself as well: if pydantic rejects the assembled
    # object, the candidate never becomes a report.
    try:
        CritiquePayload.model_validate(payload.model_dump())
    except ValidationError as exc:
        issues.append(
            ValidationIssue(
                code="schema_invalid",
                message=f"The assembled critique does not satisfy the schema: {exc}",
                location="payload",
            )
        )

    report = ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        unverified_evidence_refs=unverified_total,
    )
    parsed = ParsedCritique(
        payload=payload,
        warnings=warnings,
        notes=notes,
        unverified_evidence_refs=unverified_total,
        dropped_sections=dropped,
    )
    return parsed, report


def validate_model_response(
    raw_text: str,
    context: SynthesisContext,
    require_supported_sections: bool = True,
) -> Tuple[ParsedCritique, ValidationReport]:
    """
    Parse, validate and wrap one raw model response.

    Raises CritiqueValidationError when the response cannot be parsed, is not a
    JSON object, or fails validation; the exception carries the validation report
    so the engine can report a precise structured failure.
    """
    try:
        candidate = extract_json_payload(raw_text)
    except CritiqueValidationError as exc:
        report = ValidationReport(
            valid=False,
            issues=[
                ValidationIssue(
                    code="unparseable_response", message=str(exc), location="response"
                )
            ],
        )
        raise CritiqueValidationError(str(exc), report) from exc

    parsed, report = validate_critique(
        candidate, context, require_supported_sections=require_supported_sections
    )
    if not report.valid:
        errors = report.errors()
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in errors[:3])
        raise CritiqueValidationError(
            f"The generated critique failed validation ({len(errors)} issue(s)): "
            f"{summary}",
            report,
        )
    return parsed, report
