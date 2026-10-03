"""
AI Research Paper Critic - Streamlit frontend.

This module is presentation only. It makes exactly two backend calls:

    analyze_document(pdf_path)  - local document understanding (no model call)
    run_pipeline(pdf_path, ...)  - the full analysis pipeline

What this file does with the results
-------------------------------------
It presents the analysis of *this particular paper* in clearer language. Every
sentence shown comes from the backend analysis; nothing is generated, reworded by a
model, or invented here. For each part of the paper the interface answers:

1. What does this paper say here?   - the analysis finding for that part
2. What does that mean?             - the Critique Engine's reader-facing synthesis
3. What evidence supports it?       - the pages the analysis cited

Where the backend did not provide enough information, the interface says so instead
of filling the gap.
"""

import logging
import os
import time

import pandas as pd
import pymupdf
import streamlit as st

from backend.document_pre_analyzer.analyzer import analyze_document
from backend.pipeline import run_pipeline

logger = logging.getLogger("ai_research_paper_critic.frontend")

st.set_page_config(page_title="AI Research Paper Critic", page_icon="📄", layout="wide")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

PDF_MAGIC = b"%PDF-"
PDF_MAGIC_SEARCH_BYTES = 1024

NOT_ENOUGH_INFO = (
    "The available evidence does not provide enough information to explain this "
    "aspect confidently."
)

# The six steps of the analysis, in the language used throughout the interface.
STAGE_LABELS = [
    ("document_pre_analyzer", "Document Understanding"),
    ("jev_router", "Intelligent Routing"),
    ("rag", "Textual Evidence"),
    ("vision", "Visual Analysis"),
    ("analysis", "Research Analysis"),
    ("critique", "Final Research Critique"),
]

STAGE_DESCRIPTIONS = {
    "Document Understanding": "We identified the paper's structure and important content.",
    "Intelligent Routing": "We selected the analysis methods useful for this paper.",
    "Textual Evidence": "We found passages relevant to the paper's claims and findings.",
    "Visual Analysis": "We examined important figures and tables.",
    "Research Analysis": (
        "We examined the paper's methodology, experiments, claims, results and "
        "limitations."
    ),
    "Final Research Critique": "We combined the evidence into a structured critique.",
}

STATUS_ICON = {"completed": "✅", "running": "⏳", "failed": "⚠️", "pending": "⬜"}

# Human-readable names for the analysis dimensions returned by the backend.
DIMENSION_TITLES = {
    "research_problem": "Problem & Motivation",
    "contribution": "Contribution",
    "methodology": "Methodology",
    "data": "Data",
    "baselines": "Baselines",
    "metrics": "Metrics",
    "results": "Results",
    "reproducibility": "Reproducibility",
    "internal_consistency": "Internal Consistency",
}

# Each analysis dimension is paired with the critique section that explains it in
# reader-friendly language. Both come from the backend; neither is written here.
DIMENSION_TO_CRITIQUE = {
    "research_problem": "research_problem",
    "contribution": "contribution",
    "methodology": "methodology",
    "data": "data_and_experimental_design",
    "baselines": "baselines_and_metrics",
    "metrics": "baselines_and_metrics",
    "results": "results_and_evidence",
    "reproducibility": "reproducibility",
    "internal_consistency": "internal_consistency",
}


# -----------------------------
# Session state
# -----------------------------
# Results live here so that opening or closing a section never re-runs the
# pipeline. A new submission clears them, so two papers can never be mixed.
for _key, _default in (
    ("document_profile", None),
    ("document_profile_filename", None),
    ("analysis_elapsed_seconds", None),
    ("pdf_path", None),
    ("pipeline_result", None),
    ("pipeline_progress", None),
    ("pipeline_elapsed_seconds", None),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default


def reset_analysis_state():
    """Forget a previous analysis so a new paper starts from a clean slate."""
    st.session_state.pipeline_result = None
    st.session_state.pipeline_progress = None
    st.session_state.pipeline_elapsed_seconds = None


# -----------------------------
# Small helpers
# -----------------------------
def snippet(text, limit=160):
    """Shorten long text so a page stays scannable."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)].rstrip() + "…"


def friendly_failure(message="The analysis could not be finalized for this paper."):
    """Clean user-facing failure; the real reason stays in the server log."""
    st.error(message)
    st.caption("Please try running the analysis again.")


def stage_result(result, stage):
    """The backend record for one stage (never rebuilt or altered here)."""
    return ((result or {}).get("stages") or {}).get(stage) or {}


def stage_payload(result, stage):
    """The backend's own result object for one stage, or an empty dict."""
    return (stage_result(result, stage) or {}).get("result") or {}


def level_label(value):
    """A routing level may arrive as a plain string or as a router enum member."""
    return getattr(value, "value", value)


def page_text(reference):
    """Page numbers of one evidence reference, formatted for display."""
    pages = reference.get("pages") or []
    if pages:
        return ", ".join(str(page) for page in pages)
    single = reference.get("page")
    return str(single) if single else ""


def evidence_pages(references):
    """A readable 'Page 3, 8' line, or an empty string when no page is known."""
    pages = []
    for reference in references or []:
        if not isinstance(reference, dict):
            continue
        text = page_text(reference)
        if text and text not in pages:
            pages.append(text)
    return "Page " + ", ".join(pages) if pages else ""


def referenced_pages(references):
    """The set of page numbers a list of references points at."""
    pages = set()
    for reference in references or []:
        if not isinstance(reference, dict):
            continue
        for page in reference.get("pages") or []:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue
        single = reference.get("page")
        if single is not None:
            try:
                pages.add(int(single))
            except (TypeError, ValueError):
                continue
    return pages


def show_evidence(references):
    """Render the evidence line for a statement, or nothing when it is unknown."""
    line = evidence_pages(references)
    if line:
        st.caption(f"📍 Evidence: {line}")


def dedupe_records(records, key):
    """Deduplicate backend records that repeat across stages."""
    seen = set()
    unique = []
    for record in records or []:
        text = str(record.get(key) or "").strip().lower()
        if text and text not in seen:
            seen.add(text)
            unique.append(record)
    return unique


def upload_and_validate(uploaded_file):
    """Save the upload, confirm it is a real PDF, and return its path."""
    file_path = save_uploaded_file(uploaded_file)
    if not file_path:
        st.error("The file could not be saved. Please try again.")
        return None
    is_valid, problem = validate_pdf(file_path)
    if not is_valid:
        st.error(problem)
        return None
    return file_path


# -----------------------------
# Upload handling
# -----------------------------
def save_uploaded_file(uploaded_file):
    """Persist the uploaded file inside uploaded_files/ and return its path."""
    if uploaded_file is None:
        return None
    file_path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded_file.name))
    with open(file_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return file_path


def validate_pdf(file_path):
    """Structural check before anything is analysed."""
    try:
        size = os.path.getsize(file_path)
    except OSError:
        logger.warning("Uploaded file could not be read.")
        return False, "The uploaded file could not be read. Please upload it again."
    if size == 0:
        return False, "The uploaded file is empty (0 bytes). Please upload a valid PDF."
    if size < 8:
        return False, "The uploaded file is too small to be a valid PDF."
    try:
        with open(file_path, "rb") as handle:
            header = handle.read(PDF_MAGIC_SEARCH_BYTES)
    except OSError:
        logger.warning("Uploaded file could not be opened.")
        return False, "The uploaded file could not be opened. Please upload it again."
    if PDF_MAGIC not in header:
        return False, (
            "This file is not a valid PDF. Please upload a research paper in PDF format."
        )
    return True, ""


def analyzer_failure_message(exc):
    """Translate a document-reading failure into a friendly explanation."""
    if isinstance(exc, FileNotFoundError):
        return "The uploaded PDF could not be found. Please upload it again."
    if isinstance(exc, pymupdf.EmptyFileError):
        return "The uploaded PDF is empty and could not be read."
    if isinstance(exc, pymupdf.FileDataError):
        return (
            "The uploaded PDF could not be opened. It may be corrupted, truncated or "
            "password protected."
        )
    if isinstance(exc, MemoryError):
        return "This document is too large to read. Please try a smaller PDF."
    return (
        "This PDF could not be read. Please check that it is a valid, unencrypted "
        "research paper."
    )


def analyze_uploaded_pdf(file_path):
    """Read the paper's structure locally (no model is called here)."""
    started = time.time()
    try:
        with st.spinner("Reading the paper…"):
            profile = analyze_document(file_path)
    except Exception as exc:  # noqa: BLE001 - never expose a traceback to the user
        logger.exception("Document reading failed for %s", file_path)
        st.error(analyzer_failure_message(exc))
        return None
    if not profile or (profile.get("page_count") or 0) <= 0:
        logger.error("No readable pages found in %s", file_path)
        st.error("This PDF does not contain any readable pages.")
        return None
    st.session_state.document_profile = profile
    st.session_state.document_profile_filename = os.path.basename(file_path)
    st.session_state.analysis_elapsed_seconds = time.time() - started
    st.session_state.pdf_path = file_path
    return profile


# -----------------------------
# Progress
# -----------------------------
def render_progress(progress_state):
    """Show the six steps of the analysis in plain language."""
    if not progress_state:
        return
    st.subheader("Analysis Progress")
    for stage, label in STAGE_LABELS:
        status = (progress_state or {}).get(stage) or "pending"
        st.markdown(f"{STATUS_ICON.get(status, '⬜')} **{label}**")
        if status == "completed":
            st.caption(STAGE_DESCRIPTIONS.get(label, ""))


def _stage_records(result, field):
    """Records of one kind reported by the analysis and/or the critique."""
    records = []
    for stage in ("critique", "analysis"):
        for item in (stage_payload(result, stage) or {}).get(field) or []:
            if isinstance(item, dict):
                records.append(item)
    return records


def render_final_critique(result):
    """The main output: what this paper says, and how well it is supported."""
    critique = stage_payload(result, "critique")
    if not critique:
        st.subheader("Final Research Critique")
        st.info("The critique is not available yet. Run the full analysis to generate it.")
        return

    if critique.get("status") != "completed":
        logger.warning("Critique not completed: %s", critique.get("message"))
        st.subheader("Final Research Critique")
        friendly_failure()
        return

    st.header("Final Research Critique")
    sections = critique.get("sections") or []
    by_id = {item.get("section_id"): item for item in sections}

    summary = by_id.get("executive_summary")
    if summary:
        st.markdown("### Executive Summary")
        st.markdown(summary.get("content") or "")
        show_evidence(summary.get("evidence_refs"))

    overall = critique.get("overall_assessment") or {}
    if overall.get("content"):
        st.markdown("### Overall Assessment")
        st.markdown(overall["content"])
        show_evidence(overall.get("evidence_refs"))
        for uncertainty in (overall.get("uncertainties") or []):
            st.caption(f"Not fully settled: {uncertainty}")

    render_strengths(critique)
    render_limitations(critique)
    render_evidence_gaps(result)
    render_open_questions(result)

    st.markdown("### Full Critique")
    st.caption("Every section below was produced from this paper. Open one to read it.")
    for item in sections:
        if item.get("section_id") == "executive_summary":
            continue
        title = item.get("title") or item.get("section_id")
        with st.expander(title):
            st.markdown(item.get("content") or "")
            show_evidence(item.get("evidence_refs"))


def render_strengths(critique):
    """What this paper does well, with the reasoning the analysis recorded."""
    strengths = critique.get("strengths") or []
    if not strengths:
        return
    st.markdown("### Strengths")
    for item in strengths:
        st.markdown(f"**{item.get('statement')}**")
        if item.get("rationale"):
            st.markdown("**Simply put:**")
            st.write(item["rationale"])
        show_evidence(item.get("evidence_refs"))


def render_limitations(critique):
    """Limitations of this paper, in two separate groups."""
    limitations = critique.get("limitations") or {}
    author_items = limitations.get("author_stated") or []
    analyst_items = limitations.get("analyst_identified") or []
    if not author_items and not analyst_items:
        return

    st.markdown("### Limitations")
    if author_items:
        st.markdown("**Stated by the authors**")
        for item in author_items:
            st.write(f"- {item.get('statement')}")
            show_evidence(item.get("evidence_refs"))
    if analyst_items:
        st.markdown("**Identified during analysis**")
        st.caption("Noted while analysing the paper; not stated by the authors.")
        for item in analyst_items:
            st.write(f"- {item.get('statement')}")
            show_evidence(item.get("evidence_refs"))


def render_evidence_gaps(result):
    """Where this paper's evidence stops short."""
    gaps = dedupe_records(_stage_records(result, "evidence_gaps"), "gap")
    if not gaps:
        return
    st.markdown("### Evidence Gaps")
    st.caption(
        "Points where this paper's evidence does not settle a question; not claims "
        "that the paper is wrong."
    )
    for gap in gaps:
        st.markdown(f"**{gap.get('gap')}**")
        if gap.get("impact"):
            st.markdown("**What this means:**")
            st.write(gap["impact"])


def render_open_questions(result):
    """Questions this paper leaves open."""
    questions = dedupe_records(_stage_records(result, "open_questions"), "question")
    if not questions:
        return
    st.markdown("### Open Research Questions")
    for item in questions:
        question = str(item.get("question") or "").strip()
        if not question.endswith("?"):
            question += "?"
        st.markdown(f"**{question}**")
        if item.get("why_it_matters"):
            st.markdown("**Why it remains open:**")
            st.write(item["why_it_matters"])
        show_evidence(item.get("evidence_refs"))


def critique_text(result, section_id):
    """The Critique Engine's reader-friendly text for one section, if present."""
    critique = stage_payload(result, "critique")
    for item in (critique or {}).get("sections") or []:
        if item.get("section_id") == section_id:
            return item.get("content") or ""
    return ""


def render_dimension(result, dimension):
    # Present one part of the paper: what it says, what it means, on what evidence.
    # The first part is the analysis summary; the plain-language part is the Critique
    # Engine's own explanation, so simplification comes from the backend, not here.
    key = dimension.get("dimension")
    st.markdown(f"### {DIMENSION_TITLES.get(key, 'Analysis')}")

    summary = dimension.get("summary") or ""
    st.markdown("**What this paper says**")
    st.write(summary if summary else NOT_ENOUGH_INFO)

    findings = dimension.get("findings") or []
    st.markdown("**Simply put**")
    plain = critique_text(result, DIMENSION_TO_CRITIQUE.get(key, ""))
    if plain:
        st.write(plain)
    elif findings:
        for finding in findings:
            st.write(f"- {finding.get('statement')}")
    else:
        st.write(NOT_ENOUGH_INFO)

    references = list(dimension.get("evidence_refs") or [])
    for finding in findings:
        references.extend(finding.get("evidence_refs") or [])
    st.markdown("**Evidence**")
    if evidence_pages(references):
        show_evidence(references)
    else:
        st.caption("No page reference was recorded for this part of the paper.")


def render_claims(matrix):
    """The paper's main claims and whether its evidence supports them."""
    st.markdown("### Claims & Evidence")
    for item in matrix:
        st.markdown(f"**{item.get('claim')}**")
        st.markdown("**Simply put**")
        assessment = item.get("assessment") or ""
        st.write(assessment if assessment else NOT_ENOUGH_INFO)
        if item.get("missing_evidence"):
            st.markdown("**Still missing**")
            for value in item["missing_evidence"]:
                st.write(f"- {value}")
        references = list(item.get("supporting_evidence") or [])
        references.extend(item.get("contradicting_evidence") or [])
        st.markdown("**Evidence**")
        if evidence_pages(references):
            show_evidence(references)
        else:
            st.caption("No page reference was recorded for this claim.")


def render_research_analysis(result):
    """What this paper says about its own research, part by part."""
    analysis = stage_payload(result, "analysis")
    if not analysis:
        return

    st.header("Research Analysis")
    if analysis.get("status") != "completed" or not analysis.get("analysis"):
        logger.warning("Analysis not completed: %s", analysis.get("message"))
        friendly_failure("The research analysis could not be completed.")
        return

    payload = analysis["analysis"]
    shown = False
    for key in DIMENSION_TITLES:
        dimension = payload.get(key) or {}
        if not dimension.get("summary") and not dimension.get("findings"):
            continue
        shown = True
        render_dimension(result, dimension)

    matrix = (payload.get("claim_evidence_matrix") or {}).get("items") or []
    if matrix:
        shown = True
        render_claims(matrix)

    if not shown:
        st.info("The analysis did not produce findings for this paper.")


def findings_on_page(result, page):
    """Analysis findings whose evidence points at a given page."""
    if not page:
        return []
    try:
        page = int(page)
    except (TypeError, ValueError):
        return []
    analysis = stage_payload(result, "analysis")
    payload = (analysis or {}).get("analysis") or {}
    matches = []
    for key in DIMENSION_TITLES:
        dimension = payload.get(key) or {}
        for finding in dimension.get("findings") or []:
            references = list(finding.get("evidence_refs") or [])
            references.extend(dimension.get("evidence_refs") or [])
            if page in referenced_pages(references):
                matches.append(finding.get("statement"))
    return [text for text in matches if text]


def render_visual_analysis(result):
    """What this paper's figures and tables show, and why they matter."""
    vision = stage_payload(result, "vision")
    if not vision:
        return

    st.header("Visual Analysis")
    if not vision.get("enabled"):
        st.info("The analysis did not need visual analysis for this paper.")
        return

    assets = vision.get("results") or []
    for asset in assets:
        kind = (asset.get("asset_type") or "figure").capitalize()
        page = asset.get("page_number")
        st.markdown(f"### {kind}{f' - page {page}' if page else ''}")
        image_path = asset.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=asset.get("caption") or "")
        elif asset.get("caption"):
            st.caption(asset.get("caption"))

        st.markdown("**What the paper is showing**")
        st.write(asset.get("observation") or NOT_ENOUGH_INFO)
        st.markdown("**What it means**")
        st.write(asset.get("interpretation") or NOT_ENOUGH_INFO)

        related = findings_on_page(result, page)
        if related:
            st.markdown("**Why it matters**")
            for statement in related[:2]:
                st.write(f"- {statement}")
        if page:
            st.caption(f"📍 Evidence: Page {page}")

    unavailable = (vision.get("failures") or []) + (
        vision.get("extraction_failures") or []
    )
    for failure in unavailable:
        logger.warning("Visual asset unavailable: %s", failure.get("asset_id"))
        st.caption("Visual analysis was unavailable for this item.")

    if not assets and not unavailable:
        st.info("This paper has no figures or tables to analyse.")


def render_textual_evidence(result):
    """Passages from this paper that the analysis actually used."""
    rag = stage_payload(result, "rag")
    if not rag:
        return

    st.header("Textual Evidence")
    if not rag.get("enabled"):
        st.info("The analysis did not need textual search for this paper.")
        return

    passages = rag.get("results") or []
    if not passages:
        st.info("No relevant passage was found in this paper.")
        return

    for passage in passages:
        start = passage.get("page_start")
        end = passage.get("page_end")
        pages = f"page {start}" if start == end else f"pages {start}-{end}"
        st.markdown(f"### Evidence from {pages}")
        if passage.get("section"):
            st.caption(f"From the section: {passage['section']}")
        st.write(snippet(passage.get("text"), 1200))
        related = findings_on_page(result, start)
        if related:
            st.markdown("**Why this matters**")
            for statement in related[:2]:
                st.write(f"- {statement}")


def render_routing(result):
    """Which kinds of analysis were used for this paper, and why that suited it."""
    routing = stage_payload(result, "jev_router")
    if not routing:
        return

    st.header("Intelligent Routing")
    st.markdown("**The analysis used for this paper**")

    rag = stage_payload(result, "rag")
    vision = stage_payload(result, "vision")
    analysis = stage_payload(result, "analysis")

    chosen = []
    reasons = []
    if (rag or {}).get("enabled"):
        count = len((rag or {}).get("results") or [])
        chosen.append("Textual Evidence")
        if count:
            reasons.append(
                f"The paper contains written passages that support its claims; {count} "
                "of them were examined."
            )
    if (vision or {}).get("enabled"):
        found = (vision or {}).get("assets_found") or 0
        chosen.append("Visual Analysis")
        if found:
            reasons.append(
                f"The paper contains {found} figure(s) or table(s) that carry part of "
                "its evidence."
            )
    if (analysis or {}).get("status") == "completed":
        chosen.append("Research Analysis")

    for label in chosen:
        st.write(f"✅ {label}")
    if reasons:
        st.markdown("**Why these were useful for this paper**")
        for reason in reasons:
            st.write(f"- {reason}")


def render_technical_details(result, elapsed_seconds=None):
    """
    Small collapsed panel for people who want the engineering picture.

    Nothing here is needed to understand the critique, so it stays closed.
    """
    if not result:
        return

    with st.expander("Technical details (optional)"):
        st.caption(
            "How the analysis was produced. Useful for reproducibility; not required "
            "to read the critique."
        )
        rows = []
        for stage, _label in STAGE_LABELS:
            entry = stage_result(result, stage)
            rows.append(
                {
                    "Stage": stage,
                    "Status": entry.get("status") or "not run",
                    "Duration (s)": round((entry.get("elapsed_ms") or 0) / 1000, 1),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        critique = stage_payload(result, "critique")
        analysis = stage_payload(result, "analysis")
        rag = stage_payload(result, "rag")
        vision = stage_payload(result, "vision")
        total_ms = result.get("elapsed_ms") or 0
        runtime = elapsed_seconds or (total_ms / 1000 if total_ms else 0)
        st.write(f"- Pipeline status: {result.get('status') or 'unknown'}")
        st.write(f"- Critique status: {(critique or {}).get('status') or 'not run'}")
        st.write(
            "- Unverified references: "
            f"{(critique or {}).get('unverified_evidence_refs') or 0}"
        )
        st.write(f"- Embedding model: {(rag or {}).get('embedding_model') or 'n/a'}")
        st.write(f"- Vision model: {(vision or {}).get('model') or 'n/a'}")
        st.write(f"- Analysis model: {(analysis or {}).get('model') or 'n/a'}")
        st.write(f"- Critique model: {(critique or {}).get('model') or 'n/a'}")
        st.write(f"- Total runtime: {runtime:.1f}s")

        for stage, _label in STAGE_LABELS:
            entry = stage_result(result, stage)
            if entry.get("status") == "failed" and entry.get("error"):
                logger.warning(
                    "Stage %s failed: %s / %s",
                    stage,
                    entry.get("error_type"),
                    entry.get("error"),
                )
                st.write(f"- {stage}: {entry.get('error')}")

        warnings = (critique or {}).get("warnings") or []
        if warnings:
            logger.info("Critique produced %d internal note(s).", len(warnings))


def render_results(result, elapsed_seconds=None):
    """Render the analysis in the order a reader needs it."""
    if not result:
        return
    render_final_critique(result)
    render_research_analysis(result)
    render_visual_analysis(result)
    render_textual_evidence(result)
    render_routing(result)
    render_technical_details(result, elapsed_seconds)


# -----------------------------
# Application
# -----------------------------
st.markdown(
    """
    <style>
    .st-emotion-cache-1j22a0y.e1yxiy6j4 {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Research Paper Critic")
st.markdown("#### Understand what the paper says.")
st.write(
    "See the evidence behind its claims, and identify strengths, limitations and "
    "unanswered questions."
)

st.markdown("---")
st.subheader("Upload Research Paper")

with st.form(key="paper_upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Choose a research paper (PDF format)",
        type=["pdf"],
    )
    submitted = st.form_submit_button("Read Paper")

if submitted:
    st.session_state.document_profile = None
    st.session_state.document_profile_filename = None
    st.session_state.analysis_elapsed_seconds = None
    st.session_state.pdf_path = None
    reset_analysis_state()

    if uploaded_file is None:
        st.warning("Please choose a research paper first.")
    else:
        file_path = upload_and_validate(uploaded_file)
        if file_path:
            analyze_uploaded_pdf(file_path)


profile = st.session_state.get("document_profile")

if profile:
    st.markdown("---")
    metadata = profile.get("metadata") or {}
    filename = st.session_state.get("document_profile_filename") or "Uploaded paper"
    st.subheader("Paper")
    st.markdown(f"### {metadata.get('title') or filename}")
    st.caption(
        f"{profile.get('page_count') or 0} pages • "
        f"{profile.get('word_count') or 0:,} words"
    )
    st.markdown("---")

    if st.button(
        "Run Full Analysis",
        type="primary",
        help=(
            "This reads the paper with several analysis tools and produces a "
            "structured critique. It can take a few minutes."
        ),
    ):
        pdf_path = st.session_state.get("pdf_path")
        if not pdf_path:
            st.error("The paper could not be found. Please upload it again.")
        else:
            reset_analysis_state()
            st.session_state.pipeline_progress = {}
            started_at = time.time()
            live_line = {"holder": st.empty()}

            def _progress(stage, status):
                st.session_state.pipeline_progress[stage] = status
                label = dict(STAGE_LABELS).get(stage, stage)
                live_line["holder"].caption(f"{status.capitalize()}: {label}…")

            with st.spinner("Analysing the paper…"):
                try:
                    st.session_state.pipeline_result = run_pipeline(
                        pdf_path, progress=_progress
                    )
                except Exception as exc:  # noqa: BLE001 - UI safety net
                    logger.exception("Pipeline failed: %s", exc)
                    st.session_state.pipeline_result = {
                        "status": "failed",
                        "failed_stage": None,
                        "stages": {},
                    }
                finally:
                    st.session_state.pipeline_elapsed_seconds = time.time() - started_at
                    live_line["holder"].empty()

    render_progress(st.session_state.get("pipeline_progress"))
    render_results(
        st.session_state.get("pipeline_result"),
        st.session_state.get("pipeline_elapsed_seconds"),
    )

else:
    st.info("Choose a research paper in PDF format and press **Read Paper** to begin.")
