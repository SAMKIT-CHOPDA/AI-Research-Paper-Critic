"""
End-to-end verification of the Streamlit frontend <-> Document Pre-Analyzer link.

The real Streamlit script is executed through ``streamlit.testing.v1.AppTest``:

1. the app boots and renders the upload form
2. submitting without a file shows a warning (no analyzer call)
3. invalid, empty, corrupted and unreadable PDFs produce friendly errors only
4. uploading the benchmark paper shows only simple paper metadata (title, pages,
    words) and the analysis button - no internal document-profile output
5. the paper metadata survives a plain rerun (i.e. interacting with the page)

The internal Document Profile JSON and pre-analyzer diagnostics are intentionally
NOT part of the user interface, so this script checks that they stay hidden.

Usage (from the repository root)::

    python verify_frontend_integration.py

Exit code 0 means every check passed. The real analyzer (PyMuPDF + Docling) is
executed, so the run takes about a minute; the benchmark expectations below match
the verified "Attention Is All You Need" baseline.
"""

import json
import os
import shutil
import sys
import tempfile

from streamlit.testing.v1 import AppTest

BENCHMARK_PDF = "uploaded_files/NIPS-2017-attention-is-all-you-need-Paper.pdf"

EXPECTED = {
    "page_count": 11,
    "sections": 23,
    "figures": 2,
    "tables": 3,
    "equations": 5,
    "references": 32,
}


def write_fixtures(directory):
    """Create the malformed PDF inputs used by the error-handling checks."""
    with open(BENCHMARK_PDF, "rb") as fh:
        real_pdf_bytes = fh.read()

    fixtures = {
        "not_a_pdf.pdf": b"hello, I am not a pdf at all",
        "empty.pdf": b"",
        "truncated.pdf": real_pdf_bytes[:4000],
        "no_pages.pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>",
    }

    paths = {}
    for name, content in fixtures.items():
        path = os.path.join(directory, name)
        with open(path, "wb") as fh:
            fh.write(content)
        paths[name] = path
    return paths


def all_text(app):
    """Every human-readable string currently rendered by the app."""
    chunks = []
    for attribute in ("markdown", "text", "caption", "error", "warning", "info",
                      "success", "subheader", "title", "header", "code"):
        for element in getattr(app, attribute, []) or []:
            value = getattr(element, "value", None)
            if isinstance(value, str):
                chunks.append(value)
    for element in app.exception or []:
        chunks.append(str(getattr(element, "value", element)))
    return "\n".join(chunks)


def check(results, name, condition, detail=""):
    results.append("[%s] %s%s" % ("PASS" if condition else "FAIL", name,
                                  (" :: " + detail) if detail else ""))


def upload_and_submit(app, path):
    """Simulate a real upload followed by pressing the Submit button."""
    with open(path, "rb") as fh:
        data = fh.read()
    app.file_uploader[0].set_value((os.path.basename(path), data, "application/pdf")).run()
    app.button[0].click().run()


def run_checks(results, fixtures):
    """Full integration walkthrough; returns the rendered app at the end."""
    app = AppTest.from_file("frontend.py", default_timeout=600)
    app.run()

    # 1. Boot
    check(results, "app renders without exception", not app.exception, str(app.exception))
    check(results, "initial guidance shown",
          "research paper in PDF format" in all_text(app))
    check(results, "file uploader present", len(app.file_uploader) == 1)
    check(results, "submit button present", len(app.button) == 1)

    # 2. No file uploaded
    app.button[0].click().run()
    check(results, "no-file submit warns and calls no analyzer",
          any("Please choose a research paper" in w.value for w in app.warning),
          str([w.value for w in app.warning]))
    check(results, "no-file submit raises no exception", not app.exception, str(app.exception))

    # 3. Not a PDF at all
    upload_and_submit(app, fixtures["not_a_pdf.pdf"])
    check(results, "invalid (non-PDF) file shows error",
          any("not a valid PDF" in e.value for e in app.error),
          str([e.value for e in app.error]))
    check(results, "invalid file raises no exception", not app.exception, str(app.exception))

    # 4. Empty file
    upload_and_submit(app, fixtures["empty.pdf"])
    check(results, "empty file shows error",
          any("empty (0 bytes)" in e.value for e in app.error),
          str([e.value for e in app.error]))

    # 5. Corrupted file (valid header, broken body -> pymupdf.FileDataError)
    upload_and_submit(app, fixtures["truncated.pdf"])
    check(results, "corrupted file shows friendly analyzer error",
          any("corrupted, truncated or password protected" in e.value for e in app.error),
          str([e.value for e in app.error]))
    check(results, "corrupted file raises no exception", not app.exception, str(app.exception))

    # 6. Opens but contains no readable page
    upload_and_submit(app, fixtures["no_pages.pdf"])
    check(results, "unreadable PDF (0 pages) shows error",
          any("does not contain any readable pages" in e.value for e in app.error),
          str([e.value for e in app.error]))
    check(results, "failed uploads never show paper metadata",
          not any(s.value == "Paper" for s in app.subheader),
          str([s.value for s in app.subheader]))

    # 7. Real benchmark paper
    upload_and_submit(app, BENCHMARK_PDF)
    text = all_text(app)
    check(results, "benchmark PDF analysed without exception", not app.exception,
          str(app.exception))
    check(results, "paper metadata section shown",
          any(s.value == "Paper" for s in app.subheader),
          str([s.value for s in app.subheader]))
    check(results, "paper title rendered", "Attention is All you Need" in text)
    check(results, "page and word count rendered",
          "11 pages" in text and "4,990 words" in text, )
    check(results, "analysis button present",
          any(b.label == "Run Full Analysis" for b in app.button),
          str([b.label for b in app.button]))
    check(results, "no traceback rendered in the UI", "Traceback" not in text)

    # The pre-analyzer stays internal: none of its raw output is user facing.
    check(results, "no Document Overview section", "Document Overview" not in text)
    check(results, "no Document Profile section", "Document Profile" not in text)
    check(results, "no raw document profile JSON", len(app.json) == 0,
          "json elements: %d" % len(app.json))
    check(results, "no page analysis diagnostics", "Page Analysis" not in text)
    check(results, "no content-characteristics diagnostics",
          "Document nature" not in text)
    check(results, "no validation warnings section",
          "Validation warnings" not in text)
    check(results, "no run information section", "Run Information" not in text)
    check(results, "no backend component names in the UI",
          not any(name in text for name in
                  ("RAG Agent", "Vision Agent", "Analysis Agent",
                   "Critique Engine", "Document Pre-Analyzer")))
    check(results, "no model names before analysis",
          "gpt-4" not in text and "text-embedding" not in text)

    # 8. Plain rerun (what happens when the page is interacted with)
    app.run()
    check(results, "paper metadata survives a plain rerun",
          any(s.value == "Paper" for s in app.subheader),
          str([s.value for s in app.subheader]))

    # 9. The interface must not teach generic research-paper terminology; it
    #    explains the actual paper once an analysis has been run.
    check(results, "no generic paper-structure guide",
          "How to Read a Research Paper" not in text)
    check(results, "no generic 'what does this mean' boxes",
          "What does this mean?" not in text)
    check(results, "no generic example blocks", "Example:" not in text)

    return app, text


def main():
    fixture_dir = tempfile.mkdtemp(prefix="pre_analyzer_frontend_fixtures_")
    results = []
    rendered_summary = ""
    try:
        _, rendered_summary = run_checks(results, write_fixtures(fixture_dir))
    finally:
        shutil.rmtree(fixture_dir, ignore_errors=True)

    for line in results:
        print(line)

    failures = [line for line in results if line.startswith("[FAIL]")]
    print("\n%d checks, %d failed" % (len(results), len(failures)))

    if "--dump-summary" in sys.argv:
        print("\n----- RENDERED DOCUMENT PROFILE SUMMARY -----")
        print(rendered_summary)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
