"""
Run full Pre-Analyzer pipeline on test paper and output summary & JSON profile.
"""

import json
import time
from backend.document_pre_analyzer.analyzer import analyze_document
from backend.document_pre_analyzer.config import PreAnalyzerConfig

pdf_path = "uploaded_files/NIPS-2017-attention-is-all-you-need-Paper.pdf"
print("Analyzing document with full pipeline (PyMuPDF + Docling layout integration)...")
t0 = time.time()
profile = analyze_document(pdf_path)
t1 = time.time()
print(f"Analysis completed in {t1 - t0:.2f}s!")

print("\n==================================================")
print("HUMAN READABLE SUMMARY")
print("==================================================")
print("Filename:", profile["filename"])
print("Page Count:", profile["page_count"])
print("Text Length:", profile["text_length"], "chars")
print("Word Count:", profile["word_count"], "words")
print("Metadata Title:", profile["metadata"].get("title"))

print(f"\nSections Detected ({len(profile['sections'])}):")
for s in profile["sections"]:
    print(f"  - Page {s['page']} | Level {s['level']} | {s['title']} (conf: {s['confidence']:.2f})")

print(f"\nFigures Detected ({profile['figures']['count']} confirmed, "
      f"{profile['figures']['candidate_count']} candidates):")
for f in profile["figures"]["items"]:
    cap = f["caption"][:80] if f.get("caption") else ""
    print(f"  - {f['id']} | Page {f['page']} | Type: {f['type']} | Conf: {f['confidence']:.2f}")
    print(f"    Caption: {cap}...")
_rejected_figures = profile["figures"]["rejected_candidates"]
if _rejected_figures:
    print(f"  Rejected figure candidates ({len(_rejected_figures)}):")
    for f in _rejected_figures:
        print(f"    ! Page {f['page']} | {f['caption'][:70]} | conf {f['confidence']:.2f}")
print(f"  In-text figure mentions not treated as captions: "
      f"{profile['figures']['rejected_figure_mention_count']}")

print(f"\nTables Detected ({profile['tables']['count']} confirmed, "
      f"{profile['tables']['candidate_count']} candidates):")
for t in profile["tables"]["items"]:
    cap = t["caption"][:80] if t.get("caption") else ""
    print(f"  - {t['id']} | Page {t['page']} | Rows: {t['rows']} | Cols: {t['columns']} "
          f"| Source: {t['structure_source']} | Conf: {t['confidence']:.2f}")
    print(f"    Caption: {cap}...")
_rejected_tables = profile["tables"]["rejected_candidates"]
if _rejected_tables:
    print(f"  Rejected table candidates ({len(_rejected_tables)}):")
    for t in _rejected_tables:
        print(f"    ! Page {t['page']} | {t['caption'][:70]} | conf {t['confidence']:.2f}")
print(f"  In-text table mentions not treated as captions: "
      f"{profile['tables']['rejected_table_mention_count']}")

print(f"\nEquations Detected ({profile['equations']['count']} confirmed, "
      f"{profile['equations']['candidate_count']} candidates):")
for e in profile["equations"]["items"]:
    rep = e.get("representation", "")[:80]
    print(f"  - {e['id']} | Page {e['page']} | Number: {e.get('equation_number')} "
          f"| Source: {e.get('structure_source')} | Conf: {e['confidence']:.2f}")
    print(f"    Rep: {rep}")
_rejected_equations = profile["equations"]["rejected_candidates"]
if _rejected_equations:
    print(f"  Rejected equation candidates ({len(_rejected_equations)}):")
    for e in _rejected_equations:
        print(f"    ! Page {e['page']} | {e['representation'][:70]} | conf {e['confidence']:.2f}")


print("\nReferences:")
ref = profile["references"]
print("  - Has References:", ref["has_references"])
print("  - Count:", ref["count"])
print("  - Start Page:", ref["start_page"])
print("  - Sequential:", ref["is_sequential"])
print(f"  - Confidence: {ref['confidence']:.2f}")

print("\nContent Characteristics:")
for k, v in profile["content_characteristics"].items():
    print(f"  - {k}: {v}")

print("\n==================================================")
print("JSON DOCUMENT PROFILE (First 2000 chars preview)")
print("==================================================")
json_str = json.dumps(profile, indent=2)
print(json_str[:2000])
print("\n... [truncated for console display; total JSON length:", len(json_str), "chars]")
