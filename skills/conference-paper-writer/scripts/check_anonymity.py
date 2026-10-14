#!/usr/bin/env python3
"""Check a PDF for identifying info: metadata fields and occurrences of given terms (names, institutions, emails, URLs) in the text. Does not prove anonymity; review manually."""
import argparse, re, subprocess
ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--terms", default="")
a = ap.parse_args()
try:
    from pypdf import PdfReader
    meta = PdfReader(a.pdf).metadata or {}
    print("METADATA:", {k: str(v) for k, v in meta.items()} or "none")
except ImportError:
    print(subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True).stdout)
text = subprocess.run(["pdftotext", a.pdf, "-"], capture_output=True, text=True).stdout
flag = 0
for t in [x.strip() for x in a.terms.split(",") if x.strip()]:
    n = len(re.findall(re.escape(t), text, re.I))
    print(f"term '{t}': {n} hit(s)"); flag += n > 0
for pat, label in [(r"[\w.+-]+@[\w-]+\.[\w.]+", "email"), (r"https?://\S+", "URL"), (r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", "ORCID")]:
    hits = sorted(set(re.findall(pat, text)))
    if hits: print(f"{label}s found: {hits[:10]}"); flag += 1
print("REVIEW NEEDED" if flag else "No flagged items (still check acknowledgments, funding, self-citations manually)")
