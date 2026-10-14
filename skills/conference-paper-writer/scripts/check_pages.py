#!/usr/bin/env python3
"""Compile a LaTeX file (pdflatex+bibtex+pdflatex x2) in its own directory, or take --pdf, and report the rendered page count vs a limit. Exit 1 if over."""
import argparse, subprocess, sys, shutil
from pathlib import Path

def pages(pdf):
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except ImportError:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        for l in out.splitlines():
            if l.startswith("Pages:"): return int(l.split()[1])
    raise SystemExit("Cannot count pages: install pypdf or poppler-utils")

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

ap = argparse.ArgumentParser()
ap.add_argument("tex", nargs="?"); ap.add_argument("--pdf"); ap.add_argument("--limit", type=int, default=6)
a = ap.parse_args()
if a.pdf: pdf = Path(a.pdf)
else:
    if not a.tex or not shutil.which("pdflatex"): raise SystemExit("Need a .tex file and pdflatex (or use --pdf); page count NOT verified")
    t = Path(a.tex).resolve(); d, s = t.parent, t.stem
    run(["pdflatex", "-interaction=nonstopmode", t.name], d)
    run(["bibtex", s], d)
    run(["pdflatex", "-interaction=nonstopmode", t.name], d)
    r = run(["pdflatex", "-interaction=nonstopmode", t.name], d)
    pdf = d / f"{s}.pdf"
    if not pdf.exists(): print(r.stdout[-1500:]); raise SystemExit("Compilation failed")
n = pages(pdf)
print(f"{pdf}: {n} pages (limit {a.limit}) -> {'OK' if n <= a.limit else 'OVER LIMIT'}")
sys.exit(0 if n <= a.limit else 1)
