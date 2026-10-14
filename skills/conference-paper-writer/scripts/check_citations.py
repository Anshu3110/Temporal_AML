#!/usr/bin/env python3
"""Compare \\cite keys in a .tex file with entries in .bib file(s). Reports undefined and uncited keys."""
import re, sys
from pathlib import Path
if len(sys.argv) < 3: raise SystemExit("usage: check_citations.py main.tex refs.bib [more.bib]")
tex = Path(sys.argv[1]).read_text(errors="ignore")
tex = re.sub(r"(?<!\\)%.*", "", tex)
cited = set()
for m in re.finditer(r"\\(?:cite[a-zA-Z]*)\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
    cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
defined = set()
for b in sys.argv[2:]:
    defined |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", Path(b).read_text(errors="ignore")))
missing, unused = sorted(cited - defined), sorted(defined - cited)
print(f"cited={len(cited)} defined={len(defined)}")
print("UNDEFINED (cited, not in bib):", missing or "none")
print("UNCITED (in bib, never cited):", unused or "none")
sys.exit(1 if missing else 0)
