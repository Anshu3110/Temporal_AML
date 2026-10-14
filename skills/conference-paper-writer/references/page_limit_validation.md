# Page limit (default: 6 pages, hard)
Counts title/authors, abstract, text, figures, tables, equations, acknowledgments, references, appendices. Do not exclude references/appendices unless official rules say so; verify the current official policy (extra-page fee rules do not raise the user's maximum).
Plan budget before drafting (guide for a 6-page two-column paper): intro+related ~1.0, problem+method ~1.5-2.0, setup+results+discussion ~2.0, conclusion ~0.3, references ~0.6. Spend on contribution, method, evidence.
Loop: render -> count (scripts/check_pages.py) -> inspect last page -> compress -> re-render, until within limit.
Allowed: cut redundancy, merge paragraphs, tighten related work, drop nonessential figures/tables, shorten captions, remove uncited references, better wording.
Forbidden: smaller fonts, changed margins/line spacing/column widths/\vspace hacks, removing required declarations, dropping essential citations, methods details or limitations, misleading summaries.
If it cannot fit without hurting validity, stop and ask. Report the actual rendered page count only if it was measured.
