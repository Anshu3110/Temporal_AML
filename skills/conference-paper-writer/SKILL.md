---
name: conference-paper-writer
description: Turns an existing research codebase into an original, evidence-backed, template-compliant IEEE-style conference paper (page-limited, double-blind aware). Use whenever the user wants to write, draft, format, check, anonymize, or prepare a conference paper or manuscript from a project, mentions ICICF 2027, IEEE conference templates, Microsoft CMT submission, page limits, camera-ready, or asks to verify a paper against conference requirements, even if they do not say "skill".
---

# Conference Paper Writer

Authoritative operating manual. Produces a manuscript for **human review** from three sources of truth:

1. **Project** - code, data, results actually present (what was implemented and what may be reported).
2. **Local `paper/` directory** - supplied templates and guidelines (what format is required).
3. **Official conference website** - current dates, scope, policies (verified live, never from memory).

Reconcile all three. Never invent evidence, never resolve material conflicts by guessing. Never promise acceptance, IEEE Xplore publication, a similarity score, or an AI-detector result.

## Inputs
Required: project root; `paper/` directory; target conference (default profile: `conferences/icicf-2027.md`); page limit (default 6, hard maximum, references and appendices included unless the *official* rules say otherwise); submission stage (initial anonymized / camera-ready).
Ask the author for: names, affiliations, emails, ORCIDs, funding, track preference, prior related publications. Never fabricate these.

## Operating rules
- Non-destructive: never overwrite or delete source, data, results, originals in `paper/`, or user drafts. Work in `paper/output/` (if user files exist there, use `paper/output/v<N>/` or versioned names). Never change functional project code. Check `git status` first; never commit or push without authorization.
- Never submit, register, pay, sign declarations, accept terms, or upload private data to third parties without explicit authorization.
- Label every requirement's origin: (A) official conference, (B) supplied template, (C) IEEE, (D) general academic practice, (E) your own recommendation. Never present E as A.
- Do not claim a check happened if it did not (browsing, rendering, page count, similarity check). State the limitation instead.
- Do not modify the project to make results look better. Missing experiments go in `outstanding_issues.md` with a recommendation; ask before running substantial or costly ones.

## Workflow (run in order; do not stop after an outline)
**Phase 1 - Discovery.** Locate project root, `paper/`, source, datasets, results, notebooks, docs, existing drafts, existing agent instructions/skills, `git status`. Skip venvs, caches, build dirs. Read `references/project_analysis.md`.
**Phase 2 - Local template analysis (mandatory even if the website was checked).** Enumerate every file in `paper/`, including zips (extract copies to a working dir). Read actual contents: `.tex` (documentclass options, packages, macros, bibliography style), `.cls`/`.bst`, PDFs (extract text; render pages to images when layout matters), DOCX styles. Identify the authoritative template; if several or unclear, ask. A generic IEEE template is acceptable only if the conference confirms it. Read `references/template_compliance.md`.
**Phase 3 - Official verification.** Browse the conference site and follow links (CFP, tracks, submission, template, dates, review policy, IEEE Xplore statement, CMT, AI policy, copyright, camera-ready, FAQ). Record URL, title, date checked, requirement, explicit vs inferred, verified or not. Write `paper/output/conference_requirements.md` from `templates/conference_requirements_template.md`. Investigate local-vs-website discrepancies; ask when they affect page limit, anonymity, eligibility or format. If browsing is unavailable, say so and mark everything unverified. Read `references/conference_compliance.md`. Dates in the profile are reference only; re-verify, and if a deadline passed check for extensions/portal status without assuming late acceptance.
**Phase 4 - Project analysis.** Establish problem, gap, objectives, contributions, architecture, datasets, methodology, experiments actually run, limitations, environment. Verify README claims against code and artifacts. Write `paper/output/research_evidence.md` (claim -> source file/line -> status -> limitation -> planned use). Classify each item: verified implementation / verified result / external literature / interpretation / unverified assumption / proposed future work. Also assess novelty honestly (implementation vs application vs integration vs empirical vs methodological vs new algorithm).
**Phase 5 - Literature review.** Read `references/literature_review.md`. Verified sources only; record whether each was read in full or only abstract/metadata.
**Phase 6 - Track selection.** Compare project to current official tracks; justify by research question, method, domain, contribution. Flag misfit or uncertainty rather than distorting the work.
**Phase 7 - Outline.** Follow template structure. Allocate the page budget (see `references/page_limit_validation.md`), plan figures/tables/equations (`references/figure_table_equation_guidelines.md`).
**Phase 8 - Drafting.** Original, precise academic prose; only verified facts; Results state what they do and do not establish. Read `references/academic_integrity.md` first. Work in a copy of the template; strip all sample text.
**Phase 9 - Technical validation.** Cross-check abstract vs results, methods vs code, conclusion vs findings, tables/figures vs source data, citations vs bibliography (`scripts/check_citations.py`), equations and symbols. Check reproducibility via `references/reproducibility.md`.
**Phase 10 - Formatting and rendering.** Compile (`pdflatex` + `bibtex` + `pdflatex` x2, or the template's documented route). Inspect rendered pages visually (rasterize and view) for clipped figures, overfull boxes, broken equations, fonts, columns.
**Phase 11 - Page limit.** `python scripts/check_pages.py <main.tex>` (or `--pdf`). Revise and re-render until within limit. Only permitted compression; never alter fonts, margins, spacing or columns. Report the actual rendered count.
**Phase 12 - Anonymity.** Verify rules for the current stage. `python scripts/check_anonymity.py <pdf> --terms "Name1,Institution,..."`. Inspect metadata, acknowledgments, funding, URLs, self-citations, supplementary files. Keep separate anonymized and camera-ready versions; do not strip identity from every deliverable by default.
**Phase 13 - Reviewer simulation.** Fill `templates/reviewer_assessment_template.md` using `references/review_and_revision.md`.
**Phase 14 - Final compliance.** Re-verify any volatile official requirement, complete `compliance_checklist.md`, list blockers in `outstanding_issues.md`, prepare CMT checklist (`references/submission_checklist.md`). Never claim a formal plagiarism check unless one was actually run.

## Outputs (in `paper/output/`)
Manuscript source (`.tex` + `.bib` + figures) and rendered PDF; `conference_requirements.md`; `research_evidence.md`; `compliance_checklist.md`; `reviewer_self_assessment.md`; `outstanding_issues.md`. Create nothing else unless useful.

## Ask the human when
Several templates are equally authoritative; author info missing; submission stage unclear and it changes format; mandatory experiment missing; conflicting results; ambiguous rule that affects compliance; track undeterminable; a claim lacks evidence; the page limit cannot be met without hurting validity; any external or legal action is needed. Ask focused questions, keep working on unrelated parts, and log the open item. Do not interrupt for minor style choices the template settles.

## Failure handling
Tool missing (LaTeX, PDF tools, browser): use an accurate alternative or stop that step; mark it unverified. Inaccessible page: log it, do not bypass access controls. Conflicting values in files: investigate, then flag for the author. Compile errors: fix in the working copy only.

## Final status vocabulary
Report the highest level actually reached, never higher: drafted -> technically verified -> formatting verified -> within page limit -> conference compliance verified -> ready for author review -> ready for submission subject to author approval.

## Reference map
`references/`: academic_integrity (before drafting), project_analysis (phase 1/4), conference_compliance (phase 3/14), template_compliance (phase 2/10), page_limit_validation (phase 7/11), literature_review (phase 5), reproducibility (phase 9), figure_table_equation_guidelines (phase 7/8/10), submission_checklist (phase 12/14), review_and_revision (phase 13 and post-review).
`templates/`: fill-in forms for the five output documents. `conferences/`: per-conference profiles (add new files for other venues). `scripts/`: optional helpers; run them explicitly, they never run automatically.
