# Conference Requirements: ICICF 2027

**Checked on:** 2026-10-14  
**Target Venue:** International Conference on Intelligent Computing Frontiers (ICICF 2027)  
**Host:** JAIN (Deemed-to-be University), Faculty of Engineering and Technology, Bengaluru, India  
**Official Website:** https://set.jainuniversity.ac.in/icicf/  
**IEEE Conference Record:** #72107  
**Submission Portal:** Microsoft CMT (https://cmt3.research.microsoft.com/ICICF2027/Submission/Index)  

---

## 1. Requirements Matrix

| Requirement | Value | Source URL / Document | Origin | Explicit / Inferred | Verified (Y/N) |
|---|---|---|---|---|---|
| **Submission Deadline** | 15th October 2026 (Tomorrow) | Official Site (`#dates`) | A | Explicit | Y |
| **Notification Date** | 1st December 2026 | Official Site (`#dates`) | A | Explicit | Y |
| **Camera-Ready & Reg.** | 15th December 2026 | Official Site (`#dates`) | A | Explicit | Y |
| **Conference Dates** | 21st–23rd January 2027 | Official Site (`#dates`) | A | Explicit | Y |
| **Submission Platform** | Microsoft CMT (`ICICF2027`) | Official Site (`#submission`) | A | Explicit | Y (HTTP 302 active) |
| **Review Mode** | Double-blind (initial submission) | Official Site (`#submission`) | A | Explicit | Y |
| **Page Limit (Base)** | 6 pages maximum (including references) | Official Site (`#registration`) | A | Explicit | Y |
| **Extra Page Policy** | Permitted with fee ($30 / Rs. 800 per page) | Official Site (`#registration`) | A | Explicit | Y |
| **Target Page Limit** | 6 pages hard cap (no paid over-length) | User Requirement | E | Explicit | Y |
| **Template Format** | IEEE Conference 2-column, A4 format | Official Template PDF | A, B | Explicit | Y |
| **Presentation Mode** | Strictly physical mode in Bengaluru (no virtual) | Official Site (`#submission`) | A | Explicit | Y |
| **Publication Statement** | "Submitted for possible inclusion in IEEE Xplore" | Official Site (`#submission`) | A | Explicit | Y (No guarantee) |
| **Plagiarism Policy** | Original work only; zero tolerance | Official Site (`#submission`) | A | Explicit | Y |
| **Selected Track** | Track 3: Next Generation AI for Data, Vision & Language | Official Site (`#tracks`) | A, E | Inferred | Y |
| **Author Anonymity** | Completely anonymized PDF + metadata in CMT | Official Site + User Req | A, D | Explicit | Y |

*Origin Key: (A) Official conference website; (B) Supplied template; (C) IEEE standard; (D) General academic practice; (E) Author/Agent choice.*

---

## 2. Dates and Deadline Assessment

- **Current Date:** 2026-10-14 19:20 IST.
- **Official Submission Deadline:** 15th October 2026.
- **Portal Status:** Active on Microsoft CMT (`https://cmt3.research.microsoft.com/ICICF2027/Submission/Index`).
- **Deadline Extension:** No extension is posted on the official homepage as of 2026-10-14.
- **Urgency Warning:** Initial submission deadline is tomorrow (2026-10-15).

---

## 3. Track Selection and Rationale

The conference offers three tracks:
1. **Track 1: Frontiers in Cognitive and Autonomous Systems** (Cognitive architectures, autonomous agents, human-AI collaboration).
2. **Track 2: Information Theory & Intelligent Communication Systems** (Information theory, coding theory, IoT transmission, signal processing).
3. **Track 3: Next Generation AI for Data, Vision & Language** (Multimodal learning, foundation models, computer vision, NLP, *Bias, fairness, and interpretability in large-scale AI models*).

**Recommendation:** **Track 3** (Next Generation AI for Data, Vision & Language).  
**Subtopic:** *Bias, fairness, and interpretability in large-scale AI models* / *AI for Data*.  
**Fit Rationale & Honest Caveat:**  
- TemporalAML applies graph neural networks and GNNExplainer to large-scale financial transaction data (Elliptic Bitcoin graph with 203k nodes and 234k edges) to produce auditable regulatory explanations (SAR narratives). This aligns directly with the "AI for Data" and "Interpretability in AI models" scope of Track 3.
- *Caveat:* The fit is moderate because ICICF is not a dedicated financial cryptography or security venue (unlike FC, IEEE S&P, or ACM CCS), so emphasis must be placed on the temporal graph representation learning and explainability methodology rather than purely financial regulation.

---

## 4. Template & Formatting Verification

- **Official Template:** Downloaded and visually verified from `https://set.jainuniversity.ac.in/icicf/assets/pdf/paper-submission-template.pdf`. It is the canonical IEEE conference proceedings format:
  - Document class: `IEEEtran`, 2-column, 10pt font, A4 paper geometry.
  - Heading style: Roman numerals for primary sections (`I. INTRODUCTION`), letters for subsections (`A. Subsection`).
  - Table heads positioned *above* tables; figure captions positioned *below* figures.
  - Reference style: numbered in brackets `[1]`, IEEEtran BibTeX style.
- **Local Template Discrepancy:** The local directory `paper/` contains only `manuscript.md` (no zip archives or `.tex` files were provided in `paper/`). The canonical `IEEEtran.cls` and `IEEEtran.bst` must be supplied to compile the LaTeX source.
- **Anonymity Implementation:** The manuscript source will omit author names, affiliations, emails, funding acknowledgments, and Christians (Christ University) institutional disclosures, referring to the implementation in third-person neutral phrasing.

---

## 5. Inaccessible Pages and Limitations

- CMT requires user authentication to inspect internal CFP forms or track-specific submission checklists.
- No other inaccessible pages were encountered; all conference landing pages, tracks, leadership committees, and registration fee schedules were browsed and verified directly.
