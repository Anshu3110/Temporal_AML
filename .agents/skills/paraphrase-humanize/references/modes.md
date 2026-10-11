# Paraphrasing Modes Reference

This document provides detailed instructions, stylistic rules, and transformation patterns for each paraphrasing mode.

---

## 1. Academic Mode
- **Primary Use**: Theses, dissertations, peer-reviewed journal papers, conference submissions, research grants.
- **Tone**: Formal, analytical, objective, intellectually precise.
- **Key Techniques**:
  - Replace vague qualifiers (*"a lot of"*, *"good results"*) with precise scholarly terms (*"substantial"*, *"statistically significant gains"*).
  - Use active synthesis verbs: *demonstrates, delineates, isolates, mitigates, substantiates, exemplifies, formulates*.
  - Convert disjointed author summaries into comparative thematic analysis.
  - Preserve all mathematical notation, LaTeX symbols, and citations (`\cite{...}`).

---

## 2. Humanize / Anti-AI Detection Mode
- **Primary Use**: Bypassing AI content detectors (Turnitin AI, GPTZero, Copyleaks, ZeroGPT, Crossplag).
- **Core Engine**:
  - **High Burstiness**: Mix short sentences (3–7 words) with compound-complex sentences (25–35 words).
  - **High Perplexity**: Choose unexpected yet natural phrasing rather than high-probability LLM token pairs.
  - **Syntax Inversion**: Place conditional clauses, prepositional phrases, or adverbial phrases at the start of sentences.
  - **Remove AI Vocabulary**: Strip *delve, testament, beacon, pivotal, crucial, furthermore, moreover, multifaceted, landscape*.

---

## 3. Fluency Mode
- **Primary Use**: ESL writing correction, editing rough drafts, improving readability.
- **Tone**: Smooth, idiomatic, effortless flow.
- **Key Techniques**:
  - Fix awkward prepositions and non-idiomatic expressions.
  - Eliminate run-on sentences and excessive comma splices.
  - Optimize transition flow without overusing heavy formal conjunctive adverbs.

---

## 4. Formal Mode
- **Primary Use**: Business proposals, executive summaries, corporate compliance reports, official correspondence.
- **Tone**: Professional, authoritative, concise, confident.
- **Key Techniques**:
  - Eliminate slang, contractions, and casual idioms.
  - Structure paragraphs around direct topic sentences with logical supporting evidence.

---

## 5. Shorten / Condense Mode
- **Primary Use**: Abstracts with strict word limits, executive summaries, slide bullet points.
- **Key Techniques**:
  - Remove empty nominalizations (e.g., change *"the measurement of the parameters was performed"* to *"parameters were measured"*).
  - Strip redundant modifiers (e.g., change *"completely eliminated"* to *"eliminated"*, *"future plans for the future"* to *"future plans"*).
  - Maximize semantic density while preserving 100% of facts.

---

## 6. Expand / Elaborate Mode
- **Primary Use**: Unpacking dense methodologies, answering reviewer critiques, adding thorough technical justification.
- **Key Techniques**:
  - Articulate the underlying rationale behind architectural or algorithmic decisions.
  - Explain the practical real-world implications of experimental findings.
  - Provide intermediate logical steps without padding with filler words.
