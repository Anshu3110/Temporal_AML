---
name: paraphrase-humanize
description: Paraphrase, humanize, rewrite, and remove AI-detection and plagiarism markers from any text, academic paper, thesis, report, or essay. Triggers on requests like "paraphrase this", "quillbot", "humanize text", "reduce plagiarism", "bypass AI detector", "lower turnitin score", "make it sound human", "rewrite in academic tone", or "rephrase this document". Supports multiple rewriting modes (Academic, Standard, Fluency, Formal, Simple, Shorten, Expand).
---

# Paraphrase & Humanize (QuillBot & AI-Bypasser Skill)

A comprehensive skill for intelligent rewriting, tone shifting, plagiarism reduction, and AI content humanization while preserving factual precision, LaTeX/Markdown formatting, equations, and citations.

---

## Core Capabilities

1. **Multi-Mode Paraphrasing** (QuillBot equivalent modes: *Academic*, *Standard*, *Fluency*, *Formal*, *Simple*, *Shorten*, *Expand*).
2. **AI Content Removal & Humanizer** (Maximizes burstiness and perplexity to pass Turnitin AI, GPTZero, Copyleaks, and ZeroGPT).
3. **Plagiarism & Similarity Index Reducer** (Breaks 4+ n-gram matches, inverts sentence clauses, synthesizes literature thematically).
4. **Format & Technical Integrity Preservation** (Preserves LaTeX syntax `\cite{}`, `\ref{}`, `\begin{}...\end{}`, math equations `$..$`, Markdown tables, and code).

---

## Step 1: Mode Selection & Tone Calibration

Identify the desired rewriting mode based on user prompt or document context:

| Mode | Purpose & Tone | Key Transformations |
| :--- | :--- | :--- |
| **Academic** *(Default for theses/papers)* | Scholarly, precise, rigorous | Replaces conversational terms with scholarly equivalents; converts passive/weak prose to active analytical synthesis; preserves all citations and math. |
| **Humanize / AI-Bypass** | Eliminates robotic patterns | Modulates burstiness (sentence length variation), disrupts uniform rhythm, strips AI clichés, and injects organic sentence structures. |
| **Fluency** | Polished, natural English | Eliminates awkward phrasing, fixes prepositional errors, improves readability and sentence connection. |
| **Formal / Corporate** | Professional and direct | Crisp business vocabulary, eliminates colloquialisms, ensures confident and authoritative delivery. |
| **Standard** | Balanced rewrite | Moderate paraphrasing maintaining original tone, length, and vocabulary level. |
| **Shorten / Condense** | High information density | Strips fluff, redundancies, and filler words; retains 100% of core factual assertions. |
| **Expand / Elaborate** | Deep exposition | Unpacks dense arguments, adds contextual bridges, clarifies technical implications without repetitive fluff. |

---

## Step 2: The Humanizer & Anti-AI Engine

AI detectors (Turnitin AI, GPTZero, Copyleaks, Originality.ai) measure two core statistical properties:
1. **Perplexity**: How predictable each word is in sequence (LLMs choose top-p/greedy statistical next words; humans use idiosyncratic, context-rich phrasing).
2. **Burstiness**: The variation in sentence length and complexity (LLMs produce uniform 15–22 word declarative sentences; humans mix 4-word punches with 35-word multi-clause periodic sentences).

### Anti-AI Rules:

1. **Disrupt Uniform Sentence Length (High Burstiness)**:
   - *Never* write 3 consecutive sentences of similar length.
   - Intentionally alternate: Short sentence (4–8 words) $\to$ Complex compound/periodic sentence (25–35 words) $\to$ Medium analytical sentence (12–18 words).
2. **Strip LLM Telltale Vocabulary**:
   - 🚫 **Banned Words & Phrasings**: *delve, tapestry, testament, beacon, pivotal, crucial role, furthermore, moreover, it is important to note, arguably, in conclusion, game-changer, landscape, multifaceted, foster, underscore, at its core, not only... but also...*
   - ✅ **Human Alternatives**: Use direct action verbs (*"measured"*, *"reveals"*, *"diverges"*, *"imposes"*, *"yields"*, *"requires"*), domain-specific nouns, and natural connectors (*"however"*, *"yet"*, *"in practice"*, *"specifically"*, *"because"*).
3. **Vary Clause Architecture**:
   - Invert dependent and independent clauses (e.g., instead of *"We use X to do Y because of Z"*, use *"Because of Z, deploying X achieves Y"* or *"To achieve Y under Z constraints, X is deployed"*).
   - Use periodic sentences where the main point arrives at the end.
   - Break symmetric parallel lists (*"First... Second... Third..."* or *"X did A, Y did B, Z did C"*).

---

## Step 3: Plagiarism & Similarity Index Reduction

To reduce similarity in Turnitin, DrillBit, and iThenticate:

1. **Break 4+ N-Gram Matches**:
   - Turnitin flags identical strings of 4–8 words.
   - Never keep the original 4-word verb-noun collocations (e.g., change *"the rapid adoption of cryptocurrency networks has fundamentally altered"* $\to$ *"as cryptocurrency networks expanded globally, digital asset settlement altered..."*).
2. **Thematic Literature Synthesis (No Author Enumeration)**:
   - 🚫 *Avoid Author-by-Author Lists*: *"Weber et al. [1] did X. Then Pareja et al. [2] did Y. Later Xu et al. [3] proposed Z."*
   - ✅ *Use Thematic Synthesis*: *"Early cryptocurrency surveillance models relied on static graph aggregations [1], leaving temporal dependencies uncaptured. Subsequent efforts introduced discrete snapshot updates [2], though sub-interval fund velocity remained undetected until continuous attention formulations emerged [3]."*
3. **Table & Definition Paraphrasing**:
   - In literature tables, avoid copying full published paper titles. Use concise thematic descriptors (e.g., *"Bitcoin GNN Baselines"* instead of *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks"*).
   - Rephrase standard definitions (e.g., UTXO, Blockchain, GNN, ROC-AUC) to explain their *methodological role in your specific pipeline* rather than quoting textbook glossaries.

---

## Step 4: Technical & LaTeX Preservation Rules

When rewriting `.tex` or `.md` technical documents:

1. **Strictly Preserve Formatting Syntax**:
   - LaTeX citations: `\cite{...}`, `\citep{...}`, `\citet{...}`
   - Cross-references: `\ref{...}`, `\label{...}`, `\pageref{...}`
   - Math environments: `$x_i$`, `\[ ... \]`, `\begin{equation}...\end{equation}`
   - Structural commands: `\chapter{...}`, `\section{...}`, `\begin{figure}...\end{figure}`, `\begin{table}...\end{table}`
   - Code blocks and variable names: `\texttt{...}`, `\textbf{...}`
2. **Do Not Alter Mathematical Formulas**: Keep variable notation ($h_i^{(l)}$, $\alpha_{ij}$, $\boldsymbol{\omega}$, $\Delta t$) identical; paraphrase only the prose surrounding the equations.
3. **Retain Factual Precision**: Do not change dataset statistics, numbers, experimental results, author citations, or benchmark metrics.

---

## Quick Execution Workflow

When the user asks to paraphrase, humanize, or reduce plagiarism:

1. **Assess the Input**: Identify the document type (LaTeX chapter, abstract, paragraph, essay), target audience, and current similarity/AI risks.
2. **Apply the Transformation**:
   - Select mode (*Academic*, *Humanize*, etc.).
   - Rewrite with varied sentence lengths (burstiness) and unique vocabulary (perplexity).
   - Eliminate repetitive transition words and verbatim n-grams.
3. **Verify Integrity**: Confirm that all citations, numbers, equations, and LaTeX environments remain intact.
4. **Present the Output**: Display the polished text cleanly, highlighting key improvements made in tone, similarity reduction, and AI avoidance.
