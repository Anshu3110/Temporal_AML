# Research Evidence: TemporalAML Project Analysis

**Document Status:** Complete Audit of Codebase, Experimental Artifacts, and Phase-1 Dissertation Report  
**Target Submission:** ICICF 2027 (Double-Blind Conference Track)  
**Operating Standard:** Reconcile project code and raw artifacts as the sole source of empirical truth.

---

## 1. Problem Formulation, Objectives, and Novelty Assessment

### 1.1 Problem & Scope
- **Domain:** Forensic cryptocurrency Anti-Money Laundering (AML) on the public Elliptic Bitcoin transaction graph (203,769 nodes, 234,355 directed payment edges).
- **Core Challenge:** Detecting rare illicit transaction sub-flows under extreme class skew (~2.2% illicit), structural non-stationarity, and translating complex graph predictions into legally auditable Suspicious Activity Report (SAR) narratives for compliance officers.

### 1.2 Objectives
1. **O1 (Graph Construction & Preprocessing):** Ingest raw Elliptic transaction tables, construct contiguous 0-indexed PyG `Data` structures, enforce strict chronological train/val/test splits ($t \le 34$, $35 \le t \le 40$, $41 \le t \le 49$), and fit normalizers without lookahead.
2. **O2 (Learnable Fourier Time Encoding):** Formulate continuous harmonic time projections $\Phi(\Delta t)$ parameterized by trainable frequencies $\boldsymbol{\omega}$ inside TGAT attention to capture arrival dynamics.
3. **O3 (Multi-Pattern Detection Backbone):** Train a shared TGAT encoder with multiple specialized heads for canonical money laundering typologies (peeling chains, smurfing fan-out, circular loops).
4. **O4 (Auditable Subgraph XAI & SAR):** Deploy GNNExplainer to extract chronological evidence subgraphs and populate standardized FinCEN SAR narrative templates.

### 1.3 Honest Novelty Assessment
- **Novelty Classification:** **Methodological Integration and Domain Application with Critical Empirical Findings.**
  - *New Algorithm?* **No.** TGAT (Xu et al., ICLR 2020), Time2Vec (Kazemi et al., 2019), and GNNExplainer (Ying et al., NeurIPS 2019) are established architectures.
  - *Novel Integration?* **Yes.** Combining trainable harmonic frequencies directly inside multi-head temporal graph attention with multi-typology classification heads and automated SAR JSON compilation is a cohesive architectural integration.
  - *Empirical/Critical Contribution?* **Yes, and of high scientific value.** The empirical investigation uncovers two fundamental structural realities that challenge common assumptions in cryptocurrency AML literature:
    1. **The Intrinsic $\Delta t = 0$ Bottleneck of the Elliptic Dataset:** In the raw Elliptic benchmark, 100% of the 234,355 directed edges connect nodes within the *exact same* discrete two-week time bucket ($t_{\text{src}} = t_{\text{dst}}$). Consequently, $\Delta t = 0$ across all edges, which mathematically collapses the continuous Fourier encoding $\Phi(\Delta t)$ to a constant static bias $\sqrt{2/d_{\text{time}}}[1,\dots,0,\dots]$ and nullifies frequency gradients ($\partial \mathcal{L} / \partial \boldsymbol{\omega} = \mathbf{0}$). This formally explains why temporal attention models underperform static GNNs on this benchmark.
    2. **Bitcoin UTXO DAG Constraints on Typology Mining:** Because raw Bitcoin transactions operate on an unspent transaction output (UTXO) model, the raw transaction graph is topologically an acyclic directed graph (DAG). Directed cycles (circular laundering) are topologically impossible at the transaction level without address-level entity resolution. Furthermore, heuristic pattern mining for smurfing (degree $\ge 10$) creates circular label leakage when degree metrics are present in the feature matrix.

---

## 2. Research Evidence Matrix

| ID | Claim or Finding | Type | Source File & Line | Verified? | Limitation & Empirical Reality | Planned Use in Paper |
|---|---|---|---|---|---|---|
| **E01** | Zero temporal lookahead in graph splitting | Verified Impl | `src/data_loader.py:131-161`, `tests/test_no_leakage.py:98-175` | Yes | Train ($t \le 34$), Val ($35 \le t \le 40$), Test ($41 \le t \le 49$). Normalizer fitted strictly on $t \le 34$. Verified by automated pytest unit tests. | Sec. III (Methodology: Chronological Splitting) |
| **E02** | Feature dimensionality discrepancy (165 vs 171 vs 172) | Verified Impl | `data/processed/elliptic_graph.pt`, `src/data_loader.py:68-72`, `data/processed/elliptic_processed.pt` | Yes | `elliptic_graph.pt` (used in training/eval) has **165 features** (167 raw cols minus txId and time_step). `elliptic_processed.pt` has 171 features. Notebooks define 172 features (+6 engineered). **Source of truth for training is 165 dims.** | Sec. III & IV (Experimental Setup: Feature Taxonomies) |
| **E03** | Learnable Fourier Time Encoder parameterization | Verified Impl | `src/models.py:81-130` | Yes | Trainable parameter $\boldsymbol{\omega} \in \mathbb{R}^{d/2}$, scaling $\sqrt{2/d}$. Mathematically correct implementation of Bochner kernel projection. | Sec. III (Model Architecture: Temporal Attention) |
| **E04** | Non-zero gradient flow through Fourier parameters | Verified Impl | `tests/test_no_leakage.py:340-398` | Yes | **Implementation check only.** Verified that when synthetic $\Delta t > 0$, $\partial \mathcal{L} / \partial \boldsymbol{\omega} \neq \mathbf{0}$. Does *not* imply superior empirical performance on real data. | Sec. III (Implementation Sanity Checks) |
| **E05** | Identical $\Delta t = 0$ on 100% of Elliptic edges | Verified Finding | `data/processed/elliptic_graph.pt`, `task-118` empirical audit | Yes | Every single one of the 234,355 edges in Elliptic connects nodes within the identical time-step bucket ($\Delta t = 0$). Edge-level temporal intervals are absent. | Sec. IV & V (Empirical Results & Forensic Discussion) |
| **E06** | TemporalAML Illicit Detection AUPRC is 0.0376 (below baselines) | Verified Result | `artifacts/logs/final_results.csv:8`, `src/evaluate.py:330-374` | Yes | GraphSAGE achieves **0.2800**, GCN achieves **0.1231**, LSTM-GNN achieves **0.1265**, Heuristic achieves **0.0754**, while TemporalAML achieves **0.0376** (Ablation No-Time is **0.0383**). | Sec. IV (Comparative Benchmarking: Honest Reporting) |
| **E07** | Reconciling Table 4.3 vs Figure 4.1 in Phase-1 Report | Interpretation | `Report/Chapters/ChapterActualWork.tex:75-93`, `artifacts/logs/final_results.csv` | Yes | Table 4.3 was written prior to running the benchmark suite ("benchmarking not started"). Subsequently, `src/evaluate.py` was executed, producing `final_results.csv` and `auprc_comparison.png` (Fig 4.1), but Table 4.3 was not back-updated. | Sec. IV (Resolving Prior Preliminary Documentation) |
| **E08** | Statistical significance: TemporalAML vs Static-TGAT | Verified Result | `artifacts/logs/final_results.md:30-38`, `src/evaluate.py:405-414` | Yes | 1,000-iteration paired bootstrap yields observed $\Delta \text{F1} = -0.0016$, 95% CI $[-0.0072, +0.0043]$, $p = 0.7010$. **The difference is not statistically significant.** | Sec. IV (Statistical Significance Analysis) |
| **E09** | Circular Transfer Head produces 0 Recall / 0 F1 | Verified Result | `artifacts/logs/final_results.md:24`, `src/pattern_mining.py:52-57` | Yes | Raw Bitcoin transaction graph is an acyclic UTXO DAG. Directed cycles do not exist ($y_{\text{circ}} = 0$ for all nodes). Circular head cannot learn non-existent topological loops. | Sec. III & IV (Typology Formulation & Forensic Constraints) |
| **E10** | Smurfing Head AUC-ROC is 0.9618 (Label Circularity) | Verified Result / Caveat | `artifacts/logs/final_results.md:26`, `src/pattern_mining.py:226-245` | Yes | High performance is largely driven by label circularity: smurfing labels are defined as node degree $\ge 10$, while degree metrics are direct input features. The head learns a threshold classifier over its own inputs. | Sec. IV (Typology Head Validation & Validity Threats) |
| **E11** | Layering Peeling Chain Head AUC-ROC is 0.5772 | Verified Result | `artifacts/logs/final_results.md:25`, `src/pattern_mining.py:118-220` | Yes | Mined via topological sort (path length $\ge 4$, fan-out $\le 3$). Performance on test split is modest ($\text{AUC-ROC} = 0.5772$, $\text{AUPRC} = 0.7996$, Recall $0.2532$). | Sec. IV (Typology Results: Layering Dynamics) |
| **E12** | GNNExplainer Edge & Feature Mask Optimization | Verified Impl | `src/explain.py:82-160` | Yes | Optimizes 40-step mutual information masks directly against the trained TGAT multi-head module for flagged transactions. | Sec. III & IV (XAI Framework: Subgraph Extraction) |
| **E13** | Lack of Quantitative Fidelity / Sparsity Audits for XAI | Limitation | `src/explain.py`, `Report/Chapters/ChapterResultsDiscussionsConclusions.tex:75` | Yes | No automated edge deletion fidelity tests ($\Delta \text{F1}$), sparsity metrics, or blind human compliance officer reviews have been conducted. Only qualitative JSON templates exist. | Sec. V (Limitations and Threats to Validity) |
| **E14** | Automated SAR Narrative Generation | Verified Impl | `src/explain.py:400-510`, `artifacts/sar_reports/*.json` | Yes | Generates structured JSON reports containing suspect transaction metadata, top-5 attributed features, and chronologically ordered 2-hop causal payment paths. | Sec. III & IV (Forensic Case Studies & SAR Triage) |

---

## 3. Methodological & Scientific Positioning for the Manuscript

### 3.1 Why Honesty is the Winning Strategy for ICICF 2027
Reviewers at IEEE conferences immediately detect and reject papers that attempt to obscure poor benchmark results or make inflated claims of "state-of-the-art superiority" that contradict their own data tables. 

Instead, the paper must be positioned as a **rigorous critical and empirical investigation**:
1. **Title & Framing:** Frame the paper around *empirical realities, topological constraints, and explainability* in dynamic transaction graphs (e.g., *"Investigating Continuous-Time Graph Attention and Typological Inductive Bias on the Elliptic Bitcoin Benchmark"*).
2. **The Scientific Finding:** Highlight the discovery that public transaction benchmarks collapse inter-transaction latencies ($\Delta t = 0$), revealing a major gap between theoretical continuous-time dynamic graph literature and real-world fintech datasets.
3. **The Multi-Task & XAI Value:** Emphasize the multi-typology formulation (peeling chain extraction vs. fan-out structuring) and the auditable, chronologically ordered SAR pipeline that provides human compliance triage where black-box classifiers fail.
4. **Threats to Validity:** Openly discuss label circularity in smurfing and UTXO DAG limitations on circular loop detection. Reviewers reward rigorous self-critique and precise forensic domain awareness.
