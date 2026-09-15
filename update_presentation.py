import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def update_presentation():
    prs = Presentation('TemporalAML_Updated.pptx')
    
    # Helper to format text frame
    def set_bullet_points(shape, title_text, items, font_size=13, title_size=16):
        tf = shape.text_frame
        tf.word_wrap = True
        tf.clear()
        
        for i, item in enumerate(items):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.space_after = Pt(6)
            p.space_before = Pt(2)
            
            if isinstance(item, tuple): # (Header, Body)
                head, body = item
                r1 = p.add_run()
                r1.text = head + " "
                r1.font.bold = True
                r1.font.size = Pt(font_size)
                r1.font.color.rgb = RGBColor(0, 51, 102) # Dark Navy
                
                r2 = p.add_run()
                r2.text = body
                r2.font.bold = False
                r2.font.size = Pt(font_size)
                r2.font.color.rgb = RGBColor(40, 40, 40)
            elif isinstance(item, list): # Sub-bullets
                for sub in item:
                    sub_p = tf.add_paragraph()
                    sub_p.level = 1
                    sub_p.space_after = Pt(3)
                    r = sub_p.add_run()
                    r.text = sub
                    r.font.size = Pt(font_size - 1)
                    r.font.color.rgb = RGBColor(60, 60, 60)
            else:
                r = p.add_run()
                r.text = item
                r.font.size = Pt(font_size)
                r.font.color.rgb = RGBColor(40, 40, 40)

    # -------------------------------------------------------------
    # SLIDE 4: INTRODUCTION (Index 3)
    # -------------------------------------------------------------
    slide4 = prs.slides[3]
    # Update title
    for s in slide4.shapes:
        if s.has_text_frame and "Introduction" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Introduction: Cryptocurrency & AML Challenge"
            if len(s.text_frame.paragraphs[0].runs) > 0:
                s.text_frame.paragraphs[0].runs[0].font.size = Pt(28)
                s.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(0, 51, 153)
        elif s.has_text_frame and "Money laundering" in s.text_frame.text:
            items = [
                ("Cryptocurrency Foundation:", "Decentralized peer-to-peer digital monetary network operating on immutable public distributed ledgers (blockchains) without central financial intermediaries."),
                ("The Pseudonymity Paradox:", "Transactions utilize cryptographic address hashes (public keys) rather than verified real-world identities, enabling borderless fund flows but creating severe regulatory monitoring hurdles."),
                ("Global Laundering Scale:", "Financial crime accounts for $800B to $2 Trillion annually (2–5% of global GDP), with cryptocurrency laundering growing rapidly across darknet markets, ransomware, and fraud."),
                ("Three Core Laundering Typologies:", "Criminal syndicates execute distinct multi-hop transaction graph patterns to disguise dirty money:"),
                [
                    "• Circular Transfers: Closed transaction loops (A→B→C→A) designed to spoof trading volume and create recursive audit tracing confusion.",
                    "• Layering Chains: Rapid sequential multi-hop transfers across consecutive timestamps to distance illicit funds from crime sources.",
                    "• Smurfing: Structuring large sums into micro-transactions below mandatory regulatory reporting thresholds ($10,000 limit)."
                ],
                ("Proposed Solution (TemporalAML):", "A Temporal Graph Attention Network (TGAT) with Learnable Fourier Time Encoding that simultaneously detects all 3 patterns from shared embeddings and generates explainable subgraph evidence.")
            ]
            set_bullet_points(s, "Introduction", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 5: PROBLEM STATEMENT (Index 4)
    # -------------------------------------------------------------
    slide5 = prs.slides[4]
    for s in slide5.shapes:
        if s.has_text_frame and "Problem Statement" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Problem Statement: Limitations of Existing AML Systems"
        elif s.has_text_frame and len(s.text_frame.text.strip()) > 50 and "Existing AML" not in s.text_frame.text:
            items = [
                ("1. Static Graph Representations:", "Existing AML systems (Weber 2019, Lawal 2025) treat transaction graphs as static snapshots, causing severe future temporal data leakage and discarding transaction velocity."),
                ("2. Disconnected Spatio-Temporal Modules:", "Hybrid GNN-LSTM models (Alarab 2023) process graph structure and time in separate pipelines; the graph attention mechanism itself remains time-unaware."),
                ("3. Rigid Pre-determined Frequencies:", "Wavelet-based encodings (Lin 2026) apply fixed frequency bands that cannot adapt to the non-stationary, bursty timescales of cryptocurrency laundering."),
                ("4. Non-Actionable Binary Classification:", "All existing models output only binary (illicit/licit) labels without identifying which specific pattern typology triggered the alert (required for SAR filing)."),
                ("5. Absence of Pattern-Aware Explainability:", "Attention weights do not provide compliance-ready, time-ordered subgraph evidence connecting an alert to the underlying transaction trail.")
            ]
            set_bullet_points(s, "Problem Statement", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 6: MOTIVATION (Index 5)
    # -------------------------------------------------------------
    slide6 = prs.slides[5]
    for s in slide6.shapes:
        if s.has_text_frame and "Motivation" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Motivation & Justification for Problem Selection"
        elif s.has_text_frame and "Need for" in s.text_frame.text:
            items = [
                ("Regulatory Compliance Demand:", "Global regulators (FATF, FinCEN) mandate suspicious activity reporting (SAR) identifying exact laundering mechanisms, not just black-box anomaly scores."),
                ("Graph Topology Necessity:", "Tabular machine learning evaluates transactions in isolation; only Graph Neural Networks capture multi-hop relational fund flows across complex wallet networks."),
                ("Continuous Temporal Advantage:", "TGAT embeds continuous Fourier time deltas (Δt) directly into attention, capturing high-velocity laundering bursts and peeling chains."),
                ("Multi-Task Latent Efficiency:", "A shared 128D latent representation detects Circular, Layering, and Smurfing simultaneously in a single forward pass without 3x computational overhead."),
                ("Causal Zero-Leakage Guarantee:", "Enforces causal temporal neighborhood aggregation (tj ≤ ti), ensuring mathematically sound real-time production deployment.")
            ]
            set_bullet_points(s, "Motivation", items, font_size=13)

    # -------------------------------------------------------------
    # SLIDE 10: OBJECTIVES (Part 1: O1 & O2) (Index 9)
    # -------------------------------------------------------------
    slide10 = prs.slides[9]
    for s in slide10.shapes:
        if s.has_text_frame and "Objectives" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Research Objectives (O1 & O2)"
        elif s.has_text_frame and "Research Aim:" in s.text_frame.text:
            items = [
                ("Research Aim:", "To design, implement, and validate a Temporal Graph Attention Network with Learnable Fourier Time Encoding for explainable multi-pattern AML detection on Bitcoin data."),
                ("Objective 1 (O1) — Temporal Graph Construction & Feature Engineering:", ""),
                [
                    "• Construct a directed temporal graph G = (V, E, T) containing 203,769 transaction nodes and 234,355 directed edges from the Elliptic dataset.",
                    "• Engineer 6 domain-specific topological features (in/out degree, fan-in/out ratios, recency, neighbor time delta) yielding 172-dimensional feature vectors.",
                    "• Implement non-leaking chronological splits (t ≤ 34 Train, 35 ≤ t ≤ 42 Val, 43 ≤ t ≤ 49 Test) with StandardScaler fitted strictly on training data."
                ],
                ("Objective 2 (O2) — Learnable Fourier Time Encoding (Core Novelty):", ""),
                [
                    "• Design continuous-time sinusoidal encoding Φ(Δt) = [cos(ω·Δt + φ) || sin(ω·Δt + φ)] ∈ ℝ¹²⁸.",
                    "• Formulate trainable frequency parameters ω, φ ∈ ℝ⁶⁴ optimized via backpropagation (∂L/∂ω = -Δt·sin(ω·Δt + φ)), dynamically adapting to laundering burst velocities (+8.47% F1 gain)."
                ]
            ]
            set_bullet_points(s, "Objectives", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 11: OBJECTIVES (Part 2: O3 & O4) (Index 10)
    # -------------------------------------------------------------
    slide11 = prs.slides[10]
    for s in slide11.shapes:
        if s.has_text_frame and "Cont.." in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Research Objectives (O3 & O4)"
        elif s.has_text_frame and "O3 –" in s.text_frame.text:
            items = [
                ("Objective 3 (O3) — Shared-Embedding Multi-Pattern AML Detection:", ""),
                [
                    "• Mine ground-truth pattern labels from graph topology: Tarjan's SCC (Circular), time-monotonic BFS (Layering), and degree thresholding (Smurfing).",
                    "• Construct 2-layer TGATConv shared encoder generating unified 128-dimensional latent node representations.",
                    "• Train 3 parallel classification heads with positive class-weighted multi-task loss (L_total = Σ λ_k WBCE_k), achieving test F1 > 0.88 on all patterns."
                ],
                ("Objective 4 (O4) — Explainable Subgraph Generation (XAI):", ""),
                [
                    "• Integrate GNNExplainer to optimize continuous edge masks M ∈ [0, 1] via mutual information maximization for flagged illicit nodes.",
                    "• Generate minimal, time-ordered subgraph evidence paths connecting flagged transactions to suspicious sources for regulatory Suspicious Activity Report (SAR) filing."
                ]
            ]
            set_bullet_points(s, "Objectives Cont", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 13: DATASET OVERVIEW (Index 12)
    # -------------------------------------------------------------
    slide13 = prs.slides[12]
    for s in slide13.shapes:
        if s.has_text_frame and "Dataset Overview" in s.text_frame.text and len(s.text_frame.text.strip()) < 30:
            s.text_frame.paragraphs[0].text = "Dataset Overview: Elliptic Bitcoin Dataset"
        elif s.has_text_frame and "203,769" in s.text_frame.text:
            items = [
                ("1. Dataset Identity & Scale:", "Benchmark dataset published by Elliptic & MIT-IBM Watson AI Lab (Kaggle). Contains 203,769 transaction nodes, 234,355 directed edges, across 49 discrete time steps."),
                ("2. Attribute Breakdown (172 Total Features):", ""),
                [
                    "• Local Attributes (94 dims): Transaction fee, BTC volume, input/output counts, byte size, local aggregations.",
                    "• Neighborhood Attributes (72 dims): 1-hop statistical aggregates (mean, std, min, max) of local features.",
                    "• Engineered Temporal (6 dims): Out-degree, in-degree, fan-out ratio, fan-in ratio, temporal recency (t/49), neighbor time delta."
                ],
                ("3. Domain Importance & Label Distribution:", "Real-world forensic entity ground-truth labels: 4,545 Illicit (2.2%), 42,019 Licit (20.6%), 157,205 Unknown (77.2% — preserved for message-passing graph connectivity)."),
                ("4. Temporal Structure & Leakage-Free Split:", "49 consecutive bi-weekly snapshots (~2 years of Bitcoin history). Strict chronological splits: Train (t=1–34, 70.4%), Val (t=35–42, 16.8%), Test (t=43–49, 12.8%) with StandardScaler fitted strictly on t ≤ 34.")
            ]
            set_bullet_points(s, "Dataset Overview", items, font_size=11)

    # -------------------------------------------------------------
    # SLIDE 14: METHODOLOGY 1 (Index 13)
    # -------------------------------------------------------------
    slide14 = prs.slides[13]
    for s in slide14.shapes:
        if s.has_text_frame and "Methodology 1" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Methodology 1: Temporal Graph Construction [O1]"
        elif s.has_text_frame and "Description:" in s.text_frame.text:
            items = [
                ("Objective & Scope:", "Construct a directed temporal graph G = (V, E, T) from the 3 Elliptic CSV files with zero temporal data leakage."),
                ("Key Process Steps:", ""),
                [
                    "1. Continuous Node ID Remapping: Map arbitrary 64-bit transaction IDs to contiguous 0-indexed integers [0, 203768].",
                    "2. Directed Edge Tensor Assembly: Remap 234,355 fund transfers into PyG edge_index [2, 234355] with edge timestamps.",
                    "3. Domain Feature Engineering: Compute 6 topological metrics expanding raw 166 features to 172 dimensions.",
                    "4. Leakage-Free Normalization: Fit StandardScaler strictly on training nodes (t ≤ 34); transform validation and test nodes.",
                    "5. Chronological Split Masks: Generate boolean masks for Train (t=1–34), Val (t=35–42), and Test (t=43–49)."
                ],
                ("Deliverable Output:", "PyTorch Geometric Data object: x (203769, 172), edge_index (2, 234355), t (203769,), y (203769,), split masks.")
            ]
            set_bullet_points(s, "Methodology 1", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 15: METHODOLOGY 2 (Index 14)
    # -------------------------------------------------------------
    slide15 = prs.slides[14]
    for s in slide15.shapes:
        if s.has_text_frame and "Methodology 2" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Methodology 2: Learnable Fourier Time Encoding [O2]"
        elif s.has_text_frame and "Description:" in s.text_frame.text:
            items = [
                ("Objective & Formulation:", "Embed continuous edge time deltas Δt = t_target - t_source directly inside graph attention via learnable Fourier mapping."),
                ("Mathematical Formulation:", "Φ(Δt) = √(2/d) · [cos(ω₁Δt + φ₁), sin(ω₁Δt + φ₁), ..., cos(ω₆₄Δt + φ₆₄), sin(ω₆₄Δt + φ₆₄)] ∈ ℝ¹²⁸"),
                ("Core Algorithmic Mechanisms:", ""),
                [
                    "• Trainable Frequency Parameters: ω, φ ∈ ℝ⁶⁴ initialized as learnable PyTorch nn.Parameters.",
                    "• Gradient Backpropagation: ∂L/∂ω = -Δt · sin(ωΔt + φ) automatically tunes attention to laundering velocity bursts.",
                    "• Causal Temporal Aggregation: Attention is restricted to historical neighbors (tj ≤ ti), preventing future data leakage."
                ],
                ("Deliverable Output:", "Time-aware attention projections yielding +8.47% test F1-score gain over fixed harmonic sinusoids.")
            ]
            set_bullet_points(s, "Methodology 2", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 16: METHODOLOGY 3 (Index 15)
    # -------------------------------------------------------------
    slide16 = prs.slides[15]
    for s in slide16.shapes:
        if s.has_text_frame and "Methodology 3" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Methodology 3: Multi-Pattern TGAT Training [O3]"
        elif s.has_text_frame and "Description:" in s.text_frame.text:
            items = [
                ("Shared-Encoder Architecture:", "2-layer TGATConv (172 → 128 → 128 dims, 4 attention heads) producing unified 128D latent node embeddings."),
                ("Three Dedicated Pattern Heads:", ""),
                [
                    "• Circular Transfer Head: FFN → Sigmoid (Mined via Tarjan's SCC cycle size ≥ 2) — Test F1 = 0.88, AUC = 0.92.",
                    "• Layering Chain Head: FFN → Sigmoid (Mined via temporal monotonic BFS paths ≥ 3 hops) — Test F1 = 0.91, AUC = 0.95.",
                    "• Smurfing Fan-Out Head: FFN → Sigmoid (Mined via degree thresholding d ≥ 10) — Test F1 = 0.94, AUC = 0.97."
                ],
                ("Joint Loss Formulation:", "L_total = λ₁·WBCE_circ + λ₂·WBCE_lay + λ₃·WBCE_smurf with positive class weights (w_pos ≈ 44.8) resolving 2.2% class imbalance."),
                ("Optimization:", "Adam Optimizer (LR=0.001), CosineAnnealingLR scheduler, 30 epochs on Google Colab T4 GPU (~20 mins).")
            ]
            set_bullet_points(s, "Methodology 3", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 17: METHODOLOGY 4 (Index 16)
    # -------------------------------------------------------------
    slide17 = prs.slides[16]
    for s in slide17.shapes:
        if s.has_text_frame and "Methodology 4" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "Methodology 4: Explainability & SAR Generation [O4]"
        elif s.has_text_frame and "Description:" in s.text_frame.text:
            items = [
                ("Objective & Rationale:", "Generate compact, time-ordered subgraph evidence for regulatory Suspicious Activity Report (SAR) compliance."),
                ("Core Algorithmic Steps:", ""),
                [
                    "1. High-Risk Node Selection: Target transactions flagged with P(pattern) ≥ 0.50 by Objective 3 heads.",
                    "2. GNNExplainer Mask Optimization: Freeze model weights Θ; optimize continuous edge mask M ∈ [0, 1] over 200 epochs.",
                    "3. Mutual Information Objective: max_M MI(Y, G_s) = -log P_Θ(Y=c | G_s = A * σ(M)) + λ₁ H(M) + λ₂ ||M||₁.",
                    "4. Minimal Subgraph Extraction: Retain top-k edge paths forming the chronological laundering audit trail."
                ],
                ("Deliverable Output:", "Automated compliance evidence dossier mapping flagged transactions to source wallets, timestamps, and pattern attribution.")
            ]
            set_bullet_points(s, "Methodology 4", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 18: EXPECTED OUTCOMES (Index 17)
    # -------------------------------------------------------------
    slide18 = prs.slides[17]
    for s in slide18.shapes:
        if s.has_text_frame and "Expected Outcomes" in s.text_frame.text and len(s.text_frame.text.strip()) < 30:
            s.text_frame.paragraphs[0].text = "Expected Outcomes & Research Deliverables"
        elif s.has_text_frame and "A novel Temporal" in s.text_frame.text:
            items = [
                ("High-Precision Temporal GNN:", "A robust TGAT model with Learnable Fourier Time Encoding achieving composite test F1-score > 0.91 and AUC-ROC > 0.94 on the benchmark Elliptic Bitcoin dataset."),
                ("First Unified 3-Pattern Framework:", "Simultaneous multi-label detection of Circular Transfers (F1=0.88), Layering Chains (F1=0.91), and Smurfing (F1=0.94) from shared 128D embeddings."),
                ("Quantified Fourier Encoding Gain:", "Empirical verification demonstrating an +8.47% F1 improvement of learnable frequencies over static and fixed sinusoidal baselines."),
                ("Compliance-Ready SAR Explainability:", "Automated generation of minimal, time-ordered subgraph evidence dossiers via GNNExplainer for FinCEN/FATF regulatory filing."),
                ("Zero-Leakage Causal Guarantee:", "Formal verification that causal temporal neighborhood aggregation ensures sound real-time production deployment without future data leakage.")
            ]
            set_bullet_points(s, "Expected Outcomes", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 20: CONCLUSION (Index 19)
    # -------------------------------------------------------------
    slide20 = prs.slides[19]
    for s in slide20.shapes:
        if s.has_text_frame and "Conclusion" in s.text_frame.text and len(s.text_frame.text.strip()) < 20:
            s.text_frame.paragraphs[0].text = "Conclusion & Project Summary"
        elif s.has_text_frame and "This research proposes" in s.text_frame.text:
            items = [
                ("Core Innovation:", "TemporalAML integrates Learnable Fourier Time Encoding directly inside Graph Attention, capturing continuous transaction velocity and burst dynamics in cryptocurrency networks."),
                ("Multi-Pattern Breakthrough:", "Transitions from non-actionable binary (illicit/licit) classification to unified multi-task detection of Circular Transfers, Layering Chains, and Smurfing from shared 128D embeddings."),
                ("Empirical Validation:", "Demonstrates strong performance across all three patterns (Circular F1=0.88, Layering F1=0.91, Smurfing F1=0.94) on 203,769 real-world Bitcoin transactions."),
                ("Explainable Compliance:", "GNNExplainer integration produces time-ordered subgraph evidence dossiers ready for regulatory Suspicious Activity Report (SAR) filing."),
                ("Operational Readiness:", "Backed by a fully functional FastAPI backend (Port 8000) and an interactive dark-mode dashboard with a Real-Time Node Inspector for compliance officers.")
            ]
            set_bullet_points(s, "Conclusion", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 25: GAP ANALYSIS (Index 24)
    # -------------------------------------------------------------
    slide25 = prs.slides[24]
    for s in slide25.shapes:
        if s.has_text_frame and "Gap Analysis" in s.text_frame.text and len(s.text_frame.text.strip()) < 20:
            s.text_frame.paragraphs[0].text = "Gap Analysis: Summary of Literature Gaps"
        elif s.has_text_frame and "GAP 1" in s.text_frame.text:
            items = [
                ("GAP 1 — Static Graph Limitations:", "Existing AML systems (Weber 2019, Lawal 2025) treat transaction graphs as static snapshots, causing future data leakage and discarding transaction velocity."),
                ("GAP 2 — Disconnected Spatio-Temporal Streams:", "Hybrid GNN-LSTM models (Alarab 2023) process graph structure and time in separate pipelines; graph attention remains time-unaware."),
                ("GAP 3 — Inflexible Time Encoding:", "Wavelet models (Lin 2026) use fixed frequency bands that cannot adapt to the non-stationary burst timescales of cryptocurrency laundering."),
                ("GAP 4 — Non-Actionable Binary Output:", "Existing systems output only binary (illicit/licit) labels without identifying the specific laundering typology required for regulatory SAR filing."),
                ("GAP 5 — Absence of Subgraph Explainability:", "Prior works rely on attention weights as a proxy, failing to generate time-ordered, pattern-specific subgraph evidence trails.")
            ]
            set_bullet_points(s, "Gap Analysis", items, font_size=12)

    # -------------------------------------------------------------
    # SLIDE 26: SYSTEM ARCHITECTURE (Index 25)
    # -------------------------------------------------------------
    slide26 = prs.slides[25]
    for s in slide26.shapes:
        if s.has_text_frame and "System Architecture" in s.text_frame.text:
            s.text_frame.paragraphs[0].text = "System Architecture — 6 Modular Block Components"
        elif s.has_text_frame and "Pipeline Overview:" in s.text_frame.text:
            items = [
                ("BLOCK 1: Data Ingestion & Graph Construction [O1]:", "Ingests 3 Elliptic CSVs → Continuous ID Remapping [0, 203768] → Engineers 6 topological features (172D) → Leakage-free StandardScaler (t ≤ 34) → PyG Data container."),
                ("BLOCK 2: Learnable Fourier Time Encoding [O2]:", "Computes edge time deltas Δt = t_i - t_j → Projects into 128D sinusoidal embeddings with trainable frequencies (ω, φ ∈ ℝ⁶⁴) updated via backpropagation (+8.47% F1 gain)."),
                ("BLOCK 3: TGAT Attention & Shared Embeddings [O3 Backbone]:", "2-layer TGATConv (4 attention heads) with causal temporal neighborhood masking (tj ≤ ti) producing unified 128D latent node representations Z ∈ ℝ^(N×128)."),
                ("BLOCK 4: Multi-Pattern Classification Heads [O3 Heads]:", "Three parallel FFN heads detecting Circular Transfers (F1=0.88), Layering Chains (F1=0.91), and Smurfing (F1=0.94) trained with positive class-weighted loss."),
                ("BLOCK 5: Explainable Subgraph Generation [O4 XAI]:", "GNNExplainer optimizes continuous edge masks M ∈ [0, 1] on flagged transactions to extract minimal time-ordered audit trails for regulatory SAR reports."),
                ("BLOCK 6: Operational Platform & Node Inspector:", "FastAPI REST backend (Port 8000) serving live endpoints + Interactive Dashboard with real-time Node Inspector predicting pattern risks for any transaction ID.")
            ]
            set_bullet_points(s, "System Architecture", items, font_size=11)

    prs.save('TemporalAML_Updated.pptx')
    print('Successfully updated TemporalAML_Updated.pptx with all mentor feedback!')

if __name__ == '__main__':
    update_presentation()
