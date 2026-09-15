import sys
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def apply_feedback():
    pptx_path = 'TemporalAML_review_1(Anshu).pptx'
    prs = Presentation(pptx_path)
    print(f"Loaded {pptx_path} with {len(prs.slides)} slides.")

    DARK_NAVY = RGBColor(0, 51, 102)     # #003366 - Header bold
    DARK_CHARCOAL = RGBColor(35, 35, 35) # #232323 - Body text
    MUTED_GRAY = RGBColor(70, 70, 70)    # Sub-bullets
    BLUE_TITLE = RGBColor(0, 51, 153)    # Slide Title

    def format_title(shape, title_text, size=24):
        shape.text_frame.clear()
        p = shape.text_frame.paragraphs[0]
        p.text = title_text
        if len(p.runs) > 0:
            p.runs[0].font.size = Pt(size)
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = BLUE_TITLE

    def format_content(shape, items, font_size=11, space_after=4):
        tf = shape.text_frame
        tf.word_wrap = True
        tf.clear()
        
        for i, item in enumerate(items):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.space_after = Pt(space_after)
            p.space_before = Pt(2)
            
            if isinstance(item, tuple): # (Header, Body)
                head, body = item
                r1 = p.add_run()
                r1.text = head + " " if head else ""
                r1.font.bold = True
                r1.font.size = Pt(font_size)
                r1.font.color.rgb = DARK_NAVY
                
                if body:
                    r2 = p.add_run()
                    r2.text = body
                    r2.font.bold = False
                    r2.font.size = Pt(font_size)
                    r2.font.color.rgb = DARK_CHARCOAL
            elif isinstance(item, list): # Sub-bullets
                for sub in item:
                    sub_p = tf.add_paragraph()
                    sub_p.level = 1
                    sub_p.space_after = Pt(2)
                    sub_p.space_before = Pt(1)
                    r = sub_p.add_run()
                    r.text = sub
                    r.font.size = Pt(font_size - 1)
                    r.font.color.rgb = MUTED_GRAY
            else:
                r = p.add_run()
                r.text = str(item)
                r.font.size = Pt(font_size)
                r.font.color.rgb = DARK_CHARCOAL

    def get_shape(slide, name):
        for s in slide.shapes:
            if s.name == name:
                return s
        return None

    # =========================================================================
    # SLIDE 4: INTRODUCTION (Cryptocurrency Foundations & AML Challenges)
    # =========================================================================
    s4 = prs.slides[3]
    t4 = get_shape(s4, 'TextBox 11')
    b4 = get_shape(s4, 'TextBox 12')
    format_title(t4, "Introduction: Cryptocurrency Foundations & AML Challenge", size=22)
    items4 = [
        ("1. What is Cryptocurrency?", "A decentralized, peer-to-peer digital monetary system secured by cryptography and consensus mechanisms (Proof-of-Work / Proof-of-Stake) on immutable distributed ledgers, operating without central financial intermediaries."),
        ("2. The Pseudonymity Paradox:", "Transactions are authorized via cryptographic key-pairs (public address hashes) rather than verified identities, enabling transparent global fund flows while simultaneously creating severe tracing hurdles for regulatory authorities."),
        ("3. Global Laundering Scale:", "Financial crime launders $800 Billion to $2 Trillion annually (2–5% of global GDP); cryptocurrency laundering has doubled year-over-year across darknet markets, ransomware syndicates, and sanction evasion."),
        ("4. The Three Core Laundering Typologies:", "Illicit entities bypass monitoring using 3 distinct structural transaction graph patterns:"),
        [
            "• Circular Transfers: Closed transaction loops (A→B→C→A) designed to wash-trade, spoof volumes, and induce recursive loops in audit software.",
            "• Layering Chains: Rapid sequential multi-hop transfers across consecutive timestamps to distance illicit profits from crime sources.",
            "• Smurfing (Structuring): Splitting large sums into micro-transactions below mandatory regulatory thresholds ($10,000 limit) and aggregating at exit hubs."
        ],
        ("5. Proposed Solution (TemporalAML):", "A continuous-time Temporal Graph Attention Network (TGAT) with Learnable Fourier Time Encoding that detects all 3 typologies simultaneously and generates auditable causal subgraph evidence.")
    ]
    format_content(b4, items4, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 5: PROBLEM STATEMENT (NO PARAGRAPHS, REMOVE TEXTBOX 22)
    # =========================================================================
    s5 = prs.slides[4]
    t5 = get_shape(s5, 'TextBox 11')
    b5 = get_shape(s5, 'TextBox 12')
    p5 = get_shape(s5, 'TextBox 22')
    if p5:
        sp = p5._element
        sp.getparent().remove(sp)
        print("Removed paragraph box TextBox 22 from Slide 5.")
    format_title(t5, "Problem Statement: Limitations of Existing AML Systems", size=22)
    b5.top = Inches(1.8)
    b5.height = Inches(5.1)
    items5 = [
        ("1. Static Graph Misalignment:", "Conventional GNNs (GCN, GraphSAGE) treat transaction networks as static graphs, causing severe future temporal data leakage and discarding continuous transaction velocities."),
        ("2. Disconnected Spatio-Temporal Modules:", "Hybrid models (GNN+LSTM) decouple graph topology and time into separate sequential pipelines; the graph attention mechanism itself remains completely time-unaware."),
        ("3. Inflexible Pre-determined Frequencies:", "Wavelet and fixed sinusoidal encodings apply rigid harmonic frequencies that cannot adapt to the non-stationary, bursty timescales of cryptocurrency money laundering."),
        ("4. Non-Actionable Binary Classification:", "Current state-of-the-art models output only binary (illicit/licit) flags without identifying the underlying typology (Circular, Layering, Smurfing) required for regulatory SAR filing."),
        ("5. Absence of Pattern-Aware Explainability:", "Existing architectures lack causal, time-ordered subgraph extraction to substantiate compliance alerts with transparent forensic evidence.")
    ]
    format_content(b5, items5, font_size=12, space_after=6)

    # =========================================================================
    # SLIDE 6: MOTIVATION & PROBLEM SELECTION JUSTIFICATION
    # =========================================================================
    s6 = prs.slides[5]
    t6 = get_shape(s6, 'TextBox 11')
    b6 = get_shape(s6, 'TextBox 12')
    format_title(t6, "Motivation & Clear Justification for Problem Selection", size=22)
    b6.top = Inches(1.8)
    b6.height = Inches(5.1)
    items6 = [
        ("1. Why Cryptocurrency AML?", "Global cryptocurrency illicit volume surpassed $24B; pseudonymous wallet addresses bypass traditional banking rules, requiring automated graph intelligence for national financial security."),
        ("2. Why Graph Neural Networks (GNNs)?", "Financial crime is inherently relational; tabular machine learning (XGBoost/RF) evaluates transactions in isolation, missing multi-hop peeling chains and syndication structures."),
        ("3. Why Continuous TGAT over Static/LSTM?", "Static GNNs suffer future temporal leakage, while GNN-LSTM destroys graph parallelism; TGAT embeds continuous Fourier time deltas (Δt) directly into attention weights."),
        ("4. Why Multi-Pattern Classification over Binary?", "Financial Action Task Force (FATF) and FinCEN regulations mandate reporting the exact laundering typology to enable targeted law enforcement asset freezes."),
        ("5. Why Causal Explainability (SAR)?", "Financial intelligence units cannot act on black-box probabilities; regulatory compliance requires auditable, time-ordered evidence subgraphs for every alert.")
    ]
    format_content(b6, items6, font_size=12, space_after=6)

    # =========================================================================
    # SLIDE 11: GAP ANALYSIS
    # =========================================================================
    s11 = prs.slides[10]
    t11 = get_shape(s11, 'TextBox 11')
    b11 = get_shape(s11, 'TextBox 12')
    format_title(t11, "Gap Analysis: Summary of Literature Limitations", size=22)
    items11 = [
        ("GAP 1 — Temporal Data Leakage:", "Weber et al. (2019) & Lawal et al. (2025) evaluate static GNNs across time-aggregated graphs, allowing future transaction edges to influence past node predictions."),
        ("GAP 2 — Decoupled Time-Structure Dynamics:", "Alarab et al. (2023) use GCN followed by LSTM; message passing over edges does not condition attention on edge time intervals (Δt)."),
        ("GAP 3 — Rigid Time Representations:", "Lin et al. (2026) apply fixed-frequency wavelet transforms that fail to model bursty transaction velocities characteristic of rapid layering chains."),
        ("GAP 4 — Lack of Multi-Typology Supervision:", "All benchmark literature frames AML strictly as binary detection, failing to separate peeling chains, fan-out smurfing, and circular wash-trading."),
        ("GAP 5 — Unsubstantiated Forensic Alerts:", "Existing systems provide no automated mechanism to extract minimal time-ordered causal subgraphs for Suspicious Activity Report (SAR) compliance.")
    ]
    format_content(b11, items11, font_size=12, space_after=6)

    # =========================================================================
    # SLIDE 12: OBJECTIVES (O1 & O2 - SPECIFIC)
    # =========================================================================
    s12 = prs.slides[11]
    t12 = get_shape(s12, 'TextBox 11')
    b12 = get_shape(s12, 'TextBox 12')
    format_title(t12, "Specific Research Objectives (O1 & O2)", size=22)
    items12 = [
        ("Overarching Research Aim:", "To design, implement, and validate an end-to-end Temporal Graph Attention Network (TGAT) with Learnable Fourier Time Encoding for explainable multi-pattern AML detection on Bitcoin data."),
        ("Objective 1 (O1) — Temporal Graph Construction & Feature Engineering:", ""),
        [
            "• Construct a directed temporal graph G = (V, E, T) with 203,769 transaction nodes and 234,355 directed edges across 49 time steps from the Elliptic dataset.",
            "• Engineer 6 domain topological features (in/out degree, fan-in/out ratios, recency t/49, neighbor time delta) expanding feature dimensionality from 166 to 172.",
            "• Enforce zero-leakage chronological split masks (Train: t=1–34, Val: t=35–40, Test: t=41–49) with StandardScaler fitted strictly on training data (t ≤ 34)."
        ],
        ("Objective 2 (O2) — Continuous Learnable Fourier Time Encoding:", ""),
        [
            "• Formulate continuous-time Fourier projection Φ(Δt) = [cos(ω·Δt + φ) || sin(ω·Δt + φ)] ∈ ℝ¹²⁸ for edge time deltas Δt = t_target - t_source.",
            "• Implement trainable frequency parameters ω, φ ∈ ℝ⁶⁴ updated end-to-end via backpropagation (∂L/∂ω = -Δt·sin(ω·Δt + φ)), dynamically capturing burst laundering velocities (+8.47% F1 gain).",
            "• Enforce strict causal neighbor sampling ensuring only historical transactions (t_neighbor ≤ t_target) are aggregated."
        ]
    ]
    format_content(b12, items12, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 13: OBJECTIVES (O3 & O4 - SPECIFIC)
    # =========================================================================
    s13 = prs.slides[12]
    t13 = get_shape(s13, 'TextBox 11')
    b13 = get_shape(s13, 'TextBox 12')
    format_title(t13, "Specific Research Objectives (O3 & O4)", size=22)
    items13 = [
        ("Objective 3 (O3) — Shared-Embedding Multi-Pattern AML Detection:", ""),
        [
            "• Construct a 2-layer TGATConv shared encoder (4 heads, hidden dim 128) computing unified latent node representations Z ∈ ℝ^(N×128).",
            "• Mine topological ground-truth labels: Tarjan's SCC (Circular, cycle ≥ 2), temporal monotonic BFS (Layering, path ≥ 3 hops), and degree thresholding (Smurfing, d ≥ 10).",
            "• Deploy dedicated classification heads trained with class-weighted multi-task loss (L_total = Σ λ_k WBCE_k, w_pos ≈ 44.8) to resolve the severe 2.2% illicit class imbalance."
        ],
        ("Objective 4 (O4) — Explainable Subgraph Generation & SAR Reporting:", ""),
        [
            "• Integrate GNNExplainer to optimize continuous edge masks M ∈ [0, 1] via mutual information maximization max_M MI(Y, G_s) for flagged transactions.",
            "• Extract minimal, time-ordered causal evidence subgraphs connecting flagged transactions to source wallets and fund trails.",
            "• Auto-generate standardized regulatory Suspicious Activity Report (SAR) executive narratives and exportable FinCEN-compliant JSON dossiers."
        ]
    ]
    format_content(b13, items13, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 14: PROPOSED CONTRIBUTION
    # =========================================================================
    s14 = prs.slides[13]
    t14 = get_shape(s14, 'TextBox 11')
    b14 = get_shape(s14, 'TextBox 12')
    format_title(t14, "Summary of Key Proposed Contributions", size=22)
    items14 = [
        ("1. First Trainable Fourier Time Encoding for Crypto AML:", "Replaces fixed harmonic sinusoids with learnable frequency parameters (ω, φ) optimized via backpropagation, adapting dynamically to non-stationary fund velocities."),
        ("2. Unified Multi-Pattern Latent Representation:", "Transitions beyond black-box binary classification to simultaneous detection of Circular, Layering, and Smurfing typologies from a single shared 128D embedding backbone."),
        ("3. Mathematically Rigorous Causal Guarantee:", "Enforces strict chronological splitting (t ≤ 34 / 35–40 / 41–49) and backward-only causal neighbor aggregation, guaranteeing zero future data leakage."),
        ("4. Automated Regulatory Evidence Synthesis:", "Bridges AI detection with regulatory compliance by generating FinCEN-ready Suspicious Activity Reports (SARs) and time-ordered causal evidence subgraphs via GNNExplainer."),
        ("5. Operational Production-Grade Artifacts:", "Delivers an interactive Streamlit forensic radar dashboard and a FastAPI backend with a real-time Node Inspector for compliance teams.")
    ]
    format_content(b14, items14, font_size=12, space_after=6)

    # =========================================================================
    # SLIDE 15: DATASET OVERVIEW (Dataset -> Attributes -> Importance -> Temporal)
    # =========================================================================
    s15 = prs.slides[14]
    t15 = get_shape(s15, 'TextBox 11')
    b15 = get_shape(s15, 'TextBox 12')
    format_title(t15, "Dataset Overview: Elliptic Bitcoin Benchmark", size=22)
    items15 = [
        ("1. Dataset Identity & Scale:", "Benchmark cryptocurrency dataset published jointly by Elliptic & MIT-IBM Watson AI Lab (Kaggle). Comprises 203,769 transaction nodes and 234,355 directed edges spanning 49 discrete time steps (~2 years of Bitcoin history)."),
        ("2. Attribute Breakdown (172 Total Features):", ""),
        [
            "• Local Features (94 dims): Transaction fee, BTC output volume, input/output counts, transaction byte size.",
            "• Neighborhood Features (72 dims): 1-hop aggregated statistics (mean, standard deviation, min, max) of neighbor local features.",
            "• Engineered Domain Topological Features (6 dims): In-degree, out-degree, fan-out ratio, fan-in ratio, normalized timestamp (t/49), neighbor time delta."
        ],
        ("3. Domain Importance & Label Imbalance:", "Real-world forensic ground truth: 4,545 Illicit (2.2% — ransomware, darknet, scams), 42,019 Licit (20.6% — exchanges, miners), and 157,205 Unknown (77.2% — unlabelled, preserved for graph message passing)."),
        ("4. Temporal Structure & Zero-Leakage Split:", "Strict chronological inductive partitioning: Train Split (t=1–34, 70.4%), Validation Split (t=35–40, 16.8%, for threshold freezing), and Test Split (t=41–49, 12.8%, unseen future). StandardScaler is fitted strictly on t ≤ 34.")
    ]
    format_content(b15, items15, font_size=11, space_after=4)

    # =========================================================================
    # SLIDES 16 to 19: SYSTEM ARCHITECTURE (6 MODULAR BLOCK COMPONENTS)
    # =========================================================================
    # Slide 16: Overview of all 6 Block Components
    s16 = prs.slides[15]
    t16 = get_shape(s16, 'TextBox 21')
    format_title(t16, "System Architecture — 6 Modular Block Components", size=22)
    b16 = s16.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.1))
    items16 = [
        ("BLOCK 1: Data Ingestion & Temporal Graph Construction [O1]:", "Ingests 3 raw CSVs → Continuous 0-indexed ID remapping [0, 203768] → Engineers 6 topological features (172D) → Zero-leakage StandardScaler (t ≤ 34) → PyG Data container."),
        ("BLOCK 2: Continuous Learnable Fourier Time Encoding [O2]:", "Computes edge time deltas Δt = t_i - t_j → Projects into 128D Fourier vectors with trainable frequencies (ω, φ ∈ ℝ⁶⁴) updated via backpropagation (+8.47% F1 gain)."),
        ("BLOCK 3: TGAT Attention & Shared Latent Embeddings [O3 Backbone]:", "2-layer TGATConv (4 attention heads, hidden dim 128) with causal backward-only neighbor masking (t_j ≤ t_i) producing unified 128D latent node embeddings Z ∈ ℝ^(N×128)."),
        ("BLOCK 4: Dedicated Multi-Pattern Classification Heads [O3 Heads]:", "Three parallel 2-layer FFN heads detecting Circular Transfers, Layering Chains, and Smurfing trained with positive class-weighted multi-task loss (w_pos ≈ 44.8)."),
        ("BLOCK 5: Explainable Subgraph Generation & SAR Reporting [O4 XAI]:", "GNNExplainer optimizes continuous edge masks M ∈ [0, 1] on flagged transactions to extract minimal, time-ordered causal evidence subgraphs for automated SAR narrative generation."),
        ("BLOCK 6: Operational Radar Dashboard & Real-Time Inspector:", "FastAPI REST backend (Port 8000) serving real-time inference endpoints + Interactive Streamlit forensic radar dashboard (Port 8501) with live Node Inspector.")
    ]
    format_content(b16, items16, font_size=11, space_after=4)

    # Slide 17: Deep Dive on Block 1 (O1) & Block 2 (O2)
    s17 = prs.slides[16]
    t17 = get_shape(s17, 'TextBox 21')
    format_title(t17, "System Architecture: Block 1 (O1) & Block 2 (O2) Deep Dive", size=22)
    b17 = s17.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.1))
    items17 = [
        ("BLOCK 1: Graph Construction & Feature Engineering Pipeline [O1]:", ""),
        [
            "• Node Remapping: 64-bit non-contiguous transaction IDs mapped to [0, 203768] with bidirectional lookup mapping.",
            "• Edge Tensor Assembly: 234,355 directed transactions formatted into PyG edge_index [2, 234355] with edge timestamps.",
            "• Domain Topological Features (+6 dims): In-degree, out-degree, fan-out ratio, fan-in ratio, normalized time (t/49), neighbor Δt.",
            "• Leakage-Free Standardization: StandardScaler fitted strictly on Train partition (t ≤ 34); transforms Val and Test partitions.",
            "• Chronological Split Masks: Boolean masks for Train (t=1–34, 70.4%), Val (t=35–40, 16.8%), and Test (t=41–49, 12.8%)."
        ],
        ("BLOCK 2: Continuous Learnable Fourier Time Encoding [O2]:", ""),
        [
            "• Continuous Edge Time Delta: Δt = t_target - t_source (ensuring Δt ≥ 0 for causal consistency).",
            "• Fourier Projection: Φ(Δt) = √(2/d) · [cos(ω·Δt + φ) || sin(ω·Δt + φ)] mapping relative intervals into ℝ¹²⁸.",
            "• Trainable Parameters: ω, φ ∈ ℝ⁶⁴ initialized as learnable nn.Parameters updated via gradient backpropagation (∂L/∂ω).",
            "• Causal Neighbor Sampler: TemporalNeighborSampler restricts neighborhood to historical transactions (t_neighbor ≤ t_target, K=20)."
        ]
    ]
    format_content(b17, items17, font_size=11, space_after=4)

    # Slide 18: Deep Dive on Block 3 & Block 4 (O3 Backbone & Heads)
    s18 = prs.slides[17]
    t18 = get_shape(s18, 'TextBox 21')
    format_title(t18, "System Architecture: Block 3 & Block 4 (O3 Multi-Task) Deep Dive", size=22)
    b18 = s18.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.1))
    items18 = [
        ("BLOCK 3: TGAT Attention & Shared Representation Backbone:", ""),
        [
            "• 2-Layer TGAT Architecture: Stacked temporal attention layers (in_dim=172 → hidden_dim=128, 4 attention heads).",
            "• Time-Aware Attention Mechanism: Query q_i = [h_i || Φ(0)] W_Q, Key k_j = [h_j || Φ(Δt)] W_K, Value v_j = [h_j || Φ(Δt)] W_V.",
            "• Causal Aggregation: Computes attention weights α_ij solely over historical neighbors, producing unified node embeddings Z ∈ ℝ^(N×128)."
        ],
        ("BLOCK 4: Dedicated Multi-Pattern Classification Heads:", ""),
        [
            "• Shared Backbone Efficiency: A single forward pass generates embeddings shared across 3 parallel 2-layer FFN heads.",
            "• Circular Transfer Head: Detects wash-trading cycles (Mined via Tarjan's SCC cycle size ≥ 2) → Linear → ReLU → Sigmoid.",
            "• Layering Chain Head: Detects sequential peeling chains (Mined via temporal monotonic BFS paths ≥ 3 hops) → F1 = 0.91.",
            "• Smurfing Structuring Head: Detects fan-out dispersion & fan-in aggregation hubs (Mined via degree d ≥ 10) → F1 = 0.94.",
            "• Weighted Multi-Task Loss: L_total = λ₁·WBCE_circ + λ₂·WBCE_lay + λ₃·WBCE_smurf with w_pos ≈ 44.8 to conquer class imbalance."
        ]
    ]
    format_content(b18, items18, font_size=11, space_after=4)

    # Slide 19: Deep Dive on Block 5 (O4 XAI) & Block 6 (Operational Platform)
    s19 = prs.slides[18]
    t19 = get_shape(s19, 'TextBox 21')
    format_title(t19, "System Architecture: Block 5 (O4 XAI) & Block 6 (Platform) Deep Dive", size=22)
    b19 = s19.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.1))
    items19 = [
        ("BLOCK 5: Explainable Subgraph Generation & Automated SAR [O4]:", ""),
        [
            "• Alert Screening: Automatically activates GNNExplainer on transactions flagged with P(pattern) ≥ 0.50.",
            "• Edge Mask Optimization: Freezes model weights Θ and optimizes continuous edge mask M ∈ [0, 1] over 200 epochs via Mutual Information maximization: max_M MI(Y, G_s) = -log P_Θ(Y | A ⊙ σ(M)) + λ₁ H(M) + λ₂ ||M||₁.",
            "• Evidence Subgraph Extraction: Filters top-k causal edges, revealing chronological transaction trails and edge time deltas (Δt).",
            "• Automated SAR Dossier: Auto-generates narrative summary and standardized FinCEN-compliant SAR JSON files for download."
        ],
        ("BLOCK 6: Operational Radar Dashboard & Real-Time Node Inspector:", ""),
        [
            "• FastAPI REST Backend (Port 8000): Serves high-throughput endpoints (/api/status, /api/graph, /api/predict/{node_id}).",
            "• Interactive Streamlit Dashboard (Port 8501): Live forensic investigation interface with curated test cases and NetworkX visualization.",
            "• Real-Time Node Inspector: Allows compliance officers to query any transaction ID, inspect pattern risks, and view causal subgraphs."
        ]
    ]
    format_content(b19, items19, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 20: METHODOLOGY 1 (Temporal Graph Construction [O1])
    # =========================================================================
    s20 = prs.slides[19]
    t20 = get_shape(s20, 'TextBox 18')
    b20 = get_shape(s20, 'TextBox 19')
    format_title(t20, "Methodology 1: Temporal Graph Construction & Preprocessing [O1]", size=20)
    b20.top = Inches(1.8)
    b20.height = Inches(5.1)
    items20 = [
        ("Scope & Objective:", "Construct a directed, leakage-free temporal graph G = (V, E, T) from raw Elliptic Bitcoin CSV files with domain topological features."),
        ("Input Data Specifications:", "3 CSV files: elliptic_txs_features.csv (203,769×166), elliptic_txs_edgelist.csv (234,355 directed edges), elliptic_txs_classes.csv (ground truth)."),
        ("Methodological Operations:", ""),
        [
            "1. Continuous Node ID Remapping: Map 64-bit arbitrary transaction IDs to contiguous 0-indexed integers [0, 203768].",
            "2. Directed Edge Tensor Assembly: Remap 234,355 fund transfers into PyG edge_index [2, 234355] with edge timestamps.",
            "3. Domain Feature Engineering (+6 dims): Compute in-degree, out-degree, fan-out ratio, fan-in ratio, normalized timestamp (t/49), and neighbor temporal delta (166 → 172 dims).",
            "4. Zero-Leakage Normalization: Fit StandardScaler strictly on training nodes (t ≤ 34); transform validation and test splits.",
            "5. Chronological Split Masks: Generate boolean masks for Train (t=1–34), Val (t=35–40), and Test (t=41–49)."
        ],
        ("Deliverable Output:", "PyTorch Geometric Data container: X (203769, 172), edge_index (2, 234355), time (203769,), y (203769,), and chronological split masks.")
    ]
    format_content(b20, items20, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 21: METHODOLOGY 2 (Learnable Fourier Time Encoding [O2])
    # =========================================================================
    s21 = prs.slides[20]
    t21 = get_shape(s21, 'TextBox 18')
    b21 = get_shape(s21, 'TextBox 19')
    format_title(t21, "Methodology 2: Learnable Fourier Time Encoding [O2]", size=20)
    b21.top = Inches(1.8)
    b21.height = Inches(5.1)
    items21 = [
        ("Scope & Formulation:", "Project continuous edge time deltas Δt = t_target - t_source directly into attention projections using trainable Bochner Fourier embeddings."),
        ("Mathematical Formulation:", "Φ(Δt) = √(2/d) · [cos(ω₁Δt + φ₁), sin(ω₁Δt + φ₁), ..., cos(ω₆₄Δt + φ₆₄), sin(ω₆₄Δt + φ₆₄)] ∈ ℝ¹²⁸"),
        ("Core Algorithmic Mechanisms:", ""),
        [
            "• Trainable Frequency Vectors: ω, φ ∈ ℝ⁶⁴ initialized as learnable nn.Parameters (unlike static fixed sinusoidal baselines).",
            "• Gradient Backpropagation: ∂L/∂ω_k = -Δt · sin(ω_k Δt + φ_k) dynamically tunes attention bandwidth to burst laundering velocities.",
            "• Causal Neighborhood Sampling: TemporalNeighborSampler enforces t_neighbor ≤ t_target (K=20), preventing future data leakage."
        ],
        ("Deliverable Output:", "Time-aware attention projection tensors delivering an +8.47% test F1-score gain over fixed harmonic sinusoids.")
    ]
    format_content(b21, items21, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 22: METHODOLOGY 3 (TGAT Model Training & Pattern Detection [O3])
    # =========================================================================
    s22 = prs.slides[21]
    t22 = get_shape(s22, 'TextBox 18')
    b22 = get_shape(s22, 'TextBox 19')
    format_title(t22, "Methodology 3: Multi-Pattern TGAT Shared Backbone [O3]", size=20)
    b22.top = Inches(1.8)
    b22.height = Inches(5.1)
    items22 = [
        ("Shared-Encoder Architecture:", "2-layer TGATConv (172 → 128 → 128 dims, 4 attention heads) producing unified 128D latent node embeddings Z ∈ ℝ^(N×128)."),
        ("Dedicated Pattern Classification Heads:", ""),
        [
            "• Circular Transfer Head: 2-layer FFN → Sigmoid (Mined via Tarjan's SCC cycle size ≥ 2) — Test F1 = 0.88, AUC = 0.92.",
            "• Layering Chain Head: 2-layer FFN → Sigmoid (Mined via temporal monotonic BFS paths ≥ 3 hops) — Test F1 = 0.91, AUC = 0.95.",
            "• Smurfing Structuring Head: 2-layer FFN → Sigmoid (Mined via degree thresholding d ≥ 10) — Test F1 = 0.94, AUC = 0.97."
        ],
        ("Joint Multi-Task Loss:", "L_total = λ₁·WBCE_circ + λ₂·WBCE_lay + λ₃·WBCE_smurf with positive class weights (w_pos ≈ 44.8) resolving the severe 2.2% class imbalance."),
        ("Optimization Protocol:", "Adam Optimizer (LR=0.001), CosineAnnealingLR scheduler, batch size 1024, trained on Google Colab T4 GPU (~20 mins).")
    ]
    format_content(b22, items22, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 23: METHODOLOGY 4 (Explainability & SAR Reporting [O4])
    # =========================================================================
    s23 = prs.slides[22]
    t23 = get_shape(s23, 'TextBox 18')
    b23 = get_shape(s23, 'TextBox 19')
    format_title(t23, "Methodology 4: Explainability & Automated SAR Reporting [O4]", size=20)
    b23.top = Inches(1.8)
    b23.height = Inches(5.1)
    items23 = [
        ("Scope & Rationale:", "Extract minimal, time-ordered causal subgraph evidence trails and generate automated regulatory Suspicious Activity Reports (SAR)."),
        ("Algorithmic Workflow (GNNExplainer Integration):", ""),
        [
            "1. High-Risk Screening: Automatically select transactions flagged with P(pattern) ≥ 0.50 by Objective 3 heads.",
            "2. Edge Mask Optimization: Freeze model weights Θ; optimize continuous edge mask M ∈ [0, 1] over 200 epochs to maximize mutual information: max_M MI(Y, G_s) = -log P_Θ(Y=c | G_s = A ⊙ σ(M)) + λ₁ H(M) + λ₂ ||M||₁.",
            "3. Minimal Subgraph Extraction: Retain top-k causal edge paths forming the chronological fund audit trail.",
            "4. Automated SAR Narrative: Synthesize executive summary text and export standardized FinCEN-compliant SAR JSON dossiers."
        ],
        ("Deliverable Output:", "Interactive forensic evidence subgraphs and auditable JSON SAR dossiers deployed on the live dashboard.")
    ]
    format_content(b23, items23, font_size=11, space_after=4)

    # =========================================================================
    # SLIDE 24: EXPECTED OUTCOMES
    # =========================================================================
    s24 = prs.slides[23]
    t24 = get_shape(s24, 'TextBox 11')
    b24 = get_shape(s24, 'TextBox 12')
    format_title(t24, "Expected Outcomes & Research Deliverables", size=22)
    items24 = [
        ("1. High-Precision Continuous Temporal GNN:", "A robust TGAT model with Learnable Fourier Time Encoding achieving composite test F1 > 0.91 and AUC-ROC > 0.94 on the benchmark Elliptic Bitcoin dataset."),
        ("2. First Unified 3-Pattern Detection Framework:", "Simultaneous multi-label identification of Circular Transfers (F1=0.88), Layering Chains (F1=0.91), and Smurfing (F1=0.94) from shared 128D embeddings."),
        ("3. Quantified Fourier Encoding Superiority:", "Empirical verification demonstrating an +8.47% F1 improvement of learnable frequencies over static and fixed sinusoidal baselines."),
        ("4. Compliance-Ready SAR Explainability:", "Automated generation of minimal, time-ordered subgraph evidence dossiers via GNNExplainer for FinCEN and FATF regulatory submission."),
        ("5. Zero-Leakage Causal Guarantee:", "Formal verification that causal temporal neighborhood aggregation ensures sound real-time production deployment without future data leakage.")
    ]
    format_content(b24, items24, font_size=12, space_after=6)

    # =========================================================================
    # SLIDE 26: CONCLUSION
    # =========================================================================
    s26 = prs.slides[25]
    t26 = get_shape(s26, 'TextBox 11')
    b26 = get_shape(s26, 'TextBox 12')
    format_title(t26, "Conclusion & Project Summary", size=22)
    items26 = [
        ("Core Innovation:", "TemporalAML integrates Learnable Fourier Time Encoding directly inside Graph Attention, capturing continuous transaction velocity and burst dynamics in cryptocurrency networks."),
        ("Multi-Pattern Breakthrough:", "Transitions from non-actionable binary (illicit/licit) classification to unified multi-task detection of Circular Transfers, Layering Chains, and Smurfing from shared 128D embeddings."),
        ("Empirical Validation:", "Demonstrates strong performance across all three patterns (Circular F1=0.88, Layering F1=0.91, Smurfing F1=0.94) on 203,769 real-world Bitcoin transactions."),
        ("Explainable Compliance:", "GNNExplainer integration produces time-ordered subgraph evidence dossiers ready for regulatory Suspicious Activity Report (SAR) filing."),
        ("Operational Readiness:", "Backed by a fully functional FastAPI backend (Port 8000) and an interactive dark-mode dashboard with a Real-Time Node Inspector for compliance officers.")
    ]
    format_content(b26, items26, font_size=12, space_after=6)

    prs.save(pptx_path)
    print("SUCCESSFULLY APPLIED ALL MENTOR FEEDBACK TO TemporalAML_review_1(Anshu).pptx!")

if __name__ == '__main__':
    apply_feedback()
