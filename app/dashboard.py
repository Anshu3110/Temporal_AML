"""TemporalAML Interactive Forensic Investigation & AML Benchmark Dashboard.

Provides a Streamlit interface for:
1. Live explainability & causal evidence path visualization for individual transactions.
2. Multi-pattern AML typology scoring (Circular, Layering, Smurfing).
3. Benchmark model performance comparison (F1, AUPRC, AUC-ROC, Precision, Recall).
4. Automated Suspicious Activity Report (SAR) narrative and JSON inspection.
"""

import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import EllipticDatasetLoader
from src.explain import explain_node, get_feature_name
from src.models import (
    LearnableFourierTimeEncoder,
    MultiTaskHead,
    TemporalNeighborSampler,
    TGATEncoder,
)

# ==============================================================================
# 1. Page Configuration & Custom CSS
# ==============================================================================
st.set_page_config(
    page_title="TemporalAML Forensic Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3a 100%);
        border: 1px solid #363d4e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #90a4ae;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
        color: #f8fafc;
    }
    .badge-illicit {
        background-color: #ef4444;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-licit {
        background-color: #10b981;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-unknown {
        background-color: #6b7280;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Resource Caching: Model & Data Loading
# ==============================================================================
@st.cache_resource(show_spinner="Loading TemporalAML Graph and Models...")
def load_cached_resources() -> Dict[str, Any]:
    """Loads graph data, temporal masks, id maps, and trained model checkpoints."""
    graph_path = PROJECT_ROOT / "data/processed/elliptic_graph.pt"
    checkpoint_path = PROJECT_ROOT / "artifacts/models/temporalaml_best.pth"

    loader = EllipticDatasetLoader()
    data = loader.load(graph_path)

    # Inductive test split masks (steps 41-49)
    train_mask, val_mask, test_mask = loader.get_split_masks(train_end=34, val_end=40)
    test_labeled_mask = test_mask & (data.y != -1)

    id_map = getattr(data, "id_map", loader.id_map)
    if not id_map:
        id_map = {i: i for i in range(data.num_nodes)}
    rev_id_map = {v: k for k, v in id_map.items()}

    test_nodes = test_labeled_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    test_txids = [int(id_map.get(int(idx), idx)) for idx in test_nodes]

    # Pre-select curated ~20 interesting test nodes (mix of illicit, smurfing, and layering)
    curated_candidates = [
        50510808, 2291756, 1803867, 55483333, 37291743, 218414030,
        12649359, 164964365, 164687321, 155829812, 156307670, 155816207,
        164686469, 155809660, 38375016, 12661812, 12780483, 12661656,
        12778479, 12661090
    ]
    curated_txids = [tx for tx in curated_candidates if tx in rev_id_map]

    # Load Model Checkpoint
    chk = torch.load(checkpoint_path, map_location="cpu")
    cfg = chk.get("config", {})

    edge_times = data.time[data.edge_index[0]].float()
    sampler = TemporalNeighborSampler(data.edge_index, edge_times, num_neighbors=20)

    encoder = TGATEncoder(
        in_dim=data.x.shape[1],
        hidden_dim=cfg.get("hidden_dim", 128),
        time_dim=cfg.get("time_dim", 64),
        num_layers=cfg.get("num_tgat_layers", 2),
        num_heads=cfg.get("num_heads", 4),
        max_neighbors=cfg.get("max_temporal_neighbors", 20),
        time_encoder_type="learnable",
        sampler=sampler,
    )
    head = MultiTaskHead(hidden_dim=cfg.get("hidden_dim", 128), head_hidden_dim=64)

    encoder.load_state_dict(chk["encoder_state_dict"])
    head.load_state_dict(chk["head_state_dict"])
    encoder.eval()
    head.eval()

    return {
        "data": data,
        "id_map": id_map,
        "rev_id_map": rev_id_map,
        "test_nodes": test_nodes,
        "test_txids": test_txids,
        "curated_txids": curated_txids,
        "encoder": encoder,
        "head": head,
        "config": cfg,
    }


# ==============================================================================
# 3. Sidebar: Transaction Selection & System Meta
# ==============================================================================
res = load_cached_resources()
data = res["data"]
id_map = res["id_map"]
rev_id_map = res["rev_id_map"]
curated_txids = res["curated_txids"]
test_txids = res["test_txids"]
encoder = res["encoder"]
head = res["head"]

st.sidebar.title("🛡️ TemporalAML")
st.sidebar.caption("Continuous-Time GNN for Cryptocurrency AML")
st.sidebar.divider()

selection_mode = st.sidebar.radio(
    "Transaction Selection Pool",
    ["Curated Flagged Transactions (~20)", "All Test-Set Transactions (9,973)"],
    index=0,
)

if selection_mode.startswith("Curated"):
    tx_pool = curated_txids
    selected_txid = st.sidebar.selectbox("Select Target Transaction ID", tx_pool, index=0)
else:
    tx_pool = test_txids
    # Provide a text search or selectbox
    search_query = st.sidebar.text_input("Filter by TxID prefix", "")
    if search_query:
        filtered = [tx for tx in tx_pool if str(tx).startswith(search_query)]
        tx_pool = filtered if filtered else tx_pool
    selected_txid = st.sidebar.selectbox("Select Target Transaction ID", tx_pool[:1000], index=0)

target_node_idx = rev_id_map[selected_txid]
target_time_val = int(data.time[target_node_idx].item())
raw_label = int(data.y[target_node_idx].item())
label_str = "Illicit (1)" if raw_label == 1 else ("Licit (0)" if raw_label == 0 else "Unknown (-1)")

st.sidebar.subheader("Selected Transaction Context")
st.sidebar.markdown(f"**Transaction ID:** `{selected_txid}`")
st.sidebar.markdown(f"**Internal Node Index:** `{target_node_idx}`")
st.sidebar.markdown(f"**Time Step:** `{target_time_val}` (Test Split: 41–49)")
if raw_label == 1:
    st.sidebar.markdown('**Ground Truth:** <span class="badge-illicit">ILLICIT</span>', unsafe_allow_html=True)
elif raw_label == 0:
    st.sidebar.markdown('**Ground Truth:** <span class="badge-licit">LICIT</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('**Ground Truth:** <span class="badge-unknown">UNKNOWN</span>', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.subheader("Model Configuration")
st.sidebar.markdown(
    """
    - **Backbone:** 2-Layer Inductive TGAT
    - **Time Encoding:** Learnable Bochner Fourier ($\omega \in \mathbb{R}^{32}$)
    - **Attention Heads:** 4 Heads ($d_k=32$)
    - **Temporal Horizon:** Strictly Causal ($t_e \le t$)
    - **Supervision:** Multi-Typology Multi-Task
    """
)


# ==============================================================================
# 4. Main Panel: 3 Tabs
# ==============================================================================
tab_investigate, tab_benchmarks, tab_about = st.tabs([
    "🔍 Investigate Transaction",
    "📊 Model Benchmarks & Ablation",
    "ℹ️ Architecture & Summary",
])


# ==============================================================================
# TAB 1: INVESTIGATE
# ==============================================================================
with tab_investigate:
    st.header(f"Forensic Investigation: TxID #{selected_txid}")
    st.caption(f"Chronological Time Step {target_time_val} • Graph Evidence Subgraph & Explanation")

    # Fast Inference for Gauges
    with torch.no_grad():
        t_node_tensor = torch.tensor([target_node_idx])
        t_time_tensor = torch.tensor([float(target_time_val)])
        h = encoder(data.x, t_node_tensor, t_time_tensor)
        p_lay, p_smurf = head(h)
        p_l = float(p_lay.item())
        p_s = float(p_smurf.item())
        composite_risk = max(p_l, p_s)

    col1, col2, col3 = st.columns(3)

    def get_risk_color(val: float) -> str:
        if val >= 0.5:
            return "#ef4444"
        elif val >= 0.2:
            return "#f59e0b"
        return "#10b981"

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Composite AML Risk</div>
                <div class="metric-value" style="color: {get_risk_color(composite_risk)}">{composite_risk:.1%}</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Max Across Typologies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Layering Chain Risk</div>
                <div class="metric-value" style="color: {get_risk_color(p_l)}">{p_l:.1%}</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Peeling Chain Head</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Smurfing Structuring Risk</div>
                <div class="metric-value" style="color: {get_risk_color(p_s)}">{p_s:.1%}</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Fan-In/Out Hub Head</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive Pattern Gauge / Bar Comparison
    fig_bars = go.Figure()
    categories = ["Layering Chain", "Smurfing Structuring"]
    scores = [p_l, p_s]
    bar_colors = [get_risk_color(v) for v in scores]

    fig_bars.add_trace(
        go.Bar(
            x=scores,
            y=categories,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#1f2937", width=1.5)),
            text=[f"{v:.2%}" for v in scores],
            textposition="auto",
        )
    )
    fig_bars.update_layout(
        title="Multi-Pattern Typology Scores",
        xaxis=dict(title="Predicted Illicit Probability", range=[0, 1.0], tickformat=".0%"),
        yaxis=dict(title="AML Typology Head"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # Live Explainer Pipeline Execution
    st.subheader("⚡ Live Explainability & Evidence Path Extraction")
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        explainer_head_choice = st.selectbox(
            "Target Head for Explanation",
            ["auto (Highest Probability)", "layering", "smurfing"],
            index=0,
        )
    with col_ctrl2:
        explainer_epochs = st.slider("Explainer Optimization Epochs", min_value=10, max_value=60, value=25, step=5)

    with st.spinner("Executing GNNExplainer live over causal subgraph..."):
        head_key = explainer_head_choice.split()[0]
        sar_result = explain_node(
            target_node=target_node_idx,
            target_time=float(target_time_val),
            selected_head=head_key,
            num_explainer_epochs=explainer_epochs,
            save_sar=False,
        )

    # Render Evidence Network Subgraph
    st.subheader("🕸️ Causal Evidence Subgraph")
    evidence_edges = sar_result.get("evidence_subgraph", [])

    if evidence_edges:
        st.write(f"Identified **{len(evidence_edges)}** high-importance causal transaction path(s):")

        # Build NetworkX DiGraph
        G = nx.DiGraph()
        G.add_node(selected_txid, time_step=target_time_val, is_target=True)

        for edge in evidence_edges:
            s_id = edge["source_tx_id"]
            d_id = edge["target_tx_id"]
            t_edge = edge["time_step"]
            w = edge["importance_weight"]
            dt = target_time_val - t_edge

            G.add_node(s_id, time_step=t_edge, is_target=(s_id == selected_txid))
            G.add_node(d_id, time_step=target_time_val if d_id == selected_txid else t_edge, is_target=(d_id == selected_txid))
            G.add_edge(s_id, d_id, weight=w, time_step=t_edge, delta_t=dt)

        # Plot with Matplotlib for clear edge labels and directional arrows
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        plt.style.use("default")
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")

        try:
            pos = nx.spring_layout(G, seed=42, k=1.2)
        except Exception:
            pos = nx.circular_layout(G)

        # Color nodes by time step
        node_times = [G.nodes[n].get("time_step", target_time_val) for n in G.nodes()]
        norm = mcolors.Normalize(vmin=min(node_times) - 0.5, vmax=max(node_times) + 0.5)
        cmap = cm.plasma

        node_colors = [
            "#ef4444" if G.nodes[n].get("is_target", False) else cmap(norm(G.nodes[n]["time_step"]))
            for n in G.nodes()
        ]
        node_sizes = [900 if G.nodes[n].get("is_target", False) else 550 for n in G.nodes()]

        # Draw Nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#ffffff",
            linewidths=1.5,
        )

        # Draw Edges
        edge_widths = [1.0 + 3.0 * G[u][v].get("weight", 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#60a5fa",
            width=edge_widths,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        # Draw Labels
        labels = {n: f"{n}\n(t={G.nodes[n]['time_step']})" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_color="#f9fafb", font_weight="bold")

        # Edge labels: Delta t
        edge_labels = {(u, v): f"Δt={G[u][v]['delta_t']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, font_color="#fcd34d", bbox=dict(boxstyle="round,pad=0.2", fc="#1f2937", ec="#374151", lw=0.5))

        ax.set_title(f"Evidence Subgraph for Tx #{selected_txid} (Red = Target, Color = Time Step)", color="#f3f4f6", fontsize=11, fontweight="bold")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ℹ️ Target node operates as an isolated transaction or immediate root in the causal graph; no upstream incoming causal edges exceeded the 0.5 importance threshold.")

    # Top Triggering Features Table
    st.subheader("🔬 Top Triggering Features (Node Feature Mask)")
    top_feats = sar_result.get("triggering_features", [])
    if top_feats:
        df_feats = pd.DataFrame(top_feats)
        df_feats.columns = ["Feature Index", "Feature Taxonomy Name", "Attribution Importance"]

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.dataframe(df_feats, use_container_width=True, hide_index=True)
        with col_f2:
            fig_feat = px.bar(
                df_feats.iloc[::-1],
                x="Attribution Importance",
                y="Feature Taxonomy Name",
                orientation="h",
                title="Top Feature Contributions (GNNExplainer)",
                template="plotly_dark",
            )
            fig_feat.update_layout(height=340, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_feat, use_container_width=True)

    # Collapsible SAR JSON Narrative
    st.subheader("📋 Suspicious Activity Report (SAR)")
    st.markdown(f"**Executive Narrative:** *{sar_result.get('summary', '')}*")

    with st.expander("📄 View Full Standardized SAR JSON Document", expanded=False):
        st.json(sar_result)
        st.download_button(
            label="💾 Download SAR JSON File",
            data=json.dumps(sar_result, indent=2),
            file_name=f"SAR_TX_{selected_txid}.json",
            mime="application/json",
        )


# ==============================================================================
# TAB 2: BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.header("📈 Model Benchmarks & Comparative Evaluation")
    st.caption("Chronological Out-of-Time Test Split (Steps 41–49 • 9,973 Labeled Transactions)")

    csv_path = PROJECT_ROOT / "artifacts/logs/final_results.csv"
    png_path = PROJECT_ROOT / "artifacts/logs/auprc_comparison.png"
    md_path = PROJECT_ROOT / "artifacts/logs/final_results.md"

    if csv_path.exists():
        df_bm = pd.read_csv(csv_path)

        # Summary Metrics Table
        st.subheader("1. Comprehensive Model Comparison Table")
        st.dataframe(
            df_bm.style.format({
                "F1-Score": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "AUC-ROC": "{:.4f}",
                "AUPRC": "{:.4f}",
            }).highlight_max(subset=["F1-Score", "AUPRC", "AUC-ROC"], color="#1e3a8a"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Interactive Comparison Chart
        st.subheader("2. Interactive Metric Comparison (F1 vs AUPRC vs AUC-ROC)")
        metric_choice = st.multiselect(
            "Select Metrics to Plot",
            ["F1-Score", "AUPRC", "AUC-ROC", "Precision", "Recall"],
            default=["F1-Score", "AUPRC"],
        )

        if metric_choice:
            fig_compare = go.Figure()
            for m in metric_choice:
                if m in df_bm.columns:
                    fig_compare.add_trace(go.Bar(name=m, x=df_bm["Model"], y=df_bm[m]))
            fig_compare.update_layout(
                barmode="group",
                title="Model Performance Across Metrics",
                template="plotly_dark",
                xaxis_tickangle=-25,
                margin=dict(l=20, r=20, t=40, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")

        # Ablation and AUPRC Bar Chart Artifact
        st.subheader("3. Benchmark Visualization Artifact (artifacts/logs/auprc_comparison.png)")
        if png_path.exists():
            st.image(str(png_path), caption="AUPRC Performance across all 7 evaluated architectures.", use_container_width=True)

        st.markdown("---")

        # Statistical Significance Summary
        st.subheader("4. Paired Bootstrap Significance Audit")
        st.markdown(
            """
            > **Evaluation Protocol:**
            > - **Iterations:** 1,000 paired bootstrap resamples of the test set ($N = 9,973$)
            > - **Comparison:** TemporalAML (Learnable Fourier) vs. Static-TGAT (Fixed Sinusoidal)
            > - **Confidence Interval:** 95% empirical percentile CI on $\Delta F_1 = F_1(\\text{TemporalAML}) - F_1(\\text{Static-TGAT})$
            > - **Observed Delta:** `ΔF1 = -0.0016` (95% CI: `[-0.0072, +0.0043]`, $p = 0.701$)
            """
        )
    else:
        st.warning("Benchmark results file `artifacts/logs/final_results.csv` not found. Please run `python3 src/evaluate.py` first.")


# ==============================================================================
# TAB 3: ABOUT
# ==============================================================================
with tab_about:
    st.header("ℹ️ About TemporalAML")
    st.markdown(
        """
        **TemporalAML** is a temporal Graph Neural Network (temporal-GNN) research and detection framework
        developed for anti-money laundering across cryptocurrency transaction graphs. The architecture incorporates
        learnable Fourier time encoding with Temporal Graph Attention Networks (TGAT) to simultaneously identify
        circular transfers, layering chains, and smurfing typologies on dynamic transaction streams. Designed for
        explainability and regulatory auditing, the platform provides end-to-end capabilities spanning temporal graph
        construction, multi-task AML pattern classification, and automated Suspicious Activity Report (SAR) narrative generation.
        """
    )

    st.markdown("---")
    st.subheader("Architecture Overview")

    st.markdown(
        """
        ```mermaid
        graph TD
            A[Raw Bitcoin Transactions] --> B[EllipticDatasetLoader]
            B --> C[Temporal Causal Edge Index]
            C --> D[Causal Temporal Neighbor Sampler]
            D -->|t_edge <= t_target| E[TGAT Inductive Layers]
            F[Learnable Fourier Time Encoder] -->|Bochner Theorem| E
            E --> G[Node Embedding h_i]
            G --> H1[Layering Chain Head]
            G --> H2[Smurfing Structuring Head]
            H1 --> I[Composite AML Risk Score]
            H2 --> I
            I --> J[GNNExplainer Attribution]
            J --> K[SAR Narrative & Evidence Graph]
        ```
        """
    )

    st.markdown("---")
    st.subheader("Key Architectural Pillars")
    st.markdown(
        """
        1. **Continuous Learnable Fourier Time Encodings:**
           Instead of hand-tuned sinusoidal encodings, the network optimizes Fourier frequencies $\omega \in \mathbb{R}^{d_T/2}$ via backpropagation according to Bochner's theorem:
           $$\Phi(\Delta t) = \sqrt{\\frac{2}{d_T}} \left[ \cos(\omega \Delta t) \parallel \sin(\omega \Delta t) \\right]$$

        2. **Strict Causal Sampling with Zero Lookahead:**
           Temporal edges are sampled strictly backwards in time ($t_{\\text{edge}} \le t_{\\text{target}}$), guaranteeing zero future data leakage across dynamic evaluation windows.

        3. **Multi-Pattern Topological Supervision:**
           Addresses severe class imbalance by co-training on foundational structural typologies:
           - **Layering:** Sequential peeling chains with bounded fan-out ($\le 3$) across intermediate conduits.
           - **Smurfing:** Temporal hub aggregation/dispersion fan-in and fan-out structures.
           *(Note: Circular cycles are mathematically non-existent in raw Bitcoin UTXO transaction graphs).*

        4. **Forensic SAR Explainability:**
           Integrated with PyTorch Geometric `GNNExplainer` to produce compact, auditable causal subgraphs and automated regulatory JSON narratives.
        """
    )
