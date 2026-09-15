"""Explainability and Automated SAR Generation for TemporalAML.

Uses PyTorch Geometric's Explainer with GNNExplainer to compute edge_mask
and node_feat_mask, extracts chronological evidence paths, and compiles
Suspicious Activity Report (SAR) JSON narratives.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import MessagePassing

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import EllipticDatasetLoader
from src.models import (
    LearnableFourierTimeEncoder,
    MultiTaskHead,
    TemporalNeighborSampler,
    TGATEncoder,
)


# ==============================================================================
# Feature Taxonomy (Elliptic Bitcoin Dataset)
# ==============================================================================
FEATURE_NAMES: Dict[int, str] = {
    0: "transaction_fee",
    1: "input_count",
    2: "output_count",
    3: "transaction_volume_btc",
    4: "transaction_duration",
    5: "in_degree_local",
    6: "out_degree_local",
    7: "btc_transferred_total",
    8: "mean_input_volume",
    9: "mean_output_volume",
    10: "std_input_volume",
    11: "std_output_volume",
    12: "fees_to_volume_ratio",
    13: "output_to_input_ratio",
    14: "min_input_volume",
    15: "max_input_volume",
    16: "min_output_volume",
    17: "max_output_volume",
    18: "aggregated_tx_count_1hop",
    19: "aggregated_mean_fee_1hop",
    20: "aggregated_std_fee_1hop",
}


def get_feature_name(idx: int) -> str:
    """Returns human-readable name for a feature index or descriptive default."""
    if idx in FEATURE_NAMES:
        return FEATURE_NAMES[idx]
    elif idx < 93:
        return f"feature_{idx+1}_local"
    else:
        return f"feature_{idx+1}_aggregated_neighbor"


# ==============================================================================
# Explainer MessagePassing Model Wrapper
# ==============================================================================
class ExplainerGNNWrapper(MessagePassing):
    """MessagePassing wrapper for GNNExplainer compatibility.

    Maps node features through the trained feature projection, performs
    message passing over the local causal temporal subgraph, and applies
    the trained classification head for the target typology.
    """

    def __init__(
        self,
        in_dim: int = 165,
        hidden_dim: int = 128,
        head_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(aggr="add")
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.head = head_module or nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for GNNExplainer node-level classification."""
        h = F.relu(self.feat_proj(x))
        if edge_index.numel() > 0:
            out = self.propagate(edge_index, x=h)
            h = F.relu(out + h)
        prob = torch.sigmoid(self.head(h))
        return prob

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


# ==============================================================================
# Subgraph Extraction Helper
# ==============================================================================
def extract_causal_subgraph(
    target_node: int,
    target_time: float,
    edge_index: torch.Tensor,
    node_times: torch.Tensor,
    k_hops: int = 2,
    max_neighbors_per_hop: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], List[int]]:
    """Extracts a k-hop causal temporal subgraph around a target node.

    Args:
        target_node: Global index of the target node.
        target_time: Timestamp of the target node.
        edge_index: Full graph connectivity [2, E].
        node_times: Node timestamps [N].
        k_hops: Number of hops backwards in time.
        max_neighbors_per_hop: Maximum neighbors to include per hop.

    Returns:
        Tuple:
            - sub_edge_index: Local edge index [2, E_sub]
            - sub_edge_times: Edge timestamps [E_sub]
            - global_to_local: Dict mapping global node IDs to local indices
            - local_to_global: List mapping local indices back to global node IDs
    """
    src_all = edge_index[0].cpu().numpy()
    dst_all = edge_index[1].cpu().numpy()
    times_all = node_times.cpu().numpy()

    # Incoming edges dictionary: dst -> list of (src, edge_time)
    from collections import defaultdict
    in_edges = defaultdict(list)
    for s, d in zip(src_all, dst_all):
        t_edge = float(times_all[s])
        in_edges[d].append((int(s), t_edge))

    visited_nodes = {target_node}
    subgraph_edges: List[Tuple[int, int, float]] = []

    current_frontier = [(target_node, target_time)]

    for _ in range(k_hops):
        next_frontier = []
        for curr_node, curr_time in current_frontier:
            incoming = in_edges.get(curr_node, [])
            # Filter causal incoming edges: edge_time <= curr_time (and s != curr_node)
            causal = [(s, t) for s, t in incoming if t <= curr_time and s != curr_node]
            # Sort most-recent-first
            causal.sort(key=lambda x: x[1], reverse=True)
            sampled = causal[:max_neighbors_per_hop]

            for s, t in sampled:
                subgraph_edges.append((s, curr_node, t))
                if s not in visited_nodes:
                    visited_nodes.add(s)
                    next_frontier.append((s, t))
        current_frontier = next_frontier

    # Ensure target_node is local index 0
    local_to_global = [target_node] + [n for n in visited_nodes if n != target_node]
    global_to_local = {g: l for l, g in enumerate(local_to_global)}

    if subgraph_edges:
        sub_src = [global_to_local[s] for s, _, _ in subgraph_edges]
        sub_dst = [global_to_local[d] for _, d, _ in subgraph_edges]
        sub_times = [t for _, _, t in subgraph_edges]
        sub_edge_index = torch.tensor([sub_src, sub_dst], dtype=torch.long)
        sub_edge_times = torch.tensor(sub_times, dtype=torch.float32)
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)
        sub_edge_times = torch.empty(0, dtype=torch.float32)

    return sub_edge_index, sub_edge_times, global_to_local, local_to_global


# ==============================================================================
# SAR Generator Pipeline
# ==============================================================================
def explain_node(
    target_node: int,
    target_time: Optional[float] = None,
    checkpoint_path: str = "artifacts/models/temporalaml_best.pth",
    graph_path: str = "data/processed/elliptic_graph.pt",
    threshold: float = 0.5,
    selected_head: str = "auto",
    num_explainer_epochs: int = 40,
    save_sar: bool = True,
    sar_dir: str = "artifacts/sar_reports",
) -> Dict[str, Any]:
    """Runs the full GNNExplainer and SAR generation pipeline for a target node.

    Args:
        target_node: Global node index (0..N-1).
        target_time: Optional observation time step (defaults to node's time).
        checkpoint_path: Path to best trained weights.
        graph_path: Path to preprocessed graph.
        threshold: Decision threshold for pattern flags.
        selected_head: Head to explain ('circular', 'layering', 'smurfing', or 'auto').
        num_explainer_epochs: Optimization epochs for GNNExplainer.
        save_sar: If True, writes SAR JSON report to disk.
        sar_dir: Destination directory for SAR reports.

    Returns:
        Dict[str, Any]: Generated SAR narrative and structured explanation report.
    """
    device = torch.device("cpu")  # Explainer runs comfortably and deterministically on CPU

    # 1. Load Graph & ID Map
    loader = EllipticDatasetLoader()
    data = loader.load(graph_path)
    id_map = getattr(data, "id_map", loader.id_map)

    if not id_map:
        # Fallback identity map if not saved
        id_map = {i: i for i in range(data.num_nodes)}

    original_txid = id_map.get(target_node, target_node)

    if target_time is None:
        target_time = float(data.time[target_node].item())

    # 2. Load Trained Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})

    hidden_dim = cfg.get("hidden_dim", 128)
    time_dim = cfg.get("time_dim", 64)
    num_heads = cfg.get("num_heads", 4)
    num_layers = cfg.get("num_tgat_layers", 2)
    max_neighbors = cfg.get("max_temporal_neighbors", 20)

    edge_times = data.time[data.edge_index[0]].float()
    sampler = TemporalNeighborSampler(data.edge_index, edge_times, num_neighbors=max_neighbors)

    encoder = TGATEncoder(
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_neighbors=max_neighbors,
        time_encoder_type="learnable",
        sampler=sampler,
    )
    head = MultiTaskHead(hidden_dim=hidden_dim, head_hidden_dim=64)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    encoder.eval()
    head.eval()

    # 3. Model Predictions on Target Node
    with torch.no_grad():
        t_node_tensor = torch.tensor([target_node])
        t_time_tensor = torch.tensor([target_time])
        h = encoder(data.x, t_node_tensor, t_time_tensor)
        p_illicit, p_lay, p_smurf = head(h)

    probs = {
        "illicit": float(p_illicit.item()),
        "layering": float(p_lay.item()),
        "smurfing": float(p_smurf.item()),
    }

    pattern_names = {
        "illicit": "Illicit Activity",
        "layering": "Layering Chain",
        "smurfing": "Smurfing Structuring",
    }

    # Flagged patterns
    flagged = [pattern_names[k] for k, v in probs.items() if v >= threshold]
    if not flagged:
        # If none strictly exceeds threshold, flag top risk
        top_k = max(probs, key=probs.get)
        flagged = [f"{pattern_names[top_k]} (Max Risk Score: {probs[top_k]:.2f})"]

    # Select head to explain
    if selected_head == "auto":
        target_head_key = max(probs, key=probs.get)
    else:
        target_head_key = selected_head.lower()

    head_map = {
        "illicit": head.head_illicit,
        "layering": head.head_lay,
        "smurfing": head.head_smurf,
    }
    head_submodule = head_map.get(target_head_key, head.head_lay)

    # 4. Causal Subgraph Extraction
    sub_edge_index, sub_edge_times, g_to_l, l_to_g = extract_causal_subgraph(
        target_node=target_node,
        target_time=target_time,
        edge_index=data.edge_index,
        node_times=data.time,
        k_hops=2,
        max_neighbors_per_hop=15,
    )

    sub_nodes = torch.tensor(l_to_g, dtype=torch.long)
    sub_x = data.x[sub_nodes]
    target_local_idx = 0

    # 5. Explainer Execution
    wrapper = ExplainerGNNWrapper(
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        head_module=head_submodule,
    )
    # Transfer trained feature projection weights
    wrapper.feat_proj.load_state_dict(encoder.feat_proj.state_dict())

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=num_explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        ),
    )

    if sub_edge_index.numel() > 0:
        explanation = explainer(sub_x, sub_edge_index, index=target_local_idx)
        raw_edge_mask = explanation.edge_mask.detach().cpu().numpy()
        raw_node_mask = explanation.node_mask.detach().cpu().numpy()
    else:
        # Single isolated node
        raw_edge_mask = np.array([], dtype=float)
        # Attribute perturbation
        raw_node_mask = np.abs(sub_x.detach().cpu().numpy())

    # 6. Top Triggering Features (Top 10 from node_feat_mask)
    target_feat_importances = raw_node_mask[target_local_idx]
    top_feat_indices = np.argsort(-target_feat_importances)[:10]

    triggering_features: List[Dict[str, Any]] = []
    for idx in top_feat_indices:
        triggering_features.append({
            "feature_index": int(idx),
            "name": get_feature_name(int(idx)),
            "importance_score": round(float(target_feat_importances[idx]), 4),
        })

    # 7. Evidence Subgraph (Edges with edge_mask >= 0.5, sorted chronologically)
    evidence_subgraph: List[Dict[str, Any]] = []
    if len(raw_edge_mask) > 0:
        src_local = sub_edge_index[0].numpy()
        dst_local = sub_edge_index[1].numpy()
        times_np = sub_edge_times.numpy()

        for idx, score in enumerate(raw_edge_mask):
            if score >= 0.5 or (len(raw_edge_mask) <= 5 and score > 0.2):
                src_g = l_to_g[src_local[idx]]
                dst_g = l_to_g[dst_local[idx]]
                evidence_subgraph.append({
                    "source_tx_id": int(id_map.get(src_g, src_g)),
                    "target_tx_id": int(id_map.get(dst_g, dst_g)),
                    "time_step": int(times_np[idx]),
                    "importance_weight": round(float(score), 4),
                })

        # Sort evidence edges chronologically by time_step
        evidence_subgraph.sort(key=lambda e: (e["time_step"], -e["importance_weight"]))

    # 8. Plain-English Narrative Summary
    pattern_summary_str = ", ".join(flagged)
    top_feature_names = [f["name"] for f in triggering_features[:3]]
    hop_count = len(evidence_subgraph)

    summary_text = (
        f"Transaction {original_txid} at time step {int(target_time)} flagged for {pattern_summary_str} "
        f"with {len(flagged)} suspicious typologies, supported by a {hop_count}-edge causal transaction path "
        f"and driven by primary feature anomalies in {', '.join(top_feature_names)}."
    )

    # 9. Assemble SAR Document
    sar_report = {
        "sar_id": f"SAR-TX-{original_txid}",
        "target_tx_id": int(original_txid),
        "target_node_index": int(target_node),
        "time_step": int(target_time),
        "predicted_probabilities": probs,
        "explained_pattern": pattern_names.get(target_head_key, target_head_key),
        "flagged_patterns": flagged,
        "triggering_features": triggering_features,
        "evidence_subgraph": evidence_subgraph,
        "summary": summary_text,
    }

    # 10. Save SAR Report
    if save_sar:
        out_dir = Path(sar_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"sar_{original_txid}.json"
        with open(report_path, "w") as f:
            json.dump(sar_report, f, indent=2)
        print(f"\n✅ SAR report saved to: {report_path.resolve()}")

    return sar_report


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="TemporalAML Explainer & SAR Report Generator")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--node-idx", type=int, default=None, help="Target node index (0..N-1)")
    group.add_argument("--tx-id", type=int, default=None, help="Original Bitcoin transaction ID")

    parser.add_argument("--time-step", type=float, default=None, help="Observation time step")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision probability threshold")
    parser.add_argument(
        "--head",
        type=str,
        default="auto",
        choices=["auto", "illicit", "circular", "layering", "smurfing"],
        help="Specific pattern head to explain",
    )
    parser.add_argument("--epochs", type=int, default=30, help="GNNExplainer optimization epochs")
    parser.add_argument("--sar-dir", type=str, default="artifacts/sar_reports", help="Output SAR directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Default to node 0 or find a representative illicit node if not provided
    loader = EllipticDatasetLoader()
    graph_path = Path("data/processed/elliptic_graph.pt")
    if not graph_path.exists():
        data = loader.build_graph()
        loader.save()
    else:
        data = loader.load(graph_path)

    id_map = getattr(data, "id_map", loader.id_map)
    raw_to_idx = {tx_id: idx for idx, tx_id in id_map.items()} if id_map else {}

    target_node = args.node_idx
    if args.tx_id is not None:
        target_node = raw_to_idx.get(args.tx_id, None)
        if target_node is None:
            raise ValueError(f"Transaction ID {args.tx_id} not found in graph dataset!")

    if target_node is None:
        # Automatically pick the first illicit node from the test set
        test_illicit = ((data.y == 1) & data.test_mask).nonzero(as_tuple=True)[0]
        if len(test_illicit) > 0:
            target_node = int(test_illicit[0].item())
            print(f"Auto-selected representative test illicit node: Index {target_node} (txId: {id_map.get(target_node, target_node)})")
        else:
            target_node = 0

    print("=" * 80)
    print(f"TEMPORALAML EXPLAINER & SAR GENERATION (Node {target_node})")
    print("=" * 80)

    sar = explain_node(
        target_node=target_node,
        target_time=args.time_step,
        threshold=args.threshold,
        selected_head=args.head,
        num_explainer_epochs=args.epochs,
        save_sar=True,
        sar_dir=args.sar_dir,
    )

    print("\n" + "-" * 80)
    print("SUSPICIOUS ACTIVITY REPORT SUMMARY")
    print("-" * 80)
    print(f"Target Transaction ID : {sar['target_tx_id']}")
    print(f"Time Step             : {sar['time_step']}")
    print(f"Flagged Typologies    : {', '.join(sar['flagged_patterns'])}")
    print(f"Predicted Probs       : {sar['predicted_probabilities']}")
    print("\nTop Triggering Features:")
    for f in sar["triggering_features"][:5]:
        print(f"  • {f['name']:<28} : {f['importance_score']:.4f}")
    print(f"\nEvidence Subgraph     : {len(sar['evidence_subgraph'])} chronological payment edges")
    for e in sar["evidence_subgraph"][:3]:
        print(f"  • {e['source_tx_id']} -> {e['target_tx_id']} (step {e['time_step']}, weight {e['importance_weight']:.2f})")
    print(f"\nNarrative Summary:\n  \"{sar['summary']}\"")
    print("=" * 80)
