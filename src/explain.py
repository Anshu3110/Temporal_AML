"""Explainability and Automated SAR Generation for TemporalAML.

Optimizes GNNExplainer-style edge and feature importance masks directly
against the real trained TGATEncoder (2-layer multi-head temporal attention
+ learnable Fourier time encoding) over its actual sampled 2-hop causal
neighborhood, extracts chronological evidence paths, and compiles
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
# GNNExplainer-style mask optimization against the REAL trained TGAT
# ==============================================================================
def _entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy, used to push masks toward hard 0/1 decisions."""
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))


def optimize_explanation_masks(
    encoder: nn.Module,
    head_submodule: nn.Module,
    x: torch.Tensor,
    sampler: TemporalNeighborSampler,
    target_node: int,
    target_time: float,
    epochs: int = 40,
    lr: float = 0.1,
    edge_size_reg: float = 0.005,
    edge_ent_reg: float = 0.1,
    feat_size_reg: float = 0.005,
    feat_ent_reg: float = 0.1,
) -> Dict[str, Any]:
    """Learns edge and feature importance masks against the actual trained
    TGATEncoder (real 2-layer multi-head attention + Fourier time encoding),
    following GNNExplainer's optimization objective (maximize predicted
    probability of the target class, regularized toward small, near-binary
    masks) — instead of explaining a disconnected linear surrogate.

    Only the target node's own 2-hop causal neighborhood, exactly as sampled
    by `sampler` during real inference, is perturbed: a learnable logit per
    Hop-1 edge and per Hop-2 edge gates that edge's attention weight via
    sigmoid(edge_mask) (see TGATLayer.forward), and a learnable per-feature
    logit gates the target node's own input features. Encoder/head weights
    stay frozen throughout.

    Args:
        encoder: Trained TGATEncoder (eval mode, frozen).
        head_submodule: The specific MultiTaskHead sub-network (e.g.
            head.head_lay) whose logit is being explained.
        x: Global node feature tensor [N, in_dim].
        sampler: The same TemporalNeighborSampler used during real inference.
        target_node: Global node index to explain.
        target_time: Observation timestamp for the target node.
        epochs: Number of Adam optimization steps.
        lr: Learning rate for the mask parameters.
        edge_size_reg: L1-style penalty encouraging few important edges.
        edge_ent_reg: Entropy penalty pushing edge masks toward 0 or 1.
        feat_size_reg: L1-style penalty encouraging few important features.
        feat_ent_reg: Entropy penalty pushing feature masks toward 0 or 1.

    Returns:
        Dict[str, Any]: Detached numpy arrays for edge/feature masks plus the
        exact Hop-1/Hop-2 neighbor structure they refer to, and the final
        predicted probability under the optimized (near-real) mask.
    """
    encoder.eval()
    head_submodule.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in head_submodule.parameters():
        p.requires_grad_(False)

    device = x.device
    node_t = torch.tensor([target_node], dtype=torch.long, device=device)
    time_t = torch.tensor([target_time], dtype=torch.float32, device=device)

    # Fix the real causal neighborhood once: sampling is deterministic given
    # (node, time), so the explanation targets exactly the edges the model
    # actually attended to during inference.
    nbrs_1, times_1, mask_1 = sampler.sample_batch(node_t, time_t)          # [1, M1]
    nbrs_1_flat, times_1_flat = nbrs_1.view(-1), times_1.view(-1)
    nbrs_2, times_2, mask_2 = sampler.sample_batch(nbrs_1_flat, times_1_flat)  # [M1, M2]

    in_dim = x.shape[1]
    init_val = 3.0  # sigmoid(3.0) ~= 0.95: masks start "mostly open"
    edge_mask_hop1 = nn.Parameter(init_val + 0.01 * torch.randn_like(mask_1, dtype=torch.float32))
    edge_mask_hop2 = nn.Parameter(init_val + 0.01 * torch.randn_like(mask_2, dtype=torch.float32))
    feat_mask = nn.Parameter(init_val + 0.01 * torch.randn(1, in_dim, device=device))

    optimizer = torch.optim.Adam([edge_mask_hop1, edge_mask_hop2, feat_mask], lr=lr)

    mask_1_f = mask_1.float()
    mask_2_f = mask_2.float()

    for _ in range(max(epochs, 1)):
        optimizer.zero_grad()

        h0_target = encoder.feat_proj(x[node_t] * torch.sigmoid(feat_mask))
        h0_nbrs_1 = encoder.feat_proj(x[nbrs_1_flat])
        h0_nbrs_2 = encoder.feat_proj(x[nbrs_2])

        h1_nbrs_1 = encoder.layers[0](
            h_target=h0_nbrs_1, t_target=times_1_flat,
            h_neighbors=h0_nbrs_2, t_neighbors=times_2, mask=mask_2,
            edge_mask=edge_mask_hop2,
        ).view(1, nbrs_1.shape[1], encoder.hidden_dim)

        h2_target = encoder.layers[1](
            h_target=h0_target, t_target=time_t,
            h_neighbors=h1_nbrs_1, t_neighbors=times_1, mask=mask_1,
            edge_mask=edge_mask_hop1,
        )

        logit = head_submodule(h2_target).squeeze()
        pred_loss = -F.logsigmoid(logit)

        em1 = torch.sigmoid(edge_mask_hop1) * mask_1_f
        em2 = torch.sigmoid(edge_mask_hop2) * mask_2_f
        fm = torch.sigmoid(feat_mask)

        n_edges = mask_1_f.sum() + mask_2_f.sum() + 1e-6
        size_loss = edge_size_reg * (em1.sum() + em2.sum()) / n_edges + feat_size_reg * fm.mean()
        ent_loss = (
            edge_ent_reg * ((_entropy(em1) * mask_1_f).sum() + (_entropy(em2) * mask_2_f).sum()) / n_edges
            + feat_ent_reg * _entropy(fm).mean()
        )

        loss = pred_loss + size_loss + ent_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_prob = torch.sigmoid(logit).item()
        edge_mask_hop1_np = (torch.sigmoid(edge_mask_hop1) * mask_1_f).squeeze(0).cpu().numpy()
        edge_mask_hop2_np = (torch.sigmoid(edge_mask_hop2) * mask_2_f).cpu().numpy()
        feat_mask_np = torch.sigmoid(feat_mask).squeeze(0).cpu().numpy()

    return {
        "edge_mask_hop1": edge_mask_hop1_np,      # [M1]
        "edge_mask_hop2": edge_mask_hop2_np,      # [M1, M2]
        "feat_mask": feat_mask_np,                # [in_dim]
        "nbrs_1": nbrs_1_flat.cpu().numpy(),      # [M1]
        "times_1": times_1_flat.cpu().numpy(),    # [M1]
        "mask_1": mask_1.squeeze(0).cpu().numpy(),  # [M1]
        "nbrs_2": nbrs_2.cpu().numpy(),           # [M1, M2]
        "times_2": times_2.cpu().numpy(),         # [M1, M2]
        "mask_2": mask_2.cpu().numpy(),           # [M1, M2]
        "final_probability": final_prob,
    }


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

    # 4-5. GNNExplainer-style mask optimization against the REAL trained TGAT
    # (real 2-layer multi-head attention + learnable Fourier time encoding,
    # over exactly the Hop-1/Hop-2 neighbors the model actually attends to —
    # not a disconnected linear surrogate over a separately-reconstructed subgraph).
    mask_result = optimize_explanation_masks(
        encoder=encoder,
        head_submodule=head_submodule,
        x=data.x,
        sampler=sampler,
        target_node=target_node,
        target_time=target_time,
        epochs=num_explainer_epochs,
    )

    # 6. Top Triggering Features (Top 10 from the optimized feature mask)
    target_feat_importances = mask_result["feat_mask"]
    top_feat_indices = np.argsort(-target_feat_importances)[:10]

    triggering_features: List[Dict[str, Any]] = []
    for idx in top_feat_indices:
        triggering_features.append({
            "feature_index": int(idx),
            "name": get_feature_name(int(idx)),
            "importance_score": round(float(target_feat_importances[idx]), 4),
        })

    # 7. Evidence Subgraph: the real Hop-1/Hop-2 causal edges the model
    # attended to, scored by their optimized attention-gate importance,
    # sorted chronologically.
    evidence_subgraph: List[Dict[str, Any]] = []
    raw_scores: List[float] = []

    nbrs_1, times_1, mask_1 = mask_result["nbrs_1"], mask_result["times_1"], mask_result["mask_1"]
    em1 = mask_result["edge_mask_hop1"]
    for i in range(len(nbrs_1)):
        if not mask_1[i]:
            continue
        score = float(em1[i])
        raw_scores.append(score)
        evidence_subgraph.append({
            "source_tx_id": int(id_map.get(int(nbrs_1[i]), int(nbrs_1[i]))),
            "target_tx_id": int(original_txid),
            "time_step": int(times_1[i]),
            "importance_weight": round(score, 4),
        })

    nbrs_2, times_2, mask_2 = mask_result["nbrs_2"], mask_result["times_2"], mask_result["mask_2"]
    em2 = mask_result["edge_mask_hop2"]
    for i in range(len(nbrs_1)):
        if not mask_1[i]:
            continue
        hop1_txid = int(id_map.get(int(nbrs_1[i]), int(nbrs_1[i])))
        for j in range(nbrs_2.shape[1]):
            if not mask_2[i, j]:
                continue
            score = float(em2[i, j])
            raw_scores.append(score)
            evidence_subgraph.append({
                "source_tx_id": int(id_map.get(int(nbrs_2[i, j]), int(nbrs_2[i, j]))),
                "target_tx_id": hop1_txid,
                "time_step": int(times_2[i, j]),
                "importance_weight": round(score, 4),
            })

    if evidence_subgraph:
        # Keep edges the optimizer marked important; if none clear 0.5,
        # fall back to the single most important edge so evidence is never
        # empty purely because the mask distribution is smooth.
        threshold_score = 0.5 if max(raw_scores) >= 0.5 else max(raw_scores) - 1e-6
        evidence_subgraph = [e for e in evidence_subgraph if e["importance_weight"] >= threshold_score]
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
