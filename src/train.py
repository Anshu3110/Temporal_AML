"""Training pipeline for TemporalAML and baseline TGAT multi-pattern detection.

Trains:
1. TemporalAML (Learnable Fourier Time Encoder + TGAT + 2-Pattern MultiTaskHead)
2. Static-TGAT (Fixed Sinusoidal Time Encoder + TGAT + 2-Pattern MultiTaskHead)
3. TemporalAML Ablation (Mean-pooling without time encoding + 2-Pattern MultiTaskHead)

Tracks per-head validation F1 on TensorBoard and checkpoints best models to artifacts/models/.
"""

import copy
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import EllipticDatasetLoader
from src.evaluate import find_optimal_threshold
from src.models import (
    FixedTimeEncoder,
    LearnableFourierTimeEncoder,
    MultiTaskHead,
    TemporalAMLAblationNoTime,
    TemporalNeighborSampler,
    TGATEncoder,
)
from src.pattern_mining import mine_all_patterns


# ==============================================================================
# 1. Config Loader
# ==============================================================================
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ==============================================================================
# 1b. Focal Loss (Class-Imbalance-Aware Objective)
# ==============================================================================
def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary Focal Loss (Lin et al., 2017) computed directly from logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights the loss contribution of easy (high-confidence, correctly
    classified) majority-class examples via the (1 - p_t)^gamma modulating
    factor, and rebalances the minority (illicit) class via alpha, addressing
    the 2.2% illicit / 97.8% licit skew in the Elliptic dataset without
    relying on a fixed pos_weight multiplier.

    Args:
        logits: Raw unnormalized model outputs [B].
        targets: Binary ground-truth labels [B] in {0, 1}.
        alpha: Weight assigned to the positive (minority/illicit) class.
        gamma: Focusing parameter; higher values down-weight easy examples more.

    Returns:
        torch.Tensor: Scalar mean focal loss.
    """
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    modulating_factor = (1.0 - p_t).clamp(min=0.0) ** gamma
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * modulating_factor * ce_loss).mean()


# ==============================================================================
# 2. Evaluation Helper
# ==============================================================================
def evaluate_multi_task(
    model: nn.Module,
    data: Any,
    eval_nodes: torch.Tensor,
    y_lay_all: torch.Tensor,
    y_smurf_all: torch.Tensor,
    device: torch.device,
    head: Optional[nn.Module] = None,
    y_illicit_all: Optional[torch.Tensor] = None,
    batch_size: int = 1024,
    threshold: float = 0.5,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluates 3-pattern multi-task predictions (Illicit + Layering + Smurfing) across target nodes.

    Args:
        model: TGATEncoder or TemporalAMLAblationNoTime module.
        data: PyG Data object.
        eval_nodes: 1D Tensor of node indices to evaluate.
        y_lay_all: Ground truth layering labels [N].
        y_smurf_all: Ground truth smurfing labels [N].
        device: Active compute hardware.
        head: Optional MultiTaskHead (if model is TGATEncoder).
        y_illicit_all: Optional ground truth illicit labels [N] (defaults to (data.y == 1)).
        batch_size: Batch size for forward passes.
        threshold: Decision boundary for F1 score, used as fallback for any
            head not present in `thresholds`.
        thresholds: Optional per-head decision boundaries, e.g.
            {"illicit": 0.42, "lay": 0.31, "smurf": 0.5}, typically calibrated
            beforehand on the validation split via `find_optimal_threshold`.
            Falls back to `threshold` for any head not present.

    Returns:
        Dict[str, Any]: Metrics dictionary for illicit, layering, and smurfing.
    """
    model.eval()
    if head is not None:
        head.eval()

    num_eval = len(eval_nodes)
    if num_eval == 0:
        return {
            "f1_illicit": 0.0, "f1_lay": 0.0, "f1_smurf": 0.0, "mean_f1": 0.0,
            "metrics_illicit": {}, "metrics_lay": {}, "metrics_smurf": {},
            "probs_illicit": np.array([]), "probs_lay": np.array([]), "probs_smurf": np.array([]),
            "y_illicit": np.array([]), "y_lay": np.array([]), "y_smurf": np.array([]),
        }

    p_illicit_list = []
    p_lay_list = []
    p_smurf_list = []

    with torch.no_grad():
        for i in range(0, num_eval, batch_size):
            batch_nodes = eval_nodes[i : i + batch_size]
            batch_times = data.time[batch_nodes]

            if head is not None:
                h = model(
                    data.x.to(device),
                    batch_nodes.to(device),
                    batch_times.to(device),
                )
                p_i, p_l, p_s = head(h)
            else:
                p_i, p_l, p_s = model(
                    data.x.to(device),
                    batch_nodes.to(device),
                    batch_times.to(device),
                )

            p_illicit_list.append(p_i.cpu())
            p_lay_list.append(p_l.cpu())
            p_smurf_list.append(p_s.cpu())

    p_illicit = torch.cat(p_illicit_list).numpy()
    p_lay = torch.cat(p_lay_list).numpy()
    p_smurf = torch.cat(p_smurf_list).numpy()

    if y_illicit_all is None:
        y_illicit_all = (data.y == 1).float()

    y_i = y_illicit_all[eval_nodes].cpu().numpy().astype(int)
    y_l = y_lay_all[eval_nodes].cpu().numpy().astype(int)
    y_s = y_smurf_all[eval_nodes].cpu().numpy().astype(int)

    head_thresholds = thresholds or {}

    def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, head_key: str) -> Dict[str, float]:
        tau = head_thresholds.get(head_key, threshold)
        y_pred = (y_prob >= tau).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.5
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auprc = 0.0
        return {"f1": f1, "precision": prec, "recall": rec, "auc_roc": auc, "auprc": auprc, "threshold": tau}

    m_illicit = calc_metrics(y_i, p_illicit, "illicit")
    m_lay = calc_metrics(y_l, p_lay, "lay")
    m_smurf = calc_metrics(y_s, p_smurf, "smurf")
    mean_f1 = (m_illicit["f1"] + m_lay["f1"] + m_smurf["f1"]) / 3.0

    return {
        "f1_illicit": m_illicit["f1"],
        "f1_lay": m_lay["f1"],
        "f1_smurf": m_smurf["f1"],
        "mean_f1": mean_f1,
        "metrics_illicit": m_illicit,
        "metrics_lay": m_lay,
        "metrics_smurf": m_smurf,
        "probs_illicit": p_illicit,
        "probs_lay": p_lay,
        "probs_smurf": p_smurf,
        "y_illicit": y_i,
        "y_lay": y_l,
        "y_smurf": y_s,
    }


# ==============================================================================
# 3. Main Training Function
# ==============================================================================
def train_model(
    model_type: str = "learnable",  # "learnable" | "fixed" | "ablation"
    config: Optional[Dict[str, Any]] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Trains a specified model variant under inductive temporal split.

    Args:
        model_type: Architecture variant ('learnable', 'fixed', 'ablation').
        config: Configuration dictionary overriding defaults.
        epochs: Number of training epochs (overrides config).

    Returns:
        Dict[str, Any]: Final training summary and test metrics.
    """
    cfg = load_config("config/config.yaml")
    if config:
        cfg.update(config)

    n_epochs = epochs or cfg.get("epochs", 20)
    batch_size = cfg.get("batch_size", 1024)
    lr = float(cfg.get("learning_rate", 0.001))
    hidden_dim = cfg.get("hidden_dim", 128)
    time_dim = cfg.get("time_dim", 64)
    num_layers = cfg.get("num_tgat_layers", 2)
    num_heads = cfg.get("num_heads", 4)
    max_neighbors = cfg.get("max_temporal_neighbors", 20)

    # Task weights for 3-pattern multi-task objective (Illicit + Layering + Smurfing)
    task_weights = cfg.get("task_weights", {})
    lambda_illicit = float(task_weights.get("illicit", cfg.get("lambda_illicit", 1.0)))
    lambda_lay = float(task_weights.get("layering", cfg.get("lambda_lay", 1.0)))
    lambda_smurf = float(task_weights.get("smurfing", cfg.get("lambda_smurf", 1.0)))

    # Loss function: "bce" (pos_weight-scaled BCE, default) or "focal" (Focal Loss)
    loss_type = str(cfg.get("loss_type", "bce")).lower()
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = float(cfg.get("focal_alpha", 0.75))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_names = {
        "learnable": "TemporalAML (Learnable Fourier)",
        "fixed": "Static-TGAT (Fixed Sinusoidal)",
        "ablation": "TemporalAML Ablation (Mean-Pooling, No Time)",
    }
    ckpt_names = {
        "learnable": "temporalaml_best.pth",
        "fixed": "StaticTGAT_best.pth",
        "ablation": "TemporalAMLAblation_best.pth",
    }

    print("=" * 80, flush=True)
    print(f"STARTING {model_names.get(model_type, model_type).upper()} TRAINING ({n_epochs} Epochs)", flush=True)
    print(f"Device: {device} | Batch Size: {batch_size} | LR: {lr}", flush=True)
    print(f"Task Weights: illicit={lambda_illicit}, lay={lambda_lay}, smurf={lambda_smurf}", flush=True)
    if loss_type == "focal":
        print(f"Loss Function: Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})", flush=True)
    else:
        print(f"Loss Function: BCEWithLogits (pos_weight-scaled)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Preprocessed Graph
    loader = EllipticDatasetLoader()
    graph_path = Path("data/processed/elliptic_graph.pt")
    if graph_path.exists():
        data = loader.load(graph_path)
    else:
        data = loader.build_graph()
        loader.save()

    # 2. Load Pattern Labels
    patterns_path = Path("data/processed/pattern_labels.pt")
    if patterns_path.exists():
        print(f"Loading mined pattern targets from {patterns_path}...")
        pat_data = torch.load(patterns_path)
        y_lay = pat_data["y_lay"].float()
        y_smurf = pat_data["y_smurf"].float()
    else:
        print("Pattern targets not found. Running pattern mining pipeline...")
        _, y_l_np, y_s_np, _ = mine_all_patterns(data)
        y_lay = torch.from_numpy(y_l_np).float()
        y_smurf = torch.from_numpy(y_s_np).float()

    # 3. Splits (Inductive Temporal Masks)
    train_end = cfg.get("train_end_step", 34)
    val_end = cfg.get("val_end_step", 40)
    train_mask, val_mask, test_mask = loader.get_split_masks(train_end=train_end, val_end=val_end)

    train_labeled_mask = train_mask & (data.y != -1)
    val_labeled_mask = val_mask & (data.y != -1)
    test_labeled_mask = test_mask & (data.y != -1)

    train_nodes = train_labeled_mask.nonzero(as_tuple=True)[0]
    val_nodes = val_labeled_mask.nonzero(as_tuple=True)[0]
    test_nodes = test_labeled_mask.nonzero(as_tuple=True)[0]

    print(f"Node partitions (labeled only):", flush=True)
    print(f"  • Train : {len(train_nodes):,} nodes (steps <= {train_end})", flush=True)
    print(f"  • Val   : {len(val_nodes):,} nodes (steps {train_end+1}..{val_end})", flush=True)
    print(f"  • Test  : {len(test_nodes):,} nodes (steps > {val_end})", flush=True)

    # 4. Class Imbalance
    def compute_pos_weight(y_target: torch.Tensor, mask: torch.Tensor) -> float:
        y_sub = y_target[mask]
        pos = (y_sub == 1.0).sum().item()
        neg = (y_sub == 0.0).sum().item()
        return float(neg / max(pos, 1))

    y_illicit = (data.y == 1).float()
    pos_w_illicit = compute_pos_weight(y_illicit, train_labeled_mask)
    pos_w_lay = compute_pos_weight(y_lay, train_labeled_mask)
    pos_w_smurf = compute_pos_weight(y_smurf, train_labeled_mask)

    print(f"\nPositive class weights (pos_weight for BCE):", flush=True)
    print(f"  • Illicit  : {pos_w_illicit:.2f}", flush=True)
    print(f"  • Layering : {pos_w_lay:.2f}", flush=True)
    print(f"  • Smurfing : {pos_w_smurf:.2f}", flush=True)

    pw_illicit_tensor = torch.tensor([pos_w_illicit], dtype=torch.float, device=device)
    pw_lay_tensor = torch.tensor([pos_w_lay], dtype=torch.float, device=device)
    pw_smurf_tensor = torch.tensor([pos_w_smurf], dtype=torch.float, device=device)

    # 5. Build Causal Sampler and Models
    edge_times = data.time[data.edge_index[0]].float()
    sampler = TemporalNeighborSampler(
        edge_index=data.edge_index,
        edge_times=edge_times,
        num_neighbors=max_neighbors,
    )

    if model_type == "ablation":
        model = TemporalAMLAblationNoTime(
            in_dim=data.x.shape[1],
            hidden_dim=hidden_dim,
            head_hidden_dim=64,
            max_neighbors=max_neighbors,
            sampler=sampler,
            dropout=0.15,
        ).to(device)
        head = None
        params = list(model.parameters())
    else:
        encoder = TGATEncoder(
            in_dim=data.x.shape[1],
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_neighbors=max_neighbors,
            time_encoder_type=model_type,  # "learnable" or "fixed"
            initial_scale=0.1,
            sampler=sampler,
            dropout=0.15,
        ).to(device)
        head = MultiTaskHead(
            hidden_dim=hidden_dim,
            head_hidden_dim=64,
            dropout=0.15,
        ).to(device)
        model = encoder
        params = list(encoder.parameters()) + list(head.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    # 6. Setup Logging and Checkpoint
    log_dir = Path(f"artifacts/logs/{model_type}_MultiTask")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / ckpt_names[model_type]

    best_mean_f1 = -1.0
    best_epoch = 0

    y_illicit_d = y_illicit.to(device)
    y_lay_d = y_lay.to(device)
    y_smurf_d = y_smurf.to(device)
    x_d = data.x.to(device)

    # 7. Training Loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        if head is not None:
            head.train()

        perm = torch.randperm(len(train_nodes))
        epoch_nodes = train_nodes[perm]

        total_loss_epoch = 0.0
        loss_i_sum = 0.0
        loss_l_sum = 0.0
        loss_s_sum = 0.0
        num_batches = 0

        for i in range(0, len(epoch_nodes), batch_size):
            batch_n = epoch_nodes[i : i + batch_size]
            batch_t = data.time[batch_n].to(device)
            batch_n_d = batch_n.to(device)

            optimizer.zero_grad()

            if head is not None:
                h = model(x_d, batch_n_d, batch_t)
                logit_i, logit_l, logit_s = head.forward_logits(h)
            else:
                logit_i, logit_l, logit_s = model.forward_logits(x_d, batch_n_d, batch_t)

            if loss_type == "focal":
                l_illicit = focal_loss_with_logits(
                    logit_i, y_illicit_d[batch_n_d], alpha=focal_alpha, gamma=focal_gamma
                )
                l_lay = focal_loss_with_logits(
                    logit_l, y_lay_d[batch_n_d], alpha=focal_alpha, gamma=focal_gamma
                )
                l_smurf = focal_loss_with_logits(
                    logit_s, y_smurf_d[batch_n_d], alpha=focal_alpha, gamma=focal_gamma
                )
            else:
                l_illicit = F.binary_cross_entropy_with_logits(
                    logit_i, y_illicit_d[batch_n_d], pos_weight=pw_illicit_tensor
                )
                l_lay = F.binary_cross_entropy_with_logits(
                    logit_l, y_lay_d[batch_n_d], pos_weight=pw_lay_tensor
                )
                l_smurf = F.binary_cross_entropy_with_logits(
                    logit_s, y_smurf_d[batch_n_d], pos_weight=pw_smurf_tensor
                )

            batch_loss = lambda_illicit * l_illicit + lambda_lay * l_lay + lambda_smurf * l_smurf
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss_epoch += batch_loss.item()
            loss_i_sum += l_illicit.item()
            loss_l_sum += l_lay.item()
            loss_s_sum += l_smurf.item()
            num_batches += 1

            if num_batches % 10 == 0 or (i + batch_size) >= len(epoch_nodes):
                print(
                    f"  [Epoch {epoch:02d}/{n_epochs:02d}] Batch {num_batches}/{(len(epoch_nodes) + batch_size - 1)//batch_size} | "
                    f"Batch Loss: {batch_loss.item():.4f} (Illicit:{l_illicit.item():.3f}, Lay:{l_lay.item():.3f}, Smurf:{l_smurf.item():.3f})",
                    flush=True,
                )

        avg_loss = total_loss_epoch / max(num_batches, 1)
        avg_li = loss_i_sum / max(num_batches, 1)
        avg_ll = loss_l_sum / max(num_batches, 1)
        avg_ls = loss_s_sum / max(num_batches, 1)

        # Validation Step
        val_eval = evaluate_multi_task(
            model=model,
            head=head,
            data=data,
            eval_nodes=val_nodes,
            y_lay_all=y_lay,
            y_smurf_all=y_smurf,
            y_illicit_all=y_illicit,
            device=device,
            batch_size=batch_size,
        )

        val_f1_i = val_eval["f1_illicit"]
        val_f1_l = val_eval["f1_lay"]
        val_f1_s = val_eval["f1_smurf"]
        val_mean_f1 = val_eval["mean_f1"]

        # TensorBoard Logging
        writer.add_scalar("Loss/Train_Total", avg_loss, epoch)
        writer.add_scalar("Loss/Train_Illicit", avg_li, epoch)
        writer.add_scalar("Loss/Train_Layering", avg_ll, epoch)
        writer.add_scalar("Loss/Train_Smurfing", avg_ls, epoch)
        writer.add_scalar("F1_Val/Illicit", val_f1_i, epoch)
        writer.add_scalar("F1_Val/Layering", val_f1_l, epoch)
        writer.add_scalar("F1_Val/Smurfing", val_f1_s, epoch)
        writer.add_scalar("F1_Val/Mean", val_mean_f1, epoch)

        print(
            f"Epoch {epoch:02d}/{n_epochs:02d} | "
            f"Train Loss: {avg_loss:.4f} (Illicit:{avg_li:.3f}, Lay:{avg_ll:.3f}, Smurf:{avg_ls:.3f}) | "
            f"Val F1: Mean={val_mean_f1:.4f} [Illicit={val_f1_i:.4f}, Lay={val_f1_l:.4f}, Smurf={val_f1_s:.4f}]",
            flush=True,
        )

        # Checkpoint best model by average validation F1 across both heads
        if val_mean_f1 > best_mean_f1:
            best_mean_f1 = val_mean_f1
            best_epoch = epoch

            if head is not None:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_type": model_type,
                    "best_mean_f1": best_mean_f1,
                    "val_f1_illicit": val_f1_i,
                    "val_f1_lay": val_f1_l,
                    "val_f1_smurf": val_f1_s,
                    "encoder_state_dict": copy.deepcopy(model.state_dict()),
                    "head_state_dict": copy.deepcopy(head.state_dict()),
                    "config": cfg,
                }
            else:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_type": model_type,
                    "best_mean_f1": best_mean_f1,
                    "val_f1_illicit": val_f1_i,
                    "val_f1_lay": val_f1_l,
                    "val_f1_smurf": val_f1_s,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "config": cfg,
                }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  ⭐ Best model updated (Mean Val F1: {best_mean_f1:.4f}) -> {checkpoint_path}", flush=True)

    writer.close()
    print(f"\nTraining completed. Best model at epoch {best_epoch} (Mean Val F1={best_mean_f1:.4f}).", flush=True)

    # 8. Final Test Split Evaluation
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    if head is not None:
        model.load_state_dict(best_checkpoint["encoder_state_dict"])
        head.load_state_dict(best_checkpoint["head_state_dict"])
    else:
        model.load_state_dict(best_checkpoint["model_state_dict"])

    # Calibrate per-head decision thresholds (tau*) strictly on the validation
    # split, maximizing F1, then freeze them before touching test labels.
    val_eval_final = evaluate_multi_task(
        model=model,
        head=head,
        data=data,
        eval_nodes=val_nodes,
        y_lay_all=y_lay,
        y_smurf_all=y_smurf,
        y_illicit_all=y_illicit,
        device=device,
        batch_size=batch_size,
    )
    optimal_thresholds: Dict[str, float] = {}
    for head_key, prob_key, y_key in [
        ("illicit", "probs_illicit", "y_illicit"),
        ("lay", "probs_lay", "y_lay"),
        ("smurf", "probs_smurf", "y_smurf"),
    ]:
        tau_star, val_f1_star = find_optimal_threshold(val_eval_final[y_key], val_eval_final[prob_key])
        optimal_thresholds[head_key] = tau_star
        print(f"  • Calibrated tau* ({head_key:<7}): {tau_star:.2f} (Val F1={val_f1_star:.4f})", flush=True)

    test_eval = evaluate_multi_task(
        model=model,
        head=head,
        data=data,
        eval_nodes=test_nodes,
        y_lay_all=y_lay,
        y_smurf_all=y_smurf,
        y_illicit_all=y_illicit,
        device=device,
        batch_size=batch_size,
        thresholds=optimal_thresholds,
    )

    print("\n" + "=" * 80)
    print(f"FINAL TEST SPLIT (Steps 41–49) {model_names.get(model_type, model_type).upper()} EVALUATION")
    print("(Decision thresholds tau* calibrated on validation split, frozen before test)")
    print("=" * 80)
    header = f"{'Pattern Typology':<20} | {'Tau*':<6} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'AUC-ROC':<10} | {'AUPRC':<10}"
    print(header)
    print("-" * 80)

    for p_name, m_key in [
        ("Illicit Activity", "metrics_illicit"),
        ("Layering Chain", "metrics_lay"),
        ("Smurfing Structuring", "metrics_smurf"),
    ]:
        res = test_eval[m_key]
        print(
            f"{p_name:<20} | "
            f"{res.get('threshold', 0.5):<6.2f} | "
            f"{res.get('f1', 0.0):<10.4f} | "
            f"{res.get('precision', 0.0):<10.4f} | "
            f"{res.get('recall', 0.0):<10.4f} | "
            f"{res.get('auc_roc', 0.0):<10.4f} | "
            f"{res.get('auprc', 0.0):<10.4f}"
        )

    print("-" * 80)
    print(f"Average Multi-Pattern Test F1 : {test_eval['mean_f1']:.4f}")
    print("=" * 80)
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
    print(f"TensorBoard logs at: {log_dir.resolve()}")

    # Persist calibrated thresholds alongside the checkpoint so downstream
    # consumers (evaluate.py, the dashboard) can reuse tau* instead of 0.5.
    best_checkpoint["optimal_thresholds"] = optimal_thresholds
    torch.save(best_checkpoint, checkpoint_path)

    return {
        "model_type": model_type,
        "best_epoch": best_epoch,
        "best_val_mean_f1": best_mean_f1,
        "optimal_thresholds": optimal_thresholds,
        "test_eval": test_eval,
        "checkpoint_path": str(checkpoint_path),
    }


def train_temporalaml(epochs: Optional[int] = None) -> Dict[str, Any]:
    """Shortcut to train proposed TemporalAML model."""
    return train_model(model_type="learnable", epochs=epochs)


def train_all(epochs: Optional[int] = None) -> Dict[str, Any]:
    """Trains TemporalAML, Static-TGAT, and Ablation sequentially."""
    results = {}
    for m in ["learnable", "fixed", "ablation"]:
        results[m] = train_model(model_type=m, epochs=epochs)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TemporalAML models")
    parser.add_argument("--model", type=str, default="all", choices=["learnable", "fixed", "ablation", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.model == "all":
        train_all(epochs=args.epochs)
    else:
        train_model(model_type=args.model, epochs=args.epochs)
