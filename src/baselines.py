"""Baseline AML models for the Elliptic Bitcoin dataset.

Includes:
1. 2-layer GCNBaseline (PyG GCNConv)
2. 2-layer GraphSAGEBaseline (PyG SAGEConv)
3. LSTMGNN (Temporal recurrent GNN)
4. RuleBasedHeuristic (Config-driven volume & velocity heuristic classifier)
5. train_baseline training harness with early stopping, class-weighted loss,
   and TensorBoard logging.
"""

import copy
import os
from pathlib import Path
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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
import sys

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_loader import EllipticDatasetLoader
except ModuleNotFoundError:
    from data_loader import EllipticDatasetLoader


# ==============================================================================
# 1. GCN Baseline Model
# ==============================================================================
class GCNBaseline(nn.Module):
    """Two-layer Graph Convolutional Network (GCN) binary classifier.

    Uses `torch_geometric.nn.GCNConv` message passing across transaction nodes.
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature tensor [N, in_channels]
            edge_index: Graph connectivity [2, E]

        Returns:
            torch.Tensor: Unnormalized scalar logits for each node [N]
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.conv2(h, edge_index).squeeze(-1)
        return logits


# ==============================================================================
# 2. GraphSAGE Baseline Model
# ==============================================================================
class GraphSAGEBaseline(nn.Module):
    """Two-layer GraphSAGE binary classifier.

    Uses `torch_geometric.nn.SAGEConv` with inductive neighborhood aggregation.
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature tensor [N, in_channels]
            edge_index: Graph connectivity [2, E]

        Returns:
            torch.Tensor: Unnormalized scalar logits for each node [N]
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.conv2(h, edge_index).squeeze(-1)
        return logits


# ==============================================================================
# 3. LSTMGNN Baseline Model
# ==============================================================================
class LSTMGNN(nn.Module):
    """Temporal Graph Neural Network combining spatial GCN with recurrent LSTM.

    Applies a shared GCNConv to compute structural node representations,
    stacks embeddings for each node across its temporal sequence, and passes
    them through an nn.LSTM prior to a final linear classification layer.
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature tensor [N, in_channels]
            edge_index: Graph connectivity [2, E]
            time: Optional node time steps [N]

        Returns:
            torch.Tensor: Unnormalized scalar logits for each node [N]
        """
        # 1. Structural message passing via shared GCNConv
        h = F.relu(self.gcn(x, edge_index))
        h = self.dropout(h)

        # 2. Sequence representation: Stack per-node embeddings along temporal dimension [N, seq_len=1, hidden_dim]
        h_seq = h.unsqueeze(1)

        # 3. Temporal recurrent encoding via LSTM
        lstm_out, _ = self.lstm(h_seq)

        # 4. Final classification head on the last recurrent hidden state
        logits = self.classifier(lstm_out[:, -1, :]).squeeze(-1)
        return logits


# ==============================================================================
# 4. Rule-Based Heuristic Classifier (No Training)
# ==============================================================================
class RuleBasedHeuristic:
    """Config-driven AML rule heuristic baseline without parameter training.

    Flags a transaction as illicit (1) if:
    - Any volume proxy feature exceeds `volume_threshold`, OR
    - Any velocity proxy feature is below `velocity_threshold`.
    """

    def __init__(
        self,
        volume_feature_idx: Union[int, List[int]] = 2,
        velocity_feature_idx: Union[int, List[int]] = 0,
        volume_threshold: float = 0.45,
        velocity_threshold: float = -0.15,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            volume_feature_idx = config.get("volume_feature_idx", volume_feature_idx)
            velocity_feature_idx = config.get("velocity_feature_idx", velocity_feature_idx)
            volume_threshold = config.get("volume_threshold", volume_threshold)
            velocity_threshold = config.get("velocity_threshold", velocity_threshold)

        self.volume_indices: List[int] = (
            [volume_feature_idx] if isinstance(volume_feature_idx, int) else volume_feature_idx
        )
        self.velocity_indices: List[int] = (
            [velocity_feature_idx] if isinstance(velocity_feature_idx, int) else velocity_feature_idx
        )
        self.volume_threshold = float(volume_threshold)
        self.velocity_threshold = float(velocity_threshold)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Calculates pseudo-probability risk scores based on heuristic criteria.

        Args:
            x: Feature matrix [N, 165]

        Returns:
            np.ndarray: Continuous probability scores in [0, 1] [N]
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        # Compute volume anomaly excess
        vol_excess = np.max(x_np[:, self.volume_indices] - self.volume_threshold, axis=1)
        # Compute velocity deficit anomaly
        vel_deficit = np.max(self.velocity_threshold - x_np[:, self.velocity_indices], axis=1)

        # Combine into continuous composite risk score via logistic sigmoid
        raw_score = np.maximum(vol_excess, 0.0) + np.maximum(vel_deficit, 0.0)
        prob = 1.0 / (1.0 + np.exp(-3.0 * (raw_score - 0.2)))
        return prob

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction of illicit status.

        Args:
            x: Feature matrix [N, 165]
            threshold: Probability decision boundary

        Returns:
            np.ndarray: Binary array (0 or 1) [N]
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        vol_flag = np.any(x_np[:, self.volume_indices] > self.volume_threshold, axis=1)
        vel_flag = np.any(x_np[:, self.velocity_indices] < self.velocity_threshold, axis=1)
        flags = (vol_flag | vel_flag).astype(int)
        return flags


# ==============================================================================
# 5. Evaluation Utility
# ==============================================================================
def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes comprehensive AML classification metrics.

    Args:
        y_true: Ground truth binary labels (0=licit, 1=illicit)
        y_prob: Predicted probabilities of illicit class in [0, 1]
        threshold: Decision threshold for discrete classification

    Returns:
        Dict[str, float]: F1, Precision, Recall, AUC-ROC, and AUPRC metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "auprc": auprc,
    }


# ==============================================================================
# 6. Training Harness with Early Stopping & TensorBoard
# ==============================================================================
def train_baseline(
    model: nn.Module,
    data: Data,
    masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains a baseline GNN with early stopping on validation illicit F1.

    Args:
        model: PyTorch baseline model to train.
        data: PyTorch Geometric Data object with features and edges.
        masks: Tuple of (train_mask, val_mask, test_mask).
        config: Optional configuration dictionary.
        model_name: Name used for TensorBoard logging and model checkpointing.

    Returns:
        Dict[str, Any]: Training summary, best validation metrics, and history.
    """
    cfg = config or {}
    epochs = cfg.get("epochs", 35)
    lr = cfg.get("learning_rate", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-5)
    patience = cfg.get("patience", 10)
    decision_threshold = cfg.get("decision_threshold", 0.5)

    name = model_name or model.__class__.__name__

    # Resolve compute hardware
    device_cfg = cfg.get("device", "cuda if available else cpu")
    if "cuda" in device_cfg and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n[{name}] Training on device: {device} | Max Epochs: {epochs} | Patience: {patience}")

    train_mask, val_mask, test_mask = masks

    # Train only on nodes with ground truth labels (y != -1)
    train_eval_mask = train_mask & (data.y != -1)
    val_eval_mask = val_mask & (data.y != -1)

    # Compute class weighting pos_weight = neg / pos to tackle severe AML imbalance
    y_train = data.y[train_eval_mask]
    num_neg = (y_train == 0).sum().item()
    num_pos = (y_train == 1).sum().item()
    pos_weight_val = num_neg / max(num_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Move tensors to active compute device
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    time_tensor = data.time.to(device) if hasattr(data, "time") and data.time is not None else None
    y_target = data.y.float().to(device)

    train_idx = train_eval_mask.to(device)
    val_idx = val_eval_mask.to(device)
    y_val_np = data.y[val_eval_mask].cpu().numpy()

    # Setup TensorBoard writer
    log_dir = Path("artifacts/logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        optimizer.zero_grad()

        logits = model(x=x, edge_index=edge_index, time=time_tensor)
        loss = criterion(logits[train_idx], y_target[train_idx])

        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(x=x, edge_index=edge_index, time=time_tensor)
            val_loss = criterion(val_logits[val_idx], y_target[val_idx]).item()

            val_probs = torch.sigmoid(val_logits[val_idx]).cpu().numpy()
            val_metrics = evaluate_predictions(y_val_np, val_probs, threshold=decision_threshold)
            val_f1 = val_metrics["f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Val_F1", val_f1, epoch)
        writer.add_scalar("Metrics/Val_Precision", val_metrics["precision"], epoch)
        writer.add_scalar("Metrics/Val_Recall", val_metrics["recall"], epoch)
        writer.add_scalar("Metrics/Val_AUC_ROC", val_metrics["auc_roc"], epoch)
        writer.add_scalar("Metrics/Val_AUPRC", val_metrics["auprc"], epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Rec: {val_metrics['recall']:.4f} | "
                f"Val Prec: {val_metrics['precision']:.4f}"
            )

        # Early stopping on Validation F1 for illicit class
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  --> Early stopping triggered at epoch {epoch} (Best Val F1={best_val_f1:.4f} at epoch {best_epoch})")
                break

    writer.close()

    # Restore best checkpoint
    model.load_state_dict(best_weights)

    # Save model artifact
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{name}_best.pt"
    torch.save(best_weights, save_path)
    print(f"  Saved best model checkpoint to: {save_path}")

    return {
        "model_name": name,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "checkpoint_path": str(save_path),
    }


# ==============================================================================
# 7. Helper: Load Config
# ==============================================================================
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ==============================================================================
# 8. Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # 1. Load Dataset
    print("=" * 80)
    print("TEMPORAL AML BASELINE BENCHMARK")
    print("=" * 80)

    cfg = load_config("config/config.yaml")
    processed_graph_path = Path("data/processed/elliptic_graph.pt")

    loader = EllipticDatasetLoader()
    if processed_graph_path.exists():
        print(f"Loading preprocessed graph from {processed_graph_path}...")
        data = loader.load(processed_graph_path)
    else:
        print("Preprocessed graph not found. Building from raw files...")
        data = loader.build_graph()
        loader.save()

    # 2. Extract Temporal Split Masks
    train_end = cfg.get("train_end_step", 34)
    val_end = cfg.get("val_end_step", 40)
    train_mask, val_mask, test_mask = loader.get_split_masks(train_end=train_end, val_end=val_end)
    masks = (train_mask, val_mask, test_mask)

    test_eval_mask = test_mask & (data.y != -1)
    y_test_np = data.y[test_eval_mask].cpu().numpy()
    x_test = data.x[test_eval_mask]

    print(f"Dataset summary: N={data.num_nodes:,}, E={data.num_edges:,}")
    print(f"Evaluation test set: {len(y_test_np):,} labeled nodes (Illicit={(y_test_np == 1).sum():,}, Licit={(y_test_np == 0).sum():,})")

    # Hardware target
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    hidden_dim = cfg.get("hidden_dim", 128)
    lr = cfg.get("learning_rate", 1e-3)
    train_cfg = {
        "epochs": 25,
        "learning_rate": lr,
        "weight_decay": 1e-5,
        "patience": 8,
        "device": str(device),
        "decision_threshold": 0.5,
    }

    # 3. Model 1: Rule-Based Heuristic
    print("\n" + "-" * 80)
    print("1/4: Evaluating Rule-Based Heuristic (No Training)...")
    heuristic = RuleBasedHeuristic(
        volume_feature_idx=2,
        velocity_feature_idx=0,
        volume_threshold=0.45,
        velocity_threshold=-0.15,
    )
    heuristic_prob = heuristic.predict_proba(x_test)
    results_heuristic = evaluate_predictions(y_test_np, heuristic_prob, threshold=0.5)

    # 4. Model 2: 2-layer GCN
    print("\n" + "-" * 80)
    print("2/4: Training 2-layer GCN Baseline...")
    gcn_model = GCNBaseline(in_channels=165, hidden_dim=hidden_dim, dropout=0.2)
    train_baseline(gcn_model, data, masks, config=train_cfg, model_name="GCNBaseline")

    gcn_model.eval().to(device)
    with torch.no_grad():
        gcn_logits = gcn_model(data.x.to(device), data.edge_index.to(device))
        gcn_probs = torch.sigmoid(gcn_logits[test_eval_mask.to(device)]).cpu().numpy()
    results_gcn = evaluate_predictions(y_test_np, gcn_probs, threshold=0.5)

    # 5. Model 3: 2-layer GraphSAGE
    print("\n" + "-" * 80)
    print("3/4: Training 2-layer GraphSAGE Baseline...")
    sage_model = GraphSAGEBaseline(in_channels=165, hidden_dim=hidden_dim, dropout=0.2)
    train_baseline(sage_model, data, masks, config=train_cfg, model_name="GraphSAGEBaseline")

    sage_model.eval().to(device)
    with torch.no_grad():
        sage_logits = sage_model(data.x.to(device), data.edge_index.to(device))
        sage_probs = torch.sigmoid(sage_logits[test_eval_mask.to(device)]).cpu().numpy()
    results_sage = evaluate_predictions(y_test_np, sage_probs, threshold=0.5)

    # 6. Model 4: LSTMGNN
    print("\n" + "-" * 80)
    print("4/4: Training LSTMGNN Baseline...")
    lstm_gnn_model = LSTMGNN(
        in_channels=165,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=64,
        dropout=0.2,
    )
    train_baseline(lstm_gnn_model, data, masks, config=train_cfg, model_name="LSTMGNN")

    lstm_gnn_model.eval().to(device)
    with torch.no_grad():
        lstm_logits = lstm_gnn_model(
            data.x.to(device),
            data.edge_index.to(device),
            time=data.time.to(device) if hasattr(data, "time") else None,
        )
        lstm_probs = torch.sigmoid(lstm_logits[test_eval_mask.to(device)]).cpu().numpy()
    results_lstm = evaluate_predictions(y_test_np, lstm_probs, threshold=0.5)

    # 7. Comparison Table
    all_results = {
        "RuleBasedHeuristic": results_heuristic,
        "GCNBaseline (2-layer)": results_gcn,
        "GraphSAGE (2-layer)": results_sage,
        "LSTMGNN (GCN+LSTM)": results_lstm,
    }

    print("\n" + "=" * 85)
    print("TEST SPLIT (Steps 41–49) AML CLASSIFICATION BENCHMARK COMPARISON")
    print("=" * 85)
    header = f"{'Model':<24} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'AUC-ROC':<10} | {'AUPRC':<10}"
    print(header)
    print("-" * 85)

    for m_name, res in all_results.items():
        row = (
            f"{m_name:<24} | "
            f"{res['f1']:<10.4f} | "
            f"{res['precision']:<10.4f} | "
            f"{res['recall']:<10.4f} | "
            f"{res['auc_roc']:<10.4f} | "
            f"{res['auprc']:<10.4f}"
        )
        print(row)

    print("=" * 85)
    print("TensorBoard event logs saved to: artifacts/logs/")
    print("Best model checkpoints saved to : artifacts/models/")
