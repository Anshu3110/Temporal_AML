"""Comprehensive evaluation, benchmark comparison, and statistical significance suite.

Evaluates all models on the identical chronological test split (time steps 41–49):
1. RuleBasedHeuristic
2. GCNBaseline (2-layer)
3. GraphSAGEBaseline (2-layer)
4. LSTMGNN (GCN + LSTM)
5. Static-TGAT (Fixed sinusoidal time encoding)
6. TemporalAML Ablation (Mean-pooling, no time encoding)
7. TemporalAML (Learnable Fourier time encoding)

Key Methodology Standards:
- Threshold Selection: Optimal F1 thresholds (tau*) are calibrated strictly on the
  chronological validation split (time steps 35–40) and frozen before test evaluation.
  Test set labels are NEVER used for threshold optimization.
- Multi-Pattern Evaluation: Evaluates Layering and Smurfing typologies.
- Reports both fixed (tau=0.50) and frozen validation-optimal (tau*) metrics, AUC-ROC, and AUPRC.
- Paired bootstrap significance check (1,000 iterations) with 95% CI on F1 delta.
"""

import copy
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (
    GCNBaseline,
    GraphSAGEBaseline,
    LSTMGNN,
    RuleBasedHeuristic,
    evaluate_predictions,
)
from src.data_loader import EllipticDatasetLoader
from src.models import (
    FixedTimeEncoder,
    LearnableFourierTimeEncoder,
    MultiTaskHead,
    TemporalAMLAblationNoTime,
    TemporalNeighborSampler,
    TGATEncoder,
)


# ==============================================================================
# 1. Optimal Validation Threshold Finder
# ==============================================================================
def find_optimal_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
    num_steps: int = 91,
) -> Tuple[float, float]:
    """Finds the decision threshold maximizing F1 on the validation set.

    Args:
        y_val: Validation split binary labels [N_val].
        probs_val: Validation split predicted probabilities [N_val].
        min_thresh: Lower threshold bound.
        max_thresh: Upper threshold bound.
        num_steps: Grid resolution.

    Returns:
        Tuple[float, float]: (best_threshold, best_val_f1).
    """
    best_thresh = 0.50
    best_f1 = -1.0
    for t in np.linspace(min_thresh, max_thresh, num_steps):
        preds = (probs_val >= t).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(t)
    return best_thresh, best_f1


# ==============================================================================
# 2. Paired Bootstrap Significance Test
# ==============================================================================
def paired_bootstrap_f1_significance(
    y_true: np.ndarray,
    probs_temporal: np.ndarray,
    probs_static: np.ndarray,
    threshold_temporal: float = 0.5,
    threshold_static: float = 0.5,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Performs paired bootstrap resampling to evaluate statistical significance of F1 delta.

    Computes:
        delta_F1 = F1(TemporalAML) - F1(Static-TGAT)
    across 1,000 resamples of the test set, extracting empirical 95% confidence intervals
    and one-sided p-value against the null hypothesis (delta <= 0).
    """
    np.random.seed(seed)
    n_samples = len(y_true)

    preds_temporal = (probs_temporal >= threshold_temporal).astype(int)
    preds_static = (probs_static >= threshold_static).astype(int)

    base_f1_temporal = float(f1_score(y_true, preds_temporal, pos_label=1, zero_division=0))
    base_f1_static = float(f1_score(y_true, preds_static, pos_label=1, zero_division=0))
    observed_delta = base_f1_temporal - base_f1_static

    deltas: List[float] = []

    for _ in range(n_iterations):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_b = y_true[boot_idx]

        if (y_b == 1).sum() == 0:
            continue

        f1_t = f1_score(y_b, preds_temporal[boot_idx], pos_label=1, zero_division=0)
        f1_s = f1_score(y_b, preds_static[boot_idx], pos_label=1, zero_division=0)
        deltas.append(float(f1_t - f1_s))

    deltas_arr = np.array(deltas)

    ci_lower = float(np.percentile(deltas_arr, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(deltas_arr, 100 * (1.0 - alpha / 2.0)))
    mean_delta = float(np.mean(deltas_arr))
    std_delta = float(np.std(deltas_arr))
    p_value = float(np.mean(deltas_arr <= 0.0))

    return {
        "base_f1_temporal": base_f1_temporal,
        "base_f1_static": base_f1_static,
        "observed_delta": observed_delta,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95": (ci_lower, ci_upper),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": bool(ci_lower > 0.0 and p_value < alpha),
        "n_iterations": n_iterations,
    }


# ==============================================================================
# 3. Model Prediction Loaders
# ==============================================================================
def get_split_predictions(
    data: Any,
    eval_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Generates probability predictions for all models over a target mask."""
    predictions: Dict[str, np.ndarray] = {}

    target_nodes = eval_mask.nonzero(as_tuple=True)[0]
    target_times = data.time[target_nodes]
    x_sub = data.x[eval_mask]
    edge_index_d = data.edge_index.to(device)
    x_d = data.x.to(device)

    # 1. RuleBasedHeuristic
    heuristic = RuleBasedHeuristic(volume_feature_idx=2, velocity_feature_idx=0)
    predictions["RuleBasedHeuristic"] = heuristic.predict_proba(x_sub)

    # 2. GCNBaseline
    gcn_path = Path("artifacts/models/GCNBaseline_best.pt")
    gcn = GCNBaseline(in_channels=data.x.shape[1], hidden_dim=128).to(device)
    if gcn_path.exists():
        gcn.load_state_dict(torch.load(gcn_path, map_location=device))
    gcn.eval()
    with torch.no_grad():
        logits_gcn = gcn(x_d, edge_index_d)
        predictions["GCN (2-layer)"] = torch.sigmoid(logits_gcn[eval_mask.to(device)]).cpu().numpy()

    # 3. GraphSAGEBaseline
    sage_path = Path("artifacts/models/GraphSAGEBaseline_best.pt")
    sage = GraphSAGEBaseline(in_channels=data.x.shape[1], hidden_dim=128).to(device)
    if sage_path.exists():
        sage.load_state_dict(torch.load(sage_path, map_location=device))
    sage.eval()
    with torch.no_grad():
        logits_sage = sage(x_d, edge_index_d)
        predictions["GraphSAGE (2-layer)"] = torch.sigmoid(logits_sage[eval_mask.to(device)]).cpu().numpy()

    # 4. LSTMGNN
    lstm_path = Path("artifacts/models/LSTMGNN_best.pt")
    lstm_model = LSTMGNN(in_channels=data.x.shape[1], hidden_dim=128, lstm_hidden_dim=64).to(device)
    if lstm_path.exists():
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()
    with torch.no_grad():
        logits_lstm = lstm_model(x_d, edge_index_d)
        predictions["LSTM-GNN (GCN+LSTM)"] = torch.sigmoid(logits_lstm[eval_mask.to(device)]).cpu().numpy()

    # Shared Causal Sampler for TGAT models
    edge_times = data.time[data.edge_index[0]].float()
    sampler = TemporalNeighborSampler(data.edge_index, edge_times, num_neighbors=20)

    # 5. Static-TGAT (Fixed Sinusoidal Time Encoding)
    static_encoder = TGATEncoder(
        in_dim=data.x.shape[1], hidden_dim=128, time_dim=64, num_layers=2, num_heads=4,
        max_neighbors=20, time_encoder_type="fixed", sampler=sampler
    ).to(device)
    static_head = MultiTaskHead(128, 64).to(device)

    static_path = Path("artifacts/models/StaticTGAT_best.pth")
    if static_path.exists():
        chk_s = torch.load(static_path, map_location=device)
        static_encoder.load_state_dict(chk_s["encoder_state_dict"])
        static_head.load_state_dict(chk_s["head_state_dict"])
    static_encoder.eval()
    static_head.eval()

    probs_static = []
    with torch.no_grad():
        for i in range(0, len(target_nodes), 1024):
            batch_n = target_nodes[i : i + 1024]
            batch_t = target_times[i : i + 1024]
            h = static_encoder(x_d, batch_n.to(device), batch_t.to(device))
            pl, ps = static_head(h)
            p_comp = torch.maximum(pl, ps)
            probs_static.append(p_comp.cpu())
    predictions["Static-TGAT (Fixed Fourier)"] = torch.cat(probs_static).numpy()

    # 6. TemporalAML Ablation (Mean-Pooling, No Time)
    ablation_model = TemporalAMLAblationNoTime(
        in_dim=data.x.shape[1], hidden_dim=128, head_hidden_dim=64, max_neighbors=20, sampler=sampler
    ).to(device)
    ablation_path = Path("artifacts/models/TemporalAMLAblation_best.pth")
    if ablation_path.exists():
        chk_a = torch.load(ablation_path, map_location=device)
        if "model_state_dict" in chk_a:
            ablation_model.load_state_dict(chk_a["model_state_dict"])
    ablation_model.eval()

    probs_ablation = []
    with torch.no_grad():
        for i in range(0, len(target_nodes), 1024):
            batch_n = target_nodes[i : i + 1024]
            batch_t = target_times[i : i + 1024]
            pl, ps = ablation_model(x_d, batch_n.to(device), batch_t.to(device))
            p_comp = torch.maximum(pl, ps)
            probs_ablation.append(p_comp.cpu())
    predictions["TemporalAML (Ablation: No Time)"] = torch.cat(probs_ablation).numpy()

    # 7. TemporalAML (Learnable Fourier Time Encoding)
    temporal_encoder = TGATEncoder(
        in_dim=data.x.shape[1], hidden_dim=128, time_dim=64, num_layers=2, num_heads=4,
        max_neighbors=20, time_encoder_type="learnable", sampler=sampler
    ).to(device)
    temporal_head = MultiTaskHead(128, 64).to(device)

    temporal_path = Path("artifacts/models/temporalaml_best.pth")
    if temporal_path.exists():
        chk_t = torch.load(temporal_path, map_location=device)
        temporal_encoder.load_state_dict(chk_t["encoder_state_dict"])
        temporal_head.load_state_dict(chk_t["head_state_dict"])
    temporal_encoder.eval()
    temporal_head.eval()

    p_lay_list, p_smurf_list, p_comp_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(target_nodes), 1024):
            batch_n = target_nodes[i : i + 1024]
            batch_t = target_times[i : i + 1024]
            h = temporal_encoder(x_d, batch_n.to(device), batch_t.to(device))
            pl, ps = temporal_head(h)
            p_lay_list.append(pl.cpu())
            p_smurf_list.append(ps.cpu())
            p_comp_list.append(torch.maximum(pl, ps).cpu())

    predictions["TemporalAML (Learnable Fourier)"] = torch.cat(p_comp_list).numpy()
    predictions["_TemporalAML_Lay"] = torch.cat(p_lay_list).numpy()
    predictions["_TemporalAML_Smurf"] = torch.cat(p_smurf_list).numpy()

    return predictions


# ==============================================================================
# 4. Main Evaluation Runner
# ==============================================================================
def run_evaluation() -> Dict[str, Any]:
    """Runs full benchmark evaluation across all models and saves artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print(f"TEMPORALAML BENCHMARK EVALUATION (Device: {device})", flush=True)
    print("=" * 85, flush=True)

    # 1. Load Graph Data
    loader = EllipticDatasetLoader()
    graph_path = Path("data/processed/elliptic_graph.pt")
    data = loader.load(graph_path)

    # Chronological Splits
    train_mask, val_mask, test_mask = loader.get_split_masks(train_end=34, val_end=40)
    val_eval_mask = val_mask & (data.y != -1)
    test_eval_mask = test_mask & (data.y != -1)

    y_val = data.y[val_eval_mask].cpu().numpy().astype(int)
    y_test = data.y[test_eval_mask].cpu().numpy().astype(int)

    n_test = len(y_test)
    n_illicit = int((y_test == 1).sum())
    n_licit = int((y_test == 0).sum())
    print(f"Test split: {n_test:,} labeled transactions ({n_illicit:,} Illicit, {n_licit:,} Licit)")
    print(f"Val split : {len(y_val):,} labeled transactions (Used exclusively for threshold freezing)")

    # 2. Extract Predictions on Validation and Test Sets
    print("\n[1/4] Generating Validation Predictions (for threshold calibration)...")
    val_preds = get_split_predictions(data, val_eval_mask, device)

    print("\n[2/4] Generating Test Predictions...")
    test_preds = get_split_predictions(data, test_eval_mask, device)

    # 3. Calibrate Frozen Thresholds on Validation Split
    model_keys = [
        "RuleBasedHeuristic",
        "GCN (2-layer)",
        "GraphSAGE (2-layer)",
        "LSTM-GNN (GCN+LSTM)",
        "Static-TGAT (Fixed Fourier)",
        "TemporalAML (Ablation: No Time)",
        "TemporalAML (Learnable Fourier)",
    ]

    val_thresholds: Dict[str, float] = {}
    print("\n[3/4] Calibrating Frozen Thresholds on Validation Set:")
    for m_name in model_keys:
        p_val = val_preds[m_name]
        tau_val, f1_val = find_optimal_threshold(y_val, p_val)
        val_thresholds[m_name] = tau_val
        print(f"  • {m_name:<35}: tau* = {tau_val:.2f} (Val F1 = {f1_val:.4f})")

    # 4. Compute Benchmark Metrics
    results_rows: List[Dict[str, Any]] = []

    for m_name in model_keys:
        p_test = test_preds[m_name]
        frozen_tau = val_thresholds[m_name]

        # Test evaluation at frozen threshold
        metrics_frozen = evaluate_predictions(y_test, p_test, threshold=frozen_tau)
        # Test evaluation at standard fixed 0.50 threshold
        metrics_fixed = evaluate_predictions(y_test, p_test, threshold=0.50)

        results_rows.append({
            "Model": m_name,
            "F1 (tau=0.50)": metrics_fixed["f1"],
            "F1 (tau* val)": metrics_frozen["f1"],
            "Val tau*": frozen_tau,
            "Precision": metrics_frozen["precision"],
            "Recall": metrics_frozen["recall"],
            "AUC-ROC": metrics_frozen["auc_roc"],
            "AUPRC": metrics_frozen["auprc"],
        })

    df_results = pd.DataFrame(results_rows)

    # 5. Per-Pattern Evaluation for TemporalAML
    pat_path = Path("data/processed/pattern_labels.pt")
    if pat_path.exists():
        pat_data = torch.load(pat_path)
        yl_test = pat_data["y_lay"][test_eval_mask].cpu().numpy().astype(int)
        ys_test = pat_data["y_smurf"][test_eval_mask].cpu().numpy().astype(int)
    else:
        yl_test = np.zeros_like(y_test)
        ys_test = np.zeros_like(y_test)

    m_lay = evaluate_predictions(yl_test, test_preds["_TemporalAML_Lay"], threshold=0.5)
    m_smurf = evaluate_predictions(ys_test, test_preds["_TemporalAML_Smurf"], threshold=0.5)

    df_patterns = pd.DataFrame([
        {"Pattern Typology": "Layering Chain", "F1-Score": m_lay["f1"], "Precision": m_lay["precision"], "Recall": m_lay["recall"], "AUC-ROC": m_lay["auc_roc"], "AUPRC": m_lay["auprc"]},
        {"Pattern Typology": "Smurfing Structuring", "F1-Score": m_smurf["f1"], "Precision": m_smurf["precision"], "Recall": m_smurf["recall"], "AUC-ROC": m_smurf["auc_roc"], "AUPRC": m_smurf["auprc"]},
    ])

    # 6. Paired Bootstrap Significance Test: TemporalAML vs Static-TGAT
    print("\n[4/4] Running Paired Bootstrap Significance Test (1,000 iterations)...")
    boot_res = paired_bootstrap_f1_significance(
        y_true=y_test,
        probs_temporal=test_preds["TemporalAML (Learnable Fourier)"],
        probs_static=test_preds["Static-TGAT (Fixed Fourier)"],
        threshold_temporal=val_thresholds["TemporalAML (Learnable Fourier)"],
        threshold_static=val_thresholds["Static-TGAT (Fixed Fourier)"],
        n_iterations=1000,
        seed=42,
    )

    # 7. Save Markdown & CSV Reports
    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "final_results.csv"
    df_results.to_csv(csv_path, index=False)

    md_path = logs_dir / "final_results.md"
    with open(md_path, "w") as f:
        f.write("# 🔬 TemporalAML Comprehensive Benchmark Results\n\n")
        f.write(f"**Evaluation Dataset:** Elliptic Bitcoin Dataset — Test Split (Time Steps 41–49)\n")
        f.write(f"**Test Set Size:** {n_test:,} labeled transactions ({n_illicit:,} Illicit, {n_licit:,} Licit)\n\n")
        f.write("> **Methodology Note on Threshold Selection:**  \n")
        f.write("> Decision threshold $\\tau^*$ is calibrated exclusively to maximize F1 on the chronological ")
        f.write("validation split (time steps 35–40) and frozen before test evaluation. ")
        f.write("Ground-truth test labels are never observed during threshold calibration.\n\n")
        f.write("## 1. Overall Illicit Class Detection Performance\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 2. TemporalAML Per-Pattern Typology Performance\n\n")
        f.write(df_patterns.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 3. Statistical Significance Audit: TemporalAML vs Static-TGAT\n\n")
        f.write(f"- **Method:** Paired Bootstrap Resampling ({boot_res['n_iterations']:,} iterations)\n")
        f.write(f"- **Observed F1 (TemporalAML):** `{boot_res['base_f1_temporal']:.4f}`\n")
        f.write(f"- **Observed F1 (Static-TGAT):** `{boot_res['base_f1_static']:.4f}`\n")
        f.write(f"- **Mean F1 Delta (ΔF1):** `{boot_res['mean_delta']:+.4f}`\n")
        f.write(f"- **95% Confidence Interval on ΔF1:** `[{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}]`\n")
        f.write(f"- **Empirical p-value (H₀: ΔF1 ≤ 0):** `{boot_res['p_value']:.4f}`\n")
        sig_str = "Statistically Significant at α=0.05 (CI strictly positive)" if boot_res["is_significant"] else "Difference not strictly bounded above zero at α=0.05"
        f.write(f"- **Significance Verdict:** **{sig_str}**\n")

    print(f"✅ Saved results CSV to : {csv_path.resolve()}")
    print(f"✅ Saved results Markdown to: {md_path.resolve()}")

    # 8. Generate AUPRC Comparison Bar Chart
    chart_path = logs_dir / "auprc_comparison.png"
    plt.figure(figsize=(12, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    models_plot = df_results["Model"].tolist()
    auprc_vals = df_results["AUPRC"].tolist()

    colors = [
        "#9E9E9E",  # Heuristic (gray)
        "#64B5F6",  # GCN (blue)
        "#42A5F5",  # GraphSAGE (blue)
        "#AB47BC",  # LSTM-GNN (purple)
        "#FFB74D",  # Static-TGAT (orange)
        "#FFA726",  # Ablation (orange-dark)
        "#E53935",  # TemporalAML (red-primary highlight)
    ]

    bars = plt.barh(models_plot, auprc_vals, color=colors, edgecolor="#333", height=0.65, alpha=0.9)
    plt.xlabel("Area Under Precision-Recall Curve (AUPRC)", fontsize=11, fontweight="bold")
    plt.title("AML Illicit Transaction Detection Benchmark: AUPRC Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlim(0, max(auprc_vals) * 1.25)

    for bar, val in zip(bars, auprc_vals):
        plt.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#212121",
        )

    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved AUPRC chart to: {chart_path.resolve()}")

    # 9. Print Terminal Summary
    print("\n" + "=" * 85)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 85)
    print(df_results.to_markdown(index=False))

    print("\n" + "=" * 85)
    print("TEMPORALAML PER-PATTERN PERFORMANCE")
    print("=" * 85)
    print(df_patterns.to_markdown(index=False))

    print("\n" + "=" * 85)
    print("PAIRED BOOTSTRAP SIGNIFICANCE AUDIT")
    print("=" * 85)
    print(f"Observed F1 Delta (ΔF1) : {boot_res['observed_delta']:+.4f}")
    print(f"95% Confidence Interval : [{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}]")
    print(f"Empirical p-value       : {boot_res['p_value']:.4f}")
    print(f"Significance Verdict    : {sig_str}")
    print("=" * 85)

    return {
        "df_results": df_results,
        "df_patterns": df_patterns,
        "bootstrap": boot_res,
    }


if __name__ == "__main__":
    run_evaluation()
