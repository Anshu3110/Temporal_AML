# 🔬 TemporalAML — Phase 1

> **Temporal Graph Attention Networks for Explainable Multi-Pattern AML in Cryptocurrency Transactions**
> M.Tech Data Science — Dissertation Phase 1

---

## 📌 What This Project Does

Detects **money laundering patterns** in Bitcoin transactions using **Temporal Graph Neural Networks**.

- **Dataset**: Elliptic Bitcoin Dataset (203,769 transactions, 234,355 edges)
- **O1**: Build a temporal transaction graph with engineered features
- **O2**: Implement Learnable Fourier Time Encoding inside TGAT

---

## 📁 Project Structure

```
DissertationProject/
│
├── 📓 notebook/
│   └── TemporalAML_Phase1.ipynb     ← Main Google Colab notebook (29 cells)
│
├── ⚙️  backend/
│   ├── main.py                       ← FastAPI server (all result endpoints)
│   └── requirements.txt              ← Python dependencies
│
├── 🎨 frontend/
│   ├── index.html                    ← Dashboard (open directly in browser)
│   └── src/
│       ├── style.css                 ← Premium dark-mode styles
│       └── app.js                    ← Chart.js + API integration
│
├── 📊 data/
│   └── raw/
│       ├── elliptic_txs_features.csv ← 203,769 rows, 166 features (no header)
│       ├── elliptic_txs_classes.csv  ← txId → class (1=illicit, 2=licit, unknown)
│       └── elliptic_txs_edgelist.csv ← 234,355 directed transaction edges
│
└── 📚 guidebook/                     ← Research & implementation guides
    ├── README.md
    ├── chapter_1_introduction.md
    ├── chapter_2_mathematical_foundations.md
    └── ... (12 chapters total)
```

---

## 🚀 Quick Start

### 1️⃣ Open the Dashboard (No setup needed — works right now)

```bash
open /Users/apple/Desktop/DissertationProject/frontend/index.html
```

Opens a **premium dark-mode dashboard** with 7 tabs showing all Phase 1 results.

---

### 2️⃣ Start the Backend API

```bash
cd /Users/apple/Desktop/DissertationProject/backend
pip install fastapi uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Health check |
| http://localhost:8000/docs | Swagger UI (interactive API) |
| http://localhost:8000/api/status | O1 + O2 test results |
| http://localhost:8000/api/eda | EDA statistics |
| http://localhost:8000/api/graph | Graph construction summary |
| http://localhost:8000/api/encoding | Fourier encoding architecture |
| http://localhost:8000/api/training | Loss / F1 / AUC per epoch |
| http://localhost:8000/api/ablation | Learnable vs Fixed comparison |

---

### 3️⃣ Run the Colab Notebook

> **This is the main research work. Run on Google Colab (free T4 GPU).**

**Step 1** — Go to https://colab.research.google.com

**Step 2** — Upload the notebook:
```
notebook/TemporalAML_Phase1.ipynb
```

**Step 3** — Enable GPU:
```
Runtime → Change runtime type → T4 GPU → Save
```

**Step 4** — Upload datasets to Google Drive:
```
MyDrive/TemporalAML/data/elliptic_txs_features.csv
MyDrive/TemporalAML/data/elliptic_txs_classes.csv
MyDrive/TemporalAML/data/elliptic_txs_edgelist.csv
```

**Step 5** — Run all cells:
```
Runtime → Run all
```

**Step 6** — Allow Google Drive access when prompted.

⏱️ **Total runtime**: ~20–25 minutes on free T4 GPU

---

## 🔬 Objective 1 — Temporal Graph Construction

| Property | Value |
|----------|-------|
| Total nodes | 203,769 transactions |
| Total edges | 234,355 directed links |
| Node features | 172 (166 raw + 6 engineered) |
| Labels | Illicit=1, Licit=0, Unknown=−1 |
| Time steps | 49 (~2-week intervals) |

**6 Engineered Features:**

| Feature | Formula | AML Meaning |
|---------|---------|-------------|
| `out_degree` | count(outgoing edges) | Smurfing sources |
| `in_degree` | count(incoming edges) | Collection wallets |
| `fan_out_ratio` | out / (in + out + ε) | Layering patterns |
| `fan_in_ratio` | in / (in + out + ε) | Integration stage |
| `temporal_recency` | time_step / 49.0 | Recent activity |
| `time_delta` | mean\|t_v − t_u\| over neighbours | Temporal isolation |

**Chronological Data Splits:**

| Split | Time Steps | Nodes |
|-------|-----------|-------|
| Train | 1 – 34 | ~143,551 |
| Val | 35 – 42 | ~34,144 |
| Test | 43 – 49 | ~26,074 |

**O1 Verification**: 8/8 tests passed ✅

---

## 🌊 Objective 2 — Learnable Fourier Time Encoding

**Core Equation:**
```
φ(t) = [cos(ω₁t+φ₁), sin(ω₁t+φ₁), …, cos(ω₆₄t+φ₆₄), sin(ω₆₄t+φ₆₄)]
```

**Architecture:**

| Parameter | Shape | Init | Learnable |
|-----------|-------|------|-----------|
| omega (ω) | (64,) | N(0, 0.01) | ✅ Yes |
| phi (φ) | (64,) | zeros | ✅ Yes |
| **Output** | **(N, 128)** | — | — |

**Key Proof — Omega Actually Learns:**
```
Gradient formula: d/dω[cos(ωt+φ)] = −t · sin(ωt+φ)
```
Verified in Cell 21 — omega.grad exists and norm > 0 ✅

**Ablation Results:**

| Model | Val F1 | AUC-ROC |
|-------|--------|---------|
| No Time Encoding | 0.8124 | 0.8791 |
| Fixed Fourier | 0.8453 | 0.9012 |
| **Learnable Fourier (O2)** | **0.9300** | **0.9546** |
| **Improvement** | **+8.47%** | **+5.34%** |

**O2 Verification**: 8/8 tests passed ✅

---

## 📊 Dashboard Tabs

| Tab | Shows |
|-----|-------|
| 🏠 Overview | Stats, O1/O2 progress, mini charts |
| 📊 EDA | Class distribution, illicit ratio, degree stats |
| 🕸️ O1: Graph | Feature table, split visualiser, test results |
| 🌊 O2: Encoding | Architecture diagram, wave patterns, omega evolution |
| 📈 Training | Loss + F1 + AUC curves (20 epochs) |
| 🔬 Ablation | 3-model comparison bar charts |
| 🎓 Summary | Full dissertation review report |

---

## 📤 Outputs Saved (After Running Notebook)

All saved to `MyDrive/TemporalAML/`:

```
temporal_graph.pt              ← PyG Data object (O1)
fourier_encoding_model.pt      ← Trained FourierTimeEncoding weights
minimal_tgat.pt                ← Full MinimalTGAT model weights

results/
├── eda_class_distribution.png
├── eda_illicit_ratio.png
├── eda_feature_distribution.png
├── eda_degree_distribution.png
├── sample_subgraph.png
├── fourier_encoding_heatmap.png
├── fourier_waves.png
├── encoding_similarity.png
├── omega_evolution.png
├── training_loss.png
├── validation_f1.png
├── training_curves.png
├── ablation_comparison.png
├── class_weights.png
├── class_weights.json
├── training_history.csv
├── ablation_results.csv
├── omega_initial.npy
├── omega_final.npy
├── results_summary.json       ← Feeds the dashboard live
└── phase1_summary.txt         ← Full text report
```

---

## 🎓 For Your Dissertation Review — Show in This Order

1. **Cell 6–9** — EDA plots (dark-theme, publication-ready)
2. **Cell 16** — PyG Data object summary (all stats printed)
3. **Cell 17** — Subgraph visualisation (red=illicit, blue=licit, arrows=flow)
4. **Cell 19** — `FourierTimeEncoding` class (show the code + architecture)
5. **Cell 21** — Gradient verification (prove ω learns)
6. **Cell 25** — Omega evolution + training curves
7. **Cell 26** — Ablation table (+8.47% improvement)
8. **Cell 28** — Full Phase 1 summary report
9. **Dashboard** — Open `frontend/index.html` for live interactive summary

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Core language |
| PyTorch | 2.0+ | Deep learning |
| PyTorch Geometric | Latest | Graph neural networks |
| FastAPI | 0.104 | Backend REST API |
| Chart.js | 4.4 | Frontend charts |
| NetworkX | Latest | Graph visualisation |
| Scikit-learn | Latest | Metrics + normalisation |
| Pandas / NumPy | Latest | Data processing |
| Matplotlib / Seaborn | Latest | Research plots |

---

## ⚠️ Important Notes

- **Data leakage prevention**: `StandardScaler` is fit **only on train nodes** (t ≤ 34), then applied to all
- **Class imbalance**: Per-timestep class weights `w[t] = n_licit[t] / n_illicit[t]` applied during training
- **Reproducibility**: All seeds set to `42` (Python, NumPy, PyTorch, CUDA)
- **Phase 2 scope**: Full TGAT, pattern detection heads, GNNExplainer, baselines — NOT in this notebook

---

## 📧 Project Info

**Project**: TemporalAML — Dissertation Phase 1
**Degree**: M.Tech Data Science
**Platform**: Google Colab (Free T4 GPU)
**Dataset**: [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
