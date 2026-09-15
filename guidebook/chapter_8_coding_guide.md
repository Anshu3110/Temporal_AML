# Chapter 8: Coding Guide

This chapter details the codebase structure, configurations, and coding guidelines for the TemporalAML project. Following this structure ensures your project is modular, reproducible, and ready for deployment or review by examiners.

---

## 8.1 Directory Structure

To keep your dissertation code clean and maintainable, organize the codebase using the following modular structure:

```
TemporalAML/
  ├── config/
  │     └── config.yaml           # Hyperparameters, paths, and training settings
  ├── data/
  │     ├── raw/                  # Placeholder for Elliptic raw CSVs
  │     └── processed/            # Serialized PyTorch Geometric Data tensors
  ├── src/
  │     ├── __init__.py
  │     ├── data_loader.py        # ID mapping, cleaning, and PyG graph construction
  │     ├── models.py             # Fourier Time Encoder, TGAT layer, and heads
  │     ├── train.py              # Optimization routines and training loop
  │     ├── evaluate.py           # Metrics calculation and baseline evaluations
  │     └── explain.py            # GNNExplainer implementation and SAR generator
  ├── notebooks/
  │     └── eda.ipynb             # Exploratory Data Analysis notebooks
  ├── artifacts/
  │     ├── models/               # Saved checkpoints (.pt / .pth)
  │     └── logs/                 # CSV logs and TensorBoard records
  ├── README.md                   # Project overview and run commands
  └── requirements.txt            # Python library dependencies
```

---

## 8.2 Modular Code Architecture

### 8.2.1 Data Loader Module (`src/data_loader.py`)
This file is responsible for Objective **O1**. It handles reading CSVs, building index maps, and assembling the PyG `Data` object.

```python
import os
import pandas as pd
import torch
from torch_geometric.data import Data

class EllipticDatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_data(self) -> Data:
        # Load CSVs
        classes_df = pd.read_csv(os.path.join(self.data_dir, 'elliptic_txs_classes.csv'))
        edges_df = pd.read_csv(os.path.join(self.data_dir, 'elliptic_txs_edgelist.csv'))
        features_df = pd.read_csv(os.path.join(self.data_dir, 'elliptic_txs_features.csv'), header=None)
        
        # Build node ID map
        tx_ids = features_df[0].values
        id_map = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
        
        # Map features and times
        features = torch.tensor(features_df.iloc[:, 2:].values, dtype=torch.float)
        times = torch.tensor(features_df[1].values, dtype=torch.float)
        
        # Map labels: '1' -> 1, '2' -> 0, 'unknown' -> -1
        labels_map = {'1': 1, '2': 0, 'unknown': -1}
        labels = classes_df['class'].map(labels_map).values
        y = torch.tensor(labels, dtype=torch.long)
        
        # Map edges
        edges_df['src'] = edges_df['txId1'].map(id_map)
        edges_df['dst'] = edges_df['txId2'].map(id_map)
        # Drop edges connected to unmapped IDs (if any)
        edges_clean = edges_df.dropna(subset=['src', 'dst'])
        edge_index = torch.tensor(edges_clean[['src', 'dst']].values.T, dtype=torch.long)
        
        # Construct PyG Data object
        data = Data(x=features, edge_index=edge_index, y=y, time=times)
        return data
```

### 8.2.2 Models Module (`src/models.py`)
This file defines the neural architecture components for **O2** and **O3**.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LearnableFourierTimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.freqs = nn.Parameter(torch.randn(dim // 2))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        delta_t = delta_t.unsqueeze(-1)
        freq_prod = delta_t * self.freqs.unsqueeze(0)
        cos_part = torch.cos(freq_prod)
        sin_part = torch.sin(freq_prod)
        time_embedding = torch.cat([cos_part, sin_part], dim=-1)
        return time_embedding * (2.0 / self.dim) ** 0.5

class TemporalAMLNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 1):
        super().__init__()
        self.time_encoder = LearnableFourierTimeEncoder(hidden_dim)
        
        # Spatial-Temporal Attention
        self.conv1 = GATConv(in_dim + hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim + hidden_dim, hidden_dim, heads=1, concat=False)
        
        # Shared Embedding projecting to pattern heads
        self.circ_head = nn.Linear(hidden_dim, num_classes)
        self.lay_head = nn.Linear(hidden_dim, num_classes)
        self.smurf_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, times):
        # In a real TGAT layer, time-deltas would be calculated per edge.
        # This implementation shows how the time vectors are concatenated to node features.
        t_embed = self.time_encoder(times)
        x_temp = torch.cat([x, t_embed], dim=-1)
        
        h = self.conv1(x_temp, edge_index)
        h = torch.relu(h)
        h_temp = torch.cat([h, t_embed], dim=-1)
        
        emb = self.conv2(h_temp, edge_index)
        emb = torch.relu(emb)
        
        # Multi-task predictions
        y_circ = torch.sigmoid(self.circ_head(emb))
        y_lay = torch.sigmoid(self.lay_head(emb))
        y_smurf = torch.sigmoid(self.smurf_head(emb))
        
        return y_circ, y_lay, y_smurf, emb
```

---

## 8.3 Saving, Loading, & Model Tracking

### 8.3.1 Training Execution (`src/train.py`)
This script manages model training, logging, checkpointing, and evaluation.

```python
import torch
import torch.optim as optim
from src.data_loader import EllipticDatasetLoader
from src.models import TemporalAMLNet

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using target training device: {device}")
    
    # Load dataset
    loader = EllipticDatasetLoader(data_dir='data/raw')
    data = loader.load_data().to(device)
    
    # Initialize Model
    model = TemporalAMLNet(in_dim=data.num_node_features, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()
    
    # Create simple binary masks for training (excluding unknown class -1)
    train_mask = (data.y != -1) & (data.time <= 34)
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (Dummy target matching for multi-pattern illustration)
        y_c, y_l, y_s, _ = model(data.x, data.edge_index, data.time)
        
        # Label mask filters
        y_true = data.y[train_mask].float().unsqueeze(-1)
        
        # Calculate loss
        loss = criterion(y_c[train_mask], y_true) # Replace with multi-task loss as needed
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")
            
    # Save checkpoint
    checkpoint_path = 'artifacts/models/temporal_aml_best.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}")

if __name__ == "__main__":
    run_training()
```

---

## 8.4 Logging and Visualization Guidelines

*   **Logging**: Avoid using raw `print` statements in production code. Use Python's built-in `logging` module to output levels (`INFO`, `WARNING`, `ERROR`) to both the console and a file.
*   **TensorBoard / WandB**: Track training metrics in real-time. Log the training loss, validation F1, validation AUC-ROC, and frequency parameters $\mathbf{\omega}$ to trace learning stability.
*   **Visualizing Explanations**: Use `networkx` and `matplotlib` to plot explanatory subgraphs, highlighting high-importance edges in red and printing transaction paths chronologically.

---

## Connection to Dissertation Objectives

This chapter provides the concrete modular codebase design required to execute **O1, O2, O3, O4, and O5** as a clean, production-ready Python package.

---

## Summary
*   We defined a clean **directory structure** separating source code, data files, and saved artifacts.
*   We implemented a robust **data loader** class in `data_loader.py` representing Objective **O1**.
*   We detailed modular class designs for the **learnable time encoder** and **multi-head model** in `models.py` (**O2/O3**).
*   We provided templates for saving and loading checkpoint parameters.

## Key Takeaways
1. Code modularity is vital; avoid coding the entire pipeline in a single Jupyter Notebook.
2. Saving optimizer states along with model weights is necessary to resume training cleanly.
3. Keeping parameters configured via a separate file (e.g. `config.yaml`) enables rapid hyperparameter tuning.

## Practical Implementation Checklist
- [ ] Create the project directory structure on your system.
- [ ] Save the dependencies list in `requirements.txt`.
- [ ] Create the config directory and write parameters (learning rate, epochs) into a file.

## Recommended Research Papers
1. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

## Suggested Coding Exercises
1. Expand the model definition in `models.py` to allow configuration of hidden layers via a parameters dictionary.
2. Implement a logging function in `train.py` that writes epoch statistics to a CSV file.
