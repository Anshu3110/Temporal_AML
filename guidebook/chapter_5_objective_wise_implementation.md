# Chapter 5: Objective-wise Implementation

This chapter details the design, algorithms, libraries, and coding strategies necessary to execute the five core objectives of the TemporalAML dissertation.

---

## 5.1 Objective O1: Temporal Graph Construction
*   **Goal**: Ingest the raw Elliptic Bitcoin Dataset and construct a clean, continuous-time directed graph suitable for temporal message passing.

### 5.1.1 Algorithm & Strategy
1.  **Read and Load**: Load `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, and `elliptic_txs_features.csv` into Pandas dataframes.
2.  **Filter Unknowns for Labels**: Keep all nodes in the feature matrix, but create binary split masks. Create a training mask containing only nodes labeled as `1` or `2` up to time step 34.
3.  **Remap Node Identifiers**: Graph engines require indices in $[0, N-1]$. Build a lookup map `mapping = {raw_id: index}` and apply it to both the feature table and the edge list.
4.  **Extract Timestamps**: In the raw dataset, Column 2 is `time_step`. We treat this `time_step` (range 1 to 49) as our continuous-time parameter $t$.

### 5.1.2 Data Structures & PyTorch Geometric Classes
*   **`torch_geometric.data.Data`**: The container representing the graph.
    *   `x`: Node features. Tensors of shape $[N, D]$, where $N = 203769$, $D = 165$ (excluding transaction ID and timestamp).
    *   `edge_index`: Graph connections. Tensor of shape $[2, E]$ of type `torch.long`, where $E = 234355$.
    *   `y`: Labels. Tensor of shape $[N]$ of type `torch.long`. Values: `1` (Illicit), `0` (Licit), `-1` (Unknown).
    *   `time`: Node timestamps. Tensor of shape $[N]$ of type `torch.float`.
    *   `train_mask`, `val_mask`, `test_mask`: Boolean tensors of shape $[N]$ filtering active training, validation, and test indices.

---

## 5.2 Objective O2: Learnable Fourier Time Encoding
*   **Goal**: Implement a time-encoding module that converts raw time differences $\Delta t$ into a vector of learnable frequency sinusoids, embedded directly into the TGAT attention calculation.

### 5.2.1 Design of the Learnable Fourier Encoder
Instead of fixing frequencies to static values (e.g., $10^{-4}$), we declare a learnable parameter tensor of size $[d_t/2]$:

```python
import torch
import torch.nn as nn
import math

class LearnableFourierTimeEncoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        # Frequencies are initialized exponentially or randomly
        init_freqs = torch.randn(out_channels // 2)
        self.freqs = nn.Parameter(init_freqs)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t has shape [BatchSize] or [NumEdges]
        # Multiply by freqs: [BatchSize, 1] * [1, out_channels // 2] -> [BatchSize, out_channels // 2]
        delta_t = delta_t.unsqueeze(-1)
        freq_prod = delta_t * self.freqs.unsqueeze(0)
        
        # Calculate sine and cosine
        cos_part = torch.cos(freq_prod)
        sin_part = torch.sin(freq_prod)
        
        # Concatenate: shape [BatchSize, out_channels]
        time_code = torch.cat([cos_part, sin_part], dim=-1)
        # Apply normalization scaling
        return time_code * math.sqrt(2.0 / self.out_channels)
```

### 5.2.2 Integration inside TGAT Layer
In the TGAT attention layer, we compute dynamic time-aware features:
1.  Compute the time difference for every edge $(i, j)$ in the neighborhood: $\Delta t = t_i - t_e$.
2.  Project the time encoding: $\mathbf{t}_{ij} = \text{FourierEncoder}(\Delta t)$.
3.  Concatenate target representation and $\Phi(0)$ to form the Query $\mathbf{q}_i$.
4.  Concatenate source representation and time encoding $\mathbf{t}_{ij}$ to form the Key $\mathbf{k}_j$ and Value $\mathbf{v}_j$:
    $$\mathbf{k}_j = [\mathbf{h}_j \parallel \mathbf{t}_{ij}] \mathbf{W}_K, \quad \mathbf{v}_j = [\mathbf{h}_j \parallel \mathbf{t}_{ij}] \mathbf{W}_V$$
5.  Compute scaled dot-product attention scores to combine neighborhood messages.

---

## 5.3 Objective O3: Multi-Pattern Money Laundering Detection
*   **Goal**: Implement three classification heads sharing a single underlying TGAT embedding to identify Circular, Layering, and Smurfing patterns.

### 5.3.1 Defining Patterns (Supervised Mapping)
Since the raw Elliptic Dataset only provides binary classes (Licit/Illicit), we extract the multi-pattern ground-truth labels by running graph mining algorithms on the raw edgelist:
1.  **Circular Labels**: Run cycle detection algorithms (DFS/Tarjan's) on the directed graph. Any node belonging to a cycle of length $\le 5$ is flagged as Circular ($y_{\text{circ}} = 1$).
2.  **Layering Labels**: Track linear chains. Nodes participating in sequential paths of length $\ge 3$ (with sequential times) are flagged as Layering ($y_{\text{lay}} = 1$).
3.  **Smurfing Labels**: Nodes with an out-degree $\ge 10$ (Fan-out) or in-degree $\ge 10$ (Fan-in) within a short window are flagged as Smurfing ($y_{\text{smurf}} = 1$).

### 5.3.2 Shared Encoder and Classifier Model
The classification module shares a backbone embedding:

```python
class MultiPatternTGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.time_encoder = LearnableFourierTimeEncoder(hidden_channels)
        self.tgat_layer = TGATConv(in_channels, hidden_channels, self.time_encoder)
        
        # Classification Heads
        self.circular_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.layering_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.smurfing_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
```

---

## 5.4 Objective O4: Explainable Subgraphs (GNNExplainer)
*   **Goal**: Produce time-ordered, compliance-ready explanations for any flagged transaction using PyG's GNNExplainer module.

### 5.4.1 GNNExplainer Setup & Edge Mask Optimization
We freeze the trained `MultiPatternTGAT` model and run explainability for a target flagged node $v$:

```python
from torch_geometric.explain import Explainer, GNNExplainer

# Setup PyG Explainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass',
        task_level='node',
        return_type='probs'
    )
)

# Generate explanation
explanation = explainer(data.x, data.edge_index, index=target_node_idx)
edge_mask = explanation.edge_mask  # Importance score for each edge in the computational graph
```

### 5.4.2 Compliance-Ready Output (Suspicious Activity Reports)
For any suspicious alert, the system generates a **SAR JSON document**:
*   **Target Transaction**: Node ID, estimated amount, timestamp.
*   **Explanatory Path**: The time-ordered sequence of transactions leading to the target node.
*   **Key Features**: Feature attributes that triggered the classification (e.g. high input-to-output fee ratio).

---

## 5.5 Objective O5: Baseline Comparison & Evaluation
*   **Goal**: Set up a rigorous benchmarking suite comparing the proposed `TemporalAML` with five baselines: GCN, GraphSAGE, LSTM-GNN, static TGAT, and a rule-based system.

### 5.5.1 Baseline Setup
1.  **GCN**: Static Graph Convolution. Ignores timestamps, collapses time steps into static connections.
2.  **GraphSAGE**: Static Graph Sample and Aggregate. Uses inductive aggregation.
3.  **LSTM-GNN**: Combines an LSTM sequentially across time steps with a spatial GCN.
4.  **Static TGAT**: Uses default sinusoidal time encodings (non-learnable frequencies).
5.  **Rule-based Heuristic**: Custom script replicating classic exchange rules (e.g., alert if volume $> 50$ BTC and velocity $< 1$ minute).

### 5.5.2 Evaluation Metrics
Due to the 90.3% Licit class imbalance, comparisons must evaluate the minority class (Illicit class `1`):
*   **Precision**, **Recall**, **F1-Score**, and **AUC-ROC** are calculated on the test split (time steps 35-49).
*   **Ablation Study**: Compare `TemporalAML` (learnable frequencies) with `Static TGAT` (fixed frequencies) to demonstrate the benefit of backpropagating temporal parameters.

---

## Connection to Dissertation Objectives

This chapter maps out the explicit implementation blueprint for **Objectives O1, O2, O3, O4, and O5**, serving as the code-design manual for your dissertation development.

---

## Summary
*   **O1** requires remapping IDs and packaging features and timestamps in PyG's `Data` object.
*   **O2** defines a PyTorch custom layer with learnable parameters for Fourier time encodings.
*   **O3** extracts laundering labels via graph mining (Tarjan's, linear chain tracking) and trains three heads sharing a TGAT encoder.
*   **O4** implements GNNExplainer to extract sparse subgraphs and generate SARs.
*   **O5** details comparative validation metrics and baselines.

## Key Takeaways
1. Graph remapping is essential before tensors can be passed to PyG Conv modules.
2. Frequencies are registered as `nn.Parameter` to allow optimizer updates.
3. Unlabeled nodes are retained during message aggregation but masked out in BCE calculations.

## Practical Implementation Checklist
- [ ] Implement the `LearnableFourierTimeEncoder` class in PyTorch.
- [ ] Write a cycle-detection routine to generate Circular Transfer labels.
- [ ] Configure the `GNNExplainer` module for validation runs.

## Recommended Research Papers
1. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
2. Hamilton, W., et al. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017.

## Suggested Coding Exercises
1. Write a script to isolate the 2-hop neighborhood of a node using PyG's `k_hop_subgraph` utility.
2. Implement an FFN head with a dropout layer to prevent overfitting on minority class features.
