# Chapter 3: Dataset Understanding

This chapter details the dataset used to build the TemporalAML pipeline: the **Elliptic Bitcoin Dataset**. We will break down its file structure, column schemas, features, label distributions, graph representations, and data cleaning/preprocessing steps.

---

## 3.1 Overview of the Elliptic Bitcoin Dataset

The Elliptic Bitcoin Dataset is a transaction graph dataset mapping real-world Bitcoin transactions. It was released in 2019 by Elliptic (a blockchain analytics company) in collaboration with researchers from MIT. It is currently the most popular benchmark dataset for Anti-Money Laundering (AML) research in Graph Neural Networks.

Importantly, the graph is a **transaction-to-transaction** graph. 
*   **Nodes** represent Bitcoin *transactions* (not wallet addresses).
*   **Edges** represent the *flow* of Bitcoin from one transaction to another. (This represents the inputs and outputs of a transaction. For example, if transaction A's output is spent as an input in transaction B, there is a directed edge from node A to node B).
*   **Time steps** represent discrete blocks of time.

---

## 3.2 Dataset Structure & CSV Files

The dataset is compressed into a single zip file containing three main CSV files:

```
elliptic_bitcoin_dataset/
  ├── elliptic_txs_classes.csv
  ├── elliptic_txs_edgelist.csv
  └── elliptic_txs_features.csv
```

Let's explore each file and its schema in detail.

### 3.2.1 `elliptic_txs_classes.csv`
This file contains the classification labels for each node (transaction).

*   **Schema**:
    *   `txId`: A unique integer identifying the transaction (node).
    *   `class`: The classification label. It has three possible values:
        1.  `1`: **Illicit** (transactions linked to ransomware, scams, darknet markets, terrorist financing, etc.).
        2.  `2`: **Licit** (transactions linked to exchanges, wallet providers, miners, merchants, etc.).
        3.  `unknown`: Transactions that have not been classified.
*   **Label Distribution**:
    *   Total Nodes: $203,769$
    *   Licit Nodes (`2`): $42,019$ ($\approx 21\%$)
    *   Illicit Nodes (`1`): $4,545$ ($\approx 2\%$)
    *   Unknown Nodes (`unknown`): $157,205$ ($\approx 77\%$)

### 3.2.2 `elliptic_txs_edgelist.csv`
This file defines the directed graph topology.

*   **Schema**:
    *   `txId1`: The source transaction ID.
    *   `txId2`: The target transaction ID.
*   **Total Edges**: $234,355$ directed edges.
*   *Note*: In Bitcoin, an output can only be spent once. Therefore, the graph is a Directed Acyclic Graph (DAG) when considering transaction flows.

### 3.2.3 `elliptic_txs_features.csv`
This file contains the feature vectors for each transaction. It does not have header names in the raw CSV, but it contains 167 columns.

*   **Schema**:
    *   Column 1: `txId` (matches the classes file).
    *   Column 2: `time_step` (an integer from 1 to 49, representing two-week intervals).
    *   Columns 3 to 95: **Local Features** (93 columns).
    *   Columns 96 to 167: **Aggregated Features** (72 columns).

---

## 3.3 Detailed Feature Analysis

The 165 features (excluding `txId` and `time_step`) are anonymous for privacy and business reasons, but their structural meaning is documented:

### 3.3.1 Local Features (Columns 3 to 95)
These features represent statistics computed directly from the single transaction itself:
*   Transaction fee paid.
*   Number of inputs and number of outputs.
*   Volume of BTC transferred.
*   Average/min/max BTC sent to outputs.
*   Average/min/max BTC received from inputs.

### 3.3.2 Aggregated Features (Columns 96 to 167)
These features are obtained by aggregating local feature statistics of a node's immediate neighbors:
*   They are computed by taking one-step backward and one-step forward from the center node.
*   Examples include: sum/average fees of parent transactions, standard deviation of input sizes of child transactions, etc.
*   *Critically*, these aggregates are static and pre-computed. They represent spatial local neighborhood summaries.

---

## 3.4 Temporal Structure (Time Steps)

*   **Discrete Intervals**: The dataset is divided into **49 time steps**.
*   **Interval Spacing**: Each time step represents a window of approximately 2 weeks.
*   **Disconnected Time steps**: Edges only connect transactions that occur within the *same* time step. There are no edges in `elliptic_txs_edgelist.csv` connecting a node in time step $T$ to a node in time step $T+1$.
*   **Why is this a challenge?** Because the raw dataset has segmented the graph, we must handle this during modeling. However, for a truly temporal model, we must represent time differences. Since the dataset does not provide raw minute-level timestamps, we use the `time_step` integer or synthetic continuous offsets within time steps to simulate dynamic streaming.

---

## 3.5 Class Imbalance & The "Unknown" Node Challenge

The dataset is heavily imbalanced:

```
Illicit (4,545 nodes)  [██] 2%
Licit (42,019 nodes)   [██████████████████] 21%
Unknown (157,205 nodes)[██████████████████████████████████████████████████] 77%
```

### 3.5.1 The Class Imbalance Problem
Only $\approx 9.7\%$ of the *labeled* nodes are illicit. If a classifier predicts all nodes as "Licit", it achieves an accuracy of $90.3\%$. Thus, standard Accuracy is a useless training metric. We must use weighted losses and F1-score evaluation.

### 3.5.2 The Unknown Nodes (Semi-Supervised Learning)
The $77\%$ unknown nodes cannot be used directly in the loss function for supervised training. However, they constitute the majority of the graph structure. If we discard them, the graph breaks down into thousands of tiny disconnected components.
*   **Solution**: We retain unknown nodes in the graph structure during GNN message passing, allowing them to propagate structural features. We mask out their predictions during loss computation.

---

## 3.6 Preprocessing & Graph Construction Pipeline

To prepare the dataset for PyTorch Geometric, the raw files must go through a structured pipeline:

```
[Raw CSVs] 
    │
    ▼
[Clean Unknown Classes] ────► Mask labels for training, keep nodes for structure
    │
    ▼
[Remap IDs] ────────────────► Map txId (e.g. 230421) to 0-indexed indices (0, 1, 2...)
    │
    ▼
[Build Edge Index] ─────────► Create torch tensor of shape [2, E]
    │
    ▼
[Compile Node Features] ────► Build float tensor of shape [N, 166] (features + time)
    │
    ▼
[Generate Labels] ──────────► Map class '1' -> 1, '2' -> 0, 'unknown' -> -1
```

1.  **Index Remapping**: Transaction IDs are arbitrary large integers. We must map them to a continuous sequence from $0$ to $N-1$ to build adjacency matrices.
2.  **Handling Edges**: The source and target node IDs in the edgelist must be mapped to their new indices.
3.  **Temporal Node Assignment**: We extract the `time_step` column and save it as node timestamps.
4.  **PyTorch Geometric PyG Data object**:
    We wrap the tensors into a `torch_geometric.data.Data` object:
    *   `data.x`: Node features $[N, 166]$.
    *   `data.edge_index`: Adjacency representation $[2, E]$.
    *   `data.y`: Node labels $[N]$ (with values `0`, `1`, or `-1`).
    *   `data.time`: Node timestamps $[N]$.

---

## Connection to Dissertation Objectives

*   **Objective O1 (Construct Temporal Graph)**: This chapter defines the exact mapping process to take the raw Elliptic files and convert them into a continuous-time graph suitable for spatial-temporal message passing.
*   **Objective O5 (Comparison)**: Understanding the data splits (often splitting training on time steps 1-34 and testing on 35-49) is critical to establishing a rigorous comparison baseline.

---

## Summary
*   The **Elliptic Bitcoin Dataset** is a transaction-to-transaction DAG.
*   It consists of **3 files**: node classes, edgelist, and features (165 topological/local attributes + `txId` + `time_step`).
*   There are **49 time steps**, and edges only occur within the same time step.
*   Over **77% of nodes are unlabeled (unknown)**. They must be kept in the graph topology to propagate messages but masked during training.
*   Severe class imbalance ($\approx 9.7\%$ illicit of labeled) requires weighted loss.

## Key Takeaways
1. Nodes are transactions, edges are BTC spends.
2. The $77\%$ unknown nodes are valuable structural components for message aggregation.
3. Standard splits use a temporal split (train on early time steps, test on later time steps) to prevent temporal data leakage.

## Practical Implementation Checklist
- [ ] Download the Elliptic Bitcoin Dataset from Kaggle or GitHub.
- [ ] Write a Pandas script to parse the files and compute basic statistics (total nodes, edges, class distributions).
- [ ] Map the string classes (`1`, `2`, `unknown`) to integers (`1`, `0`, `-1`).
- [ ] Construct the `edge_index` and map transaction IDs to 0-indexed values.

## Recommended Research Papers
1. Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks.* arXiv preprint arXiv:1908.02591.
2. Hu, Y., et al. (2021). *Identify suspicious accounts in Bitcoin using graph representation learning.* IEEE Transactions on Knowledge and Data Engineering.

## Suggested Coding Exercises
1. Write a Python script to load the CSVs and plot the number of Licit and Illicit nodes per time step using matplotlib.
2. Implement the ID remapping function in pure NumPy to convert raw `txId`s in the edgelist to continuous indices.
3. Construct a PyTorch Geometric `Data` object using the processed tensors and print its attributes.
