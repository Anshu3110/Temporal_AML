# Chapter 4: Project Architecture

This chapter details the architecture of the **TemporalAML** system. We present the structural pipeline mapping how raw ledger files are ingested, transformed, modeled through deep neural attention layers, and finally translated into explainable compliance alerts.

---

## 4.1 System Pipeline Overview

The complete TemporalAML framework consists of three distinct phases:
1.  **Data Ingestion and Graph Construction**: Parsing raw CSV files and mapping them into a clean, indices-mapped temporal graph structure.
2.  **Temporal Representation Learning (TGAT)**: Computing continuous-time embeddings using multi-head attention and learnable Fourier time-delta encodings.
3.  **Multi-Pattern Classification & Explainability**: Forwarding embeddings to shared task-specific heads to detect laundering types, and deploying GNNExplainer to extract evidence subgraphs.

Here is the end-to-end data flow:

```
                  [ Raw CSV Files ]
             (Classes, Edgelist, Features)
                          │
                          ▼
                  [ Data Cleaning ]
          (Remap IDs, Mask Unlabeled Nodes)
                          │
                          ▼
               [ Graph Construction ]
           (Build PyG Spatial-Temporal Graph)
                          │
                          ▼
              [ Feature Engineering ]
        (Standardize attributes, Add continuous time)
                          │
                          ▼
            [ Learnable Fourier Encoding ]
         (Map continuous delta-t to vector)
                          │
                          ▼
           [ Temporal Graph Attention (TGAT) ]
       (Compute spatial-temporal query/key/values)
                          │
                          ▼
             [ Shared Embedding Layer ]
         (Consolidated representation h_v(t))
                          │
                          ▼
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  [ Circular Head ] [ Layering Head ] [ Smurfing Head ]
     (Multi-label Classification Output y_hat)
                          │
                          ▼
                  [ GNNExplainer ]
       (Extract sparse, high-weight dynamic subgraphs)
                          │
                          ▼
             [ Compliance Alert Packet ]
         (Evidence Subgraph + Drafted SAR)
```

---

## 4.2 Module-by-Module Description

Let's dissect each functional block in this architecture.

### 4.2.1 Data Cleaning & Graph Construction
*   **Input**: Raw `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, and `elliptic_txs_features.csv`.
*   **Operation**: 
    1.  Eliminate records containing corrupted values.
    2.  Create a dictionary mapping each string/long-int transaction ID to a continuous integer in the range $[0, N-1]$.
    3.  Create the `edge_index` PyTorch tensor.
    4.  Extract target labels (`y`) and mask arrays identifying training, validation, and test indices using the temporal step boundary (e.g., train on steps 1-34, val on 35-40, test on 41-49).
*   **Output**: A clean PyTorch Geometric `Data` object containing nodes, edges, labels, and timestamps.

### 4.2.2 Learnable Fourier Time Encoder
*   **Input**: Continuous-time difference between target prediction time $t$ and historical transaction interaction time $t_e$:
    $$\Delta t = t - t_e$$
*   **Operation**: Compute cosine and sine projections across $d_t/2$ frequencies:
    $$\Phi_{d_t}(\Delta t) = \sqrt{\frac{2}{d_t}} \left[ \cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots \right]^{\top}$$
    *Note*: The frequency vector $\mathbf{\omega} = [\omega_1, \dots]$ is registered as a PyTorch `nn.Parameter` and updated via gradients.
*   **Output**: Time-difference representation vector of dimension $d_t$.

### 4.2.3 Temporal Graph Attention Network (TGAT Encoder)
*   **Input**: Node features, remapped topological connections, and the learnable time encodings.
*   **Operation**:
    1.  Perform neighborhood sampling: extract the most recent $M$ edges for each node.
    2.  For each node $i$, project its features and time encoding $\Phi(0)$ into Query vector $\mathbf{q}_i$.
    3.  For each sampled neighbor $j$, project neighbor features and time-delta encoding $\Phi(t - t_e)$ into Key $\mathbf{k}_j$ and Value $\mathbf{v}_j$.
    4.  Compute multi-head attention weights:
        $$\alpha_{ij} = \text{Softmax}_j \left( \frac{\mathbf{q}_i \mathbf{k}_j^{\top}}{\sqrt{d_k}} \right)$$
    5.  Aggregate values:
        $$\mathbf{h}_i(t) = \text{FFN}\left( \sum_{j} \alpha_{ij} \mathbf{v}_j \right)$$
*   **Output**: A time-aware structural node representation vector $\mathbf{h}_i(t) \in \mathbb{R}^{d_{\text{embed}}}$.

### 4.2.4 Shared Embedding & Multi-Pattern Classifier Heads
Instead of deploying three completely separate networks, TemporalAML implements a **multi-task shared learning framework**.
*   **Shared Encoder**: A single TGAT stack generates the embedding $\mathbf{h}_i(t)$ capturing the general spatial-temporal dynamics of the transaction.
*   **Multi-Head Task Layers**: The shared embedding is passed to three independent Feed-Forward Neural Network (FFN) heads:
    1.  **Circular Head**: Classifies if the node is participating in a circular looping transaction.
    2.  **Layering Head**: Classifies if the node is a hop in a layering chain.
    3.  **Smurfing Head**: Classifies if the node is a source/sink in a structuring (smurfing) flow.
*   **Loss Function**: Total loss is the weighted sum of Binary Cross Entropy losses across the heads:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{circ}}\mathcal{L}_{\text{circ}} + \lambda_{\text{lay}}\mathcal{L}_{\text{lay}} + \lambda_{\text{smurf}}\mathcal{L}_{\text{smurf}}$$

### 4.2.5 GNNExplainer Interpretation Module
*   **Input**: Suspicious flagged nodes from the prediction heads, and the local computational subgraph.
*   **Operation**:
    1.  Isolate the $k$-hop temporal neighborhood around the flagged node.
    2.  Initialize a continuous mask over the neighborhood edges.
    3.  Optimize the edge masks using gradient descent to find the minimal set of transactions that preserves the model's high-confidence suspicious classification score.
    4.  Apply a threshold to the mask to filter out insignificant edges.
*   **Output**: A clean, human-readable, time-ordered subgraph containing only the transaction nodes and edges that explain the suspicious activity.

---

## Connection to Dissertation Objectives

*   **Objective O3 (Detect Circular, Layering, Smurfing)**: The shared embedding and multi-head design directly implements this multi-pattern detection task.
*   **Objective O4 (Generate Explainable Subgraphs)**: The integration of GNNExplainer at the end of the inference pipeline enables post-hoc interpretability of flagged transaction nodes.

---

## Summary
*   **TemporalAML** is designed as a unified three-phase pipeline.
*   It utilizes a **shared TGAT encoder** to map dynamic node states to vector spaces.
*   **Three task-specific heads** execute multi-label classification to identify Circular, Layering, and Smurfing patterns simultaneously.
*   **GNNExplainer** runs on flagged outputs, optimizing edge masks to isolate explanatory transaction sequences for compliance teams.

## Key Takeaways
1. Multi-task learning prevents duplicate GNN passes, saving memory and training time.
2. The shared encoder learns generalized representations of transaction dynamics, while heads focus on specific topological geometries.
3. GNNExplainer translates mathematical vector states into clean visual evidence packets.

## Practical Implementation Checklist
- [ ] Diagram the tensor input/output shapes for each block in the pipeline.
- [ ] Define the class interfaces for `FourierEncoder`, `TGATLayer`, `MultiPatternClassifier`, and `GNNExplainer`.
- [ ] Plan the database schema or data structure for caching the intermediate embeddings to speed up explainability.

## Recommended Research Papers
1. Caruana, R. (1997). *Multitask Learning.* Machine Learning, 28(1), 41-75.
2. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.

## Suggested Coding Exercises
1. Draw a flowchart showing the forward pass tensor shapes (e.g. `[batch_size, num_features] -> [batch_size, embed_dim] -> [batch_size, 3]`).
2. Write a PyTorch skeletal model wrapper combining a GNN feature extractor with three linear classification heads.
