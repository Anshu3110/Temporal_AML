# Chapter 6: Algorithms (Pseudocode)

This chapter provides the algorithmic pseudocode for the critical components of the TemporalAML pipeline. These formal algorithms serve as direct specifications for your dissertation methodology chapter and your code implementation.

---

## 6.1 Data Processing & Graph Construction

### Algorithm 1: Graph Construction and Index Mapping
*   **Goal**: Ingest raw transaction files, remap IDs, compile features, and construct spatial-temporal PyG structures.

```
Input: classes_csv, edgelist_csv, features_csv
Output: Node features X, Edge Index, Node Labels Y, Timestamps T

1:  Let raw_features = LoadCSV(features_csv)
2:  Let raw_classes = LoadCSV(classes_csv)
3:  Let raw_edgelist = LoadCSV(edgelist_csv)
4:  
5:  Initialize dictionary mapping: NodeId -> 0-indexed integers
6:  Initialize node_count = 0
7:  
8:  For each row in raw_features:
9:      Let tx_id = row['txId']
10:     If tx_id not in mapping:
11:         mapping[tx_id] = node_count
12:         node_count = node_count + 1
13:
14: Initialize X = Tensor of shape [node_count, 165]
15: Initialize T = Tensor of shape [node_count]
16: Initialize Y = Tensor of shape [node_count]
17: 
18: For each row in raw_features:
19:     Let idx = mapping[row['txId']]
20:     T[idx] = row['time_step']
21:     X[idx] = row[features_columns] (Column 3 to 167)
22: 
23: For each row in raw_classes:
24:     Let idx = mapping[row['txId']]
25:     Let cls = row['class']
26:     If cls == '1': Y[idx] = 1 (Illicit)
27:     Else If cls == '2': Y[idx] = 0 (Licit)
28:     Else: Y[idx] = -1 (Unknown)
29: 
30: Initialize mapped_edges = List of pairs
31: For each row in raw_edgelist:
32:     Let src = mapping[row['txId1']]
33:     Let dest = mapping[row['txId2']]
34:     Append (src, dest) to mapped_edges
35: 
36: Edge Index = Transpose(Tensor(mapped_edges))
37: Return X, Edge Index, Y, T
```

---

## 6.2 Temporal Operations

### Algorithm 2: Temporal Neighborhood Sampling
*   **Goal**: Sample $M$ historical edges that occurred prior to target time $t$ for a node $i$.

```
Input: Node i, Target Time t, Graph G = (V, E, T), Sample Size M
Output: Sampled Neighbor Indices, Event Timestamps

1:  Initialize candidate_edges = Empty List
2:  For each incoming edge (j, i) in E:
3:      Let t_edge = timestamp of edge (j, i)
4:      If t_edge < t:
5:          Append (j, t_edge) to candidate_edges
6:  
7:  Sort candidate_edges descending by t_edge (most recent first)
8:  
9:  If Length(candidate_edges) >= M:
10:     sampled_edges = candidate_edges[0 : M]
11: Else:
12:     sampled_edges = candidate_edges (pad with self-loops or zero vectors if needed)
13: 
14: Return list of source nodes and timestamps from sampled_edges
```

### Algorithm 3: Learnable Fourier Encoding Forward Pass
*   **Goal**: Map a batch of continuous time differences $\Delta t$ into a vector of sinusoids using learnable frequencies $\mathbf{\omega}$.

```
Input: Time difference tensor delta_t of shape [B], Learnable parameters omega of shape [d_t / 2]
Output: Time embeddings Phi of shape [B, d_t]

1:  Let B = Length(delta_t)
2:  Initialize products matrix P of shape [B, d_t / 2]
3:  
4:  For i from 0 to B-1:
5:      For j from 0 to (d_t / 2) - 1:
6:          P[i, j] = delta_t[i] * omega[j]
7:  
8:  Compute Cosines: C = cos(P)  (Element-wise)
9:  Compute Sines: S = sin(P)    (Element-wise)
10: 
11: Concatenate Cosines and Sines along channels: Phi_unscaled = Concatenate(C, S, dim=-1)
12: 
13: Scale representation: Phi = Phi_unscaled * Sqrt(2.0 / d_t)
14: 
15: Return Phi
```

---

## 6.3 Neural Architecture Forward Pass

### Algorithm 4: TGAT Layer Forward Pass (Attention Computation)
*   **Goal**: Aggregate neighborhood representations using time-aware attention mechanisms.

```
Input: Target Node i, Node Features H, Timestamps T, Adjacency E, Neighbors M, Weight Matrices W_Q, W_K, W_V, Learnable frequencies omega
Output: Updated representation h_i_new

1:  Let current_time = T[i]
2:  Get temporal neighborhood: neighbors = TemporalNeighborSample(i, current_time, G, M)
3:  
4:  Time code of zero difference: phi_zero = LearnableFourierEncode(0, omega)
5:  Query vector: q_i = Concatenate(H[i], phi_zero) * W_Q
6:  
7:  Initialize keys list K = [], values list V = []
8:  
9:  For each neighbor (j, t_e) in neighbors:
10:     Let delta_t = current_time - t_e
11:     Let phi_delta = LearnableFourierEncode(delta_t, omega)
12:     Let k_j = Concatenate(H[j], phi_delta) * W_K
13:     Let v_j = Concatenate(H[j], phi_delta) * W_V
14:     Append k_j to K
15:     Append v_j to V
16: 
17: For each neighbor index j:
18:     Compute attention score: score_ij = (q_i * K[j]) / Sqrt(d_k)
19: 
20: Apply Softmax normalization: alpha_ij = Exp(score_ij) / Sum_over_neighbors(Exp(score_ik))
21: 
22: Compute aggregated message: m_i = Sum_over_neighbors(alpha_ij * V[j])
23: 
24: Update node state: h_i_new = FeedForward(Concatenate(H[i], m_i))
25: Return h_i_new
```

### Algorithm 5: Multi-Pattern Detection Model Forward Pass
*   **Goal**: Compute task embeddings and run predictions for Circular, Layering, and Smurfing classification.

```
Input: Input Graph Data (X, Edge Index, T)
Output: Predicted probabilities y_circ, y_lay, y_smurf

1:  Pass graph structure through TGAT Encoder:
2:      H = TGAT_Encoder(X, Edge Index, T)
3:  
4:  Pass shared embedding H to classification heads:
5:      y_circ_raw = FFN_Circular(H)
6:      y_lay_raw = FFN_Layering(H)
7:      y_smurf_raw = FFN_Smurfing(H)
8:  
9:  Apply Sigmoid outputs:
10:     y_circ = Sigmoid(y_circ_raw)
11:     y_lay = Sigmoid(y_lay_raw)
12:     y_smurf = Sigmoid(y_smurf_raw)
13: 
14: Return y_circ, y_lay, y_smurf
```

---

## 6.4 Training & Optimization

### Algorithm 6: Complete End-to-End Training Loop
*   **Goal**: Optimize both TGAT weights and learnable time frequencies using weighted BCE loss.

```
Input: Data DataObject, Epochs E_max, Learning Rate lr, Positive Weight w_pos, Task weights (l1, l2, l3)
Output: Trained model parameters, Optimal frequencies omega

1:  Initialize model parameters Theta, Frequencies omega
2:  Initialize Optimizer (Adam) for Theta and omega
3:  Define loss function BCE_Weighted(y_hat, y, weight = w_pos)
4:  
5:  For epoch from 1 to E_max:
6:      Set model to Training Mode
7:      Zero out gradients in Optimizer
8:      
9:      # Forward Pass
10:     y_circ, y_lay, y_smurf = ModelForward(Data.x, Data.edge_index, Data.time)
11:     
12:     # Compute losses on training mask (excluding unknown nodes where labels are -1)
13:     loss_circ = BCE_Weighted(y_circ[train_mask], Data.y_circ[train_mask])
14:     loss_lay = BCE_Weighted(y_lay[train_mask], Data.y_lay[train_mask])
15:     loss_smurf = BCE_Weighted(y_smurf[train_mask], Data.y_smurf[train_mask])
16:     
17:     # Multi-Task Joint Loss
18:     loss_total = (l1 * loss_circ) + (l2 * loss_lay) + (l3 * loss_smurf)
19:     
20:     # Backpropagation
21:     loss_total.backward()
22:     
23:     # Optimization step (updates Theta and omega)
24:     Optimizer.step()
25:     
26:     # Validation step
27:     Run Validation metrics (F1, AUC-ROC) on Data[val_mask]
28:     If Val F1 improved:
29:         Save Model state to Disk
30: 
31: Return optimal parameters
```

---

## 6.5 Evaluation & Explainability

### Algorithm 7: Explainability Pipeline (GNNExplainer for SAR Generation)
*   **Goal**: Extract structural evidence for a suspicious transaction node and format it as a Compliance report.

```
Input: Flagged target node v, Trained Model, Graph G = (V, E, T), Threshold theta
Output: Evidence Subgraph, SAR Draft

1:  Extract local computational subgraph of node v: G_sub = k_hop_subgraph(v, k=2)
2:  Initialize edge mask parameter: M_edge = Tensor of size [Length(G_sub.edges)] initialized to 0.5
3:  
4:  Freeze Model parameters Theta and Frequencies omega
5:  Initialize optimizer for M_edge
6:  
7:  For step from 1 to Epochs_Explainer:
8:      Zero out gradients in optimizer
9:      Compute model prediction on masked subgraph: y_pred_masked = ModelForward(G_sub.x, G_sub.edges * M_edge, G_sub.time)
10:     
11:     Calculate explanation loss:
12:         loss = -Log(y_pred_masked[v]) + lambda_1 * Entropy(M_edge) + lambda_2 * Sum(M_edge)
13:     
14:     loss.backward()
15:     optimizer.step()
16:     Clamp M_edge to range [0, 1]
17: 
18: Filter edges: Retain edges where M_edge[j] >= theta
19: Identify key transactions, timestamps, and active feature elements
20: 
21: Generate SAR Report:
22:     SAR.TargetNode = v
23:     SAR.EvidencePath = Retained_Edges
24:     SAR.TriggerAlerts = ModelPrediction(v)
25: 
26: Return Evidence Subgraph, SAR Report
```

---

## Connection to Dissertation Objectives

*   **Objective O1, O2, O3, O4**: The pseudocodes written in this chapter represent the step-by-step algorithms used to implement each corresponding goal.
*   **Objective O5 (Experimental Setup)**: Algorithm 6 and the evaluation procedures represent the exact execution pipelines used to run model benchmarks.

---

## Summary
*   We detailed the **Graph Construction** process (ID mapping) in **Algorithm 1**.
*   We presented the **Temporal Sampling** (Algorithm 2) and **Fourier Encoding** (Algorithm 3) operations.
*   We detailed the math-to-code mapping of the **TGAT Layer Attention** (Algorithm 4) and **Head Forward Pass** (Algorithm 5).
*   We finalized with **Training and Optimization loops** (Algorithm 6) and **GNNExplainer interpretation** (Algorithm 7).

## Key Takeaways
1. Temporal sampling must precede Query-Key-Value projection to handle large neighborhood memory limits.
2. Training loss only applies to the subset of nodes matching the training mask.
3. GNNExplainer masks are bounded within $[0, 1]$ during backpropagation using sigmoid clamping.

## Practical Implementation Checklist
- [ ] Implement Index remapping dictionary.
- [ ] Write the temporal neighborhood extraction logic.
- [ ] Set up the optimizer to target both model weights and time encoder frequencies.

## Recommended Research Papers
1. Xu, D., et al. (2020). *Inductive Representation Learning on Temporal Graphs.* ICLR 2020.
2. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.

## Suggested Coding Exercises
1. Dry run Algorithm 2 (Neighborhood sampling) manually on a graph containing 5 nodes.
2. Write a PyTorch script to clamp a mask tensor between 0 and 1 after an optimizer update step.
