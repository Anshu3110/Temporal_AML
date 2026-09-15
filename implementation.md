# TemporalAML — Implementation Guide & Prompt Playbook

**Temporal Graph Attention Networks with Learnable Time Encodings for Explainable Cryptocurrency Anti-Money Laundering**

> **How to use this file:** This is written so you can work through it top to bottom with an AI coding assistant (Claude Code, Cursor, Copilot Chat, etc.). Each phase has a **goal**, **tasks**, **files produced**, a **ready-to-paste prompt**, and a **validation checklist**. Do NOT skip validation checklists — this project has three failure modes (temporal leakage, dropped unknown nodes, random splits) that silently produce results that look great and are wrong.

---

## 1. Project Understanding (one paragraph)

Traditional AML systems on blockchain data either use static rules or static GNNs that ignore *when* a transaction happened relative to its neighbors, which causes temporal lookahead bias and misses time-sensitive laundering patterns (rapid layering, burst smurfing). TemporalAML builds a continuous-time transaction graph from the Elliptic Bitcoin dataset, encodes time deltas with **learnable Fourier frequencies** inside a TGAT attention layer, and trains **three multi-task heads** to detect circular transfers, layering chains, and smurfing hubs from a shared embedding. A frozen, trained model is then explained per-node with **GNNExplainer** to produce a sparse, time-ordered evidence subgraph that is compiled into an automated **SAR (Suspicious Activity Report) JSON**. The system is benchmarked against GCN, GraphSAGE, LSTM-GNN, static (non-learnable) TGAT, and a rule-based heuristic, all under strict chronological train/val/test splits and imbalance-aware metrics (F1, AUPRC).

---

## 2. Architecture at a Glance

```
Raw Elliptic CSVs
      │
      ▼
[Phase 1-2] Data Loader → ID remap → PyG Data{x, edge_index, y, time}
      │
      ▼
[Phase 3] Baselines: GCN / GraphSAGE / LSTM-GNN / Rule-based  (comparison floor)
      │
      ▼
[Phase 4] Standard TGAT (fixed sinusoidal time encoding, temporal neighbor sampling)
      │
      ▼
[Phase 5] Learnable Fourier Time Encoder  ← CORE NOVELTY (ω as nn.Parameter)
      │
      ▼
[Phase 6] Pattern Mining (circular / layering / smurfing labels) → 3 FFN heads
      │      shared TGAT embedding h_i(t) → weighted multi-task BCE loss
      ▼
[Phase 7] GNNExplainer → edge/feature masks → sparse evidence subgraph → SAR JSON
      │
      ▼
[Phase 8] Evaluation: F1 / Precision / Recall / AUC-ROC / AUPRC + ablations
      │
      ▼
[Phase 9] Demo dashboard (Streamlit/Gradio) — visualize attention + SAR live
      │
      ▼
[Phase 10] Thesis chapters + Q1 journal draft
```

---

## 3. Repository Structure

```
TemporalAML/
  ├── config/
  │     └── config.yaml
  ├── data/
  │     ├── raw/                  # Elliptic CSVs (txs_features.csv, txs_classes.csv, txs_edgelist.csv)
  │     └── processed/            # Serialized PyG Data objects (.pt)
  ├── src/
  │     ├── __init__.py
  │     ├── data_loader.py        # O1
  │     ├── models.py             # O2, O3 — Fourier encoder, TGAT, heads
  │     ├── pattern_mining.py     # O3 — circular/layering/smurfing label extraction
  │     ├── train.py              # multi-task training loop
  │     ├── evaluate.py           # O5 — metrics + baseline benchmarking
  │     ├── explain.py            # O4 — GNNExplainer + SAR generator
  │     └── baselines.py          # GCN, GraphSAGE, LSTM-GNN, rule-based
  ├── app/
  │     └── dashboard.py          # Phase 9 demo
  ├── notebooks/
  │     └── eda.ipynb
  ├── artifacts/
  │     ├── models/               # .pth checkpoints
  │     ├── logs/                 # TensorBoard
  │     └── sar_reports/          # generated SAR JSONs
  ├── tests/
  │     └── test_no_leakage.py    # asserts t_e < t_target everywhere — run this constantly
  ├── README.md
  └── requirements.txt
```

---

## 4. Environment Setup (Phase 0)

**Goal:** reproducible environment before touching data.

**Tasks**
- Create `TemporalAML/` repo with the structure above.
- Pin versions: `torch`, `torch_geometric`, `torch_scatter`/`torch_sparse` (match your CUDA/CPU build), `pandas`, `numpy`, `scikit-learn`, `networkx` (for cycle/path mining), `matplotlib`/`seaborn`, `tensorboard`, `pyyaml`, `streamlit` (for Phase 9).
- Write `config/config.yaml` with: data paths, split boundaries (34/40/49), model dims, learning rate, `λ_circ/λ_lay/λ_smurf`, `M` (max sampled temporal neighbors), device.

**Prompt**
```
Set up a Python project called TemporalAML for a temporal-GNN AML dissertation.
Create the directory structure: config/, data/raw, data/processed, src/, app/,
notebooks/, artifacts/models, artifacts/logs, artifacts/sar_reports, tests/.
Create requirements.txt pinning torch, torch_geometric, torch_scatter, torch_sparse,
pandas, numpy, scikit-learn, networkx, matplotlib, seaborn, tensorboard, pyyaml,
streamlit. Create config/config.yaml with fields for: raw_data_dir, processed_data_dir,
train_end_step (34), val_end_step (40), test_end_step (49), hidden_dim (128),
time_dim (64), num_tgat_layers (2), num_heads (4), max_temporal_neighbors (20),
learning_rate (1e-3), batch_size, lambda_circ/lambda_lay/lambda_smurf (all 1.0
initially), device (cuda if available else cpu). Add a README.md stub describing
the project in 3 sentences.
```

**Validation**
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `config.yaml` loads with `yaml.safe_load`
- [ ] Repo structure matches Section 3

---

## Phase 1 — Data Ingestion & EDA (O1, Week 1)

**Goal:** understand the raw Elliptic dataset before writing any graph code.

**Tasks**
- Download `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` into `data/raw/`.
- In `notebooks/eda.ipynb`: verify counts (203,769 nodes / 234,355 edges / 49 time steps), class distribution (licit 42,019 / illicit 4,545 / unknown 157,205), and check feature column 1 is the time step.
- Plot: nodes-per-time-step, illicit-ratio-per-time-step, degree distribution.

**Prompt**
```
I have the Elliptic Bitcoin dataset in data/raw/ (elliptic_txs_features.csv,
elliptic_txs_classes.csv, elliptic_txs_edgelist.csv). Write a Jupyter notebook
notebooks/eda.ipynb that: (1) loads all three files with pandas, (2) confirms
node count 203769, edge count 234355, 49 time steps, (3) reports class balance
for licit(2)/illicit(1)/unknown(unknown), (4) plots node count per time step,
illicit ratio per time step, and log-scale in/out degree distribution, (5) checks
for and reports any txId in the edgelist missing from the features file.
Print a short markdown summary cell at the end with these findings.
```

**Validation**
- [ ] Counts match the numbers in Section "Dataset Specifications" exactly
- [ ] No edgelist txId is missing from features (or the mismatch is documented)
- [ ] You can state, in one sentence, how illicit ratio changes over time (this matters for later split sanity)

---

## Phase 2 — Graph Construction / PyG Object (O1, Week 2)

**Goal:** turn raw CSVs into a single `torch_geometric.data.Data` object with correct dtypes and a working chronological split.

**Tasks**
- `src/data_loader.py`: `EllipticDatasetLoader` class.
- Remap raw `txId` → contiguous `[0, N-1]` indices (store the mapping — you'll need it later for SAR reports referencing original tx IDs).
- Build `data.x [N,165]` (float), `data.edge_index [2,E]` (long), `data.y [N]` (1=illicit, 0=licit, -1=unknown), `data.time [N]` (int, 1..49).
- Implement `get_split_masks()` returning boolean train/val/test masks by time step (34/40/49) — **not** random.
- Serialize to `data/processed/elliptic_graph.pt`.

**Prompt**
```
Write src/data_loader.py with a class EllipticDatasetLoader that:
1. Loads elliptic_txs_features.csv (first column txId, second column time step,
   remaining 165 are features), elliptic_txs_classes.csv (txId, class in
   {'1','2','unknown'}), elliptic_txs_edgelist.csv (txId1, txId2).
2. Builds a contiguous 0-indexed node ID mapping from raw txId, and stores the
   reverse mapping (index -> original txId) as self.id_map for later SAR use.
3. Constructs a torch_geometric.data.Data object with:
   - data.x: FloatTensor [N, 165]
   - data.edge_index: LongTensor [2, E] directed, using the remapped indices
   - data.y: LongTensor [N] with 1=illicit, 0=licit, -1=unknown
   - data.time: LongTensor [N] holding the time step 1..49
4. A method get_split_masks(train_end=34, val_end=40) returning boolean tensors
   train_mask/val_mask/test_mask based on data.time, NOT random splitting.
5. A method save(path) / load(path) using torch.save/torch.load to serialize to
   data/processed/elliptic_graph.pt.
Include a __main__ block that builds the graph, prints N, E, class counts, and
split sizes, and saves it. Add type hints and docstrings.
```

**Validation**
- [ ] `data.x.shape == (203769, 165)`
- [ ] `data.y` has exactly the three values `{-1, 0, 1}` matching the counts from Phase 1
- [ ] `train_mask`, `val_mask`, `test_mask` are disjoint and time-ordered (max time in train < min time in val < min time in test)
- [ ] Unknown nodes (`y == -1`) are present in `edge_index`/`x` but excluded only at loss time (write this down now — you'll enforce it in Phase 4/6)

---

## Phase 3 — Baseline Models (O5, Week 3)

**Goal:** establish the comparison floor *before* building the novel model, so you have honest numbers to beat.

**Tasks**
- `src/baselines.py`: static `GCN`, `GraphSAGE` (both from `torch_geometric.nn`), a simple `LSTM-GNN` (per-time-step GCN embeddings fed into an LSTM), and a `RuleBasedHeuristic` (flag if volume > 50 BTC or inter-transaction velocity < 1 min, using available feature columns as proxies).
- Train GCN/GraphSAGE with early stopping on validation F1 (illicit class), using the same chronological masks from Phase 2.
- Log all metrics to `artifacts/logs/`.

**Prompt**
```
Write src/baselines.py implementing:
1. A 2-layer GCN (torch_geometric.nn.GCNConv) and a 2-layer GraphSAGE
   (torch_geometric.nn.SAGEConv) binary classifier over the Elliptic graph
   loaded via src/data_loader.py, trained only on nodes where y != -1.
2. An LSTMGNN class: run a shared GCNConv per time step to get per-node
   embeddings, stack embeddings for each node across its available time
   steps, and pass through an nn.LSTM before a final linear classifier.
3. A RuleBasedHeuristic class (no training) that flags a node illicit if any
   feature proxy for transaction volume exceeds a configurable threshold or a
   proxy for transaction velocity is below a threshold; make thresholds
   config-driven.
4. A train_baseline(model, data, masks, config) function with early stopping
   on validation F1 for the illicit class (use sklearn.metrics.f1_score),
   weighted BCE/CrossEntropy loss to handle class imbalance, and TensorBoard
   logging to artifacts/logs/<model_name>/.
5. A __main__ that trains all baselines and prints a comparison table of
   F1/Precision/Recall/AUC-ROC/AUPRC on the test split.
```

**Validation**
- [ ] All models trained/evaluated on **identical** chronological masks
- [ ] Loss/metrics computed only over `y != -1` nodes
- [ ] You have a baseline comparison table saved (you'll extend this table in Phase 8)

---

## Phase 4 — Standard TGAT (Week 4)

**Goal:** get temporal attention with *fixed* sinusoidal time encoding working correctly, with strict no-lookahead sampling — before adding the learnable novelty on top.

**Tasks**
- `src/models.py`: `FixedTimeEncoder` (standard Time2Vec-style, non-trainable frequencies), `TGATLayer` implementing the Q/K/V attention described in your brief, and `TemporalNeighborSampler` that samples the `M` most recent edges with `t_e < t_target` per node.
- **This is the phase to get temporal-leakage-proof.** Write `tests/test_no_leakage.py` now and keep running it.

**Prompt**
```
In src/models.py implement:
1. FixedTimeEncoder(nn.Module): non-trainable sinusoidal time encoding
   Phi(delta_t) using fixed geometrically-spaced frequencies, output dim
   time_dim, NOT registered as nn.Parameter (use register_buffer).
2. TemporalNeighborSampler: given edge_index, edge times (source node's time
   step), and a target (node_id, time_t), return up to M neighbor edges
   where edge_time < time_t, most-recent-first. Must raise/assert if any
   returned edge has edge_time >= time_t.
3. TGATLayer(nn.Module): multi-head attention where for target node i at
   time t and each sampled neighbor j at time t_e:
   - query = concat(h_i, time_encoder(0)) @ W_Q
   - key   = concat(h_j, time_encoder(t - t_e)) @ W_K
   - value = concat(h_j, time_encoder(t - t_e)) @ W_V
   - attention = softmax(query @ key^T / sqrt(d_k))
   - output = attention @ value, then a residual + linear projection back to
     hidden_dim.
   Accept a `time_encoder` module as a constructor argument so it can be
   swapped for the learnable version later without changing this class.
4. A TGATEncoder(nn.Module) stacking 2 TGATLayers (config-driven) that
   returns per-node embeddings h_i(t) for a batch of target (node, time) pairs.

Also write tests/test_no_leakage.py using pytest that constructs a small
synthetic temporal graph with known future edges and asserts
TemporalNeighborSampler never returns an edge with t_e >= t_target, and that
TGATEncoder's output for a node at time t is unchanged if you add new edges
with time > t (proving no lookahead).
```

**Validation**
- [ ] `pytest tests/test_no_leakage.py` passes
- [ ] TGATEncoder trains and converges on a small subgraph sanity check (loss decreases)
- [ ] `time_encoder` is a swappable argument (needed for Phase 5)

---

## Phase 5 — Learnable Fourier Time Encoding (O2, Week 5 — CORE NOVELTY)

**Goal:** implement the trainable-frequency time encoder and verify gradients actually reach `ω`.

**Tasks**
- `src/models.py`: `LearnableFourierTimeEncoder` with `ω ∈ R^{d_t/2}` as `nn.Parameter`.
- Swap it into `TGATEncoder` in place of `FixedTimeEncoder`.
- Write a gradient-flow unit test: after one backward pass, `ω.grad` must be non-zero and non-NaN.

**Prompt**
```
In src/models.py add LearnableFourierTimeEncoder(nn.Module):
- self.omega = nn.Parameter(torch.randn(time_dim // 2) * initial_scale)
  (initial_scale configurable, default small e.g. 0.1)
- forward(delta_t): returns sqrt(2/time_dim) * concat(
    cos(omega * delta_t), sin(omega * delta_t)) interleaved or concatenated
    per the formula in the spec, shape [..., time_dim].
- Must support batched delta_t tensors of arbitrary shape.

Then update TGATEncoder's constructor to accept a time_encoder_type flag
('fixed' | 'learnable') and instantiate the right class.

Write tests/test_gradient_flow.py: build a tiny TGATEncoder with the
learnable encoder, run a forward+backward pass on a dummy binary
classification loss, and assert time_encoder.omega.grad is not None,
not all zero, and contains no NaN/Inf values. Also assert the omega values
actually change after an optimizer.step().
```

**Validation**
- [ ] `omega.grad` is non-zero after backward()
- [ ] `omega` values differ before/after a few optimizer steps
- [ ] Swapping `fixed` → `learnable` doesn't change any other code (clean interface from Phase 4)

---

## Phase 6 — Pattern Mining & Multi-Task Heads (O3, Week 6)

**Goal:** generate the three structural labels and train the joint model.

**Tasks**
- `src/pattern_mining.py`: using `networkx` (build a `DiGraph` from `edge_index`+`time`), implement:
  - `find_circular(graph, max_len=5)` — cycle detection (Johnson's algorithm / `nx.simple_cycles` with length cap for tractability at this scale — expect to need pruning/sampling given 200k+ nodes).
  - `find_layering(graph, min_len=3)` — sequential peeling-chain detection (DFS along strictly increasing timestamps).
  - `find_smurfing(graph, degree_threshold=10, window)` — nodes with in/out-degree ≥ threshold inside a short time window.
- Attach three FFN heads on top of the shared `TGATEncoder` embedding; implement weighted multi-task BCE loss (`λ_circ, λ_lay, λ_smurf` from config).
- `src/train.py`: full training loop, checkpointing to `artifacts/models/`.

**Prompt**
```
Write src/pattern_mining.py with three functions operating on a networkx
DiGraph built from the Elliptic edge_index and per-node time steps:
1. find_circular_labels(G, max_cycle_len=5) -> dict[node_id, bool]: True if
   node participates in a directed cycle of length <= max_cycle_len. Because
   full simple_cycles is intractable at 200k+ nodes, restrict search to
   cycles within a bounded ego-network (e.g. BFS radius 3) per node, or run
   nx.simple_cycles per weakly-connected component if components are small
   enough; document the approximation clearly in a docstring.
2. find_layering_labels(G, min_path_len=3) -> dict[node_id, bool]: True if
   node lies on a directed path of length >= min_path_len where each edge's
   target time step is strictly greater than the source's (a "peeling
   chain").
3. find_smurfing_labels(G, degree_threshold=10, time_window=2) -> dict[node_id,
   bool]: True if node's in-degree or out-degree within any `time_window`
   consecutive time steps is >= degree_threshold.
Return all three as aligned boolean arrays over the 0-indexed node ordering
so they can be stacked into a [N, 3] multi-label target tensor. Add a
__main__ that computes and saves these to data/processed/pattern_labels.pt.

Then in src/models.py add MultiTaskHead(nn.Module): three independent
2-layer FFN heads (circular, layering, smurfing) each taking the shared
TGATEncoder embedding and outputting a sigmoid probability.

Then write src/train.py: a training loop that for each labeled node
(y != -1) at its labeled time step, samples temporal neighbors, runs
TGATEncoder + MultiTaskHead, computes weighted BCE per head (pos_weight
from config, one per head), sums with lambda_circ/lambda_lay/lambda_smurf,
backprops, and logs per-head F1 on validation each epoch to TensorBoard.
Checkpoint the best model (by average validation F1 across heads) to
artifacts/models/temporalaml_best.pth.
```

**Validation**
- [ ] Pattern label counts look plausible (not 0%, not 100% — sanity check against known illicit ratio)
- [ ] Training loss for all three heads decreases
- [ ] Validation F1 tracked per head, not just averaged (you need per-pattern numbers for the thesis)
- [ ] Checkpointing works and best model reloads correctly

---

## Phase 7 — Explainability & SAR Generation (O4, Week 7)

**Goal:** for any flagged node, produce a human-readable, compliance-style evidence report.

**Tasks**
- `src/explain.py`: load frozen best checkpoint, run `torch_geometric.explain.Explainer` with a `GNNExplainer` algorithm to get edge/node feature masks for a target node.
- Threshold masks at `θ ≥ 0.5`, extract the sparse time-ordered subgraph.
- Compile a SAR JSON: target original `txId` (via the reverse ID map from Phase 2), triggering feature names/values, the evidence subgraph (node IDs, timestamps, edge list), and which pattern head(s) fired.

**Prompt**
```
Write src/explain.py that:
1. Loads the best checkpoint from artifacts/models/temporalaml_best.pth and
   the id_map saved by EllipticDatasetLoader.
2. For a given target node index and time step, uses
   torch_geometric.explain.Explainer with algorithm=GNNExplainer to compute
   edge_mask and node_feat_mask explaining the model's prediction for that
   node (do this per prediction head, or for the max-probability head).
3. Thresholds edge_mask at 0.5, extracts the induced sparse subgraph, and
   sorts its edges by timestamp to produce a time-ordered evidence path.
4. Maps the top contributing feature indices (from node_feat_mask) back to
   human-readable names if a feature name list is available, else labels
   them "feature_<idx>".
5. Builds and saves a SAR JSON to artifacts/sar_reports/sar_<original_txid>.json
   containing: target_tx_id, flagged_patterns (list of which heads exceeded
   a probability threshold), triggering_features (name + importance score,
   top 10), evidence_subgraph (list of {source_tx_id, target_tx_id, time_step}
   sorted chronologically), and a one-line plain-English summary string.
6. Add a __main__ / CLI (argparse) that takes a node index or original txId
   and time step and runs the full pipeline end to end.
```

**Validation**
- [ ] SAR JSON is valid JSON and opens correctly
- [ ] Evidence subgraph edges are in chronological order and all satisfy `t_e < t_target`
- [ ] Run this on 3–5 known illicit nodes from the test set and eyeball whether the evidence subgraph looks plausible (not just noise)

---

## Phase 8 — Evaluation & Ablation Benchmarking (O5, Week 8)

**Goal:** produce the results table and ablation study that anchors your thesis findings.

**Tasks**
- `src/evaluate.py`: unify metric computation (F1, Precision, Recall, AUC-ROC, AUPRC) on the illicit class across: GCN, GraphSAGE, LSTM-GNN, Rule-based, Static-TGAT, TemporalAML (learnable).
- **Ablation:** Learnable Fourier vs Fixed sinusoidal vs No time encoding (mean-pool neighbors, ignore time entirely) — same architecture otherwise.
- Generate comparison tables + bar/line charts for the thesis.

**Prompt**
```
Write src/evaluate.py that:
1. Loads test-split predictions from every trained model (GCN, GraphSAGE,
   LSTM-GNN, RuleBasedHeuristic, Static-TGAT, TemporalAML) plus an ablation
   variant of TemporalAML with time encoding disabled (mean-pooled neighbor
   aggregation, no time term).
2. Computes F1, Precision, Recall, AUC-ROC, AUPRC for the illicit class for
   every model on the identical chronological test split (time steps 41-49).
3. Also computes per-pattern-head F1 (circular/layering/smurfing) for
   TemporalAML specifically.
4. Outputs a markdown table and a CSV to artifacts/logs/final_results.md /
   .csv, plus a matplotlib bar chart comparing AUPRC across all models saved
   to artifacts/logs/auprc_comparison.png.
5. Runs a paired significance check (e.g., bootstrap resampling of the test
   set, 1000 iterations) comparing TemporalAML's F1 against Static-TGAT's F1
   and reports a 95% confidence interval on the F1 delta.
```

**Validation**
- [ ] Every model evaluated on the *exact same* test node set
- [ ] Ablation clearly isolates the effect of the learnable encoding (this is your headline result — protect its rigor)
- [ ] Confidence interval on the improvement doesn't include 0 (or you honestly report that it does)

---

## Phase 9 — Demo Dashboard (for showcasing, not in original roadmap but strongly recommended)

**Goal:** a live, clickable artifact you can show in a viva, interview, or portfolio — this is what makes the project *memorable* beyond a PDF.

**Tasks**
- `app/dashboard.py` (Streamlit): pick a transaction node from the test set → show its predicted pattern probabilities → render the GNNExplainer evidence subgraph as an interactive network plot → display the generated SAR JSON side by side.
- Include a model comparison tab (bar chart from Phase 8's CSV).

**Prompt**
```
Write app/dashboard.py as a Streamlit app that:
1. Loads the trained TemporalAML model, id_map, and test-split node list.
2. Sidebar: a searchable dropdown of test-set transaction IDs (original
   txId), defaulting to a curated list of ~20 interesting flagged nodes.
3. Main panel, tab 1 "Investigate": shows the three pattern-head probabilities
   as gauges/bars, runs src/explain.py's pipeline live, renders the evidence
   subgraph with a network plot (networkx + matplotlib or plotly, nodes
   colored by time step, edges labeled with time delta), and shows the SAR
   JSON in a formatted, collapsible viewer.
4. Tab 2 "Benchmarks": loads artifacts/logs/final_results.csv and renders a
   bar chart comparing F1/AUPRC across GCN/GraphSAGE/LSTM-GNN/Rule-based/
   Static-TGAT/TemporalAML, plus the ablation chart.
5. Tab 3 "About": a short architecture diagram (static image or mermaid-like
   text) and the project's one-paragraph summary.
Keep it runnable with `streamlit run app/dashboard.py` and cache model
loading with @st.cache_resource.
```

**Validation**
- [ ] Runs locally with `streamlit run app/dashboard.py`
- [ ] Investigate tab works end-to-end for at least 5 different nodes without errors
- [ ] Benchmarks tab correctly reflects Phase 8's numbers

---

## Phase 10 — Thesis Writing & Journal Draft (Week 9)

**Goal:** turn the working system into the dissertation document + a submittable paper.

**Tasks**
- Chapters: Introduction/Motivation → Related Work (static GNNs, TGAT/TGN, AML literature) → Methodology (O1–O4 as written in your spec, verbatim math is fine) → Experimental Setup (dataset, splits, baselines) → Results & Ablation (Phase 8 tables/figures) → Explainability Case Studies (2–3 SAR examples from Phase 9) → Limitations & Future Work → Conclusion.
- Journal draft: trim to a focused paper — the learnable Fourier encoding ablation + multi-task pattern heads + SAR generation is your contribution triplet; lead with that.

**Prompt**
```
I have finished implementing TemporalAML (learnable Fourier time-encoded
TGAT with multi-task circular/layering/smurfing heads and GNNExplainer-based
SAR generation on the Elliptic Bitcoin dataset). I have: [paste Phase 8
results table], [paste 2-3 example SAR JSONs from Phase 9], and the
architecture description from this implementation.md. Help me draft a
Q1-journal-style paper (IEEE/Elsevier format) with sections: Abstract,
Introduction, Related Work, Methodology, Experiments, Results & Ablation,
Explainability Case Study, Limitations, Conclusion. Keep the Methodology
section faithful to the math I've provided. Target ~6000-8000 words.
```

**Validation**
- [ ] Every claimed number in the thesis traces back to a saved artifact (CSV/JSON) from Phase 8, not memory
- [ ] Ablation study is presented as the central result, not buried
- [ ] Limitations section honestly states the pattern-mining approximations from Phase 6

---

## 5. How to Showcase This Project

| Audience | What to show | Artifact |
|---|---|---|
| **Thesis committee / viva** | Live dashboard demo on 2-3 real flagged nodes + ablation table | Phase 9 dashboard + Phase 8 results |
| **GitHub / portfolio** | Clean README with architecture diagram, results table, one GIF of the dashboard | README.md + `docs/architecture.png` + a 15-30s screen recording |
| **Recruiters / interviews** | 60-second pitch: *"static GNNs leak future information into AML predictions; I built a temporal attention model with learnable time encodings that beats static TGAT by X% F1, and it auto-generates compliance-ready SARs"* | Same dashboard, one slide |
| **Journal submission** | Phase 10 draft, focused on the learnable-encoding ablation as the core contribution | Paper draft |
| **LinkedIn / portfolio site** | Short post: problem → novelty → one SAR example screenshot → results chart | Phase 9 screenshots |

**Concrete showcase checklist:**
- [ ] README.md with: 1-paragraph pitch, architecture diagram (reuse Section 2 of this file), results table, "how to run" instructions, dashboard screenshot/GIF
- [ ] A 2–3 minute recorded demo walking through: pick a flagged node → show attention subgraph → show SAR JSON → show it beats baselines
- [ ] A single slide with the Section 2 diagram + headline ablation number (learnable vs fixed vs no time encoding)
- [ ] Public repo cleaned of raw data (Elliptic dataset requires its own license/usage terms — check before pushing `data/raw/` publicly; typically you `.gitignore` it and document how to download it)

---

## 6. Evaluation Rubric / Success Criteria

- [ ] `tests/test_no_leakage.py` and `tests/test_gradient_flow.py` both pass
- [ ] TemporalAML outperforms Static-TGAT on illicit-class F1/AUPRC with a confidence interval that excludes 0 (or this is honestly reported as not statistically significant)
- [ ] All three pattern heads have non-trivial per-head F1 (not collapsed to predicting all-negative)
- [ ] At least 3 SAR JSONs generated and manually sanity-checked against the raw transaction data
- [ ] Full results table with 6 models (5 baselines + TemporalAML) × 5 metrics
- [ ] Dashboard runs end-to-end without manual intervention

---

## 7. Pitfalls Register (do not violate these — repeated from spec, kept here as a running checklist)

| # | Pitfall | Where it bites | Guard |
|---|---|---|---|
| 1 | Temporal lookahead (`t_e ≥ t_target`) | Phase 4 sampler | `tests/test_no_leakage.py` on every model change |
| 2 | Dropping the 77% unknown nodes | Phase 2/6 | Keep in `x`/`edge_index`; mask only in loss (`y == -1`) |
| 3 | Reporting standard accuracy | Phase 8 | Report F1/Recall/Precision/AUPRC only |
| 4 | Over-smoothing | Phase 4/5 | Cap GNN depth at 2–3 layers |
| 5 | Random train/test split | Phase 2 | Chronological only: 1–34 / 35–40 / 41–49 |

---

## 8. Suggested Week-to-Phase Mapping

| Week | Phase(s) |
|---|---|
| 1 | Phase 0 + Phase 1 |
| 2 | Phase 2 |
| 3 | Phase 3 |
| 4 | Phase 4 |
| 5 | Phase 5 |
| 6 | Phase 6 |
| 7 | Phase 7 |
| 8 | Phase 8 + start Phase 9 |
| 9 | Phase 9 (finish) + Phase 10 |

---

*Work through this file phase by phase — copy each phase's prompt block into your AI coding assistant, run the validation checklist before moving on, and don't proceed to the next phase until `tests/` pass for the current one.*
