# Chapter 7: Implementation Roadmap

This chapter provides a detailed, 9-week roadmap for implementing, evaluating, and writing your TemporalAML dissertation project. 

---

## 7.1 Overview Timeline

```
Week 1: Data Ingestion & EDA
   │
Week 2: Graph Construction (PyG Object)
   │
Week 3: Baseline Setup (GCN, GraphSAGE)
   │
Week 4: Standard TGAT Integration
   │
Week 5: Learnable Fourier Time Encodings (Novelty Implementation)
   │
Week 6: Pattern-Mining & Classifier Heads (Circular, Layering, Smurfing)
   │
Week 7: GNNExplainer Integration (Explainable AI Module)
   │
Week 8: Performance Evaluation & Statistical Analysis
   │
Week 9: Paper Drafting & Thesis Compilation
```

---

## 7.2 Week-by-Week Breakdown

### Week 1: Dataset and Preprocessing
*   **Milestones**:
    *   Download the Elliptic Bitcoin Dataset.
    *   Set up your project workspace, git repository, and environment configuration (YAML/requirements.txt file).
    *   Conduct Exploratory Data Analysis (EDA) on transaction node distributions, label counts, and average degree distributions.
*   **Key Deliverable**: An EDA script that prints class counts, transaction features, and splits by time step.

### Week 2: Graph Construction
*   **Milestones**:
    *   Write the ID remapping utility (converting raw `txId` values to 0-indexed integers).
    *   Construct the edge list tensor (`edge_index`) and feature tensor (`x`).
    *   Create training, validation, and testing masks. Use time steps 1–34 for training, 35–40 for validation, and 41–49 for testing.
*   **Key Deliverable**: A reusable `DatasetLoader` class returning a PyTorch Geometric `Data` object.

### Week 3: Baseline GCN & GraphSAGE
*   **Milestones**:
    *   Implement static GCN and GraphSAGE models in PyG.
    *   Set up a simple training pipeline with binary cross-entropy loss.
    *   Train baselines using early stopping on validation F1-score.
*   **Key Deliverable**: Trained baseline GCN and GraphSAGE checkpoints with evaluation metrics logged.

### Week 4: TGAT Layer Ingestion
*   **Milestones**:
    *   Implement a standard TGAT layer using standard fixed positional/time encodings.
    *   Integrate temporal neighborhood sampling (Algorithm 2) inside the data loader.
    *   Train the standard TGAT model on the node classification task.
*   **Key Deliverable**: A working TGAT conv layer and baseline temporal metrics showing improvements over GCN.

### Week 5: Learnable Fourier Time Encodings
*   **Milestones**:
    *   Implement the custom PyTorch module for learnable frequencies (Algorithm 3).
    *   Integrate the learnable Fourier encoder into the query/key/value computations of the TGAT conv layer (Algorithm 4).
    *   Verify gradient flow through the frequency parameter $\mathbf{\omega}$ during backpropagation.
*   **Key Deliverable**: Training log confirming learnable frequencies are updating and minimizing loss.

### Week 6: Pattern Detection
*   **Milestones**:
    *   Write structural search scripts to extract Circular, Layering, and Smurfing patterns from the dataset (creating multi-label targets).
    *   Add multi-head linear classifiers to the TGAT backbone embedding.
    *   Set up joint multi-task loss weighting.
*   **Key Deliverable**: A multi-head classification framework predicting all three laundering patterns.

### Week 7: Explainability
*   **Milestones**:
    *   Integrate PyG's GNNExplainer module.
    *   Set up edge mask optimization targeting flagged nodes.
    *   Format output subgraphs as clean JSON structures (SAR reports).
*   **Key Deliverable**: Script producing time-ordered subgraphs showing explanatory paths for alerts.

### Week 8: Evaluation
*   **Milestones**:
    *   Run comprehensive training passes across all baselines (GCN, GraphSAGE, LSTM-GCN, standard TGAT, rule-based heuristics, and TemporalAML).
    *   Compile classification reports: Precision, Recall, F1-Score, and AUC-ROC.
    *   Conduct ablation studies on time encoding (learnable vs. fixed vs. none).
*   **Key Deliverable**: Unified comparison table summarizing all results, ready for inclusion in the dissertation.

### Week 9: Paper Writing & Defense Prep
*   **Milestones**:
    *   Compile implementation results, graphs, and walkthrough screenshots.
    *   Write the introduction, methodology, and experimental sections of your thesis.
    *   Review the 100+ viva questions to prepare for the defense.
*   **Key Deliverable**: Complete draft of the M.Tech dissertation and Q1 journal draft.

---

## Connection to Dissertation Objectives

This chapter provides the concrete timeline to track your progress across all objectives (**O1–O5**), ensuring you hit key milestones systematically without falling behind before your dissertation defense.

---

## Summary
*   **Weeks 1–2** establish the data and graph structures (**O1**).
*   **Weeks 3–5** develop baselines and implement the learnable time encodings (**O2**).
*   **Week 6** designs multi-pattern heads (**O3**).
*   **Week 7** integrates GNNExplainer explainability (**O4**).
*   **Weeks 8–9** focus on evaluation (**O5**), report generation, and paper writing.

## Key Takeaways
1. Establishing static baselines first ensures your temporal metrics have a solid comparison point.
2. Building testing splits early prevents model tuning bias.
3. Gradual integration of learnable time frequencies makes debugging gradient issues easier.

## Practical Implementation Checklist
- [ ] Initialize your GitHub repository.
- [ ] Set up tracking software (e.g. Trello, Notion, or task.md) mapped to this roadmap.
- [ ] Configure virtual environments using Python 3.10+ and CUDA toolkit if using a GPU.

## Recommended Research Papers
1. Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks.*
2. Xu, D., et al. (2020). *Inductive Representation Learning on Temporal Graphs.*

## Suggested Coding Exercises
1. Write a script that checks if your PyTorch installation correctly accesses your local GPU or Apple Silicon MPS device.
2. Initialize a Git repository and commit your guidebook directory.
