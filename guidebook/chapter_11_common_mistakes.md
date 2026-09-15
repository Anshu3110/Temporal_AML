# Chapter 11: Common Mistakes & Pitfalls

This chapter outlines common technical, evaluation, research, and writing mistakes in Graph Neural Networks and AML projects. Recognizing these pitfalls early will save you weeks of debugging and ensure your dissertation is methodologically sound.

---

## 11.1 Technical & Implementation Pitfalls

### 11.1.1 Temporal Data Leakage (Lookahead Bias)
*   **The Mistake**: Allowing messages to propagate from a transaction at time step $t_2$ backward to a node at time step $t_1$ where $t_1 < t_2$. This violates physical causality.
*   **The Consequence**: The model achieves artificially near-perfect test scores during training but fails completely in deployment because it relies on future information.
*   **The Correction**: Enforce strict temporal neighborhood sampling. Ensure that for any query node $i$ evaluated at time $t$, only neighbors $j$ with transaction timestamps $t_e < t$ are aggregated.

### 11.1.2 Discarding "Unknown" Nodes
*   **The Mistake**: Dropping the 77% of nodes labeled as "unknown" from the graph dataset before training to save memory.
*   **The Consequence**: The graph breaks down into thousands of tiny, disconnected components. GNNs lose their ability to capture multi-hop structural context.
*   **The Correction**: Keep the unknown nodes in the graph structure (`edge_index` and features `x`) during message passing, but mask them out of the loss calculation.

---

## 11.2 Evaluation Pitfalls

### 11.2.1 Evaluating with Standard Accuracy
*   **The Mistake**: Using overall classification accuracy as the primary validation metric under severe class imbalance (e.g. 90.3% licit).
*   **The Consequence**: A trivial model predicting "licit" for all nodes achieves 90.3% accuracy while failing to detect any money laundering.
*   **The Correction**: Evaluate using F1-score and Area Under the Precision-Recall Curve (AUPRC) on the minority class (illicit).

### 11.2.2 Random Split vs. Temporal Split
*   **The Mistake**: Using a random train-test split on dynamic transaction graphs.
*   **The Consequence**: This causes data leakage because parent nodes can end up in the test set while their subsequent child nodes are in the training set.
*   **The Correction**: Use a strict chronological split (e.g., train on steps 1–34, test on steps 35–49).

---

## 11.3 Research & Experimental Pitfalls

### 11.3.1 Over-smoothing with Deep GNNs
*   **The Mistake**: Building models with 5 or more layers to aggregate broad structural context.
*   **The Consequence**: The GNN node embeddings converge and become highly similar, degrading classification performance.
*   **The Correction**: Restrict your GNN backbone to 2 or 3 layers. This captures local neighborhood patterns while preventing over-smoothing.

### 11.3.2 Non-Learnable Time Encodings in Dynamic Graph Tasks
*   **The Mistake**: Relying solely on fixed time-delta features without learnable representations.
*   **The Consequence**: The model cannot optimize how it processes transaction velocities, reducing its ability to identify automated laundering patterns.
*   **The Correction**: Implement learnable Fourier time encodings to let the network adapt to transaction time distributions via backpropagation.

---

## 11.4 Academic Writing Pitfalls

### 11.4.1 Lack of Reproducibility Documentation
*   **The Mistake**: Failing to document seed initializations, data splits, and model configurations in your thesis.
*   **The Consequence**: Examiners cannot verify your results, and other researchers cannot build on your work, reducing the impact of your paper.
*   **The Correction**: Log all hyperparameters, document seed values, and include clear setup instructions in your repository.

### 11.4.2 Ambiguous Variable Definitions
*   **The Mistake**: Introducing mathematical equations without explicitly defining every variable and index.
*   **The Consequence**: The methodology section becomes difficult to read, which can lead to negative reviews from journal editors.
*   **The Correction**: Use a consistent mathematical notation throughout your manuscript and double-check that all variables are defined.

---

## Connection to Dissertation Objectives

This chapter provides a guide to avoiding common methodology mistakes across all objectives (**O1–O5**), helping you pass your M.Tech evaluation with high marks.

---

## Summary
*   We detailed the dangers of **temporal data leakage** and how to prevent it.
*   We highlighted the need to retain **unknown nodes** in the graph topology during message passing.
*   We discussed key evaluation guidelines, including the use of **temporal splits** and **F1-score** metrics.

## Key Takeaways
1. Never evaluate AML model performance using standard accuracy under severe class imbalance.
2. Limit your GNN to 2–3 layers to prevent over-smoothing.
3. Enforce strict chronological splits on train/test data to prevent lookahead bias.

## Practical Implementation Checklist
- [ ] Implement early stopping based on validation F1-score rather than loss.
- [ ] Add a seed initialization function to ensure training runs are reproducible.
- [ ] Verify that no edge connections in your data loaders cross temporal split boundaries.

## Recommended Research Papers
1. Rossi, E., et al. (2020). *Temporal Graph Networks for Dynamic Graphs.*
2. Oono, K., & Suzuki, T. (2020). *Graph Neural Networks Exponentially Lose Expressive Power for Node Classification.*
