# Chapter 9: Research & Viva Questions

This chapter compiles 100 questions and detailed answers designed to prepare you for your dissertation defense (viva) and presentations. The questions are structured into six key subject areas.

---

## Section 1: Money Laundering & Crypto AML Basics (Q1 - Q20)

#### Q1: What is the primary objective of Anti-Money Laundering (AML) in financial systems?
**Answer**: The primary objective of AML is to detect and prevent the process of disguising the origin of illegally obtained funds so that they appear to have been derived from legitimate sources.

#### Q2: Explain the three classical stages of money laundering.
**Answer**:
1. **Placement**: Introducing illicit cash/assets into the financial system.
2. **Layering**: Distributing funds through complex transactions to hide source ownership.
3. **Integration**: Re-entering the laundered funds into the economy as legitimate wealth.

#### Q3: How does placement occur in cryptocurrency?
**Answer**: By purchasing coins with cash at unregulated exchanges, converting cybercrime profits (e.g. ransomware) directly to crypto, or using cash at P2P trading desks.

#### Q4: What is a peeling chain in cryptocurrency?
**Answer**: A layering technique where a wallet sends a small amount of currency to a recipient address and routes the remaining balance to a newly generated change address, repeating this process sequentially.

#### Q5: What is Circular Transfer (Looping)?
**Answer**: Transferring funds through multiple wallets such that the transactions form a directed loop, eventually returning the funds to the sender.

#### Q6: Define Smurfing (Structuring) in transaction terms.
**Answer**: Breaking a large sum of money down into small quantities below reporting thresholds, distributing them across multiple wallets (fan-out), and consolidating them later at a cash-out point (fan-in).

#### Q7: Why do traditional rule-based AML systems fail in cryptocurrency?
**Answer**: Because rule-based systems rely on static thresholds and evaluate transactions in isolation, failing to capture the global network topology and time-aware sequence patterns.

#### Q8: What does pseudonymity mean in Bitcoin and how do criminals exploit it?
**Answer**: Blockchain addresses are public, but their owner's real-world identity is not. Criminals exploit this by generating unlimited temporary addresses to hide transaction origins.

#### Q9: What is a cryptocurrency mixer (tumbler)?
**Answer**: A service that pools funds from multiple users, mixes them, and distributes them back to different addresses to sever the transaction history.

#### Q10: How do cross-chain bridges complicate AML analysis?
**Answer**: They allow users to swap assets from one blockchain to another (e.g., Bitcoin to Ethereum), which breaks traditional single-chain transaction tracking.

#### Q11: What is the role of an exchange's KYC process in AML?
**Answer**: Know Your Customer (KYC) requires users to verify their identities before trading, linking physical identities to pseudonymous wallet addresses.

#### Q12: What is an OTC desk, and why is it a risk area?
**Answer**: Over-the-Counter (OTC) desks facilitate large trades outside public order books, often with weaker KYC compliance.

#### Q13: What does the term "chain hopping" refer to?
**Answer**: Swapping rapidly between different cryptocurrencies (e.g., Bitcoin to Monero to Tether) to bypass transaction tracing networks.

#### Q14: Explain the difference between address-centric and transaction-centric graphs.
**Answer**: In an address-centric graph, nodes are wallets and edges are transfers. In a transaction-centric graph, nodes are transaction events and edges represent the spending of transaction outputs.

#### Q15: Why is the Elliptic Bitcoin Dataset modeled as a transaction-centric graph?
**Answer**: Because Bitcoin uses the UTXO (Unspent Transaction Output) model, where transactions themselves are discrete nodes and edges track the direct flow of value from input to output.

#### Q16: What is a Suspicious Activity Report (SAR)?
**Answer**: A compliance document filed by financial institutions to alert regulatory bodies about suspected money laundering or terrorist financing.

#### Q17: What are the main limitations of heuristics-based transaction monitoring?
**Answer**: High false-positive rates, inability to detect novel laundering strategies, and susceptibility to evasion via slight parameter adjustments.

#### Q18: How does the velocity of transactions help identify automated laundering?
**Answer**: Automated scripts execute transfers in seconds or minutes, while normal human economic activity usually operates on scale of hours, days, or weeks.

#### Q19: What is "Structuring" under AML regulations?
**Answer**: The practice of deliberately split-funding transactions to remain under reporting limits.

#### Q20: How does decentralization affect regulatory AML enforcement?
**Answer**: Decentralized protocols lack a central administrator to block transactions, freeze accounts, or perform KYC verification.

---

## Section 2: Graph Neural Networks & Message Passing (Q21 - Q40)

#### Q21: What is a Graph Neural Network (GNN)?
**Answer**: A class of deep learning models designed to run directly on graph structured data, capturing topological relationships and node/edge features.

#### Q22: Explain the Message Passing scheme in GNNs.
**Answer**: A mechanism where nodes send feature information to their neighbors, which aggregate these messages and update their own hidden states.

#### Q23: Why is permutation invariance required in GNN aggregation?
**Answer**: Graphs have no natural ordering of neighbors. The aggregation function must produce the same output regardless of the order in which neighbor features are processed (e.g., Sum, Mean, Max).

#### Q24: Write the mathematical expression for a simple GCN layer.
**Answer**:
$$\mathbf{h}_i^{(l+1)} = \sigma \left( \mathbf{W} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \mathbf{h}_j^{(l)} \right)$$

#### Q25: What is the purpose of adding self-loops in GCN models?
**Answer**: To ensure that a node's own features are included when aggregating its neighbors' features.

#### Q26: Explain the "over-smoothing" problem in GNNs.
**Answer**: If a GNN has too many layers, message passing causes all node embeddings to converge and become highly similar, reducing classification performance.

#### Q27: How does GraphSAGE differ from classic GCN?
**Answer**: GraphSAGE uses neighborhood sampling (aggregating from a subset of neighbors) rather than full-graph aggregation, enabling batching on large graphs.

#### Q28: What are inductive GNN models?
**Answer**: Models that can generate embeddings for unseen nodes or subgraphs without requiring retraining on the entire graph structure.

#### Q29: What is an edge index in PyTorch Geometric (PyG)?
**Answer**: A tensor of shape `[2, E]` representing the directed connections in a graph, containing source and target node indices.

#### Q30: How do you handle edge features in PyG?
**Answer**: By passing them as a parameter `edge_attr` alongside the node features `x` and connections `edge_index` in custom convolutional layers.

#### Q31: What is the spectral domain in Graph Theory?
**Answer**: Analyzing graphs using the eigenvalues and eigenvectors of the graph Laplacian matrix.

#### Q32: What is the spatial domain in GNNs?
**Answer**: Defining convolutions directly on local neighborhoods of nodes (e.g. aggregating features from immediate neighbors).

#### Q33: Why do spatial GNNs scale better than spectral GNNs?
**Answer**: Because spectral methods require expensive matrix decompositions of the entire graph Laplacian, whereas spatial methods operate locally and support batching.

#### Q34: What is the function of the graph Laplacian matrix?
**Answer**:
$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$
It measures how features vary locally across the graph's connections.

#### Q35: How does node classification differ from link prediction?
**Answer**: Node classification predicts a label for a node (e.g., Licit/Illicit), while link prediction evaluates the probability of an edge existing between two nodes.

#### Q36: Explain the difference between transductive and inductive learning on graphs.
**Answer**: Transductive learning requires all nodes (including test nodes) to be present during training. Inductive learning generalizes to new, unseen graphs or nodes.

#### Q37: What is the utility of a read-out phase in GNNs?
**Answer**: It pools individual node embeddings into a single representation to perform graph-level classification.

#### Q38: Why do standard GNNs struggle with large, highly dynamic graphs?
**Answer**: They rely on static adjacency matrices and cannot capture changes in graph topology or timestamps.

#### Q39: What is the receptive field of a GNN node?
**Answer**: The set of nodes reachable within $k$ message-passing steps, corresponding to a $k$-hop neighborhood.

#### Q40: How does dropout work on graph data?
**Answer**: It randomly zeroes out node features or drops edges during training to prevent overfitting.

---

## Section 3: Attention Mechanisms & GAT (Q41 - Q55)

#### Q41: Explain the core concept of Graph Attention Networks (GAT).
**Answer**: GAT computes learnable attention weights for each neighbor, allowing nodes to prioritize the most relevant connections rather than relying on static structural properties.

#### Q42: Write the formula for calculating attention coefficients in GAT.
**Answer**:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_k]\right)\right)}$$

#### Q43: What is the purpose of using Multi-Head Attention in GAT?
**Answer**: It stabilizes learning by computing multiple independent attention weights, allowing the model to capture diverse features across different representation subspaces.

#### Q44: How is Multi-Head Attention output combined in intermediate GAT layers?
**Answer**: By concatenating the outputs of each head.

#### Q45: How is Multi-Head Attention output combined in the final classification layer?
**Answer**: By averaging the outputs of each head to keep the final output dimensions correct.

#### Q46: What is the role of the LeakyReLU non-linearity in GAT attention?
**Answer**: It acts as a non-linear activation function on the concatenated node features, enabling the model to learn complex attention relationships.

#### Q47: Compare GAT attention weights with GCN normalization.
**Answer**: GCN normalizes messages based on fixed node degrees ($\frac{1}{\sqrt{d_i d_j}}$), whereas GAT uses dynamic, learnable attention weights ($\alpha_{ij}$) based on node features.

#### Q48: Does GAT scale to large graphs?
**Answer**: Yes, because GAT attention is computed locally between nodes and their immediate neighbors, which can be parallelized across edges.

#### Q49: What is self-attention in the context of graphs?
**Answer**: The query, key, and value vectors are all projected from the same set of input node embeddings.

#### Q50: How do edge features influence GAT attention?
**Answer**: Edge features can be concatenated with the source and target node features to compute the attention weights:
$$\alpha_{ij} = \text{softmax}_j \left( \mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_j \parallel \mathbf{W}_e \mathbf{e}_{ij}] \right)$$

#### Q51: What is the role of the softmax function in GAT?
**Answer**: It normalizes the raw attention scores across a node's neighbors so they sum to 1, creating a probability distribution.

#### Q52: How do you prevent attention scores from exploding?
**Answer**: By dividing the query-key dot product by the square root of the key dimension ($\sqrt{d_k}$) before applying the softmax function.

#### Q53: Can GAT model continuous temporal patterns?
**Answer**: No, standard GAT only calculates attention based on spatial structures and static features.

#### Q54: What happens if a node has no neighbors in GAT?
**Answer**: If self-loops are added, the node attends only to itself, preventing division-by-zero errors in the softmax denominator.

#### Q55: Why is masking applied in GAT attention?
**Answer**: To restrict the softmax calculation to actual structural neighbors, setting attention weights for non-neighbors to negative infinity.

---

## Section 4: Temporal GNNs & Learnable Time Encodings (Q56 - Q75)

#### Q56: What is a Temporal Graph Attention Network (TGAT)?
**Answer**: A temporal GNN architecture that integrates continuous time differences into GAT's attention mechanism using Fourier time encodings.

#### Q57: Write the equation for the Fourier Time Encoding used in TGAT.
**Answer**:
$$\Phi_d(\Delta t) = \sqrt{\frac{2}{d}} \left[ \cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots, \cos(\omega_{d/2} \Delta t), \sin(\omega_{d/2} \Delta t) \right]^{\top}$$

#### Q58: Why are both sine and cosine functions used in Fourier encoding?
**Answer**: Combining sines and cosines allows the model to map time differences linearly while capturing both periodic trends and translation invariance.

#### Q59: What is the difference between fixed and learnable Fourier time encodings?
**Answer**: Fixed encodings use static, predefined frequency bands. Learnable encodings treat frequencies ($\omega_k$) as parameters updated during training.

#### Q60: Explain how learnable frequencies are updated.
**Answer**: They are registered as parameters in the PyTorch graph, allowing gradients to flow back from the loss function and update the frequencies using the optimizer.

#### Q61: What is the gradient of a cosine component with respect to frequency $\omega_k$?
**Answer**:
$$\frac{\partial \cos(\omega_k \Delta t)}{\partial \omega_k} = -\Delta t \cdot \sin(\omega_k \Delta t)$$

#### Q62: Why is temporal neighborhood sampling required in TGAT?
**Answer**: To limit memory usage and enforce temporal causality by only aggregating from neighbors that interacted before the target node's time step.

#### Q63: Define the time-aware Query, Key, and Value vectors in TGAT.
**Answer**:
$$\mathbf{q}_i = [\mathbf{h}_i(t) \parallel \Phi_d(0)] \mathbf{W}_Q$$
$$\mathbf{k}_j = [\mathbf{h}_j(t_e) \parallel \Phi_d(t - t_e)] \mathbf{W}_K$$
$$\mathbf{v}_j = [\mathbf{h}_j(t_e) \parallel \Phi_d(t - t_e)] \mathbf{W}_V$$

#### Q64: What is time translation invariance in temporal representations?
**Answer**: The representation of a time interval $\Delta t$ should depend only on the elapsed time, not on the absolute start time of the transactions:
$$\Phi(t_1 - t_2) = \Phi((t_1 + \tau) - (t_2 + \tau))$$

#### Q65: How does TGAT prevent temporal information leakage (data leakage)?
**Answer**: By enforcing that queries at time $t$ only aggregate keys and values from neighbor events that occurred at time $t_e < t$.

#### Q66: What is a dynamic graph representation?
**Answer**: A representation where nodes, edges, or features change over time, captured through a sequence of graph snapshots or discrete event streams.

#### Q67: Explain the difference between discrete-time and continuous-time dynamic graphs.
**Answer**: Discrete-time graphs are represented as a sequence of static snapshots ($G_1, G_2, \dots$). Continuous-time graphs record interactions as a stream of events with precise, real-valued timestamps.

#### Q68: What is the role of time-aware self-attention?
**Answer**: It allows the model to weigh messages from neighbors based on both their structural features and how recently the interaction occurred.

#### Q69: How is the dimension $d_t$ of the time encoding vector selected?
**Answer**: Usually set to match the node feature dimension or model hidden dimension, balancing temporal resolution with computational complexity.

#### Q70: Does TGAT require a node memory state like TGN?
**Answer**: No, TGAT does not maintain an active memory state. It computes representations on-the-fly using temporal neighborhood aggregation, which avoids issues with memory staleness.

#### Q71: What is the memory staleness problem in dynamic graphs?
**Answer**: In memory-based models (like TGN), a node's memory is only updated when it participates in an event. If a node is inactive for a long time, its memory becomes stale and may not reflect the current state of the network.

#### Q72: How does TGAT handle new nodes during inference?
**Answer**: It aggregates features from the node's temporal neighborhood on-the-fly, allowing it to generalize to unseen nodes without retraining.

#### Q73: Why is continuous time encoding useful for AML peeling chains?
**Answer**: Peeling chains transfer funds through hops in quick succession. Continuous encoding captures these short time differences, allowing the model to distinguish them from slower, normal transactions.

#### Q74: Can learnable Fourier time encoding capture periodic patterns?
**Answer**: Yes, by adjusting the frequencies ($\omega_k$), the model can learn to recognize cyclical patterns in transaction volume or timing.

#### Q75: How do you initialize the learnable frequency parameters?
**Answer**: Typically using an exponential grid distribution to span a wide range of time scales:
$$\omega_k = \frac{1}{10000^{2k/d}}$$

---

## Section 5: GNNExplainer & Explainability (Q76 - Q85)

#### Q76: Why is explainability critical in AML applications?
**Answer**: Financial compliance teams must verify alerts and provide evidence subgraphs to explain why a transaction was flagged as suspicious before filing a Suspicious Activity Report (SAR).

#### Q77: What is the core mechanism of GNNExplainer?
**Answer**: It learns continuous masks over edge connections and node features to identify the most important subgraphs and attributes that drive the model's prediction.

#### Q78: Write the optimization objective of GNNExplainer.
**Answer**:
$$\min_{\mathbf{M}, \mathbf{F}} -\sum_{c=1}^C y_c \log P(Y = c \mid G_s(\mathbf{M}), X_s(\mathbf{F})) + \lambda_1 H(\mathbf{M}) + \lambda_2 H(\mathbf{F})$$

#### Q79: What is the role of the entropy terms ($H(\mathbf{M}), H(\mathbf{F})$) in GNNExplainer?
**Answer**: They act as regularization terms, forcing the masks to be sparse and binaried (close to 0 or 1) so the final explanation is compact and easy to interpret.

#### Q80: How does GNNExplainer isolate node feature importance?
**Answer**: It learns a feature mask $\mathbf{F} \in [0, 1]^D$. The model's inputs are multiplied by this mask to evaluate which attributes have the greatest impact on the classification.

#### Q81: Does GNNExplainer require retraining the GNN model?
**Answer**: No, the pre-trained GNN model parameters remain frozen. The optimization process only updates the masks for a specific target node's local subgraph.

#### Q82: How do you select the threshold for the edge mask?
**Answer**: Usually chosen empirically (e.g., $\theta \ge 0.5$ or $0.7$) to balance explanation size with coverage of the key structural pathways.

#### Q83: Explain the difference between global and local explainers on graphs.
**Answer**: Local explainers explain the prediction for a specific node or edge, while global explainers identify structural patterns that explain the model's behavior across the entire dataset.

#### Q84: What is a compliance-ready evidence package in TemporalAML?
**Answer**: A JSON package containing the target node prediction, the filtered time-ordered transaction path, and the key node features that triggered the alert.

#### Q85: What are the main limitations of GNNExplainer?
**Answer**: It must be run separately for each node, which can be slow for large batches, and it assumes that a single compact subgraph contains the entire explanation.

---

## Section 6: Metrics, Imbalance & Baselines (Q86 - Q100)

#### Q86: Why is Accuracy a poor metric for evaluating AML models?
**Answer**: Because money laundering datasets are highly imbalanced. If only 0.1% of transactions are illicit, a dummy model that predicts all transactions as licit achieves 99.9% accuracy but fails to detect any laundering.

#### Q87: Define Precision and Recall in the context of laundering alerts.
**Answer**: Precision measures the percentage of flagged alerts that are actually illicit. Recall measures the percentage of all illicit transactions that were successfully flagged by the model.

#### Q88: Write the formula for the F1-Score.
**Answer**:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Q89: What does AUC-ROC measure?
**Answer**: The probability that the model will rank a randomly chosen illicit transaction higher than a randomly chosen licit transaction across all classification thresholds.

#### Q90: How does class weighting handle data imbalance?
**Answer**: It scales the loss function so that misclassifying a minority-class node (illicit) incurs a much larger penalty than misclassifying a majority-class node (licit).

#### Q91: What is the temporal train-validation-test split?
**Answer**: Splitting the dataset chronologically (e.g., training on early time steps, testing on later ones) to evaluate how well the model generalizes to future transaction patterns.

#### Q92: Explain why random splits lead to data leakage in dynamic graphs.
**Answer**: A random split can place a transaction in the training set and its subsequent child transaction in the test set, allowing the model to look forward in time and artificially inflating performance.

#### Q93: What is the LSTM-GNN baseline?
**Answer**: A baseline model that uses a spatial GNN to extract node features at each time step and passes them through an LSTM to capture sequential patterns.

#### Q94: Why is a rule-based baseline included in the evaluation?
**Answer**: To compare the model against the standard heuristics-based systems currently used by compliance teams in industry.

#### Q95: What is an ablation study?
**Answer**: Systematically removing or modifying specific components of a model (e.g., learnable time encodings, multi-task heads) to evaluate their individual impact on performance.

#### Q96: What is a multi-task learning head?
**Answer**: Having multiple task-specific prediction layers share the same core embedding network, allowing the model to learn generalized features across different classification tasks.

#### Q97: Explain the benefit of joint optimization in multi-task networks.
**Answer**: It reduces training time compared to running separate models and helps prevent overfitting by forcing the model to learn features that generalize across all tasks.

#### Q98: How do you handle unlabeled nodes in semi-supervised training?
**Answer**: By including them in the graph structure to propagate message passing, but masking them out so they do not contribute to the loss function calculation.

#### Q99: What is hyperparameter tuning, and which parameters are most critical in TemporalAML?
**Answer**: Optimizing the model's configuration settings. Key parameters include the learning rate, the time encoding dimension ($d_t$), the number of attention heads, and the temporal sample size ($M$).

#### Q100: How do you ensure your GNN model is not overfitting?
**Answer**: By monitoring the validation F1-score and using techniques like dropout, weight decay, early stopping, and node feature masking.

---

## Summary
*   We compiled **100 defense questions** spanning core GNN concepts, temporal encoding mechanics, GNNExplainer optimization, evaluation metrics, and compliance workflows.
*   This structure prepares you for academic defense presentations and technical reviews.

## Key Takeaways
1. Always evaluate performance using F1-score and AUC-ROC on temporal splits to ensure realistic results.
2. Emphasize that learnable frequencies allow the model to adapt specifically to transaction speed distributions.
3. Highlight GNNExplainer's role in converting abstract mathematical vectors into concrete, verifiable transaction paths.

## Recommended Research Papers
1. Ying, R. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
2. Weber, M. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks.*
