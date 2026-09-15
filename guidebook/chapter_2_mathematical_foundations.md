# Chapter 2: Mathematical Foundations

This chapter details the mathematical foundations of Graph Neural Networks (GNNs), Temporal Graph Attention Networks (TGAT), Learnable Fourier Time Encodings, and GNNExplainer. For every mathematical block, we provide the underlying intuition, the formal equation, a step-by-step breakdown, a simple numerical example, an ASCII diagram, and its specific relevance to the TemporalAML project.

---

## 2.1 Graph Theory and Graph Primitives

### 2.1.1 Graph (Undirected)
*   **Intuition**: A graph is a mathematical structure representing a network of items where some pairs of items are connected.
*   **Mathematical Equation**:
    $$G = (V, E)$$
    Where $V = \{v_1, v_2, \dots, v_n\}$ is the set of nodes (vertices) and $E \subseteq V \times V$ is the set of edges (connections).
*   **Step-by-Step Explanation**:
    1.  Define the set of items (nodes $V$).
    2.  Define how they connect (edges $E$). If undirected, the connection between $v_i$ and $v_j$ is denoted as an unordered pair $\{v_i, v_j\}$.
*   **Numerical Example**: Let $V = \{1, 2, 3\}$. Let $E = \{\{1, 2\}, \{2, 3\}\}$. Node 1 is connected to Node 2. Node 2 is connected to Node 3. Node 1 and Node 3 are not directly connected.
*   **ASCII Diagram**:
    ```
    [1] ------- [2] ------- [3]
    ```
*   **Why Needed in Project**: Represents the foundational abstraction of a blockchain transaction network.

### 2.1.2 Directed Graph
*   **Intuition**: Connections have a specific direction (flow), moving from a source node to a target node.
*   **Mathematical Equation**:
    $$E \subseteq \{(v_i, v_j) \mid v_i, v_j \in V \text{ and } v_i \neq v_j\}$$
    Where $(v_i, v_j)$ is an ordered pair representing a directed edge from source $v_i$ to target $v_j$.
*   **Step-by-Step Explanation**:
    1.  Edges are ordered pairs: $(v_i, v_j) \neq (v_j, v_i)$.
    2.  Information or value flows strictly from $v_i$ to $v_j$.
*   **Numerical Example**: $V = \{1, 2, 3\}$. $E = \{(1, 2), (2, 3)\}$. Money flows from 1 to 2, and 2 to 3. Flow from 3 to 2 is not present.
*   **ASCII Diagram**:
    ```
    [1] -------> [2] -------> [3]
    ```
*   **Why Needed in Project**: Cryptocurrencies are directed transfers. Money moves from sender addresses to receiver addresses.

### 2.1.3 Dynamic and Temporal Graphs
*   **Intuition**: Real-world graphs change. A temporal graph includes timestamps on edges (events) showing exactly *when* interactions happened.
*   **Mathematical Equation**:
    $$\mathcal{G} = (V, E, t)$$
    Where $E = \{(v_i, v_j, t_k) \mid v_i, v_j \in V, t_k \in \mathbb{R}^+\}$ is the set of timed events.
*   **Step-by-Step Explanation**:
    1.  A node set $V$ remains active or is added.
    2.  Instead of a static link, edges are discrete interaction events $(v_i, v_j, t_k)$ occurring at continuous timestamp $t_k$.
*   **Numerical Example**:
    $$E = \{(1, 2, 10.0), (2, 3, 15.5)\}$$
    Transaction from 1 to 2 occurs at $t = 10.0$ seconds. Transaction from 2 to 3 occurs later at $t = 15.5$ seconds.
*   **ASCII Diagram**:
    ```
    [1] ----(t=10.0)----> [2] ----(t=15.5)----> [3]
    ```
*   **Why Needed in Project**: Money laundering patterns are highly temporal. We must enforce temporal causality: money must flow forward in time.

### 2.1.4 Node and Edge Features
*   **Intuition**: Nodes and edges carry descriptive features (attributes) represented as vector spaces.
*   **Mathematical Equation**:
    $$\mathbf{x}_v \in \mathbb{R}^d, \quad \mathbf{e}_{u,v} \in \mathbb{R}^p$$
    Where $\mathbf{x}_v$ is a $d$-dimensional feature vector for node $v$, and $\mathbf{e}_{u,v}$ is a $p$-dimensional feature vector for edge $(u,v)$.
*   **Step-by-Step Explanation**:
    1.  Identify attributes (e.g., node balance, edge amount).
    2.  Normalize and compile into continuous feature vectors.
*   **Numerical Example**: For node $1$, $\mathbf{x}_1 = [0.5, -1.2]$ (normalized transaction count, average fee). For edge $(1,2)$, $\mathbf{e}_{1,2} = [150.0]$ (transaction amount in BTC).
*   **ASCII Diagram**:
    ```
    [Node 1: x_1=[-0.2, 0.8]] --(Edge: e_12=[10.0])--> [Node 2: x_2=[1.4, -0.5]]
    ```
*   **Why Needed in Project**: Blockchain addresses have state (balance, transaction rates), and transfers have volumes and timestamps.

### 2.1.5 Adjacency Matrix
*   **Intuition**: A matrix representation of the graph topology where cell $(i,j)$ indicates if node $i$ connects to node $j$.
*   **Mathematical Equation**:
    $$\mathbf{A} \in \{0, 1\}^{N \times N}, \quad A_{ij} = \begin{cases} 1 & \text{if } (v_i, v_j) \in E \\ 0 & \text{otherwise} \end{cases}$$
*   **Step-by-Step Explanation**:
    1.  Create an $N \times N$ matrix.
    2.  Place a 1 at index $(i,j)$ if there is an edge from $i$ to $j$. Place 0 otherwise.
*   **Numerical Example**: For $V = \{1, 2, 3\}$ and $E = \{(1, 2), (2, 3)\}$:
    $$\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$
*   **ASCII Diagram**:
    ```
       1  2  3
    1 [0  1  0]
    2 [0  0  1]
    3 [0  0  0]
    ```
*   **Why Needed in Project**: Crucial for static GNN baselines and for understanding connectivity matrices.

---

## 2.2 Graph Representation Learning

### 2.2.1 Graph Embeddings
*   **Intuition**: Mapping nodes (or subgraphs) from a discrete graph structure into a low-dimensional dense continuous vector space where geometric distance reflects structural and feature similarity.
*   **Mathematical Equation**:
    $$f: V \to \mathbb{R}^k \quad (k \ll |V|)$$
*   **Step-by-Step Explanation**:
    1.  Pass the node features and graph structure through a encoder network.
    2.  Extract the dense embedding vector $\mathbf{h}_v \in \mathbb{R}^k$.
*   **Numerical Example**: Map Node 1 to a 4-dimensional dense representation: $\mathbf{h}_1 = [0.12, -0.45, 0.88, 0.03]$.
*   **ASCII Diagram**:
    ```
    Discrete Graph: [1] <-> [2]  ===[Encoder GNN]===>  Embeddings: h_1 = [ 0.12, -0.45]
                    [3]                                           h_2 = [ 0.15, -0.41]
                                                                  h_3 = [-0.88,  0.02]
    ```
*   **Why Needed in Project**: The shared embedding layer of the GNN is passed to multi-label classification heads to detect laundering.

### 2.2.2 Message Passing Scheme
*   **Intuition**: Nodes update their representation by aggregating features ("messages") sent from their local neighbors.
*   **Mathematical Equation**:
    $$\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)} \left( \mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \left\{ \mathbf{m}_u^{(l)} \mid u \in \mathcal{N}(v) \right\} \right) \right)$$
    Where $\mathbf{m}_u^{(l)} = \text{MESSAGE}^{(l)}(\mathbf{h}_u^{(l)})$ and $\mathcal{N}(v)$ is the set of neighbors of $v$.
*   **Step-by-Step Explanation**:
    1.  Each neighbor $u$ of $v$ computes a message $\mathbf{m}_u$.
    2.  Node $v$ aggregates these messages using a permutation-invariant operator (e.g., sum, mean).
    3.  Node $v$ combines its current feature $\mathbf{h}_v^{(l)}$ with the aggregated message to compute $\mathbf{h}_v^{(l+1)}$.
*   **Numerical Example**: Let $\mathcal{N}(1) = \{2, 3\}$. $\mathbf{h}_2 = [1, 0]$, $\mathbf{h}_3 = [0, 2]$. Aggregate message using Mean:
    $$\mathbf{m}_{\text{agg}} = \frac{[1,0] + [0,2]}{2} = [0.5, 1.0]$$
    Update $\mathbf{h}_1 = [0,0]$ by adding the aggregated message: $\mathbf{h}_1^{\text{new}} = [0.5, 1.0]$.
*   **ASCII Diagram**:
    ```
    [Neighbor 2] ---(m_2)---\
                             v
    [Neighbor 3] ---(m_3)---> [Aggregate] ---> [Update] ---> New State for Node 1
                             ^
    [Self Node 1] --(h_1)---/
    ```
*   **Why Needed in Project**: The mechanism through which transaction features propagate through multi-hop transfer chains.

### 2.2.3 Graph Convolution (GCN Layer)
*   **Intuition**: An approximation of spectral graph convolutions operating in the spatial domain using normalized node feature aggregation.
*   **Mathematical Equation**:
    $$\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{h}_u^{(l)} \right)$$
    Where $\tilde{d}_v = 1 + \sum_{u \in \mathcal{N}(v)} 1$, and $\mathbf{W}^{(l)}$ is a learnable weight matrix.
*   **Step-by-Step Explanation**:
    1.  Add self-loops to the adjacency matrix.
    2.  Normalize edge weights using the square root of the degrees of source and target nodes.
    3.  Multiply neighbor states by normalized weights, sum them, apply weight matrix $\mathbf{W}$, and pass through non-linearity $\sigma$.
*   **Numerical Example**: Let node $v$ have degree $\tilde{d}_v = 2$, neighbor $u$ have degree $\tilde{d}_u = 3$. The normalization factor is:
    $$\frac{1}{\sqrt{2 \times 3}} = \frac{1}{\sqrt{6}} \approx 0.408$$
*   **ASCII Diagram**:
    ```
    [u] (deg=3) ---(x 0.408)---\
                                v
    [v] (deg=2) ---(x 0.500)---> [Sum] ---> [W * Matrix] ---> [ReLU] ---> h_v_new
    ```
*   **Why Needed in Project**: Serves as the primary static GNN baseline.

---

## 2.3 Attention Mechanisms in Graphs

### 2.3.1 Attention Mechanism (GAT Layer)
*   **Intuition**: Instead of using static normalization weights like GCN, GAT computes dynamically learned attention weights to specify which neighbors are most important.
*   **Mathematical Equation**:
    $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_k]\right)\right)}$$
    Where $\mathbf{W}$ is a linear projection weight matrix, $\mathbf{a}$ is the attention parameter vector, and $\parallel$ is concatenation.
*   **Step-by-Step Explanation**:
    1.  Project node features into a higher-dimensional space using $\mathbf{W}$.
    2.  Concatenate target node embedding $\mathbf{W}\mathbf{h}_i$ and source neighbor $\mathbf{W}\mathbf{h}_j$.
    3.  Compute dot-product with vector $\mathbf{a}$, apply LeakyReLU, and normalize across neighbors using Softmax.
*   **Numerical Example**: Let projected features be $\mathbf{z}_1 = [1, 0]$ and $\mathbf{z}_2 = [0, 1]$. Concatenate: $\mathbf{z}_{1,2} = [1, 0, 0, 1]$. Let $\mathbf{a} = [1, 0.5, 0.5, 1]$. Dot product:
    $$\mathbf{a}^{\top} \mathbf{z}_{1,2} = (1)(1) + (0.5)(0) + (0.5)(0) + (1)(1) = 2.0$$
    Apply LeakyReLU: $2.0$. Softmax normalized weight will be determined relative to other neighbors.
*   **ASCII Diagram**:
    ```
    h_i, h_j ---> [Linear Project W] ---> Concatenate ---> Dot Product with 'a' ---> LeakyReLU ---> Softmax (alpha_ij)
    ```
*   **Why Needed in Project**: Allows the model to prioritize transactions coming from suspicious structural patterns over normal ones.

### 2.3.2 Multi-Head Attention
*   **Intuition**: Using multiple independent attention heads allows the model to jointly attend to information from different representation subspaces at different positions.
*   **Mathematical Equation**:
    $$\mathbf{h}_i^{(l+1)} = \parallel_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)} \right)$$
    Where $K$ is the number of heads, and $\alpha_{ij}^k$ are the attention coefficients computed by the $k$-th head.
*   **Step-by-Step Explanation**:
    1.  Run $K$ independent single-head GAT computations.
    2.  Concatenate the resulting $K$ output vectors (or average them for the final layer).
*   **Numerical Example**: Let $K = 2$. Head 1 output is $\mathbf{z}_i^1 = [0.1, 0.9]$. Head 2 output is $\mathbf{z}_i^2 = [0.5, 0.2]$. The concatenated feature is:
    $$\mathbf{h}_i^{\text{concat}} = [0.1, 0.9, 0.5, 0.2]$$
*   **ASCII Diagram**:
    ```
                     +---> [Attention Head 1] ---> h_i^1 ---+
    Input Features --|                                      |---> [Concatenate] ---> Multi-Head Output
                     +---> [Attention Head 2] ---> h_i^2 ---+
    ```
*   **Why Needed in Project**: Allows one head to focus on topological features (e.g., node degree, transaction counts) while another focuses on temporal patterns (e.g., transaction velocities).

---

## 2.4 Temporal Graph Attention Networks (TGAT)

### 2.4.1 Fourier Time Encoding
*   **Intuition**: Neural networks cannot natively generalize to raw continuous time intervals. We use a Fourier transform of time-differences to map any continuous time duration $\Delta t$ into a stable, multi-frequency vector representation.
*   **Mathematical Equation**:
    $$\Phi_d(\Delta t) = \sqrt{\frac{2}{d_t}} \left[ \cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots, \cos(\omega_{d_t/2} \Delta t), \sin(\omega_{d_t/2} \Delta t) \right]^{\top}$$
    Where $\omega_1, \dots, \omega_{d_t/2}$ are the frequencies.
*   **Step-by-Step Explanation**:
    1.  Compute the time delta: $\Delta t = t_{\text{target}} - t_{\text{event}}$.
    2.  Multiply $\Delta t$ by different frequency scales $\omega_k$.
    3.  Compute sine and cosine values for each scaled time delta.
    4.  Concatenate components into a vector of dimension $d_t$.
*   **Numerical Example**: Let $d_t = 2$, frequency $\omega_1 = 0.5$, $\Delta t = 2.0$.
    $$\Phi_2(2.0) = \sqrt{\frac{2}{2}} [\cos(0.5 \times 2), \sin(0.5 \times 2)]^{\top} = [\cos(1.0), \sin(1.0)]^{\top} \approx [0.54, 0.84]^{\top}$$
*   **ASCII Diagram**:
    ```
    Continuous time delta (\Delta t) ---> [Freq Multipliers \omega_k] ---> [Cos / Sin Kernels] ---> Time Embedding Vector
    ```
*   **Why Needed in Project**: Acts as the backbone for continuous-time learning, allowing the network to process the exact time elapsed between consecutive hops.

### 2.4.2 Learnable Fourier Encoding (O2)
*   **Intuition**: Instead of using fixed grid frequencies like in Positional Encodings (e.g., Transformer), we treat the frequencies $\omega_k$ as learnable model parameters. The model optimizes these frequencies using backpropagation to adapt specifically to the transaction velocities present in the dataset.
*   **Mathematical Equation**:
    $$\mathbf{\omega} = [\omega_1, \omega_2, \dots, \omega_{d_t/2}]^{\top} \in \mathbb{R}^{d_t/2} \quad \text{(Learnable Parameters)}$$
    $$\frac{\partial L}{\partial \omega_k} = \frac{\partial L}{\partial \Phi} \cdot \frac{\partial \Phi}{\partial \omega_k}$$
    $$\frac{\partial \Phi_{\cos, k}}{\partial \omega_k} = -\Delta t \cdot \sin(\omega_k \Delta t), \quad \frac{\partial \Phi_{\sin, k}}{\partial \omega_k} = \Delta t \cdot \cos(\omega_k \Delta t)$$
*   **Step-by-Step Explanation**:
    1.  Initialize $\mathbf{\omega}$ randomly or with exponential grid scales.
    2.  During the forward pass, calculate $\Phi_d(\Delta t)$ using the current $\mathbf{\omega}$.
    3.  During backpropagation, compute gradients with respect to $\omega_k$ and update frequencies using the optimizer.
*   **Numerical Example**: Let $\omega_k = 0.5$, $\Delta t = 2.0$. If the loss gradient upstream $\frac{\partial L}{\partial \Phi_{\cos, k}} = 0.1$, then:
    $$\frac{\partial L}{\partial \omega_k} = 0.1 \times (-2.0 \times \sin(1.0)) = -0.2 \times 0.8415 \approx -0.1683$$
    The optimizer will update $\omega_k$ using this gradient.
*   **ASCII Diagram**:
    ```
    Forward:  \omega_k ---> [Compute Encoding] ---> Loss
    Backward: Loss Gradient ---> [Compute dL/d\omega_k] ---> Update \omega_k via Adam
    ```
*   **Why Needed in Project**: Dissertations require novel methodology. Enabling *learnable* time encodings ensures the model adapts dynamically to differing transaction speeds of smart-contracts and human actions.

### 2.4.3 Temporal Neighborhood Sampling
*   **Intuition**: A node might have thousands of historical edges. To perform fast batched GNN training, we must sample a fixed size $M$ of historical edges that occurred prior to the target time $t$.
*   **Mathematical Equation**:
    $$\mathcal{N}_{\text{temp}}(i, t) = \text{Top-}M \text{ or Sample-}M \left( \{(j, t_e) \mid (j, i, t_e) \in E \text{ and } t_e < t\} \right)$$
*   **Step-by-Step Explanation**:
    1.  Filter all incoming/outgoing edges of node $i$ occurring before target time $t$.
    2.  Sort or sample $M$ transactions (typically the most recent ones).
*   **Numerical Example**: Node 1 has historical edges at times: $\{5.0, 8.0, 12.0\}$. We evaluate Node 1 at target time $t = 10.0$ with $M = 2$.
    Eligible edges: $\{5.0, 8.0\}$. The sampled neighborhood will be these two events.
*   **ASCII Diagram**:
    ```
                      Cutoff t=10.0
    Timeline: --(5.0)--[Yes]--(8.0)--[Yes]--|--(12.0)--[No/Future]--->
    Sampled: {5.0, 8.0}
    ```
*   **Why Needed in Project**: Prevents temporal data leakage and handles memory bounds during training.

### 2.4.4 Time-Aware GNN Attention (TGAT Layer Forward Pass)
*   **Intuition**: A node aggregates neighbor features by considering both the neighbor's structural embedding and the elapsed time since that neighbor interacted with the target node.
*   **Mathematical Equation**:
    $$\mathbf{q}_i = [\mathbf{h}_i(t) \parallel \Phi_d(0)] \mathbf{W}_Q$$
    $$\mathbf{k}_j = [\mathbf{h}_j(t_e) \parallel \Phi_d(t - t_e)] \mathbf{W}_K$$
    $$\mathbf{v}_j = [\mathbf{h}_j(t_e) \parallel \Phi_d(t - t_e)] \mathbf{W}_V$$
    $$\mathbf{h}_i(t) = \text{MultiHeadAttention}(\mathbf{q}_i, \mathbf{K}, \mathbf{V})$$
    Where $\mathbf{h}_j(t_e)$ is the historical embedding of neighbor $j$ at transaction time $t_e$.
*   **Step-by-Step Explanation**:
    1.  Define the Query ($\mathbf{q}_i$) using target node state at current time $t$ concatenated with the time-encoding of zero difference.
    2.  For each neighbor $j$, define Key ($\mathbf{k}_j$) and Value ($\mathbf{v}_j$) using neighbor state at transaction time $t_e$ concatenated with the time encoding of the difference $\Delta t = t - t_e$.
    3.  Compute dot-product attention scores between $\mathbf{q}_i$ and $\mathbf{k}_j$, normalize, and multiply by $\mathbf{v}_j$.
*   **Numerical Example**: Let query $\mathbf{q}_i = [1, 0]$, keys $\mathbf{k}_1 = [1, 0]$ and $\mathbf{k}_2 = [0, 1]$. Dot products: $\mathbf{q}_i \mathbf{k}_1 = 1$, $\mathbf{q}_i \mathbf{k}_2 = 0$. Normalized weights: $e^1 / (e^1 + e^0) \approx 0.73$, $e^0 / (e^1 + e^0) \approx 0.27$. Node 1 receives 73% of attention weight.
*   **ASCII Diagram**:
    ```
    Target Node i (t)   -----> Query q_i \
                                          +---> [Dot Product] ---> Softmax ---> Attention Weighted Sum
    Neighbor j (t_e)    -----> Key k_j   /
    (Delta t = t - t_e) -----> Value v_j /
    ```
*   **Why Needed in Project**: Core layer of the TemporalAML model.

---

## 2.5 Machine Learning Loss and Evaluation Primitives

### 2.5.1 Binary Cross Entropy (BCE) Loss with Class Imbalance
*   **Intuition**: Measures performance of classification models whose output is a probability value between 0 and 1. We weight positive instances heavily to handle high class imbalance.
*   **Mathematical Equation**:
    $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ w \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
    Where $y_i \in \{0, 1\}$ is the ground truth label, $\hat{y}_i$ is the predicted probability, and $w > 1.0$ is the positive class weight.
*   **Step-by-Step Explanation**:
    1.  Compute probability $\hat{y}_i$ using Sigmoid output.
    2.  If $y_i = 1$, penalize using $w \cdot \log(\hat{y}_i)$.
    3.  If $y_i = 0$, penalize using $\log(1 - \hat{y}_i)$.
*   **Numerical Example**: Let $y = 1$, $\hat{y} = 0.2$, $w = 10.0$.
    $$\mathcal{L}_i = - [ 10 \times 1 \times \log(0.2) + 0 ] \approx - [ 10 \times (-1.609) ] \approx 16.09$$
    Without weight $w$, the loss would be only $1.609$. This increases penalty for missing positive cases.
*   **ASCII Diagram**:
    ```
    Prediction (y_hat) ---\
                           +---> [Weighted Loss calculation] ---> Gradient Backprop
    Ground Truth (y)   ---/
    ```
*   **Why Needed in Project**: Necessary to handle the class imbalance in money laundering datasets (typically < 1% suspicious nodes).

### 2.5.2 Evaluation Metrics
*   **Intuition**: Evaluate classification success under heavy class imbalance.

#### A. Precision
*   **Equation**:
    $$\text{Precision} = \frac{TP}{TP + FP}$$
*   **Explanation**: Out of all predicted positive cases, what percentage were actually positive. Prevents false alarms.

#### B. Recall
*   **Equation**:
    $$\text{Recall} = \frac{TP}{TP + FN}$$
*   **Explanation**: Out of all actual positive cases, what percentage did we catch. Prevents missing real money launderers.

#### C. F1 Score
*   **Equation**:
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
*   **Explanation**: Harmonic mean of Precision and Recall.

#### D. ROC and AUC
*   **Explanation**: **ROC** (Receiver Operating Characteristic) plots True Positive Rate vs False Positive Rate at various threshold settings. **AUC** (Area Under the Curve) measures the probability that the model ranks a random positive instance higher than a random negative instance.
*   **Why Needed in Project**: High accuracy can be achieved trivially by predicting "legitimate" for all nodes due to imbalance. We must use Precision, Recall, F1, and AUC-ROC on the minority class.

---

## 2.6 Graph Neural Network Explainability

### 2.6.1 GNNExplainer
*   **Intuition**: For compliance and verification, we cannot treat GNNs as black boxes. GNNExplainer identifies a compact subgraph and a subset of node features that are most crucial for the GNN's prediction.
*   **Mathematical Equation**:
    $$\min_{\mathbf{M}, \mathbf{F}} -\sum_{c=1}^C \mathbb{E}_{\mathbf{A}_s \sim \mathbf{M}} \left[ y_c \log P(Y = c \mid G_s(\mathbf{A}_s), X_s(\mathbf{F})) \right] + \lambda_1 H(\mathbf{M}) + \lambda_2 H(\mathbf{F})$$
    Where $\mathbf{M}$ is a mask over edges, $\mathbf{F}$ is a mask over node features, $G_s(\mathbf{A}_s)$ is the masked subgraph, and $H(\cdot)$ is an entropy regularization term enforcing sparsity.
*   **Step-by-Step Explanation**:
    1.  Select a target node $v$ predicted as suspicious.
    2.  Initialize soft, learnable edge masks $\mathbf{M}$ and node feature masks $\mathbf{F}$ initialized to 0.5.
    3.  Pass the masked graph through the pre-trained, frozen GNN.
    4.  Optimize masks to minimize the change in prediction probability while maximizing sparsity of the mask.
    5.  Threshold masks to extract the final explanatory subgraph.
*   **Numerical Example**: An edge mask parameter has value $0.92$ for transactions to Node X and $0.05$ for transactions to Node Y. Node X's link is retained as evidence, whereas Node Y's link is discarded.
*   **ASCII Diagram**:
    ```
    Suspicious Prediction ---> [Freeze GNN] ---> [Learn Mask M on Edges] ---> [Threshold Mask] ---> Output Explanatory Subgraph
    ```
*   **Why Needed in Project**: Objective O4. Financial compliance requires concrete transaction subgraphs detailing exactly which sequence of payments triggered an alert.

---

## Connection to Dissertation Objectives

*   **Objective O2 (Learnable Fourier Encoding)**: Section 2.4.2 provides the explicit derivatives and intuition for learnable frequencies.
*   **Objective O4 (GNNExplainer)**: Section 2.6 defines how subgraphs are filtered to form evidence packages.

---

## Summary
*   **Graphs** provide structural representations of payment ledgers. **Temporal Graphs** add dynamic timestamps to prevent data leakage.
*   **Message passing** aggregates features along hops. **GCN** uses structural degrees, **GAT** uses learnable spatial attention, and **TGAT** integrates learnable continuous Fourier time-delta encodings.
*   Evaluation must focus on **F1 and AUC-ROC** rather than accuracy due to massive class imbalance.
*   **GNNExplainer** uses entropy-regularized mask learning to identify critical transaction sequences.

## Key Takeaways
1. Learnable time encodings allow optimization of frequency components via backpropagation.
2. Temporal graphs prevent future-to-past information leakage during message aggregation.
3. Class imbalance requires weighted binary cross entropy loss.

## Practical Implementation Checklist
- [ ] Implement a basic Cosine/Sine time encoding function in PyTorch.
- [ ] Define a custom BCE module with positive class weighting.
- [ ] Configure PyTorch Geometric variables (`edge_index`, `edge_attr`, `x`) for custom spatial datasets.

## Recommended Research Papers
1. Velickovic, P., et al. (2018). *Graph Attention Networks.* ICLR 2018.
2. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
3. Xu, D., et al. (2020). *Inductive Representation Learning on Temporal Graphs.* ICLR 2020.

## Suggested Coding Exercises
1. Write a PyTorch function for Fourier encoding of a batch of 1D time-differences.
2. Write a manual backpropagation step for a single weight update of a learnable frequency parameter $\omega$.
3. Compute Precision, Recall, and F1 manually given a small $10 \times 1$ target and prediction vector.
