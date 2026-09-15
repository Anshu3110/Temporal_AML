# 🎓 TemporalAML: Master Dissertation & Viva Defense Guide
## Complete Technical Foundation, Cryptocurrency Background, AML Patterns, and Presentation Blueprint

> **Project Title:** TemporalAML: Temporal Graph Attention Networks with Learnable Fourier Time Encoding for Explainable Multi-Pattern Anti-Money Laundering in Cryptocurrency Transactions  
> **Author:** Anshu Kumari (Reg No: 2567203) — 3MTDS, Department of AI & Data Science Engineering, CHRIST (Deemed to be University)  
> **Faculty Mentors:** Dr. Janani V S (Guide), Dr. Aruna S K (Co-Guide)

---

# 📑 TABLE OF CONTENTS
1. [Core Cryptocurrency & Blockchain Foundations (Mentor Question Prep)](#1-core-cryptocurrency--blockchain-foundations)
   - 1.1 What is a Cryptocurrency?
   - 1.2 What is Bitcoin? (History, Genesis Block, UTXO, Halving)
   - 1.3 What is Ethereum? (Smart Contracts, EVM, Account Model)
   - 1.4 What is a Digital Signature? (ECDSA, secp256k1, Math & Verification)
   - 1.5 Cryptographic Hash Functions & Merkle Trees
2. [The 3 Money Laundering Patterns in Cryptocurrency](#2-the-3-money-laundering-patterns-in-cryptocurrency)
   - 2.1 The Pseudonymity Paradox & The AML Crisis
   - 2.2 Pattern 1: Circular Transfers (Cycles & Wash Trading)
   - 2.3 Pattern 2: Layering (Sequential Chains & Velocity)
   - 2.4 Pattern 3: Smurfing (Fan-Out / Fan-In Structuring)
   - 2.5 Summary Table: How and Why Patterns are Mined & Detected
3. [Clear Justification for Problem Selection](#3-clear-justification-for-problem-selection)
   - 3.1 Why Cryptocurrency AML?
   - 3.2 Why Graph Neural Networks (GNNs)?
   - 3.3 Why Continuous Temporal Attention (TGAT) over Static/LSTM?
   - 3.4 Why Multi-Pattern Classification over Binary Classification?
4. [Dataset Deep Dive (Elliptic Bitcoin Dataset)](#4-dataset-deep-dive)
   - 4.1 Dataset Identity & Scale
   - 4.2 Attribute Breakdown (Local 94 + Neighbor 72 + Engineered 6 = 172D)
   - 4.3 Domain Importance & Ground-Truth Entity Tagging
   - 4.4 Temporal Dimension & Leakage-Free Chronological Splits
5. [System Architecture — 6 Modular Block Components](#5-system-architecture--6-modular-block-components)
6. [Specific Research Objectives (O1 to O4)](#6-specific-research-objectives)
7. [Slide-by-Slide Review Script (What to Say to Your Mentor)](#7-slide-by-slide-review-script)
8. [Top 15 Tough Viva & Mentor Questions with Model Answers](#8-top-15-tough-viva--mentor-questions)

---

# 1. CORE CRYPTOCURRENCY & BLOCKCHAIN FOUNDATIONS

This section provides the exact technical and historical answers you need when your mentor or examiner asks foundational questions.

---

### 1.1 What is a Cryptocurrency?
* **Definition:** A cryptocurrency is a **decentralized, digital, peer-to-peer monetary system** secured by cryptography rather than a central authority (like a central bank, government, or payment processor like Visa).
* **The Core Computer Science Problem Solved:** The **Double-Spending Problem**.
  * *Physical Money:* If you give a \$10 paper bill to someone, physical possession transfers; you cannot spend that same \$10 bill elsewhere.
  * *Digital Information:* Any digital file (PDF, MP3) can be duplicated infinitely at zero cost. Without a central bank, how do you prevent someone from spending the exact same digital token twice?
* **The Triad Solution:**
  1. **Asymmetric Cryptography & Digital Signatures:** Guarantees that only the legitimate private key holder can authorize a transaction.
  2. **Distributed Immutable Ledger (Blockchain):** Every validated transaction is bundled into a timestamped block and linked cryptographically to the previous block.
  3. **Consensus Mechanisms (PoW / PoS):** A distributed network of independent validator nodes agrees on the single true state of the ledger without trusting any single entity.

---

### 1.2 What is Bitcoin? (History, Genesis & Economics)
* **Origins & The 2008 Financial Crisis:**
  * In September 2008, the global subprime mortgage crisis collapsed major financial institutions, leading to massive government bailouts and fractional-reserve fiat currency devaluation.
  * On **October 31, 2008**, an anonymous cryptographer (or group) using the pseudonym **Satoshi Nakamoto** published the whitepaper: *"Bitcoin: A Peer-to-Peer Electronic Cash System"*.
  * On **January 3, 2009**, Satoshi mined the **Genesis Block (Block 0)**, embedding the historical text in the coinbase parameter:  
    `"The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"`
* **Core Economic & Protocol Properties:**
  * **Hard Supply Cap:** Exactly **21,000,000 BTC** will ever exist (disinflationary/deflationary design).
  * **Block Time:** Target of **10 minutes** per block.
  * **Difficulty Adjustment:** Every 2,016 blocks (~2 weeks), network mining difficulty recalculates so block creation time remains 10 minutes regardless of global mining hardware hash power.
  * **Halving Cycles:** Block reward cuts in half every 210,000 blocks (~4 years):  
    `50 BTC (2009) → 25 BTC (2012) → 12.5 BTC (2016) → 6.25 BTC (2020) → 3.125 BTC (2024)`
* **The UTXO Model (Unspent Transaction Output):**
  * Bitcoin **does not have account balances**.
  * The ledger consists exclusively of **UTXOs**. A transaction takes unspent outputs from previous transactions as *inputs*, consumes them entirely, and creates *new outputs* (amount sent to recipient + change returned to sender).
  * **Graph Representation:** This structure means Bitcoin transaction history is naturally a **Directed Acyclic Graph (DAG)** of transactions ($V = \text{Transactions}, E = \text{Fund Flows}, T = \text{Timestamps}$).

---

### 1.3 What is Ethereum? (Smart Contracts & World Computer)
* **Proposed in 2013** by **Vitalik Buterin**; launched in 2015.
* **Why Ethereum was created:** Bitcoin’s scripting language (`Script`) is intentionally restricted (non-Turing complete, no loops) for security as pure digital gold. It cannot execute arbitrary programmable logic.
* **Ethereum Virtual Machine (EVM):** A decentralized, Turing-complete global runtime engine.
* **Smart Contracts:** Immutable, self-executing software programs deployed on the blockchain written in languages like **Solidity** (e.g., *"If event X occurs at timestamp T, automatically transfer Y tokens to party Z"*).
* **Account-Based Model:** Unlike Bitcoin's UTXO model, Ethereum uses an **Account/Balance Model** (similar to a bank account where State = Address $\to$ Balance/Nonce/Code).

---

### 1.4 What is a Digital Signature? (ECDSA & The Mathematics)
A digital signature provides **Authentication** (proves sender identity), **Non-Repudiation** (sender cannot deny signing), and **Integrity** (proves message was not altered).

Bitcoin uses the **Elliptic Curve Digital Signature Algorithm (ECDSA)** over the standardized curve **`secp256k1`**:

$$y^2 \equiv x^3 + 7 \pmod p$$

where $p = 2^{256} - 2^{32} - 977$ (a 256-bit prime number).

```
   [ Private Key: d_A ] (Random secret 256-bit scalar integer)
            │
            ▼ (Elliptic Curve Point Multiplication: Q_A = d_A · G)
   [ Public Key: Q_A ] (Coordinate Point on secp256k1 curve)
            │
            ▼ (SHA-256 followed by RIPEMD-160)
   [ Bitcoin Wallet Address ] (Public alphanumeric string)
```

#### The Math Behind Signing & Verification:
1. **Key Generation:**
   * Private Key $d_A \in [1, n-1]$ (Secret scalar).
   * Generator Point $G$ (Standardized base point on curve).
   * Public Key $Q_A = d_A \cdot G$ (One-way point multiplication; solving for $d_A$ is the *Elliptic Curve Discrete Logarithm Problem*, which takes billions of years to break).
2. **Signing a Transaction:**
   * Calculate transaction message hash: $z = \text{SHA256}(\text{tx\_data})$.
   * Generate cryptographically secure random nonce $k$.
   * Compute point $R = k \cdot G = (x_1, y_1)$ and set $r = x_1 \pmod n$.
   * Compute scalar $s = k^{-1}(z + r \cdot d_A) \pmod n$.
   * **The Signature is the pair $(r, s)$.**
3. **Verification (by any node worldwide):**
   * Using only Public Key $Q_A$, message hash $z$, and signature $(r, s)$:
   $$w = s^{-1} \pmod n, \quad u_1 = z \cdot w \pmod n, \quad u_2 = r \cdot w \pmod n$$
   $$R' = u_1 \cdot G + u_2 \cdot Q_A$$
   * The signature is valid **if and only if** the $x$-coordinate of $R'$ equals $r$.
* **Why this matters for your project:** Digital signatures ensure only private key holders create transaction edges. However, criminals use private keys from thousands of burner/mule wallets to orchestrate multi-hop laundering chains.

---

### 1.5 Cryptographic Hash Functions & Merkle Trees
* **SHA-256 (Secure Hash Algorithm 256-bit):**
  * *Deterministic:* Same input always yields exact same 64-hex-character output.
  * *Pre-image Resistance (One-Way):* Given $H(x)$, it is computationally impossible to invert to find $x$.
  * *Collision Resistance:* Infeasible to find $x_1 \ne x_2$ such that $H(x_1) = H(x_2)$.
  * *Avalanche Effect:* Changing a single bit in the input completely randomizes the entire hash output.
* **Merkle Tree:** A binary hash tree summarizing all transactions in a block into a single 32-byte **Merkle Root** stored in the block header. Allows cryptographic $O(\log N)$ proof of inclusion (*Simplified Payment Verification*).

---

# 2. THE 3 MONEY LAUNDERING PATTERNS IN CRYPTOCURRENCY

Your mentor specifically asked: **"What are these patterns and how do we get to know they are being used for money laundering?"**

---

### 2.1 The Pseudonymity Paradox & The AML Problem
* **The Misconception:** Bitcoin is *not* anonymous; it is **pseudonymous**.
* **The Reality:** Every single transaction since 2009 is permanently public. However, real-world identities are replaced by cryptographic address hashes.
* **The Criminal Choke Point:** Illicit actors (ransomware gangs, darknet vendors, stolen exchange fund thieves) eventually need to convert cryptocurrency into fiat currency (USD, EUR, INR) at regulated exchanges with **KYC (Know Your Customer)** verification.
* **The Laundering Goal:** To break the structural link between the crime-originating wallet and the destination deposit wallet, criminals execute multi-hop graph patterns.

---

### 2.2 Pattern 1: Circular Transfers (Cycles & Volume Obfuscation)
```
       [Source Wallet A] (Illicit Source)
              │                    ▲
              ▼                    │ (t = t + Δt)
        [Proxy Wallet B] ───► [Proxy Wallet C]
```
* **What is it?** A closed directed transaction loop where funds pass through multiple intermediate wallets and eventually return to the initial entity or an affiliated wallet ($A \to B \to C \to A$).
* **How Criminals Use It for Money Laundering:**
  1. **Wash Trading & Volume Spoofing:** Simulates artificial liquidity and high legitimate transaction volume to disguise the flow of illicit funds among thousands of self-directed transactions.
  2. **Audit Trail Confusion:** Automated compliance systems tracing backward from node $A$ get stuck in infinite recursive loops or lose provenance tracking.
  3. **Mixer/Tumbler Re-routing:** Circular transfers are used to cycle funds repeatedly through unhosted pooling nodes before extraction.
* **How We Detect It Graph-Theoretically:** We apply **Tarjan’s Strongly Connected Components (SCC)** algorithm. Any directed subgraph where every node is reachable from every other node with cycle length $\ge 2$ is flagged as a circular transfer topology.

---

### 2.3 Pattern 2: Layering (Sequential Chains & Velocity Manipulation)
```
  [Crime Origin] ──► [Hop 1] ──► [Hop 2] ──► [Hop 3] ──► [Hop 4] ──► [Unregulated Exit]
      (t = 1)        (t = 2)     (t = 3)     (t = 4)     (t = 5)
```
* **What is it?** A long, linear sequential chain of rapid, consecutive transaction hops across continuous timestamps ($t_1 < t_2 < t_3 < \dots < t_k$) with path length $\ge 3$ hops.
* **How Criminals Use It for Money Laundering:**
  1. **Distance-from-Source Decay:** Traditional heuristic systems reduce suspicion scores the further a transaction is from a flagged blacklisted address (e.g., suspicion drops 50% per hop). Criminals deliberately create 10–20 rapid "peeling hops" so the destination wallet appears "clean".
  2. **Jurisdiction & Chain Hopping:** Each hop moves funds through intermediate accounts across different non-cooperative geographical services.
  3. **Time Velocity Signature:** Unlike ordinary users who hold funds for days or weeks, layering algorithms execute consecutive hops in minutes.
* **How We Detect It Graph-Theoretically:** Directed **Breadth-First Search (BFS)** with a temporal monotonicity constraint: extracting paths where $\text{length} \ge 3$ and $t_{k+1} \ge t_k$.

---

### 2.4 Pattern 3: Smurfing (Fan-Out / Fan-In Structuring)
```
  FAN-OUT (Structuring Phase):               FAN-IN (Integration Phase):
       [Large Illicit Fund]                       [Mule 1] [Mule 2] [Mule 3]
       (e.g., 50 BTC Crime)                           │        │        │
          ┌─────┼─────┐                                └────────┼────────┘
          ▼     ▼     ▼                                         ▼
       [0.5]  [0.5]  [0.5] (Mule Wallets)            [Single Destination Vault]
```
* **What is it?** 
  * **Fan-Out:** A single source node splits a large fund into dozens of micro-transactions sent to separate addresses ($d_{\text{out}} \ge 10$).
  * **Fan-In:** Multiple small accounts consolidate funds into a single destination account ($d_{\text{in}} \ge 10$).
* **How Criminals Use It for Money Laundering:**
  1. **Bypassing Mandatory Reporting Thresholds (Structuring):** Global AML regulations (FATF, FinCEN) mandate automated reporting for transactions over \$10,000. Criminals use automated "smurf" scripts to split \$500,000 into 60 transactions of \$8,500 each to stay below reporting radar.
  2. **Money Mule Networks:** Funds are distributed to hundreds of recruited "mules" who individually deposit small amounts into exchanges, avoiding suspicion.
* **How We Detect It Graph-Theoretically:** High graph degree centrality anomalies: $d_{\text{out}}(v) \ge 10$ or $d_{\text{in}}(v) \ge 10$ with high fan-out/fan-in ratios ($\frac{d_{\text{out}}}{d_{\text{in}} + d_{\text{out}} + \epsilon} > 0.8$).

---

### 2.5 Summary Table: Laundering Typologies Comparison

| Pattern Name | Graph Topology Signature | Mathematical Extraction Algorithm | Financial Money Laundering Objective |
| :--- | :--- | :--- | :--- |
| **Circular Transfer** | Directed Closed Cycle ($A \to B \to C \to A$) | **Tarjan's SCC (Size $\ge 2$)** | Wash trading, volume spoofing, creating recursive loops in audit software. |
| **Layering** | Long Monotonic Linear Path ($\ge 3$ hops) | **Temporal Directed BFS ($t_{k+1} \ge t_k$)** | Distancing dirty funds from crime origin; defeating hop-decay AML heuristics. |
| **Smurfing** | Star Subgraph (High In/Out Degree) | **Degree Thresholding ($d_{\text{out}} \ge 10 \lor d_{\text{in}} \ge 10$)** | Structuring transactions below mandatory regulatory reporting limits (\$10k). |

---

# 3. CLEAR JUSTIFICATION FOR PROBLEM SELECTION

When your mentor asks **"Why this problem and why this exact approach?"**, present these 4 clear justifications:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        4 PILLARS OF PROBLEM JUSTIFICATION                              │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ 1. Why Crypto AML?             │ • Global AML is $800B–$2T (2-5% GDP).                 │
│                                │ • Crypto laundering doubled year-over-year.           │
│                                │ • Pseudonymous addresses bypass legacy rule filters.  │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 2. Why Graph Neural Networks?  │ • Financial transactions are relational graphs.       │
│                                │ • Tabular models (XGBoost/RF) analyze nodes in        │
│                                │   isolation, missing multi-hop laundering chains.     │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 3. Why Continuous TGAT?        │ • Static GNNs cause future data leakage.              │
│                                │ • Hybrid GNN-LSTM decouples space & time.             │
│                                │ • TGAT embeds continuous Fourier time delta Δt        │
│                                │   directly inside the attention mechanism.            │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 4. Why Multi-Pattern XAI?      │ • Binary (illicit/licit) alerts are non-actionable.   │
│                                │ • Regulators require specific typology identification │
│                                │   and evidence subgraphs for SAR filing.              │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

---

# 4. DATASET DEEP DIVE (ELLIPTIC BITCOIN DATASET)

Your mentor requested: **"dataset -> attributes -> importance -> temporal should be explained clearly."**

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      ELLIPTIC BITCOIN DATASET ARCHITECTURE                             │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ 1. DATASET IDENTITY:                                                                   │
 │    • Published by Elliptic & MIT-IBM Watson AI Lab (Kaggle Benchmark).                 │
 │    • 203,769 Transaction Nodes  |  234,355 Directed Edges  |  49 Discrete Time Steps.  │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ 2. ATTRIBUTE BREAKDOWN (172 Total Features):                                           │
 │    • Local Attributes (94 dims): Tx fee, BTC volume, input/output counts, byte size.   │
 │    • Neighborhood Attributes (72 dims): 1-hop aggregate statistics (mean, std, min).   │
 │    • Domain-Engineered Temporal (6 dims): Out-degree, in-degree, fan-out ratio,       │
 │      fan-in ratio, temporal recency (t/49), neighbor time delta.                       │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ 3. DOMAIN IMPORTANCE:                                                                  │
 │    • Real-world ground-truth entities tagged by forensic investigators.                │
 │    • Severe Class Imbalance: 4,545 Illicit (2.2%) vs 42,019 Licit (20.6%) vs           │
 │      157,205 Unknown (77.2% — preserved for message-passing structure).                │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ 4. TEMPORAL STRUCTURE & SPLIT (Zero Data Leakage):                                     │
 │    • 49 consecutive time snapshots (~2-week intervals over 2 years of Bitcoin).        │
 │    • Chronological Non-Leaking Splits:                                                 │
 │      - Train Split:  Time steps t = 1 to 34  (143,551 nodes | 70.4%)                   │
 │      - Val Split:    Time steps t = 35 to 42 (34,144 nodes  | 16.8%)                   │
 │      - Test Split:   Time steps t = 43 to 49 (26,074 nodes  | 12.8% — future unseen)   │
 │    • StandardScaler is fitted EXCLUSIVELY on t ≤ 34 to prevent temporal data leakage.  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 5. SYSTEM ARCHITECTURE — 6 MODULAR BLOCK COMPONENTS

Your mentor requested: **"system architecture split it into different block components."**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORALAML MODULAR SYSTEM ARCHITECTURE                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [BLOCK 1: DATA INGESTION & TEMPORAL GRAPH CONSTRUCTION (O1)]
 ├── Ingests 3 Elliptic CSVs (features, edgelist, classes)
 ├── Continuous Index Remapping: txId ↦ [0, 203768]
 ├── Computes 6 Engineered Topological Features (166 → 172 dimensions)
 ├── Fits StandardScaler strictly on Train Mask (t ≤ 34)
 └── Assembles PyG Data Container: x (203769, 172), edge_index (2, 234355), t, masks
                                │
                                ▼
 [BLOCK 2: LEARNABLE FOURIER TIME ENCODING (O2)]
 ├── Computes Continuous Edge Time Delta: Δt = t_target - t_source
 ├── Sinusoidal Fourier Mapping: Φ(Δt) = [cos(ω·Δt + φ) || sin(ω·Δt + φ)] (d = 128)
 ├── Learnable Frequency Parameters: ω, φ ∈ ℝ⁶⁴ updated via Backpropagation
 └── Yields +8.47% F1 performance gain over fixed-frequency baselines
                                │
                                ▼
 [BLOCK 3: TEMPORAL GRAPH ATTENTION & SHARED EMBEDDINGS (O3 BACKBONE)]
 ├── 2-Layer Temporal Graph Attention Convolution (TGATConv) with 4 attention heads
 ├── Query-Key-Value Attention: q_i = [h_i || Φ(0)] W_Q,  k_j = [h_j || Φ(Δt)] W_K
 ├── Causal Temporal Masking: Only historical neighbors (t_j ≤ t_i) are attended
 └── Produces Unified Latent Node Embeddings: Z ∈ ℝ^(N × 128)
                                │
                                ▼
 [BLOCK 4: MULTI-PATTERN CLASSIFICATION HEADS (O3 HEADS)]
 ├── Head 1: Circular Transfer Detector FFN  → P(circular) ∈ [0,1] (F1 = 0.88)
 ├── Head 2: Layering Chain Detector FFN     → P(layering) ∈ [0,1] (F1 = 0.91)
 ├── Head 3: Smurfing Fan-Out Detector FFN   → P(smurfing) ∈ [0,1] (F1 = 0.94)
 └── Joint Multi-Task Loss: ℒ_total = λ₁ ℒ_circ + λ₂ ℒ_lay + λ₃ ℒ_smurf (with pos_weight)
                                │
                                ▼
 [BLOCK 5: EXPLAINABLE SUBGRAPH GENERATION (O4 XAI)]
 ├── Activates GNNExplainer (200 optimization epochs) on high-risk flagged nodes
 ├── Learns optimal edge mask M_edge ∈ [0, 1] maximizing Mutual Information
 └── Extracts minimal time-ordered evidence subgraph for Suspicious Activity Reports (SAR)
                                │
                                ▼
 [BLOCK 6: OPERATIONAL DASHBOARD & REAL-TIME NODE INSPECTOR]
 ├── FastAPI REST Backend (Port 8000) serving /api/o3/predict/{node_id}
 ├── Interactive Web Dashboard (7 Navigation Tabs, Chart.js 4.4, Dark Mode)
 └── Real-time Node Inspector: Node ID input → Probability bars → Compliance Risk Badge
```

---

# 6. SPECIFIC RESEARCH OBJECTIVES (O1 TO O4)

Your mentor requested: **"objectives should be specific."**

* **Objective 1 (O1) — Temporal Graph Construction & Feature Engineering:**
  * Construct a directed, weighted temporal graph $G = (V, E, T)$ from the Elliptic Bitcoin dataset containing 203,769 nodes and 234,355 directed edges.
  * Engineer 6 topological features (in/out degree, fan-in/out ratios, temporal recency, neighbor time delta) yielding a 172-dimensional feature space.
  * Implement strict chronological data split masks ($t \le 34$ Train, $35 \le t \le 42$ Val, $43 \le t \le 49$ Test) and leakage-free normalization.
* **Objective 2 (O2) — Learnable Fourier Time Encoding:**
  * Design and implement a continuous-time Fourier encoding module $\Phi(\Delta t) \in \mathbb{R}^{128}$ operating over time deltas $\Delta t = t_i - t_j$.
  * Formulate trainable frequency ($\mathbf{\omega} \in \mathbb{R}^{64}$) and phase ($\mathbf{\phi} \in \mathbb{R}^{64}$) parameters updated via gradient descent ($\frac{\partial \mathcal{L}}{\partial \mathbf{\omega}} = -\Delta t \sin(\mathbf{\omega} \Delta t + \mathbf{\phi})$), enabling dynamic adaptation to transaction burst velocities.
* **Objective 3 (O3) — Shared-Embedding Multi-Pattern AML Detection:**
  * Mine ground-truth multi-pattern labels from graph topology using Tarjan’s SCC (Circular), temporal BFS (Layering), and degree thresholding (Smurfing).
  * Build a 2-layer `TGATConv` shared encoder generating unified 128-dimensional latent node representations.
  * Construct 3 parallel classification heads optimized via class-weighted multi-task loss ($\mathcal{L}_{\text{total}} = \sum \lambda_k \text{WBCE}_k$), achieving test F1-scores $> 0.85$ across all three patterns.
* **Objective 4 (O4) — Explainable Subgraph Generation & SAR Reporting:**
  * Integrate `GNNExplainer` to optimize edge masks $M \in [0, 1]^{|E_s|}$ via mutual information maximization for flagged illicit nodes.
  * Generate minimal, time-ordered subgraph evidence paths connecting flagged transactions to suspicious sources for regulatory Suspicious Activity Report (SAR) filing.

---

# 7. SLIDE-BY-SLIDE REVIEW SCRIPT (WHAT TO SAY TO YOUR MENTOR)

Use this script during your review:

* **Slide 1 (Title):** *"Respected guides, my dissertation topic is TemporalAML: Temporal Graph Attention Networks with Learnable Fourier Time Encoding for Explainable Multi-Pattern Anti-Money Laundering in Cryptocurrency Transactions."*
* **Slide 4 (Introduction):** *"Cryptocurrency networks like Bitcoin rely on public, pseudonymous ledgers. Because there are no physical identities on-chain, money laundering has scaled to over $800 Billion globally. Criminals exploit three multi-hop graph patterns: Circular transfers, Layering chains, and Smurfing. Traditional rule-based filters and static GNNs fail because they ignore transaction velocity and temporal ordering."*
* **Slide 5 & 6 (Problem Statement & Motivation):** *"Existing works suffer from three flaws: First, they treat graphs as static, causing future temporal data leakage. Second, they use fixed time encodings that cannot adapt to Bitcoin transaction bursts. Third, they only output binary illicit/licit labels without identifying the specific laundering pattern. TemporalAML solves this by integrating learnable Fourier time encoding directly into graph attention, detecting all three patterns from shared embeddings."*
* **Slide 10 & 11 (Objectives):** *"We have defined 4 specific technical objectives: O1 constructs the 172-dimensional temporal graph with zero data leakage. O2 implements the backpropagated Fourier time encoder. O3 implements the multi-pattern shared TGAT architecture. O4 extracts minimal explainable evidence subgraphs via GNNExplainer."*
* **Slide 13 (Dataset Overview):** *"We utilize the Elliptic Bitcoin dataset with 203,769 nodes and 234,355 edges across 49 time-steps. We engineered 6 topological features to reach 172 dimensions, and enforce strict chronological train/val/test splits with normalization fitted strictly on training steps."*
* **Slide 14–17 (Methodology & Pipeline):** *"Our pipeline processes raw data through Fourier encoding, 2-layer TGATConv with causal temporal masking, and 3 pattern classification heads. We achieve F1-scores of 0.88 on Circular, 0.91 on Layering, and 0.94 on Smurfing with an 8.47% gain from our learnable Fourier module."*
* **Slide 26 (Architecture):** *"The system is structured into 6 modular block components, from graph construction to our FastAPI backend and real-time Node Inspector dashboard for compliance officers."*

---

# 8. TOP 15 TOUGH VIVA & MENTOR QUESTIONS WITH MODEL ANSWERS

### Q1: What is the difference between Bitcoin and Ethereum?
> **Model Answer:** "Bitcoin is a decentralized digital currency operating on the **UTXO (Unspent Transaction Output)** model with a restricted, non-Turing complete scripting language optimized for secure value transfer. Ethereum is a decentralized **World Computer** utilizing an **Account/Balance** model and the **EVM (Ethereum Virtual Machine)**, allowing developers to execute Turing-complete, arbitrary smart contracts."

### Q2: How does a digital signature work in Bitcoin?
> **Model Answer:** "Bitcoin uses **ECDSA over the secp256k1 elliptic curve** ($y^2 = x^3 + 7 \pmod p$). A user signs a transaction hash $z$ using their private key $d_A$ and a random nonce $k$, producing a signature pair $(r, s)$. Anyone with the public key $Q_A = d_A \cdot G$ can mathematically verify that $R' = (z \cdot s^{-1})G + (r \cdot s^{-1})Q_A$ has an $x$-coordinate equal to $r$. This proves the true private key holder authorized the transaction without ever revealing the private key."

### Q3: Why is Bitcoin pseudonymous rather than anonymous?
> **Model Answer:** "Bitcoin is pseudonymous because every transaction is permanently recorded on a public, transparent ledger. Real-world names are replaced by alphanumeric address hashes. Once an address is linked to an identity (e.g., through KYC at an exchange), all past and future transactions connected to that address can be completely traced."

### Q4: What is a Circular Transfer and why is it used in laundering?
> **Model Answer:** "A Circular Transfer is a closed directed cycle ($A \to B \to C \to A$) where funds return to the origin or an affiliated wallet. Criminals use it for wash trading to artificially inflate transaction volume, confuse automated audit tracing software with recursive loops, and cycle funds through pooling nodes."

### Q5: What is Layering and how does it differ from ordinary transaction sequences?
> **Model Answer:** "Layering is a rapid multi-hop sequential chain ($\ge 3$ hops) designed to create distance between dirty money and its crime origin. Unlike ordinary user transactions that occur days or weeks apart, layering hops occur within rapid time intervals to defeat hop-decay heuristic thresholds before authorities can freeze accounts."

### Q6: What is Smurfing / Structuring?
> **Model Answer:** "Smurfing (or structuring) involves splitting large sums into dozens of micro-transactions below mandatory regulatory reporting limits (\$10,000 under FATF/FinCEN rules). A fan-out phase distributes funds to mule accounts, and a subsequent fan-in phase aggregates them into a clean vault."

### Q7: Why do static Graph Neural Networks fail on financial transaction data?
> **Model Answer:** "Static GNNs (like vanilla GCN or GAT) aggregate neighborhood features without considering edge timestamps. This causes **temporal data leakage**—the model aggregates information from future transactions to predict past ones, creating artificially high benchmark scores that fail completely in real-time deployment."

### Q8: Why use Learnable Fourier Time Encoding over fixed Transformer sinusoids?
> **Model Answer:** "Fixed harmonic sinusoids (from NLP Transformers) assume periodic, uniform token distributions. Cryptocurrency transactions exhibit non-stationary, bursty power-law distributions. By making frequency parameters $\omega$ learnable via gradient descent ($\frac{\partial \mathcal{L}}{\partial \omega} = -\Delta t \sin(\omega \Delta t + \phi)$), the attention mechanism automatically tunes to the exact velocity of money laundering hops, giving an **+8.47% F1 improvement**."

### Q9: How did you extract ground-truth labels for the 3 patterns in Objective 3?
> **Model Answer:** "Because the raw Elliptic dataset only labels nodes as binary illicit/licit, we applied graph-theoretic mining restricted to illicit nodes: Tarjan's Strongly Connected Components (SCC $\ge 2$) for Circular transfers, time-monotonic BFS paths ($\text{length} \ge 3$) for Layering chains, and high in/out-degree thresholding ($d \ge 10$) for Smurfing."

### Q10: Why use a shared TGAT embedding rather than 3 separate models?
> **Model Answer:** "Training three separate GNNs triples memory and computational cost ($3 \times 986\text{k}$ parameters) and prevents cross-pattern feature sharing. A shared 2-layer TGAT encoder creates a unified 128-dimensional structural-temporal representation, from which three lightweight classification heads predict pattern probabilities simultaneously in a single forward pass."

### Q11: How do you address the severe 2.2% illicit class imbalance?
> **Model Answer:** "We implement per-pattern **Positive Class-Weighted Binary Cross-Entropy Loss** ($w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 44.8$). This heavily penalizes false negatives on illicit nodes without artificially under-sampling the graph, preserving full message-passing connectivity."

### Q12: What is the role of GNNExplainer in Objective 4?
> **Model Answer:** "GNNExplainer optimizes a continuous edge mask $M \in [0, 1]$ to maximize the mutual information between the full graph prediction and a compact subgraph. This isolates the top-k essential transaction edges responsible for the laundering alert, generating an audit trail for regulatory **Suspicious Activity Report (SAR)** filing."

### Q13: What are the 6 engineered features added in Objective 1?
> **Model Answer:** "We engineered 6 domain features: Out-degree, In-degree, Fan-out ratio, Fan-in ratio, Temporal recency ($\frac{t}{49}$), and Mean neighbor time delta ($|t_v - t_u|$). These expand the feature dimension from 166 to 172, providing immediate structural signals to the encoder."

### Q14: How do you ensure zero temporal leakage during normalization?
> **Model Answer:** "The `StandardScaler` is fitted **exclusively on training split nodes** ($t \le 34$). It transforms validation ($t = 35\dots 42$) and test nodes ($t = 43\dots 49$) using only the mean and variance learned from historical training data."

### Q15: What is the operational output of your system?
> **Model Answer:** "The system is deployed via a FastAPI backend and an interactive compliance dashboard with a **Real-Time Node Inspector**. A compliance officer enters any transaction ID to receive instant multi-pattern probability bars, an automated risk classification (HIGH / MEDIUM / LOW), and explainable transaction evidence."
