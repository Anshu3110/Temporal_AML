# TemporalAML: GNNs & Learnable Fourier Time Encoding for Anti-Money Laundering

Welcome to your dissertation guidebook. This comprehensive academic and practical reference is designed to guide you step-by-step through the research, mathematical foundations, pipeline architecture, algorithmic pseudocode, and coding implementation of your M.Tech dissertation:

> **TemporalAML: Temporal Graph Attention Networks for Explainable Multi-Pattern Anti-Money Laundering in Cryptocurrency Transactions**

---

## Guidebook Chapters

This guidebook is organized into 12 structured chapters, written as standalone textbooks in your workspace:

1. **[Chapter 1: Introduction](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_1_introduction.md)**: Contextualizes cryptocurrency AML, the specific laundering patterns (Circular Transfer, Layering, Smurfing), and why Graph Neural Networks (GNNs) and temporal modeling are required.
2. **[Chapter 2: Mathematical Foundations](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_2_mathematical_foundations.md)**: Thorough derivations and step-by-step numerical/visual explanations of Graph theory, Message Passing, Attention, TGAT, Learnable Fourier Encoding, class imbalance metrics, and GNNExplainer.
3. **[Chapter 3: Dataset Understanding](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_3_dataset_understanding.md)**: Break down of the Elliptic Bitcoin Dataset, features, labels, preprocessing pipelines, and handling class imbalances.
4. **[Chapter 4: Project Architecture](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_4_project_architecture.md)**: Visual mapping and explanation of the end-to-end pipeline, from raw data to the final explainable alerts.
5. **[Chapter 5: Objective-wise Implementation](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_5_objective_wise_implementation.md)**: Direct strategy and PyTorch Geometric implementation design for objectives O1-O5, including GNNExplainer and compliance reporting.
6. **[Chapter 6: Algorithms (Pseudocode)](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_6_algorithms.md)**: Rigorous pseudocode for all primary routines (sampling, encoding, forward propagation, explainability, training/test loops).
7. **[Chapter 7: Implementation Roadmap](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_7_implementation_roadmap.md)**: A concrete 9-week timeline from data ingestion to thesis writing.
8. **[Chapter 8: Coding Guide](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_8_coding_guide.md)**: Directory structures, modular components, logging, model tracking, and coding standards.
9. **[Chapter 9: Research & Viva Questions](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_9_research_questions.md)**: Over 100 actual viva/defense questions with rigorous, peer-reviewed level responses.
10. **[Chapter 10: Paper Writing Guide](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_10_paper_writing_guide.md)**: Systematic guide to converting this dissertation into a high-impact Q1 journal article.
11. **[Chapter 11: Common Mistakes & Pitfalls](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_11_common_mistakes.md)**: Pitfalls in graph construction, temporal leakage, evaluation design, and how to avoid them.
12. **[Chapter 12: Future Scope & Advanced Horizons](file:///Users/apple/Desktop/TemporalAML/guidebook/chapter_12_future_scope.md)**: Moving beyond TGAT to Temporal Transformers, Graph Foundation Models, LLM-based GraphRAG, and real-time streaming architectures.

---

## Dissertation Objectives Map

| Objective | Description | Respective Chapters |
| :--- | :--- | :--- |
| **O1** | Construct a temporal transaction graph from the Elliptic Bitcoin Dataset | Chapter 3, Chapter 5 (O1), Chapter 6 |
| **O2** | Design and implement Learnable Fourier Time Encoding inside TGAT attention | Chapter 2 (Fourier), Chapter 5 (O2), Chapter 6 |
| **O3** | Detect Circular Transfers, Layering Chains, and Smurfing using shared TGAT embeddings | Chapter 1 (Patterns), Chapter 5 (O3), Chapter 6 |
| **O4** | Generate explainable time-ordered transaction subgraphs using GNNExplainer | Chapter 2 (Explainability), Chapter 5 (O4), Chapter 6 |
| **O5** | Compare the proposed model with GCN, GraphSAGE, LSTM-GNN, TGAT, and rule-based baselines | Chapter 5 (O5), Chapter 7, Chapter 8 |

---

## How to Use This Guidebook

1. **Read Chronologically**: If you are new to Temporal GNNs, start with **Chapter 1** and **Chapter 2**.
2. **Consult While Coding**: Use **Chapter 5** and **Chapter 6** alongside your editor to write the implementation files.
3. **Prepare for Defense**: Go through the 100+ questions in **Chapter 9** in the weeks leading up to your viva.
4. **Draft Your Paper**: Follow **Chapter 10** to write your manuscript sections as you run experiments.
