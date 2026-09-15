# TemporalAML

TemporalAML is a temporal Graph Neural Network (temporal-GNN) research and detection framework developed for anti-money laundering across cryptocurrency transaction graphs. The architecture incorporates learnable Fourier time encoding with Temporal Graph Attention Networks (TGAT) to simultaneously identify circular transfers, layering chains, and smurfing typologies on dynamic transaction streams. Designed for explainability and regulatory auditing, the platform provides end-to-end capabilities spanning temporal graph construction, multi-task AML pattern classification, and automated Suspicious Activity Report (SAR) narrative generation.

## License

This repository's code is licensed under the [MIT License](LICENSE).

The Elliptic Bitcoin dataset used to train and evaluate the models is **not** included in this repository (`data/` is git-ignored) and is distributed separately under its own terms — **CC BY-NC-ND 4.0** (non-commercial, no derivatives) on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). That license is independent of this repository's MIT license and does not permit commercial use or redistribution of derivative works. If you download the dataset or use it to train models, review its license yourself and note that any model checkpoints or outputs derived from it may carry the same non-commercial restriction.
