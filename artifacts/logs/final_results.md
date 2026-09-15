# 🔬 TemporalAML Comprehensive Benchmark Results

**Evaluation Dataset:** Elliptic Bitcoin Dataset — Test Split (Time Steps 41–49)
**Test Set Size:** 9,973 labeled transactions (524 Illicit, 9,449 Licit)

## 1. Overall Illicit Class Detection Performance

| Model                           |   F1-Score |   Precision |    Recall |   AUC-ROC |     AUPRC |
|:--------------------------------|-----------:|------------:|----------:|----------:|----------:|
| RuleBasedHeuristic              |  0.121315  |   0.0651078 | 0.887405  |  0.653866 | 0.0753619 |
| GCN (2-layer)                   |  0.186896  |   0.105002  | 0.849237  |  0.777366 | 0.123105  |
| GraphSAGE (2-layer)             |  0.172949  |   0.096687  | 0.818702  |  0.795829 | 0.280002  |
| LSTM-GNN (GCN+LSTM)             |  0.225406  |   0.150318  | 0.450382  |  0.757268 | 0.126451  |
| Static-TGAT (Fixed Fourier)     |  0.0366492 |   0.0209353 | 0.146947  |  0.353182 | 0.0383131 |
| TemporalAML (Ablation: No Time) |  0.0366425 |   0.0235875 | 0.0820611 |  0.29826  | 0.038274  |
| TemporalAML (Learnable Fourier) |  0.0350136 |   0.020796  | 0.110687  |  0.357474 | 0.0375897 |

---

## 2. TemporalAML Per-Pattern Typology Performance

| Pattern Typology     |   F1-Score |   Precision |   Recall |   AUC-ROC |     AUPRC |
|:---------------------|-----------:|------------:|---------:|----------:|----------:|
| Circular Transfer    |   0        |    0        | 0        |  0.5      | -0        |
| Layering Chain       |   0.385634 |    0.808594 | 0.253194 |  0.577193 |  0.799636 |
| Smurfing Structuring |   0.408867 |    0.264753 | 0.897297 |  0.961841 |  0.551928 |

---

## 3. Statistical Significance Audit: TemporalAML vs Static-TGAT

- **Method:** Paired Bootstrap Resampling (1,000 iterations)
- **Observed F1 (TemporalAML):** `0.0350`
- **Observed F1 (Static-TGAT):** `0.0366`
- **Mean F1 Delta (ΔF1):** `-0.0016`
- **95% Confidence Interval on ΔF1:** `[-0.0072, +0.0043]`
- **Empirical p-value (H₀: ΔF1 ≤ 0):** `0.7010`
- **Significance Verdict:** **Difference not strictly bounded above zero at α=0.05**
