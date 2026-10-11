# AI Detection Markers and Humanizing Transformations

AI classifiers (Turnitin, GPTZero, Copyleaks) identify machine-generated text by detecting low perplexity (predictable word choices) and low burstiness (uniform sentence structures).

---

## 1. Top Banned AI Words and Phrases

| Detected AI Phrasing | Human Natural Alternatives |
| :--- | :--- |
| **"delve into" / "delving"** | examine, investigate, explore, analyze, dissect |
| **"testament to"** | reflects, demonstrates, confirms, underscores |
| **"pivotal role" / "crucial role"** | central mechanism, primary driver, critical factor |
| **"tapestry" / "landscape"** | context, domain, environment, framework |
| **"furthermore" / "moreover" (overused)** | also, additionally, in practice, equally, alongside this |
| **"it is important to remember / note that"** | notably, critically, specifically |
| **"at its core" / "at the heart of"** | fundamentally, principally, by design |
| **"multifaceted" / "myriad"** | varied, diverse, distinct, several |
| **"beacon of" / "game changer"** | benchmark, milestone, breakthrough |
| **"in conclusion" / "to summarize"** | overall, taken together, ultimately |
| **"not only... but also..." (repetitive)** | both... and..., combines X with Y |

---

## 2. Structural Patterns That Flag AI

1. **The Classic 3-Part AI Paragraph**:
   - *Sentence 1 (Broad assertion)* $\to$ *Sentence 2 (Elaboration)* $\to$ *Sentence 3 (Summary wrap-up)*.
   - **Fix**: Break the pattern by introducing questions, contrasting technical friction, or varying paragraph lengths.
2. **Symmetric Lists**:
   - AI generates lists where every bullet point has the exact same syllable count and grammatical structure (e.g., *Verb + Object + Prepositional phrase*).
   - **Fix**: Asymmetrize list items, mix sentence structures, or convert lists into flowing explanatory paragraphs.
3. **Monotonous Sentence Length (15–20 words)**:
   - AI outputs sentences of nearly identical length.
   - **Fix**: Inject 4–6 word sentences right after 30-word complex sentences.

---

## 3. High-Perplexity Humanizing Techniques

1. **Clause Inversion**:
   - AI: *"We evaluate the model on the test split because it contains unseen temporal steps."*
   - Human: *"Because the test split spans subsequent chronological intervals, evaluating the model here isolates forward-looking generalization."*
2. **Active Voice with Concrete Actors**:
   - AI: *"It can be observed from the experimental results that improvements were achieved."*
   - Human: *"The empirical results show an 8.4% gain in detection recall."*
3. **Contextual Friction & Qualification**:
   - Humans naturally include trade-offs, edge cases, and engineering friction (e.g., *"While memory consumption scales quadratically with degree density, causal masking mitigates this overhead in sparse transaction topologies"*).
