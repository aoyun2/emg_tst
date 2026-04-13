# AP Research Rubric Alignment

Official rubric source used for this rewrite: [ap25_research_scoring_guidelines.pdf](C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/ap25_research_scoring_guidelines.pdf)

## Goal

This revision was written to align with the official AP Research academic paper criteria for a top-score submission. The paper was rebuilt around the current Georgia Tech + CNN-BiLSTM + MoCapAct/XCoM project rather than the older custom-data transformer version.

## Alignment to Score-5 Expectations

### 1. Focused and justified inquiry
- The paper now asks one narrow question: whether lower wearable knee-angle RMSE corresponds to lower excess instability in paired simulation after motion-match quality is controlled.
- The gap is explicit: knee-angle models are usually evaluated with offline accuracy metrics, but those metrics do not necessarily reveal whole-body physical usefulness.

### 2. Detailed and replicable method
- The rewritten methods section defines the dataset, signal channels, preprocessing, windowing, forecast horizon, architecture, optimizer, loss, validation strategy, simulation setup, instability metric, and statistics in enough detail to reproduce the benchmark.
- The paper explicitly states that the benchmark is leave-one-file-out, not subject-holdout.
- The paper defines excess instability AUC, motion-match controls, and the rank-based Frisch-Waugh-Lovell partial Spearman procedure step by step.

### 3. Rich analysis leading to a new understanding
- The results section reports both the strong prediction benchmark and the null partial-correlation result.
- The discussion does not overclaim. Instead, it explains the specific new understanding supported by the evidence: sub-10° predictive accuracy did not independently correspond to lower excess instability in this benchmark.
- Limitations and future work are stated explicitly rather than hidden.

### 4. Clear communication
- The writing style follows the stronger early-introduction tone of the original paper rather than the later AI-generated sections.
- Definitions are embedded where the terms first appear.
- The manuscript is organized around the AP expectation that claims, method choices, and interpretations must be justified rather than merely asserted.

### 5. Citation discipline
- In-text citations are used frequently after claims, methodological decisions, and interpretive statements that require support.
- The revised package includes both a textual reference list in the paper and a machine-verifiable BibTeX file with more than 40 entries.

## Current Draft Word Count

- Main body word count estimate: **4942**
- This count excludes the references list and figure captions.
