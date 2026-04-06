# Paper Figure Captions

Generated from:
- Training: `checkpoints\tst_20260405_173725_all`
- Simulation: `artifacts\phys_eval_v2\runs\20260405_230549`

---

## Figure 1 — Pipeline overview
**File:** `figures\paper_native\fig1_pipeline_overview.png`

**Caption:** End-to-end evaluation pipeline. Raw EMG (4 channels) and inertial
measurement unit (IMU, 6-axis) signals from 55 Georgia Tech biomechanics trials
are preprocessed and windowed into 400-sample (2.0 s) segments. A CNN-BiLSTM
regressor predicts the right-knee included angle from each window; its outputs are
evaluated via 55-fold leave-one-file-out (LOFO) cross-validation, yielding a mean
held-out RMSE of 7.84°. For each held-out window, the predicted knee trajectory is
used to query a MoCapAct motion bank; the nearest matching snippet is then replayed
twice in a MuJoCo humanoid—once unmodified (REF) and once with the right-knee joint
forced to the PRED trajectory. The primary outcome is the area-under-the-curve
difference in a balance-risk heuristic (excess instability AUC = PRED − REF).

---

## Figure 2 — Prediction performance
**File:** `figures\paper_native\fig2_prediction_distribution.png`

**Caption:** Distribution of held-out subject-fold RMSE values across 55 folds of
LOFO cross-validation. (A) Violin and strip plot showing individual fold errors; the
white circle marks the median (6.85°). The dashed and dotted reference lines
correspond to the 10° and 8° thresholds discussed in the literature. (B) Empirical
cumulative distribution function of the same values, showing that 83.6% of folds
fall below 10° and 67.3% fall below 8°. Mean RMSE = 7.84° ± 4.33° (SD).

---

## Figure 3 — Physical simulation outcomes
**File:** `figures\paper_native\fig3_simulation_outcomes.png`

**Caption:** Simulation outcomes across 80 retained trials. (A) Scatter plot of REF
versus PRED simulated knee RMSE; points below the identity line indicate trials where
PRED improved knee tracking (71.3% of trials; Wilcoxon signed-rank, p < 0.001).
(B) Paired violin comparison of instability AUC for the REF and PRED conditions;
thin connecting lines show within-trial changes. (C) Histogram of excess instability
AUC (PRED − REF); 95.0% of trials show positive excess (Wilcoxon one-sided, p < 0.001),
indicating that the knee override consistently increases simulated instability even
when it improves knee tracking.

---

## Figure 4 — Correlation and confounding analysis
**File:** `figures\paper_native\fig4_correlation_confounding.png`

**Caption:** Association between prediction error and simulated instability. (A) Raw
Spearman scatter of model knee RMSE versus excess instability AUC, coloured by
motion-match quality; the association is negative but non-significant (ρ = −0.166,
p = 0.140). (B) The motion-match knee RMSE—a nuisance covariate—explains more
variance in excess instability than the model RMSE does. (C) After Frisch–Waugh–
Lovell (FWL) residualization on both motion-match controls, the partial Spearman
correlation is near zero (ρ = −0.022, p = 0.851, df = 76), indicating that
prediction accuracy per se does not drive the observed instability.

---

## Figure 5 — Representative trial
**File:** `figures\paper_native\fig5_representative_rollout.png`

**Caption:** Deep-dive into representative trial rig_002185 (source file:
gt_data035.npy). (A) Knee-angle time series showing the ground-truth target
(black), the unmodified expert-policy tracking (REF, grey dashed,
RMSE = 15.2°), and the CNN-BiLSTM-overridden tracking
(PRED, blue, RMSE = 4.9°). PRED achieves substantially
lower joint-tracking error. (B) Corresponding balance-risk traces for REF and PRED;
despite the improved knee tracking, the PRED rollout accumulates higher integrated
instability (excess AUC = 0.583), illustrating why
joint-angle accuracy alone is insufficient as a prosthetic evaluation criterion.
