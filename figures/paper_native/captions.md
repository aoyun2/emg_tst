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
held-out RMSE of 7.84 deg. For each held-out window, the predicted knee trajectory is
used to query a MoCapAct motion bank; the nearest matching snippet is then replayed
twice in a MuJoCo humanoid — once unmodified (REF) and once with the right-knee joint
forced to the PRED trajectory. The primary outcome is the area-under-the-curve
difference in a balance-risk heuristic (excess instability AUC = PRED - REF).

---

## Figure 2 — Prediction performance
**File:** `figures\paper_native\fig2_prediction_distribution.png`

**Caption:** Distribution of held-out subject-fold RMSE values across 55 folds of
LOFO cross-validation. (A) Violin and strip plot showing individual fold errors; the
white circle marks the median (6.85 deg). The dashed and dotted reference lines
correspond to the 10 deg and 8 deg thresholds discussed in the literature. (B) Empirical
cumulative distribution function of the same values, showing that 83.6% of folds
fall below 10 deg and 67.3% fall below 8 deg. Mean RMSE = 7.84 deg ± 4.33 deg (SD).

---

## Figure 3 — Physical simulation outcomes
**File:** `figures\paper_native\fig3_simulation_outcomes.png`

**Caption:** Simulation outcomes across 80 retained trials. (A) Balance-risk threshold
crossing rates: 40% of REF trials (32/80) versus 80% of PRED trials (64/80) exceeded
the instability threshold, indicating that the knee override doubled the crossing rate.
(B) Paired violin comparison of instability AUC for the REF and PRED conditions;
thin connecting lines show within-trial changes. (C) Histogram of excess instability
AUC (PRED - REF); 95% of trials show positive excess (Wilcoxon one-sided, p < 0.001),
indicating that the knee override consistently increases simulated instability.

---

## Figure 4 — Correlation and confounding analysis
**File:** `figures\paper_native\fig4_correlation_confounding.png`

**Caption:** Association between prediction error and simulated instability. (A) Raw
Spearman scatter of model knee RMSE versus excess instability AUC, coloured by
motion-match quality; the association is negative but non-significant (rho = -0.166,
p = 0.140). (B) The motion-match knee RMSE — a nuisance covariate — explains more
variance in excess instability than the model RMSE does (rho = -0.258, p = 0.021).
(C) After Frisch-Waugh-Lovell (FWL) residualization on both motion-match controls,
the partial Spearman correlation is near zero (rho = -0.022, p = 0.851, df = 76),
indicating that prediction accuracy per se does not drive the observed instability.

---

## Figure 5 — Representative trial
**File:** `figures\paper_native\fig5_representative_rollout.png`

**Caption:** Deep-dive into representative trial rig_002185 (source file:
gt_data035.npy; model RMSE = 5.1 deg;
match RMSE = 4.7 deg).
(A) Knee-angle time series showing the expert-policy target from the matched MoCapAct
clip (black), the unmodified humanoid tracking (REF, grey dashed,
RMSE = 15.2 deg vs clip), and the CNN-BiLSTM-overridden tracking
(PRED, blue, RMSE = 4.9 deg vs clip). Both RMSE values
are measured against the same matched clip trajectory; PRED is lower because the PD
controller directly targets that clip, not because it better reconstructs real-world
gait. (B) Corresponding balance-risk traces; despite nominally lower clip-tracking
error, the PRED rollout accumulates higher integrated instability
(excess AUC = 0.583), illustrating that joint-angle
accuracy against a clip is insufficient as a prosthetic evaluation criterion.

---

## Figure 6 — Simulation diagnostics
**File:** `figures\paper_native\fig6_simulation_diagnostics.png`

**Caption:** Auto-generated diagnostic panels for representative trial rig_002185.
(A) Motion-match alignment: predicted knee trajectory (blue) overlaid on the top
MoCapAct candidate, confirming a close initial pose match before rollout. (B) Simulated
knee-angle tracking for REF and PRED conditions throughout the 2.01 s trial window.
(C) Balance-risk trace per timestep; the PRED condition accumulates risk earlier and
at higher magnitude, consistent with the excess instability AUC reported in Figure 5.
