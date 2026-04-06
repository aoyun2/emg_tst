# Paper Figure Captions

Generated from:
- Training: `checkpoints\tst_20260405_173725_all`
- Simulation: `artifacts\phys_eval_v2\runs\20260405_230549`

---

## Figure 1 — Pipeline overview
**File:** `figures\paper_native\fig1_pipeline_overview.png`

**Caption:** End-to-end evaluation pipeline. Raw EMG (4 channels) and IMU (6-axis)
signals from 55 Georgia Tech subjects are windowed and fed to a CNN-BiLSTM regressor
trained via 55-fold leave-one-file-out (LOFO) cross-validation. Each held-out
prediction is used to query the MoCapAct motion bank; the nearest clip is replayed
twice in MuJoCo — once unmodified (REF) and once with the right knee overridden
(PRED). The primary outcome is excess instability AUC = PRED - REF.

---

## Figure 2 — Balance-risk metric
**File:** `figures\paper_native\fig2_balance_metric.png`

**Caption:** XCoM-based balance-risk metric. (A) Conceptual diagram: the extrapolated
centre of mass xi = x_CoM + v_CoM / omega_0 projects forward along the velocity
direction; the signed margin d to the base-of-support (BoS) boundary is negative
when xi exits the BoS — the necessary and sufficient condition for loss of dynamic
stability (Hof et al., 2005). (B) XCoM margin time series for both conditions in a
representative trial; PRED pushes the margin more negative for longer. (C) Per-step
risk score r_t combining XCoM margin, trunk tilt, and tilt rate; shaded area = AUC.
Excess AUC = integral of PRED risk minus integral of REF risk.

---

## Figure 3 — Predictive performance
**File:** `figures\paper_native\fig3_prediction_performance.png`

**Caption:** Held-out RMSE across 55 LOFO subject folds. (A) Violin and jittered
strip plot; circle marks the median (6.85 deg). (B) Empirical CDF of the same values.
Mean RMSE = 7.84 deg, SD = 4.33 deg. Cross-fold variability reflects genuine
differences in residual-muscle signal quality across subjects.

---

## Figure 4 — Simulation instability
**File:** `figures\paper_native\fig4_simulation_outcomes.png`

**Caption:** Simulation outcomes across 80 retained trials. (A) Balance-risk
threshold crossing rate: 40% (32/80) of REF trials versus 80% (64/80) of PRED
trials exceeded the instability threshold, a doubling of the high-risk rate.
(B) Paired violin of instability AUC for REF (grey) and PRED (red); connecting
lines show within-trial changes. (C) Histogram of excess instability AUC (PRED - REF);
95% of trials show positive excess (Wilcoxon signed-rank, p < 0.001, mean = 0.208).

---

## Figure 5 — FWL residualization + correlation
**File:** `figures\paper_native\fig5_fwl_correlation.png`

**Caption:** Partial Spearman analysis via Frisch-Waugh-Lovell residualization.
(A) FWL schematic: the theorem guarantees that the coefficient on X in the regression
Y ~ X + Z equals the correlation between the residuals e_X (from X ~ Z) and e_Y
(from Y ~ Z), allowing model RMSE to be isolated from motion-matching confounders.
(B) Raw association: model RMSE vs excess AUC coloured by match quality (rho = -0.166,
p = 0.140). (C) Motion-match RMSE is the stronger predictor (rho = -0.258, p = 0.021).
(D) After FWL residualization on both match controls, the partial Spearman rho
collapses to -0.022 (p = 0.851, df = 76): prediction accuracy carries no independent
association with instability once retrieval quality is controlled.
