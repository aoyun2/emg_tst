# Paper Figure Captions

---

## Figure 1 — Pipeline
**File:** `figures\paper_native\fig1_pipeline.png`

End-to-end evaluation pipeline. EMG (4 channels) and IMU (6-axis) signals from 55
Georgia Tech subjects are windowed and fed to a CNN-BiLSTM regressor evaluated by
55-fold LOFO cross-validation. Each prediction drives a motion-matching query into
the MoCapAct expert library; the retrieved clip is replayed twice in MuJoCo — once
unmodified (REF) and once with the right knee overridden by the CNN prediction (PRED).
Primary outcome: excess instability AUC = PRED - REF.

---

## Figure 2 — Representative trial
**File:** `figures\paper_native\fig2_representative_trial.png`

One simulation trial (rig_001884) illustrating the instability outcome. (A) XCoM
margin over 2.01 s; the dashed line marks the base-of-support boundary
(margin = 0 m). REF oscillates mildly; PRED drives the XCoM persistently outside
the BoS (minimum -1.55 m), indicating mechanically unstable gait per Hof et al.
(2005). (B) Per-step instability score r(t) with AUC regions shaded; integrating
PRED - REF gives the excess AUC reported in the results.

---

## Figure 3 — Prediction performance
**File:** `figures\paper_native\fig3_prediction_performance.png`

Held-out RMSE for each of the 55 LOFO subject folds, sorted smallest to largest.
Blue bars indicate folds at or below 10 deg; red bars exceed 10 deg. Vertical lines
mark the mean (7.84 deg, dashed) and median (6.85 deg, dotted).

---

## Figure 4 — Simulation instability
**File:** `figures\paper_native\fig4_simulation_instability.png`

Simulation outcomes across 80 trials. (A) Paired scatter of REF vs PRED instability
AUC; points above the diagonal (95% of trials) indicate the override increased
integrated balance risk. (B) Histogram of excess AUC (PRED - REF); 95% of values
are positive (Wilcoxon, p < 0.001, mean = 0.208).

---

## Figure 5 — FWL correlation analysis
**File:** `figures\paper_native\fig5_fwl_correlation.png`

Partial Spearman analysis via Frisch-Waugh-Lovell (FWL) residualization: regress
predictor X (model RMSE) and outcome Y (excess AUC) separately on controls Z
(match quality), then compute Spearman r of residuals. (A) Raw scatter coloured by
match RMSE: rho = -0.166, p = 0.140. (B) Match RMSE is a stronger predictor
(rho = -0.258, p = 0.021) — the dominant confound. (C) After FWL residualization
the partial rho collapses to -0.022 (p = 0.851, df = 76): model accuracy carries
no independent association with instability.
