# Analysis protocol summary

This document summarizes the completed Gait120 residual-fusion and paired
simulation experiment. Machine-readable protocols and participant-level
results are distributed with the paper; generated evidence is not tracked in
this repository.

## Prediction experiment

- Dataset: Gait120 level walking only.
- Development cohort: S001–S030. Trials 1–3 supplied scaling values and pooled
  fitting data, trial 4 selected hyperparameters, and trial 5 was held out.
- Confirmation cohort: S031–S120. The fixed model form and hyperparameters were
  refitted with pooled trials 1–3, trial 4 was secondary, and trial 5 was the
  primary test.
- Model fitting used equal total sample weight per participant. This is a
  later-trial, participant-calibrated design, not transfer to uncalibrated
  participants.
- Kinematic input: 60 recorded knee-angle frames (600 ms at 100 Hz).
- sEMG input: the final 15 recorded causal envelope frames from 12 right-leg
  channels (150 ms at 100 Hz).
- Forecast target: knee angle 100 ms after the final input frame.
- No predictor input or target was interpolated or time-normalized.
- Baseline: ridge regression using knee-angle history.
- Residual fusion: a second ridge regression estimated the remaining training
  residual from sEMG; the ablation removed only this correction.
- Primary statistical unit: participant.
- Primary summary: two-sided paired t-test and Student-t confidence interval on
  participant RMSE without sEMG minus RMSE with residual fusion. Wilcoxon and
  exact sign tests were sensitivity analyses.

## Timing control

The same sEMG history was shifted 500 ms earlier within each continuous trial.
Rows without a complete earlier history were excluded, and aligned and shifted
models were evaluated on identical target rows.

## Paired simulation

- Fixed panel: 80 one-second confirmation trial-5 windows selected in seeded,
  participant-balanced rounds before motion matching or simulation.
- Reference bank: 157 MoCapAct snippets from 90 clips described as level
  walking.
- Matching: knee-angle RMSE ranked candidates. Right-thigh pitch RMSE was
  calculated afterward as a match-quality covariate. No amplitude scaling was
  allowed.
- Prediction mapping: the sampled Gait120 prediction error was added to the
  realized knee trajectory of the same matched MoCapAct reference.
- Paired conditions shared the reference snippet, initial state, expert policy,
  warm-up, and knee controller.
- Accuracy range: 5%, 10%, 20%, 40%, 60%, 80%, and 100% of each participant's
  eligible fitting examples. Every checkpoint used the same 80 windows and
  references.
- All 560 planned prediction rollouts were retained.

## Simulation outcomes

The primary outcome was excess-instability area under the curve (AUC): the
prediction-condition value of a study-specific XCoM-based index minus the
matched unmodified-reference value. The bounded index has not been validated
as a clinical stability or fall-risk measure.

The adjusted association analysis removed ranked knee and thigh matching
errors before calculating the remaining participant-level Spearman
association between prediction RMSE and excess AUC. A high-instability flag was
reported only as a descriptive algorithmic outcome; it was not an observed
fall.

The accuracy-range analysis compares measured calibration checkpoints. It does
not isolate RMSE as a causal exposure or convert the unsampled interval between
checkpoints into a clinical safety threshold.
