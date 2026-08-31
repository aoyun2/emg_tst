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

### sEMG negative controls

The ablation alone cannot separate real sEMG information from the extra fitted
capacity of a second regression stage. Three surrogate conditions repeat the
whole confirmation fit with the sEMG replaced by a signal that keeps its
statistics but loses its correspondence with the knee:

- `circular_shift`: each participant-trial sEMG block is rotated against its own
  kinematics, preserving amplitudes, spectra, and cross-channel structure.
- `participant_swap`: each participant receives another participant's sEMG.
- `phase_randomized`: each channel is replaced by a surrogate with an identical
  power spectrum and randomized Fourier phases.

The kinematic stage is shared across all four conditions, so any difference is
attributable to sEMG content.

Attribution is judged by a paired within-participant contrast: for each
participant, the improvement with recorded sEMG minus the improvement with that
surrogate, summarized with the same bootstrap interval and sign-flip
randomization test as the primary ablation. The improvement is attributed to
sEMG content when the recorded signal clears its gate and beats every surrogate
on that contrast.

Requiring each surrogate to independently fail a significance threshold would be
the wrong rule. Walking is periodic with a cycle close to one second, so a
circular shift cannot fully break the correspondence between sEMG and knee
angle: whatever offset it lands on is still a phase of the same repeating cycle.
That surrogate sets a floor rather than a zero, and the observed residual effect
in it is expected. The participant-swap and phase-randomized surrogates, which do
not rely on a time offset, are the ones expected to reach zero.

## Timing control

The same sEMG history was shifted 500 ms earlier within each continuous trial.
Rows without a complete earlier history were excluded, and aligned and shifted
models were evaluated on identical target rows.

## Paired simulation

- Fixed panel: 80 one-second confirmation trial-5 windows selected in seeded,
  participant-balanced rounds before motion matching or simulation.
- Reference bank: the whole MoCapAct expert zoo, roughly 2589 snippets. No
  motion label restricts the candidates, since the released snippets carry
  none, so the matched references are not confined to walking. The reported run
  matched the 80 windows to 32 distinct snippets.
- Matching: knee-angle RMSE ranked candidates. Right-thigh pitch RMSE was
  calculated afterward as a match-quality covariate. No amplitude scaling was
  allowed.
- Prediction mapping: the model's predicted knee angle is the tracking target of
  the PD override on the right knee actuator, mapped into the matched clip's
  knee convention by the constant offset and sign the match established. The
  simulation is driven by the model output itself and never reads the recorded
  future target.
- Paired conditions shared the reference snippet, initial state, expert policy,
  and warm-up. They do not share the knee actuator: only the prediction
  condition retunes it, and the reference keeps the walker's native one. The
  reference condition ran the unmodified
  expert.
- Accuracy range: checkpoints sampled along the gradient-descent training path of
  the same model, from a zero initialization that predicts the participant mean
  down to the closed-form confirmation fit, which is the terminal checkpoint.
  Checkpoints are placed to tile the achieved training-RMSE range rather than
  the step index, because descent crosses the high-error region in few steps.
  Every checkpoint used the same 80 windows and references.

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

The accuracy-range analysis compares measured training checkpoints. It does not
isolate RMSE as a causal exposure or convert the unsampled interval between
checkpoints into a clinical safety threshold.

Three association views are reported. `per_checkpoint` repeats the adjusted
analysis at each accuracy level and carries the RMSE spread available to it, so
a near-zero association on a near-zero spread is not read as evidence of no
effect. `within_window` correlates each window's own accuracy against its own
excess instability across checkpoints and combines the results with a Fisher-z
one-sample test, which removes matched motion, snippet, and initial state as
sources of variation. `pooled` uses every checkpoint-window pair with a
participant-level cluster bootstrap, since each window recurs once per
checkpoint and several participants contribute two windows.

The drop-off estimate is a descriptive two-segment fit of the per-checkpoint
association against mean accuracy. It locates where the sampled association
changes; it is not a changepoint test with its own error rate.
