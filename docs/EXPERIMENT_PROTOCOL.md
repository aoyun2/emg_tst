# Definitive experiment protocol

This document is the concise public protocol for the completed Gait120
residual-fusion and physics experiment. The machine-readable protocols, source
hashes, participant-level results, and full simulation audit are preserved in
`final_submission/Additional_file_2_reproducibility_evidence.zip`.

## Prediction experiment

- Dataset: Gait120 level walking only.
- Development participants: S001–S030.
- Confirmation participants: S031–S120.
- Participant calibration: trials 1–3 for fitting, trial 4 for validation or
  reporting as specified by phase, and trial 5 for the final evaluation.
- Kinematic input: 60 recorded knee-angle frames (600 ms at 100 Hz).
- sEMG input: final 15 recorded causal envelope frames from 12 right-leg
  channels (150 ms at 100 Hz).
- Forecast target: knee angle 100 ms after the final input frame.
- No predictor input or target is interpolated or time-normalized.
- Baseline: ridge regression using knee-angle history.
- Fusion: a second ridge regression predicts the remaining training residual
  from sEMG; the ablation removes only this correction.
- Primary statistical unit: participant.
- Confirmation requirements: positive mean improvement, 95% confidence
  interval above zero, two-sided p ≤ 0.05, and more than half of participants
  improved.

The manuscript reports a conventional paired t-test on participant-level RMSE
differences. Wilcoxon signed-rank and exact sign tests are sensitivity analyses.
The finite sign-randomization calculation used by the original computational
gate remains in the evidence archive for auditability.

## Timing control

The same sEMG history is shifted 500 ms earlier within a continuous trial.
Examples without a complete earlier history are discarded without wraparound
or synthetic samples. Aligned and shifted models are evaluated on identical
target rows.

## Physics panel

- Fixed panel: 80 one-second confirmation trial-5 windows selected in seeded,
  participant-balanced rounds before matching or simulation outcomes.
- Reference bank: 157 MoCapAct snippets from 90 clips whose official
  descriptions indicate unqualified level walking.
- Matching: knee-angle RMSE ranks candidates; right-thigh pitch RMSE is recorded
  separately. No amplitude scaling is allowed.
- Reference screen: the first of at most 12 ranked candidates whose unmodified
  expert completes two deterministic stable replays becomes the reference.
- Controller systems check: at least 8 of 10 fixed windows must pass with zero
  prediction error; 10 of 10 passed.
- Prediction mapping: the sampled Gait120 prediction error is added to the
  realized knee trajectory of the same matched MoCapAct reference.
- Paired conditions: reference, residual fusion, and the same model without the
  sEMG correction use the same snippet, initial state, expert policy, and knee
  controller.
- Accuracy path: 5%, 10%, 20%, 40%, 60%, 80%, and 100% of each participant’s
  eligible training examples; every checkpoint uses the same 80 windows and
  references.
- Numerical outcomes: all 560 checkpoint-window simulations are retained.
- Media: panel indices 0, 11, 22, 33, 44, 55, 66, and 77 were selected before
  results and rendered as representative recordings.

## Physical outcomes

The primary physical outcome is excess-instability AUC: prediction-condition
instability AUC minus the matched unmodified-reference AUC. The FWL analysis
removes ranked knee and thigh matching errors before estimating the remaining
participant-level Spearman association between prediction RMSE and excess AUC.
Balance losses and early terminations remain outcomes; they are not filtered.

The accuracy-path analysis reports the adjacent measured RMSE checkpoints that
bracket the sharp rise in excess AUC and balance losses. It does not convert an
unsampled interval into an exact clinical safety threshold.
