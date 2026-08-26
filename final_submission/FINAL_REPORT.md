# Final prosthetic-regression report

Date: 26 August 2026

## Outcome

The replacement predictor met the required sEMG ablation criterion in a
separate confirmation group, and the complete prediction-to-physics experiment
finished without removing any planned numerical result.

- Confirmation participants: 90 (S031-S120)
- Forecast horizon: 100 ms
- RMSE with residual fusion: 4.748 degrees
- RMSE without sEMG: 5.093 degrees
- Mean paired improvement: 0.345 degrees
- 95% Student-t interval: 0.252 to 0.438 degrees
- Two-sided paired t-test: t(89)=7.39, p=7.50e-11
- Standardized paired effect: Cohen's dz=0.78
- Participants improved: 70 of 90

The final manuscript uses the conventional paired t-test because its null
hypothesis is simply that the mean within-participant improvement is zero. The
participant differences did not show material non-normality (Shapiro-Wilk
W=0.984, p=0.358). Wilcoxon signed-rank (p=1.53e-9) and exact sign
(p=1.14e-7) sensitivity analyses reached the same conclusion.

The 500-ms timing control also improved prediction. The manuscript therefore
describes sEMG as carrying incremental predictive information, but does not
claim that the gain was caused specifically by correctly timed pre-activation.

## Motion matching and physics

- Fixed level-walking windows: 80
- Stable references found: 80 of 80
- Unique MoCapAct snippets: 29
- Mean knee matching RMSE: 4.317 degrees
- Mean right-thigh pitch RMSE: 3.630 degrees
- Zero-error controller check: 10 of 10 passed
- Numerical simulations: 560 of 560 completed
- Complete representative recordings: 8, selected before outcomes

At the full-data checkpoint, residual fusion and the same model without sEMG
each produced two balance losses. The mean paired reduction in excess-instability
AUC favoring fusion was -0.0012 (95% Student-t interval -0.0105 to 0.0081;
paired t(64)=-0.25, p=0.801).
The confirmed prediction improvement therefore did not produce a detectable
reduction in the simulated physical outcome.

## RMSE and physical behavior

The raw participant-level Spearman association between RMSE and excess
instability was -0.117 (p=0.355). After ranked knee and thigh matching errors
were removed, partial Spearman rho was -0.145 (95% interval -0.370 to 0.082;
permutation p=0.248).

The seven-checkpoint accuracy path showed a marked deterioration between the two
least accurate checkpoints. At 13.41 degrees RMSE, three of 80 windows lost
balance; at 16.29 degrees RMSE, 32 of 80 windows lost balance. These adjacent
observations directly bracket the physical deterioration to 13.41--16.29
degrees, a 2.89-degree interval.

The manuscript reports 13.41--16.29 degrees as an observed transition interval,
not as one exact safety threshold. No checkpoint was simulated inside that
2.89-degree interval, so a narrower numerical boundary would go beyond the
measured data.

## Manuscript and visual audit

The manuscript was rebuilt as an editable Springer Nature/BMC Biomedical
Engineering LaTeX project using the official December 2024 class and Vancouver
bibliography style. It has a structured abstract, line numbers, double spacing,
short figure titles, complete declarations, and an availability statement. Its
section order now follows the approved prior draft: Introduction and its three
literature subsections; Methodology and its seven original subsections; Results;
Discussion, Limitations, and Future Work; and Conclusion. The only added Results
subsection reports the newly requested training-range experiment.

The original XCoM concept figure was retained. The data, workflow, model, and
result figures were rebuilt in the signal-based gray, blue, and red visual style
of the earlier paper, and the simulation figures were regenerated from the
final V4.1 recordings. The revised manuscript contains 13 figures and one table,
compared with 11 figures and no tables in the approved prior draft. The restored
result visuals include participant-level ablation, motion-matching quality,
three correlation panels, RMSE across all seven training-data fractions,
participant-level physics outcomes, the observed accuracy transition, and
frames and trajectories from the actual simulations. The 34-page referee PDF
was rendered at page level and every page was inspected. No clipped text,
overlapping arrows or labels, missing references, broken glyphs, or unreadable
figures were found. The Overleaf archive was extracted into a clean directory
and compiled again successfully before delivery.

## Comparison with the approved prior draft

The revised paper preserves the prior study's section order, conceptual
explanations, and central sequence: knee-angle prediction, motion matching,
paired MuJoCo evaluation, XCoM-based instability, and the relationship between
prediction error and physical behavior. The introductory simulation and balance
explanations were retained where they remained accurate. The main replacement
is the prediction experiment. The earlier custom CNN--BiLSTM and 10-ms horizon
were replaced by a participant-calibrated 100-ms residual-fusion model whose
sEMG contribution was confirmed in 90 participants who did not influence model
selection. Methods and result paragraphs were changed only where the Gait120
data, predictor, corrected simulation implementation, statistics, or numerical
outcomes were different from the prior draft.

The prior draft was easier to follow than several intermediate revisions because
it explained the study in ordinary research prose and used figures to carry the
technical sequence. The final revision returns to that approach while adding
the detail needed for replication: explicit research questions, native signal
processing, participant separation, model equations, MoCapAct reference
selection, paired simulation controls, uncertainty intervals, and limitations.
It also adds more direct evidence than the prior draft rather than replacing the
visual record with abbreviated diagrams.

## Peer-review and AP Research rubric audit

As a journal manuscript, the paper is ready for editorial submission after the
author supplies a correspondence email and confirms the affiliation shown on
the title page. Its strongest points are the untouched 90-participant
confirmation, participant-level inference, transparent negative physics result,
model-blind motion selection, stable reference checks, actual simulation media,
and complete derived evidence. These features should make the methods and result
traceable to a reviewer.

The main review risks are scientific rather than presentational. The confirmed
sEMG gain is small; the 500-ms timing control does not support a specifically
timed pre-activation interpretation; all Gait120 participants were healthy adult
males; the predictor requires within-wearer calibration; only level walking was
tested; MuJoCo instability is not a clinical fall probability; some windows
shared a MoCapAct expert snippet; and the data locate a deterioration interval
rather than one clinical RMSE threshold. These points are disclosed rather than
hidden. Reviewers may also question the custom residual-fusion model and the
clinical reach of an offline counterfactual simulation. The paper should answer
those objections by keeping its claims limited to the evaluation method and the
observed experiment.

Using the AP Research scoring rubric as a writing-quality lens, the revision now
has a narrow question carried through the method and conclusion, places prior
work in conversation to identify a gap, provides a replicable and justified
method, connects every main claim to direct evidence, explains limitations and
implications, and uses discipline-appropriate figures. Those are the features
expected of the upper rubric levels. It is not formatted as an AP submission and
should not be treated as one; the rubric was used only to audit argument,
clarity, evidence, and organization for an educated non-specialist reader.

## Submission files

- `Prosthetic_Regression_BMC_Overleaf.zip`: editable manuscript project
- `main.pdf`: visually checked compiled manuscript
- `Additional_file_1_simulation_recordings.zip`: eight H.264 recordings
- `Additional_file_2_reproducibility_evidence.zip`: protocols, participant-level
  results, complete physics summary, statistical audit, and relevant source code
- `cover_letter.txt`: journal cover letter
- `SHA256SUMS.txt`: integrity hashes for the delivered files

## One author detail still requires confirmation

No institutional affiliation or corresponding email was supplied. The project
therefore uses "Independent researcher, United States" and flags this in its
README. Confirm or replace that affiliation and add the preferred corresponding
email before uploading the submission.
