# Paper Figure Notes

These figures were generated from the native-rate publication benchmark:
- training run: `checkpoints\tst_20260405_173725_all`
- simulation run: `artifacts\phys_eval_v2\runs\20260405_230549`

## Figure 1
- file: `figures\paper_native\fig1_pipeline_overview.png`
- use: methodology overview
- caption: Native-rate Georgia Tech pipeline used in the publication benchmark, from GT preprocessing through CNN-BiLSTM prediction, motion matching, MuJoCo rollout, and excess-instability analysis.

## Figure 2
- file: `figures\paper_native\fig2_prediction_distribution.png`
- use: predictive performance / generalization
- caption: Distribution of held-out subject-fold RMSE values across 55 folds. Most folds remain below the 10 deg threshold, with mean test RMSE 7.84 deg and median 6.85 deg.

## Figure 3
- file: `figures\paper_native\fig3_simulation_outcomes.png`
- use: simulation outcomes
- caption: Paired REF versus PRED outcomes across the 80-window simulation benchmark. PRED improves knee tracking on average, but excess instability remains positive overall.

## Figure 4
- file: `figures\paper_native\fig4_correlation_confounding.png`
- use: correlation / confounding
- caption: Raw model RMSE is not significantly associated with excess instability, while motion-match quality carries more signal. After Frisch-Waugh-Lovell residualization on the motion-match controls, the partial association is near zero.

## Figure 5
- file: `figures\paper_native\fig5_representative_rollout.png`
- use: representative case study
- caption: Representative trial `rig_001312` from `gt_data021.npy`. The model improves simulated knee tracking (REF 21.55 deg vs PRED 7.65 deg) even though excess instability remains positive (0.395), illustrating why RMSE alone is not sufficient.
