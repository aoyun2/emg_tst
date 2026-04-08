# Paper Rewrite Plan

This file is a repo-faithful rewrite guide for replacing the current transformer/custom-data draft in [Research Paper - Aaron (3).docx](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/Research%20Paper%20%E2%80%94%20Aaron%20%283%29.docx) with the current Georgia Tech + CNN-BiLSTM + MuJoCo pipeline.

Use this together with the `Paper Handoff` section in [README.md](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/README.md).

Important:
- the methodology below should follow the current native-rate `200 Hz` code path
- the benchmark numbers listed here are from the current native-rate publication run
- the canonical training run is `checkpoints/tst_20260405_173725_all`
- the canonical simulation/statistics run is `artifacts/phys_eval_v2/runs/20260406_205003`

## Main Change

The current Word draft describes:
- custom self-recorded EMG + IMU data
- a transformer / TST model
- the older custom-data simulation setup

The current repo actually implements:
- Georgia Tech processed biomechanics data converted into `gt_data*.npy`
- a CNN-BiLSTM last-step regressor
- GT-compatible motion matching and MuJoCo evaluation
- a completed 80-window simulation run with partial-Spearman analysis

So the paper should be revised as a Georgia Tech / CNN-BiLSTM / simulation-correlation study, not as a custom-data transformer study.

## Recommended Title

Pick one of these:

1. `Prediction Error vs. Simulated Balance Risk in Knee-Angle Regression from EMG and Thigh IMU`
2. `Do Lower Knee-Prediction Errors Correspond to Lower Simulated Prosthetic Risk?`
3. `Relating Wearable-Sensor Knee Prediction Error to Physics-Based Balance Risk`

## Recommended Abstract Structure

Use 4 parts:

1. Problem
   - continuous knee-angle prediction from wearable EMG and thigh IMU is often evaluated only with statistical error metrics
   - it is unclear whether lower numerical prediction error corresponds to lower excess instability in simulation

2. Method
   - Georgia Tech processed biomechanics data
   - CNN-BiLSTM trained on preprocessed EMG + thigh IMU
   - subject-holdout benchmark
   - motion matching into MoCapAct
   - MuJoCo `REF` vs `PRED` prosthetic override
   - partial Spearman correlation with motion-match controls

3. Main results
   - held-out test RMSE `7.84 deg`
   - 80 successful simulation windows
   - motion-match knee RMSE below `8 deg` on average
   - raw RMSE vs excess instability non-significant
   - partial correlation after controls remains non-significant

4. Interpretation
   - lower model error did not show a meaningful independent association with excess simulated instability once motion-matching quality was controlled
   - motion-match quality was a stronger driver of risk in this setup

## Section-by-Section Rewrite

### 1. Introduction

Keep:
- prosthetic knee-control motivation
- importance of EMG and IMU for continuous knee estimation
- concern that numerical regression error may not reflect real physical usefulness

Change:
- remove claims that the main experiment is based on your self-recorded dataset
- remove claims that the main model is a transformer
- frame the paper around the question:
  - whether prediction RMSE is independently related to simulation risk

Suggested replacement thesis sentence:

`This study examines whether lower wearable-sensor knee-prediction error corresponds to lower excess simulated instability when the predicted trajectory is injected into a physics-based prosthetic-override pipeline.`

### 1.1 Prior Work

Keep:
- background on classical ML, CNN/LSTM, transformer literature
- simulation motivation

Change:
- do not overstate transformer superiority if the implemented study no longer uses a transformer
- cite transformer work as background, not as the model actually used here
- bring CNN-LSTM / CNN-BiLSTM literature forward because it now matches the implemented method

Suggested framing:

`Although attention-based architectures have shown strong results in wearable biomechanical forecasting, the implemented model in this study is a CNN-BiLSTM chosen for compatibility with the current Georgia Tech benchmark and for stable end-to-end integration with the simulation pipeline.`

### 2. Methodology

This section needs the biggest rewrite.

Replace the old custom-data / transformer subsections with:

1. Data source
   - Georgia Tech processed biomechanics dataset
   - EMG channels: `RRF`, `RBF`, `RVL`, `RMGAS`
   - thigh IMU channels: `RAThigh_ACC{X,Y,Z}`, `RAThigh_GYRO{X,Y,Z}`
   - label source: `knee_angle_r`
   - motion-match thigh proxy: `hip_flexion_r`

2. Preprocessing
   - high-pass EMG at `20 Hz`
   - rectify
   - low-pass at `5 Hz`
   - keep GT angle + IMU at native `200 Hz`
   - resample the filtered EMG envelope onto that same native `200 Hz` timebase
   - 2.0-second windows, `400` samples
   - z-score EMG per recording
   - global feature scaler on training recordings
   - target normalized by `180`

3. Model
   - CNN-BiLSTM last-step regressor
   - `Conv1d(10 -> 32, k=5)`
   - `Conv1d(32 -> 32, k=5)`
   - `BiLSTM(hidden=64, layers=2, bidirectional=True)`
   - `Linear(128 -> 64) -> GELU -> Dropout(0.10) -> Linear(64 -> 1)`
   - `10 ms` forecast because `LABEL_SHIFT = 2` at `200 Hz`

4. Training
   - subject-holdout split for the GT benchmark used here
   - optimizer `Adam`
   - LR `1e-3`
   - weight decay `1e-4`
   - batch size `128`
   - Huber loss with `delta = 5Â°`
   - max `6` epochs, patience `2`

5. Simulation
   - build `samples_dataset.npy` from GT windows
   - motion match each held-out window into the MoCapAct bank using scalar `thigh_knee_d`
   - publication-default matching weights: `knee=1.0`, `thigh=0.0`
   - local refine radius: `30`
   - run `REF` and `PRED` MuJoCo rollouts
   - override only the right knee

6. Statistics
   - predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
   - outcome: `sim.excess.instability_auc_delta`
   - nuisance controls:
     - `match.rmse_knee_deg`
     - `match.rms_thigh_ori_err_deg`
   - rank-transform variables
   - residualize predictor and outcome on controls
   - Pearson correlation on residuals = partial Spearman via Frisch-Waugh-Lovell
   - justify the outcome as `PRED - REF` because the instability trace is heuristic and absolute `PRED` AUC inherits clip difficulty and reference bias
   - define the instability trace as an XCoM-margin-only score rather than a literal fall probability

### 2.x Current Native-Rate Numbers

Training result from [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/checkpoints/tst_20260405_173725_all/metrics_summary.json):
- `n_folds = 55`
- mean `best_val_rmse = 6.05Â°`
- mean `test_rmse = 7.84Â°`
- median `test_rmse = 6.85Â°`
- mean `test_seq_rmse = 7.88Â°`
- mean `test_mae = 6.11Â°`

Simulation result from [summary_metrics_native.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260406_205003/summary_metrics_native.json):
- `80` successful trials
- fixed seed `42`
- mean predictor RMSE across simulation windows: `8.80Â°`
- mean motion-match knee RMSE: `7.93Â°`
- median motion-match knee RMSE: `5.86Â°`
- mean reference instability AUC: `0.819`
- mean predicted instability AUC: `1.019`
- mean excess instability AUC: `0.200`
- mean reference simulated knee RMSE: `12.09Â°`
- mean predicted simulated knee RMSE: `9.60Â°`

Correlation result from [partial_spearman_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260406_205003/analysis/partial_spearman_summary.json):
- `n = 80`
- partial Spearman on excess instability `rho = -0.019`
- `p = 0.867`

Raw and control correlations from [paper_plot_stats_excess.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260406_205003/analysis/paper_plot_stats_excess.json):
- raw RMSE vs excess instability: `rho = -0.168`, `p = 0.136`
- match knee error vs excess instability: `rho = -0.267`, `p = 0.0168`
- match thigh error vs excess instability: `rho = -0.236`, `p = 0.0353`

## Recommended Results Section

Use this structure:

### 3.1 Predictive Performance

Key points:
- the CNN-BiLSTM achieved sub-10Â° held-out RMSE on the GT subject-holdout benchmark
- report the exact mean `7.84Â°` test RMSE and `6.11Â°` MAE
- note that this establishes a model accurate enough for the planned simulation benchmark

Suggested sentence:

`On the current Georgia Tech subject-holdout benchmark, the CNN-BiLSTM achieved a mean held-out test RMSE of 7.84Â° and MAE of 6.11Â°, indicating sub-10Â° knee-angle prediction accuracy on the implemented dataset split.`

### 3.2 Simulation Outcomes

Key points:
- 80 successful windows were evaluated
- `PRED` often improved simulated knee tracking relative to `REF`
- but risk did not necessarily decrease with lower model error

Suggested sentence:

`Across the 80-window simulation run, the model-conditioned rollout had lower mean simulated knee RMSE than the reference rollout (9.60 deg vs. 12.09 deg), yet its mean heuristic instability AUC was higher (1.019 vs. 0.819), indicating that improved local knee tracking did not automatically translate into lower simulated instability.`

### 3.3 Correlation Results

Key points:
- raw RMSE-risk link is weakly negative and non-significant
- after controlling for motion-match quality, the association disappears

Suggested paragraph:

`Using excess instability AUC, defined as the model-conditioned instability AUC minus the reference instability AUC for the matched clip, raw prediction error showed no meaningful monotonic association with outcome (Spearman rho = -0.168, p = 0.136). After controlling for motion-match knee error and thigh-orientation error using a partial Spearman procedure based on the Frisch-Waugh-Lovell theorem, the association remained non-significant (rho = -0.019, p = 0.867).`

## Recommended Discussion Section

Main message:
- sub-10 deg prediction error is achievable on the current GT benchmark
- sub-10 deg motion matching is also achievable on the current GT benchmark
- but prediction RMSE is not a significant predictor of excess instability in the present pipeline
- the instability trace should be framed as a heuristic stability cost, not a literal fall probability

Suggested discussion claims:

1. `The study does not support using RMSE alone as a proxy for physical usefulness in this simulation setting.`
2. `The non-significant partial correlation suggests that once motion-matching quality is accounted for, prediction error contributes limited additional explanatory power for excess instability AUC.`
3. `This indicates that future improvements in the simulator pipeline may need to focus as much on match quality and whole-body contextual consistency as on lowering knee-prediction RMSE.`

## Figures To Insert

Use these files directly:

1. Integrated pipeline and model architecture
   - [fig1_pipeline.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig1_pipeline.png)

2. Representative instability trial
   - [fig2_representative_trial.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig2_representative_trial.png)

3. Prediction performance distribution
   - [fig3_prediction_performance.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig3_prediction_performance.png)

4. Simulation instability outcomes
   - [fig4_simulation_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig4_simulation_instability.png)

5. Correlation and confounding
   - [fig5_fwl_correlation.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig5_fwl_correlation.png)

6. Example replay frames
   - [fig6_simulation_frames.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig6_simulation_frames.png)

7. Motion-matching process
   - [fig7_motion_matching.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig7_motion_matching.png)

Caption notes for all seven:
- [captions.md](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/captions.md)

## What To Delete From The Current Draft

Delete or rewrite any passages that claim:
- the main dataset is your self-recorded EMG/IMU dataset
- the model is an encoder-only transformer or TST
- the main feature set is `43` handcrafted EMG/quaternion features
- the current main study uses the old custom-data non-overlapping 200 Hz window pipeline

Those claims describe an earlier project state, not the current repo.

## What To Keep From The Current Draft

You can keep and adapt:
- the broad prosthetics motivation
- the argument that numerical error alone may not imply functional success
- the motivation for simulation as an intermediate evaluation environment
- the statistical rationale for using motion-match controls

## Caution

Two caveats should stay explicit in the paper:

1. The current repo benchmark uses the publicly accessible GT processed dataset integrated here, which may not be identical to the exact corpus used in the external GT paper.
2. For the GT path, the repo stores a marker-derived `thigh_quat_wxyz`, but the current publication-default matcher is still scalar `thigh_knee_d` because it gave the best knee match RMSE on the exact 80-window held-out pool.
3. The scalar `predicted_fall_risk` in the saved artifacts is a heuristic instability score derived from XCoM-margin dynamics, not a calibrated fall probability. For the paper, describe it accordingly and prefer the excess instability outcome relative to `REF`.

