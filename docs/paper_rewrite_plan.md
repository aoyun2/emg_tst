# Paper Rewrite Plan

This file is a repo-faithful rewrite guide for replacing the current transformer/custom-data draft in [Research Paper — Aaron (3).docx](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/Research%20Paper%20%E2%80%94%20Aaron%20%283%29.docx) with the current Georgia Tech + CNN-BiLSTM + MuJoCo pipeline.

Use this together with the `Paper Handoff` section in [README.md](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/README.md).

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
   - it is unclear whether lower numerical prediction error corresponds to safer physical behavior in simulation

2. Method
   - Georgia Tech processed biomechanics data
   - CNN-BiLSTM trained on preprocessed EMG + thigh IMU
   - subject-holdout benchmark
   - motion matching into MoCapAct
   - MuJoCo `REF` vs `PRED` prosthetic override
   - partial Spearman correlation with motion-match controls

3. Main results
   - held-out test RMSE `8.96°`
   - 80 successful simulation windows
   - raw RMSE-risk association weakly positive
   - partial correlation after controls approximately zero

4. Interpretation
   - lower model error did not show a meaningful independent association with simulated balance-risk AUC once motion-matching quality was controlled
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

`This study examines whether lower wearable-sensor knee-prediction error corresponds to lower simulated balance risk when the predicted trajectory is injected into a physics-based prosthetic-override pipeline.`

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
   - resample GT branch to `100 Hz`
   - 2.0-second windows, `200` samples
   - z-score EMG per recording
   - global feature scaler on training recordings
   - target normalized by `180`

3. Model
   - CNN-BiLSTM last-step regressor
   - `Conv1d(10 -> 32, k=5)`
   - `Conv1d(32 -> 32, k=5)`
   - `BiLSTM(hidden=64, layers=2, bidirectional=True)`
   - `Linear(128 -> 64) -> GELU -> Dropout(0.10) -> Linear(64 -> 1)`
   - one-sample-ahead forecast because `LABEL_SHIFT = 1`

4. Training
   - subject-holdout split for the GT benchmark used here
   - optimizer `Adam`
   - LR `1e-3`
   - weight decay `1e-4`
   - batch size `128`
   - Huber loss with `delta = 5°`
   - max `6` epochs, patience `2`

5. Simulation
   - build `samples_dataset.npy` from GT windows
   - motion match each held-out window into the MoCapAct bank using scalar thigh-pitch proxy plus knee dynamics
   - run `REF` and `PRED` MuJoCo rollouts
   - override only the right knee

6. Statistics
   - predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
   - outcome: `sim.pred.balance_risk_auc`
   - nuisance controls:
     - `match.rmse_knee_deg`
     - `match.rms_thigh_ori_err_deg`
   - rank-transform variables
   - residualize predictor and outcome on controls
   - Pearson correlation on residuals = partial Spearman via Frisch-Waugh-Lovell

### 2.x Exact Numbers To Use

Training result from [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/gt_full_subject_holdout/all/metrics_summary.json):
- `n_train = 88,298`
- `n_val = 1,802`
- `n_test = 9,010`
- `best_val_rmse = 21.49°`
- `test_rmse = 8.96°`
- `test_seq_rmse = 8.96°`
- `test_mae = 6.11°`

Simulation result from [summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260404_165818/summary.json):
- `80` successful trials
- fixed seed `42`
- mean predictor RMSE across simulation windows: `8.63°`
- mean reference balance-risk AUC: `0.677`
- mean predicted balance-risk AUC: `0.891`
- mean reference simulated knee RMSE: `12.26°`
- mean predicted simulated knee RMSE: `9.82°`

Correlation result from [partial_spearman_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260404_165818/analysis/partial_spearman_summary.json):
- `n = 80`
- partial Spearman `rho = -0.0105`
- `p = 0.927`

Raw and control correlations from [figures/gt_correlation](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation):
- raw RMSE vs risk: `rho = 0.234`, `p = 0.037`
- match knee error vs risk: `rho = 0.506`, `p = 1.66e-6`
- match thigh error vs risk: `rho = 0.598`, `p = 4.75e-9`

## Recommended Results Section

Use this structure:

### 3.1 Predictive Performance

Key points:
- the CNN-BiLSTM achieved sub-10° held-out RMSE on the GT subject-holdout benchmark
- report the exact `8.96°` test RMSE and `6.11°` MAE
- note that this establishes a model accurate enough for the planned simulation benchmark

Suggested sentence:

`On the current Georgia Tech subject-holdout benchmark, the CNN-BiLSTM achieved a held-out test RMSE of 8.96° and MAE of 6.11°, indicating sub-10° knee-angle prediction accuracy on the implemented dataset split.`

### 3.2 Simulation Outcomes

Key points:
- 80 successful windows were evaluated
- `PRED` often improved simulated knee tracking relative to `REF`
- but risk did not necessarily decrease with lower model error

Suggested sentence:

`Across the 80-window simulation run, the model-conditioned rollout had lower mean simulated knee RMSE than the reference rollout (9.82° vs. 12.26°), yet its mean balance-risk AUC was higher (0.891 vs. 0.677), indicating that improved local knee tracking did not automatically translate into lower simulated balance risk.`

### 3.3 Correlation Results

Key points:
- raw RMSE-risk link is weakly positive
- after controlling for motion-match quality, the association disappears

Suggested paragraph:

`Raw prediction error showed only a weak positive relationship with simulated balance-risk AUC (Spearman rho = 0.234, p = 0.037). However, after controlling for motion-match knee error and thigh-orientation error using a partial Spearman procedure based on the Frisch-Waugh-Lovell theorem, the association was effectively null (rho = -0.0105, p = 0.927). In contrast, the motion-match controls themselves were substantially more correlated with risk, particularly thigh-match error.`

## Recommended Discussion Section

Main message:
- sub-10° prediction error is achievable on the current GT benchmark
- but prediction RMSE is not the dominant determinant of simulation risk in the present pipeline
- motion-match quality is the stronger bottleneck

Suggested discussion claims:

1. `The study does not support using RMSE alone as a proxy for physical usefulness in this simulation setting.`
2. `The null partial correlation suggests that once motion-matching quality is accounted for, prediction error contributes little additional explanatory power for balance-risk AUC.`
3. `This indicates that future improvements in the simulator pipeline may need to focus as much on match quality and whole-body contextual consistency as on lowering knee-prediction RMSE.`

## Figures To Insert

Use these files directly:

1. Raw relationship
   - [raw_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/raw_rmse_vs_risk.png)

2. Partial / adjusted relationship
   - [partial_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/partial_rmse_vs_risk.png)

3. Motion-match confounders
   - [match_controls_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/match_controls_vs_risk.png)

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
2. For the GT path, motion matching uses a scalar thigh-pitch proxy derived from `hip_flexion_r`, not a directly measured thigh quaternion.
