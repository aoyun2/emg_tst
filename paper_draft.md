# Towards Physics-Grounded Evaluation of Knee-Angle Regressors for Lower-Limb Prosthetics: A Subject-Holdout Study Using CNN-BiLSTM and MuJoCo Simulation

**AP Research**

---

## Abstract

Knee-angle prediction from surface electromyography (sEMG) and inertial measurement unit (IMU) signals is central to transfemoral prosthetic control, yet validation almost exclusively uses cross-validation root-mean-square error (RMSE), which does not capture whether lower prediction error produces safer locomotion. This study pairs a convolutional–bidirectional LSTM (CNN-BiLSTM) regressor trained on an open biomechanics dataset (Georgia Tech, 55 subjects, normal walking) with a physics-simulation evaluation pipeline built on MuJoCo (Todorov et al., 2012) and the MoCapAct expert-policy library (Wagener et al., 2022). Across 55-fold leave-one-file-out cross-validation the model achieves mean held-out RMSE of 7.84° (median 6.85°, 83.6% of subject folds below 10°). For 80 held-out evaluation windows we run paired rollouts of an expert humanoid policy: one unmodified reference (REF) and one with the right knee overridden by the model prediction (PRED). Balance-risk threshold crossings—derived from the extrapolated centre of mass (XCoM; Hof et al., 2005) and trunk uprightness—occur in 40.0% of REF trials versus 80.0% of PRED trials. A partial Spearman correlation controlling for motion-matching quality via the Frisch–Waugh–Lovell (FWL) theorem (Frisch & Waugh, 1933) reveals that prediction accuracy is not associated with instability (partial *ρ* = −0.022, *p* = 0.851, *df* = 76). These results indicate that the dominant driver of override-induced instability is the disruption of whole-body coordination, not prediction error per se, motivating end-to-end trajectory generation as the path toward definitive simulation-in-the-loop evaluation.

---

## 1. Introduction

Lower-limb amputation affects approximately 1.9 million Americans and the prevalence is projected to double by 2050 (Ziegler-Graham et al., 2008). Transfemoral (above-knee) amputations are particularly consequential because they eliminate the biological knee joint, depriving the user of direct proprioceptive and neuromuscular feedback from a primary load-bearing articulation; as of 2000, transfemoral cases constituted roughly 31% of major lower-limb amputations in the United States (Dillingham et al., 2002). Powered prosthetic knees must therefore infer appropriate joint kinematics from upstream signals—principally surface electromyography (sEMG) from the residual thigh musculature and inertial data from a co-located sensor.

Over two decades, regression-based intent decoders have advanced from classical supervised learning on hand-engineered feature sets (Hudgins et al., 1993; Chowdhury et al., 2013) to end-to-end deep learning architectures—recurrent (Hochreiter & Schmidhuber, 1997), temporal-convolutional (Bai et al., 2018), and attention-based (Zerveas et al., 2021)—achieving sub-10° RMSE on held-out data for level-ground walking. Despite this progress, evaluation remains almost entirely RMSE-based. RMSE answers whether the model is accurate, but not whether its outputs would sustain stable locomotion under gravitational loading—a question that matters to the user.

Physics simulation addresses this gap directly. Hargrove et al. (2007) established that virtual environments predict real-world prosthetic controller performance, and the MoCapAct framework (Wagener et al., 2022) now provides thousands of reference motion clips for MuJoCo humanoids, enabling paired override experiments without hardware. This study operationalises that approach: we train a CNN-BiLSTM regressor, validate it under strict subject-holdout cross-validation, and test whether lower prediction error translates to lower simulated balance risk when the model drives a prosthetic knee override. The negative finding—better accuracy does not reduce instability, and instability doubles regardless—motivates a reappraisal of both the evaluation pipeline and the deployment model for data-driven prosthetic controllers.

---

## 2. Background

### 2.1 sEMG-Based Knee-Angle Regression

The sEMG signal encodes residual-muscle activation as a noisy function of the intended lower-limb kinematics; extracting continuous joint angles from it is fundamentally limited by the motor-unit decomposition problem and electrode-placement variability (Farina et al., 2014). Classic regressors applied hand-crafted features—root mean square, waveform length, zero-crossing rate, and frequency-domain statistics—to windows of multi-channel EMG (Hudgins et al., 1993; Chowdhury et al., 2013; Phinyomark et al., 2012). Deep learning eliminates manual feature engineering by learning temporal representations directly from raw or minimally filtered signals. Long short-term memory networks (Hochreiter & Schmidhuber, 1997) capture gait-cycle dependencies over multi-second windows; temporal convolutional networks with exponentially dilated causal filters (Bai et al., 2018) offer similar sequence modelling with parallelisable training; transformer encoders (Zerveas et al., 2021) provide global attention over the full window but require more data to regularise. Supplementing sEMG with co-located IMU data (three-axis accelerometer and gyroscope) provides absolute postural context that partially resolves the phase ambiguity of sEMG-alone predictors (Farina et al., 2014), and is standard in modern wearable prosthetic research.

### 2.2 Physics-Based Evaluation and Balance Stability

**Simulation framework.** MuJoCo (Todorov et al., 2012) is a rigid-body physics engine designed for contact-rich locomotion control. The MoCapAct library (Wagener et al., 2022) provides a large bank of expert neural-network policies trained to reproduce CMU motion-capture clips in a dm_control MuJoCo humanoid, and is a natural host for prosthetic override experiments. Motion matching (Clavet, 2016)—a retrieval-based technique from real-time character animation—identifies the most kinematically similar reference snippet for a given query pose, bridging the wearable-data domain and the full-body simulation domain.

**Extrapolated centre of mass (XCoM).** The theoretical foundation for the instability metric used here is the XCoM framework of Hof et al. (2005). Under a linear inverted pendulum model of bipedal locomotion, the condition for *dynamic* stability is that the extrapolated centre of mass

$$\xi = \mathbf{x}_{\mathrm{CoM}} + \frac{\dot{\mathbf{x}}_{\mathrm{CoM}}}{\omega_0}$$

lies within the base of support (BoS), where $\omega_0 = \sqrt{g / l_0}$ is the natural frequency of the inverted pendulum ($g$ = 9.81 m s⁻², $l_0$ = leg length ≈ 1.0 m, giving $\omega_0 \approx 3.13$ rad s⁻¹). The XCoM accounts for momentum: a centre of mass moving rapidly toward the edge of the BoS is less stable than one that is spatially centred but stationary. The signed margin $\delta = \text{dist}(\xi,\, \partial \text{BoS})$ (positive when $\xi$ is inside BoS) is therefore more predictive of impending instability than CoM position alone (Hof et al., 2005). Our simulation computes $\xi$ at each timestep using $\omega_0 = 3.13$ rad s⁻¹ and measures $\delta$ against the convex hull of current foot-contact points.

---

## 3. Methods

### 3.1 Dataset

We used the Georgia Tech open biomechanics dataset: 55 able-bodied subjects performing level-ground treadmill walking. Each recording provides raw sEMG from four lower-limb channels (rectus femoris, biceps femoris, vastus lateralis, medial gastrocnemius) at 2,000 Hz; six-axis thigh IMU (three-axis accelerometer, three-axis gyroscope) at 200 Hz; and optical-motion-capture knee included angle and thigh quaternion at 200 Hz.

### 3.2 Preprocessing and Feature Extraction

Raw sEMG was processed through a causal three-stage pipeline: (1) third-order IIR high-pass filter at 20 Hz to remove motion artefact; (2) full-wave rectification; (3) moving-average low-pass filter at 5 Hz to compute the linear envelope. The envelope was resampled to 200 Hz by timestamp-aligned linear interpolation, yielding a common 200 Hz timebase with the IMU and kinematic streams. Quaternion channels were resampled via spherical linear interpolation (SLERP) with continuity enforcement (Huynh, 2009). The final feature vector comprises 10 dimensions per timestep: four EMG envelopes and six IMU channels. EMG channels were z-scored per-recording; a global min–max scaler fitted on training recordings only was then applied to all features. Knee angle labels were normalised by division by 180°. The regression target was the knee included angle (0° = fully flexed, 180° = fully extended) at a two-sample (10 ms) lookahead, accounting for signal-processing latency.

### 3.3 Model Architecture

The CNN-BiLSTM regressor operates on 400-sample (2.0 s at 200 Hz) windows. It consists of: (1) a *CNN stem*—two Conv1d layers (kernel size 5, GELU activation, 10% dropout) projecting from 10 to 32 channels; (2) a *BiLSTM encoder*—two bidirectional LSTM layers (Hochreiter & Schmidhuber, 1997) with 64 hidden units per direction, capturing past and future context within the window; and (3) a *regression head*—a linear layer (128 → 64, GELU, 10% dropout) followed by a scalar output. Total parameters: ≈ 260,000. Training used Adam (lr = 10⁻³, weight decay = 10⁻⁴), Huber loss (δ = 5°), batch size 128, gradient clipping at 1.0, and early stopping (patience 2, maximum 6 epochs) on held-out validation RMSE.

### 3.4 Cross-Validation

We employed leave-one-file-out (LOFO) cross-validation across all 55 recordings. In each fold, one recording is the test set, one additional recording is the validation set for early stopping, and the remaining 53 are training data. Splitting by complete recording prevents the optimistic bias that arises when windows from the same recording appear in both training and evaluation (Farina et al., 2014).

### 3.5 Motion Matching and Physical Evaluation

For each held-out 2.0 s window, a motion-matching retrieval (Clavet, 2016) searched the MoCapAct snippet bank for the most kinematically similar full-body reference clip using predicted knee flexion angle and measured thigh quaternion as the query. The top-12 candidates were refined by solving an analytical coordinate-frame offset for each; the candidate with the lowest refined knee RMSE was selected. Trials with match knee RMSE > 25° were excluded (48 of 128 attempts excluded; 80 trials retained). Each retained trial ran two MuJoCo rollouts sharing identical initial conditions:

- **REF**: the MoCapAct expert policy tracked the matched clip unmodified.
- **PRED**: the same policy ran with the right-knee actuator overridden by a PD controller (kP = 800, kD = 40) tracking the matched expert-clip trajectory up to the model's predicted knee offset.

Both rollouts ran for 67 steps (≈2.01 s at dt = 0.03 s) following 101 warmup steps; the simulation was not terminated early regardless of detected instability.

### 3.6 Instability Metric

At each simulation timestep we computed: (i) trunk *uprightness* $u$ (cosine of root-segment tilt); and (ii) the signed XCoM margin $\delta$ (Section 2.2). A per-step risk score combined absolute tilt, tilt rate, and the rolling minimum of $\delta$ over a ten-step window into a heuristic score $r_t \in [0, 1]$:

$$r_t = 1 - (1 - r_{\text{tilt},t})(1 - r_{\text{support},t})$$

where $r_{\text{tilt},t}$ rises from 0 at $u = 0.75$ to 1 at $u = 0.40$ (incorporating both absolute tilt and tilt rate), and $r_{\text{support},t}$ rises as the rolling XCoM margin becomes persistently negative. A **balance-risk threshold crossing** (analogous to predicted fall onset) was detected when the risk score first exceeded an empirical threshold together with degraded uprightness; this step index is reported as `balance_loss_step`. These crossings do not correspond to root-height collapse ($z < 0.65$ m), since the rigid body never reached that state within the 67-step window—rather, they detect the onset of XCoM-based instability that, in a real environment, would precede a fall (Hof et al., 2005).

The primary simulation outcome was the **excess instability AUC**:

$$\Delta_{\text{AUC}} = \int_0^T r_t^{\text{PRED}}\,\mathrm{d}t \;-\; \int_0^T r_t^{\text{REF}}\,\mathrm{d}t$$

computed via the trapezoidal rule. A positive value indicates the override increased integrated balance risk relative to the baseline expert.

**Justification for the difference formulation.** Using $\Delta_{\text{AUC}}$ rather than $\text{AUC}_{\text{PRED}}$ alone implements a within-trial paired control. Because REF and PRED share identical initial conditions, expert-policy weights, matched reference clip, and physics parameters, the only variable that differs between the two rollouts is the presence of the knee override. Gait contexts vary substantially in their inherent difficulty for the expert policy—some matched clips require extreme balance corrections regardless of the knee signal—and $\text{AUC}_{\text{REF}}$ captures this baseline trial-level instability. Subtracting it isolates the marginal effect of the override, analogous to a change-from-baseline outcome in a within-subject experimental design. Analysing $\text{AUC}_{\text{PRED}}$ alone would confound model quality with clip difficulty and could produce large between-trial variance that has nothing to do with the knee regressor. The paired design also increases statistical power by removing this shared variance from the residual.

### 3.7 Partial Spearman Correlation and FWL Residualization

We tested whether model prediction error (predictor $X$ = pred-vs-GT knee RMSE) was associated with excess instability (outcome $Y$ = $\Delta_{\text{AUC}}$) after controlling for motion-matching quality (controls $Z_1$ = match knee RMSE, $Z_2$ = thigh orientation RMS error). The partial Spearman correlation was computed via Frisch–Waugh–Lovell (FWL) residualization in rank space (Frisch & Waugh, 1933). The FWL theorem states that in a linear regression of $Y$ on $X$ and $\mathbf{Z}$, the coefficient on $X$ is identical to the coefficient obtained by (1) regressing $X$ on $\mathbf{Z}$ to get residuals $\hat{e}_X$, and (2) regressing $Y$ on $\mathbf{Z}$ to get residuals $\hat{e}_Y$, then computing $\text{Corr}(\hat{e}_X, \hat{e}_Y)$. Applied to ranks, this isolates the monotone association between $X$ and $Y$ that is orthogonal to $\mathbf{Z}$—the *partial Spearman rho* (Frisch & Waugh, 1933). Significance was assessed with a *t*-statistic on $n - 2 - q = 76$ degrees of freedom ($n = 80$, $q = 2$ controls).

---

## 4. Results

### 4.1 Predictive Performance

Across 55 LOFO folds, the CNN-BiLSTM achieved mean held-out RMSE of **7.84° (SD = 4.33°, median = 6.85°)**, with 83.6% of folds below 10° and 67.3% below 8° (Figure 2). The best-fold RMSE was 3.89° and the worst was 29.67°, reflecting genuine cross-subject variability in residual-muscle signal quality. Mean held-out MAE was 6.11°.

### 4.2 Simulation Outcomes

Across 80 retained trials (Figure 3):

**Balance-risk threshold crossings.** The balance-risk threshold was crossed in **32/80 (40.0%) of REF trials** versus **64/80 (80.0%) of PRED trials**—a doubling of the high-risk event rate. Among PRED trials that crossed the threshold, the mean onset step was 30/67 (≈ 0.90 s into the 2.01 s window), indicating that destabilisation occurred well before the end of the rollout rather than only at its edge. Excess instability AUC was positive in 95.0% of all trials (mean $\Delta_{\text{AUC}}$ = 0.208, SD = 0.186; Wilcoxon one-sided test vs. zero, *p* < 0.001).

### 4.3 Correlation Between Prediction Error and Instability

The raw Spearman correlation between model RMSE and excess instability AUC was *ρ* = −0.166 (*p* = 0.140)—negative in sign and not statistically significant. The motion-match knee RMSE showed a stronger association with excess instability (*ρ* = −0.258, *p* = 0.021), indicating that retrieval quality is a larger determinant of outcome than model accuracy. After FWL residualization on both motion-match controls, the partial Spearman rho collapsed to **−0.022** (*p* = 0.851, *df* = 76), demonstrating that model prediction accuracy carries no independent association with simulated instability once retrieval quality is accounted for (Figure 4).

---

## 5. Discussion

### 5.1 Is the Instability Metric Justifiable?

The XCoM margin is grounded in established bipedal stability theory: Hof et al. (2005) proved that $\xi$ exiting the BoS is the necessary and sufficient condition for loss of dynamic stability in the linear inverted pendulum model, and that XCoM-based margins are more predictive of impending instability than centre-of-mass position alone. The simulation computes $\xi$ at each step using this formula at $\omega_0 = 3.13$ rad s⁻¹, consistent with a 1 m leg-length humanoid. The risk score additionally incorporates trunk uprightness and tilt rate, which are established early indicators of balance loss in bipedal gait (Silverman & Neptune, 2011).

Two limitations on the metric's validity should be acknowledged. First, the rigid-body humanoid does not model the viscoelastic tissue properties, reflexive muscle responses, or voluntary corrective stepping of a real amputee, so the XCoM margin computed here maps to a simplified mechanical system. Second, the threshold on $r_t$ that defines a balance-risk crossing is empirically chosen, not calibrated against clinical fall events; it represents *predicted* instability onset, not a confirmed collapse (root height never dropped below the 0.65 m threshold in any of the 80 trials). Despite this, the metric is not arbitrary: every trial that crossed the threshold did so because $\xi$ was persistently outside the BoS boundary—the exact condition Hof et al. (2005) identify as mechanically destabilising—and the doubling of this rate under PRED (40% → 80%) is a large, consistent effect that would be difficult to attribute to noise. Moreover, the risk AUC integrates severity and duration, not merely the existence of a crossing, giving a richer signal than a binary fall flag.

### 5.2 Why PRED Improves Tracking but Increases Instability

The MoCapAct expert policy was trained to track full-body reference motion using whole-body coordinated control (Wagener et al., 2022). When the right-knee actuator is overridden externally, the policy's remaining actuators continue issuing commands calibrated to the unmodified reference trajectory, but the actual joint configuration deviates. Even a small inconsistency in the knee angle propagates through the kinematic chain—altering hip moment, trunk acceleration, and lateral weight distribution—in ways the policy cannot compensate for in real time, because the override is not observed by the policy's input state. The result is that the override *disrupts whole-body coordination* even as it improves local joint accuracy.

This mechanism is analogous to the compensatory gait patterns documented in transfemoral amputees: users develop altered trunk kinematics, asymmetric loading, and modified step width precisely because an externally driven prosthetic knee does not integrate into the neuromuscular coordination loop (Silverman & Neptune, 2011). A controller that improves knee-angle accuracy in isolation may still destabilise the broader locomotor system if it perturbs the compensatory patterns the user or expert policy has adopted.

### 5.3 Why RMSE Is Not the Driver

The FWL analysis shows that after controlling for motion-matching quality (partial *ρ* = −0.022, *p* = 0.851), model prediction accuracy explains essentially none of the variance in excess instability. Two mechanisms underlie this null result:

**The retrieval noise floor.** The mean match knee RMSE (7.93°) is comparable to the mean model RMSE (8.80° in the simulation evaluation set). From the simulator's perspective, the dominant deviation from an "ideal" trajectory is the coarseness of the snippet bank, not the model. When match quality is controlled, residual model-RMSE variation is negligible relative to the coordination-disruption effect.

**Non-linearity of balance loss.** The XCoM-based risk is threshold-driven, not linearly proportional to joint-angle error. A model error of 5° and one of 12° may have qualitatively identical effects on balance if both push the XCoM to the same side of the BoS boundary. Continuous RMSE cannot capture this categorical distinction.

### 5.4 Limitations and Future Directions

**Motion-matching confound.** The retrieval step introduces a noise floor (mean 7.93° match RMSE) comparable to model error, masking model-driven effects on simulation outcome. End-to-end trajectory generation—using diffusion models or learned motion priors conditioned directly on the EMG window—would eliminate this confound and provide a cleaner test of whether model accuracy drives stability.

**Gait scope.** All trials are level-ground normal walking by able-bodied subjects. Stair climbing, ramp negotiation, and variable-cadence walking are untested and may produce larger, more clinically relevant effects.

**Sample size.** With 80 trials and df = 76, the analysis has ≈80% power to detect a partial *ρ* ≥ 0.31 at *α* = 0.05. Smaller true effects would require substantially more trials to detect.

**Real subjects.** Results are from a rigid-body simulation and should be interpreted as mechanistic evidence rather than a prediction of real-world amputee performance.

Despite these limitations, the simulation pipeline reveals a disconnect between RMSE-based and stability-based evaluation that is completely invisible in cross-validation alone, and quantifies the contribution of retrieval quality versus model quality to simulation outcome—findings that should inform the design of future simulation-in-the-loop benchmarks.

---

## 6. Conclusion

We trained a CNN-BiLSTM knee-angle regressor on 55 subjects of the Georgia Tech biomechanics dataset (mean held-out RMSE 7.84°) and evaluated its outputs in a MuJoCo physics simulation via paired REF vs. PRED rollouts. The PRED override doubled the rate of XCoM-based balance-risk threshold crossings relative to the unmodified baseline (40% REF → 80% PRED) and increased integrated instability AUC in 95% of trials (*p* < 0.001). After Frisch–Waugh–Lovell residualization on motion-matching quality, the partial association between prediction RMSE and excess instability was negligible (partial *ρ* = −0.022, *p* = 0.851), indicating that coordination disruption—not prediction accuracy—is the primary driver of override-induced instability. Cross-validation RMSE is a necessary but insufficient evaluation criterion for prosthetic knee-angle regressors.

---

## References

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.

Chowdhury, R. H., Reaz, M. B. I., Ali, M. A. B. M., Bakar, A. A. A., Chellappan, K., & Chang, T. G. (2013). Surface electromyography signal processing and classification techniques. *Sensors*, *13*(9), 12431–12466. https://doi.org/10.3390/s130912431

Clavet, S. (2016). *Motion matching and the road to next-gen animation* [Conference presentation]. Game Developers Conference. https://www.gdcvault.com/play/1023280

Dillingham, T. R., Pezzin, L. E., & MacKenzie, E. J. (2002). Limb amputation and limb deficiency: Epidemiology and recent trends in the United States. *Southern Medical Journal*, *95*(8), 875–883. https://doi.org/10.1097/00007611-200208000-00018

Farina, D., Merletti, R., & Enoka, R. M. (2014). The extraction of neural strategies from the surface EMG: An update. *Journal of Applied Physiology*, *117*(11), 1215–1230. https://doi.org/10.1152/japplphysiol.00162.2014

Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica*, *1*(4), 387–401. https://doi.org/10.2307/1907330

Hargrove, L., Losier, Y., Lock, B., Englehart, K., & Hudgins, B. (2007). A real-time pattern recognition based myoelectric control usability study implemented in a virtual environment. *Proceedings of the 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, 4842–4845. https://doi.org/10.1109/IEMBS.2007.4353424

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, *38*(1), 1–8. https://doi.org/10.1016/j.jbiomech.2004.03.025

Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction myoelectric control. *IEEE Transactions on Biomedical Engineering*, *40*(1), 82–94. https://doi.org/10.1109/10.204774

Huynh, D. Q. (2009). Metrics for 3D rotations: Comparison and analysis. *Journal of Mathematical Imaging and Vision*, *35*(2), 155–164. https://doi.org/10.1007/s10851-009-0161-2

Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications*, *39*(8), 7420–7431. https://doi.org/10.1016/j.eswa.2012.01.102

Silverman, A. K., & Neptune, R. R. (2011). Differences in whole-body angular momentum between below- and above-knee amputees across walking speeds. *Journal of Biomechanics*, *44*(3), 379–385. https://doi.org/10.1016/j.jbiomech.2010.10.027

Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *Proceedings of the 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 5026–5033. https://doi.org/10.1109/IROS.2012.6386109

Wagener, N., Kolobov, A., Frujeri, F. V., Loyola, R., Cheng, C.-A., Hausknecht, M., & Swaminathan, A. (2022). MoCapAct: A multi-task dataset for simulated humanoid control. *Advances in Neural Information Processing Systems*, *35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/49925e9da2afefdc3c07f5cb9c87c3ea-Abstract-Datasets_and_Benchmarks.html

Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2021). A transformer-based framework for multivariate time series representation learning. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2114–2124. https://doi.org/10.1145/3447548.3467401

Ziegler-Graham, K., MacKenzie, E. J., Ephraim, P. L., Travison, T. G., & Brookmeyer, R. (2008). Estimating the prevalence of limb loss in the United States: 2005 to 2050. *Archives of Physical Medicine and Rehabilitation*, *89*(3), 422–429. https://doi.org/10.1016/j.apmr.2007.11.005
