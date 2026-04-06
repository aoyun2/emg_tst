# Towards Physics-Grounded Evaluation of Knee-Angle Regressors for Lower-Limb Prosthetics: A Subject-Holdout Study Using CNN-BiLSTM and MuJoCo Simulation

**AP Research**

---

## Abstract

Knee-angle prediction from surface electromyography (sEMG) and inertial measurement unit (IMU) signals is central to transfemoral prosthetic control, yet validation almost exclusively uses cross-validation root-mean-square error (RMSE), which does not capture whether lower prediction error produces safer locomotion. This study pairs a convolutional–bidirectional long short-term memory (CNN-BiLSTM) regressor trained on an open biomechanics dataset (Georgia Tech, 55 subjects, normal walking) with a physics-simulation evaluation pipeline built on MuJoCo (Todorov et al., 2012) and the MoCapAct expert-policy library (Wagener et al., 2022). Motion matching (Clavet, 2016) bridges the wearable-sensor domain and the full-body simulator by retrieving the most kinematically similar reference clip for each held-out prediction. Paired MuJoCo rollouts then compare an unmodified humanoid expert (REF) against one whose right knee is overridden by the model output (PRED). Across 55-fold leave-one-file-out cross-validation the model achieves mean held-out RMSE of 7.84° (median 6.85°). The PRED override increases integrated balance risk—quantified as the area under a per-step extrapolated-centre-of-mass (XCoM) instability score—in 95% of the 80 simulation trials (mean excess AUC = 0.208, Wilcoxon *p* < 0.001). A partial Spearman correlation controlling for motion-matching quality via the Frisch–Waugh–Lovell (FWL) theorem (Frisch & Waugh, 1933) finds no independent association between prediction accuracy and instability magnitude within the observed RMSE range (partial *ρ* = −0.022, *p* = 0.851). These results indicate that at the accuracy scale achieved here, the dominant driver of override-induced instability is whole-body coordination disruption rather than prediction error magnitude, motivating end-to-end trajectory generation as a more rigorous path for simulation-in-the-loop evaluation.

---

## 1. Introduction

Lower-limb amputation affects approximately 1.9 million Americans and is projected to double in prevalence by 2050 (Ziegler-Graham et al., 2008). Transfemoral—or above-knee—amputations are particularly consequential because they eliminate the biological knee joint, depriving the user of the proprioceptive feedback and neuromuscular coordination that normally govern load-bearing locomotion. As of 2000, transfemoral cases constituted roughly 31% of major lower-limb amputations in the United States (Dillingham et al., 2002). Powered prosthetic knees must therefore infer appropriate joint kinematics from upstream signals—principally surface electromyography (sEMG, the electrical activity of residual thigh muscles measured by skin-surface electrodes) and inertial measurement unit (IMU, a combined accelerometer and gyroscope) data from a co-located wearable.

Over two decades, regression-based intent decoders have advanced from classical supervised learning on hand-engineered feature sets (Hudgins et al., 1993; Chowdhury et al., 2013) to end-to-end deep learning architectures—recurrent (Hochreiter & Schmidhuber, 1997), temporal-convolutional (Bai et al., 2018), and attention-based (Zerveas et al., 2021)—achieving sub-10° RMSE on held-out walking data. Despite this progress, evaluation remains almost entirely RMSE-based: the model is trained and tested on the same sensor modality, and success is defined as accurate joint-angle prediction in isolation. RMSE answers whether the model is accurate, but not whether its outputs would sustain stable locomotion under gravitational loading—the question that actually matters to the user.

Physics simulation addresses this gap by coupling the model's outputs to a full-body mechanical environment. Hargrove et al. (2007) established that virtual environments predict real-world prosthetic controller performance. However, there is a domain mismatch: wearable sensors capture a single segment's kinematics, whereas a physics simulator requires a complete specification of the humanoid's joint configuration at every timestep. Bridging this gap requires *motion matching* (Clavet, 2016)—a retrieval technique borrowed from real-time character animation. Given the ground-truth knee trajectory and the measured thigh orientation from the held-out window, motion matching searches a library of pre-recorded full-body motion-capture clips for the kinematically closest reference, supplying the simulator with a realistic whole-body context in which to test the predicted knee signal. The MoCapAct framework (Wagener et al., 2022) provides exactly such a library: thousands of CMU motion-capture clips together with trained neural-network expert policies that reproduce those clips in a MuJoCo humanoid (Todorov et al., 2012), making it an ideal host for paired override experiments.

This study operationalises that approach: we train a CNN-BiLSTM regressor, validate it under strict subject-holdout cross-validation, use motion matching to place each held-out prediction into a full-body simulation context, and test whether lower prediction error translates to lower simulated balance risk when the model drives a prosthetic knee override. The negative finding—that prediction accuracy carries no independent association with instability once motion-matching quality is controlled—motivates a reappraisal of both the evaluation pipeline and the deployment model for data-driven prosthetic controllers.

---

## 2. Background

### 2.1 sEMG-Based Knee-Angle Regression

The sEMG signal encodes residual-muscle activation as a noisy proxy for intended lower-limb kinematics. Extracting continuous joint angles from it is fundamentally constrained by electrode-placement variability and inter-subject differences in muscle geometry (Farina et al., 2014). Classic regressors applied hand-crafted time- and frequency-domain features to multi-channel EMG windows (Hudgins et al., 1993; Chowdhury et al., 2013; Phinyomark et al., 2012). Deep learning eliminates manual feature engineering by learning temporal representations directly from minimally filtered signals. Long short-term memory (LSTM) networks (Hochreiter & Schmidhuber, 1997) capture multi-second gait-cycle dependencies; temporal convolutional networks (TCN) with dilated causal filters (Bai et al., 2018) offer parallelisable training with similar sequence-modelling capacity; transformer encoders (Zerveas et al., 2021) provide global attention across the full window at higher data cost. Supplementing sEMG with IMU data—three-axis accelerometer and gyroscope from the same thigh—provides absolute postural context that partially resolves the phase ambiguity inherent to sEMG-only predictors (Farina et al., 2014).

### 2.2 Physics-Based Evaluation and Balance Stability

**Motion matching.** Wearable-sensor data describes a single body segment, but physics simulation requires a complete joint configuration for the humanoid at every timestep. Motion matching (Clavet, 2016) resolves this by treating the ground-truth knee trajectory (the sensor label for the held-out window) and the measured thigh quaternion as a query into a library of full-body motion-capture clips, then selecting the clip whose kinematics most closely match the query. Using the label—rather than the model's prediction—as the retrieval key is essential: it ensures that both the REF (unmodified expert) and PRED (knee-overridden expert) rollouts are embedded in the same biomechanical context. Any difference in outcome is therefore attributable solely to the knee-angle override rather than to differences in the retrieved clip. The retrieved clip provides the simulator with a consistent whole-body pose, and its associated expert policy handles actuation—allowing a controlled override experiment in which only the knee joint is altered by the model. This study searches the MoCapAct bank (Wagener et al., 2022), which contains clips from the CMU Motion Capture Database with corresponding expert neural-network policies trained to reproduce each clip inside a dm_control MuJoCo humanoid.

**XCoM balance stability.** The instability metric used in this study is grounded in the extrapolated centre of mass (XCoM) framework of Hof et al. (2005). Under a linear inverted pendulum model of bipedal gait—which treats the body as a point mass on a rigid leg—the necessary and sufficient condition for *dynamic* stability is that the XCoM

$$\xi = \mathbf{x}_{\mathrm{CoM}} + \frac{\dot{\mathbf{x}}_{\mathrm{CoM}}}{\omega_0}$$

lies within the base of support (BoS), defined as the convex hull of current foot-contact points. Here $\omega_0 = \sqrt{g / l_0}$ is the pendulum's natural frequency ($g$ = 9.81 m s⁻², $l_0$ ≈ 1.0 m leg length, giving $\omega_0 \approx 3.13$ rad s⁻¹). The XCoM velocity term $\dot{\mathbf{x}}_{\mathrm{CoM}} / \omega_0$ is a forward-looking momentum projection: a centre of mass moving rapidly toward the BoS edge is less stable than a stationary one at the same position. The signed margin $\delta = \text{dist}(\xi, \partial \text{BoS})$ (positive inside BoS, negative outside) is therefore a more predictive pre-fall indicator than CoM position alone (Hof et al., 2005). Figure 2 illustrates the geometry, defines the AUC instability scalar, and shows how the paired excess AUC isolates the override effect.

---

## 3. Methods

### 3.1 Dataset

We used the Georgia Tech open biomechanics dataset: 55 able-bodied subjects performing level-ground treadmill walking. Each recording provides raw sEMG from four right-leg channels (rectus femoris, biceps femoris, vastus lateralis, medial gastrocnemius) at 2,000 Hz; a six-axis thigh IMU (three-axis accelerometer, three-axis gyroscope) at 200 Hz; and optical-motion-capture knee included angle and thigh quaternion at 200 Hz.

### 3.2 Preprocessing

Raw sEMG was processed through three sequential stages: (1) high-pass filtering at 20 Hz (third-order IIR) to suppress motion artefact; (2) full-wave rectification; and (3) a 5 Hz moving-average low-pass filter to extract the linear envelope—the slowly-varying amplitude of muscle activation. The resulting envelope was resampled to 200 Hz by timestamp-aligned linear interpolation, matching the IMU and kinematic timebase. Thigh quaternion channels were resampled with spherical linear interpolation (SLERP), which correctly interpolates rotations by traversing the shortest arc on the unit sphere rather than linearly mixing components (Huynh, 2009). The final feature vector is 10-dimensional per timestep: four EMG envelopes and six IMU channels. Features were normalised using per-recording z-scoring for EMG channels and a global min–max scaler fit on training data only. The regression target was the knee included angle (0° = fully flexed, 180° = fully extended) at a 10 ms lookahead.

### 3.3 Model Architecture

The CNN-BiLSTM regressor processes 400-sample (2.0 s at 200 Hz) input windows. The architecture has three stages: (1) a *convolutional stem*—two one-dimensional convolutional layers (kernel width 5, GELU activation, 10% dropout) projecting from 10 input channels to 32 feature channels, which extract local temporal patterns shared across the whole window; (2) a *bidirectional LSTM encoder*—two stacked bidirectional LSTM layers (Schuster & Paliwal, 1997; Hochreiter & Schmidhuber, 1997) with 64 hidden units per direction, each direction reading the sequence forward and backward to capture both prior and subsequent context; and (3) a *regression head*—a fully-connected layer (128 → 64, GELU, 10% dropout) followed by a scalar output. Total parameters: approximately 260,000.

Training used the Adam optimiser (Kingma & Ba, 2015) with learning rate 10⁻³ and weight decay 10⁻⁴. Loss was Huber (Huber, 1964) with δ = 5°, which behaves as mean squared error for small residuals and mean absolute error for large ones, reducing sensitivity to outlier windows. Batch size was 128 and gradients were clipped to a maximum norm of 1.0. Early stopping with patience 2 (up to 6 epochs) selected the checkpoint with lowest validation RMSE.

### 3.4 Cross-Validation

We employed leave-one-file-out (LOFO) cross-validation across all 55 recordings. In each fold one complete recording is the test set and one additional recording is the validation set; the remaining 53 recordings form the training set. Splitting by complete recording—rather than by individual windows—prevents data leakage: sequential windows from the same walking bout are highly correlated, and mixing them across train/test sets would inflate measured accuracy (Farina et al., 2014).

### 3.5 Motion Matching and Simulation

The simulation pipeline (see Figure 1) operates on the held-out windows generated by LOFO. For each 2.0 s window, the ground-truth knee trajectory (the held-out sensor label) and the measured thigh quaternion were used as a query to search the MoCapAct snippet bank. Using the label as the retrieval key—rather than the model's prediction—ensures that both REF and PRED rollouts share the same clip and initial conditions; the only variable that differs between them is the presence or absence of the knee override. The top-12 candidate clips were refined by solving for a coordinate-frame alignment offset; the candidate with the lowest refined knee RMSE was selected. Clips with a match knee RMSE exceeding 25°—indicating the label kinematics were too dissimilar to any available reference—were excluded, retaining 80 of 128 attempted trials. Figure 3b shows the distributions of match knee RMSE and match thigh orientation error across the 80 retained trials; the mean match knee RMSE (≈7.9°) is comparable in magnitude to the mean model RMSE (7.84°), establishing that the retrieval noise floor is a non-negligible source of variance in the simulation outcomes.

Each retained trial ran two MuJoCo rollouts sharing the same initial conditions, expert-policy weights, and reference clip:

- **REF**: the MoCapAct expert policy tracked the matched clip without intervention. The expert policy is a neural network trained via imitation learning to reproduce the reference mocap trajectory; it manages all joints simultaneously and produces naturally balanced locomotion.
- **PRED**: the same expert policy ran, but the right-knee actuator was overridden by a proportional-derivative (PD) controller—a feedback controller that applies torque proportional to the angle error and its rate of change (kP = 800, kD = 40)—targeting the model's predicted knee angle. All other joints remained under the expert policy's control.

Both rollouts ran for 67 steps (≈2.01 s at dt = 0.03 s) following 101 warm-up steps to reach a stable gait cycle.

### 3.6 Instability Metric

At each timestep we computed two quantities from the simulation state: (i) trunk *uprightness u* (cosine of the root-segment tilt angle from vertical, ranging from 1 = perfectly upright to 0 = horizontal); and (ii) the signed XCoM margin δ (Section 2.2). These were combined into a per-step risk score r_t ∈ [0, 1]:

$$r_t = 1 - (1 - r_{\text{tilt},t})(1 - r_{\text{support},t})$$

where r_tilt,t increases from 0 toward 1 as trunk uprightness degrades (incorporating both absolute tilt and tilt rate), and r_support,t increases as the rolling minimum XCoM margin over a ten-step window becomes persistently negative. A score of 0 indicates no detected risk; a score of 1 indicates both trunk collapse and sustained XCoM-outside-BoS instability. Figure 2 (Panels B and C) shows representative XCoM margin and risk-score traces for REF and PRED in the same trial.

The **primary outcome** is the excess instability area under the curve (AUC)—the time-integrated risk across a trial minus the same quantity for the paired REF rollout:

$$\Delta_{\text{AUC}} = \int_0^T r_t^{\text{PRED}}\,\mathrm{d}t - \int_0^T r_t^{\text{REF}}\,\mathrm{d}t$$

computed via the trapezoidal rule. Using the *difference* rather than the PRED AUC alone implements a within-trial paired control. REF and PRED share identical initial conditions, reference clip, and expert-policy weights; the only variable that differs is the presence of the knee override. Different clips vary substantially in their inherent balance difficulty regardless of any override—a clip that demands extreme lateral corrections will produce high PRED AUC for any model. Subtracting AUC_REF removes this clip-level baseline and isolates the marginal effect attributable to the override, analogous to a change-from-baseline design in a controlled experiment. A positive Δ_AUC indicates the override increased integrated balance risk relative to what the expert would have produced unassisted.

### 3.7 Partial Spearman Correlation

The primary scientific question is whether prediction accuracy (predictor X = model RMSE against GT wearable data) independently predicts instability (outcome Y = Δ_AUC). However, motion-matching quality—how closely the retrieved clip resembles the predicted motion—also influences the simulation outcome and is correlated with model RMSE. A naive Spearman correlation between X and Y would conflate model quality with retrieval quality.

We controlled for this using Frisch–Waugh–Lovell (FWL) residualization (Frisch & Waugh, 1933) with two controls (Z₁ = match knee RMSE, Z₂ = thigh orientation RMS error). Figure 5 illustrates the three-step procedure: (1) regress ranked(X) on ranked(Z) to obtain residuals e_X; (2) regress ranked(Y) on ranked(Z) to obtain residuals e_Y; (3) compute Pearson r(e_X, e_Y). The FWL theorem guarantees this equals the partial regression coefficient on X in the full model Y ~ X + Z, so r(e_X, e_Y) is the *partial Spearman rho*—the monotone association between X and Y that is orthogonal to Z. Significance was assessed with a t-statistic on n − 2 − q = 76 degrees of freedom (n = 80 trials, q = 2 controls).

---

## 4. Results

### 4.1 Predictive Performance

Across 55 LOFO folds, the CNN-BiLSTM achieved mean held-out RMSE of **7.84° (SD = 4.33°, median = 6.85°)** (Figure 3). The best-fold RMSE was 3.89° and the worst was 29.67°, reflecting genuine cross-subject variability in residual-muscle signal quality. Mean held-out mean absolute error (MAE) was 6.11°.

### 4.2 Simulation Outcomes

Across 80 retained trials (Figure 4), the PRED override increased integrated instability relative to REF in 95.0% of cases (mean Δ_AUC = 0.208, SD = 0.186; Wilcoxon one-sided test vs. zero, *p* < 0.001). The median instability AUC under PRED was substantially higher than under REF, and paired within-trial lines in Figure 4B show the consistent upward shift. As an illustrative secondary observation, the per-step risk score exceeded a high-risk level in roughly twice as many PRED trials (80%) as REF trials (40%), with destabilisation typically occurring at step 30 of 67 (≈0.90 s into the 2.01 s window); this binary count depends on an empirically chosen threshold and is reported for context rather than as a primary claim.

### 4.3 Association Between Prediction Error and Instability

The raw Spearman correlation between model RMSE and excess instability AUC was ρ = −0.166 (*p* = 0.140)—negative in sign but not statistically significant. Motion-match knee RMSE showed a stronger association with excess instability (ρ = −0.258, *p* = 0.021), indicating that retrieval quality is a larger determinant of outcome than model accuracy. After FWL residualization on both motion-match controls, the partial Spearman rho collapsed to **−0.022** (*p* = 0.851, *df* = 76), demonstrating that within the observed RMSE range, prediction accuracy carries no independent association with simulated instability once retrieval quality is accounted for (Figure 5).

---

## 5. Discussion

### 5.1 Validity of the XCoM Instability Metric

The XCoM margin has a rigorous mechanical foundation: Hof et al. (2005) proved that ξ exiting the BoS is the necessary and sufficient condition for loss of dynamic stability in the linear inverted pendulum model, and that this margin is more predictive of impending instability than CoM position alone. The combined risk score additionally incorporates trunk uprightness and tilt rate, which are established early indicators of balance deterioration in amputee gait (Silverman & Neptune, 2011). Two caveats apply. First, the rigid-body humanoid does not model viscoelastic tissue, reflexive muscle responses, or voluntary corrective stepping; the XCoM margin maps to a simplified mechanical system. Second, the 67-step window is short (≈2 s): the simulation was never terminated early and root height never dropped below 0.65 m in any trial, so the metric captures instability *onset* rather than confirmed collapse. Despite these limitations, the 95% positive excess AUC finding is large and consistent, difficult to attribute to noise, and is grounded in a theoretically defensible stability condition.

### 5.2 Mechanism of Override-Induced Balance Disruption

The MoCapAct expert policy was trained via imitation learning to reproduce full-body reference trajectories through coordinated, whole-body actuation (Wagener et al., 2022). When the right-knee actuator is externally overridden, the remaining actuators continue issuing commands calibrated to the unmodified reference trajectory—but the actual joint configuration has changed. Even a small knee-angle discrepancy alters hip moment, trunk acceleration, and lateral weight distribution in ways the policy cannot compensate for in real time, because the override is invisible to the policy's input state. The result is that the override disrupts whole-body coordination even when the knee angle itself closely follows the target.

This mechanism parallels the compensatory gait patterns documented in transfemoral amputees: users develop altered trunk kinematics and asymmetric loading precisely because an externally driven prosthetic knee does not integrate into the body's neuromuscular coordination loop (Silverman & Neptune, 2011). A controller that improves knee-angle accuracy in isolation may still destabilise the broader locomotor system if it perturbs compensatory strategies the user or expert policy has already adopted.

### 5.3 Retrieval Noise Floor Masks Prediction Error Effects at the Observed Accuracy Scale

The FWL analysis (partial ρ = −0.022, *p* = 0.851) shows that once motion-matching quality is controlled, model RMSE explains essentially none of the additional variance in excess instability *within the observed accuracy range* (held-out RMSE 3.89°–29.67°, mean 7.84°). This is not a claim that prediction accuracy is irrelevant in general—a grossly inaccurate controller would trivially produce unsafe knee trajectories—but that at the accuracy levels achieved here, the retrieval noise floor dominates any residual variation in model quality.

*Retrieval noise floor.* The mean match knee RMSE (7.93°) is comparable in magnitude to the mean model RMSE (7.84°). From the simulator's perspective, the dominant source of deviation between the predicted trajectory and an ideal reference is the coarseness of the snippet bank, not the model itself. When match quality is held constant, the remaining variation in model RMSE is too small relative to the retrieval error to produce a detectable independent effect.

*Threshold non-linearity of balance loss.* The XCoM-based risk is threshold-driven in its physical interpretation: ξ is either inside or outside the BoS. Within the accuracy range studied, a model error of 5° and one of 12° may produce qualitatively identical balance outcomes if both push the XCoM to the same side of the boundary. Continuous RMSE cannot capture this categorical distinction, further attenuating its association with the risk-score integral.

### 5.4 Limitations and Future Directions

The retrieval step introduces a noise floor (mean 7.93° match RMSE) comparable to model error, masking model-driven effects on simulation outcome. End-to-end trajectory generation—using learned motion priors conditioned directly on the EMG window—would eliminate this confound and provide a cleaner test of whether model accuracy drives stability. All trials are level-ground normal walking by able-bodied subjects; stair climbing, ramp negotiation, and variable-cadence walking are untested. With 80 trials the analysis has approximately 80% power to detect a partial ρ ≥ 0.31 at α = 0.05; smaller effects would require substantially more trials. Results derive from rigid-body simulation and should be interpreted as mechanistic evidence rather than predictions of real-world amputee outcomes.

---

## 6. Conclusion

We trained a CNN-BiLSTM knee-angle regressor on 55 Georgia Tech subjects (mean held-out RMSE 7.84°) and evaluated its outputs in a MuJoCo physics simulation via paired REF vs. PRED rollouts, with motion matching bridging the wearable-sensor and simulation domains. The PRED knee override increased integrated XCoM-based instability in 95% of 80 trials (*p* < 0.001, mean excess AUC = 0.208). After Frisch–Waugh–Lovell residualization on motion-matching quality, the partial association between prediction RMSE and excess instability was negligible (partial ρ = −0.022, *p* = 0.851), indicating that at the accuracy scale achieved here, whole-body coordination disruption is the primary driver of override-induced instability—not differences in prediction error magnitude within the observed range. These results establish that cross-validation RMSE is a necessary but insufficient evaluation criterion for prosthetic knee-angle regressors.

---

## References

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.

Chowdhury, R. H., Reaz, M. B. I., Ali, M. A. B. M., Bakar, A. A. A., Chellappan, K., & Chang, T. G. (2013). Surface electromyography signal processing and classification techniques. *Sensors*, *13*(9), 12431–12466. https://doi.org/10.3390/s130912431

Clavet, S. (2016). *Motion matching and the road to next-gen animation* [Conference presentation]. Game Developers Conference. https://www.gdcvault.com/play/1023280

Dillingham, T. R., Pezzin, L. E., & MacKenzie, E. J. (2002). Limb amputation and limb deficiency: Epidemiology and recent trends in the United States. *Southern Medical Journal*, *95*(8), 875–883. https://doi.org/10.1097/00007611-200208000-00018

Farina, D., Merletti, R., & Enoka, R. M. (2014). The extraction of neural strategies from the surface EMG: An update. *Journal of Applied Physiology*, *117*(11), 1215–1230. https://doi.org/10.1152/japplphysiol.00162.2014

Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica*, *1*(4), 387–401. https://doi.org/10.2307/1907330

Hargrove, L., Losier, Y., Lock, B., Englehart, K., & Hudgins, B. (2007). A real-time pattern recognition based myoelectric control usability study implemented in a virtual environment. *Proceedings of the 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, 4842–4845. https://doi.org/10.1109/IEMBS.2007.4353424

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics*, *35*(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, *38*(1), 1–8. https://doi.org/10.1016/j.jbiomech.2004.03.025

Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction myoelectric control. *IEEE Transactions on Biomedical Engineering*, *40*(1), 82–94. https://doi.org/10.1109/10.204774

Huynh, D. Q. (2009). Metrics for 3D rotations: Comparison and analysis. *Journal of Mathematical Imaging and Vision*, *35*(2), 155–164. https://doi.org/10.1007/s10851-009-0161-2

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications*, *39*(8), 7420–7431. https://doi.org/10.1016/j.eswa.2012.01.102

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, *45*(11), 2673–2681. https://doi.org/10.1109/78.650093

Silverman, A. K., & Neptune, R. R. (2011). Differences in whole-body angular momentum between below- and above-knee amputees across walking speeds. *Journal of Biomechanics*, *44*(3), 379–385. https://doi.org/10.1016/j.jbiomech.2010.10.027

Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *Proceedings of the 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 5026–5033. https://doi.org/10.1109/IROS.2012.6386109

Wagener, N., Kolobov, A., Frujeri, F. V., Loyola, R., Cheng, C.-A., Hausknecht, M., & Swaminathan, A. (2022). MoCapAct: A multi-task dataset for simulated humanoid control. *Advances in Neural Information Processing Systems*, *35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/49925e9da2afefdc3c07f5cb9c87c3ea-Abstract-Datasets_and_Benchmarks.html

Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2021). A transformer-based framework for multivariate time series representation learning. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2114–2124. https://doi.org/10.1145/3447548.3467401

Ziegler-Graham, K., MacKenzie, E. J., Ephraim, P. L., Travison, T. G., & Brookmeyer, R. (2008). Estimating the prevalence of limb loss in the United States: 2005 to 2050. *Archives of Physical Medicine and Rehabilitation*, *89*(3), 422–429. https://doi.org/10.1016/j.apmr.2007.11.005
