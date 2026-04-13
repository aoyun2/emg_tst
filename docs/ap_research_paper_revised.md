# When Lower RMSE Is Not Enough: Relating Wearable Knee-Angle Prediction Error to Excess Instability in Physics-Based Prosthetic Simulation

**AP Research**

**Word Count:** 4942

## Abstract

Continuous knee-angle prediction from wearable sensors is often evaluated with numerical regression metrics such as root mean squared error (RMSE) and mean absolute error (MAE), but those statistics do not directly answer whether a predicted knee trajectory would remain functionally acceptable once it is inserted into a whole-body locomotion system. That gap matters in transfemoral prosthetics, where the biological knee joint is absent and a controller must infer knee motion from incomplete biomechanical and neuromuscular information (Ahkami et al., 2023; Cimolato et al., 2022). This study therefore asked whether lower model error on a wearable knee-angle prediction task corresponded to lower excess instability in a paired physics-based simulation benchmark. Public normal-walk recordings from the Georgia Tech lower-limb biomechanics dataset were converted into a uniform file-based benchmark of 55 trials, each containing four surface electromyography (sEMG) channels and six thigh-IMU channels sampled on a 200 Hz prediction timeline (Camargo et al., 2021). A convolutional bidirectional long short-term memory network (CNN-BiLSTM) was trained with leave-one-file-out cross-validation to predict knee included angle 10 ms into the future from 2.0 s windows of wearable history (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997). Held-out predictions were then embedded into a MuJoCo humanoid control pipeline grounded in MoCapAct motion clips, producing paired reference (REF) and prediction-conditioned (PRED) rollouts for 80 retained windows (Todorov et al., 2012; Wagener et al., 2022). Instability was quantified as the area under an extrapolated center of mass (XCoM)-based instability trace, and the primary outcome was excess instability AUC = AUC_PRED - AUC_REF (Hof et al., 2005; Hof, 2008). Across the 55 held-out folds, the model achieved a mean RMSE of 7.84° and a median RMSE of 6.85°, with 46 of 55 folds below 10°. Across the 80 simulation trials, mean query-window model RMSE was 8.80° and mean motion-match knee RMSE was 7.93°. However, RMSE did not show a meaningful independent relationship with excess instability after controlling for motion-match knee error and thigh-orientation error via a rank-based Frisch-Waugh-Lovell residualization procedure: partial Spearman rho = -0.019, p = .867 (Frisch & Waugh, 1933; Lovell, 1963; Kim, 2015). The results suggest that, in this pipeline, lower prediction error alone is not a sufficient proxy for better simulated locomotor stability.

## 1. Introduction

Lower-limb amputation remains a substantial rehabilitation problem in the United States, and transfemoral amputation is especially disruptive because it removes the biological knee joint and many of the mechanical advantages that joint normally provides during stance, swing, sitting, and terrain transitions (Dillingham et al., 2002; Pran et al., 2021). Even with contemporary microprocessor knees, users frequently walk more slowly, expend more mechanical work per stride, and experience functional limitations that vary considerably from one person to another (Pinhey et al., 2022). As a result, the core control problem is not simply to recreate a generic knee trajectory. It is to infer a knee trajectory that remains usable across changing gait states, variable sensor conditions, and body-specific differences in movement.

One major challenge in transfemoral prosthetics is that direct information about the missing or mechanically replaced knee is unavailable by definition. A controller must instead infer knee behavior from nearby signals that are available, practical, and non-invasive. Surface electromyography (sEMG) provides a window into muscle activation, while inertial measurement units (IMUs) provide segment acceleration and angular-velocity information that can constrain the current kinematic state of the limb (De Luca, 1997; Farina et al., 2014). Those signals are valuable for different reasons. EMG contains neuromuscular intent, but it is noisy, non-stationary, and sensitive to fatigue, electrode placement, skin impedance, and inter-session variability (Huang et al., 2008; Chowdhury et al., 2013; Phinyomark et al., 2018). IMUs are comparatively stable and geometrically informative, but they do not directly encode muscle drive or volitional intent. The practical regression task therefore involves fusing noisy biological input with contextual kinematic input in order to estimate a quantity that is not directly measured at the point of control.

That inference problem has been studied with both classical and deep-learning approaches. Earlier work in myoelectric control relied heavily on feature-engineered models and pattern-recognition pipelines, including support vector methods, relevance vector regression, and handcrafted EMG descriptors (Hudgins et al., 1993; Hargrove et al., 2007; Li et al., 2023). More recent studies show that convolutional and recurrent neural networks can model time dependencies in continuous joint-angle prediction more effectively than many classical baselines, particularly when EMG and kinematic information are fused into the same temporal model (Sun et al., 2022; Zhang et al., 2022; Zhu et al., 2022; Moghadam et al., 2023). Transformer-based models have also shown strong results in wearable biomechanics and myoelectric prediction, particularly when sequence length and cross-subject generalization become central concerns (Zerveas et al., 2021; Liang et al., 2023; Li et al., 2023; Lin & He, 2024; Lin et al., 2025). At the same time, systematic reviews of EMG-driven lower-limb prosthetic control still describe the field as methodologically heterogeneous, especially with respect to preprocessing, validation strategy, and functional evaluation criteria (Cimolato et al., 2022; Ahkami et al., 2023).

That last issue is the one this study addresses. Most knee-angle prediction papers stop at offline predictive metrics such as RMSE, MAE, or normalized error. Those metrics are useful and necessary, but they do not by themselves establish that a model’s output will remain biomechanically tolerable when it is inserted into a full locomotor system. Prior work in myoelectric control has already shown that offline classification or regression quality can fail to capture real-time usability, especially once the human-machine loop or the task environment becomes more complicated (Hargrove et al., 2007; Krasoulis et al., 2019). In other words, a model can look statistically strong and still be physically awkward, unstable, or poorly aligned with whole-body behavior.

Physics-based simulation offers a way to test that missing layer without immediately moving into unsafe or logistically expensive human-subject trials. MuJoCo makes it possible to simulate articulated dynamics under controlled perturbations, while MoCapAct provides a large bank of reference humanoid motions and policies grounded in motion-capture trajectories (Todorov et al., 2012; Wagener et al., 2022). If a predicted knee signal is inserted into a simulated gait controller while the rest of the body is constrained by matched reference motion, the resulting rollout can be inspected for biomechanical plausibility rather than judged only by pointwise angular error. That creates an opportunity to study a narrower and more consequential question: whether lower wearable-sensor prediction error actually corresponds to better physical outcomes after the prediction is embedded into a whole-body locomotor system.

The present study uses that framework to test a specific research question: **To what extent does lower held-out knee-angle prediction error correspond to lower excess instability in a paired physics-based simulation, once motion-match quality is controlled?** The initial expectation was that lower RMSE would correspond to lower excess instability, because more accurate knee targets should, in principle, produce less disruptive overrides. However, the project was intentionally designed so that this assumption could be tested rather than taken for granted. To do that, the study moved away from the earlier custom-data transformer direction and instead adopted a fully public, reproducible benchmark built from the Georgia Tech open biomechanics dataset, a fixed CNN-BiLSTM model, a file-holdout validation design, and an XCoM-based excess-instability outcome defined relative to a matched reference rollout (Camargo et al., 2021; Hof et al., 2005; Hof, 2008). By comparing predictor RMSE to **excess** instability instead of absolute instability, the study asks not whether a clip is globally hard, but whether substituting the learned knee trajectory makes that clip materially worse than its own matched reference condition.

[[FIGURE:figures/paper_native/fig1_pipeline.png|Figure 1. End-to-end study pipeline. The figure shows the wearable input window, the CNN-BiLSTM architecture, the predicted knee trace, motion matching into the MoCapAct bank, and the paired REF/PRED simulation benchmark.]]

## 2. Methodology

### 2.1 Data Source and Benchmark Construction

The raw source for the current benchmark was the Georgia Tech open lower-limb biomechanics dataset, a public resource containing synchronized electromyography, inertial, kinematic, kinetic, and motion-capture measurements across level-ground walking, ramps, stairs, and transitions (Camargo et al., 2021). The present benchmark intentionally restricted itself to **normal-walk** recordings so that the prediction task and the downstream motion-matching bank would both remain centered on steady locomotion rather than highly heterogeneous task classes. Using the project’s conversion script, the processed Georgia Tech files were converted into 55 repo-native recordings named `gt_data000.npy` through `gt_data054.npy`, each representing one file-level unit for cross-validation.

This choice matters because the unit of holdout in the present study is the **recording file**, not an individual window. Adjacent windows cut from the same recording share both sensor history and gait-state context, so random window splitting would substantially inflate apparent generalization by allowing nearly duplicated samples to appear in both training and testing partitions. For structured temporal data, that kind of leakage is a known problem, and blocked or grouped validation designs are generally preferred over naive random splitting when the goal is realistic out-of-sample performance (Roberts et al., 2017; Tougui et al., 2021). Accordingly, the current study used leave-one-file-out (LOFO) evaluation across the 55 converted recordings, with one additional training file held out inside each fold for validation.

### 2.2 Signals, Units, and Preprocessing
At the raw-signal level, the benchmark used four EMG channels and six thigh-IMU channels. The EMG channels were right rectus femoris (`RRF`), right biceps femoris (`RBF`), right vastus lateralis (`RVL`), and right medial gastrocnemius (`RMGAS`). The thigh-IMU channels were the right anterior-thigh accelerometer and gyroscope axes (`RAThigh_ACC{X,Y,Z}` and `RAThigh_GYRO{X,Y,Z}`). These channels were chosen because they are present in the public dataset, directly relevant to lower-limb control, and sufficiently local to the thigh-knee system to support continuous prediction without depending on sensors that would be unavailable on a transfemoral prosthesis itself (Camargo et al., 2021).

The raw EMG timeline was sampled at approximately 2000 Hz, whereas the IMU and angle timelines were native 200 Hz. Rather than force the entire benchmark onto an artificial lower rate, the project retained the native 200 Hz angle/IMU timebase and projected the EMG onto that same timeline after standard envelope extraction. The preprocessing pipeline followed conventional EMG practice: second-order high-pass filtering at 20 Hz to reduce motion artifact, full-wave rectification, second-order low-pass filtering at 5 Hz to obtain a linear envelope, and timestamp-aligned interpolation onto the 200 Hz output grid (De Luca, 1997; Chowdhury et al., 2013; Phinyomark et al., 2018). The resulting per-timestep feature vector therefore consisted of 10 values: four EMG envelope channels and six thigh-IMU channels.

The prediction target was the right-knee **included angle**, defined so that 180° corresponds to full extension and smaller values correspond to greater flexion. In the converted Georgia Tech recordings, this target was derived from `knee_angle_r` as `knee_included_deg = 180 - clip(-knee_angle_r, 0, 180)`. This convention aligned the learned output with the downstream simulator and motion-matching code, both of which were written around an included-angle representation rather than a signed flexion angle. All labels were divided by 180 before training so that the network operated on a normalized target range.

### 2.3 Windowing and Forecast Horizon

Each sample window contained 400 timesteps, equivalent to 2.0 s of history at 200 Hz. The model did not predict the final sample in the input window directly. Instead, it predicted the knee included angle **2 samples ahead**, corresponding to a 10 ms lookahead. This short forecast horizon was chosen as a minimal nonzero prediction interval: long enough to avoid framing the task as zero-latency identity tracking, but still close enough to the current state that the learned target remains interpretable as near-term control (Sun et al., 2022; Lin et al., 2025).

Training windows were generated lazily at stride 1 so that every valid 2.0 s history segment could, in principle, be sampled without materializing the full combinatorial window set in memory. However, to keep epochs computationally manageable and avoid overemphasizing redundant neighboring windows, the trainer sampled 8,192 windows per epoch from the available stride-1 pool. The held-out evaluation pool used wider spacing when `samples_dataset.npy` was built for the simulation protocol, because the goal there was benchmark diversity rather than dense stochastic optimization.

### 2.4 Model Architecture

The publication model was a **CNN-BiLSTM last-step regressor**. A convolutional frontend was used to extract local temporal patterns in the 10-channel signal history, while the bidirectional LSTM aggregated longer context across the full 2.0 s input window before a small multilayer perceptron (MLP) mapped the final latent state to a single knee-angle prediction. This general CNN-LSTM family is well established in continuous EMG-based kinematic estimation, and recent lower-limb studies continue to report strong performance for convolutional-recurrent hybrids that fuse wearable signals over time (Sun et al., 2022; Zhu et al., 2022; Moghadam et al., 2023).

The exact architecture used in the current codebase was:

- input shape: `[B, 400, 10]`
- `Conv1d(10 -> 32, kernel=5, padding=2)` + GELU
- `Conv1d(32 -> 32, kernel=5, padding=2)` + GELU + dropout 0.10
- 2-layer bidirectional LSTM with hidden size 64 in each direction
- last-timestep readout from the bidirectional sequence output, giving a 128-dimensional vector
- MLP head: `Linear(128 -> 64)` + GELU + dropout 0.10 + `Linear(64 -> 1)`

The network therefore learned from the full historical window but regressed only the single final output associated with `t + 10 ms`. Because the target lies just beyond the end of the input window, the bidirectional recurrence does not leak future information relative to the label; it simply summarizes the already observed past in both temporal directions within the window (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997).

### 2.5 Training Configuration

Optimization used Adam with learning rate `1e-3`, weight decay `1e-4`, batch size 128, gradient clipping at 1.0, and a Huber loss with delta = 5° (Huber, 1964). Huber loss was used instead of plain mean squared error because it preserves quadratic sensitivity near the target while reducing the influence of occasional large deviations, which is useful in physiological time-series data where small label or sensor misalignments can produce locally large angular errors. The maximum epoch count was 6, with early stopping after 2 non-improving validation epochs. For each fold, the best checkpoint was selected strictly by validation RMSE and saved before the held-out test file was scored.

The file-holdout design produced 55 total folds. Each fold held out one converted Georgia Tech recording for testing and one additional training recording for validation. The remaining files formed the training pool. The final benchmark metrics reported in this paper were computed by aggregating the per-fold results from `checkpoints/tst_20260405_173725_all`.

[[FIGURE:figures/paper_native/fig3_prediction_performance.png|Figure 2. Sorted held-out RMSE across the 55 leave-one-file-out folds. Each bar corresponds to one held-out recording; the dashed line marks the 10° threshold.]]

### 2.6 Motion Matching and Physics Evaluation

Offline prediction accuracy was only the first stage of evaluation. The second stage inserted the learned knee trajectory into a whole-body locomotion setting. To do that, each held-out query window was passed into the motion-matching system, which searched a MoCapAct reference bank for the closest locomotion snippet under the current paper configuration. In the finalized benchmark, candidate ranking was driven entirely by knee-trajectory RMSE with local temporal refinement radius 30, while thigh-orientation RMS was still recorded as a separate diagnostic and control variable. This design was not meant to claim that thigh orientation is irrelevant. Rather, it was the empirically best matching rule on the paper benchmark and therefore the correct rule for the current analysis.

Once a clip was selected, two paired MuJoCo rollouts were produced:

1. **REF**, which tracked the matched clip with the simulator’s reference policy and no learned knee substitution.
2. **PRED**, which was identical except that the right knee actuator (`walker/rtibiarx`) was overridden by a PD controller targeting the model’s predicted knee angle at each timestep.

The PD gains used for the paper run were `kp = 800`, `kd = 40`, and maximum force `800`. The purpose of this design was to isolate the effect of the learned knee trajectory while keeping the rest of the body under the same matched whole-body context and the same general control stack (Todorov et al., 2012; Wagener et al., 2022).

The paper simulation benchmark retained 80 successful held-out windows under this protocol. Each retained trial therefore had four key quantities: model knee RMSE on the query window, motion-match knee RMSE, motion-match thigh orientation RMS, and the paired REF/PRED simulation outcomes.

[[FIGURE:figures/paper_native/fig7_motion_matching.png|Figure 3. Representative motion-matching example. The figure shows the selected clip, its ranked competitors, and the resulting knee and thigh alignment errors for one held-out query.]]

### 2.7 Instability Metric

The project originally experimented with a more ad hoc mixed metric that combined XCoM-based support information with an uprightness term. That version was removed for the final paper because it blurred two different ideas: dynamic capture-point behavior and gross body orientation. The continuous outcome used here is instead **XCoM-margin only**. At each simulation step, the system estimated the support polygon from foot-ground contacts, computed center of mass (COM) and extrapolated center of mass (XCoM), and derived a signed XCoM support margin. Positive margin values indicate that the extrapolated center of mass remains inside the support polygon; sufficiently negative values indicate that the system would need a corrective step or other recovery action to avoid continued loss of balance (Hof et al., 2005; Hof, 2008).

From that margin, the code generated a per-step instability trace in `[0, 1]` based on margin magnitude and short-horizon margin trend. The paper outcome was then the area under that trace: `instability_auc = integral(instability_trace dt)`. The primary simulation outcome was not absolute PRED instability. It was `excess_instability_auc = instability_auc_PRED - instability_auc_REF`. This subtraction is important. Absolute instability depends partly on how difficult a matched clip is even before the learned model is involved. Excess instability therefore measures the *additional* instability associated with substituting the learned knee trajectory for the matched reference trajectory. A positive value means the prediction-conditioned rollout was worse than its paired reference. A near-zero value means the substitution made little difference. Because the trace is built from XCoM support behavior rather than an empirically calibrated fall detector, the resulting AUC should be interpreted as a **heuristic instability score**, not as a literal fall probability (Hof et al., 2005; Hof, 2008).

[[FIGURE:figures/paper_native/fig2_representative_trial.png|Figure 4. Representative held-out trial showing the predicted knee trace, the matched reference trajectory, the XCoM-margin signal, and the resulting instability trace from which instability AUC was computed.]]

### 2.8 Statistical Analysis

The statistical question in this study was not simply whether higher RMSE and higher instability tended to co-occur. Motion-match quality could confound that relationship, because some queries are intrinsically easier to match into the MoCapAct bank than others. To address that issue, the primary analysis used a **partial Spearman correlation** between model RMSE and excess instability AUC while controlling for (a) motion-match knee RMSE and (b) motion-match thigh-orientation RMS.

The implementation followed the Frisch-Waugh-Lovell (FWL) logic on ranked variables. First, the predictor, outcome, and both control variables were converted to average ranks (Spearman, 1904). Second, ordinary least squares was used to residualize ranked model RMSE against the ranked controls, and the same residualization was applied to ranked excess instability. Third, the Pearson correlation between the two residual vectors was computed, yielding a partial Spearman estimate because the calculation was performed in rank space rather than raw-value space (Frisch & Waugh, 1933; Lovell, 1963; Kim, 2015). This approach was appropriate because the study cared about monotonic rather than strictly linear association and because the raw variables showed skew and unequal spread across trials.

The primary reported statistic was therefore:

`rho_partial = corr(resid(rank(RMSE) ~ rank(match_knee) + rank(match_thigh)), resid(rank(excess_auc) ~ rank(match_knee) + rank(match_thigh))).`

The associated two-sided p-value was computed from the usual t approximation applied to the residualized correlation with degrees of freedom `n - 2 - q`, where `q = 2` controls.

[[FIGURE:figures/paper_native/fig5_fwl_correlation.png|Figure 5. Raw and residualized relationships used in the Frisch-Waugh-Lovell partial Spearman analysis. The left panel shows the raw relationship between model RMSE and excess instability; the center panel shows motion-match error as a confounding structural variable; the right panel shows the residualized relationship after both control variables were removed from the ranked predictor and ranked outcome.]]

## 3. Results
### 3.1 Predictive Accuracy

Across the 55 leave-one-file-out folds, the CNN-BiLSTM achieved a mean held-out test RMSE of **7.84°**, a median held-out RMSE of **6.85°**, a mean held-out sequence RMSE of **7.88°**, and a mean held-out MAE of **6.11°**. The interquartile range of fold RMSEs was **5.32° to 8.89°**, and **46 of the 55 folds** (83.6%) fell below the 10° threshold that had been treated as a practical target throughout development. These results establish that, at the pure prediction stage, the final model was not merely functional. It was consistently sub-10° on most held-out recordings.

That detail matters for interpreting the later simulation results. If the predictor had remained in the 15°-20° regime, a null simulation result could be dismissed as a trivial consequence of poor modeling. The final benchmark does not allow that explanation so easily. The model performed well by conventional regression standards, especially relative to the lower-limb wearable literature in which continuous knee-angle prediction often remains in the high single digits or low teens depending on the exact dataset, preprocessing, and validation protocol (Sun et al., 2022; Liang et al., 2023; Moghadam et al., 2023).

### 3.2 Motion Matching and Paired Simulation Outcomes

The simulation benchmark retained **80** successful held-out windows. Across those windows, the mean model RMSE measured on the query windows used by the simulator was **8.80°**. The mean motion-match knee RMSE was **7.93°**, with a median of **5.86°**, indicating that the final matching configuration produced generally close reference clips even before the learned override was applied. Mean motion-match thigh-orientation RMS was **8.39°**, with an interquartile range from **4.01°** to **12.76°**.

The paired simulation outcome was less favorable than the prediction benchmark alone would suggest. Mean REF instability AUC was **0.819**, whereas mean PRED instability AUC was **1.019**, producing a mean **excess instability AUC of 0.200**. Moreover, **76 of the 80 trials** (95%) had positive excess instability, meaning that, on most retained windows, replacing the matched reference knee with the learned prediction made the rollout at least somewhat more unstable than its own paired reference condition.

This result does not imply that the learned model was unusable or that the motion matcher failed. On the contrary, both the prediction benchmark and the matching benchmark were reasonably strong. Instead, it shows that a locally accurate knee-angle predictor can still degrade a whole-body rollout once that prediction is inserted into an already constrained locomotion context. The important point is not that the model “failed” in a naive sense. The important point is that the model’s success under offline regression metrics did not automatically translate into neutral or beneficial physical substitution.

[[FIGURE:figures/paper_native/fig4_simulation_instability.png|Figure 6. Distribution of REF instability AUC, PRED instability AUC, and the resulting excess instability AUC across the 80 retained simulation trials.]]

### 3.3 Relationship Between Prediction Error and Excess Instability

Contrary to the initial hypothesis, lower model RMSE did **not** correspond to lower excess instability in a meaningful way. The raw Spearman association between model RMSE and excess instability AUC was **rho = -0.168** with **p = .136**, which was already weak and not statistically significant. After ranked motion-match knee RMSE and ranked thigh-orientation RMS were controlled through the FWL residualization procedure, the association became essentially zero: **partial Spearman rho = -0.019**, **p = .867**.

This is the central empirical result of the study. Within the current Georgia Tech to MoCapAct evaluation pipeline, and after controlling for motion-match quality, **better offline knee-angle prediction did not reliably predict lower added instability in simulation**. The null result persisted even though the model had already achieved sub-10° held-out RMSE on most folds and sub-10° mean motion-match knee error across the simulation pool.

## 4. Discussion

The original expectation behind this project was straightforward: if a wearable knee-angle model becomes more accurate, then overriding a simulated prosthetic knee with that model should, all else equal, make the resulting motion less unstable. That expectation is intuitive, and it is one reason offline metrics such as RMSE are so often treated as stand-ins for functional quality. The present results complicate that assumption. In the final benchmark, the model was objectively accurate by standard regression criteria, yet that accuracy did not show a meaningful independent relationship with excess instability once motion-match quality was controlled.

The most defensible interpretation is not that prediction error is irrelevant in every setting. It is that **prediction error alone was not the dominant determinant of excess instability in this specific pipeline**. Several explanations are plausible. First, whole-body locomotion is constrained by more than sagittal knee angle, so small local improvements in knee tracking may not matter if the selected reference clip, support configuration, and policy state already impose a narrow basin of stable behavior. Second, an RMSE summary collapses phase information. Two trajectories can have similar RMSE while differing sharply in whether their largest errors occur during swing, early stance, or support transition, and those phases do not contribute equally to balance. Third, once mean model error is already in the high single digits, the remaining error variation may simply be too small or too phase-nonspecific to explain much variance in a whole-body instability measure.

In that sense, the study supports the broader claim that **offline predictive accuracy is an incomplete evaluation criterion for prosthetic control research**, a point that earlier myoelectric and virtual-environment studies hinted at but did not test in the exact form used here (Hargrove et al., 2007; Krasoulis et al., 2019). The gap identified in the introduction was not whether neural networks can predict knee motion. The literature already shows that they can (Sun et al., 2022; Liang et al., 2023; Lin et al., 2025). The gap was whether the usual accuracy metric captures downstream biomechanical usefulness once the prediction is embedded into an interactive locomotor context. In the current benchmark, it does not.

That finding is useful even though it is not the simple monotonic result originally expected. AP Research emphasizes the production of a new understanding that is justified by method and evidence rather than by whether the results confirm the starting hypothesis. The new understanding here is that a model can satisfy a strong offline benchmark and still fail to produce corresponding gains in a paired physics-based outcome. Put differently, the present study shows that **sub-10° prediction is not, by itself, evidence of simulated locomotor benefit**.

This does not mean that the simulation benchmark should replace predictive benchmarking entirely. The predictor still needs to be accurate, and the model still needs to generalize across held-out files. What the present study suggests is that the evaluation stack should be hierarchical rather than singular. A prosthetic prediction model should first be screened for prediction accuracy, then checked for motion-match compatibility, and finally evaluated for paired whole-body effects. Each stage answers a different question, and collapsing all three into RMSE alone obscures that structure.

### 4.1 Limitations

Several limitations constrain how far the present conclusion should be generalized. First, the benchmark used **public able-bodied normal-walk recordings**, not data from transfemoral prosthesis users. That choice improved reproducibility and public verifiability, but it also means the study tested a prosthetic-evaluation framework on a proxy biomechanics dataset rather than on amputee data directly (Camargo et al., 2021). Second, the holdout strategy was **file-holdout LOFO**, not subject-holdout. This was the correct design for the current converted benchmark, but it is still a narrower generalization claim than true across-subject evaluation (Roberts et al., 2017). Third, the instability outcome was based on **XCoM-derived support behavior**, which is well motivated in gait stability research but still remains a simplified heuristic rather than a complete fall detector or a full-body stability theory (Hof et al., 2005; Hof, 2008). It is therefore more accurate to describe the outcome as a measure of excess simulated instability than as a literal probability of falling.

Fourth, the simulation pipeline depended on motion matching into a MoCapAct bank. Although the final configuration achieved mean motion-match knee RMSE below 8°, that matching stage still constrains the downstream physics evaluation and is itself a source of structure in the outcome. This is exactly why the paper controlled for match quality in the correlation analysis, but it also means that the conclusion applies specifically to a prediction-plus-matching-plus-simulation stack, not to every possible use of the predictor. Finally, the final benchmark used one main model family, CNN-BiLSTM, rather than a broad architecture bake-off. That decision was deliberate because the research question concerned the relationship between accuracy and physical usefulness, not the global search for an architecture winner. Even so, future work should test whether the same null relationship persists for stronger or more causal model families.

### 4.2 Implications and Future Work

The clearest next step is not simply to lower RMSE further and assume that the physical outcome will follow. A more informative next step would be to **systematically sweep model quality** and observe whether any threshold or phase-specific regime emerges in which prediction error begins to matter physically. Another extension would be to move from normal-walk public data to prosthetic-user data while keeping the same paired REF/PRED evaluation structure. That would separate a limitation of the current dataset from a limitation of the metric itself.

Future work should also examine whether more phase-sensitive error metrics, such as stance-weighted RMSE or event-aligned error around support transitions, explain instability better than window-level RMSE does. The present null result does not prove that all prediction error summaries are equally poor. It only shows that one widely used summary, even when impressively low, did not independently explain the added instability produced by the learned trajectory in this benchmark.

## 5. Conclusion

This study tested whether lower wearable knee-angle prediction error corresponded to lower added instability in a paired physics-based prosthetic simulation. Using 55 leave-one-file-out Georgia Tech recordings, a CNN-BiLSTM model achieved a mean held-out RMSE of 7.84°, with most folds below 10°. Using 80 held-out simulation windows, the motion-matching system achieved mean knee error below 8°, and the simulator produced paired REF and PRED rollouts under a shared whole-body context. Even so, model RMSE did not show a meaningful independent association with excess instability AUC after motion-match quality was controlled. The resulting conclusion is narrow but important: within this benchmark, better offline predictive accuracy was **not** a sufficient proxy for better downstream physical behavior. For prosthetic machine learning, that means evaluation should not end at RMSE.

## References

Ahkami, B., Ahmed, K., Thesleff, A., Hargrove, L., & Ortiz-Catalan, M. (2023). Electromyography-Based Control of Lower Limb Prostheses: A Systematic Review. IEEE Transactions on Medical Robotics and Bionics, 5(3), 547–562. https://doi.org/10.1109/tmrb.2023.3282325

Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021). A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. Journal of Biomechanics, 119, 110320. https://doi.org/10.1016/j.jbiomech.2021.110320

Chowdhury, R., Reaz, M., Ali, M., Bakar, A., Chellappan, K., & Chang, T. (2013). Surface Electromyography Signal Processing and  Classification Techniques. Sensors, 13(9), 12431–12466. https://doi.org/10.3390/s130912431

Cimolato, A., Driessen, J. J. M., Mattos, L. S., De Momi, E., Laffranchi, M., & De Michieli, L. (2022). EMG-driven control in lower limb prostheses: a topic-based systematic review. Journal of NeuroEngineering and Rehabilitation, 19(1). https://doi.org/10.1186/s12984-022-01019-1

Clavet, S. (2016). Motion Matching and the Road to Next-Gen Animation. Game Developers Conference talk. https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road

De Luca, C. J. (1997). The Use of Surface Electromyography in Biomechanics. Journal of Applied Biomechanics, 13(2), 135–163. https://doi.org/10.1123/jab.13.2.135

DILLINGHAM, T. R., PEZZIN, L. E., & MACKENZIE, E. J. (2002). Limb Amputation and Limb Deficiency: Epidemiology and Recent Trends in the United States. Southern Medical Journal, 95(8), 875–883. https://doi.org/10.1097/00007611-200208000-00018

Farina, D., Merletti, R., & Enoka, R. M. (2014). The extraction of neural strategies from the surface EMG: an update. Journal of Applied Physiology, 117(11), 1215–1230. https://doi.org/10.1152/japplphysiol.00162.2014

Frisch, R., & Waugh, F. V. (1933). Partial Time Regressions as Compared with Individual Trends. Econometrica, 1(4), 387. https://doi.org/10.2307/1907330

Gill, S. V., Narain, A., Arora, H., Smith, C., & Bhamidipati, P. (2019). The Utility of the Extrapolated Center of Mass in the Assessment of Human Balance after Foot Placement. Journal of Biomechanics, 97, 109356. https://doi.org/10.1016/j.jbiomech.2019.109356

Hargrove, L., Losier, Y., Lock, B., Englehart, K., & Hudgins, B. (2007). A Real-Time Pattern Recognition Based Myoelectric Control Usability Study Implemented in a Virtual Environment. In 2007 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (pp. 4842–4845). IEEE. https://doi.org/10.1109/iembs.2007.4353424

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hof, A., Gazendam, M., & Sinke, W. (2005). The condition for dynamic stability. Journal of Biomechanics, 38(1), 1–8. https://doi.org/10.1016/j.jbiomech.2004.03.025

Hof, A. L. (2008). The ‘extrapolated center of mass’ concept suggests a simple control of balance in walking. Human Movement Science, 27(1), 112–125. https://doi.org/10.1016/j.humov.2007.08.003

Huang, H., Zhou, P., Li, G., & Kuiken, T. A. (2008). An Analysis of EMG Electrode Configuration for Targeted Muscle Reinnervation Based Neural Machine Interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 16(1), 37–45. https://doi.org/10.1109/tnsre.2007.910282

Huber, P. J. (1964). Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Hudgins, B., Parker, P., & Scott, R. (1993). A new strategy for multifunction myoelectric control. IEEE Transactions on Biomedical Engineering, 40(1), 82–94. https://doi.org/10.1109/10.204774

Hur, B., Baek, S., Kang, I., & Kim, D. (2025). Learning based lower limb joint kinematic estimation using open source IMU data. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-89716-4

Huynh, D. Q. (2009). Metrics for 3D Rotations: Comparison and Analysis. Journal of Mathematical Imaging and Vision, 35(2), 155–164. https://doi.org/10.1007/s10851-009-0161-2

Keleş, A. D., Türksoy, R. T., & Yucesoy, C. A. (2023). The use of nonnormalized surface EMG and feature inputs for LSTM-based powered ankle prosthesis control algorithm development. Frontiers in Neuroscience, 17. https://doi.org/10.3389/fnins.2023.1158280

Kim, S. (2015). ppcor: An R Package for a Fast Calculation to Semi-partial Correlation Coefficients. Communications for Statistical Applications and Methods, 22(6), 665–674. https://doi.org/10.5351/csam.2015.22.6.665

Krasoulis, A., Vijayakumar, S., & Nazarpour, K. (2019). Effect of User Practice on Prosthetic Finger Control With an Intuitive Myoelectric Decoder. Frontiers in Neuroscience, 13. https://doi.org/10.3389/fnins.2019.00891

Li, H. B., Guan, X. R., Li, Z., Zou, K. F., & He, L. (2023). Estimation of Knee Joint Angle from Surface EMG Using Multiple Kernels Relevance Vector Regression. Sensors, 23(10), 4934. https://doi.org/10.3390/s23104934

Li, X., Zhang, X., Zhang, L., Chen, X., & Zhou, P. (2023). A Transformer-Based Multi-Task Learning Framework for Myoelectric Pattern Recognition Supporting Muscle Force Estimation. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 3255–3264. https://doi.org/10.1109/tnsre.2023.3298797

Liang, T., Sun, N., Wang, Q., Bu, J., Li, L., Chen, Y., Cao, M., Ma, J., & Liu, T. (2023). sEMG-Based End-to-End Continues Prediction of Human Knee Joint Angles Using the Tightly Coupled Convolutional Transformer Model. IEEE Journal of Biomedical and Health Informatics, 27(11), 5272–5280. https://doi.org/10.1109/jbhi.2023.3304639

Lin, C., & He, Z. (2024). A rotary transformer cross-subject model for continuous estimation of finger joints kinematics and a transfer learning approach for new subjects. Frontiers in Neuroscience, 18. https://doi.org/10.3389/fnins.2024.1306050

Lin, C., Zhang, X., & Zhao, C. (2025). A parallel and efficient transformer deep learning network for continuous estimation of hand kinematics from electromyographic signals. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-16268-y

Lin, C., Zhao, C., & Zhang, X. (2025). Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets. arXiv. https://arxiv.org/abs/2510.13443

Lovell, M. C. (1963). Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis. Journal of the American Statistical Association, 58(304), 993–1010. https://doi.org/10.1080/01621459.1963.10480682

Moghadam, S. M., Yeung, T., & Choisne, J. (2023). A comparison of machine learning models’ accuracy in predicting lower-limb joints’ kinematics, kinetics, and muscle forces from wearable sensors. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-31906-z

Phinyomark, A., N. Khushaba, R., & Scheme, E. (2018). Feature Extraction and Selection for Myoelectric Control Based on Wearable EMG Sensors. Sensors, 18(5), 1615. https://doi.org/10.3390/s18051615

Pickle, N. T., Wilken, J. M., Aldridge Whitehead, J. M., & Silverman, A. K. (2018). The Inverted Pendulum Model Is Insufficient to Explain the Reactive Stepping Strategies Used in Bipedal Walking on a Known Slippery Surface. Journal of Biomechanics, 77, 176–183. https://doi.org/10.1016/j.jbiomech.2018.06.024

Pinhey, S. R., Murata, H., Hisano, G., Ichimura, D., Hobara, H., & Major, M. J. (2022). Effects of walking speed and prosthetic knee control type on external mechanical work in transfemoral prosthesis users. Journal of Biomechanics, 134, 110984. https://doi.org/10.1016/j.jbiomech.2022.110984

Pran, L., Baijoo, S., Harnanan, D., Slim, H., Maharaj, R., & Naraynsingh, V. (2021). Quality of Life Experienced by Major Lower Extremity Amputees. Cureus. https://doi.org/10.7759/cureus.17440

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera‐Arroita, G., Hauenstein, S., Lahoz‐Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross‐validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

Schuster, M., & Paliwal, K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681. https://doi.org/10.1109/78.650093

Spearman, C. (1904). The Proof and Measurement of Association between Two Things. The American Journal of Psychology, 15(1), 72. https://doi.org/10.2307/1412159

Sun, N., Cao, M., Chen, Y., Chen, Y., Wang, J., Wang, Q., Chen, X., & Liu, T. (2022). Continuous Estimation of Human Knee Joint Angles by Fusing Kinematic and Myoelectric Signals. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 30, 2446–2455. https://doi.org/10.1109/tnsre.2022.3200485

Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 5026–5033). IEEE. https://doi.org/10.1109/iros.2012.6386109

Tougui, I., Jilbab, A., & Mhamdi, J. E. (2021). Impact of the Choice of Cross-Validation Techniques on the Results of Machine Learning-Based Diagnostic Applications. Healthcare Informatics Research, 27(3), 189–199. https://doi.org/10.4258/hir.2021.27.3.189

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems 30 (pp. 5998–6008). https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

Wagener, N., Kolobov, A., Frujeri, F. V., Loynd, R., Cheng, C. A., & Hausknecht, M. (2022). MoCapAct: A Multi-task Dataset for Simulated Humanoid Control. In Advances in Neural Information Processing Systems 35 (pp. 35418–35431). https://proceedings.neurips.cc/paper_files/paper/2022/hash/e47ad085450fd2f1a5eabadf9907378f-Abstract-Datasets_and_Benchmarks.html

Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2021). A Transformer-based Framework for Multivariate Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining (pp. 2114–2124). ACM. https://doi.org/10.1145/3447548.3467401

Zhang, S., Lu, J., Huo, W., Yu, N., & Han, J. (2022). Estimation of knee joint movement using single-channel sEMG signals with a feature-guided convolutional neural network. Frontiers in Neurorobotics, 16. https://doi.org/10.3389/fnbot.2022.978014

Zhu, M., Guan, X., Li, Z., He, L., Wang, Z., & Cai, K. (2022). sEMG-Based Lower Limb Motion Prediction Using CNN-LSTM with Improved PCA Optimization Algorithm. Journal of Bionic Engineering, 20(2), 612–627. https://doi.org/10.1007/s42235-022-00280-3
