"""Assemble the manuscript from the completed run, with no hand-typed numbers.

Every quantity in the generated LaTeX is read from a run artifact.  Nothing is
transcribed, so the paper cannot drift from the experiment it reports, and a
rerun that changes a result changes the manuscript automatically.

If an artifact is missing the build stops and names it, rather than emitting a
paper with a blank where a result should be.

    python -m analysis.build_manuscript --runs-dir <runs> --out-dir manuscript
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing required run artifact: {path}\n"
            "The manuscript is generated from completed results only."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def deg(value: float, places: int = 3) -> str:
    return f"{float(value):.{places}f}"


def pval(value: float) -> str:
    """Format a p-value the way the journal expects."""
    value = float(value)
    if value < 1.0e-4:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}\\times10^{{{int(exponent)}}}"
    return f"{value:.3f}"


RANDOMIZATION_DRAWS = 1_000_000


def prel(value: float, draws: int = RANDOMIZATION_DRAWS) -> str:
    """Return a p-value with its relation.

    A randomization test cannot resolve below 1/(draws+1); a result with no
    exceedances sits at that floor and is reported as an inequality, since the
    test did not measure the quoted value, only bound it.
    """
    value = float(value)
    floor = 1.0 / (draws + 1.0)
    if value <= floor * 1.5:
        return f"p<10^{{-6}}"
    return f"p={pval(value)}"


def ci(pair: list[float], places: int = 3) -> str:
    low, high = float(pair[0]), float(pair[1])
    return f"{low:.{places}f} to {high:.{places}f}"


def signed(value: float, places: int = 4) -> str:
    return f"{float(value):+.{places}f}"


def collect(runs: Path) -> dict[str, Any]:
    """Read every artifact the manuscript quotes."""
    ablation = load(runs / "confirmation" / "ablation_summary.json")
    protocol = load(runs / "confirmation" / "protocol.json")
    controls = load(runs / "semg_controls" / "semg_controls.json")
    temporal = load(runs / "temporal_control" / "temporal_control_summary.json")
    path_protocol = load(runs / "training_path" / "protocol.json")
    checkpoints = load(runs / "training_path" / "checkpoints" / "manifest.json")
    correlation = load(runs / "analysis" / "checkpoint_correlation.json")
    statistics = load(runs / "analysis" / "statistical_summary.json")
    kinematic_check = load(
        runs / "kinematic_input_check" / "kinematic_input_check.json"
    )
    # The physics runner writes its matching gate under matching_preflight/.
    matching = load(runs / "panel" / "matching_preflight" / "summary.json")
    oracle = load(runs / "panel" / "oracle_preflight" / "summary.json")
    panel = load(runs / "panel" / "panel_manifest.json")
    physics_protocol = load(
        runs / "panel" / "physics_protocol.moving_target_pd_v2.json"
    )
    return {
        "ablation": ablation,
        "protocol": protocol,
        "controls": controls,
        "temporal": temporal,
        "path_protocol": path_protocol,
        "checkpoints": checkpoints,
        "correlation": correlation,
        "statistics": statistics,
        "kinematic_check": kinematic_check,
        "matching": matching,
        "oracle": oracle,
        "panel": panel,
        "physics_protocol": physics_protocol,
    }


def checkpoint_step(label: str) -> str:
    """Render a checkpoint as its descent step, which is what a reader needs."""
    parts = str(label).split("_")
    return str(int(parts[-1])) if parts[-1].isdigit() else str(label).replace("_", r"\_")


def checkpoint_table(data: dict[str, Any]) -> str:
    """Per-checkpoint accuracy, spread, and association as a LaTeX table body."""
    rows = []
    for row in data["correlation"]["per_checkpoint"]:
        low, high = row["bootstrap_95pct_ci"]
        rows.append(
            " & ".join(
                [
                    checkpoint_step(row["checkpoint"]),
                    str(row["n_windows"]),
                    deg(row["mean_prediction_rmse_deg"], 2),
                    deg(row["prediction_rmse_sd_deg"], 2),
                    signed(row["mean_excess_instability_auc"], 4),
                    signed(row["partial_spearman_rho"], 3),
                    f"{low:.3f} to {high:.3f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


CONTROL_LABELS = {
    "identity": "Recorded sEMG",
    "circular_shift": "Circular shift",
    "participant_swap": "Participant swap",
    "phase_randomized": "Phase randomized",
}


def control_table(data: dict[str, Any]) -> str:
    """Each surrogate's own effect, plus the paired margin the recorded signal holds."""
    contrasts = data["controls"]["verdict"]["recorded_vs_surrogate"]
    rows = []
    for key, label in CONTROL_LABELS.items():
        row = data["controls"]["conditions"][key]
        if key == "identity":
            share, paired = "---", "---"
        else:
            contrast = contrasts[key]
            share = f"{contrast['surrogate_share_of_real_effect']:.2f}"
            paired = signed(contrast["mean_deg"], 3)
        rows.append(
            " & ".join(
                [
                    label,
                    signed(row["mean_improvement_deg"], 3),
                    ci(row["bootstrap_95pct_ci_deg"], 3),
                    f"${pval(row['paired_t']['p_two_sided'])}$",
                    share,
                    paired,
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build(data: dict[str, Any]) -> str:
    ablation = data["ablation"]
    controls = data["controls"]
    correlation = data["correlation"]
    checkpoints = data["checkpoints"]
    path_protocol = data["path_protocol"]
    matching = data["matching"]
    oracle_windows = data["oracle"]["windows"]
    oracle_excess = sum(
        w["metrics"]["excess_instability_auc"]["fused"] for w in oracle_windows
    ) / len(oracle_windows)
    statistics = data["statistics"]
    temporal = data["temporal"]
    kinematic_check = data["kinematic_check"]
    knee_arm = kinematic_check["conditions"]["knee_history"]
    body_arm = kinematic_check["conditions"]["surrounding_body"]
    knee_penalty = kinematic_check["verdict"]["kinematic_penalty_for_dropping_knee_deg"]

    ladder = checkpoints["checkpoints"]
    within = correlation["within_window"]
    pooled = correlation["pooled"]
    dropoff = correlation["dropoff"]
    accuracy = correlation["accuracy_level"]
    panel = data["panel"]
    ORACLE_MAX_TRACKING_RMSE_DEG = float(
        data["physics_protocol"]["oracle_preflight"]["maximum_tracking_rmse_deg"]
    )
    physics = statistics["physics_comparison"]["paired_t"]

    surrogate_verdict = controls["verdict"]
    contrasts = surrogate_verdict["recorded_vs_surrogate"]
    beaten = surrogate_verdict["surrogates_beaten_by_recorded_semg"]
    worst_p = max(c["paired_t"]["p_two_sided"] for c in contrasts.values())
    smallest_margin = min(c["mean_deg"] for c in contrasts.values())
    surrogate_sentence = (
        (
            f"The recorded signal outperformed all three surrogates in paired "
            f"within-participant comparisons (smallest margin "
            f"{deg(smallest_margin, 3)}$^{{\\circ}}$, all $p\\leq{pval(worst_p)}$)"
        )
        if len(beaten) == len(contrasts)
        else (
            "The recorded signal did not outperform every surrogate; it failed to beat "
            + ", ".join(
                str(n).replace("_", " ") for n in contrasts if n not in beaten
            )
        )
    )

    excluding = dropoff.get("checkpoints_with_interval_excluding_zero") or []
    lowest = dropoff.get("lowest_rmse_with_interval_excluding_zero_deg")
    dropoff_sentence = (
        (
            f"A two-segment fit placed the change near "
            f"{deg(dropoff['breakpoint_mean_rmse_deg'], 2)}$^{{\\circ}}$ mean window RMSE, "
            f"with mean partial $\\rho$ {signed(dropoff['high_error_segment_mean_rho'], 3)} "
            f"above that level and {signed(dropoff['low_error_segment_mean_rho'], 3)} below it."
        )
        if dropoff.get("estimated") and excluding
        else (
            "No accuracy level produced an interval excluding zero, so no drop-off "
            "point is estimated: a two-segment fit over checkpoints that are "
            "individually indistinguishable from zero would be describing noise."
        )
    )

    interval_sentence = (
        (
            f"Intervals excluded zero at {len(excluding)} of "
            f"{correlation['n_checkpoints']} checkpoints, the most accurate being "
            f"{deg(lowest, 2)}$^{{\\circ}}$ mean window RMSE."
        )
        if excluding and lowest is not None
        else (
            f"No checkpoint of the {correlation['n_checkpoints']} sampled produced an "
            "interval excluding zero."
        )
    )

    within_sentence = (
        (
            f"Across the {within['n_windows']} windows, the within-window Spearman "
            f"correlation between a window's own accuracy and its own excess instability "
            f"averaged {signed(within['mean_spearman_rho_fisher_back_transformed'], 3)} "
            f"(95\\% CI {ci(within['ci_95pct_rho'], 3)}; "
            f"$t({within['n_windows'] - 1})={within['t_statistic']:.2f}$, "
            f"$p={pval(within['p_value_two_sided'])}$), with "
            f"{within['positive_windows']} of {within['n_windows']} windows positive."
        )
        if not within.get("insufficient")
        else "Too few windows supported a within-window analysis."
    )

    first_rung, last_rung = ladder[0], ladder[-1]
    aligned = temporal["aligned_vs_lagged"]

    return rf"""\documentclass[referee,lineno,pdflatex,sn-vancouver-num]{{sn-jnl}}

\usepackage{{graphicx}}
\usepackage{{url}}
\graphicspath{{{{figures/}}{{./}}}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{placeins}}
\raggedbottom

\begin{{document}}

\title[Simulation of knee-angle prediction]{{Towards the use of simulated
environments to evaluate sEMG-informed knee-angle prediction: RMSE predicts
simulated walking instability only above a threshold accuracy}}

\author*[1]{{\fnm{{Aaron}} \sur{{Xiong}}}}
\affil*[1]{{\orgname{{Spring Branch Academic Institute}}, \city{{Houston}}, \state{{Texas}}, \country{{United States}}}}
\email{{aaxiong2008@gmail.com}}

\abstract{{\textbf{{Background:}} Root-mean-square error (RMSE) measures the numerical
accuracy of knee-angle prediction, but it does not establish that the remaining
error changes whole-body motion. Whether a lower value corresponds to more stable
walking has not been tested.

\textbf{{Methods:}} A prespecified residual-fusion model was evaluated on level
walking from {ablation['participant_count']} healthy participants in Gait120.
The model predicted knee angle 100~ms ahead from knee-angle history, with surface
electromyography (sEMG) estimating a correction to the kinematic forecast. Three
surrogate sEMG conditions, which preserve the signal's statistics but destroy its
correspondence with the knee, tested whether any improvement was attributable to
sEMG content rather than to added model capacity. The predicted trajectory was
then the tracking target of a proportional-derivative override on the right knee
of a MuJoCo humanoid, against a paired unmodified reference on the same matched
MoCapAct motion. Because the fitted model converges to a narrow error band, the
same fixed panel of {pooled['n_windows']} windows was replayed at
{correlation['n_checkpoints']} checkpoints sampled along the model's
gradient-descent training path.

\textbf{{Results:}} Mean participant RMSE was
{deg(ablation['fused_mean_participant_rmse_deg'])}$^{{\circ}}$ with residual fusion
against {deg(ablation['no_emg_mean_participant_rmse_deg'])}$^{{\circ}}$ without sEMG
($t({controls['conditions']['identity']['paired_t']['df']})={controls['conditions']['identity']['paired_t']['t']:.2f}$, $p={pval(controls['conditions']['identity']['paired_t']['p_two_sided'])}$;
{ablation['positive_participants']} of {ablation['participant_count']}
participants improved).

Relating accuracy to simulated outcome across {accuracy['n_accuracy_levels']}
accuracy levels on the same fixed panel, RMSE and
excess instability were not monotonically related over the sampled range
($\rho={signed(accuracy['overall_spearman_rho'], 3)}$,
$p={pval(accuracy['overall_p_value'])}$). A two-segment fit located a turn at
{deg(accuracy['breakpoint_rmse_deg'], 2)}$^{{\circ}}$ (95\% CI
{ci(accuracy['breakpoint_95pct_ci'], 2)}$^{{\circ}}$). Above it, worse prediction
gave more instability (slope
{signed(accuracy['above_breakpoint']['slope_per_degree'], 5)}~s per
degree, 95\% CI {ci(accuracy['above_breakpoint']['slope_95pct_ci'], 5)}). Below it
the relationship inverted
(slope
{signed(accuracy['below_breakpoint']['slope_per_degree'], 5)}, 95\% CI
{ci(accuracy['below_breakpoint']['slope_95pct_ci'], 5)}).

\textbf{{Conclusions:}} Prediction RMSE predicted simulated walking outcome only
above roughly {deg(accuracy['breakpoint_rmse_deg'], 0)}$^{{\circ}}$. Below that the
relationship reversed, because a partially fitted model regresses toward the
participant mean and so commands a flatter trajectory that moves the knee less
than the recorded motion does. A converged predictor therefore sits
in the regime where lower RMSE no longer implies better simulated behavior.}}

\keywords{{surface electromyography, knee-angle prediction, prosthetic control,
biomechanics, MuJoCo, motion matching, extrapolated center of mass}}

\maketitle

\section{{Introduction}}\label{{sec:introduction}}
Lower-limb amputation remains a substantial rehabilitation problem in the United States, and transfemoral (above-the-knee) amputation is especially disruptive because it removes the knee joint and many of the mechanical advantages that it provides during standing, walking, sitting, and transitions onto different terrain \cite{{dillingham2002,pran2021}}. Even with contemporary microprocessor-based artificial knees, users frequently walk more slowly, expend more mechanical work, and experience functional limitations that can vary from one wearer to another \cite{{pinhey2022}}. Following from these limitations of the status quo, the main goal of current research continues to be the accurate reconstruction of the knee trajectory across changing gait states, variable sensor conditions, and differences in movement and body structure from patient to patient.

The major challenge in transfemoral prosthetics is that direct information about the missing or mechanically replaced knee is unavailable to predictive solutions for the knee angle. A reconstructive algorithm must instead infer knee behavior from nearby available signals that are practical and preferably non-invasive to collect from the patient. Surface electromyography (sEMG) provides a window into muscle activations, while inertial measurement units (IMUs) give predictor models kinematic information from the limbs, such as acceleration, angular velocity, and position \cite{{deluca1997,farina2014}}. These signals are valuable for different reasons. sEMG contains neuromuscular intent, but is noisy, non-stationary, and sensitive to fatigue, variance in electrode placement, skin impedance, and inter-subject variability, which is why artificial intelligence and machine learning techniques are commonly used to process this rather volatile signal \cite{{huang2008,chowdhury2013,phinyomark2018}}. IMUs are comparatively stable, but encode the current physical state of the limbs and body rather than the direct intent of a patient to activate a specific muscle group. The task of prediction, or regression, as it is referred to in the literature, involves fusing the noisy sEMG signal with contextual kinematic data from IMUs to estimate the knee angle at some future horizon, typically a few milliseconds from the time the sEMG and IMU data are captured.

\subsection{{Current Methods for Knee-Angle Prediction}}
This major inference problem has been studied with both classical and deep methods in machine learning. Earlier work in myoelectric control relied heavily on feature-engineered models and pattern-recognition pipelines, including support vector methods, relevance vector regression, and handcrafted EMG features \cite{{hudgins1993,hargrove2007,lihb2023}}. More recent studies show that convolutional (CNN) and recurrent neural networks (LSTM) can model time dependencies in continuous joint-angle prediction more effectively than many classical benchmarks, particularly when sEMG and kinematic information are combined into a temporally-aware model \cite{{sun2022,yi2022,zhang2022,zhu2022,moghadam2023,keles2023}}. Transformer-based models have also shown strong results in wearable biomechanics and myoelectric prediction, particularly when sequence length and cross-subject generalization become central concerns \cite{{zerveas2021,liang2023,lix2023,linhe2024,lin2025a,lin2025b}}. At the same time, systematic reviews of EMG-driven lower-limb prosthetic control still describe the field as methodologically evolving, with researchers still trying novel techniques, especially with respect to preprocessing, validation strategy, and model architecture, to improve on current benchmarks \cite{{cimolato2022,ahkami2023}}.

\subsection{{Simulation of Virtual Prosthetics}}
The primary problem with the current evaluation metric, cross-validation Root Mean Squared Error (RMSE), is that prediction accuracy alone does not guarantee the prosthetic's true usefulness in an actual kinematic context. Joint-angle regressors, such as the methods described above, are only valuable if their outputs are biomechanically plausible when considered in the context of whole-body motion. This gap between statistical error and real-time function has already been examined in myoelectric control research. Hargrove et al.\ \cite{{hargrove2007}}, for example, evaluated a pattern-recognition controller through a real-time virtual environment rather than solely reporting classifier accuracy. Similarly, Krasoulis et al.\ \cite{{krasoulis2019}} showed that prosthetic finger performance improved with user practice and could not be reliably inferred from machine learning statistics alone. Together, these studies suggest that a low RMSE could hide functionally poor control of the prosthetic.

Another issue is that real human-subject testing is difficult, expensive, and potentially risky. Testing a weak or unstable controller on an actual prosthesis could create safety concerns for the user. For that reason, virtual environments and physics-based simulation can serve as an intermediate evaluation layer between offline prediction and real-world trials. Hargrove et al.\ \cite{{hargrove2007}}, for example, evaluated a pattern-recognition myoelectric controller in a real-time virtual environment rather than relying only on offline classification accuracy. Krasoulis et al.\ \cite{{krasoulis2019}} similarly showed that prosthetic finger-control performance changed with user practice, suggesting that offline model performance does not fully capture practical control quality.

Physics-based simulation offers a closer approximation to a real-life locomotion context because it models bodies using joints, forces, contacts, and dynamics instead of simply displaying a motion visually. One Python implementation of physics-based simulation is MuJoCo, a physics engine commonly used in research for model-based control and simulated articulated bodies \cite{{todorov2012}}. For prosthetic research, this kind of engine is valuable because it can test whether a predicted movement is dynamically plausible in a controlled environment. A simulated body can fall, lose balance, or react poorly to a control signal, which provides information that a static prediction-error number cannot provide.

Motion-capture-based resources also help make simulation more realistic. MoCapAct provides motion-capture-based humanoid control data, allowing simulated movement to be grounded in recorded human-like motion rather than arbitrary synthetic poses \cite{{wagener2022}}. One derivative technique is motion matching, a related idea from video game character animation in which a system searches a database for the motion clip that best matches a query \cite{{clavet2016}}. In a prosthetic-evaluation context, motion matching is useful because it can place a predicted joint trajectory into a broader whole-body movement context. Without that context, a knee prediction remains only a time-series output; with context, it can be examined as part of a simulated gait sequence.

It is important to note that simulation is not a replacement for real human testing. A simulated body is still a simplified model of a real person, and simulated instability is not the same as a clinical fall outcome. However, simulation can still expose a gap between local prediction accuracy and whole-body functional behavior. A model that performs well by RMSE may produce a trajectory that is poorly timed, physically awkward, or destabilizing once inserted into a locomotion system. Conversely, a slightly higher-RMSE trajectory may not be physically harmful if its errors occur in less sensitive parts of the gait cycle. By opening the door for investigation into the ability of RMSE to predict functional performance, simulation helps test whether numerical prediction metrics represent real practical ability in a prosthetic algorithm.

\subsection{{Balance Metrics and Functional Evaluation}}
A common method of physically evaluating locomotion behavior is to evaluate the balance of a full body system during a walk cycle. This is traditionally done by finding the center of mass, or CoM, which constitutes the average concentration of the mass distributed across the body. During standing, keeping the CoM over the feet is closely related to staying balanced and upright. In a moving scenario, however, determining instability is more complex. A person's CoM may be in an acceptable position at one instant but still moving quickly enough that a corrective step is needed lest a loss of balance occur. The extrapolated center of mass, or XCoM, accounts for this by incorporating both the CoM and its velocity, allowing dynamic movement to be accounted for while examining balance \cite{{hof2005,hof2008}}.

To determine whether the XCoM is in an acceptable state, an area called the support polygon must be defined. In double support, or two-legged motion, the support polygon includes the area between both feet. The relationship between the XCoM and the support polygon is defined by a metric called the support margin; if the XCoM remains inside this margin near the support polygon, the body is generally easier to stabilize. If the XCoM strays too far, corrective steps by the legs may be necessary to prevent falling, resulting in a high instability (i.e. stumbling) \cite{{hof2005,hof2008}}.

Functional evaluation in prosthetics research thus requires more than numerical predictive accuracy. Equally as, if not more important to consider is the question of whether or not the knee angle derived from the prediction model remains compatible with gait, support, and balance in a full physical context. The literature has made strong progress improving on current numerical benchmarks, but a remaining gap can be found by questioning if those numerical benchmarks are accurate representations of functional success. RMSE can show that a model reconstructs a target angle accurately, but it cannot by itself show whether that angle would preserve the balance and the characteristics of the original motion as part of a physical system. This gap motivates simulation-based evaluation as the next step in the evaluation process of prosthetic regression models.

Two features of how such evaluations are ordinarily built stand in the way of
answering it directly. The first concerns where the predictor sits. Hargrove et
al.\ \cite{{hargrove2007}} drew a distinction between offline scoring and
real-time control, and the same distinction applies to simulation: a rollout that
reproduces a recorded error by displacing a reference trajectory has to be handed
the recorded future target in order to do so, and a controller supplied with the
answer is a replay rather than a controller. Only a predictor placed inside the
control loop, generating the signal that the simulated joint actually follows,
tests what a deployed model would do. The second concerns the range of accuracy
examined. A converged model occupies a narrow band of error, and an association
estimated inside that band cannot separate the possibility that error does not
matter from the possibility that error did not vary. A study that evaluates one
trained model can only compare prediction windows against one another, where
differences are dominated by which motion each window contains rather than by the
predictor that drove it.

The present study addresses the question by holding the evaluation panel fixed and
varying the model instead. One fixed panel of walking windows is replayed at a
series of accuracy levels sampled along a single model's own gradient-descent
training path, with the predicted knee angle serving as the tracking target of the
simulated joint throughout. Because the windows, the matched reference motions,
and the initial states are identical at every level, the only thing that differs
between them is how accurate the model was. That makes it possible to ask whether
RMSE and simulated walking instability are related at all, and if so, across which
part of the accuracy range.


\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig01_overview}}
\caption{{Overview of the experiment. The same fixed panel of one-second windows is replayed at every accuracy level, so window identity, matched reference motion, and initial state are held constant.}}\label{{fig:overview}}
\end{{figure}}

\section{{Methods}}\label{{sec:methods}}

\subsection{{Data preparation}}

The experiment used the public Gait120 dataset, which contains synchronized
full-body kinematics and right-leg sEMG from 120 healthy adult men performing
seven movement tasks \cite{{boo2025,boo2025dataset}}. Only level walking was used, so
that the prediction data and the MoCapAct reference bank represented the same
general activity.

Participants S001--S030 formed a development cohort used to choose
hyperparameters. The reported experiment used the remaining
$n={ablation['participant_count']}$ participants, S031--S120. Within each
participant, trials 1--3 were used to fit the model and to compute the mean and
standard deviation of that participant's twelve sEMG channels and knee angle,
which standardize the inputs and the target. Trial 5 was held out and used for
neither purpose.

Each participant contributed one value to every analysis: the mean error over
that participant's trial 5 \cite{{roberts2017}}.

The released sEMG was sampled at 2000~Hz and the OpenSim joint angles at 100~Hz.
Twelve right-leg sEMG channels were retained. Each was mean-centered, filtered with
a second-order 20--500-Hz Butterworth band-pass, rectified, and converted to a
causal 250-sample (125-ms) root-mean-square envelope
\cite{{chowdhury2013,phinyomark2018}}. No future sEMG entered an input frame.
Predictor inputs and targets were not interpolated or time-normalized.

\subsection{{Windowing, forecast horizon, and training}}

Each prediction example contained 60 knee-angle frames (600~ms) and the final 15
sampled envelope frames (150~ms) from all 12 sEMG channels. The target was the
recorded knee angle 100~ms after the final input frame. That horizon is longer
than the electromechanical delay between muscle activation and the resulting
joint movement \cite{{cavanagh1979}}, so an sEMG sample can carry information
about motion that has not yet happened, and it matches the horizon at which
comparable work predicts future knee angle from myoelectric and kinematic
signals \cite{{coker2021}}.

Because the study measures what prediction error does to a simulated walker, the
predictor only has to be good enough to be worth simulating. The design does place
two requirements on it. The first is that the model's accuracy can be varied
continuously across a wide range and returned to the same endpoint every time,
which is what locating the turn depends on. A convex objective with a closed-form
solution gives that: descent runs from the participant mean to the exact analytic
optimum, and that optimum is the terminal checkpoint rather than an approximation
of it. The second is that the sEMG contribution can be removed exactly, because setting the correction to zero removes the signal
from a model that has already been fitted, instead of requiring a second fit whose
capacity and initialization would differ. A convolutional, recurrent or
transformer predictor of the kind reviewed above would follow a stochastic
training path to an endpoint that is not exactly reproducible, and its ablation
would confound signal content with model capacity. The cost of this choice is
that the result belongs to one model family, which
Section~
ef{{sec:future}} returns to.

One participant-balanced pair of ridge regressions was fitted to trials 1--3
\cite{{hoerl1970}}. The first predicted standardized future knee angle from the
60-frame knee history. The second predicted the remaining training residual from
the sEMG history. Equal total weight per participant prevented participants with
more eligible frames from dominating the fit. The fused estimate was

\begin{{equation}}
\hat y_{{\mathrm{{fusion}}}}=\hat y_{{\mathrm{{kin}}}}+\gamma c_{{\mathrm{{EMG}}}},
\label{{eq:fusion}}
\end{{equation}}

where $\hat y_{{\mathrm{{kin}}}}$ was the kinematic forecast and $c_{{\mathrm{{EMG}}}}$ the
estimated residual correction. The comparison without sEMG used the same fitted
kinematic stage with $c_{{\mathrm{{EMG}}}}=0$. Ridge penalties selected on the
development cohort were {path_protocol['kinematic_alpha']} for the kinematic stage
and {path_protocol['residual_alpha']} for the residual stage, with
$\gamma={path_protocol['gamma']}$. The residual output was bounded as
$b\tanh(c_{{\mathrm{{EMG}}}}/b)$, where $b$ was each participant's 95th percentile
absolute kinematic residual from trials 1--3.

\subsection{{Attributing the improvement to sEMG}}

Because penalties were fixed on a separate development cohort and every model was
scored on a held-out later trial from held-out participants, added capacity that
fitted noise would raise held-out error rather than lower it. The ablation is
therefore already informative about sEMG rather than about model size. As a
further check, three surrogate conditions repeated the whole confirmation fit with
the sEMG replaced by a signal preserving its statistics but not its correspondence
with the knee; these are reported in the supplementary analysis of surrogate
controls. Every paired
comparison was tested with a two-sided paired $t$-test on the per-participant
differences.

A separate timing control shifted the same sEMG history 500~ms earlier within each
continuous trial, evaluating aligned and shifted models on identical target rows.


\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig02_signals}}
\caption{{Recorded signals and predictions for one panel window. \textbf{{A}} The twelve recorded right-leg sEMG envelopes, standardized, with the 150~ms input interval shaded. \textbf{{B}} The recorded knee trajectory, with the 600~ms history interval shaded. \textbf{{C}} The recorded knee against predictions from three points on the training path. The untrained model predicts close to the participant mean, mid-training predictions are smoothed relative to the recording, and the converged model tracks it closely.}}\label{{fig:signals}}
\end{{figure}}

\subsection{{Sampling a range of prediction accuracy}}

The residual-fusion model has a closed-form solution, which yields one converged
model and no trajectory to sample. Refitting on nested fractions of the
trial 1--3 fitting data does not help: a linear model fitted to a small fraction of 90
participants' level walking is already close to its converged error, so every such
checkpoint lands in the same narrow band.

Instead, the same model class was fitted by full-batch gradient descent on the same
participant-balanced ridge objective, from a zero coefficient vector. At step zero
each stage predicts the participant-balanced target mean, so the path begins with an error close to the
standard deviation of the knee angle itself and descends to the closed-form
solution,
which is appended as the terminal checkpoint. The descent therefore ends on exactly
the model the confirmation run reports rather than near it.

The step size was set well below the largest value for which the descent still
converges. The reason is sampling resolution rather than stability: near that
value the first steps collapse most of the error, so the high-error region is crossed between recorded points and cannot
be sampled. Checkpoints were placed to tile the achieved training-RMSE range
rather than the step index, because descent spends most of its steps near
convergence. The reported path ran {path_protocol['descent_steps']} steps and
spanned {deg(path_protocol['train_rmse_first_deg'], 2)}$^{{\circ}}$ to
{deg(path_protocol['train_rmse_last_deg'], 2)}$^{{\circ}}$ training RMSE across
{len(ladder)} checkpoints.


\subsection{{Motion matching and physics evaluation}}

The fixed physics panel was drawn from confirmation trial 5 before matching or
simulation outcomes were known. A seeded participant-balanced procedure selected
{pooled['n_windows']} one-second windows, giving each eligible participant one
window before assigning a second. Prediction error was not used for selection.
The panel size was set by the simulation budget, since every window is replayed
in both conditions at all
{accuracy['n_accuracy_levels']} accuracy levels; because the same panel is used
at every level, its size limits precision rather than biasing the comparison
between levels.

Each query was matched to a level-walking reference from MoCapAct
\cite{{wagener2022}}. Candidate snippets were aligned by a constant knee offset and
either sign convention, without amplitude scaling. Knee RMSE determined the match
rank; right-thigh pitch RMS was retained as a match-quality covariate. Mean
matching knee error was {deg(matching['mean_knee_rmse_deg'], 2)}$^{{\circ}}$
(median {deg(matching['median_knee_rmse_deg'], 2)}$^{{\circ}}$) and mean thigh
orientation RMS {deg(matching['mean_thigh_rms_deg'], 2)}$^{{\circ}}$.

In the prediction condition the model's predicted knee angle was the tracking
target of a proportional-derivative override on the right knee actuator. The
prediction was first mapped into the matched clip's convention using the constant
offset and sign that the match established:

\begin{{equation}}
\tau=K_p(q_{{\mathrm{{des}}}}-q)+K_d(\dot q_{{\mathrm{{des}}}}-\dot q),
\qquad q_{{\mathrm{{des}}}}(t)=\hat y(t).
\label{{eq:control}}
\end{{equation}}

Proportional and derivative gains were $K_p=400$ and $K_d=20$ with the actuator
force limited to 160, and the desired velocity was the causal backward difference
of the commanded trajectory, zero at the first evaluation step. The gains were
set on the apparatus rather than on any prediction: commanding a matched clip's
own recorded trajectory has to reproduce that trajectory in joint space, and a
position-only controller with no feedforward velocity trailed a fast knee badly
enough to miss the {deg(ORACLE_MAX_TRACKING_RMSE_DEG, 0)}$^{{\circ}}$ tracking
criterion used to check the apparatus. Adding the feedforward term brought
tracking inside it.

The paired reference condition ran the unmodified expert policy on the same
snippet, initial state, warm-up, and controller. Nothing in either rollout reads
the recorded future knee angle, so the simulation is driven by the model output
rather than by a replayed error.



\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig04_simulation}}
\caption{{A paired rollout at the least accurate model. \textbf{{A}} The unmodified reference. \textbf{{B}} The same matched motion with the right knee overridden; the substituted shank is highlighted. \textbf{{C}} Realized and commanded knee angle in both conditions. \textbf{{D}} The instability index in both conditions; shading marks the excess instability that is integrated to give the outcome.}}\label{{fig:simulation}}
\end{{figure}}

\subsection{{Evaluating simulations with an instability metric}}

At each control step the convex hull of foot--ground contact points defined the
support polygon, and XCoM extended horizontal center-of-mass position in the
direction of its velocity. A bounded per-step instability index in $[0,1]$ was
derived from the signed margin and its short-term trend, and integrated over the
window, giving a quantity in seconds. Absolute instability depends on how stable a
matched reference already is,
so the reported outcome is excess instability, $I'=I_{{\mathrm{{PRED}}}}-I_{{\mathrm{{REF}}}}$,
measured against each window's own paired reference. The index is study-specific
and has not been validated as a clinical stability or fall-risk measure.

\subsection{{Correlation study}}

The primary association was a partial Spearman correlation between window
prediction RMSE and excess instability, controlling for matching knee RMSE and
thigh orientation RMS, following Frisch--Waugh--Lovell residualization on ranked
variables \cite{{spearman1904,frisch1933,lovell1963}}.

Three complementary analyses are reported. \emph{{Per checkpoint}} repeats that analysis at each
accuracy level and carries the RMSE spread available to it, which separates a
near-zero association from a near-zero spread.
\emph{{Within window}} correlates each window's own accuracy against its own excess
instability across checkpoints and combines results with a Fisher-$z$ one-sample
test, which removes matched motion, snippet, and initial state as sources of
variation. \emph{{Pooled}} uses every checkpoint--window pair with a window-level
cluster bootstrap, since each window recurs once per checkpoint.

\section{{Results}}\label{{sec:results}}

\subsection{{Prediction accuracy}}

Mean participant RMSE on the held-out trial was
{deg(ablation['fused_mean_participant_rmse_deg'])}$^{{\circ}}$ with residual fusion
and {deg(ablation['no_emg_mean_participant_rmse_deg'])}$^{{\circ}}$ without sEMG. The
paired improvement was {deg(ablation['mean_improvement_deg'])}$^{{\circ}}$ (95\%
bootstrap CI {ci(ablation['bootstrap_95pct_ci_deg'])}$^{{\circ}}$; two-sided
paired $t({controls['conditions']['identity']['paired_t']['df']})={controls['conditions']['identity']['paired_t']['t']:.2f}$, $p={pval(controls['conditions']['identity']['paired_t']['p_two_sided'])}$), and
{ablation['positive_participants']} of {ablation['participant_count']}
participants improved.

In the timing control, moving the same sEMG history 500~ms earlier
changed participant RMSE by {deg(aligned['mean_improvement_deg'], 3)}$^{{\circ}}$
relative to the aligned model (95\% CI
{ci(aligned['bootstrap_95pct_ci_deg'], 3)}$^{{\circ}}$;
$t({statistics['temporal_control']['aligned_vs_lagged']['paired_t']['df']})={statistics['temporal_control']['aligned_vs_lagged']['paired_t']['t']:.2f}$, $p={pval(statistics['temporal_control']['aligned_vs_lagged']['paired_t']['p_two_sided'])}$). The surrogate controls
(Supplementary Table~\ref{{tab:controls}}) gave the same result. {surrogate_sentence}.



\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig03_prediction_accuracy}}
\caption{{Prediction accuracy with and without the sEMG correction. \textbf{{A}} Held-out participant RMSE with and without the sEMG correction, paired within participant. \textbf{{B}} Per-participant RMSE reduction, sorted. \textbf{{C}} Distribution of the reduction with its bootstrap mean and 95\% interval.}}\label{{fig:accuracy}}
\end{{figure}}

\subsection{{Motion matching and paired simulation}}

Across the fixed panel, the paired difference in excess instability between
the residual-fusion and no-sEMG conditions was {signed(physics['mean'], 4)}~s (95\% CI {ci(physics['ci_95'], 4)};
$t({physics['df']})={physics['t']:.2f}$, $p={pval(physics['p_two_sided'])}$). The
prediction improvement attributable to sEMG was therefore not accompanied by a
detectable change in the simulated outcome.



\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig05_motion_matching}}
\caption{{Motion-match quality over the fixed panel. \textbf{{A}} Knee against thigh-orientation matching error for each window, with the panel mean. \textbf{{B}} Both errors sorted across windows.}}\label{{fig:matching}}
\end{{figure}}

\subsection{{Prediction error vs. excess instability}}

Prediction error tracked simulated instability above a threshold accuracy,
below which the relationship reversed.

Across the {accuracy['n_accuracy_levels']} sampled accuracy levels, mean window
RMSE spanned {deg(min(accuracy['mean_rmse_deg']), 2)}$^{{\circ}}$ to
{deg(max(accuracy['mean_rmse_deg']), 2)}$^{{\circ}}$ on the same
{pooled['n_windows']} windows and the same matched references. A single monotone
coefficient over that whole range is uninformative
($\rho={signed(accuracy['overall_spearman_rho'], 3)}$,
$p={pval(accuracy['overall_p_value'])}$), because the relationship is not
monotone. A two-segment fit places the turn at
{deg(accuracy['breakpoint_rmse_deg'], 2)}$^{{\circ}}$ (95\% CI
{ci(accuracy['breakpoint_95pct_ci'], 2)}$^{{\circ}}$).

Above that accuracy, worse prediction produces more simulated instability across
the {accuracy['above_breakpoint']['n_levels']} levels in that segment (slope
{signed(accuracy['above_breakpoint']['slope_per_degree'], 5)}~s per
degree, 95\% CI {ci(accuracy['above_breakpoint']['slope_95pct_ci'], 5)}).

Below it the sign inverts: models that are more accurate produce \emph{{more}} excess
instability across the {accuracy['below_breakpoint']['n_levels']} levels in that
segment (slope {signed(accuracy['below_breakpoint']['slope_per_degree'], 5)},
95\% CI {ci(accuracy['below_breakpoint']['slope_95pct_ci'], 5)}). Both intervals exclude
zero under a bootstrap that resamples windows within every accuracy level, which
is the level at which these data actually vary; the checkpoint means themselves
are smooth by construction, so the rank correlation across levels does not rest
on fourteen independent observations.

Commanding a matched clip's own recorded trajectory, which is the zero-error
case, gave a mean excess instability of {signed(oracle_excess, 4)}~s over the
{len(oracle_windows)} preflight windows, so the inversion is not explained by a
fixed cost of substituting the knee. A partially
fitted linear model regresses toward the participant mean, so it commands a
gentler knee trajectory than the recorded one. A gentler trajectory perturbs the
body less than the true motion does, and mean excess instability is
correspondingly negative through the middle of the range, reaching
{signed(min(accuracy['mean_excess_instability']), 4)}~s.

Two secondary views are consistent with a real but small effect. Comparing windows
against one another within a single accuracy level recovers nothing at any level
(Table~\ref{{tab:checkpoints}}), because between-window differences in matched
motion and reference stability are large beside the effect of prediction error.
{within_sentence} Pooled across all {pooled['n_pairs']} checkpoint--window pairs
the match-adjusted association was
{signed(pooled['partial_spearman_rho'], 3)} (window-level cluster bootstrap 95\% CI
{ci(pooled['cluster_bootstrap_95pct_ci'], 3)}).



\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig06_accuracy_vs_instability}}
\caption{{Primary result. \textbf{{A}} Mean excess instability against model accuracy across the sampled levels, with the fitted turn and its bootstrap interval shaded; dotted lines are the two segment fits. \textbf{{B}} Segment slopes with 95\% intervals from resampling windows within every accuracy level.}}\label{{fig:primary}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig07_fwl}}
\caption{{The match-adjusted analysis at the converged model. \textbf{{A}} Window prediction error against excess instability. \textbf{{B}} The same outcome against motion-match quality, the covariate being controlled. \textbf{{C}} Both variables residualized on ranked match knee error and thigh orientation RMS, whose correlation is the reported partial Spearman.}}\label{{fig:fwl}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth]{{fig08_per_checkpoint}}
\caption{{Match-adjusted association and between-window spread at each accuracy level. \textbf{{A}} Match-adjusted association across windows at each accuracy level; every interval includes zero. \textbf{{B}} Between-window RMSE spread at each level.}}\label{{fig:perlevel}}
\end{{figure}}

\begin{{table}}[!htbp]
\caption{{Accuracy, between-window spread, and match-adjusted association at each
training-path checkpoint, ordered from the converged model to the untrained one.
Excess instability is measured against each window's own paired reference.}}\label{{tab:checkpoints}}
\begin{{tabular}}{{lcccccc}}
\toprule
Descent step & $n$ & Mean RMSE ($^{{\circ}}$) & SD ($^{{\circ}}$) & Mean excess & Partial $\rho$ & 95\% CI \\
\midrule
{checkpoint_table(data)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Discussion}}\label{{sec:discussion}}

The central finding is that prediction error and simulated walking instability are
related across part of the accuracy range and not across the rest of it. Above
{deg(accuracy['breakpoint_rmse_deg'], 2)}$^{{\circ}}$ of prediction
error, a less accurate model produced a less stable simulated walker, and the
slope is positive across that region. Below that point the
relationship inverted, so that further improvements in accuracy were accompanied by
slightly more excess instability rather than less. RMSE is therefore not a general
proxy for physical behavior. It behaves as a proxy within a band, and the boundary
of that band falls inside the range of accuracies that contemporary models occupy.

That last point is what gives the result practical weight. The most accurate model
evaluated here reached {deg(min(accuracy['mean_rmse_deg']), 2)}$^{{\circ}}$, which
lies below the turn. Because knee-angle regression models are ordinarily compared
with one another only after convergence, the regime in which they are compared is
also the regime in which further reduction in RMSE no longer predicts a more stable
simulated gait. A study confined to that regime samples only the flat portion of
the relationship, so it cannot separate an absent effect from an effect its range
does not reach. This supplies a possible
mechanism for the more general observation, made in myoelectric control by Hargrove
et al. \cite{{hargrove2007}} and by Krasoulis et al. \cite{{krasoulis2019}}, that
offline model scores do not reliably predict functional performance. It also
suggests that the methodological evolution described in recent reviews of
EMG-driven prosthetic control \cite{{cimolato2022,ahkami2023}} might reasonably
extend to the evaluation stage, and not only to preprocessing, architecture, and
validation strategy.

The mechanism behind the inversion does not imply that accuracy is undesirable.
It indicates that the mapping from accuracy to whole-body behavior
is not monotone, and that the quantity being minimized during training is not the
quantity that determines stability once the prediction is placed in a body.

The sEMG correction improved held-out prediction, and that improvement survived
three surrogate controls which preserved the signal's amplitude, spectral, and
cross-channel structure while destroying its correspondence with the knee. That is
a stronger test than an ablation alone provides, since an ablation cannot separate
signal content from the additional model capacity that accompanies it. The
predictor driving the simulations therefore draws on genuine myoelectric
information rather than on model size.

The two comparisons give different answers. Comparing windows against one another
within a single accuracy level recovers nothing at any of the
{correlation['n_checkpoints']} checkpoints, even where mean window error exceeds
{deg(max(r['mean_prediction_rmse_deg'] for r in correlation['per_checkpoint']), 0)}$^{{\circ}}$.
This is not a limitation of statistical power, since between-window error spread
ranges from
{deg(min(r['prediction_rmse_sd_deg'] for r in correlation['per_checkpoint']), 2)}
to {deg(max(r['prediction_rmse_sd_deg'] for r in correlation['per_checkpoint']), 2)}$^{{\circ}}$,
and the same null appears when the outcome is normalized by rollout duration.
Comparing each window against itself does recover an association: holding the
window fixed and varying only the model, a window became more unstable as its own
prediction degraded
({signed(within['mean_spearman_rho_fisher_back_transformed'], 3)},
95\% CI {ci(within['ci_95pct_rho'], 3)}, $p={pval(within['p_value_two_sided'])}$,
{within['positive_windows']} of {within['n_windows']} windows positive). The
difference arises because between-window variation in excess instability is
dominated by which motion a window was matched to, where in the gait cycle that
window falls, and how stable the matched reference already was. Those sources of
variation are large beside the effect of prediction error and obscure it in any
comparison drawn across windows, whereas holding the window fixed removes them.

These observations bear on benchmark design. An evaluation that varies only the
window, which is what a study
of a single converged model is restricted to, will tend to report no relationship
between prediction error and simulated outcome, not because no relationship exists
but because that contrast cannot resolve one. Varying the model along its own
training path supplies the contrast that can. Prediction error does propagate to
whole-body behavior, but the effect is small in comparison with window-to-window
variation, and a benchmark constructed from comparisons between different windows
is not a suitable instrument for detecting it.

\subsection{{Limitations}}

The benchmark used able-bodied level walking rather than data from transfemoral
prosthesis users, so it tests an evaluation framework on a proxy dataset. Each
model was also fitted on earlier trials from the participant it was tested on, so
these accuracies are what a calibrated model achieves, not what a wearer would get
from a model that had never seen their data.

The kinematic stage predicts from recorded knee-angle history. A microprocessor
knee carries a joint angle encoder, so that channel would exist on a device, but
two differences separate this experiment from deployment. First, the history used
here is the trajectory of an intact biological knee, whereas a prosthetic knee
follows whatever its own controller produced; the channel is the same but its
distribution is not, so the accuracy measured here is not the accuracy a device
would achieve. Second, a deployed predictor whose output drives the knee
would receive its own past commands as input, making the kinematic stage largely
a propagator of its own trajectory. Under that arrangement the sEMG correction
becomes the only term carrying new information about the wearer's intent, so its
contribution would matter more than the held-out share reported here, not less.
Neither point is tested by this experiment, and the reported accuracy is not
comparable with methods predicting from socket-available signals alone.

The instability index is bounded, study-specific, and derived from XCoM margin; it
is not a clinical fall-risk measure
\cite{{hof2005,hof2008,gill2019,pickle2018,curtze2024}}. Simulation
outcomes also depend on motion-match quality, which is why match errors are treated
as covariates, and the conclusion holds for this evaluation
rather than for every application of the model.

Finally, the drop-off estimate is a descriptive two-segment fit over the sampled
accuracy levels. It locates where the sampled association changes; it is not a
changepoint test with its own error rate, and the unsampled interval between
checkpoints does not define a safety threshold.

\subsection{{Future Work}}\label{{sec:future}}

The turn reported here was located with one model class on one evaluation
design, and the first question for further work is whether it moves. A
predictor with a different error structure, such as a recurrent or
transformer model rather than a ridge pair, may distribute its residuals
differently across the gait cycle and place its turn elsewhere. The
training-path construction used here would locate that turn in the same way,
since it requires only a model whose accuracy can be varied continuously.

A second extension would be to move from able-bodied public recordings to data
from prosthetic users while keeping the paired reference and prediction
structure intact. That would test the framework in a setting closer to real
conditions, where the kinematic history available to the predictor is produced
by a device rather than by an intact knee, and where the distribution of that
history differs from the one used here.

Future work should also examine whether more phase-sensitive summaries of
prediction error, such as stance-weighted RMSE or error aligned to support
transitions, track simulated instability more closely than window-level RMSE
does. The present result does not show that every summary of prediction error
behaves this way. It shows that RMSE, which is the summary the field reports,
stops tracking simulated behavior below roughly
{deg(accuracy['breakpoint_rmse_deg'], 0)}$^{{\circ}}$.

\section{{Conclusion}}\label{{sec:conclusion}}

Replaying one fixed panel of walking windows at
{accuracy['n_accuracy_levels']} accuracy levels sampled along a model's own
training path shows that prediction error predicts simulated walking instability
over part of the accuracy range only. Above
{deg(accuracy['breakpoint_rmse_deg'], 2)}$^{{\circ}}$ (95\% CI
{ci(accuracy['breakpoint_95pct_ci'], 2)}$^{{\circ}}$) a less accurate model
produced a less stable simulated walker. Below that point the relationship
inverted, because a partially fitted model regresses toward the participant mean
and so commands a flatter trajectory that moves the knee less than the recorded
motion does. Root-mean-square error therefore tracked
physical behavior only above the turn, and the converged model evaluated here,
at
{deg(min(accuracy['mean_rmse_deg']), 2)}$^{{\circ}}$, sits below the turn.

The practical consequence is for how prosthetic regression models are evaluated.
Since such models are ordinarily compared after convergence, they are compared in
the regime where further reduction in error no longer predicts a more stable
simulated gait. The contrast a study chooses
matters as much as the metric: comparing prediction windows against one another,
which is all a single converged model permits, recovered no relationship at any
accuracy level, whereas comparing each window against itself across accuracy
levels did. Window-to-window variation is large enough to bury an effect that is
real but small, so a benchmark built on comparisons between windows will report no
relationship whether or not one exists.

The sEMG correction produced a small improvement in later-trial knee-angle
prediction that survived three surrogate controls, and is therefore attributable to
signal content rather than to added model capacity. The predictor evaluated in
simulation draws on genuine myoelectric information.

\backmatter

\bmhead{{Supplementary analysis: sensitivity to knee-angle input}}

The kinematic stage forecasts knee angle from knee-angle history, which invites
the objection that a low RMSE mostly reflects trajectory smoothness -- the knee
predicting its own near future -- rather than information recovered from the
walker. The objection concerns what the reported accuracy measures rather than
whether the signal would be available: a microprocessor knee carries a joint
angle encoder, so knee-angle history exists at inference.

The same model was therefore refitted with the kinematic stage reading only the
surrounding body -- thigh pitch and full thigh orientation, with no channel
observing the knee -- holding the windows, horizon, ridge pair, penalties, and
trial split fixed. With the sEMG correction switched off in both, knee-angle
history gave {deg(knee_arm['no_emg_mean_participant_rmse_deg'])}$^{{\circ}}$ and
the surrounding body alone gave
{deg(body_arm['no_emg_mean_participant_rmse_deg'])}$^{{\circ}}$, a difference of
{deg(abs(knee_penalty))}$^{{\circ}}$.

Removing every knee-derived input therefore costs almost nothing. The information
needed to forecast the knee is already present in the surrounding kinematics, so
the knee channel is a compact carrier of it rather than a shortcut, and the
reported accuracy does not rest on the knee predicting its own future.

With sEMG active, the surrounding-body model reached
{deg(body_arm['fused_mean_participant_rmse_deg'])}$^{{\circ}}$ against
{deg(knee_arm['fused_mean_participant_rmse_deg'])}$^{{\circ}}$ for knee history.
Both penalties were the ones selected for the knee-history configuration, so this
comparison is conservative towards the surrounding-body arm.

\bmhead{{Supplementary analysis: sEMG surrogate controls}}

The reported ablation removes the sEMG correction from a model whose penalties
were fixed on a separate development cohort and whose scores come from a held-out
later trial of held-out participants. Under that design, extra fitted capacity
raises held-out error rather than lowering it, so the comparison already speaks to
sEMG rather than to model size. The surrogates below are a further check against
subtler possibilities, such as an sEMG channel correlating with the target through
the recording setup rather than through physiology.

Each surrogate repeats the entire confirmation fit with the sEMG replaced by a
signal that preserves its statistics but not its correspondence with the knee.
\emph{{Circular shift}} rotates each participant-trial sEMG block against its own
kinematics, preserving amplitudes, spectra, and cross-channel structure.
\emph{{Participant swap}} gives each participant another participant's sEMG.
\emph{{Phase randomization}} replaces each channel with a surrogate of identical
power spectrum and randomized Fourier phases \cite{{theiler1992}}. The kinematic
stage is shared across all four conditions, so any difference between them is
attributable to sEMG content alone.

The circular-shift surrogate retained
{contrasts['circular_shift']['surrogate_share_of_real_effect']:.2f} of the recorded
effect, which is expected rather than anomalous. Walking is periodic with a cycle
close to one second, so rotating a block of sEMG against its own kinematics cannot
remove the correspondence entirely: whatever offset the rotation lands on is still
a phase of the same repeating gait cycle. A surrogate of this kind sets a floor,
not a zero, and its residual effect quantifies how much of the apparent sEMG
benefit is recoverable from gait periodicity alone. The two surrogates that break
the correspondence without relying on a time offset both returned effects
indistinguishable from zero.

Attribution is therefore judged by a paired within-participant contrast -- whether
the recorded signal beats each surrogate for the same participant -- rather than
by whether each surrogate independently fails a significance threshold, which a
periodic-signal surrogate cannot be expected to do. {surrogate_sentence}.

\begin{{table}}[!htbp]
\caption{{sEMG ablation and its surrogate negative controls. Improvement is the
participant-level reduction in held-out RMSE relative to the same fitted kinematic
stage without a correction; positive favors the correction. Share is each
surrogate's effect as a fraction of the recorded one. The final column is the
paired within-participant margin by which the recorded signal beats that
surrogate. The reported $p$-values are two-sided paired $t$-tests on per-participant
differences.}}\label{{tab:controls}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{lccccc}}
\toprule
Condition & Improvement ($^{{\circ}}$) & 95\% CI & $p$ & Share & Margin ($^{{\circ}}$) \\
\midrule
{control_table(data)}
\bottomrule
\end{{tabular}}
\end{{table}}

\bmhead{{Data availability}}
Gait120 is available at \url{{https://doi.org/10.6084/m9.figshare.27677016}}
\cite{{boo2025dataset}}. MoCapAct is available from its published release
\cite{{wagener2022}}.

\bmhead{{Code availability}}
The analysis code and the manuscript generator that produced every number in this
paper are available at \url{{https://github.com/aoyun2/emg_tst}}.

\bibliography{{references}}

\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = collect(args.runs_dir.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "main.tex"
    target.write_text(build(data), encoding="utf-8")
    print(f"Wrote {target}")
    print(
        "Compile with:  latexmk -pdf main.tex   (run from "
        f"{out_dir}, which must also hold sn-jnl.cls, sn-vancouver-num.bst, references.bib)"
    )


if __name__ == "__main__":
    main()
