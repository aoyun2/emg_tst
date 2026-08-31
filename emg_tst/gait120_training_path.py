"""Gradient-descent training path for the residual-fusion predictor.

The confirmation experiment fits the residual-fusion model in closed form, which
gives a single converged model and no trajectory to sample. This module fits the
same model class by full-batch gradient descent on the same participant-balanced
ridge objective, from a zero coefficient vector.

At step zero the kinematic stage predicts the participant-balanced target mean
and the residual correction is zero. Checkpoints are sampled along the descent;
the closed-form solution is appended as a separate terminal checkpoint, and the
relative gap between it and the last descent iterate is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_RESIDUAL_FUSION_GRADIENT_TRAINING_PATH_V1"

# Step size as a fraction of the stability bound 2/L, where L is the largest
# curvature of the objective.
STEP_SIZE_SAFETY = 0.005

MAX_STEPS = 400_000

# Descent stops when the gradient norm falls to this fraction of its starting
# value.
CONVERGENCE_TOLERANCE = 1.0e-8

# Number of sampled checkpoints, including the untrained and converged ends.
N_CHECKPOINTS = 14

CHECKPOINT_MODES = ("rmse_ladder", "log_step")


@dataclass(frozen=True)
class StageProblem:
    """Standardized least-squares problem for one residual-fusion stage.

    ``normal`` and ``rhs`` are the normal-equation terms Z'VZ + alpha*I and
    Z'Vy.  The descent uses them instead of touching Z each step: the gradient of
    the ridge objective is exactly ``2(normal @ w - rhs)``, so a step costs
    O(features^2) rather than O(rows * features).  With hundreds of thousands of
    training rows and hundreds of thousands of steps that is the difference
    between seconds and days, and it is an algebraic identity, not an
    approximation.
    """

    z: np.ndarray
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    alpha: float
    step_size: float
    normal: np.ndarray
    rhs: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.z.shape[1])


def _standardize(
    x: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted standardization matching ``run_gait120_residual_fusion._fit_ridge``."""
    weight_sum = float(np.sum(weights))
    mean = np.sum(weights[:, None] * x, axis=0) / weight_sum
    centered = x - mean[None, :]
    variance = np.sum(weights[:, None] * np.square(centered), axis=0) / weight_sum
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1.0e-8] = 1.0
    return centered / std[None, :], mean, std


def build_stage(
    x: np.ndarray, subject_number: np.ndarray, *, alpha: float, y_centered: np.ndarray
) -> StageProblem:
    """Prepare one stage of the descent, including its stable step size."""
    x = np.asarray(x, dtype=np.float64)
    weights = fusion._participant_balanced_weights(subject_number)
    z, mean, std = _standardize(x, weights)
    zw = z * np.sqrt(weights)[:, None]
    gram = zw.T @ zw
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    # Objective is ||sqrt(V)(y - Zw)||^2 + alpha||w||^2, so curvature is 2*gram.
    curvature = 2.0 * float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise RuntimeError("Stage curvature is not positive; cannot set a step size")
    return StageProblem(
        z=z,
        weights=weights,
        feature_mean=mean,
        feature_std=std,
        alpha=float(alpha),
        step_size=float(2.0 * STEP_SIZE_SAFETY / curvature),
        normal=gram,
        rhs=z.T @ (weights * np.asarray(y_centered, dtype=np.float64)),
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def _stage_gradient(
    stage: StageProblem, coefficient: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Gradient of the ridge objective from the precomputed normal equations."""
    return 2.0 * (stage.normal @ coefficient - rhs)


def _as_ridge_model(
    stage: StageProblem, coefficient: np.ndarray, intercept: float
) -> fusion.RidgeModel:
    return fusion.RidgeModel(
        feature_mean=stage.feature_mean,
        feature_std=stage.feature_std,
        coefficient=np.asarray(coefficient, dtype=np.float64),
        intercept=float(intercept),
        alpha=float(stage.alpha),
    )


def closed_form_coefficient(stage: StageProblem, rhs: np.ndarray) -> np.ndarray:
    """Solve the same stage exactly, for the terminal checkpoint."""
    return np.linalg.solve(stage.normal, np.asarray(rhs, dtype=np.float64))


@dataclass(frozen=True)
class ErrorQuadratic:
    """Training RMSE in degrees as a quadratic in the stacked coefficients.

    The fused residual is ``Z_k w_kin + Z_e w_res - y_centered`` and the error in
    degrees rescales it per row, so mean squared error is ``w'Mw - 2g'w + h``.
    Precomputing M, g, and h keeps the per-step cost independent of how many
    training rows there are.
    """

    m: np.ndarray
    g: np.ndarray
    h: float

    def rmse(self, coefficient: np.ndarray) -> float:
        value = float(
            coefficient @ self.m @ coefficient - 2.0 * self.g @ coefficient + self.h
        )
        return float(np.sqrt(max(value, 0.0)))


@dataclass
class TrainingPath:
    """Recorded coefficients and training accuracy along the descent.

    ``steps`` and ``train_rmse_deg`` cover every step of the descent.  The
    coefficient arrays cover only the selected checkpoints, listed in
    ``checkpoint_indices`` as positions within ``steps``; a descent long enough
    to resolve the early error drop has far too many steps to keep a coefficient
    vector for each.  The final entry is the exact closed-form solution rather
    than the last iterate, so the ladder ends on the model the confirmation run
    reports.
    """

    steps: np.ndarray
    train_rmse_deg: np.ndarray
    checkpoint_indices: np.ndarray
    kinematic_coefficients: np.ndarray
    residual_coefficients: np.ndarray
    kinematic_intercept: float
    residual_intercept_path: np.ndarray
    kinematic_stage: StageProblem
    residual_stage: StageProblem
    converged_step: int
    closed_form_gap: float


def _descend(
    kinematic_stage: StageProblem,
    residual_stage: StageProblem,
    cross: np.ndarray,
    objective: ErrorQuadratic,
    *,
    max_steps: int,
    capture: set[int] | None,
):
    """Run the coupled descent, yielding state at step 0 and each step after.

    The kinematic stage descends on the standardized target.  The residual stage
    descends on the *current* kinematic residual, so the two co-evolve.  Once the
    kinematic stage settles its residual is fixed, and the residual stage then
    converges on exactly the target the sequential closed-form fit uses, so the
    descent and the confirmation fit share a fixed point.

    The residual stage's right-hand side is ``Z_e'V(y_c - Z_k w_kin)``, which is
    ``rhs - cross @ w_kin`` with ``cross = Z_e'V Z_k`` precomputed, so tracking
    the moving target costs a small matrix-vector product rather than a pass over
    the training rows.

    ``capture`` selects which steps yield a coefficient copy; every step yields
    its training error regardless.
    """
    w_kin = np.zeros(kinematic_stage.n_features, dtype=np.float64)
    w_res = np.zeros(residual_stage.n_features, dtype=np.float64)

    initial_gradient = (
        float(np.linalg.norm(_stage_gradient(kinematic_stage, w_kin, kinematic_stage.rhs)))
        + 1.0e-30
    )

    for step in range(int(max_steps) + 1):
        wanted = capture is None or int(step) in capture
        yield (
            int(step),
            objective.rmse(np.concatenate([w_kin, w_res])),
            (w_kin.copy() if wanted else None),
            (w_res.copy() if wanted else None),
        )

        grad_kin = _stage_gradient(kinematic_stage, w_kin, kinematic_stage.rhs)
        if float(np.linalg.norm(grad_kin)) / initial_gradient < CONVERGENCE_TOLERANCE:
            return
        grad_res = _stage_gradient(
            residual_stage, w_res, residual_stage.rhs - cross @ w_kin
        )
        w_kin = w_kin - kinematic_stage.step_size * grad_kin
        w_res = w_res - residual_stage.step_size * grad_res


def fit_training_path(
    train: gait.ExampleSet,
    *,
    kinematic_alpha: float,
    residual_alpha: float,
    max_steps: int = MAX_STEPS,
    checkpoint_mode: str = "rmse_ladder",
    n_checkpoints: int = N_CHECKPOINTS,
) -> TrainingPath:
    """Trace the descent, choose checkpoints, then materialize just those.

    The descent runs twice.  The first pass records the training error at every
    step, which is what the checkpoint schedule is chosen from.  The second pass
    repeats the identical deterministic descent and keeps coefficients only at
    the chosen steps, so resolving the path finely costs time rather than memory.
    """
    train_k, train_e = fusion._features(train)
    y = np.asarray(train.y_standardized, dtype=np.float64)

    weights = fusion._participant_balanced_weights(train.subject_number)
    kinematic_intercept = _weighted_mean(y, weights)
    y_centered = y - kinematic_intercept

    kinematic_stage = build_stage(
        train_k, train.subject_number, alpha=kinematic_alpha, y_centered=y_centered
    )
    residual_stage = build_stage(
        train_e, train.subject_number, alpha=residual_alpha, y_centered=y_centered
    )
    # Z_k is weighted-centered, so the residual intercept is zero at every step.
    residual_intercept = _weighted_mean(y - kinematic_intercept, weights)
    if abs(residual_intercept) > 1.0e-8:
        raise RuntimeError(
            f"Residual stage intercept is {residual_intercept:.3e}, expected zero; "
            "the kinematic design is not weighted-centered."
        )
    cross = residual_stage.z.T @ (weights[:, None] * kinematic_stage.z)

    target_std = np.asarray(train.target_std_deg, dtype=np.float64)
    scale = np.square(target_std)
    stacked = np.concatenate([kinematic_stage.z, residual_stage.z], axis=1)
    n_rows = float(stacked.shape[0])
    objective = ErrorQuadratic(
        m=(stacked.T @ (scale[:, None] * stacked)) / n_rows,
        g=(stacked.T @ (scale * y_centered)) / n_rows,
        h=float(np.sum(scale * np.square(y_centered)) / n_rows),
    )

    steps: list[int] = []
    rmse: list[float] = []
    for step, error, _w_kin, _w_res in _descend(
        kinematic_stage,
        residual_stage,
        cross,
        objective,
        max_steps=max_steps,
        capture=set(),
    ):
        steps.append(step)
        rmse.append(error)
    converged_step = int(steps[-1])

    # Appends the closed-form solution as the terminal rung.
    exact_kin = closed_form_coefficient(kinematic_stage, kinematic_stage.rhs)
    exact_res = closed_form_coefficient(
        residual_stage, residual_stage.rhs - cross @ exact_kin
    )
    exact_intercept = 0.0

    steps.append(converged_step + 1)
    rmse.append(objective.rmse(np.concatenate([exact_kin, exact_res])))

    provisional = TrainingPath(
        steps=np.asarray(steps, dtype=np.int64),
        train_rmse_deg=np.asarray(rmse, dtype=np.float64),
        checkpoint_indices=np.zeros(0, dtype=np.int64),
        kinematic_coefficients=np.zeros((0, kinematic_stage.n_features)),
        residual_coefficients=np.zeros((0, residual_stage.n_features)),
        kinematic_intercept=float(kinematic_intercept),
        residual_intercept_path=np.zeros(0, dtype=np.float64),
        kinematic_stage=kinematic_stage,
        residual_stage=residual_stage,
        converged_step=converged_step,
        closed_form_gap=float("nan"),
    )
    picks = select_checkpoints(
        provisional, mode=checkpoint_mode, count=int(n_checkpoints)
    )

    terminal_index = int(provisional.steps.size) - 1
    wanted_steps = {int(provisional.steps[i]) for i in picks.tolist() if i != terminal_index}
    # Keep the last iterate so the gap diagnostic below is defined.
    wanted_steps.add(converged_step)
    captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step, _error, w_kin, w_res in _descend(
        kinematic_stage,
        residual_stage,
        cross,
        objective,
        max_steps=max_steps,
        capture=wanted_steps,
    ):
        if w_kin is not None and w_res is not None:
            captured[int(step)] = (w_kin, w_res)

    kin_rows: list[np.ndarray] = []
    res_rows: list[np.ndarray] = []
    intercepts: list[float] = []
    for i in picks.tolist():
        if i == terminal_index:
            kin_rows.append(exact_kin)
            res_rows.append(exact_res)
        else:
            w_kin, w_res = captured[int(provisional.steps[i])]
            kin_rows.append(w_kin)
            res_rows.append(w_res)
        intercepts.append(float(exact_intercept))

    # Relative gap between the last descent iterate and the closed-form
    # solution.
    gap = float(
        np.linalg.norm(captured[converged_step][0] - exact_kin)
        / (np.linalg.norm(exact_kin) + 1.0e-12)
    )

    return TrainingPath(
        steps=provisional.steps,
        train_rmse_deg=provisional.train_rmse_deg,
        checkpoint_indices=picks,
        kinematic_coefficients=np.asarray(kin_rows, dtype=np.float64),
        residual_coefficients=np.asarray(res_rows, dtype=np.float64),
        kinematic_intercept=float(kinematic_intercept),
        residual_intercept_path=np.asarray(intercepts, dtype=np.float64),
        kinematic_stage=kinematic_stage,
        residual_stage=residual_stage,
        converged_step=converged_step,
        closed_form_gap=gap,
    )


def select_checkpoints(
    path: TrainingPath, *, mode: str = "rmse_ladder", count: int = N_CHECKPOINTS
) -> np.ndarray:
    """Choose which recorded points along the path become checkpoints.

    ``rmse_ladder`` tiles the achieved training-RMSE range as evenly as the path
    allows.  Because descent spends most of its steps near convergence, a
    step-indexed schedule crowds the ladder into the converged end and leaves the
    transition region -- the part the drop-off analysis is about -- resolved by
    one or two points.  ``log_step`` keeps the plain step-indexed schedule for
    comparison.
    """
    mode = str(mode).strip().lower()
    if mode not in CHECKPOINT_MODES:
        raise ValueError(f"Unsupported checkpoint mode {mode!r}")
    n_recorded = int(path.steps.size)
    count = int(min(max(2, count), n_recorded))

    if mode == "log_step":
        targets = np.unique(
            np.round(np.geomspace(1.0, float(path.steps[-1] + 1), num=count)).astype(np.int64)
        )
        picks = [int(np.argmin(np.abs(path.steps - (target - 1)))) for target in targets]
    else:
        rmse = np.asarray(path.train_rmse_deg, dtype=np.float64)
        levels = np.linspace(float(rmse[-1]), float(rmse[0]), num=count)
        picks = [int(np.argmin(np.abs(rmse - level))) for level in levels]
    return np.unique(np.asarray([0, *picks, n_recorded - 1], dtype=np.int64))


def checkpoint_label(order: int, step: int) -> str:
    return f"step_{int(order):02d}_{int(step):08d}"


def materialize_checkpoint(
    path: TrainingPath, order: int, train: gait.ExampleSet
) -> tuple[fusion.RidgeModel, fusion.RidgeModel, dict[int, float]]:
    """Rebuild the two ridge stages and the correction caps at one checkpoint.

    ``order`` indexes the checkpoint ladder, not the full step trajectory.

    Correction caps are recomputed from the checkpoint's own kinematic residual,
    matching the confirmation protocol, so an early checkpoint is not bounded by
    a converged model's residual scale.
    """
    kinematic = _as_ridge_model(
        path.kinematic_stage,
        path.kinematic_coefficients[int(order)],
        path.kinematic_intercept,
    )
    residual = _as_ridge_model(
        path.residual_stage,
        path.residual_coefficients[int(order)],
        float(path.residual_intercept_path[int(order)]),
    )
    train_k, _ = fusion._features(train)
    base = fusion._predict_standardized(kinematic, train_k)
    caps = fusion._training_correction_caps(train, base)
    return kinematic, residual, caps
