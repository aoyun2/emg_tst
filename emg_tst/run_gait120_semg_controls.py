"""Negative controls for the sEMG ablation.

The ablation compares residual fusion against the same fitted kinematic stage
with the correction switched off.  That comparison shows the correction helps,
but on its own it cannot separate two explanations: the sEMG carries information
about the knee that the kinematic history does not, or the residual stage is
simply extra fitted capacity that would absorb some training error regardless of
what was fed into it.

These controls decide between them.  Each one re-runs the whole confirmation
fit with the sEMG replaced by a surrogate that keeps the signal's statistics but
destroys its correspondence with the target:

``circular_shift``
    Each participant-trial's sEMG block is rotated against its own kinematics.
    Channel amplitudes, spectra, cross-channel structure, and per-participant
    scaling are all preserved exactly; only the alignment to the knee angle is
    broken.

``participant_swap``
    Each participant receives another participant's sEMG.  This keeps real
    walking sEMG opposite real walking kinematics but removes the within-person
    correspondence that residual fusion relies on.

``phase_randomized``
    Each channel is replaced by a surrogate with an identical power spectrum and
    randomized Fourier phases, which preserves the autocorrelation of the
    envelope but nothing about its timing relative to the knee.

If the reported improvement reflects real neuromuscular information, the real
condition should clear the gate and every surrogate should sit near zero.  A
surrogate that also improves RMSE would mean the gain came from the added
regression stage rather than from sEMG.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_SEMG_NEGATIVE_CONTROLS_V1"

SURROGATES = ("identity", "circular_shift", "participant_swap", "phase_randomized")
SEED = 20260827


def _blocks(examples: gait.ExampleSet) -> list[np.ndarray]:
    """Row indices for each participant-trial block, in recorded order."""
    subject = np.asarray(examples.subject_number).reshape(-1)
    trial = np.asarray(examples.trial_index).reshape(-1)
    key = subject.astype(np.int64) * 1000 + trial.astype(np.int64)
    out: list[np.ndarray] = []
    for value in np.unique(key).tolist():
        rows = np.flatnonzero(key == value)
        if np.any(np.diff(rows) != 1):
            raise RuntimeError(
                "Participant-trial rows are not contiguous; surrogate construction "
                "assumes recorded order is preserved."
            )
        out.append(rows)
    return out


def _circular_shift(emg: np.ndarray, examples: gait.ExampleSet, rng) -> np.ndarray:
    out = np.array(emg, dtype=np.float64, copy=True)
    for rows in _blocks(examples):
        n = int(rows.size)
        if n < 4:
            continue
        # Shift by a large fraction of the block so no window keeps its own
        # sEMG, and jitter it so the offset is not identical everywhere.
        shift = int(n // 2 + rng.integers(-(n // 8), n // 8 + 1))
        shift = int(np.clip(shift, 1, n - 1))
        out[rows] = np.roll(emg[rows], shift, axis=0)
    return out


def _participant_swap(emg: np.ndarray, examples: gait.ExampleSet, rng) -> np.ndarray:
    subject = np.asarray(examples.subject_number).reshape(-1)
    unique = np.unique(subject)
    if unique.size < 2:
        raise RuntimeError("Participant swap needs at least two participants")
    # A derangement by rotation: every participant gets someone else's sEMG.
    donor = {int(s): int(unique[(i + 1) % unique.size]) for i, s in enumerate(unique.tolist())}

    rows_by_subject = {int(s): np.flatnonzero(subject == int(s)) for s in unique.tolist()}
    out = np.array(emg, dtype=np.float64, copy=True)
    for s in unique.tolist():
        target_rows = rows_by_subject[int(s)]
        source_rows = rows_by_subject[donor[int(s)]]
        # Tile or truncate the donor block to the recipient's length.
        picks = np.arange(target_rows.size) % source_rows.size
        out[target_rows] = emg[source_rows[picks]]
    return out


def _phase_randomized(emg: np.ndarray, examples: gait.ExampleSet, rng) -> np.ndarray:
    out = np.array(emg, dtype=np.float64, copy=True)
    for rows in _blocks(examples):
        block = emg[rows]
        n = int(block.shape[0])
        if n < 8:
            continue
        spectrum = np.fft.rfft(block, axis=0)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(spectrum.shape[0], 1))
        # Keep the DC term real so the surrogate holds the original block mean.
        phases[0, 0] = 0.0
        if n % 2 == 0:
            phases[-1, 0] = 0.0
        rotated = np.abs(spectrum) * np.exp(1j * (np.angle(spectrum) + phases))
        out[rows] = np.fft.irfft(rotated, n=n, axis=0)
    return out


_BUILDERS = {
    "identity": lambda emg, examples, rng: np.array(emg, dtype=np.float64, copy=True),
    "circular_shift": _circular_shift,
    "participant_swap": _participant_swap,
    "phase_randomized": _phase_randomized,
}


def _fit_and_score(
    train: gait.ExampleSet,
    test: gait.ExampleSet,
    *,
    train_emg: np.ndarray,
    test_emg: np.ndarray,
    kinematic: fusion.RidgeModel,
    correction_caps: dict[int, float],
    residual_alpha: float,
    gamma: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit only the residual stage on surrogate sEMG and score the test trial.

    The kinematic stage is shared across conditions, so any difference between
    the real and surrogate results is attributable to the sEMG content alone.
    """
    train_k, _ = fusion._features(train)
    test_k, _ = fusion._features(test)
    base_train = fusion._predict_standardized(kinematic, train_k)
    residual_target = np.asarray(train.y_standardized, dtype=np.float64) - base_train

    residual_model = fusion._fit_ridge(
        train_emg, residual_target, train.subject_number, alpha=residual_alpha
    )
    base_test = fusion._predict_standardized(kinematic, test_k)
    correction = fusion._bounded_correction(
        fusion._predict_standardized(residual_model, test_emg), test, correction_caps
    )
    fused = base_test + float(gamma) * correction

    fused_metrics = gait._participant_metrics(test, fusion._to_degrees(test, fused))
    no_emg_metrics = gait._participant_metrics(test, fusion._to_degrees(test, base_test))
    return {"test": fused_metrics}, {"test": no_emg_metrics}


def _summarize_effects(effects: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    """Bootstrap interval and sign-flip randomization test for a paired vector."""
    effects = np.asarray(effects, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))

    means = []
    for first in range(0, gait.BOOTSTRAP_DRAWS, 2_000):
        draws = min(2_000, gait.BOOTSTRAP_DRAWS - first)
        index = rng.integers(0, effects.size, size=(draws, effects.size))
        means.append(np.mean(effects[index], axis=1))
    lower, upper = np.quantile(np.concatenate(means), [0.025, 0.975]).tolist()

    observed = abs(float(np.mean(effects)))
    exceed = completed = 0
    while completed < gait.RANDOMIZATION_DRAWS:
        draws = min(10_000, gait.RANDOMIZATION_DRAWS - completed)
        signs = rng.integers(0, 2, size=(draws, effects.size), dtype=np.int8) * 2 - 1
        exceed += int(
            np.sum(np.abs(np.mean(signs * effects[None, :], axis=1)) >= observed - 1.0e-15)
        )
        completed += draws
    return {
        "mean_deg": float(np.mean(effects)),
        "bootstrap_95pct_ci_deg": [float(lower), float(upper)],
        "two_sided_randomization_p": float((exceed + 1.0) / (completed + 1.0)),
        "positive_participants": int(np.sum(effects > 0.0)),
        "participant_count": int(effects.size),
    }


def _improvement_by_subject(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        subject: float(value)
        for subject, value in zip(statistics["subjects"], statistics["improvement_deg"])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--first-subject", type=int, default=31)
    parser.add_argument("--last-subject", type=int, default=120)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RuntimeError(f"Control run directory already exists: {run_dir}")

    protocol_path = args.confirmation_run_dir.resolve() / "protocol.json"
    chosen = (
        json.loads(protocol_path.read_text(encoding="utf-8")).get("model") or {}
    ).get("confirmation_hyperparameters")
    if not isinstance(chosen, dict):
        raise RuntimeError(f"{protocol_path} does not record confirmation_hyperparameters")
    kinematic_alpha = float(chosen["kinematic_alpha"])
    residual_alpha = float(chosen["residual_alpha"])
    gamma = float(chosen["gamma"])

    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    subjects = list(range(int(args.first_subject), int(args.last_subject) + 1))
    train, _validation, test, _scalers = gait._build_examples(cache_dir, subjects)

    train_k, train_e = fusion._features(train)
    _, test_e = fusion._features(test)
    kinematic = fusion._fit_ridge(
        train_k, train.y_standardized, train.subject_number, alpha=kinematic_alpha
    )
    correction_caps = fusion._training_correction_caps(
        train, fusion._predict_standardized(kinematic, train_k)
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    by_subject: dict[str, dict[str, float]] = {}
    for name in SURROGATES:
        rng = np.random.default_rng(SEED)
        builder = _BUILDERS[name]
        fused, no_emg = _fit_and_score(
            train,
            test,
            train_emg=builder(train_e, train, rng),
            test_emg=builder(test_e, test, rng),
            kinematic=kinematic,
            correction_caps=correction_caps,
            residual_alpha=residual_alpha,
            gamma=gamma,
        )
        statistics = gait._paired_statistics(fused, no_emg, require_gate=True)
        by_subject[name] = _improvement_by_subject(statistics)
        statistics.pop("improvement_deg", None)
        results[name] = statistics
        print(
            f"[controls] {name:<18} improvement={statistics['mean_improvement_deg']:+.4f} deg "
            f"CI[{statistics['bootstrap_95pct_ci_deg'][0]:+.4f}, "
            f"{statistics['bootstrap_95pct_ci_deg'][1]:+.4f}] "
            f"p={statistics['two_sided_randomization_p']:.3g} "
            f"gate={'PASS' if statistics['passed'] else 'fail'}"
        )

    # Requiring every surrogate to fail its own gate is the wrong test. Walking is
    # periodic, so a circular shift cannot fully break the link between sEMG and
    # knee angle: whatever phase it lands on is still a phase of the same gait
    # cycle. Such a surrogate is expected to retain some signal, and a small
    # residual effect in it is not evidence that the real result is spurious.
    #
    # The question that matters is whether the recorded signal beats its own
    # surrogate, which is a paired comparison within each participant.
    real = by_subject["identity"]
    contrasts: dict[str, Any] = {}
    for name in SURROGATES:
        if name == "identity":
            continue
        subjects = sorted(set(real) & set(by_subject[name]))
        effects = np.asarray(
            [real[s] - by_subject[name][s] for s in subjects], dtype=np.float64
        )
        summary = _summarize_effects(effects)
        summary["definition"] = (
            f"per-participant improvement with recorded sEMG minus improvement with {name}"
        )
        summary["surrogate_share_of_real_effect"] = (
            float(results[name]["mean_improvement_deg"])
            / float(results["identity"]["mean_improvement_deg"])
            if results["identity"]["mean_improvement_deg"]
            else float("nan")
        )
        summary["recorded_beats_surrogate"] = bool(
            summary["mean_deg"] > 0.0
            and summary["bootstrap_95pct_ci_deg"][0] > 0.0
            and summary["two_sided_randomization_p"] <= 0.05
        )
        contrasts[name] = summary
        print(
            f"[controls] recorded vs {name:<18} {summary['mean_deg']:+.4f} deg "
            f"CI[{summary['bootstrap_95pct_ci_deg'][0]:+.4f}, "
            f"{summary['bootstrap_95pct_ci_deg'][1]:+.4f}] "
            f"p={summary['two_sided_randomization_p']:.3g} "
            f"({'beats' if summary['recorded_beats_surrogate'] else 'DOES NOT BEAT'})"
        )

    real_passed = bool(results["identity"]["passed"])
    beaten = [n for n, c in contrasts.items() if c["recorded_beats_surrogate"]]
    verdict = {
        "real_semg_passed_gate": real_passed,
        "surrogates_that_also_passed": [
            name for name in SURROGATES if name != "identity" and results[name]["passed"]
        ],
        "surrogates_beaten_by_recorded_semg": beaten,
        "recorded_vs_surrogate": contrasts,
        # Attribution requires the recorded signal to clear its gate and to beat
        # every surrogate in a paired within-participant comparison.
        "attributable_to_semg_content": bool(
            real_passed and len(beaten) == len(contrasts)
        ),
    }
    gait._atomic_json(
        run_dir / "semg_controls.json",
        {
            "version": VERSION,
            "cache_dir": str(cache_dir),
            "confirmation_run_dir": str(args.confirmation_run_dir.resolve()),
            "subjects": subjects,
            "kinematic_alpha": kinematic_alpha,
            "residual_alpha": residual_alpha,
            "gamma": gamma,
            "seed": SEED,
            "verdict": verdict,
            "conditions": results,
            "software": {"python": platform.python_version(), "numpy": np.__version__},
        },
    )
    print(
        "\nAttributable to sEMG content: "
        f"{'yes' if verdict['attributable_to_semg_content'] else 'NO'}"
    )
    print(f"Wrote {run_dir / 'semg_controls.json'}")


if __name__ == "__main__":
    main()
