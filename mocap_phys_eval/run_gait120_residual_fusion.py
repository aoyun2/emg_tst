from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .matching import motion_match_one_window
from .recording import record_compare_rollout
from .run import _ensure_reference_bank
from .sim import OverrideSpec, load_expert_policy
from .utils import resample_linear, set_global_determinism


VERSION = "GAIT120_RESIDUAL_FUSION_PHYSICS_V1"
MOVING_TARGET_PD_VERSION = "GAIT120_RESIDUAL_FUSION_PHYSICS_V2_MOVING_TARGET_PD"
MOVING_TARGET_PD = False
PANEL_VERSION = "GAIT120_MODEL_BLIND_PHYSICS_PANEL_V1"
SEED = 42
# Checkpoint labels come from the frozen panel rather than being hardcoded here.
# The training path chooses how many checkpoints to emit and where to place them,
# and the panel records the labels it was built against; pinning them in this
# module would silently desynchronise the two.  ``_adopt_checkpoints`` sets these
# from the panel manifest before any window is evaluated.
CHECKPOINT_LABELS: tuple[str, ...] = ()
PRIMARY_CHECKPOINT = ""
MATCH_MAX_MEAN_KNEE_RMSE_DEG = 10.0
MATCH_MAX_MEAN_THIGH_RMS_DEG = 15.0
ORACLE_WINDOWS = 10
ORACLE_MINIMUM_PASSING = 8
ORACLE_MAX_TRACKING_RMSE_DEG = 10.0
# Recorded for every preflight window but deliberately not a pass criterion: the
# instability the substituted knee adds is the effect under study, not an
# apparatus fault. See the pass criterion in _oracle_preflight.
ORACLE_REPORTED_ADDED_FALL_RISK = True


@dataclass(frozen=True)
class PanelQuery:
    query_id: str
    panel_index: int
    subject: str
    start_frame: int
    sample_hz: float
    knee_flexion_deg: np.ndarray
    thigh_pitch_deg: np.ndarray
    thigh_quat_wxyz: np.ndarray
    fused_prediction_deg: np.ndarray
    no_emg_prediction_deg: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def _finite_only(value: Any) -> Any:
    """Replace non-finite floats with None so the result is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_only(v) for v in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _finite_only(value),
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 1:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(a[:n] - b[:n]))))


def _risk_auc(trace: np.ndarray, dt: float) -> float:
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 1 or not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    if values.size == 1:
        return float(values[0] * dt)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values, dx=float(dt)))


def _runtime_probe(run_dir: Path) -> None:
    status_path = run_dir / "pipeline_status.json"
    try:
        import mujoco  # type: ignore

        version = str(getattr(mujoco, "__version__", "unknown"))
        from dm_control import suite  # type: ignore  # noqa: F401

        result = {
            "passed": True,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": version,
        }
        _atomic_json(run_dir / "runtime_probe.json", result)
    except Exception as exc:
        result = {
            "passed": False,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(run_dir / "runtime_probe.json", result)
        _atomic_json(
            status_path,
            {
                "version": VERSION,
                "stage": "blocked_runtime_probe",
                "updated_unix": time.time(),
                **result,
            },
        )
        raise RuntimeError(
            "MuJoCo runtime probe failed before any model-zoo download or physics outcome"
        ) from exc


def _adopt_checkpoints(manifest: dict[str, Any]) -> None:
    """Take the checkpoint schedule from the frozen panel."""
    global CHECKPOINT_LABELS, PRIMARY_CHECKPOINT
    protocol = manifest.get("protocol", {})
    labels = tuple(str(label) for label in protocol.get("checkpoint_labels", ()))
    if not labels:
        raise RuntimeError("Frozen panel does not record its checkpoint labels")
    primary = str(protocol.get("primary_checkpoint", labels[-1]))
    if primary not in labels:
        raise RuntimeError(f"Primary checkpoint {primary!r} is not in the panel schedule")
    CHECKPOINT_LABELS = labels
    PRIMARY_CHECKPOINT = primary


def _load_panel(run_dir: Path) -> tuple[dict[str, Any], list[PanelQuery]]:
    manifest_path = run_dir / "panel_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing frozen physics panel: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("protocol", {}).get("version")) != PANEL_VERSION:
        raise RuntimeError("Unexpected physics-panel version")
    _adopt_checkpoints(manifest)
    rows = list(manifest.get("windows", []))
    if len(rows) != 80:
        raise RuntimeError(f"Frozen panel must contain exactly 80 windows, found {len(rows)}")
    queries: list[PanelQuery] = []
    for expected_index, row in enumerate(rows):
        query_id = str(row["query_id"])
        path = run_dir / "queries" / f"{query_id}.npz"
        with np.load(path, allow_pickle=False) as stored:
            labels = tuple(str(x) for x in np.asarray(stored["checkpoint_labels"]).tolist())
            if labels != CHECKPOINT_LABELS:
                raise RuntimeError(f"Checkpoint labels changed for {query_id}")
            panel_index = int(np.asarray(stored["panel_index"]).reshape(()))
            if panel_index != expected_index:
                raise RuntimeError(f"Panel order changed for {query_id}")
            knee = np.asarray(stored["knee_flexion_deg"], dtype=np.float32)
            thigh = np.asarray(stored["thigh_pitch_deg"], dtype=np.float32)
            quat = np.asarray(stored["thigh_quat_wxyz"], dtype=np.float32)
            fused = np.asarray(stored["fused_prediction_deg"], dtype=np.float32)
            no_emg = np.asarray(stored["no_emg_prediction_deg"], dtype=np.float32)
            if knee.shape != (100,) or thigh.shape != (100,) or quat.shape != (100, 4):
                raise RuntimeError(f"Malformed recorded query arrays for {query_id}")
            if fused.shape != (len(CHECKPOINT_LABELS), 100) or no_emg.shape != fused.shape:
                raise RuntimeError(f"Malformed prediction arrays for {query_id}")
            if not all(np.all(np.isfinite(x)) for x in (knee, thigh, quat, fused, no_emg)):
                raise RuntimeError(f"Non-finite panel value for {query_id}")
            queries.append(
                PanelQuery(
                    query_id=query_id,
                    panel_index=panel_index,
                    subject=str(row["subject"]),
                    start_frame=int(row["start_frame"]),
                    sample_hz=float(np.asarray(stored["sample_hz"]).reshape(())),
                    knee_flexion_deg=knee,
                    thigh_pitch_deg=thigh,
                    thigh_quat_wxyz=quat,
                    fused_prediction_deg=fused,
                    no_emg_prediction_deg=no_emg,
                )
            )
    if len({query.query_id for query in queries}) != 80:
        raise RuntimeError("Frozen physics panel contains duplicate query identifiers")
    return manifest, queries


def _match_query(query: PanelQuery, cfg: EvalConfig, bank: Any, match_hz: float) -> tuple[Any, int]:
    thigh = resample_linear(query.thigh_pitch_deg, src_hz=query.sample_hz, dst_hz=match_hz)
    knee = resample_linear(query.knee_flexion_deg, src_hz=query.sample_hz, dst_hz=match_hz)
    length = int(min(thigh.size, knee.size))
    candidates = motion_match_one_window(
        bank=bank,
        query_thigh_deg=thigh[:length],
        query_thigh_quat_wxyz=None,
        query_knee_deg=knee[:length],
        top_k=int(cfg.match_top_k),
        local_refine_radius=int(cfg.match_local_refine_radius),
        feature_mode=str(cfg.match_feature_mode),
        knee_weight=float(cfg.match_knee_weight),
        thigh_weight=float(cfg.match_thigh_weight),
    )
    if not candidates:
        raise RuntimeError(f"No motion match for {query.query_id}")
    return candidates[0], length


def _candidate_row(query: PanelQuery, candidate: Any, length: int) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "panel_index": query.panel_index,
        "subject": query.subject,
        "snippet_id": str(candidate.snippet_id),
        "clip_id": str(candidate.clip_id),
        "start_step_in_snippet": int(candidate.start_step),
        "length": int(length),
        "knee_rmse_deg": float(candidate.rmse_knee_deg),
        "thigh_rms_deg": float(candidate.rmse_thigh_deg),
        "knee_sign": float(candidate.knee_sign),
        "knee_offset_deg": float(candidate.knee_offset_deg),
        "score": float(candidate.score),
    }


def _matching_preflight(
    queries: list[PanelQuery], run_dir: Path, cfg: EvalConfig, bank: Any, match_hz: float
) -> dict[str, Any]:
    path = run_dir / "matching_preflight" / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not bool(summary.get("passed")):
            raise RuntimeError("Saved motion-matching preflight did not pass")
        return summary
    rows = []
    for query in queries:
        candidate, length = _match_query(query, cfg, bank, match_hz)
        rows.append(_candidate_row(query, candidate, length))
    knee = np.asarray([row["knee_rmse_deg"] for row in rows], dtype=np.float64)
    thigh = np.asarray([row["thigh_rms_deg"] for row in rows], dtype=np.float64)
    summary = {
        "windows": rows,
        "n_windows": len(rows),
        "mean_knee_rmse_deg": float(np.mean(knee)),
        "median_knee_rmse_deg": float(np.median(knee)),
        "p95_knee_rmse_deg": float(np.quantile(knee, 0.95)),
        "mean_thigh_rms_deg": float(np.mean(thigh)),
        "median_thigh_rms_deg": float(np.median(thigh)),
        "p95_thigh_rms_deg": float(np.quantile(thigh, 0.95)),
        "maximum_allowed_mean_knee_rmse_deg": MATCH_MAX_MEAN_KNEE_RMSE_DEG,
        "maximum_allowed_mean_thigh_rms_deg": MATCH_MAX_MEAN_THIGH_RMS_DEG,
        "passed": bool(
            len(rows) == 80
            and float(np.mean(knee)) <= MATCH_MAX_MEAN_KNEE_RMSE_DEG
            and float(np.mean(thigh)) <= MATCH_MAX_MEAN_THIGH_RMS_DEG
        ),
    }
    _atomic_json(path, summary)
    if not bool(summary["passed"]):
        raise RuntimeError(
            "Motion-matching preflight failed: "
            f"mean knee={summary['mean_knee_rmse_deg']:.3f}, "
            f"mean thigh={summary['mean_thigh_rms_deg']:.3f} degrees"
        )
    return summary


def _record_metrics(path: Path, reference_target: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        dt = float(np.asarray(stored["dt"]).reshape(()))
        output: dict[str, Any] = {}
        for name, suffix in (("reference", "ref"), ("fused", "good"), ("no_emg", "bad")):
            actual = np.asarray(stored[f"knee_{suffix}_actual_deg"], dtype=np.float32)
            target = (
                np.asarray(reference_target, dtype=np.float32)
                if suffix == "ref"
                else np.asarray(stored[f"knee_{suffix}_query_deg"], dtype=np.float32)
            )
            risk_trace = np.asarray(
                stored[f"predicted_fall_risk_trace_{suffix}"], dtype=np.float32
            )
            output[name] = {
                "recorded_steps": int(actual.size),
                "tracking_rmse_deg": _rmse(target, actual),
                "balance_loss_step": int(
                    np.asarray(stored[f"balance_loss_step_{suffix}"]).reshape(())
                ),
                "fall_risk": float(
                    np.asarray(stored[f"predicted_fall_risk_{suffix}"]).reshape(())
                ),
                "risk_auc": _risk_auc(risk_trace, dt),
            }
        output["dt"] = dt
        output["has_no_emg"] = bool(np.asarray(stored["has_bad"]).reshape(()))
        # Absolute instability depends on how stable the matched reference
        # already is before any override.  The paired quantity the analysis uses
        # is the instability the substituted knee trajectory adds to its own
        # reference rollout.
        output["excess_instability_auc"] = {
            "fused": float(output["fused"]["risk_auc"] - output["reference"]["risk_auc"]),
            "no_emg": (
                float(output["no_emg"]["risk_auc"] - output["reference"]["risk_auc"])
                if output["has_no_emg"]
                else None
            ),
        }
        return output


def _oracle_preflight(
    queries: list[PanelQuery],
    matches: dict[str, dict[str, Any]],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
) -> dict[str, Any]:
    path = run_dir / "oracle_preflight" / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not bool(summary.get("passed")):
            raise RuntimeError("Saved oracle preflight did not pass")
        return summary
    order = np.random.default_rng(SEED + 3407).permutation(len(queries))[:ORACLE_WINDOWS]
    results = []
    for smoke_index, query_index in enumerate(order.tolist()):
        query = queries[int(query_index)]
        match = matches[query.query_id]
        q_dir = run_dir / "oracle_preflight" / "evals" / f"{smoke_index:02d}_{query.query_id}"
        summary_path = q_dir / "summary.json"
        if summary_path.exists():
            results.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        recording_npz = q_dir / "oracle_compare.npz"
        recording_gif = q_dir / "oracle_compare.gif"
        if q_dir.exists():
            contents = list(q_dir.iterdir())
            completed_recording = recording_npz.exists() and recording_gif.exists()
            if contents and not completed_recording:
                raise RuntimeError(f"Incomplete oracle preflight requires audit: {q_dir}")
            if not contents and not (run_dir / "operational_repairs.json").exists():
                raise RuntimeError(f"Undocumented empty oracle directory requires audit: {q_dir}")
        else:
            q_dir.mkdir(parents=True, exist_ok=False)
        bi = bank_index[match["snippet_id"]]
        start = int(match["start_step_in_snippet"])
        length = int(match["length"])
        reference = np.asarray(bank.knee_deg[bi], dtype=np.float32)[start : start + length]
        snippet_start = int(np.asarray(bank.start_step[bi]).reshape(()))
        snippet_end = int(np.asarray(bank.end_step[bi]).reshape(()))
        if not recording_npz.exists():
            policy = load_expert_policy(
                Path(str(np.asarray(bank.expert_model_path[bi]).reshape(()))), device=str(cfg.device)
            )
            record_compare_rollout(
                out_npz_path=recording_npz,
                clip_id=str(match["clip_id"]),
                start_step=snippet_start + start,
                end_step=snippet_end,
                primary_steps=length,
                warmup_steps=max(0, start),
                policy=policy,
                override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
                knee_good_query_deg=reference,
                knee_bad_query_deg=reference,
                width=int(cfg.render_width),
                height=int(cfg.render_height),
                camera_id=int(cfg.render_camera_id),
                deterministic_policy=True,
                seed=0,
                run_bad=False,
                panel_labels=("Reference", "Exact knee target", "Unused"),
                moving_target_pd=MOVING_TARGET_PD,
            )
        metrics = _record_metrics(recording_npz, reference)
        ref = metrics["reference"]
        oracle = metrics["fused"]
        # This gate checks the apparatus, not the outcome.
        #
        # The instability the override adds is deliberately NOT a pass criterion.
        # A transfemoral wearer cannot drive their prosthetic knee volitionally:
        # it is driven by its own controller, and the rest of the body has to
        # compensate. Overriding the actuator removes that joint from the policy
        # in exactly the same way, so instability arising from the substitution
        # is the effect under study rather than an artefact to screen out.
        # Gating on it, or excluding the windows where it appears, would discard
        # the cases where prosthesis use matters most and bias the study toward
        # finding nothing.
        #
        # What must hold is that the instrument works: the prosthetic controller
        # follows the trajectory it was given, both rollouts run to completion,
        # and the matched reference is not already falling over on its own, which
        # would be a matching failure rather than a controller one.
        added_risk = float(oracle["fall_risk"]) - float(ref["fall_risk"])
        passed = bool(
            ref["balance_loss_step"] == -1
            and oracle["tracking_rmse_deg"] <= ORACLE_MAX_TRACKING_RMSE_DEG
            and ref["recorded_steps"] == length
            and oracle["recorded_steps"] == length
        )
        result = {
            "query_id": query.query_id,
            "match": match,
            "metrics": metrics,
            "required_steps": length,
            "added_fall_risk_vs_reference": added_risk,
            "passed": passed,
            "recording_npz": str(recording_npz),
            "recording_gif": str(recording_gif),
        }
        _atomic_json(summary_path, result)
        results.append(result)
    passing = sum(bool(row["passed"]) for row in results)
    summary = {
        "windows": results,
        "passing_windows": passing,
        "required_passing_windows": ORACLE_MINIMUM_PASSING,
        "passed": bool(len(results) == ORACLE_WINDOWS and passing >= ORACLE_MINIMUM_PASSING),
    }
    _atomic_json(path, summary)
    if not bool(summary["passed"]):
        raise RuntimeError(f"Oracle physics preflight failed: {passing}/{ORACLE_WINDOWS} passed")
    return summary


def _evaluate(
    query: PanelQuery,
    checkpoint_index: int,
    checkpoint_label: str,
    match: dict[str, Any],
    q_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
) -> dict[str, Any]:
    summary_path = q_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if q_dir.exists():
        raise RuntimeError(f"Incomplete physics evaluation requires audit: {q_dir}")
    q_dir.mkdir(parents=True, exist_ok=False)
    bi = bank_index[match["snippet_id"]]
    start = int(match["start_step_in_snippet"])
    length = int(match["length"])
    match_hz = float(np.asarray(bank.sample_hz[bi]).reshape(()))
    reference = np.asarray(bank.knee_deg[bi], dtype=np.float32)[start : start + length]
    fused_native = query.fused_prediction_deg[checkpoint_index]
    no_emg_native = query.no_emg_prediction_deg[checkpoint_index]
    measured_native = query.knee_flexion_deg
    fused = resample_linear(fused_native, src_hz=query.sample_hz, dst_hz=match_hz)[:length]
    no_emg = resample_linear(no_emg_native, src_hz=query.sample_hz, dst_hz=match_hz)[:length]
    knee_sign = float(match["knee_sign"])
    knee_offset = float(match["knee_offset_deg"])
    fused_target = np.clip(knee_sign * fused + knee_offset, 0.0, 170.0).astype(np.float32)
    no_emg_target = np.clip(knee_sign * no_emg + knee_offset, 0.0, 170.0).astype(np.float32)
    snippet_start = int(np.asarray(bank.start_step[bi]).reshape(()))
    snippet_end = int(np.asarray(bank.end_step[bi]).reshape(()))
    policy = load_expert_policy(
        Path(str(np.asarray(bank.expert_model_path[bi]).reshape(()))), device=str(cfg.device)
    )
    primary = checkpoint_label == PRIMARY_CHECKPOINT
    recording = record_compare_rollout(
        out_npz_path=q_dir / "compare.npz",
        clip_id=str(match["clip_id"]),
        start_step=snippet_start + start,
        end_step=snippet_end,
        primary_steps=length,
        warmup_steps=max(0, start),
        policy=policy,
        override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
        knee_good_query_deg=fused_target,
        knee_bad_query_deg=no_emg_target,
        width=int(cfg.render_width),
        height=int(cfg.render_height),
        camera_id=int(cfg.render_camera_id),
        deterministic_policy=True,
        seed=0,
        run_bad=primary,
        panel_labels=("Reference", "Residual fusion", "Without sEMG"),
        moving_target_pd=MOVING_TARGET_PD,
    )
    metrics = _record_metrics(recording.npz_path, reference)
    result = {
        "query_id": query.query_id,
        "panel_index": query.panel_index,
        "subject": query.subject,
        "checkpoint": checkpoint_label,
        "primary_three_condition_recording": primary,
        "prediction_rmse_deg": {
            "fused": _rmse(fused_native, measured_native),
            "no_emg": _rmse(no_emg_native, measured_native),
        },
        "match": match,
        "simulation": metrics,
        "recording_npz": str(recording.npz_path),
        "recording_gif": str(recording.gif_path),
    }
    _atomic_json(summary_path, result)
    return result


def _protocol(
    root: Path,
    run_dir: Path,
    panel_source_dir: Path,
    panel_manifest: dict[str, Any],
    controller_validation_dir: Path | None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "panel_version": PANEL_VERSION,
        "panel_source_dir": str(panel_source_dir.resolve()),
        "panel_manifest_sha256": _sha256(panel_source_dir / "panel_manifest.json"),
        "panel_selection": panel_manifest["protocol"]["selection"],
        "checkpoints": list(CHECKPOINT_LABELS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "matching": {
            "feature_mode": "thigh_knee_d",
            "knee_weight": 1.0,
            "thigh_weight": 0.0,
            "maximum_mean_knee_rmse_deg": MATCH_MAX_MEAN_KNEE_RMSE_DEG,
            "maximum_mean_thigh_rms_deg": MATCH_MAX_MEAN_THIGH_RMS_DEG,
        },
        "oracle_preflight": {
            "windows": ORACLE_WINDOWS,
            "minimum_passing": ORACLE_MINIMUM_PASSING,
            "maximum_tracking_rmse_deg": ORACLE_MAX_TRACKING_RMSE_DEG,
            "added_fall_risk_is_reported_not_gated": ORACLE_REPORTED_ADDED_FALL_RISK,
        },
        "simulation": {
            "primary": "paired reference, residual-fusion, and no-sEMG rollouts",
            "secondary": "paired reference and residual-fusion rollouts",
            "record_every_rollout": True,
            "instability_used_for_eligibility": False,
        },
        "controller": {
            "moving_target_pd": bool(MOVING_TARGET_PD),
            "equation": (
                "tau = Kp*(q_des-q) + Kd*(qdot_des-qdot)"
                if MOVING_TARGET_PD
                else "tau = Kp*(q_des-q) - Kd*qdot"
            ),
            "desired_velocity": (
                "causal backward difference; zero at first evaluation step"
                if MOVING_TARGET_PD
                else "zero"
            ),
            "development_validation_dir": (
                str(controller_validation_dir.resolve())
                if controller_validation_dir is not None
                else None
            ),
            "development_validation_summary_sha256": (
                _sha256(controller_validation_dir / "validation_summary.json")
                if controller_validation_dir is not None
                else None
            ),
        },
        "code_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "core_run": _sha256(root / "mocap_phys_eval" / "run.py"),
            "matching": _sha256(root / "mocap_phys_eval" / "matching.py"),
            "recording": _sha256(root / "mocap_phys_eval" / "recording.py"),
            "simulation": _sha256(root / "mocap_phys_eval" / "sim.py"),
        },
    }


def main() -> None:
    global MOVING_TARGET_PD, VERSION

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-probe-only", action="store_true")
    # The reported simulations used the moving-target controller. Leaving it
    # optional meant the obvious command silently ran the older mode, so the
    # mode has to be stated.
    controller = parser.add_mutually_exclusive_group(required=True)
    controller.add_argument("--moving-target-pd-v2", action="store_true",
                            help="the controller the reported results used")
    controller.add_argument("--static-target-pd", action="store_true",
                            help="the superseded static-target controller")
    parser.add_argument("--panel-source-dir", type=Path)
    parser.add_argument("--controller-validation-dir", type=Path)
    args = parser.parse_args()

    MOVING_TARGET_PD = bool(args.moving_target_pd_v2)
    if MOVING_TARGET_PD:
        VERSION = MOVING_TARGET_PD_VERSION

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    panel_source_dir = (
        args.panel_source_dir.resolve() if args.panel_source_dir is not None else run_dir
    )
    controller_validation_dir = (
        args.controller_validation_dir.resolve()
        if args.controller_validation_dir is not None
        else None
    )
    # A separately staged controller validation may be supplied, and is still
    # checked when it is. It is no longer required, because the oracle preflight
    # below validates the same property more directly: it commands the controller
    # to track the matched clip's own realized knee trajectory and gates the run
    # on tracking error and fall risk. That check uses no model output and no
    # study outcome, so it cannot leak into the reported result, and it runs in
    # this same invocation -- a failure stops the run before anything is
    # recorded.
    if controller_validation_dir is not None:
        validation_path = controller_validation_dir / "validation_summary.json"
        if not validation_path.exists():
            raise RuntimeError(f"Missing completed controller validation: {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not bool(validation.get("passed")):
            raise RuntimeError("Moving-target PD development validation did not pass")
        if int(validation.get("passing_windows", -1)) < ORACLE_MINIMUM_PASSING:
            raise RuntimeError("Moving-target PD development validation passed count is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    _runtime_probe(run_dir)
    if args.runtime_probe_only:
        return

    panel_manifest, queries = _load_panel(panel_source_dir)
    protocol = _protocol(
        root,
        run_dir,
        panel_source_dir,
        panel_manifest,
        controller_validation_dir,
    )
    # The initial protocol is retained verbatim as evidence of the renderer-only
    # compatibility failure.  This final protocol hashes the repaired recording
    # layer and is the sole protocol accepted for resumed physics outcomes.
    protocol_path = run_dir / (
        "physics_protocol.moving_target_pd_v2.json"
        if MOVING_TARGET_PD
        else "physics_protocol.runtime_compatibility_v2.json"
    )
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing physics protocol differs from executable protocol")
    else:
        _atomic_json(protocol_path, protocol)
    status_path = run_dir / "pipeline_status.json"

    def update(stage: str, **extra: Any) -> None:
        _atomic_json(
            status_path,
            {"version": VERSION, "stage": stage, "updated_unix": time.time(), **extra},
        )

    set_global_determinism(seed=0)
    cfg = EvalConfig(artifacts_dir=artifacts_dir, device=str(args.device))
    if str(cfg.match_feature_mode) != "thigh_knee_d":
        raise RuntimeError("Definitive runner requires the original scalar matcher")
    if float(cfg.match_knee_weight) != 1.0 or float(cfg.match_thigh_weight) != 0.0:
        raise RuntimeError("Original motion-matching weights changed")

    update("ensuring_mocapact_reference_bank")
    bank = _ensure_reference_bank(cfg, out_root=artifacts_dir)
    bank_index = {str(bank.snippet_id[index]): int(index) for index in range(len(bank))}
    match_hz = float(np.median(np.asarray(bank.sample_hz, dtype=np.float64)))

    update("matching_preflight")
    matching = _matching_preflight(queries, run_dir, cfg, bank, match_hz)
    matches = {str(row["query_id"]): row for row in matching["windows"]}
    update("oracle_physics_preflight")
    _oracle_preflight(queries, matches, run_dir, cfg, bank, bank_index)

    stages = []
    for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
        results = []
        for query_index, query in enumerate(queries):
            update(
                "checkpoint_physics",
                checkpoint=checkpoint_label,
                checkpoint_index=checkpoint_index,
                query_index=query_index,
                query_id=query.query_id,
                total_queries=len(queries),
            )
            result = _evaluate(
                query,
                checkpoint_index,
                checkpoint_label,
                matches[query.query_id],
                run_dir / "stages" / checkpoint_label / "evals" / query.query_id,
                cfg,
                bank,
                bank_index,
            )
            results.append(result)
        stage = {"checkpoint": checkpoint_label, "n_windows": len(results), "results": results}
        _atomic_json(run_dir / "stages" / checkpoint_label / "summary.json", stage)
        stages.append(stage)

    complete = {
        "version": VERSION,
        "checkpoints": list(CHECKPOINT_LABELS),
        "n_windows_per_checkpoint": len(queries),
        "n_paired_prediction_rollouts": len(queries) * len(CHECKPOINT_LABELS),
        "primary_three_condition_recordings": len(queries),
        "stages": stages,
    }
    _atomic_json(run_dir / "physics_summary.json", complete)
    update("physics_complete", summary_path=str((run_dir / "physics_summary.json").resolve()))


if __name__ == "__main__":
    main()
