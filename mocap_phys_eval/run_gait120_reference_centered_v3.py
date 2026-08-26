from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .recording import record_compare_rollout
from .reference_centered import (
    capture_unmodified_reference,
    inject_prediction_error,
    load_reference_baseline,
)
from .run import _ensure_reference_bank
from .run_gait120_residual_fusion import (
    CHECKPOINT_LABELS,
    MATCH_MAX_MEAN_KNEE_RMSE_DEG,
    MATCH_MAX_MEAN_THIGH_RMS_DEG,
    ORACLE_MAX_FALL_RISK,
    ORACLE_MAX_TRACKING_RMSE_DEG,
    ORACLE_MINIMUM_PASSING,
    ORACLE_WINDOWS,
    PANEL_VERSION,
    PRIMARY_CHECKPOINT,
    SEED,
    PanelQuery,
    _atomic_json,
    _load_panel,
    _record_metrics,
    _rmse,
    _sha256,
)
from .sim import OverrideSpec, load_expert_policy, make_tracking_env
from .utils import set_global_determinism


BASE_VERSION = "GAIT120_RESIDUAL_FUSION_PHYSICS_V3_REFERENCE_CENTERED"
SPARSE_MEDIA_VERSION = "GAIT120_RESIDUAL_FUSION_PHYSICS_V3_1_SPARSE_MEDIA"
GROUPED_VERSION = "GAIT120_RESIDUAL_FUSION_PHYSICS_V3_2_GROUPED_SNIPPETS"
VERSION = BASE_VERSION
DEVELOPMENT_VERSION = "GAIT120_REFERENCE_CENTERED_DEVELOPMENT_VALIDATION_V1"
REPRESENTATIVE_PANEL_INDICES = (0, 11, 22, 33, 44, 55, 66, 77)


def _update(path: Path, stage: str, **extra: Any) -> None:
    _atomic_json(
        path,
        {"version": VERSION, "stage": stage, "updated_unix": time.time(), **extra},
    )


def _runtime_probe(run_dir: Path) -> None:
    try:
        import mujoco  # type: ignore
        from dm_control import suite  # type: ignore  # noqa: F401

        result = {
            "passed": True,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": str(getattr(mujoco, "__version__", "unknown")),
        }
    except Exception as exc:
        result = {
            "passed": False,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(run_dir / "runtime_probe.json", result)
        _update(run_dir / "pipeline_status.json", "blocked_runtime_probe", **result)
        raise RuntimeError("V3 MuJoCo runtime probe failed") from exc
    _atomic_json(run_dir / "runtime_probe.json", result)


def _require_development_validation(path: Path) -> dict[str, Any]:
    summary_path = path / "validation_summary.json"
    protocol_path = path / "protocol.json"
    if not summary_path.exists() or not protocol_path.exists():
        raise RuntimeError("V3 requires the completed development validation evidence")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("version")) != DEVELOPMENT_VERSION:
        raise RuntimeError("Unexpected V3 development-validation version")
    if not bool(summary.get("passed")) or int(summary.get("passing_windows", -1)) != 10:
        raise RuntimeError("V3 development validation did not pass all frozen windows")
    return summary


def _load_frozen_matches(
    panel_source_dir: Path,
    queries: list[PanelQuery],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    path = panel_source_dir / "matching_preflight" / "summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing frozen V1 motion matches: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = list(summary.get("windows", []))
    if not bool(summary.get("passed")) or len(rows) != len(queries) or len(rows) != 80:
        raise RuntimeError("Frozen V1 motion-matching summary is malformed or failed")
    if float(summary.get("mean_knee_rmse_deg", np.inf)) > MATCH_MAX_MEAN_KNEE_RMSE_DEG:
        raise RuntimeError("Frozen V1 mean knee matching error no longer passes")
    if float(summary.get("mean_thigh_rms_deg", np.inf)) > MATCH_MAX_MEAN_THIGH_RMS_DEG:
        raise RuntimeError("Frozen V1 mean thigh matching error no longer passes")
    query_ids = [query.query_id for query in queries]
    row_ids = [str(row.get("query_id")) for row in rows]
    if row_ids != query_ids or len(set(row_ids)) != len(row_ids):
        raise RuntimeError("Frozen V1 motion-match order differs from the frozen panel")
    return summary, {str(row["query_id"]): row for row in rows}, path


def _load_policy(match: dict[str, Any], cfg: EvalConfig, bank: Any, bank_index: dict[str, int]) -> Any:
    bank_row = bank_index[str(match["snippet_id"])]
    model_path = Path(str(np.asarray(bank.expert_model_path[bank_row]).reshape(())))
    return load_expert_policy(model_path, device=str(cfg.device))


def _make_prebuilt_envs(
    policy: Any,
    match: dict[str, Any],
    bank: Any,
    bank_index: dict[str, int],
    *,
    include_bad: bool = True,
) -> tuple[Any, Any, Any | None]:
    bank_row = bank_index[str(match["snippet_id"])]
    start = int(match["start_step_in_snippet"])
    length = int(match["length"])
    snippet_start = int(np.asarray(bank.start_step[bank_row]).reshape(()))
    snippet_end = int(np.asarray(bank.end_step[bank_row]).reshape(()))
    ref_steps_array = np.asarray(getattr(policy, "ref_steps", (0,)), dtype=np.int64).reshape(-1)
    ref_steps = tuple(int(value) for value in ref_steps_array.tolist()) or (0,)
    max_ref_step = int(np.max(ref_steps_array)) if ref_steps_array.size else 0
    evaluation_start = snippet_start + start
    minimum_end = evaluation_start + max(0, length - 1) + max_ref_step
    use_end = max(snippet_end, minimum_end)
    environments = tuple(
        make_tracking_env(
            clip_id=str(match["clip_id"]),
            start_step=snippet_start,
            end_step=use_end,
            ref_steps=ref_steps,
            seed=0,
        )
        for _ in range(3 if include_bad else 2)
    )
    if include_bad:
        return environments  # type: ignore[return-value]
    return environments[0], environments[1], None


def _reference_baseline(
    query: PanelQuery,
    match: dict[str, Any],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
    *,
    policy: Any | None,
    prebuilt_envs: tuple[Any, Any, Any | None] | None = None,
    render_media: bool = True,
) -> tuple[np.ndarray, Any | None]:
    q_dir = run_dir / "reference_baselines" / query.query_id
    summary_path = q_dir / "summary.json"
    npz_path = q_dir / "reference_capture.npz"
    gif_path = q_dir / "reference_capture.gif"
    bank_row = bank_index[str(match["snippet_id"])]
    length = int(match["length"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("query_id")) != query.query_id or int(summary.get("steps", -1)) != length:
            raise RuntimeError(f"Saved reference summary changed for {query.query_id}")
        return load_reference_baseline(npz_path, required_steps=length), policy
    if q_dir.exists():
        contents = list(q_dir.iterdir())
        completed = npz_path.exists() and (gif_path.exists() if render_media else True)
        if contents and not completed:
            raise RuntimeError(f"Incomplete reference capture requires audit: {q_dir}")
    else:
        q_dir.mkdir(parents=True, exist_ok=False)
    start = int(match["start_step_in_snippet"])
    nominal = np.asarray(bank.knee_deg[bank_row], dtype=np.float32)[start : start + length]
    snippet_start = int(np.asarray(bank.start_step[bank_row]).reshape(()))
    snippet_end = int(np.asarray(bank.end_step[bank_row]).reshape(()))
    if not npz_path.exists():
        if policy is None:
            policy = _load_policy(match, cfg, bank, bank_index)
        capture_unmodified_reference(
            out_npz_path=npz_path,
            clip_id=str(match["clip_id"]),
            start_step=snippet_start + start,
            end_step=snippet_end,
            primary_steps=length,
            warmup_steps=max(0, start),
            policy=policy,
            nominal_reference_deg=nominal,
            override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
            width=int(cfg.render_width),
            height=int(cfg.render_height),
            camera_id=int(cfg.render_camera_id),
            render_media=render_media,
            prebuilt_envs=prebuilt_envs,
        )
    baseline = load_reference_baseline(npz_path, required_steps=length)
    metrics = _record_metrics(npz_path, baseline)
    if metrics["reference"]["recorded_steps"] != length:
        raise RuntimeError(f"Reference capture truncated for {query.query_id}")
    summary = {
        "query_id": query.query_id,
        "steps": length,
        "match": match,
        "metrics": metrics,
        "recording_npz": str(npz_path.resolve()),
        "recording_gif": str(gif_path.resolve()) if gif_path.exists() else None,
        "media_rendered": bool(gif_path.exists()),
        "deterministic_replay_identical": True,
    }
    _atomic_json(summary_path, summary)
    return baseline, policy


def _oracle_preflight(
    queries: list[PanelQuery],
    matches: dict[str, dict[str, Any]],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
    status_path: Path,
) -> dict[str, Any]:
    aggregate_path = run_dir / "oracle_preflight" / "summary.json"
    if aggregate_path.exists():
        summary = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not bool(summary.get("passed")):
            raise RuntimeError("Saved V3 oracle preflight did not pass")
        return summary
    order = np.random.default_rng(SEED + 3407).permutation(len(queries))[:ORACLE_WINDOWS]
    results: list[dict[str, Any]] = []
    for smoke_index, query_index in enumerate(order.tolist()):
        query = queries[int(query_index)]
        match = matches[query.query_id]
        _update(
            status_path,
            "reference_centered_oracle_preflight",
            query_index=smoke_index,
            total_queries=ORACLE_WINDOWS,
            query_id=query.query_id,
        )
        q_dir = run_dir / "oracle_preflight" / "evals" / f"{smoke_index:02d}_{query.query_id}"
        summary_path = q_dir / "summary.json"
        if summary_path.exists():
            results.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        q_dir.mkdir(parents=True, exist_ok=True)
        recording_npz = q_dir / "oracle_compare.npz"
        recording_gif = q_dir / "oracle_compare.gif"
        if list(q_dir.iterdir()) and not (recording_npz.exists() and recording_gif.exists()):
            raise RuntimeError(f"Incomplete V3 oracle recording requires audit: {q_dir}")
        policy = _load_policy(match, cfg, bank, bank_index)
        baseline, policy = _reference_baseline(
            query, match, run_dir, cfg, bank, bank_index, policy=policy
        )
        bank_row = bank_index[str(match["snippet_id"])]
        start = int(match["start_step_in_snippet"])
        length = int(match["length"])
        snippet_start = int(np.asarray(bank.start_step[bank_row]).reshape(()))
        snippet_end = int(np.asarray(bank.end_step[bank_row]).reshape(()))
        if not recording_npz.exists():
            record_compare_rollout(
                out_npz_path=recording_npz,
                clip_id=str(match["clip_id"]),
                start_step=snippet_start + start,
                end_step=snippet_end,
                primary_steps=length,
                warmup_steps=max(0, start),
                policy=policy,
                override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
                knee_good_query_deg=baseline,
                knee_bad_query_deg=baseline,
                width=int(cfg.render_width),
                height=int(cfg.render_height),
                camera_id=int(cfg.render_camera_id),
                deterministic_policy=True,
                seed=0,
                run_bad=False,
                panel_labels=("Reference", "Zero-error injection", "Unused"),
                moving_target_pd=True,
            )
        metrics = _record_metrics(recording_npz, baseline)
        reference = metrics["reference"]
        oracle = metrics["fused"]
        passed = bool(
            reference["balance_loss_step"] == -1
            and oracle["balance_loss_step"] == -1
            and reference["fall_risk"] < ORACLE_MAX_FALL_RISK
            and oracle["fall_risk"] < ORACLE_MAX_FALL_RISK
            and oracle["tracking_rmse_deg"] <= ORACLE_MAX_TRACKING_RMSE_DEG
            and reference["recorded_steps"] == length
            and oracle["recorded_steps"] == length
        )
        result = {
            "query_id": query.query_id,
            "panel_index": query.panel_index,
            "match": match,
            "metrics": metrics,
            "required_steps": length,
            "passed": passed,
            "recording_npz": str(recording_npz.resolve()),
            "recording_gif": str(recording_gif.resolve()),
        }
        _atomic_json(summary_path, result)
        results.append(result)
    passing = sum(bool(row["passed"]) for row in results)
    summary = {
        "version": VERSION,
        "fixed_order_panel_indices": [int(value) for value in order.tolist()],
        "windows": results,
        "passing_windows": passing,
        "required_passing_windows": ORACLE_MINIMUM_PASSING,
        "passed": bool(len(results) == ORACLE_WINDOWS and passing >= ORACLE_MINIMUM_PASSING),
    }
    _atomic_json(aggregate_path, summary)
    if not bool(summary["passed"]):
        raise RuntimeError(f"V3 oracle physics preflight failed: {passing}/{ORACLE_WINDOWS}")
    return summary


def _evaluate(
    query: PanelQuery,
    checkpoint_index: int,
    checkpoint_label: str,
    match: dict[str, Any],
    baseline: np.ndarray,
    q_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
    policy: Any,
    *,
    render_media: bool = True,
    prebuilt_envs: tuple[Any, Any, Any | None] | None = None,
) -> dict[str, Any]:
    summary_path = q_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    primary = checkpoint_label == PRIMARY_CHECKPOINT
    separate_no_emg = bool(
        primary and prebuilt_envs is not None and prebuilt_envs[2] is None
    )
    recording_npz = q_dir / "compare.npz"
    recording_gif = q_dir / "compare.gif"
    no_emg_npz = q_dir / "no_emg_compare.npz"
    if q_dir.exists():
        contents = list(q_dir.iterdir())
        main_complete = recording_npz.exists() and (
            recording_gif.exists() if render_media else True
        )
        known_partial = bool(separate_no_emg and recording_npz.exists())
        if contents and not main_complete and not known_partial:
            raise RuntimeError(f"Incomplete V3 physics evaluation requires audit: {q_dir}")
    else:
        q_dir.mkdir(parents=True, exist_ok=False)
    bank_row = bank_index[str(match["snippet_id"])]
    start = int(match["start_step_in_snippet"])
    length = int(match["length"])
    match_hz = float(np.asarray(bank.sample_hz[bank_row]).reshape(()))
    fused_native = query.fused_prediction_deg[checkpoint_index]
    no_emg_native = query.no_emg_prediction_deg[checkpoint_index]
    measured_native = query.knee_flexion_deg
    fused_target, fused_error = inject_prediction_error(
        baseline,
        fused_native,
        measured_native,
        knee_sign=float(match["knee_sign"]),
        control_hz=match_hz,
    )
    no_emg_target, no_emg_error = inject_prediction_error(
        baseline,
        no_emg_native,
        measured_native,
        knee_sign=float(match["knee_sign"]),
        control_hz=match_hz,
    )
    if fused_target.size != length or no_emg_target.size != length:
        raise RuntimeError(f"Exact frame selection changed matched length for {query.query_id}")
    snippet_start = int(np.asarray(bank.start_step[bank_row]).reshape(()))
    snippet_end = int(np.asarray(bank.end_step[bank_row]).reshape(()))
    if not recording_npz.exists():
        record_compare_rollout(
            out_npz_path=recording_npz,
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
            run_bad=bool(primary and not separate_no_emg),
            panel_labels=("Reference", "Residual fusion", "Without sEMG"),
            moving_target_pd=True,
            render_media=render_media,
            prebuilt_envs=prebuilt_envs,
        )
    metrics = _record_metrics(recording_npz, baseline)
    if separate_no_emg:
        if not no_emg_npz.exists():
            record_compare_rollout(
                out_npz_path=no_emg_npz,
                clip_id=str(match["clip_id"]),
                start_step=snippet_start + start,
                end_step=snippet_end,
                primary_steps=length,
                warmup_steps=max(0, start),
                policy=policy,
                override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
                knee_good_query_deg=no_emg_target,
                knee_bad_query_deg=no_emg_target,
                width=int(cfg.render_width),
                height=int(cfg.render_height),
                camera_id=int(cfg.render_camera_id),
                deterministic_policy=True,
                seed=0,
                run_bad=False,
                panel_labels=("Reference", "Without sEMG", "Unused"),
                moving_target_pd=True,
                render_media=False,
                prebuilt_envs=prebuilt_envs,
            )
        no_emg_metrics = _record_metrics(no_emg_npz, baseline)
        metrics["no_emg"] = no_emg_metrics["fused"]
        metrics["has_no_emg"] = True
    result = {
        "query_id": query.query_id,
        "panel_index": query.panel_index,
        "subject": query.subject,
        "checkpoint": checkpoint_label,
        "primary_three_condition_recording": bool(primary and not separate_no_emg),
        "primary_no_emg_simulated_separately": separate_no_emg,
        "media_rendered": bool(recording_gif.exists()),
        "prediction_rmse_deg": {
            "source_100hz": {
                "fused": _rmse(fused_native, measured_native),
                "no_emg": _rmse(no_emg_native, measured_native),
            },
            "exact_simulation_frames": {
                "fused": _rmse(fused_error, np.zeros_like(fused_error)),
                "no_emg": _rmse(no_emg_error, np.zeros_like(no_emg_error)),
            },
        },
        "mapping": {
            "equation": "realized_reference + knee_sign*(prediction-measurement)",
            "source_hz": query.sample_hz,
            "control_hz": match_hz,
            "selected_source_frame_indices": list(range(0, 100, 3)),
            "interpolation": False,
            "clipping": False,
        },
        "match": match,
        "simulation": metrics,
        "recording_npz": str(recording_npz.resolve()),
        "no_emg_recording_npz": str(no_emg_npz.resolve()) if no_emg_npz.exists() else None,
        "recording_gif": str(recording_gif.resolve()) if recording_gif.exists() else None,
    }
    _atomic_json(summary_path, result)
    return result


def _protocol(
    root: Path,
    panel_source_dir: Path,
    panel_manifest: dict[str, Any],
    matching_path: Path,
    development_dir: Path,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "panel_version": PANEL_VERSION,
        "panel_source_dir": str(panel_source_dir.resolve()),
        "panel_manifest_sha256": _sha256(panel_source_dir / "panel_manifest.json"),
        "panel_selection": panel_manifest["protocol"]["selection"],
        "frozen_matching_summary": str(matching_path.resolve()),
        "frozen_matching_summary_sha256": _sha256(matching_path),
        "development_validation_dir": str(development_dir.resolve()),
        "development_validation_summary_sha256": _sha256(
            development_dir / "validation_summary.json"
        ),
        "checkpoints": list(CHECKPOINT_LABELS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "mapping": {
            "baseline": "realized knee from an unmodified deterministic expert replay",
            "target": "baseline + knee_sign*(prediction-measurement)",
            "source_to_control": "direct source-frame indices 0,3,6,...",
            "interpolation": False,
            "prediction_or_data_clipping": False,
            "scaling": False,
        },
        "oracle_preflight": {
            "windows": ORACLE_WINDOWS,
            "selection_seed": SEED + 3407,
            "minimum_passing": ORACLE_MINIMUM_PASSING,
            "maximum_tracking_rmse_deg": ORACLE_MAX_TRACKING_RMSE_DEG,
            "maximum_reference_and_oracle_fall_risk": ORACLE_MAX_FALL_RISK,
            "balance_loss_permitted": False,
        },
        "simulation": {
            "controller": "moving-target PD; unchanged Kp=800, Kd=40, force=800",
            "primary": "paired reference, residual-fusion, and no-sEMG rollouts",
            "secondary": "paired reference and residual-fusion rollouts",
            "record_every_rollout": True,
            "falls_retained": True,
            "instability_used_for_eligibility": False,
        },
        "code_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "protocol": _sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "reference_centered": _sha256(root / "mocap_phys_eval" / "reference_centered.py"),
            "recording": _sha256(root / "mocap_phys_eval" / "recording.py"),
            "matching": _sha256(root / "mocap_phys_eval" / "matching.py"),
            "simulation": _sha256(root / "mocap_phys_eval" / "sim.py"),
        },
    }


def main() -> None:
    global VERSION

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--panel-source-dir", type=Path, required=True)
    parser.add_argument("--development-validation-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-probe-only", action="store_true")
    parser.add_argument("--sparse-media-v3-1", action="store_true")
    parser.add_argument("--sparse-media-validation-dir", type=Path)
    parser.add_argument("--grouped-snippet-v3-2", action="store_true")
    args = parser.parse_args()

    grouped_snippets = bool(args.grouped_snippet_v3_2)
    sparse_media = bool(args.sparse_media_v3_1 or grouped_snippets)
    VERSION = (
        GROUPED_VERSION
        if grouped_snippets
        else SPARSE_MEDIA_VERSION
        if sparse_media
        else BASE_VERSION
    )

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    panel_source_dir = args.panel_source_dir.resolve()
    development_dir = args.development_validation_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    sparse_validation_dir = (
        args.sparse_media_validation_dir.resolve()
        if args.sparse_media_validation_dir is not None
        else None
    )
    if run_dir == panel_source_dir or run_dir == development_dir:
        raise RuntimeError("V3 output must not overwrite source evidence")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "pipeline_status.json"
    _runtime_probe(run_dir)
    if args.runtime_probe_only:
        return
    _require_development_validation(development_dir)
    panel_manifest, queries = _load_panel(panel_source_dir)
    _, matches, matching_path = _load_frozen_matches(panel_source_dir, queries)
    protocol = _protocol(
        root, panel_source_dir, panel_manifest, matching_path, development_dir
    )
    if sparse_media:
        if sparse_validation_dir is None:
            raise RuntimeError("Sparse-media resume requires --sparse-media-validation-dir")
        sparse_summary_path = sparse_validation_dir / "validation_summary.json"
        if not sparse_summary_path.exists():
            raise RuntimeError("Sparse-media compatibility summary is missing")
        sparse_summary = json.loads(sparse_summary_path.read_text(encoding="utf-8"))
        if not bool(sparse_summary.get("passed")) or int(
            sparse_summary.get("n_compared_numeric_fields", 0)
        ) < 1:
            raise RuntimeError("Sparse-media compatibility validation did not pass")
        base_protocol_path = run_dir / "physics_protocol.reference_centered_v3.json"
        if not base_protocol_path.exists():
            raise RuntimeError("Sparse-media resume requires the preserved V3 base protocol")
        base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
        if str(base_protocol.get("version")) != BASE_VERSION:
            raise RuntimeError("Sparse-media resume found an unexpected base protocol")
        if base_protocol.get("panel_manifest_sha256") != protocol.get("panel_manifest_sha256"):
            raise RuntimeError("Sparse-media resume panel differs from the base protocol")
        if base_protocol.get("frozen_matching_summary_sha256") != protocol.get(
            "frozen_matching_summary_sha256"
        ):
            raise RuntimeError("Sparse-media resume matches differ from the base protocol")
        protocol["amendment"] = {
            "reason": "user-authorized runtime reduction after the V3 gate passed",
            "base_protocol": str(base_protocol_path.resolve()),
            "base_protocol_sha256": _sha256(base_protocol_path),
            "numerical_conditions_unchanged": True,
            "all_numeric_trajectories_and_states_saved": True,
            "environment_reuse_requires_exact_deterministic_equivalence_validation": True,
            "compatibility_validation_dir": str(sparse_validation_dir.resolve()),
            "compatibility_validation_summary_sha256": _sha256(sparse_summary_path),
            "exactly_compared_numeric_fields": int(
                sparse_summary["n_compared_numeric_fields"]
            ),
            "rendered_primary_panel_indices": list(REPRESENTATIVE_PANEL_INDICES),
            "representative_rule": "fixed evenly spaced panel indices; independent of outcomes",
            "nonrepresentative_pixel_frames": False,
            "nonrepresentative_primary_execution": (
                "fused and no-sEMG are run sequentially from identical deterministic resets "
                "of the same validated environment"
            ),
            "previously completed media_preserved": True,
        }
        if grouped_snippets:
            sparse_protocol_path = run_dir / "physics_protocol.sparse_media_v3_1.json"
            if not sparse_protocol_path.exists():
                raise RuntimeError("Grouped resume requires the preserved V3.1 protocol")
            protocol["amendment"].update(
                {
                    "parent_sparse_protocol": str(sparse_protocol_path.resolve()),
                    "parent_sparse_protocol_sha256": _sha256(sparse_protocol_path),
                    "grouping_key": "frozen MoCapAct snippet_id",
                    "unique_groups": len(
                        {str(row["snippet_id"]) for row in matches.values()}
                    ),
                    "group_order": "descending incomplete group size, then snippet_id",
                    "scientific_order_dependence": False,
                }
            )
            protocol_path = run_dir / "physics_protocol.grouped_snippets_v3_2.json"
        else:
            protocol_path = run_dir / "physics_protocol.sparse_media_v3_1.json"
    else:
        protocol_path = run_dir / "physics_protocol.reference_centered_v3.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing V3 protocol differs from executable protocol")
    else:
        _atomic_json(protocol_path, protocol)

    set_global_determinism(seed=0)
    cfg = EvalConfig(artifacts_dir=artifacts_dir, device=str(args.device))
    if str(cfg.match_feature_mode) != "thigh_knee_d":
        raise RuntimeError("V3 requires the original scalar matcher")
    if float(cfg.match_knee_weight) != 1.0 or float(cfg.match_thigh_weight) != 0.0:
        raise RuntimeError("Original motion-matching weights changed")
    _update(status_path, "ensuring_mocapact_reference_bank")
    bank = _ensure_reference_bank(cfg, out_root=artifacts_dir)
    bank_index = {str(bank.snippet_id[index]): int(index) for index in range(len(bank))}
    missing = sorted(
        {str(match["snippet_id"]) for match in matches.values()} - set(bank_index)
    )
    if missing:
        raise RuntimeError(f"Frozen matched snippets are missing: {missing[:3]}")

    _update(status_path, "reference_centered_oracle_preflight")
    oracle = _oracle_preflight(
        queries, matches, run_dir, cfg, bank, bank_index, status_path
    )
    _update(
        status_path,
        "oracle_preflight_complete",
        passed=True,
        passing_windows=int(oracle["passing_windows"]),
    )

    by_checkpoint: dict[str, list[dict[str, Any]]] = {
        label: [] for label in CHECKPOINT_LABELS
    }
    incomplete_queries: list[PanelQuery] = []
    for query in queries:
        complete_paths = [
            run_dir / "stages" / label / "evals" / query.query_id / "summary.json"
            for label in CHECKPOINT_LABELS
        ]
        if all(path.exists() for path in complete_paths):
            for label, path in zip(CHECKPOINT_LABELS, complete_paths):
                by_checkpoint[label].append(json.loads(path.read_text(encoding="utf-8")))
        else:
            incomplete_queries.append(query)

    if grouped_snippets:
        grouped: dict[str, list[PanelQuery]] = {}
        for query in incomplete_queries:
            snippet_id = str(matches[query.query_id]["snippet_id"])
            grouped.setdefault(snippet_id, []).append(query)
        batches = [
            grouped[key]
            for key in sorted(grouped, key=lambda value: (-len(grouped[value]), value))
        ]
    else:
        batches = [[query] for query in incomplete_queries]

    for batch_index, batch in enumerate(batches):
        first_match = matches[batch[0].query_id]
        snippet_id = str(first_match["snippet_id"])
        if any(str(matches[query.query_id]["snippet_id"]) != snippet_id for query in batch):
            raise RuntimeError("Grouped execution mixed frozen MoCapAct snippets")
        policy = _load_policy(first_match, cfg, bank, bank_index)
        prebuilt_envs = (
            _make_prebuilt_envs(
                policy,
                first_match,
                bank,
                bank_index,
                include_bad=any(
                    query.panel_index in REPRESENTATIVE_PANEL_INDICES for query in batch
                ),
            )
            if sparse_media
            else None
        )
        for query in batch:
            match = matches[query.query_id]
            _update(
                status_path,
                "capturing_reference_baseline",
                query_index=query.panel_index,
                total_queries=len(queries),
                query_id=query.query_id,
                group_index=batch_index,
                total_groups=len(batches),
                snippet_id=snippet_id,
                group_windows=len(batch),
            )
            baseline, policy = _reference_baseline(
                query,
                match,
                run_dir,
                cfg,
                bank,
                bank_index,
                policy=policy,
                prebuilt_envs=prebuilt_envs,
                render_media=not sparse_media,
            )
            query_envs = prebuilt_envs
            if (
                sparse_media
                and prebuilt_envs is not None
                and query.panel_index not in REPRESENTATIVE_PANEL_INDICES
            ):
                query_envs = (prebuilt_envs[0], prebuilt_envs[1], None)
            for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
                _update(
                    status_path,
                    "checkpoint_physics",
                    checkpoint=checkpoint_label,
                    checkpoint_index=checkpoint_index,
                    query_index=query.panel_index,
                    total_queries=len(queries),
                    query_id=query.query_id,
                    group_index=batch_index,
                    total_groups=len(batches),
                    snippet_id=snippet_id,
                    group_windows=len(batch),
                )
                result = _evaluate(
                    query,
                    checkpoint_index,
                    checkpoint_label,
                    match,
                    baseline,
                    run_dir / "stages" / checkpoint_label / "evals" / query.query_id,
                    cfg,
                    bank,
                    bank_index,
                    policy,
                    render_media=bool(
                        not sparse_media
                        or (
                            checkpoint_label == PRIMARY_CHECKPOINT
                            and query.panel_index in REPRESENTATIVE_PANEL_INDICES
                        )
                    ),
                    prebuilt_envs=query_envs,
                )
                by_checkpoint[checkpoint_label].append(result)

    stages = []
    for checkpoint_label in CHECKPOINT_LABELS:
        results = by_checkpoint[checkpoint_label]
        if len(results) != len(queries):
            raise RuntimeError(f"Incomplete V3 stage {checkpoint_label}")
        stage = {
            "checkpoint": checkpoint_label,
            "n_windows": len(results),
            "results": results,
        }
        _atomic_json(run_dir / "stages" / checkpoint_label / "summary.json", stage)
        stages.append(stage)
    complete = {
        "version": VERSION,
        "checkpoints": list(CHECKPOINT_LABELS),
        "n_windows_per_checkpoint": len(queries),
        "n_paired_prediction_rollouts": len(queries) * len(CHECKPOINT_LABELS),
        "primary_three_condition_recordings": len(queries),
        "oracle_preflight": oracle,
        "stages": stages,
    }
    _atomic_json(run_dir / "physics_summary.json", complete)
    _update(
        status_path,
        "physics_complete",
        summary_path=str((run_dir / "physics_summary.json").resolve()),
    )


if __name__ == "__main__":
    main()
