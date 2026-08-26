from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .level_walking import (
    CMU_ACTIVITY_INDEX_SHA256,
    CMU_ACTIVITY_INDEX_URL,
    LEVEL_WALKING_CLIP_IDS,
    LEVEL_WALKING_NORMALIZED_LABELS,
    filter_expert_bank_to_level_walking,
)
from .matching import motion_match_one_window
from .recording import record_compare_rollout
from .reference_centered import capture_unmodified_reference, load_reference_baseline
from .run import _ensure_reference_bank
from .run_gait120_reference_centered_v3 import (
    REPRESENTATIVE_PANEL_INDICES,
    _evaluate,
    _load_policy,
    _make_prebuilt_envs,
    _require_development_validation,
)
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
    _sha256,
)
from .sim import OverrideSpec
from .utils import resample_linear, set_global_determinism


BASE_VERSION = "GAIT120_LEVEL_WALKING_REFERENCE_CENTERED_PHYSICS_V4"
VERSION = "GAIT120_LEVEL_WALKING_REFERENCE_CENTERED_PHYSICS_V4_1_JSON_ORDER_REPAIR"
MATCHING_VERSION = "GAIT120_LEVEL_WALKING_STABILITY_QUALIFIED_MATCHING_V4"


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
        raise RuntimeError("V4 MuJoCo runtime probe failed") from exc
    _atomic_json(run_dir / "runtime_probe.json", result)


def _candidate_row(
    query: PanelQuery, candidate: Any, length: int, rank: int
) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "panel_index": query.panel_index,
        "subject": query.subject,
        "candidate_rank": int(rank),
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


def _rank_candidates(
    queries: list[PanelQuery],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
) -> dict[str, list[dict[str, Any]]]:
    path = run_dir / "matching_preflight" / "ranked_candidates.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if str(stored.get("version")) != MATCHING_VERSION:
            raise RuntimeError("Saved V4 candidate manifest has an unexpected version")
        rows = dict(stored.get("candidates_by_query", {}))
        expected_ids = [query.query_id for query in queries]
        if len(rows) != len(expected_ids) or set(rows) != set(expected_ids):
            raise RuntimeError("Saved V4 candidate order differs from the frozen panel")
        return {query_id: list(rows[query_id]) for query_id in expected_ids}

    ranked: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        match_hz = float(np.asarray(bank.sample_hz[0]).reshape(()))
        thigh = resample_linear(
            query.thigh_pitch_deg, src_hz=query.sample_hz, dst_hz=match_hz
        )
        knee = resample_linear(
            query.knee_flexion_deg, src_hz=query.sample_hz, dst_hz=match_hz
        )
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
            raise RuntimeError(f"No labeled level-walking match for {query.query_id}")
        ranked[query.query_id] = [
            _candidate_row(query, candidate, length, rank)
            for rank, candidate in enumerate(candidates)
        ]
    _atomic_json(
        path,
        {
            "version": MATCHING_VERSION,
            "n_queries": len(queries),
            "top_k": int(cfg.match_top_k),
            "candidates_by_query": ranked,
        },
    )
    return ranked


def _screen_attempt(
    query: PanelQuery,
    match: dict[str, Any],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
    policy: Any,
    prebuilt_envs: tuple[Any, Any, Any | None],
) -> dict[str, Any]:
    rank = int(match["candidate_rank"])
    safe_snippet = str(match["snippet_id"]).replace("/", "_").replace("\\", "_")
    q_dir = run_dir / "reference_screening" / query.query_id / f"rank_{rank:02d}_{safe_snippet}"
    summary_path = q_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    q_dir.mkdir(parents=True, exist_ok=True)
    npz_path = q_dir / "reference_capture.npz"
    bank_row = bank_index[str(match["snippet_id"])]
    start = int(match["start_step_in_snippet"])
    length = int(match["length"])
    nominal = np.asarray(bank.knee_deg[bank_row], dtype=np.float32)[start : start + length]
    snippet_start = int(np.asarray(bank.start_step[bank_row]).reshape(()))
    snippet_end = int(np.asarray(bank.end_step[bank_row]).reshape(()))
    if not npz_path.exists():
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
            render_media=False,
            prebuilt_envs=prebuilt_envs,
        )
    metrics = _record_metrics(npz_path, nominal)
    deterministic = False
    reason = ""
    try:
        load_reference_baseline(npz_path, required_steps=length)
        deterministic = True
    except RuntimeError as exc:
        reason = str(exc)
    reference = metrics["reference"]
    replay = metrics["fused"]
    passed = bool(
        deterministic
        and reference["recorded_steps"] == length
        and replay["recorded_steps"] == length
        and reference["balance_loss_step"] == -1
        and replay["balance_loss_step"] == -1
        and reference["fall_risk"] < ORACLE_MAX_FALL_RISK
        and replay["fall_risk"] < ORACLE_MAX_FALL_RISK
    )
    if not passed and not reason:
        reason = "unmodified expert failed the fixed completion, balance, or risk criterion"
    result = {
        "version": MATCHING_VERSION,
        "query_id": query.query_id,
        "match": match,
        "required_steps": length,
        "deterministic_replay_identical": deterministic,
        "metrics": metrics,
        "passed": passed,
        "failure_reason": None if passed else reason,
        "recording_npz": str(npz_path.resolve()),
        "recording_sha256": _sha256(npz_path),
    }
    _atomic_json(summary_path, result)
    return result


def _materialize_selected_baseline(
    query: PanelQuery, attempt: dict[str, Any], run_dir: Path
) -> None:
    q_dir = run_dir / "reference_baselines" / query.query_id
    q_dir.mkdir(parents=True, exist_ok=True)
    source = Path(str(attempt["recording_npz"]))
    target = q_dir / "reference_capture.npz"
    if target.exists():
        if _sha256(target) != str(attempt["recording_sha256"]):
            raise RuntimeError(f"Selected V4 baseline changed for {query.query_id}")
    else:
        shutil.copy2(source, target)
    summary = {
        "version": MATCHING_VERSION,
        "query_id": query.query_id,
        "steps": int(attempt["required_steps"]),
        "match": attempt["match"],
        "metrics": attempt["metrics"],
        "recording_npz": str(target.resolve()),
        "recording_sha256": _sha256(target),
        "selected_from_screening": str(source.resolve()),
        "selected_from_screening_sha256": str(attempt["recording_sha256"]),
        "media_rendered": False,
        "deterministic_replay_identical": True,
    }
    summary_path = q_dir / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise RuntimeError(f"Selected V4 baseline summary changed for {query.query_id}")
    else:
        _atomic_json(summary_path, summary)


def _stability_qualified_matching(
    queries: list[PanelQuery],
    ranked: dict[str, list[dict[str, Any]]],
    run_dir: Path,
    cfg: EvalConfig,
    bank: Any,
    bank_index: dict[str, int],
    status_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    aggregate_path = run_dir / "matching_preflight" / "summary.json"
    if aggregate_path.exists():
        summary = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not bool(summary.get("passed")):
            raise RuntimeError("Saved V4 stability-qualified matching did not pass")
        rows = list(summary.get("windows", []))
        return summary, {str(row["query_id"]): row for row in rows}

    selected: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {query.query_id: [] for query in queries}
    max_rank = max(len(rows) for rows in ranked.values())
    for rank in range(max_rank):
        unresolved = [query for query in queries if query.query_id not in selected]
        if not unresolved:
            break
        grouped: dict[str, list[tuple[PanelQuery, dict[str, Any]]]] = {}
        for query in unresolved:
            candidates = ranked[query.query_id]
            if rank < len(candidates):
                match = candidates[rank]
                grouped.setdefault(str(match["snippet_id"]), []).append((query, match))
        for snippet_id in sorted(grouped):
            group = grouped[snippet_id]
            first_match = group[0][1]
            policy = _load_policy(first_match, cfg, bank, bank_index)
            prebuilt_envs = _make_prebuilt_envs(
                policy, first_match, bank, bank_index, include_bad=False
            )
            for query, match in group:
                _update(
                    status_path,
                    "model_blind_reference_screening",
                    candidate_rank=rank,
                    query_id=query.query_id,
                    selected_windows=len(selected),
                    total_queries=len(queries),
                    snippet_id=snippet_id,
                )
                result = _screen_attempt(
                    query,
                    match,
                    run_dir,
                    cfg,
                    bank,
                    bank_index,
                    policy,
                    prebuilt_envs,
                )
                attempts[query.query_id].append(result)
                if bool(result["passed"]):
                    selected[query.query_id] = match
                    _materialize_selected_baseline(query, result, run_dir)
    if len(selected) != len(queries):
        missing = [query.query_id for query in queries if query.query_id not in selected]
        _atomic_json(
            run_dir / "matching_preflight" / "failed_summary.json",
            {
                "version": MATCHING_VERSION,
                "passed": False,
                "unmatched_queries": missing,
                "attempts_by_query": attempts,
            },
        )
        raise RuntimeError(f"No stable level-walking reference for {missing[:3]}")

    rows = [selected[query.query_id] for query in queries]
    knee = np.asarray([row["knee_rmse_deg"] for row in rows], dtype=np.float64)
    thigh = np.asarray([row["thigh_rms_deg"] for row in rows], dtype=np.float64)
    summary = {
        "version": MATCHING_VERSION,
        "windows": rows,
        "n_windows": len(rows),
        "n_unique_selected_snippets": len({str(row["snippet_id"]) for row in rows}),
        "selected_candidate_ranks": [int(row["candidate_rank"]) for row in rows],
        "mean_knee_rmse_deg": float(np.mean(knee)),
        "median_knee_rmse_deg": float(np.median(knee)),
        "p95_knee_rmse_deg": float(np.quantile(knee, 0.95)),
        "maximum_knee_rmse_deg": float(np.max(knee)),
        "mean_thigh_rms_deg": float(np.mean(thigh)),
        "median_thigh_rms_deg": float(np.median(thigh)),
        "p95_thigh_rms_deg": float(np.quantile(thigh, 0.95)),
        "maximum_thigh_rms_deg": float(np.max(thigh)),
        "maximum_allowed_mean_knee_rmse_deg": MATCH_MAX_MEAN_KNEE_RMSE_DEG,
        "maximum_allowed_mean_thigh_rms_deg": MATCH_MAX_MEAN_THIGH_RMS_DEG,
        "model_predictions_used_for_matching_or_screening": False,
        "passed": bool(
            len(rows) == 80
            and float(np.mean(knee)) <= MATCH_MAX_MEAN_KNEE_RMSE_DEG
            and float(np.mean(thigh)) <= MATCH_MAX_MEAN_THIGH_RMS_DEG
        ),
        "attempts_by_query": attempts,
    }
    _atomic_json(aggregate_path, summary)
    if not bool(summary["passed"]):
        raise RuntimeError("V4 stability-qualified motion-matching gate failed")
    return summary, selected


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
            raise RuntimeError("Saved V4 oracle preflight did not pass")
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
        length = int(match["length"])
        baseline_path = run_dir / "reference_baselines" / query.query_id / "reference_capture.npz"
        baseline = load_reference_baseline(baseline_path, required_steps=length)
        policy = _load_policy(match, cfg, bank, bank_index)
        prebuilt_envs = _make_prebuilt_envs(
            policy, match, bank, bank_index, include_bad=False
        )
        bank_row = bank_index[str(match["snippet_id"])]
        start = int(match["start_step_in_snippet"])
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
                render_media=False,
                prebuilt_envs=prebuilt_envs,
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
            "media_rendered": False,
        }
        _atomic_json(summary_path, result)
        results.append(result)
    passing = sum(bool(row["passed"]) for row in results)
    summary = {
        "version": VERSION,
        "parent_version": BASE_VERSION,
        "fixed_order_panel_indices": [int(value) for value in order.tolist()],
        "windows": results,
        "passing_windows": passing,
        "required_passing_windows": ORACLE_MINIMUM_PASSING,
        "passed": bool(len(results) == ORACLE_WINDOWS and passing >= ORACLE_MINIMUM_PASSING),
    }
    _atomic_json(aggregate_path, summary)
    if not bool(summary["passed"]):
        raise RuntimeError(f"V4 oracle physics preflight failed: {passing}/{ORACLE_WINDOWS}")
    return summary


def _protocol(
    root: Path,
    panel_source_dir: Path,
    panel_manifest: dict[str, Any],
    development_dir: Path,
    sparse_validation_dir: Path,
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "parent_version": BASE_VERSION,
        "panel_version": PANEL_VERSION,
        "panel_source_dir": str(panel_source_dir.resolve()),
        "panel_manifest_sha256": _sha256(panel_source_dir / "panel_manifest.json"),
        "panel_selection": panel_manifest["protocol"]["selection"],
        "checkpoints": list(CHECKPOINT_LABELS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "level_walking_reference_gate": {
            "official_activity_index_url": CMU_ACTIVITY_INDEX_URL,
            "official_activity_index_sha256": CMU_ACTIVITY_INDEX_SHA256,
            "retrieved_date": "2026-08-25",
            "accepted_normalized_labels": list(LEVEL_WALKING_NORMALIZED_LABELS),
            "eligible_clip_ids": sorted(LEVEL_WALKING_CLIP_IDS),
            "eligible_clip_count": len(LEVEL_WALKING_CLIP_IDS),
            "eligible_expert_snippet_count": 157,
            "candidate_manifest": str(candidate_manifest_path.resolve()),
            "candidate_manifest_sha256": _sha256(candidate_manifest_path),
            "ranked_candidates_per_window": 12,
            "selection": (
                "first rank-ordered candidate whose deterministic unmodified expert "
                "completes all steps twice without balance loss and with both fall-risk "
                "values below 0.70"
            ),
            "screening_uses_model_predictions": False,
            "failed_candidate_attempts_retained": True,
            "maximum_mean_knee_rmse_deg": MATCH_MAX_MEAN_KNEE_RMSE_DEG,
            "maximum_mean_thigh_rms_deg": MATCH_MAX_MEAN_THIGH_RMS_DEG,
        },
        "development_validation_dir": str(development_dir.resolve()),
        "development_validation_summary_sha256": _sha256(
            development_dir / "validation_summary.json"
        ),
        "sparse_media_validation_dir": str(sparse_validation_dir.resolve()),
        "sparse_media_validation_summary_sha256": _sha256(
            sparse_validation_dir / "validation_summary.json"
        ),
        "oracle_preflight": {
            "windows": ORACLE_WINDOWS,
            "selection_seed": SEED + 3407,
            "minimum_passing": ORACLE_MINIMUM_PASSING,
            "maximum_tracking_rmse_deg": ORACLE_MAX_TRACKING_RMSE_DEG,
            "maximum_reference_and_oracle_fall_risk": ORACLE_MAX_FALL_RISK,
            "balance_loss_permitted": False,
        },
        "mapping": {
            "baseline": "realized knee from the selected unmodified expert replay",
            "target": "baseline + knee_sign*(prediction-measurement)",
            "source_to_control": "direct source-frame indices 0,3,6,...",
            "interpolation": False,
            "prediction_or_data_clipping": False,
            "scaling": False,
        },
        "simulation": {
            "controller": "moving-target PD; unchanged Kp=800, Kd=40, force=800",
            "fixed_windows": 80,
            "rendered_primary_panel_indices": list(REPRESENTATIVE_PANEL_INDICES),
            "representative_rule": "fixed evenly spaced panel indices; independent of outcomes",
            "all_numeric_rollouts_saved": True,
            "all_prediction_falls_retained": True,
            "prediction_instability_used_for_eligibility": False,
            "grouping_key": "selected MoCapAct snippet_id",
        },
        "analysis_unchanged": [
            "XCoM and support-margin traces",
            "excess-instability AUC",
            "participant-aware FWL partial Spearman analysis",
            "seven-point prediction-accuracy path and change-point analysis",
        ],
        "code_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "protocol": _sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "level_walking": _sha256(root / "mocap_phys_eval" / "level_walking.py"),
            "reference_centered": _sha256(root / "mocap_phys_eval" / "reference_centered.py"),
            "recording": _sha256(root / "mocap_phys_eval" / "recording.py"),
            "matching": _sha256(root / "mocap_phys_eval" / "matching.py"),
            "simulation": _sha256(root / "mocap_phys_eval" / "sim.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--panel-source-dir", type=Path, required=True)
    parser.add_argument("--development-validation-dir", type=Path, required=True)
    parser.add_argument("--sparse-media-validation-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-probe-only", action="store_true")
    parser.add_argument("--matching-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    panel_source_dir = args.panel_source_dir.resolve()
    development_dir = args.development_validation_dir.resolve()
    sparse_validation_dir = args.sparse_media_validation_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    if run_dir in {panel_source_dir, development_dir, sparse_validation_dir}:
        raise RuntimeError("V4 output must not overwrite prior evidence")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "pipeline_status.json"
    _runtime_probe(run_dir)
    if args.runtime_probe_only:
        return
    _require_development_validation(development_dir)
    sparse_summary_path = sparse_validation_dir / "validation_summary.json"
    if not sparse_summary_path.exists():
        raise RuntimeError("V4 requires the sparse-media compatibility validation")
    sparse_summary = json.loads(sparse_summary_path.read_text(encoding="utf-8"))
    if not bool(sparse_summary.get("passed")) or int(
        sparse_summary.get("n_compared_numeric_fields", 0)
    ) != 76:
        raise RuntimeError("Sparse-media compatibility validation did not pass 76 fields")

    panel_manifest, queries = _load_panel(panel_source_dir)
    set_global_determinism(seed=0)
    cfg = EvalConfig(artifacts_dir=artifacts_dir, device=str(args.device))
    if str(cfg.match_feature_mode) != "thigh_knee_d":
        raise RuntimeError("V4 requires the original scalar matcher")
    if float(cfg.match_knee_weight) != 1.0 or float(cfg.match_thigh_weight) != 0.0:
        raise RuntimeError("Original motion-matching weights changed")
    if int(cfg.match_top_k) != 12:
        raise RuntimeError("V4 requires the prospectively fixed top-12 candidate panel")

    _update(status_path, "ensuring_mocapact_reference_bank")
    full_bank = _ensure_reference_bank(cfg, out_root=artifacts_dir)
    bank, source_indices = filter_expert_bank_to_level_walking(full_bank)
    if len(bank) != 157 or source_indices.size != 157:
        raise RuntimeError("V4 level-walking reference filter changed")
    bank_index = {str(bank.snippet_id[index]): int(index) for index in range(len(bank))}

    _update(status_path, "matching_level_walking_candidates")
    ranked = _rank_candidates(queries, run_dir, cfg, bank)
    candidate_manifest_path = run_dir / "matching_preflight" / "ranked_candidates.json"
    protocol = _protocol(
        root,
        panel_source_dir,
        panel_manifest,
        development_dir,
        sparse_validation_dir,
        candidate_manifest_path,
    )
    parent_protocol_path = run_dir / "physics_protocol.level_walking_v4.json"
    if not parent_protocol_path.exists():
        raise RuntimeError("V4.1 requires the preserved pre-repair V4 protocol")
    protocol["operational_repair"] = {
        "parent_protocol": str(parent_protocol_path.resolve()),
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "failure_stage": "loading the saved ranked-candidate JSON before reference screening",
        "failure_cause": "JSON key sorting differed from frozen panel order",
        "repair": "validate the exact identifier set, then restore frozen panel order explicitly",
        "scientific_design_changed": False,
        "physics_outcomes_before_repair": 0,
        "reference_screening_outcomes_before_repair": 0,
    }
    protocol_path = run_dir / "physics_protocol.level_walking_v4_1.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing V4 protocol differs from executable protocol")
    else:
        _atomic_json(protocol_path, protocol)
    if args.matching_only:
        first_rows = [ranked[query.query_id][0] for query in queries]
        knee = np.asarray([row["knee_rmse_deg"] for row in first_rows], dtype=np.float64)
        thigh = np.asarray([row["thigh_rms_deg"] for row in first_rows], dtype=np.float64)
        _atomic_json(
            run_dir / "matching_preflight" / "top_rank_dry_summary.json",
            {
                "version": MATCHING_VERSION,
                "n_windows": len(first_rows),
                "mean_knee_rmse_deg": float(np.mean(knee)),
                "mean_thigh_rms_deg": float(np.mean(thigh)),
                "passed_mean_error_gates": bool(
                    float(np.mean(knee)) <= MATCH_MAX_MEAN_KNEE_RMSE_DEG
                    and float(np.mean(thigh)) <= MATCH_MAX_MEAN_THIGH_RMS_DEG
                ),
            },
        )
        _update(status_path, "matching_dry_validation_complete")
        return

    matching_summary, matches = _stability_qualified_matching(
        queries, ranked, run_dir, cfg, bank, bank_index, status_path
    )
    _update(
        status_path,
        "stability_qualified_matching_complete",
        selected_windows=len(matches),
        mean_knee_rmse_deg=matching_summary["mean_knee_rmse_deg"],
        mean_thigh_rms_deg=matching_summary["mean_thigh_rms_deg"],
    )

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
        paths = [
            run_dir / "stages" / label / "evals" / query.query_id / "summary.json"
            for label in CHECKPOINT_LABELS
        ]
        if all(path.exists() for path in paths):
            for label, path in zip(CHECKPOINT_LABELS, paths):
                by_checkpoint[label].append(json.loads(path.read_text(encoding="utf-8")))
        else:
            incomplete_queries.append(query)

    grouped: dict[str, list[PanelQuery]] = {}
    for query in incomplete_queries:
        grouped.setdefault(str(matches[query.query_id]["snippet_id"]), []).append(query)
    batches = [
        grouped[key]
        for key in sorted(grouped, key=lambda value: (-len(grouped[value]), value))
    ]
    completed_before = len(queries) - len(incomplete_queries)
    for batch_index, batch in enumerate(batches):
        first_match = matches[batch[0].query_id]
        snippet_id = str(first_match["snippet_id"])
        policy = _load_policy(first_match, cfg, bank, bank_index)
        prebuilt_envs = _make_prebuilt_envs(
            policy,
            first_match,
            bank,
            bank_index,
            include_bad=any(
                query.panel_index in REPRESENTATIVE_PANEL_INDICES for query in batch
            ),
        )
        for query in batch:
            match = matches[query.query_id]
            baseline = load_reference_baseline(
                run_dir / "reference_baselines" / query.query_id / "reference_capture.npz",
                required_steps=int(match["length"]),
            )
            query_envs = prebuilt_envs
            if query.panel_index not in REPRESENTATIVE_PANEL_INDICES:
                query_envs = (prebuilt_envs[0], prebuilt_envs[1], None)
            for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
                _update(
                    status_path,
                    "checkpoint_physics",
                    checkpoint=checkpoint_label,
                    checkpoint_index=checkpoint_index,
                    query_index=query.panel_index,
                    completed_windows=completed_before,
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
                        checkpoint_label == PRIMARY_CHECKPOINT
                        and query.panel_index in REPRESENTATIVE_PANEL_INDICES
                    ),
                    prebuilt_envs=query_envs,
                )
                by_checkpoint[checkpoint_label].append(result)
            completed_before += 1

    stages = []
    for checkpoint_label in CHECKPOINT_LABELS:
        results = sorted(
            by_checkpoint[checkpoint_label], key=lambda row: int(row["panel_index"])
        )
        if len(results) != len(queries):
            raise RuntimeError(f"Incomplete V4 stage {checkpoint_label}")
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
        "matching_preflight": matching_summary,
        "oracle_preflight": oracle,
        "stages": stages,
    }
    _atomic_json(run_dir / "physics_summary.json", complete)
    _update(
        status_path,
        "physics_complete",
        completed_windows=len(queries),
        completed_simulations=len(queries) * len(CHECKPOINT_LABELS),
        summary_path=str((run_dir / "physics_summary.json").resolve()),
    )


if __name__ == "__main__":
    main()
