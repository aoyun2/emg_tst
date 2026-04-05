from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bvh import extract_right_leg_thigh_quat_and_knee_included_deg, load_bvh
from .config import EvalConfig
from .experts import discover_expert_snippets, ensure_full_expert_zoo
from .matching import MatchCandidate, motion_match_one_window
from .plots import plot_balance_traces, plot_motion_match, plot_simulation_knee, plot_thigh_quat_match
from .recording import record_compare_rollout
from .reference_bank import ExpertSnippetBank, build_expert_snippet_bank
from .sim import OverrideSpec, load_expert_policy
from .utils import (
    dataclass_to_json_dict,
    download_to,
    ensure_dir,
    now_run_id,
    quat_conj_wxyz,
    quat_geodesic_deg_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    resample_linear,
    resample_quat_slerp_wxyz,
    set_global_determinism,
    write_json,
)


@dataclass(frozen=True)
class QueryWindow:
    query_id: str
    source: str
    sample_hz: float
    thigh_quat_wxyz: np.ndarray  # (W,4) at sample_hz
    knee_included_deg: np.ndarray  # (W,) at sample_hz, included angle (0 bent, 180 straight)
    thigh_pitch_deg: np.ndarray | None = None  # optional demo-only visualization
    X_raw: np.ndarray | None = None  # (W,F) raw features (rigtest only), for model inference
    dataset_index: int | None = None
    file_id: int | None = None
    file_name: str | None = None
    window_start: int | None = None
    outer_fold: int | None = None
    held_out_file_name: str | None = None
    tst_model: LoadedTstModel | None = None


@dataclass(frozen=True)
class LoadedTstModel:
    model: Any
    mean: np.ndarray  # (F,)
    std: np.ndarray  # (F,)
    feature_cols: np.ndarray  # (n_vars,)
    device: str
    ckpt_path: str
    # Number of leading features that received per-recording z-score at training time.
    # 0 means no per-recording normalization (older checkpoints).
    n_emg_norm: int = 0
    label_shift: int = 0
    label_scale: float = 180.0


@dataclass(frozen=True)
class FoldAssignedTstModel:
    fold: int
    held_out_file_ids: tuple[int, ...]
    held_out_file_names: tuple[str, ...]
    test_indices: np.ndarray
    model: LoadedTstModel
    run_dir: str


@dataclass(frozen=True)
class QueryEvalResult:
    result: dict[str, Any]
    compare_npz: Path
    compare_gif: Path
    motion_match_plot: Path
    thigh_quat_plot: Path


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 1:
        return float("nan")
    return float(np.sqrt(float(np.mean((a[:n] - b[:n]) ** 2))))


def _mean_abs(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 1:
        return float("nan")
    return float(np.mean(np.abs(a)))


def _risk_auc(risk_trace: np.ndarray, *, dt: float) -> float:
    risk = np.asarray(risk_trace, dtype=np.float64).reshape(-1)
    risk = risk[np.isfinite(risk)]
    if risk.size < 1:
        return float("nan")
    step = float(dt)
    if not np.isfinite(step) or step <= 0.0:
        return float("nan")
    if risk.size == 1:
        return float(risk[0] * step)
    return float(np.trapezoid(risk, dx=step))


def _make_bad_knee_prediction(
    knee_good_deg: np.ndarray,
    *,
    sample_hz: float,
    target_rmse_deg: float,
    lowpass_hz: float,
    seed: int = 0,
) -> np.ndarray:
    """Smooth deterministic "bad model" knee with ~target_rmse (deg) vs knee_good.

    We intentionally avoid large discontinuities (e.g., 0<->180 spikes) by generating
    band-limited noise and scaling to the desired RMSE.
    """
    knee_good = np.asarray(knee_good_deg, dtype=np.float64).reshape(-1)
    n = int(knee_good.size)
    if n < 2:
        return knee_good.astype(np.float32)

    hz = float(max(1e-6, float(sample_hz)))
    lp = float(max(0.10, float(lowpass_hz)))
    lp = min(lp, 0.45 * hz)  # keep below Nyquist

    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal(n).astype(np.float64)

    # FFT low-pass (deterministic and simple).
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    mask = freqs <= lp
    X[~mask] = 0.0
    e = np.fft.irfft(X, n=n)

    # Normalize to target RMSE.
    e = e - float(np.mean(e))
    rms = float(np.sqrt(float(np.mean(e**2))))
    if not np.isfinite(rms) or rms < 1e-8:
        return knee_good.astype(np.float32)
    e = e * (float(target_rmse_deg) / rms)

    bad = knee_good + e
    bad = np.clip(bad, 0.0, 170.0)
    return bad.astype(np.float32)


def _load_tst_checkpoint(ckpt_path: Path, *, device: str) -> LoadedTstModel:
    try:
        import torch

        from emg_tst.model import build_last_step_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load the sensor-fusion checkpoint.") from e

    dev = str(device).strip().lower()
    torch_dev = torch.device("cpu" if dev in ("", "cpu") else dev)

    ckpt = torch.load(ckpt_path, map_location=torch_dev, weights_only=False)
    cfg = ckpt["model_cfg"]
    reg = build_last_step_model(**cfg).to(torch_dev)
    state = ckpt.get("model_state_dict", ckpt.get("reg_state_dict"))
    reg.load_state_dict(state, strict=True)
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)

    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    std = np.asarray(scaler.get("std", None), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.size != std.size or mean.size < 1:
        raise RuntimeError(f"Invalid scaler in checkpoint: {str(ckpt_path)!r}")

    extra = ckpt.get("extra", {})
    cols = np.asarray(extra.get("feature_cols", None), dtype=np.int64)
    if cols.ndim != 1 or cols.size != int(cfg["n_vars"]):
        raise RuntimeError(
            f"Checkpoint is missing feature_cols (needed to reproduce training-time feature selection): {str(ckpt_path)!r}"
        )

    n_emg_norm = int(extra.get("n_emg_norm", 0))
    task_cfg = ckpt.get("task_cfg", {})
    label_shift = int(task_cfg.get("label_shift", extra.get("label_shift", 0)))
    label_scale = float(task_cfg.get("label_scale", extra.get("label_scale", 180.0)))

    return LoadedTstModel(
        model=reg,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_cols=cols.astype(np.int64),
        device=str(torch_dev),
        ckpt_path=str(ckpt_path),
        n_emg_norm=int(n_emg_norm),
        label_shift=int(label_shift),
        label_scale=float(label_scale),
    )


def _all_training_run_dirs() -> list[Path]:
    ckpts_root = Path("checkpoints")
    if not ckpts_root.exists():
        return []
    run_dirs = sorted(
        {
            p.parent.parent
            for p in ckpts_root.glob("*_all/fold_*/reg_best.pt")
            if p.parent.parent.is_dir()
        },
        key=lambda p: p.name,
    )
    return run_dirs


def _checkpoint_expected_feature_count(ckpt_path: Path) -> int | None:
    try:
        import torch
    except Exception:
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    if mean.ndim != 1 or mean.size < 1:
        return None
    return int(mean.size)


def _run_expected_feature_count(run_dir: Path) -> int | None:
    for ckpt_path in sorted(run_dir.glob("fold_*/reg_best.pt")):
        feat_count = _checkpoint_expected_feature_count(ckpt_path)
        if feat_count is not None:
            return int(feat_count)
    return None


def _discover_latest_all_training_run(
    *,
    expected_n_features: int | None = None,
    run_dir_override: Path | None = None,
) -> Path | None:
    if run_dir_override is not None:
        cand = Path(run_dir_override).expanduser()
        if cand.exists() and cand.is_dir():
            if expected_n_features is None or _run_expected_feature_count(cand) == int(expected_n_features):
                return cand
        return None
    run_dirs = _all_training_run_dirs()
    if expected_n_features is None:
        return run_dirs[-1] if run_dirs else None

    compatible = [
        run_dir
        for run_dir in run_dirs
        if _run_expected_feature_count(run_dir) == int(expected_n_features)
    ]
    return compatible[-1] if compatible else None


def _training_seed_for_run(run_dir: Path) -> int:
    base_name = str(run_dir.name)
    if base_name.endswith("_all"):
        base_name = base_name[:-4]
    summary_path = run_dir.parent / base_name / "ablation_summary.json"
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(obj.get("config", {}).get("seed", 7))
    except Exception:
        return 7


def _load_samples_dataset(cfg: EvalConfig) -> dict[str, Any] | None:
    samples_path = Path(cfg.rig_samples_path).expanduser()
    if not samples_path.exists():
        return None

    data = np.load(samples_path, allow_pickle=True).item()
    X = np.asarray(data["X"], dtype=np.float32)
    y_seq = np.asarray(data.get("y_seq", None), dtype=np.float32)
    if y_seq is None or y_seq.ndim != 2:
        raise RuntimeError(f"{str(samples_path)!r} is missing required key 'y_seq'. Rebuild with split_to_samples.py.")

    N, W, F = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    if W != int(cfg.window_n):
        raise RuntimeError(
            f"samples_dataset window={W} does not match mocap_phys_eval window_n={int(cfg.window_n)}. "
            "Keep train/eval windows the same length."
        )
    if int(y_seq.shape[1]) != W:
        raise RuntimeError(f"Unexpected y_seq shape {y_seq.shape}; expected (N,{W}).")

    thigh_n = int(data.get("thigh_n_features", 0))
    thigh_mode = str(data.get("thigh_mode", "")).strip().lower()
    thigh_pitch_seq = np.asarray(data.get("thigh_pitch_seq", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    thigh_quat_seq = np.asarray(data.get("thigh_quat_seq", np.zeros((0, 0, 0), dtype=np.float32)), dtype=np.float32)
    has_pitch_seq = bool(thigh_pitch_seq.ndim == 2 and int(thigh_pitch_seq.shape[0]) == N and int(thigh_pitch_seq.shape[1]) == W)
    has_quat_seq = bool(
        thigh_quat_seq.ndim == 3
        and int(thigh_quat_seq.shape[0]) == N
        and int(thigh_quat_seq.shape[1]) == W
        and int(thigh_quat_seq.shape[2]) == 4
    )
    if not has_quat_seq and not has_pitch_seq and (thigh_n != 4 or thigh_mode != "quat"):
        raise RuntimeError(
            f"samples_dataset has thigh_mode={thigh_mode!r} thigh_n_features={thigh_n}; "
            "this evaluation pipeline requires either thigh_quat_wxyz (4D quaternion) "
            "or stored thigh_quat_seq / thigh_pitch_seq for motion matching."
        )

    data["_samples_path"] = str(samples_path)
    data["_n_samples"] = int(N)
    data["_n_features"] = int(F)
    return data


def _load_cv_manifest_for_run(run_dir: Path, *, samples_data: dict[str, Any]) -> dict[str, Any]:
    def _file_names_array() -> np.ndarray:
        return np.asarray(samples_data.get("file_names", np.array([])))

    def _file_ids_array() -> np.ndarray:
        return np.asarray(samples_data.get("file_id", np.zeros((0,), dtype=np.int32)), dtype=np.int64).reshape(-1)

    def _resolve_split_manifest_path(fold_entry: dict[str, Any], fold: int) -> Path | None:
        candidates: list[Path] = []
        ref = fold_entry.get("split_manifest", None)
        if isinstance(ref, str) and ref.strip():
            p = Path(ref)
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(run_dir / p)
                candidates.append(run_dir / p.name)
        candidates.append(run_dir / f"fold_{int(fold):02d}" / "split_manifest.json")
        seen: set[str] = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists():
                return p
        return None

    def _normalize_manifest(manifest_obj: dict[str, Any]) -> dict[str, Any]:
        file_ids = _file_ids_array()
        file_names = _file_names_array()
        seed0 = int(manifest_obj.get("seed", _training_seed_for_run(run_dir)))
        folds_in = list(manifest_obj.get("folds", []))
        total_folds = int(manifest_obj.get("n_folds", len(folds_in) or 1))
        split_strategy0 = str(manifest_obj.get("split_strategy", "unknown"))
        folds_out: list[dict[str, Any]] = []

        for i, raw_fold in enumerate(folds_in, start=1):
            fold_entry = dict(raw_fold)
            fold = int(fold_entry.get("fold", i))
            fold_entry.setdefault("fold", int(fold))
            fold_entry.setdefault("seed", int(seed0))
            fold_entry.setdefault("split_strategy", split_strategy0)

            test_idx = np.asarray(fold_entry.get("test_indices", []), dtype=np.int64).reshape(-1)
            if test_idx.size < 1:
                split_path = _resolve_split_manifest_path(fold_entry, int(fold))
                if split_path is not None:
                    try:
                        split_obj = json.loads(split_path.read_text(encoding="utf-8"))
                    except Exception:
                        split_obj = {}
                    if split_obj:
                        if test_idx.size < 1:
                            test_idx = np.asarray(split_obj.get("test_indices", []), dtype=np.int64).reshape(-1)
                        if not fold_entry.get("test_file_ids"):
                            fold_entry["test_file_ids"] = list(split_obj.get("test_file_ids", []))
                        if not fold_entry.get("test_file_names"):
                            fold_entry["test_file_names"] = list(split_obj.get("test_file_names", []))

            if test_idx.size < 1:
                test_file_ids = np.asarray(fold_entry.get("test_file_ids", []), dtype=np.int64).reshape(-1)
                if test_file_ids.size < 1:
                    test_file_names = [str(x) for x in fold_entry.get("test_file_names", [])]
                    if test_file_names and int(file_names.size) > 0:
                        matched: list[int] = []
                        for name in test_file_names:
                            hits = np.where(file_names.astype(str) == str(name))[0].astype(np.int64)
                            matched.extend(int(x) for x in hits.tolist())
                        test_file_ids = np.asarray(sorted(set(matched)), dtype=np.int64)
                if test_file_ids.size > 0 and file_ids.size > 0:
                    test_idx = np.where(np.isin(file_ids, test_file_ids))[0].astype(np.int64)

            if test_idx.size < 1 and split_strategy0 == "kfold" and int(file_ids.size) > 0:
                rng = np.random.default_rng(int(fold_entry.get("seed", seed0)))
                perm = rng.permutation(int(file_ids.size))
                split = np.array_split(perm, int(max(1, total_folds)))
                pos = int(fold) - 1
                if 0 <= pos < len(split):
                    test_idx = np.asarray(split[pos], dtype=np.int64).reshape(-1)

            if test_idx.size > 0:
                fold_entry["test_indices"] = test_idx.astype(np.int64).tolist()
                if not fold_entry.get("test_file_ids") and file_ids.size > 0:
                    fold_entry["test_file_ids"] = np.unique(file_ids[test_idx]).astype(np.int64).tolist()
                if not fold_entry.get("test_file_names") and int(file_names.size) > 0:
                    names: list[str] = []
                    for fid in fold_entry.get("test_file_ids", []):
                        fid_i = int(fid)
                        if 0 <= fid_i < int(file_names.size):
                            names.append(str(file_names[fid_i]))
                    fold_entry["test_file_names"] = names

            folds_out.append(fold_entry)

        out = dict(manifest_obj)
        out["seed"] = int(seed0)
        out["split_strategy"] = split_strategy0
        out["n_folds"] = int(len(folds_out))
        out["folds"] = folds_out
        return out

    manifest_path = run_dir / "cv_manifest.json"
    if manifest_path.exists():
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(obj.get("folds", None), list) and obj["folds"]:
                return _normalize_manifest(obj)
        except Exception:
            pass

    split_paths = sorted(run_dir.glob("fold_*/split_manifest.json"))
    if split_paths:
        folds = []
        for p in split_paths:
            folds.append(json.loads(p.read_text(encoding="utf-8")))
        return _normalize_manifest({
            "label": "ALL",
            "split_strategy": str(folds[0].get("split_strategy", "unknown")),
            "seed": int(folds[0].get("seed", _training_seed_for_run(run_dir))),
            "samples_file": str(folds[0].get("samples_file", samples_data.get("_samples_path", ""))),
            "n_folds": int(len(folds)),
            "folds": folds,
        })

    file_ids = np.asarray(samples_data.get("file_id", np.zeros((0,), dtype=np.int32)), dtype=np.int64).reshape(-1)
    file_names = np.asarray(samples_data.get("file_names", np.array([])))
    n = int(file_ids.size)
    if n < 1:
        raise RuntimeError("samples_dataset is empty.")

    fold_dirs = sorted(run_dir.glob("fold_*"))
    if not fold_dirs:
        raise RuntimeError(f"No fold checkpoints found under {run_dir}.")

    seed = _training_seed_for_run(run_dir)
    unique_files = np.unique(file_ids)
    folds: list[dict[str, Any]] = []
    if unique_files.size >= 2:
        if len(fold_dirs) != int(unique_files.size):
            raise RuntimeError(
                f"Training run {run_dir.name!r} has {len(fold_dirs)} fold dirs, but samples_dataset has "
                f"{int(unique_files.size)} LOFO files. Re-run training to refresh fold manifests."
            )
        for i, fid in enumerate(unique_files.tolist(), start=1):
            test_idx = np.where(file_ids == int(fid))[0].astype(np.int64)
            name = str(file_names[int(fid)]) if 0 <= int(fid) < int(len(file_names)) else f"file_{int(fid)}"
            folds.append(
                {
                    "fold": int(i),
                    "split_strategy": "lofo",
                    "seed": int(seed),
                    "samples_file": str(samples_data.get("_samples_path", "")),
                    "test_indices": test_idx.tolist(),
                    "test_file_ids": [int(fid)],
                    "test_file_names": [name],
                }
            )
    else:
        k = int(len(fold_dirs))
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        split = np.array_split(perm, k)
        for i, test_idx in enumerate(split, start=1):
            folds.append(
                {
                    "fold": int(i),
                    "split_strategy": "kfold",
                    "seed": int(seed),
                    "samples_file": str(samples_data.get("_samples_path", "")),
                    "test_indices": np.asarray(test_idx, dtype=np.int64).reshape(-1).tolist(),
                    "test_file_ids": [0],
                    "test_file_names": [
                        str(file_names[0]) if int(len(file_names)) >= 1 else "file_0",
                    ],
                }
            )

    return _normalize_manifest({
        "label": "ALL",
        "split_strategy": str(folds[0].get("split_strategy", "unknown")) if folds else "unknown",
        "seed": int(seed),
        "samples_file": str(samples_data.get("_samples_path", "")),
        "n_folds": int(len(folds)),
        "folds": folds,
    })


def _load_tst_fold_models_if_available(
    *,
    device: str,
    samples_data: dict[str, Any],
    run_dir_override: Path | None = None,
    allow_partial_coverage: bool = False,
) -> tuple[Path, dict[int, FoldAssignedTstModel]] | None:
    expected_n_features = int(np.asarray(samples_data["X"], dtype=np.float32).shape[2])
    run_dir = _discover_latest_all_training_run(
        expected_n_features=expected_n_features,
        run_dir_override=run_dir_override,
    )
    if run_dir is None:
        run_dirs = _all_training_run_dirs()
        if not run_dirs:
            return None
        details = []
        for cand in run_dirs:
            feat_count = _run_expected_feature_count(cand)
            suffix = "unknown" if feat_count is None else f"F={int(feat_count)}"
            details.append(f"{cand.name} ({suffix})")
        raise RuntimeError(
            "Found training runs, but none are compatible with the current samples_dataset feature layout. "
            f"Current samples_dataset has F={expected_n_features} features. "
            "Re-run `python -m emg_tst.run_experiment` after rebuilding samples_dataset.npy. "
            f"Available `_all` runs: {', '.join(details)}"
        )

    manifest = _load_cv_manifest_for_run(run_dir, samples_data=samples_data)
    sample_to_fold: dict[int, FoldAssignedTstModel] = {}

    for fold_entry in manifest.get("folds", []):
        fold = int(fold_entry.get("fold", 0))
        if fold <= 0:
            raise RuntimeError(f"Invalid fold entry in cv manifest: {fold_entry!r}")
        ckpt_path = run_dir / f"fold_{fold:02d}" / "reg_best.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Missing checkpoint for fold {fold}: {ckpt_path}")
        model = _load_tst_checkpoint(ckpt_path, device=device)
        test_indices = np.asarray(fold_entry.get("test_indices", []), dtype=np.int64).reshape(-1)
        assigned = FoldAssignedTstModel(
            fold=int(fold),
            held_out_file_ids=tuple(int(x) for x in fold_entry.get("test_file_ids", [])),
            held_out_file_names=tuple(str(x) for x in fold_entry.get("test_file_names", [])),
            test_indices=test_indices.astype(np.int64),
            model=model,
            run_dir=str(run_dir),
        )
        for idx in test_indices.tolist():
            idx = int(idx)
            if idx in sample_to_fold:
                raise RuntimeError(f"Duplicate held-out assignment for sample index {idx}.")
            sample_to_fold[idx] = assigned

    expected_n = int(np.asarray(samples_data["X"]).shape[0])
    if len(sample_to_fold) != expected_n:
        if bool(allow_partial_coverage) and len(sample_to_fold) > 0:
            return run_dir, sample_to_fold
        raise RuntimeError(
            f"Held-out pool coverage mismatch: mapped {len(sample_to_fold)} sample windows, expected {expected_n}. "
            "Re-run training so each outer fold writes a split manifest."
        )
    return run_dir, sample_to_fold


def _load_rigtest_query_pool(
    cfg: EvalConfig,
    *,
    samples_data: dict[str, Any],
    sample_to_fold: dict[int, FoldAssignedTstModel],
    seed: int,
) -> list[QueryWindow]:
    X = np.asarray(samples_data["X"], dtype=np.float32)
    y_seq = np.asarray(samples_data["y_seq"], dtype=np.float32)
    N, W, F = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    file_names = np.asarray(samples_data.get("file_names", np.array([])))
    file_id = np.asarray(samples_data.get("file_id", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    starts = np.asarray(samples_data.get("start", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    sample_hz = float(samples_data.get("sample_hz", cfg.window_hz))
    thigh_pitch_seq = np.asarray(samples_data.get("thigh_pitch_seq", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    has_pitch_seq = bool(thigh_pitch_seq.ndim == 2 and int(thigh_pitch_seq.shape[0]) == N and int(thigh_pitch_seq.shape[1]) == W)

    idxs = np.asarray(sorted(sample_to_fold.keys()), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idxs = rng.permutation(idxs)

    out: list[QueryWindow] = []
    for idx in idxs.tolist():
        idx = int(idx)
        xw = X[idx]
        if has_pitch_seq:
            q_pitch = np.asarray(thigh_pitch_seq[idx], dtype=np.float32).reshape(-1)
            q = _pitch_deg_to_quat_series_wxyz(q_pitch)
        else:
            q_pitch = None
            q = _normalize_quat_series_wxyz(xw[:, F - 4 : F])
        knee_inc = np.asarray(y_seq[idx], dtype=np.float32).reshape(-1)
        if knee_inc.size != W:
            knee_inc = knee_inc[:W]

        fid = int(file_id[idx]) if file_id.size == N else -1
        st = int(starts[idx]) if starts.size == N else -1
        file_name = str(file_names[fid]) if 0 <= fid < int(len(file_names)) else None
        fold_info = sample_to_fold[idx]
        held_out_name = fold_info.held_out_file_names[0] if fold_info.held_out_file_names else file_name

        src = (
            f"rig:{Path(str(samples_data.get('_samples_path', cfg.rig_samples_path))).name}:idx={idx}"
            f":file={file_name if file_name is not None else fid}:start={st}:fold={fold_info.fold}"
        )

        out.append(
            QueryWindow(
                query_id=f"rig_{idx:06d}",
                source=src,
                sample_hz=float(sample_hz),
                thigh_quat_wxyz=q.astype(np.float32),
                knee_included_deg=knee_inc.astype(np.float32),
                thigh_pitch_deg=None if q_pitch is None else q_pitch.astype(np.float32),
                X_raw=xw.astype(np.float32),
                dataset_index=int(idx),
                file_id=fid,
                file_name=file_name,
                window_start=st,
                outer_fold=int(fold_info.fold),
                held_out_file_name=held_out_name,
                tst_model=fold_info.model,
            )
        )
    return out


def _load_tst_model_if_available(*, device: str, expected_n_features: int | None = None) -> LoadedTstModel | None:
    """Auto-discover and load a sensor-fusion checkpoint if present.

    Selection policy (no CLI flags):
    - Prefer the latest training run directory that ends with `_all` (the default
      "ALL FEATURES" model produced by emg_tst/run_experiment.py).
    - Within that run, pick the fold with the lowest `metrics.json.best_rmse`.
    - If no metrics are available, fall back to the most recently modified checkpoint.
    """
    run_dir = _discover_latest_all_training_run(expected_n_features=expected_n_features)
    if run_dir is None:
        return None
    run_candidates = sorted(run_dir.glob("fold_*/reg_best.pt"))
    if not run_candidates:
        return None

    best_path = None
    best_rmse = None
    for p in run_candidates:
        mpath = p.parent / "metrics.json"
        if not mpath.exists():
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            rmse = float(m.get("best_rmse"))
        except Exception:
            continue
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_path = p

    ckpt_path = best_path if best_path is not None else max(run_candidates, key=lambda p: p.stat().st_mtime)
    return _load_tst_checkpoint(ckpt_path, device=device)


def _predict_knee_included_deg_for_window(model: LoadedTstModel, X_raw: np.ndarray) -> np.ndarray:
    """Return per-timestep knee included-angle prediction in degrees for one window."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run the sensor-fusion model.") from e

    x = np.asarray(X_raw, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"X_raw must be (W,F), got {x.shape}")
    W, F = int(x.shape[0]), int(x.shape[1])
    if model.mean.size != F or model.std.size != F:
        raise ValueError(f"Checkpoint scaler expects F={model.mean.size}, but window has F={F}")

    # Match training pipeline:
    # 1. Per-recording EMG z-score (applied to first n_emg_norm features during training).
    #    At inference we use per-window stats as a proxy (equivalent when the window
    #    is representative of its recording, which holds for 1-second walking windows).
    xn = x.copy()
    n_emg = int(model.n_emg_norm)
    if n_emg > 0 and n_emg <= F:
        emg_block = xn[:, :n_emg].astype(np.float64)
        mu = emg_block.mean(axis=0)
        sd = np.maximum(emg_block.std(axis=0), 1e-6)
        xn[:, :n_emg] = ((emg_block - mu) / sd).astype(np.float32)

    # 2. Global scaler (fitted on per-recording-normalized data during training).
    xn = (xn - model.mean[None, :]) / model.std[None, :]
    xn = xn[:, model.feature_cols]

    xb = torch.from_numpy(xn).to(model.device)
    with torch.no_grad():
        from emg_tst.model import rolling_last_step_predict

        pred = rolling_last_step_predict(model.model, xb).numpy().astype(np.float32)
    if pred.shape != (W,):
        pred = pred.reshape(-1)[:W].astype(np.float32)
    return (pred * float(model.label_scale)).astype(np.float32)


def _align_future_forecast_to_control(
    pred_future: np.ndarray,
    current_observed: np.ndarray,
    *,
    label_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align a future-angle forecast to a causal control target.

    `pred_future[j]` predicts the angle at time `j + label_shift`. For control,
    we keep the observed current angle for the prefix where no forecast has
    matured yet, then inject the model forecast once its target time arrives.

    Returns:
      aligned:    (W,) absolute angle sequence suitable for immediate control
      valid_mask: (W,) True where aligned values come from the model forecast
    """
    pred = np.asarray(pred_future, dtype=np.float32).reshape(-1)
    obs = np.asarray(current_observed, dtype=np.float32).reshape(-1)
    W = int(min(pred.size, obs.size))
    if W < 1:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    shift = int(max(0, label_shift))
    aligned = obs[:W].copy()
    valid = np.zeros((W,), dtype=bool)
    if shift <= 0:
        aligned = pred[:W].copy()
        valid[:] = True
        return aligned.astype(np.float32), valid

    if shift < W:
        aligned[shift:] = pred[: W - shift]
        valid[shift:] = True
    return aligned.astype(np.float32), valid


def _normalize_quat_series_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    if q.shape[0] < 1:
        return q.astype(np.float32)
    # Normalize and enforce sign continuity.
    qn = quat_normalize_wxyz(q)
    out = np.asarray(qn, dtype=np.float64)
    for i in range(1, int(out.shape[0])):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] *= -1.0
    return out.astype(np.float32)


def _pitch_deg_to_quat_series_wxyz(pitch_deg: np.ndarray) -> np.ndarray:
    pitch = np.asarray(pitch_deg, dtype=np.float64).reshape(-1)
    if pitch.size < 1:
        return np.zeros((0, 4), dtype=np.float32)
    half = 0.5 * np.deg2rad(pitch)
    q = np.stack(
        [
            np.cos(half),
            np.zeros_like(half),
            np.sin(half),
            np.zeros_like(half),
        ],
        axis=1,
    )
    return _normalize_quat_series_wxyz(q)


def _load_rigtest_query_windows(cfg: EvalConfig, *, max_n: int, seed: int) -> list[QueryWindow]:
    samples_path = Path(cfg.rig_samples_path).expanduser()
    if not samples_path.exists():
        return []

    data = np.load(samples_path, allow_pickle=True).item()
    X = np.asarray(data["X"], dtype=np.float32)  # (N,W,F)
    y_seq = np.asarray(data.get("y_seq", None), dtype=np.float32)  # (N,W)
    if y_seq is None or y_seq.ndim != 2:
        raise RuntimeError(f"{str(samples_path)!r} is missing required key 'y_seq'. Rebuild with split_to_samples.py.")

    N, W, F = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    if W != int(cfg.window_n):
        raise RuntimeError(
            f"samples_dataset window={W} does not match mocap_phys_eval window_n={int(cfg.window_n)}. "
            "Keep train/eval windows the same length."
        )
    if int(y_seq.shape[1]) != W:
        raise RuntimeError(f"Unexpected y_seq shape {y_seq.shape}; expected (N,{W}).")

    thigh_n = int(data.get("thigh_n_features", 0))
    thigh_mode = str(data.get("thigh_mode", "")).strip().lower()
    thigh_pitch_seq = np.asarray(data.get("thigh_pitch_seq", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    thigh_quat_seq = np.asarray(data.get("thigh_quat_seq", np.zeros((0, 0, 0), dtype=np.float32)), dtype=np.float32)
    has_pitch_seq = bool(thigh_pitch_seq.ndim == 2 and int(thigh_pitch_seq.shape[0]) == N and int(thigh_pitch_seq.shape[1]) == W)
    has_quat_seq = bool(
        thigh_quat_seq.ndim == 3
        and int(thigh_quat_seq.shape[0]) == N
        and int(thigh_quat_seq.shape[1]) == W
        and int(thigh_quat_seq.shape[2]) == 4
    )
    if not has_quat_seq and not has_pitch_seq and (thigh_n != 4 or thigh_mode != "quat"):
        raise RuntimeError(
            f"samples_dataset has thigh_mode={thigh_mode!r} thigh_n_features={thigh_n}; "
            "this evaluation pipeline requires either thigh_quat_wxyz (4D quaternion) "
            "or stored thigh_quat_seq / thigh_pitch_seq for motion matching."
        )

    file_names = data.get("file_names", None)
    file_id = np.asarray(data.get("file_id", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    starts = np.asarray(data.get("start", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)

    n_take = int(min(int(max_n), int(N)))
    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(N, size=n_take, replace=False) if n_take < N else np.arange(N, dtype=np.int64)

    out: list[QueryWindow] = []
    for idx in idxs.tolist():
        idx = int(idx)
        xw = X[idx]
        if has_quat_seq:
            q = _normalize_quat_series_wxyz(np.asarray(thigh_quat_seq[idx], dtype=np.float32).reshape(-1, 4))
            q_pitch = None
            if has_pitch_seq:
                q_pitch = np.asarray(thigh_pitch_seq[idx], dtype=np.float32).reshape(-1)
        elif has_pitch_seq:
            q_pitch = np.asarray(thigh_pitch_seq[idx], dtype=np.float32).reshape(-1)
            q = _pitch_deg_to_quat_series_wxyz(q_pitch)
        else:
            q_pitch = None
            q = _normalize_quat_series_wxyz(xw[:, F - 4 : F])
        knee_inc = np.asarray(y_seq[idx], dtype=np.float32).reshape(-1)
        if knee_inc.size != W:
            knee_inc = knee_inc[:W]

        src = f"rig:{samples_path.name}:idx={idx}"
        try:
            fid = int(file_id[idx]) if file_id.size == N else -1
            st = int(starts[idx]) if starts.size == N else -1
            if file_names is not None and fid >= 0:
                fn = str(np.asarray(file_names[fid]).reshape(()))
                src += f":file={fn}:start={st}"
        except Exception:
            pass

        out.append(
            QueryWindow(
                query_id=f"rig_{idx:06d}",
                source=src,
                sample_hz=float(cfg.window_hz),
                thigh_quat_wxyz=q.astype(np.float32),
                knee_included_deg=knee_inc.astype(np.float32),
                thigh_pitch_deg=None if q_pitch is None else q_pitch.astype(np.float32),
                X_raw=xw.astype(np.float32),
            )
        )
    return out


def _download_demo_bvh(cfg: EvalConfig, *, dst: Path, demo_idx: int) -> tuple[Path, str]:
    urls = tuple(getattr(cfg, "query_bvh_urls", ()))
    if not urls:
        raise RuntimeError("EvalConfig.query_bvh_urls is empty.")
    start_i = int(demo_idx) % int(len(urls))
    errs: list[str] = []
    for k in range(int(len(urls))):
        u = str(urls[(start_i + k) % int(len(urls))])
        print(f"[mocap_phys_eval] status: downloading demo BVH (try={k+1}/{len(urls)})  url={u}")
        try:
            p = download_to(u, dst, force=False, timeout_s=180.0)
            return p, u
        except Exception as e:
            errs.append(f"{u} :: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Failed to download any demo BVH.\n  - " + "\n  - ".join(errs))


def _rank_query_window_starts(
    thigh_pitch_deg: np.ndarray,
    knee_flex_deg: np.ndarray,
    *,
    window_n: int,
    sample_hz: float,
    skip_start_s: float,
    top_k: int,
) -> list[int]:
    th = np.asarray(thigh_pitch_deg, dtype=np.float32).reshape(-1)
    kn = np.asarray(knee_flex_deg, dtype=np.float32).reshape(-1)
    n = int(min(th.size, kn.size))
    w = int(window_n)
    if n <= w:
        return [0]

    nstarts = int(n - w + 1)
    skip = int(round(float(skip_start_s) * float(sample_hz)))
    skip = int(max(0, min(skip, nstarts - 1)))

    dth = np.abs(np.diff(th[:n], prepend=th[0])).astype(np.float64)
    dkn = np.abs(np.diff(kn[:n], prepend=kn[0])).astype(np.float64)
    energy = (dth + dkn).astype(np.float64)
    win = np.ones((w,), dtype=np.float64)
    score = np.convolve(energy, win, mode="valid")
    if skip > 0:
        score[:skip] = -np.inf

    order = np.argsort(score)[::-1]
    order = order[np.isfinite(score[order])]
    if order.size < 1:
        return [int(skip)]

    k = int(max(1, min(int(top_k), int(order.size))))
    starts = [int(x) for x in order[:k].tolist()]
    if int(skip) not in starts:
        starts.append(int(skip))
    return starts


def _load_demo_query_windows(cfg: EvalConfig, *, run_dir: Path, demo_idx: int, n_windows: int) -> list[QueryWindow]:
    bvh_path, bvh_url = _download_demo_bvh(cfg, dst=run_dir / "demo_query.bvh", demo_idx=demo_idx)

    print("[mocap_phys_eval] status: parsing demo BVH and extracting right-leg kinematics")
    bvh = load_bvh(bvh_path)
    thigh_pitch_deg, thigh_quat_wxyz, knee_included_deg, bvh_hz = extract_right_leg_thigh_quat_and_knee_included_deg(bvh)

    knee_flex_deg = (180.0 - np.asarray(knee_included_deg, dtype=np.float32)).astype(np.float32)
    print(f"[mocap_phys_eval] status: resampling demo BVH to {float(cfg.window_hz):.1f}Hz")
    th200 = resample_linear(thigh_pitch_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    kn200 = resample_linear(knee_flex_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    thq200 = resample_quat_slerp_wxyz(
        np.asarray(thigh_quat_wxyz, dtype=np.float32), src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz)
    )

    starts = _rank_query_window_starts(
        th200,
        kn200,
        window_n=int(cfg.window_n),
        sample_hz=float(cfg.window_hz),
        skip_start_s=float(cfg.query_window_skip_s),
        top_k=max(int(cfg.query_window_top_k), int(n_windows)),
    )
    n_take = int(max(1, int(n_windows)))
    # Deterministic variety: randomly sample from the top-ranked candidates.
    # (Demo-only; rigtest windows are sampled directly from samples_dataset.npy.)
    starts = starts[: int(max(int(cfg.query_window_top_k), int(n_take)))]
    if len(starts) > n_take:
        rng = np.random.default_rng(1000 + int(demo_idx))
        starts = rng.choice(np.asarray(starts, dtype=np.int64), size=int(n_take), replace=False).tolist()
    starts = sorted(int(s) for s in starts[:n_take])

    out: list[QueryWindow] = []
    for i, s in enumerate(starts):
        s = int(s)
        w = int(cfg.window_n)
        out.append(
            QueryWindow(
                query_id=f"demo_{demo_idx:03d}_{i:02d}",
                source=f"bvh_url={bvh_url} start_s={float(s)/float(cfg.window_hz):.3f}",
                sample_hz=float(cfg.window_hz),
                thigh_quat_wxyz=_normalize_quat_series_wxyz(thq200[s : s + w]),
                knee_included_deg=(180.0 - kn200[s : s + w]).astype(np.float32),  # included for schema
                thigh_pitch_deg=th200[s : s + w].astype(np.float32),
                X_raw=None,
            )
        )
    return out


def _npz_has_key(path: Path, key: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as d:
            return bool(key in getattr(d, "files", ()))
    except Exception:
        return False


def _ensure_reference_bank(cfg: EvalConfig, *, out_root: Path) -> ExpertSnippetBank:
    ensure_full_expert_zoo(experts_root=cfg.experts_root, downloads_dir=cfg.experts_downloads_dir)
    snippets = discover_expert_snippets(cfg.experts_root)
    print(f"[mocap_phys_eval] status: experts discovered (n_snippets={len(snippets)})")
    if len(snippets) < 2500:
        raise RuntimeError(
            f"Expected ~2589 expert snippets, but found only {len(snippets)} under {str(cfg.experts_root)!r}. "
            "This usually means the expert zoo download/extraction is incomplete."
        )

    bank_path = out_root / "reference_bank" / "mocapact_expert_snippets_right.npz"
    bank_ok = False
    if bank_path.exists():
        try:
            print(f"[mocap_phys_eval] status: loading reference bank ({bank_path})")
            bank = ExpertSnippetBank.load_npz(bank_path)
            bank_ok = bool(len(bank) >= int(len(snippets))) and _npz_has_key(bank_path, "thigh_anat_quat_world_wxyz")
        except Exception:
            bank_ok = False
    if not bank_ok:
        print("[mocap_phys_eval] status: building expert snippet reference bank (this may take a while)")
        bank = build_expert_snippet_bank(experts_root=cfg.experts_root, side="right")
        bank.save_npz(bank_path)
    return bank


def _align_query_thigh_quat(*, query_thq: np.ndarray, cand: MatchCandidate) -> np.ndarray:
    qoff = quat_normalize_wxyz(np.asarray(cand.thigh_quat_offset_wxyz, dtype=np.float64).reshape(4))
    q = quat_normalize_wxyz(np.asarray(query_thq, dtype=np.float64).reshape(-1, 4))
    if bool(getattr(cand, "thigh_quat_conjugated", False)):
        q = quat_conj_wxyz(q)
    q_al = quat_mul_wxyz(qoff[None, :], q)
    return np.asarray(q_al, dtype=np.float32)


def _evaluate_query_window(
    *,
    q: QueryWindow,
    q_dir: Path,
    query_mode: str,
    cfg: EvalConfig,
    bank: ExpertSnippetBank,
    bank_snip_to_i: dict[str, int],
    match_hz: float,
    noise_seed: int,
) -> QueryEvalResult:
    plots_dir = ensure_dir(q_dir / "plots")
    replay_dir = ensure_dir(q_dir / "replay")

    np.savez_compressed(
        q_dir / "query_window.npz",
        query_id=np.asarray(str(q.query_id)),
        source=np.asarray(str(q.source)),
        mode=np.asarray(str(query_mode)),
        sample_hz=np.asarray(float(q.sample_hz), dtype=np.float32),
        thigh_quat_wxyz=np.asarray(q.thigh_quat_wxyz, dtype=np.float32),
        knee_included_deg=np.asarray(q.knee_included_deg, dtype=np.float32),
        dataset_index=np.asarray(-1 if q.dataset_index is None else int(q.dataset_index), dtype=np.int64),
        file_id=np.asarray(-1 if q.file_id is None else int(q.file_id), dtype=np.int64),
        file_name=np.asarray("" if q.file_name is None else str(q.file_name)),
        window_start=np.asarray(-1 if q.window_start is None else int(q.window_start), dtype=np.int64),
        outer_fold=np.asarray(-1 if q.outer_fold is None else int(q.outer_fold), dtype=np.int64),
        held_out_file_name=np.asarray("" if q.held_out_file_name is None else str(q.held_out_file_name)),
        thigh_pitch_deg=(
            np.asarray(q.thigh_pitch_deg, dtype=np.float32)
            if q.thigh_pitch_deg is not None
            else np.zeros((0,), dtype=np.float32)
        ),
    )

    feature_mode = str(cfg.match_feature_mode).strip().lower()
    thq = resample_quat_slerp_wxyz(q.thigh_quat_wxyz, src_hz=float(q.sample_hz), dst_hz=float(match_hz))
    kn_inc = resample_linear(q.knee_included_deg, src_hz=float(q.sample_hz), dst_hz=float(match_hz))
    kn_flex = (180.0 - np.asarray(kn_inc, dtype=np.float32)).astype(np.float32)
    th_deg = None
    if q.thigh_pitch_deg is not None:
        th_deg = resample_linear(np.asarray(q.thigh_pitch_deg, dtype=np.float32), src_hz=float(q.sample_hz), dst_hz=float(match_hz))

    if feature_mode == "thigh_knee_d" and th_deg is not None:
        L = int(min(int(th_deg.size), int(kn_flex.size)))
    else:
        L = int(min(int(thq.shape[0]), int(kn_flex.size)))
    if L < 2:
        raise RuntimeError(f"Query window too short after resampling (L={L}).")
    thq = thq[:L]
    kn_flex = kn_flex[:L]
    if th_deg is not None:
        th_deg = np.asarray(th_deg[:L], dtype=np.float32)

    print(f"[mocap_phys_eval] status: motion matching ({q.query_id})  bank={len(bank)}  L={L}")
    candidates = motion_match_one_window(
        bank=bank,
        query_thigh_deg=th_deg,
        query_thigh_quat_wxyz=(None if feature_mode == "thigh_knee_d" else thq),
        query_knee_deg=kn_flex,
        top_k=int(cfg.match_top_k),
        local_refine_radius=int(cfg.match_local_refine_radius),
        feature_mode=feature_mode,
        knee_weight=float(cfg.match_knee_weight),
        thigh_weight=float(cfg.match_thigh_weight),
    )
    if not candidates:
        raise RuntimeError("No motion-match candidates found.")
    cand = candidates[0]
    if cand.snippet_id not in bank_snip_to_i:
        raise RuntimeError(f"Matched snippet_id {cand.snippet_id!r} not found in reference bank index.")
    bi = int(bank_snip_to_i[str(cand.snippet_id)])

    ref_kn_full = np.asarray(bank.knee_deg[bi], dtype=np.float32).reshape(-1)
    ref_kn = ref_kn_full[int(cand.start_step) : int(cand.start_step) + L]
    if feature_mode == "thigh_knee_d":
        if th_deg is None:
            raise RuntimeError("Scalar motion matching requires query thigh pitch.")
        ref_th_full = np.asarray(bank.thigh_pitch_deg[bi], dtype=np.float32).reshape(-1)
        ref_th = ref_th_full[int(cand.start_step) : int(cand.start_step) + L]
        query_th_al = (float(cand.thigh_sign) * th_deg + float(cand.thigh_offset_deg)).astype(np.float32)
        thigh_err_deg = np.abs(query_th_al - ref_th).astype(np.float32)
        query_thq_al = None
        ref_thq = None
    else:
        ref_thq_full = np.asarray(bank.thigh_anat_quat_world_wxyz[bi], dtype=np.float32).reshape(-1, 4)
        ref_thq = ref_thq_full[int(cand.start_step) : int(cand.start_step) + L]
        query_thq_al = _align_query_thigh_quat(query_thq=thq, cand=cand)
        thigh_err_deg = quat_geodesic_deg_wxyz(
            quat_normalize_wxyz(ref_thq),
            quat_normalize_wxyz(query_thq_al),
        ).astype(np.float32)
        ref_th = None
        query_th_al = None
    kn_al = (float(cand.knee_sign) * kn_flex + float(cand.knee_offset_deg)).astype(np.float32)

    mm_plot = plot_motion_match(
        out_path=plots_dir / "motion_match.png",
        sample_hz=float(match_hz),
        ref_thigh_deg=ref_th,
        ref_knee_deg=ref_kn,
        query_thigh_aligned_deg=query_th_al,
        query_knee_aligned_deg=kn_al,
        rmse_thigh_deg=float(cand.rmse_thigh_deg),
        rmse_knee_deg=float(cand.rmse_knee_deg),
        thigh_ori_err_deg=np.asarray(thigh_err_deg, dtype=np.float32),
        title=f"Motion Match  query={q.query_id}  snippet={cand.snippet_id}  start_in_snip={cand.start_step}  L={L}",
    )

    if feature_mode == "thigh_knee_d":
        quat_plot = mm_plot
    else:
        quat_plot = plot_thigh_quat_match(
            out_path=plots_dir / "thigh_quat_match.png",
            sample_hz=float(match_hz),
            ref_thigh_quat_wxyz=np.asarray(ref_thq, dtype=np.float32),
            query_thigh_quat_aligned_wxyz=np.asarray(query_thq_al, dtype=np.float32),
            thigh_ori_err_deg=np.asarray(thigh_err_deg, dtype=np.float32),
            title="Thigh Quaternion Match (ref vs query aligned)",
        )

    good_kn_flex = kn_flex.copy()
    pred_included_200 = None
    pred_flex_200 = None
    control_valid_200 = None
    tst = q.tst_model
    if tst is not None and q.X_raw is not None:
        pred_included_200 = _predict_knee_included_deg_for_window(tst, q.X_raw)
        pred_flex_200 = (180.0 - np.asarray(pred_included_200, dtype=np.float32)).astype(np.float32)
        gt_flex_200 = (180.0 - np.asarray(q.knee_included_deg, dtype=np.float32)).astype(np.float32)
        pred_control_200, control_valid_200 = _align_future_forecast_to_control(
            pred_flex_200,
            gt_flex_200,
            label_shift=int(tst.label_shift),
        )
        good_kn_flex = resample_linear(pred_control_200, src_hz=float(q.sample_hz), dst_hz=float(match_hz))[:L].astype(
            np.float32
        )

    bad_kn_flex = _make_bad_knee_prediction(
        good_kn_flex,
        sample_hz=float(match_hz),
        target_rmse_deg=float(cfg.bad_knee_rmse_deg),
        lowpass_hz=float(cfg.bad_knee_lowpass_hz),
        seed=int(noise_seed),
    )[:L].astype(np.float32)

    pred_bad_rmse = float(_rmse(bad_kn_flex, good_kn_flex))

    pred_vs_gt_rmse = None
    if pred_flex_200 is not None:
        gt_flex_200 = (180.0 - np.asarray(q.knee_included_deg, dtype=np.float32)).astype(np.float32)
        W200 = int(min(pred_flex_200.size, gt_flex_200.size))
        shift200 = int(max(0, int(tst.label_shift) if tst is not None else 0))
        if shift200 <= 0:
            pred_vs_gt_rmse = float(_rmse(pred_flex_200[:W200], gt_flex_200[:W200]))
        elif shift200 < W200:
            pred_vs_gt_rmse = float(_rmse(pred_flex_200[: W200 - shift200], gt_flex_200[shift200:W200]))

    good_target_kn = (float(cand.knee_sign) * good_kn_flex + float(cand.knee_offset_deg)).astype(np.float32)
    bad_target_kn = (float(cand.knee_sign) * bad_kn_flex + float(cand.knee_offset_deg)).astype(np.float32)
    good_target_kn = np.clip(good_target_kn, 0.0, 170.0).astype(np.float32)
    bad_target_kn = np.clip(bad_target_kn, 0.0, 170.0).astype(np.float32)

    clip_id = str(cand.clip_id)
    snip_abs_start = int(np.asarray(bank.start_step[bi]).reshape(()))
    snip_abs_end = int(np.asarray(bank.end_step[bi]).reshape(()))
    eval_start_abs = int(snip_abs_start + int(cand.start_step))
    warmup_steps = int(max(0, int(cand.start_step)))
    if (eval_start_abs + int(L) - 1) > int(snip_abs_end):
        raise RuntimeError("Matched window extends past snippet end; this should never happen.")

    expert_model_path = Path(str(np.asarray(bank.expert_model_path[bi]).reshape(())))
    policy = load_expert_policy(expert_model_path, device=str(cfg.device))
    override = OverrideSpec(knee_actuator=str(cfg.knee_actuator), knee_sign=1.0, knee_offset_deg=0.0)

    # BAD panel is only meaningful for demo/oracle mode (no real model).
    # When a trained model is available, show only REF | PRED (the model's override).
    run_bad = bool(tst is None or q.X_raw is None)

    sim_label = "REF|PRED|BAD" if run_bad else "REF|PRED"
    print(
        f"[mocap_phys_eval] status: sim ({sim_label})  clip={clip_id}  "
        f"start_abs={eval_start_abs} warmup={warmup_steps} L={L}"
    )
    rec_paths = record_compare_rollout(
        out_npz_path=replay_dir / "compare.npz",
        clip_id=str(clip_id),
        start_step=int(eval_start_abs),
        end_step=int(snip_abs_end),
        primary_steps=int(L),
        warmup_steps=int(warmup_steps),
        policy=policy,
        override=override,
        knee_good_query_deg=np.asarray(good_target_kn, dtype=np.float32),
        knee_bad_query_deg=np.asarray(bad_target_kn, dtype=np.float32),
        width=int(cfg.render_width),
        height=int(cfg.render_height),
        camera_id=int(cfg.render_camera_id),
        deterministic_policy=True,
        seed=0,
        run_bad=bool(run_bad),
    )

    with np.load(rec_paths.npz_path, allow_pickle=True) as rr0:
        rr = {str(k): rr0[k] for k in rr0.files}
    dt = float(np.asarray(rr["dt"]).reshape(()))
    knee_ref_actual = np.asarray(rr["knee_ref_actual_deg"], dtype=np.float32)
    knee_good_actual = np.asarray(rr["knee_good_actual_deg"], dtype=np.float32)
    knee_bad_actual = np.asarray(rr["knee_bad_actual_deg"], dtype=np.float32)

    sim_knee_plot = plot_simulation_knee(
        out_path=plots_dir / "simulation_knee.png",
        sample_hz=float(match_hz),
        ref_target_knee_deg=np.asarray(ref_kn, dtype=np.float32),
        good_target_knee_deg=np.asarray(good_target_kn[: knee_good_actual.size], dtype=np.float32),
        bad_target_knee_deg=np.asarray(bad_target_kn[: knee_bad_actual.size], dtype=np.float32),
        ref_actual_knee_deg=knee_ref_actual,
        good_actual_knee_deg=knee_good_actual,
        bad_actual_knee_deg=knee_bad_actual,
        title="Simulation: Right Knee Flexion (target vs actual)",
    )

    bal_plot = plot_balance_traces(
        out_path=plots_dir / "simulation_balance.png",
        sample_hz=float(match_hz),
        xcom_margin_ref_m=np.asarray(rr["balance_xcom_margin_ref_m"], dtype=np.float32),
        xcom_margin_good_m=np.asarray(rr["balance_xcom_margin_good_m"], dtype=np.float32),
        xcom_margin_bad_m=np.asarray(rr["balance_xcom_margin_bad_m"], dtype=np.float32),
        upright_ref=np.asarray(rr["upright_ref"], dtype=np.float32),
        upright_good=np.asarray(rr["upright_good"], dtype=np.float32),
        upright_bad=np.asarray(rr["upright_bad"], dtype=np.float32),
        risk_trace_ref=np.asarray(rr["predicted_fall_risk_trace_ref"], dtype=np.float32),
        risk_trace_good=np.asarray(rr["predicted_fall_risk_trace_good"], dtype=np.float32),
        risk_trace_bad=np.asarray(rr["predicted_fall_risk_trace_bad"], dtype=np.float32),
        balance_loss_step_ref=int(np.asarray(rr["balance_loss_step_ref"]).reshape(())),
        balance_loss_step_good=int(np.asarray(rr["balance_loss_step_good"]).reshape(())),
        balance_loss_step_bad=int(np.asarray(rr["balance_loss_step_bad"]).reshape(())),
        risk_ref=float(np.asarray(rr["predicted_fall_risk_ref"]).reshape(())),
        risk_good=float(np.asarray(rr["predicted_fall_risk_good"]).reshape(())),
        risk_bad=float(np.asarray(rr["predicted_fall_risk_bad"]).reshape(())),
        title="Simulation Balance Signals (COM margin, uprightness) + predicted fall risk",
    )

    ctrl_pol_good = np.asarray(rr.get("ctrl_knee_good_policy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ctrl_app_good = np.asarray(rr.get("ctrl_knee_good_applied", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ctrl_tgt_good = np.asarray(rr.get("ctrl_knee_good_target", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ctrl_pol_bad = np.asarray(rr.get("ctrl_knee_bad_policy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ctrl_app_bad = np.asarray(rr.get("ctrl_knee_bad_applied", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    ctrl_tgt_bad = np.asarray(rr.get("ctrl_knee_bad_target", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

    ctrl_diag = {
        "good": {
            "mean_abs_policy_minus_applied": float(_mean_abs(ctrl_pol_good - ctrl_app_good)),
            "mean_abs_applied_minus_target": float(_mean_abs(ctrl_app_good - ctrl_tgt_good)),
        },
        "bad": {
            "mean_abs_policy_minus_applied": float(_mean_abs(ctrl_pol_bad - ctrl_app_bad)),
            "mean_abs_applied_minus_target": float(_mean_abs(ctrl_app_bad - ctrl_tgt_bad)),
        },
    }

    ref_metrics = {
        "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_ref"]).reshape(())),
        "balance_risk_auc": _risk_auc(np.asarray(rr["predicted_fall_risk_trace_ref"], dtype=np.float32), dt=dt),
        "balance_loss_step": int(np.asarray(rr["balance_loss_step_ref"]).reshape(())),
        "knee_rmse_deg": float(_rmse(ref_kn, knee_ref_actual)),
    }
    pred_metrics = {
        "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_good"]).reshape(())),
        "balance_risk_auc": _risk_auc(np.asarray(rr["predicted_fall_risk_trace_good"], dtype=np.float32), dt=dt),
        "balance_loss_step": int(np.asarray(rr["balance_loss_step_good"]).reshape(())),
        "knee_rmse_deg": float(_rmse(good_target_kn, knee_good_actual)),
    }
    bad_metrics = {
        "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_bad"]).reshape(())),
        "balance_risk_auc": _risk_auc(np.asarray(rr["predicted_fall_risk_trace_bad"], dtype=np.float32), dt=dt),
        "balance_loss_step": int(np.asarray(rr["balance_loss_step_bad"]).reshape(())),
        "knee_rmse_deg": float(_rmse(bad_target_kn, knee_bad_actual)),
    }
    sim_metrics = {
        "ref": ref_metrics,
        "pred": pred_metrics,
        # Compatibility alias for older tooling and older manuscript wording.
        "good": dict(pred_metrics),
        "bad": bad_metrics,
    }

    top_candidates = [
        {
            "snippet_id": str(c.snippet_id),
            "start_step_in_snip": int(c.start_step),
            "score": float(c.score),
            "rmse_knee_deg": float(c.rmse_knee_deg),
            "rms_thigh_ori_err_deg": float(c.rmse_thigh_deg),
        }
        for c in candidates[: min(5, len(candidates))]
    ]

    result = {
        "query_id": str(q.query_id),
        "source": str(q.source),
        "mode": str(query_mode),
        "query_meta": {
            "dataset_index": (None if q.dataset_index is None else int(q.dataset_index)),
            "file_id": (None if q.file_id is None else int(q.file_id)),
            "file_name": (None if q.file_name is None else str(q.file_name)),
            "window_start": (None if q.window_start is None else int(q.window_start)),
            "outer_fold": (None if q.outer_fold is None else int(q.outer_fold)),
            "held_out_file_name": (None if q.held_out_file_name is None else str(q.held_out_file_name)),
        },
        "match": {
            "snippet_id": str(cand.snippet_id),
            "clip_id": str(cand.clip_id),
            "start_step_in_snip": int(cand.start_step),
            "start_step_abs": int(eval_start_abs),
            "L": int(L),
            "rmse_knee_deg": float(cand.rmse_knee_deg),
            "rms_thigh_ori_err_deg": float(cand.rmse_thigh_deg),
            "knee_sign": float(cand.knee_sign),
            "knee_offset_deg": float(cand.knee_offset_deg),
            "thigh_quat_offset_wxyz": [float(x) for x in cand.thigh_quat_offset_wxyz],
            "thigh_quat_conjugated": bool(getattr(cand, "thigh_quat_conjugated", False)),
            "score": float(cand.score),
            "top_candidates": top_candidates,
        },
        "model": {
            "tst_ckpt": (None if tst is None else str(tst.ckpt_path)),
            "pred_is_oracle": bool(tst is None or q.X_raw is None),
            "good_is_oracle": bool(tst is None or q.X_raw is None),
            "label_shift_samples": (0 if tst is None else int(tst.label_shift)),
            "label_shift_ms": (0.0 if tst is None else float(1000.0 * float(tst.label_shift) / float(q.sample_hz))),
            "pred_vs_gt_knee_flex_rmse_deg": pred_vs_gt_rmse,
            "bad_target_rmse_deg": float(cfg.bad_knee_rmse_deg),
            "bad_actual_rmse_deg": float(pred_bad_rmse),
        },
        "override": {
            "knee_actuator": str(cfg.knee_actuator),
            "prosthetic_knee_kp": float(np.asarray(rr.get("prosthetic_knee_kp", float("nan"))).reshape(())),
            "prosthetic_knee_kd": float(np.asarray(rr.get("prosthetic_knee_kd", float("nan"))).reshape(())),
            "prosthetic_knee_force": float(np.asarray(rr.get("prosthetic_knee_force", float("nan"))).reshape(())),
            "ctrl_override_diag": ctrl_diag,
        },
        "sim": sim_metrics,
        "artifacts": {
            "motion_match_plot": str(mm_plot),
            "thigh_quat_plot": str(quat_plot),
            "simulation_knee_plot": str(sim_knee_plot),
            "simulation_balance_plot": str(bal_plot),
            "compare_npz": str(rec_paths.npz_path),
            "compare_gif": str(rec_paths.gif_path),
        },
    }

    write_json(q_dir / "summary.json", result)
    return QueryEvalResult(
        result=result,
        compare_npz=rec_paths.npz_path,
        compare_gif=rec_paths.gif_path,
        motion_match_plot=mm_plot,
        thigh_quat_plot=quat_plot,
    )


def _session_run_dir(cfg: EvalConfig, *, out_root: Path) -> tuple[Path, int, str]:
    runs_root = ensure_dir(out_root / "runs")
    try:
        demo_idx = sum(1 for p in runs_root.iterdir() if p.is_dir())
    except Exception:
        demo_idx = 0
    run_id = now_run_id()
    run_dir = ensure_dir(runs_root / run_id)
    ensure_dir(run_dir / "evals")
    write_json(run_dir / "config.json", dataclass_to_json_dict(cfg))
    return run_dir, int(demo_idx), str(run_id)


def main() -> None:
    cfg = EvalConfig()
    set_global_determinism(seed=0)

    out_root = ensure_dir(cfg.artifacts_dir)
    run_dir, demo_idx, run_id = _session_run_dir(cfg, out_root=out_root)

    models_dir_env = os.environ.get("MOCAPACT_MODELS_DIR")
    print(f"[mocap_phys_eval] status: init (run_id={run_id})")
    print(f"[mocap_phys_eval] config: python={sys.executable}")
    print(f"[mocap_phys_eval] config: MOCAPACT_MODELS_DIR={models_dir_env!r}")
    print(f"[mocap_phys_eval] config: experts_root={str(Path(cfg.experts_root).resolve())}")
    print(f"[mocap_phys_eval] config: artifacts_dir={str(Path(cfg.artifacts_dir).resolve())}")

    # 1) Query windows.
    samples_data = _load_samples_dataset(cfg)
    training_run_dir: Path | None = None
    query_mode = "demo_bvh"
    target_successes = int(max(1, int(getattr(cfg, "demo_n_windows", 1))))
    queries: list[QueryWindow] = []

    if samples_data is not None:
        loaded = _load_tst_fold_models_if_available(
            device=str(cfg.device),
            samples_data=samples_data,
            run_dir_override=cfg.tst_run_dir_override,
            allow_partial_coverage=bool(cfg.allow_partial_coverage),
        )
        if loaded is None:
            raise RuntimeError(
                "Found samples_dataset.npy, but no trained LOFO `_all` run with fold checkpoints was found. "
                "Run `python -m emg_tst.run_experiment` first so the evaluator can follow the paper protocol exactly."
            )
        training_run_dir, sample_to_fold = loaded
        target_successes = int(max(1, int(getattr(cfg, "paper_eval_n_trials", 80))))
        queries = _load_rigtest_query_pool(
            cfg,
            samples_data=samples_data,
            sample_to_fold=sample_to_fold,
            seed=int(getattr(cfg, "paper_eval_seed", 42)),
        )
        query_mode = "rigtest_paper_lofo"
        if len(queries) < target_successes and bool(cfg.allow_partial_coverage):
            target_successes = int(len(queries))
        if len(queries) < target_successes:
            raise RuntimeError(
                f"Paper protocol requires {target_successes} successful held-out trials, but the held-out pool only "
                f"contains {len(queries)} windows. Record/train on more files before running mocap_phys_eval."
            )
        print(
            f"[mocap_phys_eval] status: loaded latest compatible `_all` training run: {training_run_dir}  "
            f"(candidate held-out windows={len(queries)} seed={int(getattr(cfg, 'paper_eval_seed', 42))})"
        )
    else:
        queries = _load_demo_query_windows(
            cfg,
            run_dir=run_dir,
            demo_idx=int(demo_idx),
            n_windows=int(max(1, int(getattr(cfg, "demo_n_windows", 1)))),
        )
        query_mode = "demo_bvh"
        print("[mocap_phys_eval] status: no rigtest samples found; running demo BVH sanity-check mode")

    if not queries:
        raise RuntimeError("No query windows available (neither rigtest held-out pool nor demo BVH).")

    # 2) Expert zoo + reference bank.
    print("[mocap_phys_eval] status: ensuring MoCapAct expert zoo + reference bank")
    bank = _ensure_reference_bank(cfg, out_root=out_root)
    bank_snip_to_i = {str(bank.snippet_id[i]): int(i) for i in range(len(bank))}
    match_hz = float(np.median(np.asarray(bank.sample_hz, dtype=np.float64)))
    print(f"[mocap_phys_eval] status: bank ready (n_snippets={len(bank)}; sample_hz~{match_hz:.2f})")

    # 3) Evaluate windows.
    eval_results: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    last_compare_npz: Path | None = None
    failures_root = ensure_dir(run_dir / "failures")
    tmp_root = ensure_dir(run_dir / "_attempts")

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    items = list(enumerate(queries))
    it = items if tqdm is None else tqdm(items, desc="Eval attempts", unit="win")

    for attempt_i, q in it:
        if len(eval_results) >= target_successes:
            break
        attempt_i = int(attempt_i)
        attempt_dir = ensure_dir(tmp_root / f"{attempt_i:03d}_{q.query_id}")
        success_i = int(len(eval_results))
        final_dir = run_dir / "evals" / f"{success_i:02d}_{q.query_id}"
        try:
            eval_out = _evaluate_query_window(
                q=q,
                q_dir=attempt_dir,
                query_mode=query_mode,
                cfg=cfg,
                bank=bank,
                bank_snip_to_i=bank_snip_to_i,
                match_hz=float(match_hz),
                noise_seed=1234 + int(attempt_i),
            )

            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.move(str(attempt_dir), str(final_dir))

            result = dict(eval_out.result)
            old_prefix = str(attempt_dir)
            new_prefix = str(final_dir)
            for k, v in list(result.get("artifacts", {}).items()):
                result["artifacts"][k] = str(v).replace(old_prefix, new_prefix)
            result["paper_eval"] = {
                "attempt_index": int(attempt_i),
                "retained_index": int(success_i),
                "target_successes": int(target_successes),
                "sampling_seed": (
                    int(getattr(cfg, "paper_eval_seed", 42))
                    if query_mode == "rigtest_paper_lofo"
                    else None
                ),
            }
            write_json(final_dir / "summary.json", result)
            eval_results.append(result)

            compare_npz = Path(str(result["artifacts"]["compare_npz"]))
            compare_gif = Path(str(result["artifacts"]["compare_gif"]))
            motion_match_plot = Path(str(result["artifacts"]["motion_match_plot"]))
            thigh_quat_plot = Path(str(result["artifacts"]["thigh_quat_plot"]))
            last_compare_npz = compare_npz

            try:
                shutil.copyfile(compare_npz, out_root / "latest_compare.npz")
            except Exception:
                pass
            try:
                if compare_gif.exists():
                    shutil.copyfile(compare_gif, out_root / "latest_compare.gif")
            except Exception:
                pass
            try:
                shutil.copyfile(motion_match_plot, out_root / "latest_motion_match.png")
            except Exception:
                pass
            try:
                shutil.copyfile(thigh_quat_plot, out_root / "latest_thigh_quat_match.png")
            except Exception:
                pass
        except Exception as e:
            failure = {
                "attempt_index": int(attempt_i),
                "query_id": str(q.query_id),
                "source": str(q.source),
                "dataset_index": (None if q.dataset_index is None else int(q.dataset_index)),
                "file_name": (None if q.file_name is None else str(q.file_name)),
                "outer_fold": (None if q.outer_fold is None else int(q.outer_fold)),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            failed_attempts.append(failure)
            write_json(failures_root / f"{attempt_i:03d}_{q.query_id}.json", failure)
            try:
                if attempt_dir.exists():
                    shutil.rmtree(attempt_dir)
            except Exception:
                pass
            print(
                f"[mocap_phys_eval] warning: discarded failed trial "
                f"(attempt={attempt_i} query={q.query_id} error={type(e).__name__}: {e})"
            )
            continue

    session = {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "mode": str(query_mode),
        "protocol": {
            "paper_exact": bool(query_mode == "rigtest_paper_lofo"),
            "target_successes": int(target_successes),
            "successful_trials": int(len(eval_results)),
            "attempted_trials": int(min(len(queries), len(eval_results) + len(failed_attempts))),
            "failed_trials": int(len(failed_attempts)),
            "sampling_seed": (
                int(getattr(cfg, "paper_eval_seed", 42))
                if query_mode == "rigtest_paper_lofo"
                else None
            ),
            "training_run_dir": (None if training_run_dir is None else str(training_run_dir)),
            "candidate_pool_size": int(len(queries)),
        },
        "n_windows": int(len(eval_results)),
        "bank_n_snippets": int(len(bank)),
        "bank_sample_hz": float(match_hz),
        "failures": failed_attempts,
        "results": eval_results,
    }
    write_json(run_dir / "summary.json", session)

    print(f"[mocap_phys_eval] run_id={run_id}")
    print(
        f"[mocap_phys_eval] mode={query_mode} windows={len(eval_results)} "
        f"failures={len(failed_attempts)} bank_snippets={len(bank)}"
    )
    print(f"[mocap_phys_eval] wrote: {run_dir / 'summary.json'}")
    print(f"[mocap_phys_eval] latest replay: {out_root / 'latest_compare.npz'}")

    if query_mode == "rigtest_paper_lofo" and len(eval_results) < target_successes:
        raise RuntimeError(
            f"Paper protocol incomplete: obtained {len(eval_results)}/{target_successes} successful trials. "
            f"See failures under {failures_root}."
        )

    # Launch viewer for the last recording in a separate process so it stays open.
    if last_compare_npz is not None:
        try:
            subprocess.Popen([sys.executable, "-m", "mocap_phys_eval.replay", str(last_compare_npz)])
        except Exception:
            pass


if __name__ == "__main__":
    main()
