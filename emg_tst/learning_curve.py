from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from emg_tst.run_experiment import (
    SEED,
    K_FOLDS,
    lofo_indices,
    kfold_indices,
    train_one_fold,
    set_seed,
)

# ==========================================
# Hard-coded learning curve config (NO FLAGS)
# ==========================================
SAMPLES_FILE = Path("samples_dataset.npy")
OUT_ROOT = Path("artifacts") / "learning_curve"

# For runtime: learning curves do not need every outer fold. Increase for a more precise estimate.
MAX_OUTER_FOLDS = 8

# Train sizes are in minutes of non-overlapping 1.0s windows (see split_to_samples.py).
MIN_TRAIN_MINUTES = 5
MAX_POINTS = 8  # including "full"

# Keep this <= run_experiment.py settings unless you explicitly want a faster, less-accurate curve.
EPOCHS_PRETRAIN = 20
EPOCHS_FINETUNE = 20
EARLY_STOP_PATIENCE = 5

TARGET_RMSE_DEG = 3.0


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def _dataset_signature(data: dict) -> dict:
    X = data["X"]
    y = data["y"]
    y_seq = data.get("y_seq", None)
    file_id = data.get("file_id", None)
    file_names = data.get("file_names", np.array([]))
    return {
        "n_samples": int(X.shape[0]),
        "window": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "has_y_seq": bool(y_seq is not None),
        "n_files": int(len(np.unique(file_id))) if file_id is not None else 1,
        "file_names": [str(x) for x in list(file_names)],
        "label_shift": int(data.get("label_shift", 0)),
        "sample_hz": float(data.get("sample_hz", 0.0)),
        "thigh_mode": str(data.get("thigh_mode", "")),
        "thigh_n_features": int(data.get("thigh_n_features", 0)),
        "y_shape": [int(x) for x in np.asarray(y).shape],
    }


def _select_outer_folds(file_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a subset of outer CV folds for the learning curve."""
    folds = lofo_indices(file_ids)
    if len(folds) <= MAX_OUTER_FOLDS:
        return folds

    # Evenly spaced selection over file ids for a representative estimate.
    idxs = np.linspace(0, len(folds) - 1, num=MAX_OUTER_FOLDS, dtype=int)
    idxs = np.unique(idxs)
    return [folds[int(i)] for i in idxs]


def _split_train_val_by_file(train_idx: np.ndarray, file_ids: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one entire file_id for validation (fallback: window split)."""
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    if train_idx.size < 2:
        return train_idx[:1], train_idx[:0]

    rng = np.random.default_rng(int(seed))
    train_files = np.unique(file_ids[train_idx])

    if train_files.size >= 2:
        val_file = int(rng.choice(train_files))
        val_mask = file_ids[train_idx] == val_file
        val_idx = train_idx[val_mask]
        train_idx2 = train_idx[~val_mask]
    else:
        perm = rng.permutation(train_idx)
        n_val = max(1, int(round(0.1 * perm.size)))
        val_idx = perm[:n_val]
        train_idx2 = perm[n_val:]

    if train_idx2.size < 1:
        perm = rng.permutation(train_idx)
        val_idx = perm[:1]
        train_idx2 = perm[1:]

    return train_idx2.astype(np.int64), val_idx.astype(np.int64)


def _minutes_schedule(max_minutes: float) -> list[int]:
    """Log-spaced-ish minutes schedule ending at full size."""
    if max_minutes <= 0:
        return [0]

    full = int(max(1, round(max_minutes)))
    mins = []
    m = int(MIN_TRAIN_MINUTES)
    if full < m:
        return [full]

    while m < full and len(mins) < max(0, int(MAX_POINTS) - 1):
        mins.append(int(m))
        m *= 2

    mins.append(full)
    mins = sorted(set(int(x) for x in mins if x >= 1))
    return mins


@dataclass(frozen=True)
class SummaryPoint:
    minutes: int
    hours: float
    n_train_windows: int
    mean_rmse: float
    ci95_rmse: float
    mean_seq_rmse: float
    ci95_seq_rmse: float
    n_folds: int


def _summarize(results: dict, *, window_sec: float) -> list[SummaryPoint]:
    by_minutes = results.get("folds", {})
    mins_all = sorted({int(m) for fold in by_minutes.values() for m in fold.get("train_sizes", {}).keys()})

    out: list[SummaryPoint] = []
    for minutes in mins_all:
        rmses = []
        seq_rmses = []
        n_train_windows = None
        for fold_key, fold in by_minutes.items():
            _ = fold_key
            metrics = fold.get("train_sizes", {}).get(str(minutes), None)
            if metrics is None:
                continue
            rmses.append(float(metrics["test_rmse"]))
            seq_rmses.append(float(metrics["test_seq_rmse"]))
            if n_train_windows is None:
                n_train_windows = int(metrics.get("n_train", 0))

        if not rmses:
            continue

        rmse = float(np.mean(rmses))
        seq = float(np.mean(seq_rmses))
        rmse_std = float(np.std(rmses, ddof=1)) if len(rmses) >= 2 else 0.0
        seq_std = float(np.std(seq_rmses, ddof=1)) if len(seq_rmses) >= 2 else 0.0
        n = int(len(rmses))
        ci_rmse = float(1.96 * rmse_std / math.sqrt(max(n, 1)))
        ci_seq = float(1.96 * seq_std / math.sqrt(max(n, 1)))
        hours = float(minutes) / 60.0
        n_train_windows = int(n_train_windows if n_train_windows is not None else int(round((minutes * 60.0) / max(window_sec, 1e-6))))

        out.append(SummaryPoint(
            minutes=int(minutes),
            hours=float(hours),
            n_train_windows=int(n_train_windows),
            mean_rmse=rmse,
            ci95_rmse=ci_rmse,
            mean_seq_rmse=seq,
            ci95_seq_rmse=ci_seq,
            n_folds=n,
        ))

    return out


def _write_csv(path: Path, points: list[SummaryPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("minutes,hours,n_train_windows,mean_rmse,ci95_rmse,mean_seq_rmse,ci95_seq_rmse,n_folds\n")
    for p in points:
        lines.append(
            f"{p.minutes},{p.hours:.4f},{p.n_train_windows},{p.mean_rmse:.6f},{p.ci95_rmse:.6f},{p.mean_seq_rmse:.6f},{p.ci95_seq_rmse:.6f},{p.n_folds}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _plot(points: list[SummaryPoint], *, out_path: Path) -> None:
    if not points:
        return

    xs = np.array([p.hours for p in points], dtype=np.float64)
    rmse = np.array([p.mean_rmse for p in points], dtype=np.float64)
    rmse_ci = np.array([p.ci95_rmse for p in points], dtype=np.float64)
    seq = np.array([p.mean_seq_rmse for p in points], dtype=np.float64)
    seq_ci = np.array([p.ci95_seq_rmse for p in points], dtype=np.float64)

    plt.figure(figsize=(9.5, 5.2))
    plt.errorbar(xs, rmse, yerr=rmse_ci, fmt="-o", linewidth=2.2, capsize=4, label="Predicted RMSE (last timestep)")
    plt.errorbar(xs, seq, yerr=seq_ci, fmt="--o", linewidth=1.6, capsize=4, alpha=0.85, label="Seq RMSE (full 1.0s window)")
    plt.axhline(float(TARGET_RMSE_DEG), color="black", linestyle=":", linewidth=1.2, alpha=0.8, label=f"Target {TARGET_RMSE_DEG:.1f} deg")

    plt.xlabel("Training data (hours)")
    plt.ylabel("RMSE (deg)")
    plt.title("Learning Curve (Outer CV by Recording File)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(SEED)

    if not SAMPLES_FILE.exists():
        raise SystemExit(f"Missing {SAMPLES_FILE}. Run: python split_to_samples.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(SAMPLES_FILE, allow_pickle=True).item()

    # Enforce quaternion thigh features (no scalar thigh_angle mode).
    thigh_mode = str(data.get("thigh_mode", ""))
    thigh_n = int(data.get("thigh_n_features", 0))
    if thigh_mode != "quat" or thigh_n != 4:
        raise SystemExit(
            "This pipeline requires thigh quaternion features (thigh_quat_wxyz, 4 dims). "
            f"Got thigh_mode={thigh_mode!r}, thigh_n_features={thigh_n}."
        )

    X_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.float32)
    y_seq_all = data["y_seq"].astype(np.float32)
    file_ids = data["file_id"].astype(np.int32)
    file_names = data.get("file_names", np.array([]))

    n_all = int(X_all.shape[0])
    window = int(X_all.shape[1])
    n_vars = int(X_all.shape[2])
    unique_files = np.unique(file_ids)
    n_files = int(unique_files.size)
    sample_hz = float(data.get("sample_hz", 0.0)) or 200.0
    window_sec = float(window) / float(sample_hz)

    print(f"Learning curve dataset: N={n_all} windows, window={window} ({window_sec:.3f}s), features={n_vars}, files={n_files}")

    # Outer folds.
    if n_files >= 2:
        outer_folds = _select_outer_folds(file_ids)
        print(f"Outer CV: LOFO by file_id (using {len(outer_folds)}/{n_files} folds for the curve)")
    else:
        outer_folds = kfold_indices(n_all, K_FOLDS, seed=SEED)
        print(f"Outer CV: random {K_FOLDS}-fold (single file; LOFO not possible)")

    # Create or resume an output run dir (only resumes if the dataset signature matches).
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sig = _dataset_signature(data)
    latest_path = OUT_ROOT / "_latest.json"

    run_dir = None
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            prev_sig = latest.get("dataset_sig", None)
            prev_dir = Path(str(latest.get("run_dir", "")))
            if prev_sig == sig and prev_dir.exists():
                run_dir = prev_dir
        except Exception:
            run_dir = None

    if run_dir is None:
        run_dir = OUT_ROOT / datetime.now().strftime("lc_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(latest_path, {"run_dir": str(run_dir), "dataset_sig": sig, "created_at": datetime.now().isoformat(timespec="seconds")})

    results_path = run_dir / "results.json"
    config_path = run_dir / "config.json"

    results = {}
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    if not results:
        results = {
            "dataset_sig": sig,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "max_outer_folds": int(MAX_OUTER_FOLDS),
                "min_train_minutes": int(MIN_TRAIN_MINUTES),
                "max_points": int(MAX_POINTS),
                "epochs_pretrain": int(EPOCHS_PRETRAIN),
                "epochs_finetune": int(EPOCHS_FINETUNE),
                "early_stop_patience": int(EARLY_STOP_PATIENCE),
                "target_rmse_deg": float(TARGET_RMSE_DEG),
            },
            "folds": {},
        }

    _write_json_atomic(config_path, results.get("config", {}))

    # Plan schedule based on the smallest training split among the selected folds (so all points are comparable).
    train_minutes_per_fold = []
    fold_plans = []
    for fold_i, (train_idx_full, test_idx) in enumerate(outer_folds, start=1):
        train_idx2, val_idx = _split_train_val_by_file(train_idx_full, file_ids, seed=SEED + fold_i * 1000)
        max_minutes = float(train_idx2.size) * float(window_sec) / 60.0
        train_minutes_per_fold.append(max_minutes)
        test_file_id = int(file_ids[int(test_idx[0])]) if test_idx.size > 0 else -1
        fold_plans.append((fold_i, train_idx2, val_idx, test_idx, test_file_id))

    max_minutes_common = float(np.min(train_minutes_per_fold)) if train_minutes_per_fold else 0.0
    mins_schedule = _minutes_schedule(max_minutes_common)
    if not mins_schedule:
        raise SystemExit("Not enough data to build a learning curve (need at least a couple windows).")

    print(f"Train sizes (minutes): {mins_schedule}")
    print(f"Outputs: {run_dir}")

    # All features.
    feature_cols = np.arange(n_vars, dtype=np.int64)

    # Run jobs.
    n_jobs = int(len(fold_plans) * len(mins_schedule))
    job_i = 0
    for fold_i, train_idx2_full, val_idx, test_idx, test_file_id in fold_plans:
        test_name = None
        if 0 <= test_file_id < len(file_names):
            test_name = str(file_names[test_file_id])

        fold_key = f"fold_{fold_i:02d}_test_{test_file_id:03d}"
        fold_rec = results["folds"].setdefault(fold_key, {
            "fold": int(fold_i),
            "test_file_id": int(test_file_id),
            "test_file_name": test_name,
            "n_test": int(test_idx.size),
            "train_sizes": {},
        })

        rng = np.random.default_rng(SEED + fold_i * 17)
        perm = rng.permutation(train_idx2_full)

        for minutes in mins_schedule:
            minutes = int(minutes)
            if str(minutes) in fold_rec["train_sizes"]:
                continue

            n_train = int(round((float(minutes) * 60.0) / max(window_sec, 1e-6)))
            n_train = max(1, min(int(n_train), int(perm.size)))
            train_idx2 = perm[:n_train]

            job_i += 1
            print(f"\n[Job {job_i}/{n_jobs}] fold={fold_key} train={minutes} min ({n_train} windows)")

            metrics = train_one_fold(
                X_all=X_all,
                y_all=y_all,
                y_seq_all=y_seq_all,
                file_ids=file_ids,
                train_idx=train_idx2,
                val_idx=val_idx,
                test_idx=test_idx,
                feature_cols=feature_cols,
                device=device,
                run_dir=None,  # do not write checkpoints for learning curves
                fold_i=int(fold_i),
                label="LEARNING_CURVE",
                data_meta=data,
                quiet=False,
                early_stop_patience=int(EARLY_STOP_PATIENCE),
                epochs_pretrain=int(EPOCHS_PRETRAIN),
                epochs_finetune=int(EPOCHS_FINETUNE),
            )

            fold_rec["train_sizes"][str(minutes)] = {
                "minutes": int(minutes),
                "n_train": int(metrics["n_train"]),
                "n_val": int(metrics["n_val"]),
                "n_test": int(metrics["n_test"]),
                "best_val_rmse": float(metrics["best_val_rmse"]),
                "test_rmse": float(metrics["test_rmse"]),
                "test_seq_rmse": float(metrics["test_seq_rmse"]),
                "test_mae": float(metrics["test_mae"]),
                "best_epoch": int(metrics["best_epoch"]),
            }

            _write_json_atomic(results_path, results)

    # Summarize + plot.
    points = _summarize(results, window_sec=window_sec)
    _write_csv(run_dir / "summary.csv", points)
    _write_json_atomic(run_dir / "summary.json", {"points": [p.__dict__ for p in points]})
    _plot(points, out_path=run_dir / "learning_curve.png")

    print("\nSummary (mean test RMSE across outer folds; error=95% CI):")
    for p in points:
        print(f"  {p.minutes:>4d} min  ({p.hours:>6.2f} h)  rmse={p.mean_rmse:6.3f} +/- {p.ci95_rmse:5.3f}  seq={p.mean_seq_rmse:6.3f} +/- {p.ci95_seq_rmse:5.3f}  folds={p.n_folds}")

    # Simple threshold check.
    first_ok = next((p for p in points if p.mean_rmse <= float(TARGET_RMSE_DEG)), None)
    if first_ok is None:
        print(f"\nTarget not reached: mean RMSE never <= {TARGET_RMSE_DEG:.1f} deg on this dataset.")
    else:
        print(f"\nEstimated data to reach <= {TARGET_RMSE_DEG:.1f} deg (mean): ~{first_ok.hours:.2f} hours (at {first_ok.minutes} minutes point).")

    print(f"\nWrote learning curve report to: {run_dir}")


if __name__ == "__main__":
    main()

