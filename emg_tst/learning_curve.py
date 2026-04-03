from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from emg_tst.run_experiment import (
    CONTEXT_WINDOW,
    EPOCHS,
    SAVE_DIR,
    SEED,
    SOURCE_WINDOW,
    TRAIN_STRIDE,
    build_fold_windows,
    load_recording_corpus,
    set_seed,
    train_one_fold,
)

OUT_ROOT = Path("artifacts") / "learning_curve"
MAX_OUTER_FOLDS = 8
MIN_TRAIN_MINUTES = 5
MAX_POINTS = 8
TARGET_RMSE_DEG = 10.0


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _minutes_schedule(max_minutes: float) -> list[int]:
    full = int(max(1, round(float(max_minutes))))
    if full <= int(MIN_TRAIN_MINUTES):
        return [full]
    vals = []
    m = int(MIN_TRAIN_MINUTES)
    while m < full and len(vals) < int(MAX_POINTS) - 1:
        vals.append(int(m))
        m *= 2
    vals.append(int(full))
    return sorted(set(vals))


def _add_interval(intervals: list[tuple[int, int]], start: int, end: int) -> tuple[list[tuple[int, int]], int]:
    if end <= start:
        return intervals, 0
    merged: list[tuple[int, int]] = []
    delta = 0
    inserted = False
    s = int(start)
    e = int(end)
    for a, b in intervals:
        if b < s:
            merged.append((int(a), int(b)))
            continue
        if e < a:
            if not inserted:
                merged.append((int(s), int(e)))
                delta += int(e - s)
                inserted = True
            merged.append((int(a), int(b)))
            continue
        s = min(int(s), int(a))
        e = max(int(e), int(b))
        delta -= int(b - a)
    if not inserted:
        merged.append((int(s), int(e)))
        delta += int(e - s)
    return merged, int(delta)


def _prefix_coverage_minutes(windows: np.ndarray, *, sample_hz: float) -> np.ndarray:
    wins = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    if wins.size < 1:
        return np.zeros((0,), dtype=np.float64)
    intervals_by_file: dict[int, list[tuple[int, int]]] = {}
    total_samples = 0
    out = np.zeros((wins.shape[0],), dtype=np.float64)
    for i, (fid, start) in enumerate(wins.tolist()):
        prev = intervals_by_file.get(int(fid), [])
        merged, delta = _add_interval(prev, int(start), int(start) + int(SOURCE_WINDOW))
        intervals_by_file[int(fid)] = merged
        total_samples += int(delta)
        out[i] = float(total_samples) / float(sample_hz) / 60.0
    return out


@dataclass(frozen=True)
class SummaryPoint:
    minutes: int
    actual_minutes: float
    hours: float
    mean_rmse: float
    ci95_rmse: float
    mean_seq_rmse: float
    ci95_seq_rmse: float
    n_folds: int


def _write_csv(path: Path, points: list[SummaryPoint]) -> None:
    lines = [
        "requested_minutes,actual_minutes,hours,mean_rmse,ci95_rmse,mean_seq_rmse,ci95_seq_rmse,n_folds\n"
    ]
    for p in points:
        lines.append(
            f"{p.minutes},{p.actual_minutes:.4f},{p.hours:.4f},{p.mean_rmse:.6f},{p.ci95_rmse:.6f},"
            f"{p.mean_seq_rmse:.6f},{p.ci95_seq_rmse:.6f},{p.n_folds}\n"
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
    plt.errorbar(xs, rmse, yerr=rmse_ci, fmt="-o", linewidth=2.2, capsize=4, label="Last-step RMSE")
    plt.errorbar(xs, seq, yerr=seq_ci, fmt="--o", linewidth=1.6, capsize=4, alpha=0.85, label="Rolling seq RMSE")
    plt.axhline(float(TARGET_RMSE_DEG), color="black", linestyle=":", linewidth=1.2, alpha=0.8, label=f"Target {TARGET_RMSE_DEG:.1f} deg")
    plt.xlabel("Training data (hours)")
    plt.ylabel("RMSE (deg)")
    plt.title(f"Learning Curve ({SOURCE_WINDOW / 200.0:.1f}s source, {CONTEXT_WINDOW / 200.0:.1f}s context)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(SEED)
    corpus = load_recording_corpus()
    n_files = int(len(corpus.file_names))
    if n_files < 3:
        raise SystemExit("Learning curve needs at least 3 files so each outer fold still has train/val/test.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml

            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")

    out_dir = OUT_ROOT / datetime.now().strftime("lc_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_file_ids = list(range(n_files))
    if len(outer_file_ids) > int(MAX_OUTER_FOLDS):
        pick = np.linspace(0, len(outer_file_ids) - 1, num=int(MAX_OUTER_FOLDS), dtype=int)
        outer_file_ids = [outer_file_ids[int(i)] for i in np.unique(pick).tolist()]

    fold_jobs = []
    max_minutes_per_fold = []
    for fold_i, held_out in enumerate(outer_file_ids, start=1):
        train_file_ids = tuple(fid for fid in range(n_files) if fid != int(held_out))
        train_windows, val_windows, test_windows, train_files_final, val_files = build_fold_windows(
            corpus,
            train_file_ids=train_file_ids,
            test_file_ids=(int(held_out),),
            seed=int(SEED + fold_i * 1000),
            train_stride=int(TRAIN_STRIDE),
            eval_stride=1,
        )
        rng = np.random.default_rng(int(SEED + fold_i * 17))
        perm = rng.permutation(train_windows.shape[0])
        train_windows_perm = train_windows[perm]
        prefix_minutes = _prefix_coverage_minutes(train_windows_perm, sample_hz=float(corpus.sample_hz))
        fold_jobs.append((fold_i, int(held_out), train_windows_perm, prefix_minutes, val_windows, test_windows, train_files_final, val_files))
        max_minutes_per_fold.append(float(prefix_minutes[-1]) if prefix_minutes.size > 0 else 0.0)

    schedule = _minutes_schedule(float(np.min(np.asarray(max_minutes_per_fold, dtype=np.float64))))
    print(f"Learning curve schedule (minutes): {schedule}")
    print(f"Outputs: {out_dir}")

    results: dict[str, dict[str, dict]] = {}
    for fold_i, held_out, train_windows_perm, prefix_minutes, val_windows, test_windows, train_files_final, val_files in fold_jobs:
        fold_key = f"fold_{int(fold_i):02d}_test_{int(held_out):03d}"
        results[fold_key] = {}
        for minutes in schedule:
            pos = int(np.searchsorted(prefix_minutes, float(minutes), side="left"))
            pos = max(0, min(int(pos), int(prefix_minutes.size) - 1))
            use_windows = train_windows_perm[: pos + 1]
            actual_minutes = float(prefix_minutes[pos])
            unique_train_files = tuple(sorted(set(int(x) for x in use_windows[:, 0].tolist())))
            print(
                f"\n[LC] fold={fold_key} requested={int(minutes)} min actual={actual_minutes:.2f} "
                f"windows={int(use_windows.shape[0])}"
            )
            metrics = train_one_fold(
                corpus=corpus,
                train_windows=use_windows,
                val_windows=val_windows,
                test_windows=test_windows,
                train_file_ids=unique_train_files,
                val_file_ids=tuple(val_files),
                test_file_ids=(int(held_out),),
                feature_cols=np.arange(corpus.n_features, dtype=np.int64),
                n_emg_selected=int(corpus.n_emg),
                n_imu_selected=int(corpus.n_imu),
                device=device,
                run_dir=None,
                fold_i=int(fold_i),
                label="LEARNING_CURVE",
                quiet=False,
                epochs=int(EPOCHS),
                early_stop_patience=5,
                train_stride=int(TRAIN_STRIDE),
            )
            results[fold_key][str(int(minutes))] = {
                "actual_minutes": float(actual_minutes),
                "test_rmse": float(metrics["test_rmse"]),
                "test_seq_rmse": float(metrics["test_seq_rmse"]),
            }
            _write_json(out_dir / "results.json", results)

    points: list[SummaryPoint] = []
    for minutes in schedule:
        rmses = []
        seqs = []
        actuals = []
        for fold_key, fold_vals in results.items():
            _ = fold_key
            item = fold_vals.get(str(int(minutes)))
            if item is None:
                continue
            rmses.append(float(item["test_rmse"]))
            seqs.append(float(item["test_seq_rmse"]))
            actuals.append(float(item["actual_minutes"]))
        n = int(len(rmses))
        rmse_std = float(np.std(rmses, ddof=1)) if n >= 2 else 0.0
        seq_std = float(np.std(seqs, ddof=1)) if n >= 2 else 0.0
        mean_actual = float(np.mean(actuals)) if actuals else float(minutes)
        points.append(
            SummaryPoint(
                minutes=int(minutes),
                actual_minutes=float(mean_actual),
                hours=float(mean_actual) / 60.0,
                mean_rmse=float(np.mean(rmses)) if rmses else float("nan"),
                ci95_rmse=float(1.96 * rmse_std / math.sqrt(max(n, 1))),
                mean_seq_rmse=float(np.mean(seqs)) if seqs else float("nan"),
                ci95_seq_rmse=float(1.96 * seq_std / math.sqrt(max(n, 1))),
                n_folds=int(n),
            )
        )

    _write_json(out_dir / "summary.json", {"points": [p.__dict__ for p in points]})
    _write_csv(out_dir / "summary.csv", points)
    _plot(points, out_path=out_dir / "learning_curve.png")

    print("\nSummary:")
    for p in points:
        print(
            f"  {p.minutes:>4d} min ({p.hours:>6.2f} h)  "
            f"rmse={p.mean_rmse:6.3f} +/- {p.ci95_rmse:5.3f}  "
            f"seq={p.mean_seq_rmse:6.3f} +/- {p.ci95_seq_rmse:5.3f}  folds={p.n_folds}"
        )

    print(f"\nWrote learning curve report to: {out_dir}")


if __name__ == "__main__":
    main()
