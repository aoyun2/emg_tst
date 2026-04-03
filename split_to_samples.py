import glob
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
from emg_tst.data import load_recording

# ==========================================
# Hard-coded sample builder (NO CLI FLAGS)
# Note: the main transformer trainer now reads raw `data*.npy` recordings
# directly. This script remains important for physical-eval query pools,
# held-out window bookkeeping, and visualization.
# ==========================================
DATA_GLOB = "data*.npy"
OUT_FILE = Path("samples_dataset.npy")

# At ~200Hz effective sample rate (BWT901CL at 200Hz):
#   WINDOW=200 -> 1.0s windows
#   WINDOW=100 -> 0.5s windows
#   WINDOW=40  -> 0.2s windows
WINDOW = 200         # 1 second at 200Hz
LABEL_SHIFT = 0      # samples of lookahead

# Stride for overlapping windows.
#   STRIDE = WINDOW  -> non-overlapping (original behaviour)
#   STRIDE = 1       -> maximum overlap; every timestep starts a new window
#                       (WARNING: with ~800K timesteps this produces ~26 GiB)
#   STRIDE = 10      -> 95% overlap; ~80K windows per 800K timesteps (~2.6 GiB)
#   STRIDE = WINDOW  -> non-overlapping (original behaviour)
#
# LOFO validity: all windows from the same recording file keep the same
# file_id regardless of stride, so Leave-One-File-Out cross-validation is
# unaffected (no leakage between files). Overlap within a file is fine because
# the model sees windows from the same file only in train or only in test, never
# mixed – which is what matters.
STRIDE = 30


def make_windows(X, y, w=WINDOW, stride=STRIDE, label_shift=LABEL_SHIFT):
    """
    Sliding-window partition with configurable stride.

    When stride < w the windows overlap.  All windows from the same recording
    file receive the same file_id, preserving LOFO integrity.

    Returns:
      Xs:    (N, W, F)   input windows
      ys:    (N,)         scalar label (last timestep of window + shift)
      y_seq: (N, W)       full angle trajectory per window
      starts:(N,)         start index of each window in the source recording
    """
    T = int(X.shape[0])
    F = int(X.shape[1])
    stride = int(stride)
    w      = int(w)

    # Number of full windows that fit
    n = max(0, (T - w) // stride + 1)
    if n <= 0:
        return (np.empty((0, w, F), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, w), dtype=np.float32),
                np.empty((0,), dtype=np.int32))

    starts = np.arange(n, dtype=np.int32) * stride

    Xs    = np.empty((n, w, F), dtype=np.float32)
    ys    = np.empty((n,),      dtype=np.float32)
    y_seq = np.empty((n, w),    dtype=np.float32)

    for i, t in enumerate(starts):
        Xs[i] = X[t:t + w].astype(np.float32)
        ys[i] = y[min(int(t + w - 1 + label_shift), T - 1)]
        for j in range(w):
            y_seq[i, j] = y[min(int(t + j + label_shift), T - 1)]

    return Xs, ys, y_seq, starts


def main():
    paths = sorted(glob.glob(DATA_GLOB))
    # Don't accidentally treat the sample dataset as a recording
    paths = [p for p in paths if Path(p).name != OUT_FILE.name and "samples_" not in Path(p).name]

    if not paths:
        raise SystemExit(f"No recordings found matching {DATA_GLOB}. (Expected data0.npy, data1.npy, ...)")

    file_names = [Path(p).name for p in paths]

    all_X, all_y, all_y_seq, all_file_id, all_start = [], [], [], [], []
    first_meta = None

    for file_id, p in enumerate(tqdm(paths, desc="Processing recordings", unit="file")):
        X, y, meta = load_recording(Path(p))
        if first_meta is None:
            first_meta = dict(meta)
        else:
            # Make mismatches obvious (instead of failing later in np.concatenate).
            if int(meta.get("n_features", -1)) != int(first_meta.get("n_features", -2)):
                raise RuntimeError(
                    f"Feature count mismatch: {Path(p).name} has F={int(meta.get('n_features', -1))}, "
                    f"but earlier files had F={int(first_meta.get('n_features', -1))}. "
                    "Rebuild recordings so they all have the same feature layout."
                )
            if str(meta.get("thigh_mode", "")) != str(first_meta.get("thigh_mode", "")) or int(meta.get("thigh_n_features", 0)) != int(first_meta.get("thigh_n_features", 0)):
                raise RuntimeError(
                    f"Thigh feature mismatch: {Path(p).name} has thigh_mode={meta.get('thigh_mode')!r}, "
                    f"thigh_n_features={int(meta.get('thigh_n_features', 0))}, but earlier files had "
                    f"thigh_mode={first_meta.get('thigh_mode')!r}, thigh_n_features={int(first_meta.get('thigh_n_features', 0))}."
                )
            if str(meta.get("emg_feature_mode", "")) != str(first_meta.get("emg_feature_mode", "")) or int(meta.get("n_emg_features_per_sensor", 0)) != int(first_meta.get("n_emg_features_per_sensor", 0)):
                raise RuntimeError(
                    f"EMG feature mismatch: {Path(p).name} has emg_feature_mode={meta.get('emg_feature_mode')!r}, "
                    f"n_emg_features_per_sensor={int(meta.get('n_emg_features_per_sensor', 0))}, but earlier files had "
                    f"emg_feature_mode={first_meta.get('emg_feature_mode')!r}, "
                    f"n_emg_features_per_sensor={int(first_meta.get('n_emg_features_per_sensor', 0))}."
                )

        eff = float(meta.get("effective_hz", 0.0))
        orig = float(meta.get("orig_hz", eff))
        if bool(meta.get("resampled", False)):
            hz_msg = f"{orig:.1f} -> {eff:.1f} Hz"
        else:
            hz_msg = f"~{eff:.1f} Hz"
        emg_mode = str(meta.get("emg_feature_mode", "unknown"))
        emg_per_sensor = int(meta.get("n_emg_features_per_sensor", 0))
        tqdm.write(
            f"{Path(p).name}: T={X.shape[0]} features={X.shape[1]} "
            f"(emg_mode={emg_mode} per_sensor={emg_per_sensor}) {hz_msg}"
        )

        Xs, ys, y_seq, starts = make_windows(X, y)
        all_X.append(Xs)
        all_y.append(ys)
        all_y_seq.append(y_seq)
        all_file_id.append(np.full((Xs.shape[0],), file_id, dtype=np.int32))
        all_start.append(starts)
        overlap_pct = int(100 * (1 - STRIDE / WINDOW))
        print(f"  -> {Xs.shape[0]} windows (window={WINDOW}, stride={STRIDE}, {overlap_pct}% overlap)")

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    y_seq_out = np.concatenate(all_y_seq, axis=0)
    file_id_out = np.concatenate(all_file_id, axis=0)
    start_out = np.concatenate(all_start, axis=0)

    assert first_meta is not None
    dataset = {
        "X": X_out,                       # (N, WINDOW, F): emg + omega + thigh quat
        "y": y_out,                       # (N,)
        "y_seq": y_seq_out,               # (N, WINDOW)
        "file_id": file_id_out,           # (N,)
        "start": start_out,               # (N,)
        "file_names": np.array(file_names),
        "window": np.int32(WINDOW),
        "label_shift": np.int32(LABEL_SHIFT),
        "mode": "overlap" if STRIDE < WINDOW else "nonoverlap",
        "stride": np.int32(STRIDE),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Dataset sample rate after resampling (used for TST + physics eval).
        "sample_hz": np.float32(first_meta.get("effective_hz", 0.0)),
        "orig_hz": np.float32(first_meta.get("orig_hz", first_meta.get("effective_hz", 0.0))),
        # Feature layout metadata (from first file)
        "n_channels": np.int32(first_meta["n_channels"]),
        "n_raw_features": np.int32(first_meta.get("n_raw_features", 0)),
        "has_raw_emg": bool(first_meta.get("has_raw_emg", False)),
        "emg_feature_mode": str(first_meta.get("emg_feature_mode", "unknown")),
        "n_emg_features_per_sensor": np.int32(first_meta.get("n_emg_features_per_sensor", 0)),
        "raw_window_samples": np.int32(first_meta.get("raw_window_samples", 0)),
        "thigh_mode": str(first_meta.get("thigh_mode", "unknown")),
        "thigh_n_features": np.int32(first_meta.get("thigh_n_features", 1)),
        "n_angular_velocity_features": np.int32(first_meta.get("n_angular_velocity_features", 0)),
    }

    np.save(OUT_FILE, dataset, allow_pickle=True)
    print(f"\nWrote: {OUT_FILE}  X={X_out.shape}  y={y_out.shape}  y_seq={y_seq_out.shape}")

if __name__ == "__main__":
    main()
