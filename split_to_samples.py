import glob
from pathlib import Path
import numpy as np
from datetime import datetime
from emg_tst.data import load_recording

# ==========================================
# Hard-coded sample builder (NO CLI FLAGS)
# ==========================================
DATA_GLOB = "data*.npy"
OUT_FILE = Path("samples_dataset.npy")

# At ~200Hz effective sample rate (BWT901CL at 200Hz):
#   WINDOW=200 -> 1.0s windows
#   WINDOW=100 -> 0.5s windows
#   WINDOW=40  -> 0.2s windows
WINDOW = 200         # ← 1 second at 200Hz
LABEL_SHIFT = 0      # samples of lookahead


def make_nonoverlapping_windows(X, y, w=WINDOW, label_shift=LABEL_SHIFT):
    """Partition into consecutive, non-overlapping windows. Drops remainder.

    Returns:
      Xs:    (N, W, F)   input windows (13 features: 12 EMG + 1 thigh angle)
      ys:    (N,)         scalar label (last timestep + shift)
      y_seq: (N, W)       full angle trajectory per window (with shift)
      starts:(N,)         start indices
    """
    T = int(X.shape[0])
    F = int(X.shape[1])
    n = T // int(w)
    if n <= 0:
        return (np.empty((0, w, F), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, w), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    starts = (np.arange(n, dtype=np.int32) * int(w))

    Xs_list = []
    ys_list = []
    y_seq_list = []
    for t in starts:
        Xs_list.append(X[t:t+w].astype(np.float32))

        # Labels
        ys_list.append(y[min(int(t + w - 1 + label_shift), T - 1)])
        seq = np.empty(w, dtype=np.float32)
        for i in range(w):
            seq[i] = y[min(int(t + i + label_shift), T - 1)]
        y_seq_list.append(seq)

    Xs = np.stack(Xs_list)
    ys = np.array(ys_list, dtype=np.float32)
    y_seq = np.stack(y_seq_list)
    return Xs, ys, y_seq, starts


def main():
    paths = sorted(glob.glob(DATA_GLOB))
    # Don't accidentally treat the sample dataset as a recording
    paths = [p for p in paths if Path(p).name != OUT_FILE.name and "samples_" not in Path(p).name]

    if not paths:
        raise SystemExit(f"No recordings found matching {DATA_GLOB}. (Expected data0.npy, data1.npy, ...)")

    file_names = [Path(p).name for p in paths]

    all_X, all_y, all_y_seq, all_file_id, all_start = [], [], [], [], []

    for file_id, p in enumerate(paths):
        X, y, meta = load_recording(Path(p))
        print(f"{Path(p).name}: T={X.shape[0]} features={X.shape[1]} "
              f"({meta['n_channels']} bins/sensor) ~{meta['effective_hz']:.1f} Hz")

        Xs, ys, y_seq, starts = make_nonoverlapping_windows(X, y)
        all_X.append(Xs)
        all_y.append(ys)
        all_y_seq.append(y_seq)
        all_file_id.append(np.full((Xs.shape[0],), file_id, dtype=np.int32))
        all_start.append(starts)
        print(f"  -> {Xs.shape[0]} non-overlapping windows (window={WINDOW})")

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    y_seq_out = np.concatenate(all_y_seq, axis=0)
    file_id_out = np.concatenate(all_file_id, axis=0)
    start_out = np.concatenate(all_start, axis=0)

    dataset = {
        "X": X_out,                       # (N, WINDOW, F): spectr + raw_features + thigh
        "y": y_out,                       # (N,)
        "y_seq": y_seq_out,               # (N, WINDOW)
        "file_id": file_id_out,           # (N,)
        "start": start_out,               # (N,)
        "file_names": np.array(file_names),
        "window": np.int32(WINDOW),
        "label_shift": np.int32(LABEL_SHIFT),
        "mode": "nonoverlap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Feature layout metadata (from first file)
        "n_channels": np.int32(meta["n_channels"]),
        "n_raw_features": np.int32(meta.get("n_raw_features", 0)),
        "has_raw_emg": bool(meta.get("has_raw_emg", False)),
    }

    np.save(OUT_FILE, dataset, allow_pickle=True)
    print(f"\nWrote: {OUT_FILE}  X={X_out.shape}  y={y_out.shape}  y_seq={y_seq_out.shape}")

if __name__ == "__main__":
    main()
