import glob
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
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
WINDOW = 200         # 1 second at 200Hz
LABEL_SHIFT = 0      # samples of lookahead


def make_nonoverlapping_windows(X, y, w=WINDOW, label_shift=LABEL_SHIFT):
    """Partition into consecutive, non-overlapping windows. Drops remainder.

    Returns:
      Xs:    (N, W, F)   input windows (EMG features + thigh orientation features)
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

        eff = float(meta.get("effective_hz", 0.0))
        orig = float(meta.get("orig_hz", eff))
        if bool(meta.get("resampled", False)):
            hz_msg = f"{orig:.1f} -> {eff:.1f} Hz"
        else:
            hz_msg = f"~{eff:.1f} Hz"
        tqdm.write(
            f"{Path(p).name}: T={X.shape[0]} features={X.shape[1]} "
            f"({meta['n_channels']} bins/sensor) {hz_msg}"
        )

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

    assert first_meta is not None
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
        # Dataset sample rate after resampling (used for TST + physics eval).
        "sample_hz": np.float32(first_meta.get("effective_hz", 0.0)),
        "orig_hz": np.float32(first_meta.get("orig_hz", first_meta.get("effective_hz", 0.0))),
        # Feature layout metadata (from first file)
        "n_channels": np.int32(first_meta["n_channels"]),
        "n_raw_features": np.int32(first_meta.get("n_raw_features", 0)),
        "has_raw_emg": bool(first_meta.get("has_raw_emg", False)),
        "thigh_mode": str(first_meta.get("thigh_mode", "unknown")),
        "thigh_n_features": np.int32(first_meta.get("thigh_n_features", 1)),
    }

    np.save(OUT_FILE, dataset, allow_pickle=True)
    print(f"\nWrote: {OUT_FILE}  X={X_out.shape}  y={y_out.shape}  y_seq={y_seq_out.shape}")

if __name__ == "__main__":
    main()
