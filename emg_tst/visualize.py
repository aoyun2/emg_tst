from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from emg_tst.model import TSTEncoder, TSTRegressor
from emg_tst.data import N_FFT_BANDS

# ==========================================
# Hard-coded visualization config (NO FLAGS)
# ==========================================
SAMPLES_FILE = Path("samples_dataset.npy")
# Point this to a saved fold checkpoint you want to inspect:
CKPT_PATH = Path("checkpoints")  # <-- edit to e.g. checkpoints/tst_YYYYMMDD_HHMMSS/fold_01/reg_best.pt

# Which sample indices to plot:
INDICES = [0, 10, 100]
OUT_DIR = Path("viz_outputs")

# Raw feature names (per sensor): 5 time-domain + N_FFT_BANDS spectral
RAW_FEAT_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC"] + [f"band{b}" for b in range(N_FFT_BANDS)]
N_RAW_PER_SENSOR = len(RAW_FEAT_NAMES)  # 13


def load_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    encoder = TSTEncoder(
        n_vars=int(cfg["n_vars"]),
        seq_len=int(cfg["seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    reg = TSTRegressor(encoder, out_dim=1).to(device)
    reg.load_state_dict(ckpt["reg_state_dict"], strict=True)
    reg.eval()

    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    std = np.asarray(scaler.get("std", None), dtype=np.float32)
    return reg, mean, std


def _parse_feature_layout(data: dict, F: int):
    """Parse feature layout from samples dataset metadata.

    New recordings (has_raw=True):  [raw_features x 3 sensors] + [thigh]
      raw_features = 5 time-domain + N_FFT_BANDS spectral per sensor
      total = (5 + N_FFT_BANDS) * 3 + thigh_n

    Legacy recordings (has_raw=False): [device_spectr x 3 sensors] + [thigh]
      total = n_ch * 3 + thigh_n
    """
    n_ch = int(data.get("n_channels", 16))
    n_raw = int(data.get("n_raw_features", 0))
    has_raw = bool(data.get("has_raw_emg", False))
    thigh_n = int(data.get("thigh_n_features", 1))
    if thigh_n < 1 or thigh_n > F:
        thigh_n = 1

    # Fallback: infer from feature count
    if not has_raw:
        expected_raw = N_RAW_PER_SENSOR * 3
        if F == expected_raw + 1:
            has_raw = True
            n_raw = expected_raw

    thigh_start = int(F - thigh_n)
    thigh_slice = slice(thigh_start, int(F))

    if has_raw:
        return {
            "n_ch": n_ch, "has_raw": True, "has_spectr": False,
            "raw_start": 0, "raw_end": n_raw,
            "spectr_start": None, "spectr_end": None,
            "thigh_slice": thigh_slice,
            "thigh_n": int(thigh_n),
        }
    else:
        spectr_total = n_ch * 3
        return {
            "n_ch": n_ch, "has_raw": False, "has_spectr": True,
            "raw_start": None, "raw_end": None,
            "spectr_start": 0, "spectr_end": spectr_total,
            "thigh_slice": thigh_slice,
            "thigh_n": int(thigh_n),
        }


def _plot_channels(x: np.ndarray, layout: dict, window: int, idx: int, out_dir: Path):
    """Plot all feature groups for one sample window."""
    W, F = x.shape
    n_ch = layout["n_ch"]
    has_raw = layout["has_raw"]
    has_spectr = layout.get("has_spectr", False)

    # Count rows
    n_rows = 1  # thigh always
    if has_spectr:
        n_rows += 3  # 3 spectr sensors
    if has_raw:
        n_rows += 6  # 3 time-domain + 3 spectral-bands

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    ax_i = 0

    # --- Spectral channels per sensor (legacy only) ---
    if has_spectr:
        spectr_start = layout["spectr_start"]
        for s in range(3):
            ax = axes[ax_i]; ax_i += 1
            base = spectr_start + s * n_ch
            for c in range(n_ch):
                col = x[:, base + c]
                if np.abs(col).max() > 1e-6:
                    ax.plot(col, linewidth=0.7, alpha=0.85, label=f"ch{c}")
            ax.set_title(f"Sensor {s+1} - device_spectr ({n_ch} ch)")
            ax.legend(ncol=min(n_ch, 8), fontsize=6, frameon=False)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylabel("z-score")

    # --- Raw EMG features ---
    if has_raw:
        raw_start = layout["raw_start"]

        for s in range(3):
            sensor_offset = raw_start + s * N_RAW_PER_SENSOR

            # Time-domain features: RMS, MAV, WL, ZC, SSC
            ax_td = axes[ax_i]; ax_i += 1
            for f_i in range(5):
                col = x[:, sensor_offset + f_i]
                ax_td.plot(col, linewidth=0.8, alpha=0.85, label=RAW_FEAT_NAMES[f_i])
            ax_td.set_title(f"Sensor {s+1} - Raw Time-Domain (RMS, MAV, WL, ZC, SSC)")
            ax_td.legend(ncol=5, fontsize=7, frameon=False)
            ax_td.grid(True, linestyle="--", alpha=0.4)
            ax_td.set_ylabel("z-score")

            # FFT band features
            ax_fft = axes[ax_i]; ax_i += 1
            for f_i in range(N_FFT_BANDS):
                col = x[:, sensor_offset + 5 + f_i]
                ax_fft.plot(col, linewidth=0.7, alpha=0.85, label=RAW_FEAT_NAMES[5 + f_i])
            ax_fft.set_title(f"Sensor {s+1} - Raw FFT Power ({N_FFT_BANDS} bands)")
            ax_fft.legend(ncol=min(N_FFT_BANDS, 8), fontsize=6, frameon=False)
            ax_fft.grid(True, linestyle="--", alpha=0.4)
            ax_fft.set_ylabel("z-score")

    # --- Thigh angle ---
    ax_th = axes[ax_i]
    th = x[:, layout["thigh_slice"]]
    if int(layout.get("thigh_n", 1)) == 4 and th.shape[1] == 4:
        ax_th.plot(th[:, 0], linewidth=0.9, label="thigh_q_w")
        ax_th.plot(th[:, 1], linewidth=0.9, label="thigh_q_x")
        ax_th.plot(th[:, 2], linewidth=0.9, label="thigh_q_y")
        ax_th.plot(th[:, 3], linewidth=0.9, label="thigh_q_z")
        ax_th.set_title("Thigh Quaternion (wxyz) - uMyo sensor 2")
    else:
        ax_th.plot(th[:, 0], linewidth=1.0, color="orange", label="thigh angle")
        ax_th.set_title("Thigh Angle - uMyo sensor 2")
    ax_th.legend(fontsize=8, frameon=False)
    ax_th.grid(True, linestyle="--", alpha=0.4)
    ax_th.set_ylabel("z-score")
    ax_th.set_xlabel(f"t (samples, window={window})")

    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{idx:06d}_channels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SAMPLES_FILE.exists():
        raise SystemExit(f"Missing {SAMPLES_FILE}. Run: python split_to_samples.py")

    # Find an actual checkpoint file if CKPT_PATH is a directory
    ckpt_file = CKPT_PATH
    if ckpt_file.is_dir():
        candidates = sorted(ckpt_file.glob("**/reg_best.pt"))
        if not candidates:
            raise SystemExit("No reg_best.pt found under checkpoints/. Edit CKPT_PATH.")
        ckpt_file = candidates[-1]

    reg, mean, std = load_ckpt(ckpt_file, device=device)

    data = np.load(SAMPLES_FILE, allow_pickle=True).item()
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    y_seq = data.get("y_seq", None)
    if y_seq is not None:
        y_seq = y_seq.astype(np.float32)

    window = int(data.get("window", X.shape[1]))
    F = X.shape[2]
    layout = _parse_feature_layout(data, F)

    ts = layout["thigh_slice"]
    ts_label = f"cols{int(ts.start)}..{int(ts.stop) - 1}"
    print(f"Feature layout: {F} total - spectr={layout['n_ch']}x3  "
          f"raw={'yes ('+str(layout['raw_end']-layout['raw_start'])+')' if layout['has_raw'] else 'no'}  "
          f"thigh={ts_label}")

    # normalize
    Xn = (X - mean[None, None, :]) / std[None, None, :]

    for idx in INDICES:
        if idx < 0 or idx >= Xn.shape[0]:
            continue
        x = Xn[idx]  # (W,F)
        xb = torch.from_numpy(x[None, :, :]).to(device)
        with torch.no_grad():
            out = reg(xb)[0, :, 0].cpu().numpy()

        pred_last = float(out[-1])
        true_label = float(y[idx])

        # Plot: pred seq vs true trajectory
        fig = plt.figure(figsize=(14, 5))
        plt.plot(out, label="pred_angle_seq", linewidth=1.5)
        if y_seq is not None:
            plt.plot(y_seq[idx], label="true_angle_seq", linewidth=1.5, linestyle="--")
        else:
            plt.axhline(true_label, linestyle="--", linewidth=1.0, label="true_label")
        plt.title(f"sample {idx}: pred_last={pred_last:.3f} true={true_label:.3f}")
        plt.xlabel(f"t (samples, window={window})")
        plt.ylabel("knee angle (degrees)")
        plt.legend()
        fig.savefig(OUT_DIR / f"sample_{idx:06d}_pred.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Plot: all feature channels
        _plot_channels(x, layout, window, idx, OUT_DIR)

    print("Wrote images to:", OUT_DIR)
    print("Used checkpoint:", ckpt_file)

if __name__ == "__main__":
    main()
