from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from emg_tst.data import emg_feature_layout_from_meta
from emg_tst.model import SensorFusionTransformer, rolling_last_step_predict

SAMPLES_FILE = Path("samples_dataset.npy")
CKPT_PATH = Path("checkpoints")
INDICES = [0, 10, 100]
OUT_DIR = Path("viz_outputs")


def _checkpoint_feature_count(path: Path) -> int | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    if mean.ndim != 1 or mean.size < 1:
        return None
    return int(mean.size)


def _resolve_checkpoint_file(path: Path, *, expected_n_features: int) -> Path:
    if path.is_file():
        feat_count = _checkpoint_feature_count(path)
        if feat_count != int(expected_n_features):
            raise SystemExit(
                f"Checkpoint {path} expects F={feat_count}, but samples_dataset.npy has F={expected_n_features}."
            )
        return path

    candidates = sorted(path.glob("**/reg_best.pt"))
    if not candidates:
        raise SystemExit("No reg_best.pt found under checkpoints/. Run training first.")

    compatible = [cand for cand in candidates if _checkpoint_feature_count(cand) == int(expected_n_features)]
    if compatible:
        return compatible[-1]

    raise SystemExit(
        f"No compatible checkpoint found for samples_dataset.npy feature count F={expected_n_features}. "
        "Re-run `python -m emg_tst.run_experiment`."
    )


def load_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    model_type = str(cfg.get("model_type", ""))
    if model_type != "sensor_fusion_last_step_transformer":
        raise SystemExit(f"Unsupported checkpoint model_type={model_type!r}; retrain with the integrated transformer.")
    model = SensorFusionTransformer(
        n_emg_vars=int(cfg["n_emg_vars"]),
        n_imu_vars=int(cfg["n_imu_vars"]),
        seq_len=int(cfg["seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
        causal=bool(cfg.get("causal", False)),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("reg_state_dict"))
    model.load_state_dict(state, strict=True)
    model.eval()
    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    std = np.asarray(scaler.get("std", None), dtype=np.float32)
    extra = ckpt.get("extra", {})
    feature_cols = np.asarray(extra.get("feature_cols", []), dtype=np.int64).reshape(-1)
    return model, mean, std, feature_cols


def _parse_feature_layout(data: dict, F: int):
    n_ch = int(data.get("n_channels", 16))
    emg_layout = emg_feature_layout_from_meta(data)
    n_emg_per_sensor = int(emg_layout["per_sensor"])
    emg_mode = str(emg_layout["mode"])
    thigh_n = int(data.get("thigh_n_features", 1))
    n_omega = int(data.get("n_angular_velocity_features", 0))
    thigh_start = int(F - thigh_n)
    omega_end = thigh_start
    omega_start = max(0, omega_end - n_omega)
    emg_end = omega_start
    return {
        "n_ch": n_ch,
        "emg_mode": emg_mode,
        "emg_names": list(emg_layout["names"]),
        "emg_per_sensor": n_emg_per_sensor,
        "emg_start": 0,
        "emg_end": int(emg_end),
        "omega_slice": slice(omega_start, omega_end),
        "n_omega": int(n_omega),
        "thigh_slice": slice(thigh_start, F),
        "thigh_n": int(thigh_n),
    }


def _plot_channels(x: np.ndarray, layout: dict, window: int, idx: int, out_dir: Path):
    W, _ = x.shape
    emg_mode = str(layout["emg_mode"])
    emg_per_sensor = int(layout["emg_per_sensor"])
    emg_names = list(layout["emg_names"])
    n_omega = int(layout.get("n_omega", 0))
    n_rows = 3 + (1 if n_omega > 0 else 0) + 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    ax_i = 0

    for s in range(3):
        ax = axes[ax_i]
        ax_i += 1
        base = int(layout["emg_start"]) + s * emg_per_sensor
        sensor_block = x[:, base : base + emg_per_sensor]
        if emg_mode == "raw_snippets":
            ax.imshow(
                sensor_block.T,
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
                interpolation="nearest",
                extent=[0, W - 1, 0, emg_per_sensor - 1],
            )
            ax.set_ylabel("lag")
            ax.set_title(f"Sensor {s + 1} - Raw EMG snippet ({emg_per_sensor} lags)")
        else:
            for c in range(sensor_block.shape[1]):
                label = emg_names[c] if c < len(emg_names) else f"f{c}"
                ax.plot(sensor_block[:, c], linewidth=0.7, alpha=0.85, label=label)
            ax.set_ylabel("z-score")
            ax.set_title(f"Sensor {s + 1} - EMG inputs ({emg_mode})")
            if sensor_block.shape[1] <= 16:
                ax.legend(ncol=min(sensor_block.shape[1], 8), fontsize=6, frameon=False)
            ax.grid(True, linestyle="--", alpha=0.4)

    if n_omega > 0:
        ax_omega = axes[ax_i]
        ax_i += 1
        om = x[:, layout["omega_slice"]]
        labels = ["omega_x", "omega_y", "omega_z"][: om.shape[1]]
        for j in range(om.shape[1]):
            ax_omega.plot(om[:, j], linewidth=0.9, label=labels[j] if j < len(labels) else f"omega_{j}")
        ax_omega.set_title("Thigh Angular Velocity")
        ax_omega.legend(fontsize=8, frameon=False)
        ax_omega.grid(True, linestyle="--", alpha=0.4)
        ax_omega.set_ylabel("z-score")

    ax_th = axes[ax_i]
    th = x[:, layout["thigh_slice"]]
    ax_th.plot(th[:, 0], linewidth=0.9, label="thigh_q_w")
    ax_th.plot(th[:, 1], linewidth=0.9, label="thigh_q_x")
    ax_th.plot(th[:, 2], linewidth=0.9, label="thigh_q_y")
    ax_th.plot(th[:, 3], linewidth=0.9, label="thigh_q_z")
    ax_th.set_title("Thigh Quaternion (wxyz)")
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

    data = np.load(SAMPLES_FILE, allow_pickle=True).item()
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    y_seq = data.get("y_seq", None)
    if y_seq is not None:
        y_seq = y_seq.astype(np.float32)

    window = int(data.get("window", X.shape[1]))
    F = int(X.shape[2])
    layout = _parse_feature_layout(data, F)
    ckpt_file = _resolve_checkpoint_file(CKPT_PATH, expected_n_features=int(F))
    model, mean, std, feature_cols = load_ckpt(ckpt_file, device=device)

    print(
        f"Feature layout: F={F} emg_mode={layout['emg_mode']} emg_per_sensor={layout['emg_per_sensor']} "
        f"omega={layout['n_omega']} thigh_n={layout['thigh_n']}"
    )

    Xn = (X - mean[None, None, :]) / std[None, None, :]
    for idx in INDICES:
        if idx < 0 or idx >= Xn.shape[0]:
            continue
        x_full = Xn[idx]
        x_sel = x_full[:, feature_cols]
        pred = rolling_last_step_predict(model, x_sel).numpy().astype(np.float32) * 180.0
        pred_last = float(pred[-1])
        true_label = float(y[idx])

        fig = plt.figure(figsize=(14, 5))
        plt.plot(pred, label="pred_angle_seq", linewidth=1.5)
        if y_seq is not None:
            plt.plot(y_seq[idx], label="true_angle_seq", linewidth=1.5, linestyle="--")
        else:
            plt.axhline(true_label, linestyle="--", linewidth=1.0, label="true_label")
        plt.title(f"sample {idx}: pred_last={pred_last:.3f} true_last={true_label:.3f}")
        plt.xlabel(f"t (samples, window={window})")
        plt.ylabel("knee angle (degrees)")
        plt.legend()
        fig.savefig(OUT_DIR / f"sample_{idx:06d}_pred.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        _plot_channels(x_full, layout, window, idx, OUT_DIR)

    print("Wrote images to:", OUT_DIR)
    print("Used checkpoint:", ckpt_file)


if __name__ == "__main__":
    main()
