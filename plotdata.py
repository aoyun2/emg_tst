import numpy as np
import matplotlib.pyplot as plt
import os


def plot_emg_and_angle(filename: str, *, assume_hz: float = 10.0):
    """
    Loads a recording and plots all channels:
      - Knee angle (label)
      - Thigh orientation (input feature): scalar thigh_angle (legacy) and/or thigh_quat_wxyz (wxyz)
      - device_spectr per sensor (active channels only)
      - Raw EMG waveform per sensor (if available)
    """
    if not os.path.exists(filename):
        print(f"Error: File not found at path: {filename}")
        return

    try:
        data = np.load(filename, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading NumPy file '{filename}': {e}")
        return

    required_keys = ["imu", "emg_sensor1", "emg_sensor2", "emg_sensor3"]
    if not all(k in data for k in required_keys):
        print(f"Error: Missing required keys. Required: {required_keys}")
        print(f"Available keys: {list(data.keys())}")
        return

    knee_angle = np.asarray(data["imu"]).reshape(-1).astype(np.float32)
    s1 = np.asarray(data["emg_sensor1"], dtype=np.float32)  # (N_CH, T)
    s2 = np.asarray(data["emg_sensor2"], dtype=np.float32)
    s3 = np.asarray(data["emg_sensor3"], dtype=np.float32)

    hz = float(data.get("effective_hz", assume_hz)) or assume_hz
    n_ch = s1.shape[0]

    has_raw = "raw_emg_sensor1" in data
    if has_raw:
        raw1 = np.asarray(data["raw_emg_sensor1"], dtype=np.float64)
        raw2 = np.asarray(data["raw_emg_sensor2"], dtype=np.float64)
        raw3 = np.asarray(data["raw_emg_sensor3"], dtype=np.float64)

    timestamps = None
    if "timestamps" in data:
        try:
            timestamps = np.asarray(data["timestamps"]).reshape(-1).astype(np.float32)
        except Exception:
            timestamps = None

    T = min(knee_angle.shape[0], s1.shape[1], s2.shape[1], s3.shape[1])
    if timestamps is not None:
        T = min(T, timestamps.shape[0])

    if T <= 1:
        print("Error: Not enough samples to plot.")
        return

    knee_angle = knee_angle[:T]
    s1 = s1[:, :T]
    s2 = s2[:, :T]
    s3 = s3[:, :T]

    if timestamps is None or timestamps.shape[0] < T:
        t = np.arange(T, dtype=np.float32) / float(hz)
    else:
        t = timestamps[:T]
        if not np.isfinite(t[0]) or not np.isfinite(t[-1]) or t[-1] == t[0]:
            t = np.arange(T, dtype=np.float32) / float(hz)

    duration = t[-1] - t[0]

    # Count active spectr channels (nonzero std)
    active_ch = set()
    for s in [s1, s2, s3]:
        for c in range(n_ch):
            if s[c].std() > 0.01:
                active_ch.add(c)
    active_ch = sorted(active_ch)

    # Layout: knee + thigh + 3 spectr + (3 raw if available)
    has_thigh_quat = "thigh_quat_wxyz" in data
    n_rows = (6 if has_thigh_quat else 5) + (3 if has_raw else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows), sharex=False)

    title = f"{os.path.basename(filename)}  |  {len(active_ch)}/{n_ch} active ch  |  ~{hz:.1f} Hz  |  {T} samples  |  {duration:.1f}s"
    if has_raw:
        raw_hz = len(raw1) / duration if duration > 0 else 0
        title += f"  |  raw ~{raw_hz:.0f} Hz"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Knee angle
    axes[0].plot(t, knee_angle, linewidth=1.0)
    axes[0].set_title("Knee Angle (label)")
    axes[0].set_ylabel("Degrees")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Thigh angle (legacy scalar feature, kept for debugging)
    if "thigh_angle" in data:
        thigh = np.asarray(data["thigh_angle"]).reshape(-1).astype(np.float32)[:T]
        axes[1].plot(t, thigh, linewidth=1.0, color="orange", label="thigh_angle")
    axes[1].set_title("Thigh Angle (legacy scalar feature)")
    axes[1].set_ylabel("Degrees")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=False)

    # Thigh quaternion (new feature)
    ax_offset = 0
    if has_thigh_quat:
        ax_q = axes[2]
        q = np.asarray(data["thigh_quat_wxyz"], dtype=np.float32).reshape(-1, 4)[:T]
        ax_q.plot(t, q[:, 0], linewidth=0.9, label="w")
        ax_q.plot(t, q[:, 1], linewidth=0.9, label="x")
        ax_q.plot(t, q[:, 2], linewidth=0.9, label="y")
        ax_q.plot(t, q[:, 3], linewidth=0.9, label="z")
        ax_q.set_title("Thigh Quaternion (wxyz) - input feature")
        ax_q.set_ylabel("Quat")
        ax_q.grid(True, linestyle="--", alpha=0.5)
        ax_q.legend(ncol=4, fontsize=8, frameon=False)
        ax_offset = 1

    # device_spectr per sensor (only active channels)
    for idx, (s, name) in enumerate([(s1, "Sensor 1"), (s2, "Sensor 2"), (s3, "Sensor 3")]):
        ax = axes[idx + 2 + ax_offset]
        for c in active_ch:
            ax.plot(t, s[c, :], linewidth=0.6, alpha=0.85, label=f"ch{c}")
        ax.set_title(f"{name} — device_spectr (active ch only)")
        ax.set_ylabel("Spectral")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(ncol=min(len(active_ch), 8), fontsize=7, frameon=False)

    axes[4 + ax_offset].set_xlabel("Time (s)")

    # Raw EMG waveform per sensor
    if has_raw:
        for idx, (raw, name) in enumerate([(raw1, "Sensor 1"), (raw2, "Sensor 2"), (raw3, "Sensor 3")]):
            ax = axes[5 + ax_offset + idx]
            R = len(raw)
            t_raw = np.linspace(t[0], t[-1], R)
            ax.plot(t_raw, raw, linewidth=0.3, alpha=0.8, color=f"C{idx}")
            ax.set_title(f"{name} — Raw EMG waveform ({R} samples)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    plot_emg_and_angle("data0.npy")
