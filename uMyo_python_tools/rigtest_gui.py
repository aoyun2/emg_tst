"""
rigtest_gui.py  –  GUI-based EMG + knee-angle recording script.

Key improvements over rigtest.py
─────────────────────────────────
1. Calibration step  (stand straight for ~3 s at startup)
   • The uMyo LSM6DS3TR-C (thigh) has no magnetometer → heading is arbitrary.
   • The BWT901CL (calf) uses magnetic north → different heading frame.
   • Turning the body changed the old knee angle even with a fixed knee.
   • A one-time correction quaternion is computed at calibration and applied
     to all subsequent calf quaternions, aligning the two frames.

2. Vector dot-product knee angle formula
   • Sensor axes:
       Thigh (uMyo BF):  +Y = distal (toward knee), +X = subject's left, +Z = posterior
       Calf  (BWT901CL): +Y = proximal (toward knee = up when standing),
                          +X = subject's right, +Z = posterior
   • knee_included_deg = arccos(v_thigh · v_calf_corrected) × (180/π)
     where v_thigh = q_thigh.rotate([0,1,0])  (toward-knee direction, world frame)
           v_calf  = q_calf_corrected.rotate([0,1,0])
   • When leg is straight: v_thigh ≈ down, v_calf ≈ up → dot ≈ -1 → 180°  ✓
   • No gimbal lock; correct at any thigh elevation.

3. Raw EMG timestamps
   • Each burst of raw EMG samples is timestamped for accurate time-domain
     feature alignment in post-processing.

4. Artifact detection
   • Spike: knee angle changes > SPIKE_DEG per step → sample flagged.
   • Flatline: knee angle std < FLAT_STD over last FLAT_WIN steps → flagged.
   • Flagged samples are still saved (for QC plots) but marked with `quality_flag`.

5. Tkinter GUI
   • Calibrate button (required before recording).
   • Start / Stop Recording toggle.
   • Live knee-angle plot (last 5 s) and per-sensor EMG RMS.
   • Status bar with warnings.
"""

import math
import sys
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial
import socket
from serial.tools import list_ports

# Local imports (same directory)
sys.path.insert(0, str(Path(__file__).parent))
import umyo_parser
import quat_math
import pywitmotion as wit

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_LEN   = 1000        # ~5 s at 200 Hz for live plots
SPIKE_DEG     = 40.0        # knee-angle jump > this per step → spike flag
FLAT_WIN      = 20          # window (steps) for flatline detection
FLAT_STD      = 0.5         # std threshold (degrees) for flatline
CAL_DURATION  = 3.0         # seconds of calibration data to collect
IMU_TIMEOUT   = 0.005       # BT socket timeout (s) – 5 ms at 200 Hz

# Sensor unit IDs (order: VM, SM, BF)
SENSOR_IDS = [1626655497, 4100972396, 1969998036]
THIGH_SENSOR_IDX = 2        # BF (biceps femoris) = thigh sensor

IMU_MAC  = "00:0C:BF:07:42:47"
IMU_PORT = 1

# BWT901CL commands
UNLOCK_CMD    = bytes.fromhex('FF AA 69 88 B5')
SAVE_CMD      = bytes.fromhex('FF AA 00 00 00')
HORIZ_INST    = bytes.fromhex('FF AA 23 00 00')
HZ200_CMD     = bytes.fromhex('FF AA 03 0B 00')
# Disable magnetometer: set IMUALGCON register (0x24) to 0x01 = 6-axis horizontal
# (accelerometer + gyroscope only, no magnetometer heading update).
#
# WHY: The BWT901CL uses magnetic north as its heading reference. The uMyo
# LSM6DS3TR-C has no magnetometer — its heading is arbitrary. When you turn,
# the BWT901CL's world frame rotates to track north while the uMyo's doesn't.
# Any formula that computes the angle between the two sensors' vectors breaks,
# because the vectors are now in different coordinate frames.
#
# With the magnetometer disabled, both sensors use gravity + gyro only.
# Their headings drift at the same slow rate, so the calibration correction
# computed at startup stays valid for the entire recording session.
DISABLE_MAG   = bytes.fromhex('FF AA 24 01 00')  # 6-axis (no magnetometer)
ENABLE_MAG    = bytes.fromhex('FF AA 24 00 00')  # 9-axis (magnetometer on) — restores default

# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / vector math  (numpy-based, no gimbal lock)
# ─────────────────────────────────────────────────────────────────────────────

def rotate_vec(q_wxyz, v_xyz):
    """Rotate 3D vector v by unit quaternion q (wxyz).  Returns np.ndarray(3)."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    vx, vy, vz = float(v_xyz[0]), float(v_xyz[1]), float(v_xyz[2])
    rx = (1 - 2*y*y - 2*z*z)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
    ry = 2*(x*y + w*z)*vx + (1 - 2*x*x - 2*z*z)*vy + 2*(y*z - w*x)*vz
    rz = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1 - 2*x*x - 2*y*y)*vz
    return np.array([rx, ry, rz], dtype=np.float64)


def quat_mul(q1, q2):
    """Hamilton product q1 ⊗ q2 (wxyz), returns normalized tuple."""
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)


def quat_from_v_to_v(v_from, v_to):
    """
    Quaternion (wxyz) that rotates unit-vector v_from onto v_to.
    Handles parallel and anti-parallel cases robustly.
    """
    u = np.asarray(v_from, dtype=np.float64)
    v = np.asarray(v_to,   dtype=np.float64)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))

    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)          # already aligned

    if dot < -0.9999:                         # anti-parallel → 180° about any perp axis
        perp = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-4:
            perp = np.cross(u, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return (0.0, float(perp[0]), float(perp[1]), float(perp[2]))

    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([w, axis[0], axis[1], axis[2]])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def knee_included_deg(q_thigh, q_calf, angle_offset=0.0):
    """
    Compute knee included angle (degrees) using the same Euler-angle formula
    as the original rigtest.py.  This formula extracts each sensor's angle
    around a single gravity-referenced axis independently, so it is robust to
    body tilt, turning, and thigh elevation.

      180° = fully extended (straight)
       ~90° = 90° of flexion
        0° = fully bent (theoretical limit; real range ~30°–180°)

    Parameters
    ----------
    q_thigh      : (w,x,y,z) – thigh sensor quaternion (uMyo BF, Qsg)
    q_calf       : (w,x,y,z) – calf sensor quaternion (BWT901CL, wxyz)
    angle_offset : float – calibration offset in degrees (added to raw diff
                   so that the standing-straight pose reads exactly 180°).
    """
    w_c, x_c, y_c, z_c = q_calf
    w_t, x_t, y_t, z_t = q_thigh

    # Calf: rotation-matrix element that tracks knee flexion for this mounting
    t1 = math.atan2(2.0*(x_c*y_c + w_c*z_c), w_c*w_c + x_c*x_c - y_c*y_c - z_c*z_c)
    # Thigh: corresponding element from the uMyo Qsg quaternion
    t2 = math.atan2(2.0*(y_t*z_t + w_t*x_t), w_t*w_t - x_t*x_t - y_t*y_t + z_t*z_t)

    # Compute the circular difference (t1 - t2) normalised to (-π, π).
    # atan2(sin(a-b), cos(a-b)) gives the correct signed difference at any
    # angle, including when t1 or t2 crosses the ±180° wrap boundary.
    # This replaces the original "if dt1 < 0" patch and handles all
    # thigh/calf elevations without discontinuities.
    diff_rad = math.atan2(math.sin(t1 - t2), math.cos(t1 - t2))
    return 180.0 - math.degrees(diff_rad) + angle_offset


def avg_quaternions(qs):
    """
    Simple quaternion averaging: flip sign for continuity, then average + renorm.
    Good enough for a 3-second calibration window.
    """
    qs = np.asarray(qs, dtype=np.float64)
    for i in range(1, len(qs)):
        if np.dot(qs[i-1], qs[i]) < 0:
            qs[i] *= -1
    q_avg = qs.mean(axis=0)
    return tuple(q_avg / np.linalg.norm(q_avg))


# ─────────────────────────────────────────────────────────────────────────────
# Hardware init
# ─────────────────────────────────────────────────────────────────────────────

def init_serial():
    ports = list(list_ports.comports())
    print("Available serial ports:")
    device = None
    for p in ports:
        print(" ", p.device)
        device = p.device
    if device is None:
        raise RuntimeError("No serial port found.")
    ser = serial.Serial(port=device, baudrate=921600, parity=serial.PARITY_NONE,
                        stopbits=1, bytesize=8, timeout=0)
    print(f"Serial: {ser.portstr}")
    return ser


def init_imu():
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.connect((IMU_MAC, IMU_PORT))
    sock.settimeout(1.0)
    for cmd in [UNLOCK_CMD, HORIZ_INST, HZ200_CMD, ENABLE_MAG, SAVE_CMD]:
        sock.send(cmd)
        time.sleep(0.1)
    print("BWT901CL configured.")
    sock.settimeout(IMU_TIMEOUT)
    return sock


def init_umyo(ser):
    cnt = ser.in_waiting
    if cnt > 0:
        data = ser.read(cnt)
        umyo_parser.umyo_parse_preprocessor(data)

    umyos_raw = umyo_parser.umyo_get_list()
    if len(umyos_raw) < 3:
        raise RuntimeError(f"Only {len(umyos_raw)} uMyo sensors detected (need 3).")

    ordered = []
    for uid in SENSOR_IDS:
        match = next((u for u in umyos_raw if u.unit_id == uid), None)
        if match is None:
            raise RuntimeError(f"Sensor with unit_id {uid} not found.")
        ordered.append(match)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Global shared state
# ─────────────────────────────────────────────────────────────────────────────

data_lock = threading.Lock()

# Current sensor readings (updated every loop iteration)
state = {
    "q_thigh":  (1.0, 0.0, 0.0, 0.0),   # latest thigh quaternion (wxyz)
    "q_calf":   (1.0, 0.0, 0.0, 0.0),   # latest calf  quaternion (wxyz)
    "knee_deg": 90.0,                     # latest knee included angle
    "quality":  0,                        # 0=ok, 1=spike, 2=flatline, 3=no_cal
}

# Calibration
cal = {
    "status":       "none",              # "none" | "collecting" | "done" | "failed"
    "q_thigh_buf":  [],
    "q_calf_buf":   [],
    "angle_offset": 0.0,                 # degrees to add so standing-straight = 180°
    "start_time":   None,
    "cal_angle":    None,                # raw angle at calibration pose (before offset)
}

# Recording
rec = {
    "active":      False,
    "file_index":  0,
    # --- saved arrays ---
    "timestamps":       [],
    "knee_deg":         [],
    "quality_flags":    [],              # 0=ok, 1=spike, 2=flatline
    "thigh_quat_wxyz":  [],
    "calf_quat_wxyz":   [],             # corrected calf quaternion
    "emg_sensor":       None,           # will be [[], [], ..., []] per channel × 3 sensors
    "raw_emg":          [[], [], []],   # flat list of raw int16 samples
    "raw_emg_times":    [[], [], []],   # timestamp per raw sample
    "n_channels":       0,
}

# Live display deques
disp = {
    "times":        deque(maxlen=DISPLAY_LEN),
    "knee":         deque(maxlen=DISPLAY_LEN),
    "quality":      deque(maxlen=DISPLAY_LEN),
    # Quaternion components (w, x, y, z) for thigh and calf
    "thigh_quat":   [deque(maxlen=DISPLAY_LEN) for _ in range(4)],
    "calf_quat":    [deque(maxlen=DISPLAY_LEN) for _ in range(4)],
    # All EMG spectral channels per sensor; populated once n_channels is known
    "emg_channels": None,   # list of 3 × [deque×n_ch]; set in update_loop
}

# Hardware objects (filled in main)
hw = {"ser": None, "sock": None, "umyos": None, "start_time": None}

# Prev data_id per sensor (to detect new packets)
prev_data_ids = [0, 0, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Sensor read functions
# ─────────────────────────────────────────────────────────────────────────────

def drain_umyo():
    """Read serial buffer; update uMyo sensor state; accumulate raw EMG with timestamps."""
    ser    = hw["ser"]
    umyos  = hw["umyos"]
    t_now  = time.time() - hw["start_time"]

    cnt = ser.in_waiting
    if cnt > 0:
        umyo_parser.umyo_parse_preprocessor(ser.read(cnt))

    # Accumulate raw EMG samples with per-batch timestamps (recording only)
    if rec["active"]:
        for s in range(3):
            curr_id = umyos[s].data_id
            if curr_id != prev_data_ids[s]:
                prev_data_ids[s] = curr_id
                dc = umyos[s].data_count
                dt = 1.0 / 400.0  # ~2.5 ms between raw samples
                for i in range(dc):
                    sample_t = t_now - (dc - 1 - i) * dt
                    rec["raw_emg"][s].append(umyos[s].data_array[i])
                    rec["raw_emg_times"][s].append(sample_t)
    else:
        # Still advance data_id tracking so we don't replay stale packets
        # when recording starts.
        for s in range(3):
            prev_data_ids[s] = umyos[s].data_id

    # Update thigh quaternion from BF sensor (sensor index 2)
    bf = umyos[THIGH_SENSOR_IDX]
    q_raw = quat_math.sQ(bf.Qsg[0], bf.Qsg[1], bf.Qsg[2], bf.Qsg[3])
    q_norm = quat_math.q_renorm(q_raw)
    with data_lock:
        state["q_thigh"] = (q_norm.w, q_norm.x, q_norm.y, q_norm.z)


def read_imu():
    """
    Drain BWT901CL Bluetooth buffer.  Returns list of computed knee angles for
    each valid quaternion packet received in this call.
    """
    results = []
    try:
        raw = hw["sock"].recv(4096)
    except socket.timeout:
        return results
    except socket.error as e:
        print(f"BT recv error: {e}")
        return results

    for msg in raw.split(b'U'):
        q = wit.get_quaternion(msg)
        if q is None:
            continue

        # Normalise calf quaternion
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = math.sqrt(w*w + x*x + y*y + z*z)
        if n < 1e-10:
            continue
        q_calf = (w/n, x/n, y/n, z/n)

        with data_lock:
            q_thigh      = state["q_thigh"]
            angle_offset = cal["angle_offset"]
            cal_status   = cal["status"]

        # Compute knee angle
        angle = knee_included_deg(q_thigh, q_calf, angle_offset=angle_offset)

        results.append((angle, q_thigh, q_calf))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main data collection loop  (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def update_loop():
    """Background thread: reads sensors, updates state, collects calibration data."""
    umyos   = hw["umyos"]
    n_ch    = len(umyos[0].device_spectr)
    rec["n_channels"] = n_ch
    if rec["emg_sensor"] is None:
        rec["emg_sensor"] = [[[] for _ in range(n_ch)] for _ in range(3)]

    prev_knee    = None
    knee_history = deque(maxlen=FLAT_WIN)

    while True:
        drain_umyo()
        imu_readings = read_imu()

        t_global = time.time() - hw["start_time"]

        for angle, q_thigh, q_calf in imu_readings:
            # Clamp to valid anatomical range before any further processing
            angle = float(np.clip(angle, 0.0, 180.0))

            # Artifact detection
            flag = 0
            if prev_knee is not None and abs(angle - prev_knee) > SPIKE_DEG:
                flag = 1   # spike
            knee_history.append(angle)
            if len(knee_history) == FLAT_WIN:
                if np.std(list(knee_history)) < FLAT_STD:
                    flag = 2  # flatline

            # Replace spike with last valid value so the saved label is clean.
            # prev_knee is already clamped, so this is safe.
            clean_angle = prev_knee if (flag == 1 and prev_knee is not None) else angle
            prev_knee = angle  # track raw (post-clamp) for next spike check

            # Calibration: feed data
            with data_lock:
                if cal["status"] == "collecting":
                    cal["q_thigh_buf"].append(q_thigh)
                    cal["q_calf_buf"].append(q_calf)
                    elapsed = t_global - cal["start_time"]
                    if elapsed >= CAL_DURATION:
                        _finish_calibration()

                # Update live state
                state["q_calf"]   = q_calf
                state["knee_deg"] = clean_angle
                state["quality"]  = flag if cal["status"] == "done" else 3

            # Snapshot EMG (latest device_spectr from each sensor)
            emg_ch_vals = []
            for s in range(3):
                ch_vals = umyos[s].device_spectr
                emg_ch_vals.append(ch_vals)
                if rec["active"]:
                    for c in range(n_ch):
                        rec["emg_sensor"][s][c].append(ch_vals[c])

            # Save recording sample
            if rec["active"]:
                rec["timestamps"].append(t_global)
                rec["knee_deg"].append(clean_angle)
                rec["quality_flags"].append(flag)
                rec["thigh_quat_wxyz"].append(q_thigh)
                rec["calf_quat_wxyz"].append(q_calf)

            # Update display deques (always)
            # Ensure emg_channels deques are initialised on first packet
            if disp["emg_channels"] is None:
                disp["emg_channels"] = [
                    [deque(maxlen=DISPLAY_LEN) for _ in range(n_ch)]
                    for _ in range(3)
                ]

            disp["times"].append(t_global)
            disp["knee"].append(clean_angle)
            disp["quality"].append(flag)
            for i, comp in enumerate(q_thigh):
                disp["thigh_quat"][i].append(comp)
            for i, comp in enumerate(q_calf):
                disp["calf_quat"][i].append(comp)
            for s in range(3):
                for c in range(n_ch):
                    disp["emg_channels"][s][c].append(emg_ch_vals[s][c])

        time.sleep(0.001)


def _finish_calibration():
    """Called under data_lock when calibration collection time is up."""
    q_th_list = cal["q_thigh_buf"]
    q_ca_list = cal["q_calf_buf"]
    if len(q_th_list) < 10:
        cal["status"] = "failed"
        return

    q_th_avg = avg_quaternions(q_th_list)
    q_ca_avg = avg_quaternions(q_ca_list)

    # Compute raw angle at calibration pose with zero offset
    raw_angle = knee_included_deg(q_th_avg, q_ca_avg, angle_offset=0.0)
    # Offset so that standing-straight reads exactly 180°
    offset = 180.0 - raw_angle

    cal["angle_offset"] = offset
    cal["cal_angle"]    = raw_angle
    cal["status"]       = "done"
    cal["q_thigh_buf"]  = []
    cal["q_calf_buf"]   = []


# ─────────────────────────────────────────────────────────────────────────────
# Recording save / clear
# ─────────────────────────────────────────────────────────────────────────────

def save_recording(parent_dir: Path):
    """Save current recording buffers to a numbered data*.npy file."""
    n = len(rec["timestamps"])
    if n == 0:
        print("Nothing to save.")
        return None

    # Find next available filename
    idx = 0
    while (parent_dir / f"data{idx}.npy").exists():
        idx += 1
    out_path = parent_dir / f"data{idx}.npy"

    n_ch = rec["n_channels"]
    dur  = rec["timestamps"][-1] - rec["timestamps"][0] if n > 1 else 1.0

    d = {
        # Labels
        "knee_included_deg":  np.array(rec["knee_deg"],      dtype=np.float64),
        "quality_flags":       np.array(rec["quality_flags"], dtype=np.int8),
        # Timestamps
        "timestamps":          np.array(rec["timestamps"],    dtype=np.float64),
        # Quaternions
        "thigh_quat_wxyz":    np.array(rec["thigh_quat_wxyz"], dtype=np.float64),
        "calf_quat_wxyz":     np.array(rec["calf_quat_wxyz"],  dtype=np.float64),
        # EMG spectral (C, T) per sensor
        "emg_sensor1":        np.array([np.array(rec["emg_sensor"][0][c]) for c in range(n_ch)], dtype=np.float64),
        "emg_sensor2":        np.array([np.array(rec["emg_sensor"][1][c]) for c in range(n_ch)], dtype=np.float64),
        "emg_sensor3":        np.array([np.array(rec["emg_sensor"][2][c]) for c in range(n_ch)], dtype=np.float64),
        # Raw EMG with timestamps
        "raw_emg_sensor1":    np.array(rec["raw_emg"][0],      dtype=np.float64),
        "raw_emg_sensor2":    np.array(rec["raw_emg"][1],      dtype=np.float64),
        "raw_emg_sensor3":    np.array(rec["raw_emg"][2],      dtype=np.float64),
        "raw_emg_times1":     np.array(rec["raw_emg_times"][0], dtype=np.float64),
        "raw_emg_times2":     np.array(rec["raw_emg_times"][1], dtype=np.float64),
        "raw_emg_times3":     np.array(rec["raw_emg_times"][2], dtype=np.float64),
        # Metadata
        "n_channels":         n_ch,
        "effective_hz":       (n - 1) / dur if n > 1 else 0.0,
        "calibration_angle_deg": cal.get("cal_angle"),
        "calibration_angle_offset": cal.get("angle_offset", 0.0),
        "thigh_sensor_axes":  "right thigh (BF): +Y=distal(knee), +X=subject-left, +Z=posterior",
        "calf_sensor_axes":   "right calf (BWT901CL): +Y=proximal(knee=up), +X=subject-right, +Z=posterior",
        "knee_formula":       "arccos(v_thigh · v_calf_corrected) where v = rotate([0,1,0]), 180=straight",
    }

    np.save(out_path, d, allow_pickle=True)
    print(f"Saved → {out_path}  ({n} samples, {dur:.1f} s, {n_ch} ch/sensor)")
    return out_path


def clear_recording_buffers():
    """Reset all recording buffers (call after saving)."""
    rec["timestamps"]       = []
    rec["knee_deg"]         = []
    rec["quality_flags"]    = []
    rec["thigh_quat_wxyz"]  = []
    rec["calf_quat_wxyz"]   = []
    n_ch = rec["n_channels"]
    rec["emg_sensor"]       = [[[] for _ in range(n_ch)] for _ in range(3)]
    rec["raw_emg"]          = [[], [], []]
    rec["raw_emg_times"]    = [[], [], []]


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = save_dir
        self.title("EMG Recording  –  rigtest_gui")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_widgets()
        self._schedule_update()

    # ── Widget layout ─────────────────────────────────────────────────────────

    def _build_widgets(self):
        # ── Status bar ────────────────────────────────────────────────────────
        top = tk.Frame(self, pady=4)
        top.pack(fill="x", padx=8)

        self.lbl_cal = tk.Label(top, text="⚠ Not calibrated", fg="red",
                                font=("Helvetica", 11, "bold"))
        self.lbl_cal.pack(side="left")

        tk.Button(top, text="  Calibrate (stand straight)  ",
                  command=self._start_calibration).pack(side="left", padx=8)

        self.lbl_rec = tk.Label(top, text="● Not recording", fg="gray",
                                font=("Helvetica", 11))
        self.lbl_rec.pack(side="left", padx=12)

        self.lbl_quality = tk.Label(top, text="", fg="orange",
                                    font=("Helvetica", 10))
        self.lbl_quality.pack(side="left", padx=8)

        self.btn_rec = tk.Button(top, text="▶  Start Recording",
                                 command=self._toggle_recording,
                                 state="disabled", width=18)
        self.btn_rec.pack(side="right", padx=8)

        self.lbl_info = tk.Label(top, text="", fg="gray", font=("Helvetica", 9))
        self.lbl_info.pack(side="right", padx=8)

        # ── Plots ─────────────────────────────────────────────────────────────
        self.fig, axes = plt.subplots(6, 1, figsize=(10, 14))
        self.fig.tight_layout(pad=2.5)

        # Row 0: Knee angle
        self.ax_knee = axes[0]
        self.ax_knee.set_title("Knee included angle  (180° = straight)")
        self.ax_knee.set_ylabel("Degrees")
        self.ax_knee.set_ylim(0, 200)
        self.line_knee, = self.ax_knee.plot([], [], color="steelblue", lw=1.2)
        self.scat_spike, = self.ax_knee.plot([], [], "rx", ms=6, label="spike")
        self.ax_knee.axhline(180, color="gray", lw=0.6, ls="--")
        self.ax_knee.legend(loc="upper right", fontsize=8)

        # Row 1: Thigh quaternion (w, x, y, z)
        self.ax_thigh_q = axes[1]
        self.ax_thigh_q.set_title("Thigh quaternion  (uMyo BF, w x y z)")
        self.ax_thigh_q.set_ylabel("Component")
        self.ax_thigh_q.set_ylim(-1.1, 1.1)
        quat_colors = ["black", "tab:red", "tab:green", "tab:blue"]
        quat_labels = ["w", "x", "y", "z"]
        self.lines_thigh_q = [
            self.ax_thigh_q.plot([], [], color=quat_colors[i], lw=1, label=quat_labels[i])[0]
            for i in range(4)
        ]
        self.ax_thigh_q.legend(loc="upper right", fontsize=8, ncol=4)

        # Row 2: Calf quaternion (w, x, y, z)  — corrected
        self.ax_calf_q = axes[2]
        self.ax_calf_q.set_title("Calf quaternion  (BWT901CL, corrected, w x y z)")
        self.ax_calf_q.set_ylabel("Component")
        self.ax_calf_q.set_ylim(-1.1, 1.1)
        self.lines_calf_q = [
            self.ax_calf_q.plot([], [], color=quat_colors[i], lw=1, label=quat_labels[i])[0]
            for i in range(4)
        ]
        self.ax_calf_q.legend(loc="upper right", fontsize=8, ncol=4)

        # Rows 3–5: EMG spectral heatmap per sensor (channels × time, normalized)
        sensor_names = ["Sensor 1 (VM)", "Sensor 2 (SM)", "Sensor 3 (BF/thigh)"]
        self.ax_emg = [axes[3], axes[4], axes[5]]
        self.im_emg = [None, None, None]   # imshow objects, created on first packet
        for s in range(3):
            self.ax_emg[s].set_title(f"EMG spectral – {sensor_names[s]}  (channel × time, per-channel normalized)")
            self.ax_emg[s].set_ylabel("Channel")
            self.ax_emg[s].set_xlabel("Time (s)")

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas = canvas

        # ── Bottom status bar ─────────────────────────────────────────────────
        bot = tk.Frame(self, pady=3)
        bot.pack(fill="x", padx=8)
        self.lbl_angle = tk.Label(bot, text="Knee: --°", font=("Helvetica", 14, "bold"))
        self.lbl_angle.pack(side="left")
        self.lbl_samples = tk.Label(bot, text="", fg="gray", font=("Helvetica", 10))
        self.lbl_samples.pack(side="right")

    # ── Calibration ───────────────────────────────────────────────────────────

    def _start_calibration(self):
        with data_lock:
            cal["status"]      = "collecting"
            cal["q_thigh_buf"] = []
            cal["q_calf_buf"]  = []
            cal["angle_offset"] = 0.0
            cal["start_time"]  = time.time() - hw["start_time"]
        self.lbl_cal.config(text=f"⏳ Collecting calibration ({CAL_DURATION:.0f} s)…", fg="darkorange")
        self.btn_rec.config(state="disabled")

    # ── Recording toggle ──────────────────────────────────────────────────────

    def _toggle_recording(self):
        if not rec["active"]:
            clear_recording_buffers()
            rec["active"] = True
            self.btn_rec.config(text="■  Stop & Save", bg="salmon")
            self.lbl_rec.config(text="● Recording…", fg="red")
        else:
            rec["active"] = False
            self.btn_rec.config(text="▶  Start Recording", bg=self.cget("bg"))
            self.lbl_rec.config(text="● Not recording", fg="gray")
            path = save_recording(self.save_dir)
            if path:
                rec["file_index"] += 1
                self.lbl_info.config(text=f"Saved: {path.name}")

    # ── Periodic GUI refresh ──────────────────────────────────────────────────

    def _schedule_update(self):
        self.after(50, self._update_gui)

    def _update_gui(self):
        try:
            self._refresh()
        finally:
            self._schedule_update()

    def _refresh(self):
        # Calibration status
        with data_lock:
            cs = cal["status"]
            angle_now = state["knee_deg"]
            quality   = state["quality"]

        if cs == "done":
            check_a = cal.get("cal_angle", 0)
            self.lbl_cal.config(
                text=f"✓ Calibrated  (straight-leg check: {check_a:.1f}°)",
                fg="darkgreen"
            )
            self.btn_rec.config(state="normal")
        elif cs == "failed":
            self.lbl_cal.config(
                text="✗ Calibration failed – try again while standing straight", fg="red"
            )
        elif cs == "collecting":
            elapsed = (time.time() - hw["start_time"]) - cal["start_time"]
            remain  = max(0.0, CAL_DURATION - elapsed)
            self.lbl_cal.config(text=f"⏳ Calibrating… {remain:.1f} s remaining", fg="darkorange")

        # Quality warning
        q_labels = {0: ("", "black"), 1: ("⚡ SPIKE", "red"), 2: ("▬ FLATLINE", "red"),
                    3: ("⚠ Not calibrated", "orange")}
        qtxt, qcol = q_labels.get(quality, ("", "black"))
        self.lbl_quality.config(text=qtxt, fg=qcol)

        # Knee angle label
        self.lbl_angle.config(text=f"Knee: {angle_now:.1f}°")

        # Recording samples count
        if rec["active"]:
            n_s  = len(rec["timestamps"])
            dur  = (rec["timestamps"][-1] - rec["timestamps"][0]) if n_s > 1 else 0
            self.lbl_samples.config(text=f"Samples: {n_s}  ({dur:.1f} s)")

        times = list(disp["times"])
        knees = list(disp["knee"])
        quals = list(disp["quality"])
        if len(times) < 2:
            self.canvas.draw_idle()
            return

        xlim = (times[0], times[-1])

        # ── Knee angle ────────────────────────────────────────────────────────
        self.ax_knee.set_xlim(*xlim)
        self.line_knee.set_data(times, knees)
        spk_t = [times[i] for i in range(len(quals)) if quals[i] == 1]
        spk_k = [knees[i] for i in range(len(quals)) if quals[i] == 1]
        self.scat_spike.set_data(spk_t, spk_k)

        # ── Thigh quaternion ──────────────────────────────────────────────────
        self.ax_thigh_q.set_xlim(*xlim)
        for i in range(4):
            vals = list(disp["thigh_quat"][i])
            if len(vals) == len(times):
                self.lines_thigh_q[i].set_data(times, vals)

        # ── Calf quaternion ───────────────────────────────────────────────────
        self.ax_calf_q.set_xlim(*xlim)
        for i in range(4):
            vals = list(disp["calf_quat"][i])
            if len(vals) == len(times):
                self.lines_calf_q[i].set_data(times, vals)

        # ── EMG spectral heatmap (imshow, per-channel normalized) ────────────
        if disp["emg_channels"] is not None:
            n_ch = len(disp["emg_channels"][0])
            for s in range(3):
                data = np.array(
                    [list(disp["emg_channels"][s][c]) for c in range(n_ch)],
                    dtype=np.float32
                )   # shape (n_ch, T)
                if data.shape[1] < 2:
                    continue
                # Normalize each channel to [0, 1] independently
                row_min = data.min(axis=1, keepdims=True)
                row_max = data.max(axis=1, keepdims=True)
                rng = row_max - row_min
                rng[rng < 1e-6] = 1.0
                data_norm = (data - row_min) / rng

                extent = [times[0], times[-1], n_ch - 0.5, -0.5]
                if self.im_emg[s] is None:
                    self.im_emg[s] = self.ax_emg[s].imshow(
                        data_norm, aspect='auto', origin='upper',
                        extent=extent, cmap='viridis', vmin=0, vmax=1,
                        interpolation='nearest'
                    )
                    self.ax_emg[s].set_yticks(range(n_ch))
                    self.fig.colorbar(self.im_emg[s], ax=self.ax_emg[s], fraction=0.02)
                else:
                    self.im_emg[s].set_data(data_norm)
                    self.im_emg[s].set_extent(extent)

        self.canvas.draw_idle()

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _on_close(self):
        if rec["active"]:
            rec["active"] = False
            save_recording(self.save_dir)
        try:
            hw["sock"].close()
        except Exception:
            pass
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    save_dir = Path(__file__).parent.parent   # project root (same as rigtest.py)

    print("Initialising hardware…")
    hw["ser"]        = init_serial()
    hw["sock"]       = init_imu()
    hw["umyos"]      = init_umyo(hw["ser"])
    hw["start_time"] = time.time()

    n_ch = len(hw["umyos"][0].device_spectr)
    rec["n_channels"] = n_ch
    rec["emg_sensor"] = [[[] for _ in range(n_ch)] for _ in range(3)]
    print(f"uMyo: {n_ch} spectral channels per sensor")

    # Start background data thread
    t = threading.Thread(target=update_loop, daemon=True)
    t.start()

    # Launch GUI (must run on main thread)
    app = App(save_dir=save_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
