import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import socket
import umyo_parser
import pywitmotion as wit
import math
import quat_math
import numpy as np
from serial.tools import list_ports

# ─── Serial port setup ───────────────────────────────────────────────
port = list(list_ports.comports())
print("available ports:")
for p in port:
    print(p.device)
    device = p.device
print("===")

try:
    ser = serial.Serial(port=device, baudrate=921600, parity=serial.PARITY_NONE,
                        stopbits=1, bytesize=8, timeout=0)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# ─── IMU Bluetooth setup ─────────────────────────────────────────────
imu = "00:0C:BF:07:42:47"
imu_port = 1
try:
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.connect((imu, imu_port))
    sock.settimeout(1.0)

    # BWT901CL commands (per datasheet section 6.2)
    UNLOCK_COMMAND = bytes.fromhex('FF AA 69 88 B5')
    SAVE_COMMAND   = bytes.fromhex('FF AA 00 00 00')
    HORIZ_INSTALL  = bytes.fromhex('FF AA 23 00 00')
    HZ200          = bytes.fromhex('FF AA 03 0B 00')  # 0x0B = 200Hz (section 6.2.9)

    def send_command(sock, command):
        print(f"Sending command: {command.hex()}")
        sock.send(command)
        time.sleep(0.1)

    try:
        send_command(sock, UNLOCK_COMMAND)
        send_command(sock, HORIZ_INSTALL)
        send_command(sock, HZ200)
        send_command(sock, SAVE_COMMAND)
        print("Configuration updated and saved.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
except socket.error as e:
    print(f"Error connecting to IMU: {e}")
    exit()

print("conn: " + ser.portstr)

# ─── Initial sensor discovery ─────────────────────────────────────────
cnt = ser.in_waiting
if cnt > 0:
    data = ser.read(cnt)
    parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data)
    umyos = umyo_parser.umyo_get_list()

    if len(umyos) < 3:
        raise RuntimeError("Less than 3 sensors detected.")

    tmp = []
    tmp.append(next((u for u in umyos if u.unit_id == 1626655497), None))
    tmp.append(next((u for u in umyos if u.unit_id == 4100972396), None))
    tmp.append(next((u for u in umyos if u.unit_id == 1969998036), None))
    umyos = tmp
    for u in umyos:
        print(u.unit_id)
else:
    raise RuntimeError("No serial data available. Check sensor connection.")

# Full device_spectr per sensor (16 frequency bands)
N_CHANNELS = len(umyos[0].device_spectr)
print(f"EMG: {N_CHANNELS} device_spectr channels per sensor")

# ─── Data storage ─────────────────────────────────────────────────────
DISPLAY_LEN = 1000  # ~5 seconds at 200Hz

xs = deque(maxlen=DISPLAY_LEN)
xssave = []

ys_imu = deque(maxlen=DISPLAY_LEN)
ys_thigh = deque(maxlen=DISPLAY_LEN)  # thigh orientation from uMyo sensor 2
ys_emg = [[deque(maxlen=DISPLAY_LEN) for _ in range(N_CHANNELS)] for _ in range(3)]

yssave_imu = []
yssave_thigh = []  # thigh angle (feature input, NOT label)
yssave_emg = [[[] for _ in range(N_CHANNELS)] for _ in range(3)]

# Raw EMG accumulator (native rate ~400Hz, 8 samples per uMyo packet)
yssave_raw = [[], [], []]     # flat lists of raw int16 samples per sensor
prev_data_ids = [0, 0, 0]     # track data_id to detect new packets

# Quaternion state
a = quat_math.sQ(0, 0, 0, 0)

data_lock = threading.Lock()
parse_unproc_cnt = 0

# ─── Sensor read functions ────────────────────────────────────────────
def drain_umyo():
    """Drain serial buffer to update uMyo sensor state.
    Also accumulates raw EMG samples when new packets arrive."""
    global parse_unproc_cnt, a

    cnt = ser.in_waiting
    if cnt > 0:
        data = ser.read(cnt)
        parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data)

    # Accumulate raw EMG when new data arrives (native rate)
    for s in range(3):
        curr_id = umyos[s].data_id
        if curr_id != prev_data_ids[s]:
            prev_data_ids[s] = curr_id
            dc = umyos[s].data_count
            for i in range(dc):
                yssave_raw[s].append(umyos[s].data_array[i])

    # Update quaternion from sensor 2
    aq = quat_math.sQ(umyos[2].Qsg[0], umyos[2].Qsg[1], umyos[2].Qsg[2], umyos[2].Qsg[3])
    a = quat_math.q_renorm(aq)


def snapshot_emg():
    """Save full device_spectr snapshot from each sensor."""
    ch0 = umyos[0].device_spectr
    ch1 = umyos[1].device_spectr
    ch2 = umyos[2].device_spectr
    for i in range(N_CHANNELS):
        yssave_emg[0][i].append(ch0[i])
        yssave_emg[1][i].append(ch1[i])
        yssave_emg[2][i].append(ch2[i])
        ys_emg[0][i].append(ch0[i])
        ys_emg[1][i].append(ch1[i])
        ys_emg[2][i].append(ch2[i])


def read_imu():
    """Read ALL pending IMU quaternions from Bluetooth buffer.

    Returns list of (knee_angle, thigh_angle) tuples — one per valid packet.
    At 200Hz IMU, a single recv may contain multiple packets.
    """
    results = []
    try:
        data = sock.recv(4096)
        data = data.split(b'U')
        for msg in data:
            q = wit.get_quaternion(msg)
            if q is not None:
                qp = quat_math.sQ(q[0], q[1], q[2], q[3])
                qp = quat_math.q_renorm(qp)

                # Roll from IMU (shin)
                t1 = math.atan2(
                    2.0 * (qp.x * qp.y + qp.w * qp.z),
                    qp.w * qp.w + qp.x * qp.x - qp.y * qp.y - qp.z * qp.z
                )
                # Yaw from uMyo sensor quaternion (thigh)
                t2 = math.atan2(
                    2.0 * (a.y * a.z + a.w * a.x),
                    a.w * a.w - a.x * a.x - a.y * a.y + a.z * a.z
                )
                dt1 = t1 * 180 / math.pi
                if dt1 < 0:
                    dt1 = 180 + (180 + dt1)
                dt2 = t2 * 180 / math.pi
                diff = 180 - (dt1 - dt2)

                results.append((diff, dt2))
        return results
    except socket.timeout:
        return results
    except socket.error as e:
        print(f"Error receiving data: {e}")
        return results


# ─── Main data collection loop ────────────────────────────────────────
start_time = time.time()

# Reduce BT socket timeout so we don't block the loop
sock.settimeout(0.005)  # 5ms — at 200Hz packets arrive every 5ms

def update_time():
    while True:
        with data_lock:
            # 1) Drain serial buffer — updates uMyo sensor data in place
            drain_umyo()

            # 2) Drain BT buffer — get ALL pending IMU quaternions
            imu_readings = read_imu()

            # 3) Save one sample per IMU packet, each with the same EMG
            #    snapshot (uMyo updates at its own rate, we read latest values)
            global_time = time.time() - start_time
            for knee_angle, thigh_angle in imu_readings:
                snapshot_emg()  # saves current EMG state (1 row per IMU packet)
                ys_imu.append(knee_angle)
                yssave_imu.append(knee_angle)
                ys_thigh.append(thigh_angle)
                yssave_thigh.append(thigh_angle)
                xs.append(global_time)
                xssave.append(global_time)

        time.sleep(0.001)  # 1ms — well under 5ms IMU packet interval

threading.Thread(target=update_time, daemon=True).start()

# ─── Plot setup ───────────────────────────────────────────────────────
fig, ax = plt.subplots(4, 1, figsize=(10, 8))

line_imu, = ax[0].plot([], [], label="Knee Angle", color="blue")
ax[0].set_title("IMU Data")
ax[0].set_xlabel("Time (sec)")
ax[0].set_ylabel("Angle (degrees)")
ax[0].set_ylim(0, 200)
ax[0].legend()

lines_emg = []
for i in range(3):
    sensor_lines = [ax[i + 1].plot([], [], linewidth=0.8)[0] for _ in range(N_CHANNELS)]
    lines_emg.append(sensor_lines)
    ax[i + 1].set_title(f"Sensor {i + 1} (device_spectr)")
    ax[i + 1].set_xlabel("Time (sec)")
    ax[i + 1].set_ylabel("Normalized")
    ax[i + 1].set_ylim(-0.05, 1.05)


def animate(frame):
    with data_lock:
        if len(xs) < 2:
            return [line_imu] + [l for s in lines_emg for l in s]

        line_imu.set_data(xs, ys_imu)
        ax[0].set_xlim(xs[0], xs[-1])

        for i, sensor_lines in enumerate(lines_emg):
            for j, line in enumerate(sensor_lines):
                vals = list(ys_emg[i][j])
                if vals:
                    lo = min(vals)
                    hi = max(vals)
                    rng = hi - lo if hi != lo else 1.0
                    normed = [(v - lo) / rng for v in vals]
                    line.set_data(list(xs), normed)
                else:
                    line.set_data([], [])
            ax[i + 1].set_xlim(xs[0], xs[-1])

    return [line_imu] + [l for s in lines_emg for l in s]


ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)

# ─── Save on exit ─────────────────────────────────────────────────────
def done():
    print("Saving data...")
    with data_lock:
        n_samples = len(xssave)
        if n_samples == 0:
            print("No data to save.")
            return

        data_dict = {
            "timestamps": np.array(xssave, dtype=np.float64),
            "imu": np.array(yssave_imu, dtype=np.float64),
            "thigh_angle": np.array(yssave_thigh, dtype=np.float64),  # uMyo sensor 2 orientation (model input feature)
            "emg_sensor1": np.array([np.array(yssave_emg[0][i], dtype=np.float64)
                                      for i in range(N_CHANNELS)]),  # (C, T)
            "emg_sensor2": np.array([np.array(yssave_emg[1][i], dtype=np.float64)
                                      for i in range(N_CHANNELS)]),
            "emg_sensor3": np.array([np.array(yssave_emg[2][i], dtype=np.float64)
                                      for i in range(N_CHANNELS)]),
            # Raw EMG at native rate (~400Hz) for offline feature extraction
            "raw_emg_sensor1": np.array(yssave_raw[0], dtype=np.float64),
            "raw_emg_sensor2": np.array(yssave_raw[1], dtype=np.float64),
            "raw_emg_sensor3": np.array(yssave_raw[2], dtype=np.float64),
            "n_channels": N_CHANNELS,
            "effective_hz": n_samples / (xssave[-1] - xssave[0]) if xssave[-1] > xssave[0] else 0,
        }

        raw_len = len(yssave_raw[0])
        duration = xssave[-1] - xssave[0] if xssave[-1] > xssave[0] else 1
        raw_hz = raw_len / duration

        from pathlib import Path
        file_path = Path("data0.npy")
        datai = 1
        while file_path.exists():
            file_path = Path(f"data{datai}.npy")
            datai += 1
        np.save(file_path, data_dict)

        hz = data_dict["effective_hz"]
        print(f"Saved to {file_path} ({n_samples} samples, {N_CHANNELS} ch/sensor, ~{hz:.1f} Hz effective)")
        print(f"  Raw EMG: {raw_len} samples/sensor (~{raw_hz:.0f} Hz native rate)")

    print("Closing socket...")
    sock.close()

try:
    plt.tight_layout()
    plt.show()
finally:
    done()
