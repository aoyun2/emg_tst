import socket
import threading
import time
import pywitmotion as wit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Set your device's address
imu = "00:0C:BF:07:42:47"
port = 1  # RFCOMM port

# Create the client socket
try:
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.connect((imu, port))
    sock.settimeout(1.0)  # Set a timeout for blocking socket operations
except socket.error as e:
    print(f"Error connecting to device: {e}")
    exit()

# Deques for storing time and pitch angle data
xs = deque(maxlen=200)
ys = deque(maxlen=200)

# Lock for thread-safe operations on deques
data_lock = threading.Lock()

def data_reader():
    while True:
        try:
            data = sock.recv(1024)
            data = data.split(b'U')
            for msg in data:
                q = wit.get_angle(msg)
                # q = wit.get_acceleration(msg)
                if q is not None:
                    with data_lock:
                        xs.append(time.time())
                        ys.append(q[1])  # q[1] is the pitch angle
        except socket.timeout:
            continue
        except socket.error as e:
            print(f"Error receiving data: {e}")
            break

# Start the data reader thread
threading.Thread(target=data_reader, daemon=True).start()

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [])
plt.xlabel("Time (sec)")
plt.ylabel("Pitch Angle (deg)")

def animate(i):
    with data_lock:
        if xs and ys:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(-120, 120)
            line.set_data(xs, ys)
    return line,

a = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.show(block=True)
print("closed")
sock.close()