import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import keyboard
from collections import deque
import Config

# ================== 涓插彛閰嶇疆 ==================
SENSOR_PORT = 'COM4'
ARDUINO_PORT = 'COM5'
BAUD = 115200

sensor = serial.Serial(SENSOR_PORT, BAUD, timeout=0)
arduino = serial.Serial(ARDUINO_PORT, BAUD, timeout=0.1)

time.sleep(2)
sensor.write(b"100 Hz,1")
time.sleep(2)

print("Sensor & Arduino connected")

# ================== 鍙傛暟 ==================
SAMPLE_RATE = 100
WINDOW = 300
FEATURE_WIN = 100
PLOT_INTERVAL = 0.02
EPS = 1e-6

# ---------- 鑺傛媿鎺у埗锛堟牳蹇冿級 ----------
STEP_INTERVAL = 0.4   # 500 ms
last_step_time = 0

def can_step():
    global last_step_time
    now = time.time()
    if now - last_step_time >= STEP_INTERVAL:
        last_step_time = now
        return True
    return False

# ---------- 瀹為獙鍙傛暟 ----------
FZ_CONTACT = 2.2
FZ_MAX = 10.0

INDENT_MM = 3.0
Z_STEP_MM = 1    # 瑕佸拰arduion閲岄潰鐨凷TEP_MM鐨勫€间竴鏍?
INDENT_STEPS = int(INDENT_MM / Z_STEP_MM)

SLIDE_MM = 20.0
SLIDE_STEPS = int(SLIDE_MM / Z_STEP_MM)

LIFT_EXTRA_MM = 20.0
LIFT_EXTRA_STEPS = int(LIFT_EXTRA_MM / Z_STEP_MM)

# ================== 缂撳啿鍖?==================
# ================== 鏁版嵁缂撳啿 ==================
fx_buf = deque(maxlen=WINDOW)
fy_buf = deque(maxlen=WINDOW)
fz_buf = deque(maxlen=WINDOW)
ft_buf = deque(maxlen=WINDOW)
mu_buf = deque(maxlen=WINDOW)
dfz_buf = deque(maxlen=WINDOW)

# ================== 鐗瑰緛缂撳啿锛堢敤浜庣敾鍥撅級 ==================
k_eff_buf = deque(maxlen=WINDOW)
fz_peak_buf = deque(maxlen=WINDOW)
mu_mean_buf = deque(maxlen=WINDOW)
mu_std_buf = deque(maxlen=WINDOW)
slip_buf = deque(maxlen=WINDOW)
micro_buf = deque(maxlen=WINDOW)

# ================== 鏁版嵁淇濆瓨 ==================
MATERIAL_LABEL = "Material_Predict_6"
os.makedirs(Config.RAW_DIR, exist_ok=True)
# os.makedirs(Config.PREDICTION_RAW_DIR, exist_ok=True)

csv_file = open(os.path.join(Config.RAW_DIR, f"{MATERIAL_LABEL}.csv"), "w", newline="")
# csv_file = open(os.path.join(Config.PREDICTION_RAW_DIR, f"{MATERIAL_LABEL}.csv"), "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "time","Fx","Fy","Fz","Ft","mu",
    "k_eff","mu_mean","mu_std","slip","micro","material"
])

# ================== Matplotlib鐢诲浘 ==================
plt.ion()
fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
axes = axes.flatten()

titles = [
    "Fx (N)", "Fy (N)", "Fz (N)",
    "k_eff = |dFz/dt|", "渭 = Ft/Fz", "std(渭)",
    "Fz_peak", "Slip intensity", "Micro vibration RMS(Ft)"
]

lines = []
for ax, title in zip(axes, titles):
    line, = ax.plot([], [])
    ax.set_title(title)
    ax.grid(True)
    lines.append(line)

axes[0].set_ylim(-25, 25.0)
axes[1].set_ylim(-25, 25.0)
axes[2].set_ylim(0, 15.0)
axes[3].set_ylim(0, 6.0)       # k_eff = |dFz/dt| (N/s)
axes[4].set_ylim(0, 5.0)       # 渭 = Ft/Fz
axes[5].set_ylim(0, 3.0)       # std(渭)
axes[6].set_ylim(0, 12.0)       # Fz_peak
axes[7].set_ylim(0, 10)       # Slip intensity
axes[8].set_ylim(0, 3.0)       # Micro vibration RMS(Ft)

last_plot = time.time()
last_fz = None

# ================== 鎺у埗鐘舵€?==================
state = "idle"
enabled = False
indent_count = 0
slide_count = 0
lift_count = 0

print("System ready.")
print("[s] start  |  [p] pause  |  Ctrl+C exit")

# ================== 涓诲惊鐜?==================
try:
    while True:

        # ---------- 浜哄伐鎺у埗 ----------
        if keyboard.is_pressed('s'):
            enabled = True
            state = "approach"
            indent_count = 0
            slide_count = 0
            lift_count = 0
            print(">>> START")
            time.sleep(0.3)

        if keyboard.is_pressed('p'):
            enabled = False
            state = "idle"
            arduino.write(b'S\n')
            print(">>> PAUSE")
            time.sleep(0.3)

        # ---------- 鎵嬪姩鐐瑰姩鎺у埗锛坕dle 鐘舵€侊級 ----------
        if not enabled and state == "idle":

            if keyboard.is_pressed('d'):
                if can_step():
                    arduino.write(b'D\n')

            elif keyboard.is_pressed('u'):
                if can_step():
                    arduino.write(b'U\n')

            elif keyboard.is_pressed('x'):
                if can_step():
                    arduino.write(b'X\n')

            elif keyboard.is_pressed('z'):   # 缁欎綘涓€涓笉鍐茬獊鐨勫弽鍚戦敭
                if can_step():
                    arduino.write(b'x\n')

            elif keyboard.is_pressed('y'):
                if can_step():
                    arduino.write(b'Y\n')

            elif keyboard.is_pressed('h'):   # Y-
                if can_step():
                    arduino.write(b'y\n')

            elif keyboard.is_pressed('s'):
                arduino.write(b'S\n')


        # ---------- 璇诲彇浼犳劅鍣?----------
        raw = sensor.read(sensor.in_waiting or 1)
        if not raw:
            continue

        for l in raw.split(b'\n'):
            parts = l.strip().split()
            if len(parts) < 5:
                continue

            try:
                fy = float(parts[1])
                fx = float(parts[2])  # 杩欓噷鎴戞妸x鍜寉瀵规崲浜嗭紝鎸夌収鎴戠殑涓夎酱婊戝彴鐨勫潗鏍囩郴鏉?
                fz = float(parts[3])
            except:
                continue

            fx_buf.append(fx)
            fy_buf.append(fy)
            fz_buf.append(fz)

            ft = np.sqrt(fx**2 + fy**2)
            mu = ft / (fz + EPS)

            ft_buf.append(ft)
            mu_buf.append(mu)

            if last_fz is not None:
                dfz_buf.append((fz - last_fz) * SAMPLE_RATE)
            last_fz = fz

            # ---------- 浣嶇Щ鐘舵€佹満锛堜弗鏍艰妭鎷嶏級 ----------
            if not enabled and state != "idle":
                arduino.write(b'S\n')


            elif state == "approach":
                if can_step():
                    arduino.write(b'D\n')
                if fz > FZ_CONTACT:
                    state = "indent"
                    indent_count = 0
                    print("Contact detected")

            elif state == "indent":
                if fz > FZ_MAX:
                    enabled = False
                    arduino.write(b'S\n')
                    print("!!! FORCE LIMIT STOP")
                elif indent_count < INDENT_STEPS:
                    if can_step():
                        arduino.write(b'D\n')
                        indent_count += 1
                else:
                    arduino.write(b'S\n')
                    state = "hold_z"
                    print("Indent complete")

            elif state == "hold_z":
                slide_count = 0
                state = "slide_x"
                print("Start friction path")

            elif state == "slide_x":
                if slide_count < SLIDE_STEPS:
                    if can_step():
                        arduino.write(b'X\n')
                        slide_count += 1
                else:
                    slide_count = 0
                    state = "slide_y"

            elif state == "slide_y":
                if slide_count < SLIDE_STEPS:
                    if can_step():
                        arduino.write(b'y\n')
                        slide_count += 1
                else:
                    slide_count = 0
                    state = "slide_back"

            elif state == "slide_back":
                if slide_count < SLIDE_STEPS:
                    if can_step():
                        arduino.write(b'x\n')
                        arduino.write(b'Y\n')
                        slide_count += 1
                else:
                    lift_count = 0
                    state = "lift_off"
                    print("Friction path done")

            elif state == "lift_off":
                if can_step():
                    arduino.write(b'U\n')
                if fz < FZ_CONTACT:
                    state = "retreat"
                    lift_count = 0

            elif state == "retreat":
                if lift_count < LIFT_EXTRA_STEPS:
                    if can_step():
                        arduino.write(b'U\n')
                        lift_count += 1
                else:
                    arduino.write(b'S\n')
                    enabled = False
                    state = "idle"
                    print("Cycle complete")

            # ---------- 鐗瑰緛 ----------
            if len(fz_buf) >= FEATURE_WIN:
                fz_win = np.array(list(fz_buf)[-FEATURE_WIN:])
                ft_win = np.array(list(ft_buf)[-FEATURE_WIN:])
                mu_win = np.array(list(mu_buf)[-FEATURE_WIN:])
                dfz_win = np.array(list(dfz_buf)[-FEATURE_WIN:])

                k_eff = np.mean(np.abs(dfz_win))
                fz_peak = np.max(fz_win)
                mu_mean = np.mean(mu_win)
                mu_std = np.std(mu_win)
                slip = np.max(np.abs(dfz_win))
                micro = np.sqrt(np.mean((ft_win - np.mean(ft_win))**2))

                csv_writer.writerow([
                    time.time(), fx, fy, fz, ft, mu,
                    k_eff, mu_mean, mu_std, slip, micro,
                    MATERIAL_LABEL
                ])
                csv_file.flush()

                k_eff_buf.append(k_eff)
                fz_peak_buf.append(fz_peak)
                mu_mean_buf.append(mu_mean)
                mu_std_buf.append(mu_std)
                slip_buf.append(slip)
                micro_buf.append(micro)

        # ---------- 鐢诲浘 ----------
        if time.time() - last_plot > PLOT_INTERVAL:
            x = range(len(fx_buf))
            plot_data = [
                fx_buf, fy_buf, fz_buf,
                k_eff_buf, mu_mean_buf, mu_std_buf,
                fz_peak_buf, slip_buf, micro_buf
            ]
            for line, data in zip(lines, plot_data):
                line.set_data(x, data)
            for ax in axes:
                ax.set_xlim(0, WINDOW)
            plt.pause(0.001)
            last_plot = time.time()

except KeyboardInterrupt:
    print("Stopped")

finally:
    arduino.write(b'S\n')
    sensor.close()
    arduino.close()
    csv_file.close()
    plt.ioff()
    plt.show()

