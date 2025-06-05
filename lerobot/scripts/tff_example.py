import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import RTDETFFController, AxisMode, TaskFrameCommand, Command

# ----------------------------------------
# 1. Configure and start the controller
# ----------------------------------------
config = URArmConfig(
    robot_ip="mock_hostname",
    frequency=125,
    gain=300,
    max_pos_speed=0.5,
    max_rot_speed=1.0,
    payload_mass=None,
    payload_cog=None,
    joints_init=None,
    joints_init_speed=1.0,
    shm_manager=None,
    receive_keys=None,
    get_max_k=10,
    soft_real_time=False,
    rt_core=3,
    verbose=False,
    launch_timeout=5.0,
    mock=True
)

# Instantiate and start the controller (in its own process)
controller = RTDETFFController(config)
controller.start()  # non-blocking

# Wait until the controller signals ready
while not controller.is_ready:
    time.sleep(0.01)

# ----------------------------------------
# 2. Prepare data structures for logging
# ----------------------------------------
max_len = 200
times = deque(maxlen=max_len)
xs = deque(maxlen=max_len)
ys = deque(maxlen=max_len)
zs = deque(maxlen=max_len)

traj_x = []
traj_y = []
traj_z = []

start_time = time.time()

# ----------------------------------------
# 3. Set up the matplotlib figure
# ----------------------------------------
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 1)

# Upper subplot: time‐series for X, Y, Z
ax_ts = fig.add_subplot(gs[0, 0])
line_x, = ax_ts.plot([], [], label='X', color='r')
line_y, = ax_ts.plot([], [], label='Y', color='g')
line_z, = ax_ts.plot([], [], label='Z', color='b')
ax_ts.set_xlim(0, 10)
ax_ts.set_ylim(-1, 1)
ax_ts.set_ylabel('Position (m)')
ax_ts.legend(loc='upper right')

# Lower subplot: 3D scatter
ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
scatter = ax_3d.scatter([], [], [], c='k', s=5)
ax_3d.set_xlim(-1, 1)
ax_3d.set_ylim(-1, 1)
ax_3d.set_zlim(-1, 1)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

# ----------------------------------------
# 4. Function to send random velocity commands
# ----------------------------------------
def send_random_vel_command():
    # Random velocities in [-0.2, 0.2] m/s for x, y, z
    vx, vy, vz = np.random.uniform(-0.2, 0.2, size=3)
    # Build TaskFrameCommand: velocity in x,y,z, FIXED rotation
    mode = [AxisMode.IMPEDANCE_VEL]*3 + [AxisMode.POS]*3
    target = np.array([vx, vy, vz, 0.0, 0.0, 0.0])
    kp = np.array([100.0, 100.0, 100.0, 300.0, 300.0, 300.0])
    kd = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    T_WF = np.eye(4)
    cmd = TaskFrameCommand(
        cmd=Command.TF_SET,
        T_WF=T_WF,
        mode=mode,
        target=target,
        kp=kp,
        kd=kd
    )
    controller.send_cmd(cmd)

# ----------------------------------------
# 5. Animation update function
# ----------------------------------------
def update(frame):
    # Send a new random velocity command every 10 frames (≈1 second at 100 ms interval)
    if frame % 10 == 0:
        send_random_vel_command()

    # Read the latest TCP pose from the controller
    state = controller.get_state(k=1)
    pose = state['ActualTCPPose'][0]  # [x, y, z, Rx, Ry, Rz]
    x, y, z = pose[:3]

    # Timestamp
    t = time.time() - start_time
    times.append(t)
    xs.append(x)
    ys.append(y)
    zs.append(z)
    traj_x.append(x)
    traj_y.append(y)
    traj_z.append(z)

    # Update time‐series lines
    line_x.set_data(times, xs)
    line_y.set_data(times, ys)
    line_z.set_data(times, zs)
    # Extend x‐axis if needed
    if t > ax_ts.get_xlim()[1]:
        ax_ts.set_xlim(0, t + 1)

    # Update 3D scatter
    scatter._offsets3d = (traj_x, traj_y, traj_z)

    return line_x, line_y, line_z, scatter

# ----------------------------------------
# 6. Create and start the animation
# ----------------------------------------
ani = FuncAnimation(fig, update, interval=100, blit=False)

# Display the interactive plot (blocks until closed)
plt.show()

# ----------------------------------------
# 7. Cleanup: stop the controller after window closes
# ----------------------------------------
controller.stop()
