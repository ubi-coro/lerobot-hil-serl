import time
import matplotlib.pyplot as plt
from collections import deque

import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import RTDETFFController, TaskFrameCommand, \
    Command, GripperCommand, AxisMode
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseExpert

USE_ROT = True

# ----------------------------------------
# 1. Configure and start the controller
# ----------------------------------------
config = URArmConfig(
    robot_ip="172.22.22.2",
    frequency=500,
    gain=300,
    payload_mass=None,
    payload_cog=None,
    shm_manager=None,
    receive_keys=None,
    get_max_k=10,
    soft_real_time=True,
    rt_core=3,
    verbose=False,
    launch_timeout=5.0,
    mock=False,
    use_gripper=True,
    wrench_limits=[0.40, 0.40, 0.40, 0.4, 0.4, 4.0]
)

# Instantiate and start the controller (in its own process)
controller = RTDETFFController(config)
controller.start()  # non-blocking

spacemouse_expert = SpaceMouseExpert()
action_scale = np.array([1 / 10] * 3 + [1.0] * 3)

# setup tff command
if USE_ROT:
    cmd = TaskFrameCommand(
        T_WF=[0.1811, -0.3745, 0.11, 2.221, -2.221, 0.0],
        target=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mode=6 * [AxisMode.IMPEDANCE_VEL],
        kp=np.array([2500, 2500, 2500, 100, 100, 100]),
        kd=np.array([160, 160, 320, 6, 6, 6])
    )
else:
    cmd = TaskFrameCommand(
        T_WF=[0.1811, -0.3745, 0.11, 2.221, -2.221, 0.0],
        target=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mode=6 * [AxisMode.IMPEDANCE_VEL],
        kp=np.array([2500, 2500, 2500, 100, 100, 100]),
        kd=np.array([160, 160, 320, 6, 6, 6])
    )

# Wait until the controller signals ready
while not controller.is_ready:
    time.sleep(0.01)

frequency = 10  # Hz
while controller.is_alive():
    t_start = time.perf_counter()

    action, buttons = spacemouse_expert.get_action()

    action = action_scale * action

    if USE_ROT:
        cmd.target = action
    else:
        cmd.target[0] = -action[0]
        cmd.target[1] = action[1]
        cmd.target[2] = -action[2]

    controller.send_cmd(cmd)

    if buttons[0]:
        controller.send_cmd(GripperCommand(cmd=Command.OPEN))
    if buttons[1]:
        controller.send_cmd(GripperCommand(cmd=Command.CLOSE))

    print("EE Pose:", controller.get_robot_state()["ActualTCPPose"][:3])

    t_loop = time.perf_counter() - t_start
    time.sleep(1 / frequency - t_loop)

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

# ----------------------------------------
# 5. Animation update function
# ----------------------------------------
def update(frame):
    # Send a new random velocity command every 10 frames (≈1 second at 100 ms interval)

    # Read the latest TCP pose from the controller
    state = controller.get_state(k=1)
    pose = state['ActualTCPPose'][0]  # [x, y, z, Rx, Ry, Rz]
    x, y, z = pose[:3]
    print(x, y, z)

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

    #ax_ts.set_ylim(
    #    min([min(x), min(y), min(z)]),
    #    max([max(x), max(y), max(z)])
    #)

    ax_3d.set_xlim(min(xs), max(xs))
    ax_3d.set_ylim(min(ys), max(ys))
    ax_3d.set_zlim(min(zs), max(zs))

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
