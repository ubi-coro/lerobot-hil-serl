import time
import matplotlib.pyplot as plt
from collections import deque

import numpy as np
from jedi.debug import speed
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import RTDETFFController, TaskFrameCommand, AxisMode
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseExpert

USE_ROT = False

# ----------------------------------------
# 1. Configure and start the controller
# ----------------------------------------
config = URArmConfig(
    robot_ip="172.22.22.2",
    frequency=500,
    payload_mass=1.080,
    payload_cog=[-0.000, 0.000, 0.071],
    shm_manager=None,
    receive_keys=None,
    get_max_k=10,
    soft_real_time=True,
    rt_core=3,
    verbose=False,
    launch_timeout=5.0,
    mock=False,
    use_gripper=True,
    speed_limits=[15.0, 15.0, 15.0, 0.40, 0.40, 1.0],
    wrench_limits=[30.0, 30.0, 30.0, 15.0, 15.0, 10.0],
    enable_contact_aware_force_scaling=[True, True, False, False, False, True],
    contact_desired_wrench=[3.0, 3.0, 0, 0, 0, 0.5],
    contact_limit_scale_min=[0.09, 0.09, 0, 0, 0, 0.04],
    debug=False,
    debug_axis=5
)

# Instantiate and start the controller (in its own process)
controller = RTDETFFController(config)
controller.start()  # non-blocking

spacemouse_expert = SpaceMouseExpert()
action_scale = np.array([1 / 10] * 3 + [1.0] * 3)

# setup tff command
if USE_ROT:
    cmd = TaskFrameCommand(
        T_WF=[0.03383, -0.25478, 0.138, float(np.pi), 0.0, 0.0],
        target=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mode=3 * [AxisMode.IMPEDANCE_VEL],
        kp=np.array([2500, 2500, 2500, 100, 100, 100]),
        kd=np.array([160, 160, 320, 6, 6, 6])
    )
else:
    cmd = TaskFrameCommand(
        T_WF=[0.03383, -0.25478, 0.138, float(np.pi), 0.0, 0.0],
        target=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mode=2 * [AxisMode.PURE_VEL] + 1 * [AxisMode.IMPEDANCE_VEL] + 2 * [AxisMode.POS] + 1 * [AxisMode.PURE_VEL],
        kp=np.array([2500, 2500, 2500, 100, 100, 100]),
        kd=np.array([960, 960, 320, 6, 6, 6])
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
        cmd.target[0] = -action[0]
        cmd.target[1] = action[1]
        cmd.target[2] = -action[2]
        cmd.target[3] = 0.2 * -action[3]
        cmd.target[4] = 0.2 * action[4]
        cmd.target[5] = 0.2 * -action[5]
    else:
        cmd.target[0] = 0.5 * action[0]
        cmd.target[1] = 0.5 * -action[1]
        cmd.target[2] = 0.5 * -action[2]
        cmd.target[5] = 1.0 * -action[5]

    controller.send_cmd(cmd)

    print("EE Pose:", controller.get_robot_state()["ActualTCPPose"][:])
    #print("EE Wrench:", controller.get_robot_state()["ActualTCPForce"][:])

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
