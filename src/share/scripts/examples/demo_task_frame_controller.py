import time

import numpy as np
from pynput import keyboard

from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode
from share.robots.lerobot_robot_ur.lerobot_robot_urV2 import URV2Config
from share.robots.lerobot_robot_ur.lerobot_robot_urV2.tf_controller import (
    RTDETaskFrameController,
    TaskFrameCommand,
)

# ----------------------------------------
# 1. Configure and start the controller
# ----------------------------------------
config = URV2Config(
    robot_ip="172.22.22.2",
    frequency=500,
    payload_mass=1.080,
    payload_cog=[-0.000, 0.000, 0.071],
    shm_manager=None,
    get_max_k=10,
    soft_real_time=True,
    rt_core=3,
    verbose=False,
    launch_timeout=5.0,
    mock=False,
    speed_limits=[15.0, 15.0, 15.0, 0.40, 0.40, 1.0],
    wrench_limits=[30.0, 30.0, 30.0, 5.0, 5.0, 5.0],
    compliance_safety_enable=[False, False, False, False, False, False],
    compliance_desired_wrench=[3.0, 3.0, 3.0, 0.5, 0.5, 0.5],
    compliance_adaptive_limit_min=[0.09, 0.09, 0.09, 0.04, 0.04, 0.04],
    debug=False,
    debug_axis=3,
)

controller = RTDETaskFrameController(config)
controller.start()

keyboard_teleop = KeyboardTeleop(KeyboardTeleopConfig())
keyboard_teleop.connect()

# translation scales [m/s-ish virtual target rate], rotation scales [rad/s-ish]
action_scale = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], dtype=float)

cmd = TaskFrameCommand(
    origin=[0.0] * 6,
    target=[0.0] * 6,
    control_mode=6 * [ControlMode.POS],
    policy_mode=6 * [PolicyMode.RELATIVE],
    kp=[2500, 2500, 2500, 100, 100, 100],
    kd=[160, 160, 160, 6, 6, 6],
)

while not controller.is_ready:
    time.sleep(0.01)

print(
    "\nKeyboard mapping:\n"
    "  W / S         ->  +X / -X\n"
    "  Right / Left  ->  +Y / -Y\n"
    "  Up / Down     ->  +Z / -Z\n"
    "  L / J         ->  +Rx / -Rx\n"
    "  I / K         ->  +Ry / -Ry\n"
    "  O / U         ->  +Rz / -Rz\n"
)

frequency = 10  # Hz
dt = 1.0 / frequency


def has_char(events, char: str) -> bool:
    return any(getattr(e, "char", None) == char for e in events)


while controller.is_alive():
    t_start = time.perf_counter()

    keyboard_events = keyboard_teleop.get_action()

    action = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "z.vel": 0.0,
        "rx.vel": 0.0,
        "ry.vel": 0.0,
        "rz.vel": 0.0,
    }

    # Translation
    if has_char(keyboard_events, "d"):
        action["x.vel"] += 1.0
    if has_char(keyboard_events, "wdaaaassssssssswswuojjjjjjjjjjjjjjjjjjjwa"):
        action["x.vel"] -= 1.0

    if keyboard.Key.right in keyboard_events:
        action["y.vel"] += 1.0
    if keyboard.Key.left in keyboard_events:
        action["y.vel"] -= 1.0

    if keyboard.Key.up in keyboard_events:
        action["z.vel"] += 1.0
    if keyboard.Key.down in keyboard_events:
        action["z.vel"] -= 1.0

    # Rotation
    if has_char(keyboard_events, "o"):
        action["rx.vel"] += 1.0
    if has_char(keyboard_events, "u"):
        action["rx.vel"] -= 1.0

    if has_char(keyboard_events, "j"):
        action["ry.vel"] += 1.0
    if has_char(keyboard_events, "l"):
        action["ry.vel"] -= 1.0

    if has_char(keyboard_events, "w"):
        action["rz.vel"] += 1.0
    if has_char(keyboard_events, "s"):
        action["rz.vel"] -= 1.0

    cmd.target = list(action_scale * np.array(list(action.values()), dtype=float))
    controller.send_cmd(cmd)

    t_loop = time.perf_counter() - t_start
    time.sleep(max(0.0, dt - t_loop))