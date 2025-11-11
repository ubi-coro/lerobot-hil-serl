import time

import numpy as np

from experiments import UR5eBimanualPolytecEnvConfig
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.robotiq_controller import RTDERobotiqController
from lerobot.robots.ur.tf_controller import RTDETFFController, TaskFrameCommand, AxisMode
from lerobot.teleoperators.spacemouse import SpacemouseConfig, SpaceMouse


# ----------------------------------------
# 1. Configure and start the controller
# ----------------------------------------
env_cfg = UR5eBimanualPolytecEnvConfig()
robot_name = "right"

# Instantiate and start the controller (in its own process)
robot_cfg = env_cfg.robot[robot_name]
controller = RTDETFFController(robot_cfg)
controller.start()  # non-blocking

use_gripper = env_cfg.processor.gripper.use_gripper[robot_name]
if use_gripper:
    gripper = RTDERobotiqController(
        hostname=robot_cfg.robot_ip,
        frequency=robot_cfg.gripper_frequency,
        soft_real_time=robot_cfg.gripper_soft_real_time,
        rt_core=robot_cfg.gripper_rt_core,
        verbose=robot_cfg.verbose
    )
    gripper.connect()

spacemouse_expert = SpaceMouse(SpacemouseConfig())
spacemouse_expert.connect()
action_scale = [1.0] * 6

# setup tff command
cmd = env_cfg.processor.task_frame.command[robot_name]

# Wait until the controller signals ready
while not controller.is_ready:
    time.sleep(0.01)

# this is really important
controller.zero_ft()

frequency = 10  # Hz

waiting_for_release = False

initial_x = cmd.min_pose_rpy[0]

while controller.is_alive():
    t_start = time.perf_counter()

    action = spacemouse_expert.get_action()
    if action["gripper.pos"] > 0.5:
        if not waiting_for_release:
            print("Tighten bound")
            cmd.min_pose_rpy[0] += 0.005
            waiting_for_release = True
    else:
        waiting_for_release = False

    for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
        if cmd.mode[i] != AxisMode.POS:
            cmd.target[i] = action[f"{ax}.vel"] * action_scale[i]

    controller.send_cmd(cmd)

    if use_gripper:
        gripper_pos = (1 - env_cfg.max_gripper_pos) + env_cfg.max_gripper_pos * action["gripper.pos"]
        gripper.move(gripper_pos, vel=robot_cfg.gripper_vel, force=robot_cfg.gripper_force)

    ee_pose = controller.get_robot_state()["ActualTCPPose"][:]
    print(f"x-limit: {cmd.min_pose_rpy[0]:.4f}", "".join([f"{p:.4f}, " for p in ee_pose]))
    #print("EE Wrench:", controller.get_robot_state()["ActualTCPForce"][:])

    t_loop = time.perf_counter() - t_start
    time.sleep(max([0.0, 1 / frequency - t_loop]))