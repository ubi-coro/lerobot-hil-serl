import argparse
import sys
import threading
from collections import deque
from typing import Type

import matplotlib.pyplot as plt
import logging
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from threading import Lock
from typing import Annotated, Any, Dict, Tuple
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, log_say

from lerobot.scripts.server.kinematics import MRKinematics, RobotKinematics

robot_cfg = init_hydra_config("../../configs/robot/aloha-mr.yaml")
robot = make_robot(robot_cfg)
robot.connect()
robot.follower_arms["main"].write("Position_I_Gain", [50] * 9)

# Define interpolation parameters
n_steps = 250
trajectory_dt = 1.0 / 50
interpolation_time = 2.0  # seconds
alpha = 1.0 / n_steps

reset_joint_position = np.array([-0.08789062, -21.18164, -21.00586, 62.666016, 62.578125, -0.87890625, -39.55078, .08789062, 0])

# Get robot + kinematics
kinematics = MRKinematics(robot.config.follower_model)
fk = kinematics.fk_gripper_tip
ik = kinematics.ik

# Read current joint state and compute FK pose
initial_pose = fk(reset_joint_position)
initial_orientation = R.from_matrix(initial_pose[:3, :3])
reset_follower_position(robot, reset_joint_position)

# Define target orientation as rotation around z by 90 degrees
target_orientation = R.from_euler('y', 100, degrees=True)

# Interpolate orientations using SLERP
# Define the interpolation times and rotations
key_times = [0, 1]
key_rots = R.concatenate([initial_orientation, target_orientation])

# Create the SLERP interpolator
slerp = Slerp(key_times, key_rots)

# Query interpolated rotations at evenly spaced times
interp_rots = slerp(np.linspace(0, 1, n_steps))

# Fixed EE position (no translation)
fixed_position = initial_pose[:3, 3]

# Execute motion
for i in range(n_steps):
    rot_matrix = interp_rots[i].as_matrix()

    # Build full 4x4 pose matrix
    target_pose = np.eye(4)
    target_pose[:3, :3] = rot_matrix
    target_pose[:3, 3] = fixed_position

    # Compute joint solution
    current_joint_pos = robot.follower_arms["main"].read("Present_Position")
    joint_target = ik(
        current_joint_pos,
        target_pose,
        position_only=False,
        fk_func=fk,
    )

    # Send action
    robot.send_action(torch.tensor(joint_target))

    # Busy wait to control timing
    time.sleep(trajectory_dt)
