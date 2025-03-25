import hydra
from omegaconf import DictConfig
import numpy as np
from pynput import keyboard
import threading
import time

import argparse
import sys
import threading
from collections import deque
from typing import Type

import matplotlib.pyplot as plt
import logging
import time
from scipy.spatial.transform import Rotation as R
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
from tests.utils import robot_type

logging.basicConfig(level=logging.INFO)

# === Keyboard control logic ===
class KeyboardTeleop:
    def __init__(self, robot, fps, step, kinematic):
        self.robot = robot
        self.fps = fps
        self.step = step
        self.kinematic = kinematic
        self.lock = threading.Lock()
        self.running = True
        self.desired_ee_pos = kinematic.fk_gripper_tip(robot.follower_arms["main"].read("Present_Position"))

        # Active motion vector
        self.motion_vector = np.zeros(3)
        self.active_keys = set()

        # Start main control loop
        self.control_thread = threading.Thread(target=self._run_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        # Keyboard listeners
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        listener.start()

    def _on_press(self, key):
        with self.lock:
            self.active_keys.add(key)

    def _on_release(self, key):
        with self.lock:
            if key in self.active_keys:
                self.active_keys.remove(key)
            if key == keyboard.Key.esc:
                self.running = False
                print("[Teleop] Exiting...")

    def _compute_motion_vector(self):
        delta = np.zeros(3)
        for key in self.active_keys:
            try:
                if key.char == 'w':
                    delta[1] += 1
                elif key.char == 's':
                    delta[1] -= 1
                elif key.char == 'a':
                    delta[0] -= 1
                elif key.char == 'd':
                    delta[0] += 1
            except AttributeError:
                if key == keyboard.Key.shift_l:
                    delta[2] += 1
                elif key == keyboard.Key.shift_r:
                    delta[2] -= 1
        return delta

    def _run_loop(self):
        rate = 1.0 / self.fps
        while self.running:
            start_loop_t = time.perf_counter()
            with self.lock:
                direction = self._compute_motion_vector()
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)

                    self.desired_ee_pos[:3, 3] += direction * self.step
                    target_joint_pos = self.kinematic.ik(
                        self.robot.follower_arms["main"].read("Present_Position"),
                        self.desired_ee_pos,
                        position_only=False
                    )
                    self.robot.send_action(torch.from_numpy(target_joint_pos))
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / self.fps - dt_s)


def main():
    from scipy.spatial.transform import Rotation as R
    reset_joint_position = np.array([-0.08789062, -21.18164, -21.00586, 62.666016, 62.578125, -0.87890625, -39.55078, .08789062, 0])
    fps = 50
    step = 0.005
    kinematic = MRKinematics("vx300s-bota")

    robot_cfg = init_hydra_config("../../configs/robot/aloha-mr.yaml")
    robot = make_robot(robot_cfg)
    robot.connect()

    # increase P gain to improve tracking accuracy
    robot.follower_arms["main"].write("Position_I_Gain", [50] * 9)

    # reset to desired ee pose
    T_sb = kinematic.fk_gripper_tip(reset_joint_position)
    T_sb[:3, :3] = R.from_euler('y', 80, degrees=True).as_matrix()
    reset_joint_position = kinematic.ik(reset_joint_position, T_sb)
    reset_follower_position(robot, reset_joint_position)

    print("[Teleop] Press W/A/S/D for XY control, Shift for Z-axis, R to reset, ESC to exit.")
    def on_key(key):
        if key == keyboard.KeyCode.from_char('r'):
            reset_follower_position(robot, reset_joint_position)

    # Start a second listener for 'r' key (reset)
    reset_listener = keyboard.Listener(on_press=on_key)
    reset_listener.start()

    KeyboardTeleop(robot, step=step, fps=fps, kinematic=kinematic)

    # Keep the script alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Stopped.")

if __name__ == "__main__":
    main()

