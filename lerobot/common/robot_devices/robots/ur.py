# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.motors.rtde_tff_controller import RTDETFFController, TaskFrameCommand
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class UR:
    def __init__(
        self,
        config: URConfig,
    ):
        self.config = config
        self.robot_type = self.config.type

        # -------------------------- shared memory ------------------------ #
        self.shm = SharedMemoryManager()
        self.shm.start()

        # -------------------------- URs ------------------------- #
        self.controllers = {}
        for name, follower_config in self.config.follower_arms.items():
            follower_config.shm_manager = self.shm
            controller = RTDETFFController(follower_config)
            self.controllers[name] = controller

        # -------------------------- dynamixel & cameras ------------------------- #
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)

        # runtime vars
        self.is_connected = False
        self.logs = {}
        self._last_pose = None
        self._step_count = 0

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.image.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        features = dict()

        # state observations
        example_obs = self.capture_observation()
        for key, value in example_obs.items():
            if "image" in key:
                continue

            if key.endswith("q_pos"):
                state_names = [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ]
            elif key.endswith("eef_pos") and len(value) == 7:
                    state_names = ["x", "y", "z", "a", "b", "c", "gripper"]
            else:
                state_names = ["x", "y", "z", "a", "b", "c"]

            assert len(state_names) == len(value)
            features[key] = {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names
            }

        # action
        action_names = []
        for name, controller in self.controllers.items():
            if getattr(controller.config, 'use_gripper', False):
                action_names.extend([f"{name}_{ax}" for ax in ["x", "y", "z", "a", "b", "c", "gripper"]])
            else:
                action_names.extend([f"{name}_{ax}" for ax in ["x", "y", "z", "a", "b", "c"]])

        features["action"] = {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        }

        return features

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.controllers and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.controllers:
            logging.info(f"Connecting {name} follower arm.")
            self.controllers[name].start()
        for name in self.leader_arms:
            logging.info(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        self.activate_calibration()

        # Check if all components can be read
        for name in self.controllers:
            self.controllers[name].get_robot_state()
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                logging.warn(f"Missing calibration file '{arm_calib_path}'")

                if not all(arm.read("Torque_Enable") == TorqueMode.DISABLED.value):
                    input(f"Press <enter> to disable the torque of {self.robot_type} {name} {arm_type}... ")
                    arm.write("Torque_Enable", TorqueMode.DISABLED.value)

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss", "lekiwi"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pass

    def capture_observation(self):
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        The returned observations do not have a batch dimension.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        obs = {}

        # Capture images from cameras
        for name in self.cameras:
            img = self.cameras[name].async_read()
            obs[f"observation.image.{name}"] = torch.from_numpy(img)

        # both have more than n_obs_steps data
        for name, controller in self.controllers.items():
            controller_data = controller.get_robot_state()

            if getattr(controller.config, 'use_gripper', False):
                gripper_pos = controller.get_robot_state()["width_mm"]
                obs[f'observation.{name}_eef_pos'] = np.concatenate([controller_data['ActualTCPPose'], [gripper_pos]])
            else:
                obs[f'observation.{name}_eef_pos'] = controller_data['ActualTCPPose']

            obs[f'observation.{name}_eef_pos'] = torch.from_numpy(obs[f'observation.{name}_eef_pos'])
            obs[f'observation.{name}_eef_speed'] = torch.from_numpy(controller_data['ActualTCPSpeed'])
            obs[f'observation.{name}_eef_wrench'] = torch.from_numpy(controller_data['ActualTCPForce'])
            obs[f'observation.{name}_q_pos'] = torch.from_numpy(controller_data['ActualQ'])

        return obs

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command follower arms (and gripper if used) to targets.

        For each controller, if `controller.config.use_gripper` is True, expects 7 values: 6-joint targets + 1 gripper width (m).
        Otherwise expects 6 values: joint targets only.
        Returns the action actually sent.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        arr = action.detach().cpu().numpy().ravel()
        idx = 0
        total_len = arr.size
        # Compute expected length
        expected = sum((7 if r.config.use_gripper else 6) for r in self.controllers.values())
        if total_len != expected:
            raise ValueError(f"Action length ({total_len}) does not match expected ({expected}) for current controllers.")

        for name, controller in self.controllers.items():
            length = 7 if getattr(controller.config, 'use_gripper', False) else 6
            seg = arr[idx: idx + length]
            idx += length

            # first 6 values are robot targets
            cmd = TaskFrameCommand(target=seg[:6])
            controller.send_cmd(cmd)

            # if gripper flag, last element is width in metres
            if length == 7:
                width_mm = float(seg[6])
                # assumes controller.gripper exists
                controller.gripper.move(width_mm, wait=False)
        return action


    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.controllers:
            self.controllers[name].stop()

        self.shm.shutdown()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
