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
import logging
import time
from functools import cached_property
from multiprocessing.managers import SharedMemoryManager
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.rtde_robotiq_controller import RTDERobotiqController
from lerobot.robots.ur.tff_controller import TaskFrameCommand, RTDETFFController, AxisMode
from lerobot.robots.ur.tff_mock_controller import RTDETFFMockController
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError

logger = logging.getLogger(__name__)


class TF_UR(Robot):

    config_class = URConfig
    name = "ur"

    joint_names = state_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, config: URConfig):
        self.config = config
        self.robot_type = self.name
        self.id = config.id
        self.task_frame = TaskFrameCommand.make_default_cmd()

        self.shm = SharedMemoryManager()
        self.shm.start()
        config.shm_manager = self.shm

        if config.mock:
            self.controller = RTDETFFMockController(config)
        else:
            self.controller = RTDETFFController(config)

        if self.config.use_gripper:
            self.gripper = RTDERobotiqController(
                hostname=config.robot_ip,
                shm_manager=self.shm,
                frequency=config.gripper_frequency,
                soft_real_time=config.gripper_soft_real_time,
                rt_core=config.gripper_rt_core,
                verbose=config.verbose
            )
        else:
            self.gripper = None

        self.cameras = make_cameras_from_configs(config.cameras)

        # runtime vars
        self.logs = {}
        self.last_robot_action = TaskFrameCommand()

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {}
        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            ft[f"{ax}.ee_pos"] = float
            ft[f"{ax}.ee_vel"] = float
            ft[f"{ax}.ee_wrench"] = float

        for i, joint_name in enumerate(self.joint_names):
            ft[f"{joint_name}.q_pos"] = float

        if self.gripper is not None:
            ft["gripper.pos"] = float

        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        ft = self.task_frame.to_robot_action()

        if self.gripper is not None:
            ft["gripper.pos"] = float

        return ft

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        _is_connected = self.controller.is_ready
        if self.gripper is not None:
            _is_connected &= self.gripper.is_ready
        return _is_connected

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.controller.start()
        self.controller.zero_ft()
        if self.gripper is not None:
            self.gripper.start()

        for name in self.cameras:
            self.cameras[name].connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        return None

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_dict = {}
        controller_data = self.controller.get_robot_state()

        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            obs_dict[f"{ax}.ee_pos"] = controller_data['ActualTCPPose'][i]
            obs_dict[f"{ax}.ee_vel"] = controller_data['ActualTCPSpeed'][i]
            obs_dict[f"{ax}.ee_wrench"] = controller_data['ActualTCPForce'][i]

        for i, joint_name in enumerate(self.joint_names):
            obs_dict[f"{joint_name}.q_pos"] = controller_data['ActualQ'][i]

        if self.gripper is not None:
            obs_dict["gripper.pos"] = float(self.gripper.get_state()["width"]) / 255.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Pass task-frame command to the low-level controller.



        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            if f"{ax}.pos" in action:
                self.task_frame.target[i] = action[f"{ax}.pos"]
                self.task_frame.mode[i] = AxisMode.POS
            elif f"{ax}.vel" in action:
                self.task_frame.target[i] = action[f"{ax}.vel"]
                self.task_frame.mode[i] = AxisMode.PURE_VEL
            elif f"{ax}.wrench" in action:
                self.task_frame.target[i] = action[f"{ax}.wrench"]
                self.task_frame.mode[i] = AxisMode.FORCE

        if self.gripper is not None and f"{GRIPPER_KEY}.pos" in action:
            self.send_gripper_action(action[f"{GRIPPER_KEY}.pos"])

        self.controller.send_cmd(self.task_frame)

        return action

    def send_gripper_action(self, gripper_action: float):
        self.gripper.move(gripper_action, vel=self.config.gripper_vel, force=self.config.gripper_force)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.controller.stop()
        for cam in self.cameras.values():
            cam.disconnect()
        self.shm.shutdown()

        logger.info(f"{self} disconnected.")
