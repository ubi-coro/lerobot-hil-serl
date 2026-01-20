
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

import logging
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.sim.configs import SimViperXConfig
from lerobot.sim.simrobot import SimRobot
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

HORN_RADIUS = 0.022
ARM_LENGTH = 0.036


def gripper_to_linear(gripper_pos):
    a1 = HORN_RADIUS * math.sin(gripper_pos)
    c = math.sqrt(pow(HORN_RADIUS, 2) - pow(a1, 2))
    a2 = math.sqrt(pow(ARM_LENGTH, 2) - pow(c, 2))
    return a1 + a2


def linear_to_gripper(linear_position):
    result = math.pi / 2.0 - math.acos(
        (pow(HORN_RADIUS, 2) + pow(linear_position, 2) - pow(ARM_LENGTH, 2))
        / (2 * HORN_RADIUS * linear_position)
    )
    return result


class SimViperX(SimRobot):
    """
    [ViperX](https://www.trossenrobotics.com/viperx-300) developed by Trossen Robotics
    """

    config_class = SimViperXConfig
    name = "simviperx"

    def __init__(
        self,
        config: SimViperXConfig,
    ):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_motor_obs = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors = {}
        return motors

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self.get_observation()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return


    def configure(self) -> None:
        """
        Read current motor registers (with torque ON). If all match our desired
        configuration, skip the torque-off writes. Otherwise, torque-off and apply.
        """

        logger.info(f"{self}: applying motor configuration (requires torque-off).")


    def get_observation(self) -> dict[str, Any]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = {}
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items() if not motor.endswith("_shadow")}
        #obs_dict["finger.pos"] = gripper_to_linear(obs_dict.pop("gripper.pos"))
        dt_ms = (time.perf_counter() - start) * 1e3
        self._last_motor_obs = dict(obs_dict)
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if "finger.pos" in action:
            # Convert finger position to gripper position
            action = action.copy()  # Avoid modifying the original action
            action["gripper.pos"] = linear_to_gripper(action["finger.pos"])
            del action["finger.pos"]

        goal_pos = {key: action.get(key, self._last_motor_obs[key]) for key in self._last_motor_obs}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, self._last_motor_obs[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        goal_pos = {key.removesuffix(".pos"): value for key, value in goal_pos.items()}

        # Send goal position to the arm
        #TODO(jzilke) send action to sim
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")



    def _settings_match(self, current: dict[str, dict[str, int]], desired: dict[str, dict[str, int]]) -> bool:
        """
        Compare current vs desired. For Drive_Mode, require bit2 set (mask check).
        For others, require exact equality.
        """

        return True
