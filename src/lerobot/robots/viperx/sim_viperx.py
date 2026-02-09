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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.robots.viperx import SimViperXConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

class SimViperX(Robot):
    config_class = SimViperXConfig
    name = "sim_viperx"

    def __init__(
        self,
        config: SimViperXConfig,
    ):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_motor_obs = None
        self.sim = SimManager.get()

        self.joint_names = ["waist","shoulder","elbow","forearm_roll","wrist_angle","wrist_rotate","gripper"]

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors = {f"{joint}.pos": float for joint in self.joint_names}
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
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        # self.get_observation()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True # TODO(jzilke)

    def calibrate(self) -> None:
        pass # TODO(jzilke)

    def configure(self) -> None:
        """
        Read current motor registers (with torque ON). If all match our desired
        configuration, skip the torque-off writes. Otherwise, torque-off and apply.
        """
        pass # TODO(jzilke)

    def get_observation(self) -> dict[str, Any]:
        """The returned observations do not have a batch dimension."""

        obs_dict = { f"{joint_name}.pos": self.sim.get_observation(f"{self.id}.{joint_name}.pos") for joint_name in self.joint_names}
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
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

        goal_pos = {key: action.get(key, self._last_motor_obs[key]) for key in self._last_motor_obs}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, self._last_motor_obs[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.sim.add_action(self.id, goal_pos)
        return goal_pos

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _read_current_motor_settings(self) -> dict[str, dict[str, int]]:
        """
        Snapshot current settings while torque is ON.
        Returns a dict: {setting_name: {motor_name: value}} for fast comparisons.
        """
        snap: dict[str, dict[str, int]] = {}
        # TODO(jzilke)
        return snap

    def _desired_motor_settings(self) -> dict[str, dict[str, int]]:
        """
        Compute desired settings exactly as configure() would apply.
        NOTE: For Drive_Mode, we only *require* that bit2 (time-based profiles) is set.
        We don't force other bits here; comparison will be bitwise.
        """
        desired: dict[str, dict[str, int]] = {}
        # TODO(jzilke)
        return desired

    def _settings_match(self, current: dict[str, dict[str, int]], desired: dict[str, dict[str, int]]) -> bool:
        """
        Compare current vs desired. For Drive_Mode, require bit2 set (mask check).
        For others, require exact equality.
        """
        # TODO(jzilke)
        return True
