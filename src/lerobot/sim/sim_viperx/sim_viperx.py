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
from collections import OrderedDict
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.robots.viperx import ViperXConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager
from lerobot.sim.sim_viperx import SimViperXConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

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
        # if self.is_connected:
        #     raise DeviceAlreadyConnectedError(f"{self} already connected")


        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        # self.get_observation()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """
        Read current motor registers (with torque ON). If all match our desired
        configuration, skip the torque-off writes. Otherwise, torque-off and apply.
        """
        pass

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

        # # Common per-motor regs
        # for reg in ("Return_Delay_Time", "Drive_Mode", "Operating_Mode", "Profile_Velocity"):
        #     snap[reg] = self.bus.sync_read(reg)
        #
        # # Only relevant motors for Secondary_ID
        # sec = {}
        # if "shoulder_shadow" in self.bus.motors:
        #     sec["shoulder_shadow"] = self.bus.read("Secondary_ID", "shoulder_shadow")
        # if "elbow_shadow" in self.bus.motors:
        #     sec["elbow_shadow"] = self.bus.read("Secondary_ID", "elbow_shadow")
        # snap["Secondary_ID"] = sec

        return snap

    def _desired_motor_settings(self) -> dict[str, dict[str, int]]:
        """
        Compute desired settings exactly as configure() would apply.
        NOTE: For Drive_Mode, we only *require* that bit2 (time-based profiles) is set.
        We don't force other bits here; comparison will be bitwise.
        """
        desired: dict[str, dict[str, int]] = {}

        # # Return delay time set by bus.configure_motors(return_delay_time=0)
        # desired["Return_Delay_Time"] = {m: 0 for m in self.bus.motors}
        #
        # # Drive mode: ensure bit 2 set (time-based profile)
        # # We'll compare with a mask rather than exact equality.
        # desired["Drive_Mode"] = {}  # placeholder; comparison uses bit mask only
        #
        # # Operating mode
        # desired["Operating_Mode"] = {}
        # for m in self.bus.motors:
        #     if m == "gripper":
        #         desired["Operating_Mode"][m] = OperatingMode.CURRENT_POSITION.value
        #     else:
        #         desired["Operating_Mode"][m] = OperatingMode.EXTENDED_POSITION.value
        #
        # # Profile velocity from moving_time (seconds) -> ms
        # pv = int(self.config.moving_time * 1000)
        # desired["Profile_Velocity"] = {m: pv for m in self.bus.motors}
        #
        # # Secondary IDs
        # desired["Secondary_ID"] = {}
        # if "shoulder_shadow" in self.bus.motors:
        #     desired["Secondary_ID"]["shoulder_shadow"] = 2
        # if "elbow_shadow" in self.bus.motors:
        #     desired["Secondary_ID"]["elbow_shadow"] = 4

        return desired

    def _settings_match(self, current: dict[str, dict[str, int]], desired: dict[str, dict[str, int]]) -> bool:
        """
        Compare current vs desired. For Drive_Mode, require bit2 set (mask check).
        For others, require exact equality.
        """
        # 1) Return_Delay_Time exact
        # for m, want in desired["Return_Delay_Time"].items():
        #     have = current["Return_Delay_Time"].get(m)
        #     if have != want:
        #         logger.debug(f"Mismatch Return_Delay_Time[{m}]: have={have}, want={want}")
        #         return False
        #
        # # 2) Secondary_ID where applicable
        # for m, want in desired["Secondary_ID"].items():
        #     have = current["Secondary_ID"].get(m)
        #     if have != want:
        #         logger.debug(f"Mismatch Secondary_ID[{m}]: have={have}, want={want}")
        #         return False
        #
        # # 3) Drive_Mode: bit2 (time-profile) must be set
        # mask = 1 << 2
        # for m, have in current["Drive_Mode"].items():
        #     if (have & mask) == 0:
        #         logger.debug(f"Mismatch Drive_Mode[{m}]: bit2 not set (have=0b{have:b})")
        #         return False
        #
        # # 4) Operating_Mode exact
        # for m, want in desired["Operating_Mode"].items():
        #     have = current["Operating_Mode"].get(m)
        #     if have != want:
        #         logger.debug(f"Mismatch Operating_Mode[{m}]: have={have}, want={want}")
        #         return False
        #
        # # 5) Profile_Velocity exact
        # for m, want in desired["Profile_Velocity"].items():
        #     have = current["Profile_Velocity"].get(m)
        #     if have != want:
        #         logger.debug(f"Mismatch Profile_Velocity[{m}]: have={have}, want={want}")
        #         return False

        return True
