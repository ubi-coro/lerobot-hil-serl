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
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.robots.viperx import ViperXConfig
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


class ViperX(Robot):
    """
    [ViperX](https://www.trossenrobotics.com/viperx-300) developed by Trossen Robotics
    """

    config_class = ViperXConfig
    name = "viperx"

    def __init__(
        self,
        config: ViperXConfig,
    ):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": Motor(1, "xm540-w270", MotorNormMode.RADIANS),
                "shoulder": Motor(2, "xm540-w270", MotorNormMode.RADIANS),
                "shoulder_shadow": Motor(3, "xm540-w270", MotorNormMode.RADIANS),
                "elbow": Motor(4, "xm540-w270", MotorNormMode.RADIANS),
                "elbow_shadow": Motor(5, "xm540-w270", MotorNormMode.RADIANS),
                "forearm_roll": Motor(6, "xm540-w270", MotorNormMode.RADIANS),
                "wrist_angle": Motor(7, "xm540-w270", MotorNormMode.RADIANS),
                "wrist_rotate": Motor(8, "xm430-w350", MotorNormMode.RADIANS),
                "gripper": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_1),
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_motor_obs = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors = {f"{motor}.pos": float for motor in self.bus.motors if not motor.endswith("_shadow")}
        #motors["finger.pos"] = float  # Special case for gripper position
        #motors.pop("gripper.pos", None)  # Remove gripper position, as it is not used directly
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
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # we disable torque to write calibration parameters to the servos
        # then we enable it again to read them when we check if our calibration matches (weird ik)
        # then we turn it off again during motor configuration and keep that
        self.bus.connect()

        with self.bus.torque_disabled():
            if self.calibration:
                self.bus.write_calibration(self.calibration)

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self.bus.disable_torque()
        self.get_observation()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        with self.bus.torque_disabled():
            self.calibration = {
                "waist": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
                "shoulder": MotorCalibration(id=2, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
                "shoulder_shadow": MotorCalibration(
                    id=3, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
                ),
                "elbow": MotorCalibration(id=4, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
                "elbow_shadow": MotorCalibration(
                    id=5, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
                ),
                "forearm_roll": MotorCalibration(
                    id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
                ),
                "wrist_angle": MotorCalibration(id=7, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
                "wrist_rotate": MotorCalibration(
                    id=8, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
                ),
                "gripper": MotorCalibration(id=9, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            }

            print(
                f"{self} arm: move the gripper joint through its range of motion"
            )

            range_mins, range_maxes = self.bus.record_ranges_of_motion(motors="gripper", display_values=False)
            self.calibration["gripper"].range_min = range_mins["gripper"]
            self.calibration["gripper"].range_max = range_maxes["gripper"]

            self.bus.write_calibration(self.calibration)
            self._save_calibration()
            logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            self.bus.write("Secondary_ID", "shoulder_shadow", 2)
            self.bus.write("Secondary_ID", "elbow_shadow", 4)

            # Set a velocity limit of 131 as advised by Trossen Robotics
            # TODO(aliberts): remove as it's actually useless in position control
            # self.bus.write("Velocity_Limit", 131)

            for motor in self.bus.motors:

                # Set the drive mode to time-based profile to set moving time via velocity profiles
                drive_mode = self.bus.read('Drive_Mode', motor)
                drive_mode |= 1 << 2  # set third bit to enable time-based profiles
                self.bus.write('Drive_Mode', motor, drive_mode)

                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
                else:
                    self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

                # Set time profile after setting operation mode
                self.bus.write("Profile_Velocity", motor, int(self.config.moving_time * 1000))

    def get_observation(self) -> dict[str, Any]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
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
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
