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
from lerobot.robots.viperx.viperx import gripper_to_linear
from lerobot.teleoperators import Teleoperator, TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class WidowX(Teleoperator):
    """
    [ViperX](https://www.trossenrobotics.com/viperx-300) developed by Trossen Robotics
    """

    config_class = WidowXConfig
    name = "widowx"

    def __init__(
        self,
        config: WidowXConfig,
    ):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": Motor(1, "xm430-w350", MotorNormMode.RADIANS),
                "shoulder": Motor(2, "xm430-w350", MotorNormMode.RADIANS),
                "shoulder_shadow": Motor(3, "xm430-w350", MotorNormMode.RADIANS),
                "elbow": Motor(4, "xm430-w350", MotorNormMode.RADIANS),
                "elbow_shadow": Motor(5, "xm430-w350", MotorNormMode.RADIANS),
                "forearm_roll": Motor(6, "xm430-w350", MotorNormMode.RADIANS),
                "wrist_angle": Motor(7, "xm430-w350", MotorNormMode.RADIANS),
                "wrist_rotate": Motor(8, "xl430-w250", MotorNormMode.RADIANS),
                "gripper": Motor(9, "xl430-w250", MotorNormMode.RANGE_0_1),
            }
        )
        self._last_motor_obs = None
        self._torque_enabled = False

    @property
    def action_features(self) -> dict[str, type]:
        motors = {f"{motor}.pos": float for motor in self.bus.motors if not motor.endswith("_shadow")}
        #motors["finger.pos"] = float  # Special case for gripper position
        #motors.pop("gripper.pos", None)  # Remove gripper position, as it is not used directly
        return motors

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

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

        self.configure()
        self.get_action()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()

        self.calibration = {
            "waist": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "shoulder": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "shoulder_shadow": MotorCalibration(
                id=3, drive_mode=1, homing_offset=0, range_min=0, range_max=4095
            ),
            "elbow": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "elbow_shadow": MotorCalibration(
                id=5, drive_mode=1, homing_offset=0, range_min=0, range_max=4095
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
        self.bus.disable_torque()
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

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items() if not motor.endswith("_shadow")}
        #action["finger.pos"] = gripper_to_linear(action.pop("gripper.pos"))
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        self._last_motor_obs = action

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.enable_torque()

        goal_pos = {key: feedback.get(key, self._last_motor_obs[key]) for key in self._last_motor_obs}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, self._last_motor_obs[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        goal_pos = {key.removesuffix(".pos"): value for key, value in goal_pos.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

    def enable_torque(self):
        if not self._torque_enabled:
            self.bus.enable_torque()
            self._torque_enabled = True

    def disable_torque(self):
        if self._torque_enabled:
            self.bus.disable_torque()
            self._torque_enabled = False

    def get_teleop_events(self) -> dict[TeleopEvents, Any]:
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
