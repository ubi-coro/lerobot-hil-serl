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
        default_calibration = {
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
                "gripper": Motor(9, "xm430-w350", MotorNormMode.RADIANS),
            },
            calibration=default_calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        motors = {f"{motor}.pos": float for motor in self.bus.motors if not motor.endswith("_shadow")}
        motors["finger.pos"] = float  # Special case for gripper position
        motors.pop("gripper.pos", None)  # Remove gripper position, as it is not used directly
        return motors

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

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

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO(aliberts): adapt code below (copied from koch
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

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

            # Set the drive mode to time-based profile to set moving time via velocity profiles
            drive_mode = self.bus.read('Drive_Mode')
            for i in range(len(self.bus.motor_names)):
                drive_mode[i] |= 1 << 2  # set third bit to enable time-based profiles
            self.bus.write('Drive_Mode', drive_mode)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
            # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
            # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point.
            # See: https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Use 'position control current based' for follower gripper to be limited by the limit of the
            # current. It can grasp an object without forcing too much even tho, it's goal position is a
            # complete grasp (both gripper fingers are ordered to join and reach a touch).
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # Set time profile after setting operation mode
            self.bus.write("Profile_Velocity", int(self.config.moving_time * 1000))

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items() if not motor.endswith("_shadow")}
        action["finger.pos"] = gripper_to_linear(action.pop("gripper.pos"))
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
