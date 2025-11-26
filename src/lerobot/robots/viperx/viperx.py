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
            motors=OrderedDict({
                "waist": Motor(1, "xm540-w270", MotorNormMode.RADIANS),
                "shoulder": Motor(2, "xm540-w270", MotorNormMode.RADIANS),
                "shoulder_shadow": Motor(3, "xm540-w270", MotorNormMode.RADIANS),
                "elbow": Motor(4, "xm540-w270", MotorNormMode.RADIANS),
                "elbow_shadow": Motor(5, "xm540-w270", MotorNormMode.RADIANS),
                "forearm_roll": Motor(6, "xm540-w270", MotorNormMode.RADIANS),
                "wrist_angle": Motor(7, "xm540-w270", MotorNormMode.RADIANS),
                "wrist_rotate": Motor(8, "xm430-w350", MotorNormMode.RADIANS),
                "gripper": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_1),
            }),
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
        # Robot is considered connected if bus is connected
        # Cameras are optional (lazy loading via connect_cameras parameter)
        return self.bus.is_connected

    def connect(self, calibrate: bool = True, connect_cameras: bool = True) -> None:
        """
        Connect to the robot, applying calibration and configuration only if needed.
        
        This method is optimized to avoid disabling torque when the robot is already
        configured correctly. This prevents the arm from dropping when reconnecting
        to an already-torqued robot.
        
        Args:
            calibrate: Whether to run interactive calibration if needed
            connect_cameras: Whether to connect cameras (set False for lazy loading)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        
        # Check what needs to be done BEFORE disabling torque
        needs_calibration_write = self.calibration and not self._calibration_matches()
        # Only need interactive calibration if we have NO calibration file AND calibrate=True
        # If we have a calibration file (self.calibration is truthy), we use that instead
        needs_interactive_calibration = not self.calibration and calibrate
        needs_configuration = not self._configuration_matches()
        
        # Only disable torque if something actually needs to change
        if needs_calibration_write or needs_interactive_calibration or needs_configuration:
            logger.info(f"{self}: configuration needed (calib_write={needs_calibration_write}, "
                       f"interactive_calib={needs_interactive_calibration}, config={needs_configuration})")
            with self.bus.torque_disabled():
                if needs_calibration_write:
                    self.bus.write_calibration(self.calibration)
                    logger.info(f"{self}: calibration written to motors")
                
                if needs_interactive_calibration:
                    self._run_calibration()
                
                if needs_configuration:
                    self._apply_configuration()
        else:
            logger.info(f"{self}: already calibrated and configured; keeping torque ON")
            # Still need to cache calibration in bus for position normalization
            if self.calibration:
                self.bus.calibration = dict(self.calibration)

        # Ensure torque is enabled after configuration
        self.bus.enable_torque()

        # Connect cameras if requested
        if connect_cameras:
            for cam in self.cameras.values():
                cam.connect()

        self.get_observation()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Run interactive calibration. Requires torque to be disabled."""
        with self.bus.torque_disabled():
            self._run_calibration()

    def _run_calibration(self) -> None:
        """Internal calibration routine. Caller must ensure torque is disabled."""
        self.calibration = {
            "waist": MotorCalibration(id=1, drive_mode=4, homing_offset=0, range_min=0, range_max=4095),
            "shoulder": MotorCalibration(id=2, drive_mode=5, homing_offset=0, range_min=0, range_max=4095),
            "shoulder_shadow": MotorCalibration(
                id=3, drive_mode=4, homing_offset=0, range_min=0, range_max=4095
            ),
            "elbow": MotorCalibration(id=4, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
            "elbow_shadow": MotorCalibration(
                id=5, drive_mode=4, homing_offset=0, range_min=0, range_max=4095
            ),
            "forearm_roll": MotorCalibration(
                id=6, drive_mode=4, homing_offset=0, range_min=0, range_max=4095
            ),
            "wrist_angle": MotorCalibration(id=7, drive_mode=5, homing_offset=0, range_min=0, range_max=4095),
            "wrist_rotate": MotorCalibration(
                id=8, drive_mode=4, homing_offset=0, range_min=0, range_max=4095
            ),
            "gripper": MotorCalibration(id=9, drive_mode=4, homing_offset=0, range_min=0, range_max=4095),
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
        """
        Apply motor configuration. If settings already match, skips torque-off.
        """
        if self._configuration_matches():
            logger.info(f"{self}: motor settings already match; skipping configuration.")
            return

        logger.info(f"{self}: applying motor configuration (requires torque-off).")
        with self.bus.torque_disabled():
            self._apply_configuration()

    def _apply_configuration(self) -> None:
        """Internal configuration routine. Caller must ensure torque is disabled."""
        # Return delay time for all motors
        self.bus.configure_motors(return_delay_time=0)

        # Secondary IDs for shadow motors
        if "shoulder_shadow" in self.bus.motors:
            self.bus.write("Secondary_ID", "shoulder_shadow", 2)
        if "elbow_shadow" in self.bus.motors:
            self.bus.write("Secondary_ID", "elbow_shadow", 4)

        # Drive_Mode: set bit2 (time-based profiles) while preserving other bits
        for motor in self.bus.motors:
            dm = self.bus.read("Drive_Mode", motor)
            dm |= (1 << 2)
            self.bus.write("Drive_Mode", motor, dm)

        # Operating Mode
        for motor in self.bus.motors:
            if motor != "gripper":
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
            else:
                self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

        # Profile velocity (ms)
        pv = int(self.config.moving_time * 1000)
        for motor in self.bus.motors:
            self.bus.write("Profile_Velocity", motor, pv)
        
        logger.info(f"{self}: motor configuration applied.")

    def _configuration_matches(self) -> bool:
        """Check if current motor configuration matches desired settings."""
        current = self._read_current_motor_settings()
        desired = self._desired_motor_settings()
        return self._settings_match(current, desired)

    def _calibration_matches(self) -> bool:
        """
        Check if calibration values on the motors match our cached calibration.
        This avoids unnecessary torque-off for writing already-correct values.
        
        Note: For Drive_Mode, we only compare bits 0-1 (direction bits), since
        bit 2 (time-based profiles) is set by _apply_configuration() separately.
        """
        if not self.calibration:
            return True  # No calibration to write
        
        try:
            for motor, calib in self.calibration.items():
                # Read current values from motor
                homing = self.bus.read("Homing_Offset", motor)
                min_pos = self.bus.read("Min_Position_Limit", motor)
                max_pos = self.bus.read("Max_Position_Limit", motor)
                drive_mode = self.bus.read("Drive_Mode", motor)
                
                # Compare with desired calibration
                if homing != calib.homing_offset:
                    logger.debug(f"Calibration mismatch {motor} Homing_Offset: have={homing}, want={calib.homing_offset}")
                    return False
                if min_pos != calib.range_min:
                    logger.debug(f"Calibration mismatch {motor} Min_Position_Limit: have={min_pos}, want={calib.range_min}")
                    return False
                if max_pos != calib.range_max:
                    logger.debug(f"Calibration mismatch {motor} Max_Position_Limit: have={max_pos}, want={calib.range_max}")
                    return False
                # For Drive_Mode, only compare bits 0-1 (direction), ignore bit 2 (time-based profiles)
                # Bit 2 is handled by _apply_configuration()
                direction_mask = 0b11  # bits 0-1
                if (drive_mode & direction_mask) != (calib.drive_mode & direction_mask):
                    logger.debug(f"Calibration mismatch {motor} Drive_Mode direction: have={drive_mode & direction_mask}, want={calib.drive_mode & direction_mask}")
                    return False
            
            # All values match - cache calibration in bus for normalization
            self.bus.calibration = dict(self.calibration)
            return True
        except Exception as e:
            logger.debug(f"Error reading calibration values: {e}")
            return False  # If we can't read, assume we need to write

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

    def _read_current_motor_settings(self) -> dict[str, dict[str, int]]:
        """
        Snapshot current settings while torque is ON.
        Returns a dict: {setting_name: {motor_name: value}} for fast comparisons.
        """
        snap: dict[str, dict[str, int]] = {}

        # Common per-motor regs
        for reg in ("Return_Delay_Time", "Drive_Mode", "Operating_Mode", "Profile_Velocity"):
            snap[reg] = self.bus.sync_read(reg)

        # Only relevant motors for Secondary_ID
        sec = {}
        if "shoulder_shadow" in self.bus.motors:
            sec["shoulder_shadow"] = self.bus.read("Secondary_ID", "shoulder_shadow")
        if "elbow_shadow" in self.bus.motors:
            sec["elbow_shadow"] = self.bus.read("Secondary_ID", "elbow_shadow")
        snap["Secondary_ID"] = sec

        return snap

    def _desired_motor_settings(self) -> dict[str, dict[str, int]]:
        """
        Compute desired settings exactly as configure() would apply.
        NOTE: For Drive_Mode, we only *require* that bit2 (time-based profiles) is set.
        We don't force other bits here; comparison will be bitwise.
        """
        desired: dict[str, dict[str, int]] = {}

        # Return delay time set by bus.configure_motors(return_delay_time=0)
        desired["Return_Delay_Time"] = {m: 0 for m in self.bus.motors}

        # Drive mode: ensure bit 2 set (time-based profile)
        # We'll compare with a mask rather than exact equality.
        desired["Drive_Mode"] = {}  # placeholder; comparison uses bit mask only

        # Operating mode
        desired["Operating_Mode"] = {}
        for m in self.bus.motors:
            if m == "gripper":
                desired["Operating_Mode"][m] = OperatingMode.CURRENT_POSITION.value
            else:
                desired["Operating_Mode"][m] = OperatingMode.EXTENDED_POSITION.value

        # Profile velocity from moving_time (seconds) -> ms
        pv = int(self.config.moving_time * 1000)
        desired["Profile_Velocity"] = {m: pv for m in self.bus.motors}

        # Secondary IDs
        desired["Secondary_ID"] = {}
        if "shoulder_shadow" in self.bus.motors:
            desired["Secondary_ID"]["shoulder_shadow"] = 2
        if "elbow_shadow" in self.bus.motors:
            desired["Secondary_ID"]["elbow_shadow"] = 4

        return desired

    def _settings_match(self, current: dict[str, dict[str, int]], desired: dict[str, dict[str, int]]) -> bool:
        """
        Compare current vs desired. For Drive_Mode, require bit2 set (mask check).
        For others, require exact equality.
        """
        # 1) Return_Delay_Time exact
        for m, want in desired["Return_Delay_Time"].items():
            have = current["Return_Delay_Time"].get(m)
            if have != want:
                logger.debug(f"Mismatch Return_Delay_Time[{m}]: have={have}, want={want}")
                return False

        # 2) Secondary_ID where applicable
        for m, want in desired["Secondary_ID"].items():
            have = current["Secondary_ID"].get(m)
            if have != want:
                logger.debug(f"Mismatch Secondary_ID[{m}]: have={have}, want={want}")
                return False

        # 3) Drive_Mode: bit2 (time-profile) must be set
        mask = 1 << 2
        for m, have in current["Drive_Mode"].items():
            if (have & mask) == 0:
                logger.debug(f"Mismatch Drive_Mode[{m}]: bit2 not set (have=0b{have:b})")
                return False

        # 4) Operating_Mode exact
        for m, want in desired["Operating_Mode"].items():
            have = current["Operating_Mode"].get(m)
            if have != want:
                logger.debug(f"Mismatch Operating_Mode[{m}]: have={have}, want={want}")
                return False

        # 5) Profile_Velocity exact
        for m, want in desired["Profile_Velocity"].items():
            have = current["Profile_Velocity"].get(m)
            if have != want:
                logger.debug(f"Mismatch Profile_Velocity[{m}]: have={have}, want={want}")
                return False

        return True
