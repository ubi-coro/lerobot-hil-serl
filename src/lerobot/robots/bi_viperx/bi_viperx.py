#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.robots.viperx import ViperX
from lerobot.robots.viperx.config_viperx import ViperXConfig

from ..robot import Robot
from .config_bi_viperx import BiViperXConfig

logger = logging.getLogger(__name__)


class BiViperX(Robot):
    """
    [Bimanual ViperX Arms] designed by Lennart
    This bimanual robot can also be easily adapted to use ViperX arms, just replace the ViperX class with viperx and ViperXConfig with viperxConfig.
    """

    config_class = BiViperXConfig
    name = "bi_viperx"

    def __init__(self, config: BiViperXConfig):
        super().__init__(config)
        self.config = config

        left_arm_id = config.left_arm_id or (f"{config.id}_left" if config.id else None)
        left_arm_calibration_dir = config.left_arm_calibration_dir or config.calibration_dir

        right_arm_id = config.right_arm_id or (f"{config.id}_right" if config.id else None)
        right_arm_calibration_dir = config.right_arm_calibration_dir or config.calibration_dir

        left_arm_config = ViperXConfig(
            id=left_arm_id,
            calibration_dir=left_arm_calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            moving_time=config.left_arm_moving_time,
            cameras={},
        )

        right_arm_config = ViperXConfig(
            id=right_arm_id,
            calibration_dir=right_arm_calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            moving_time=config.right_arm_moving_time,
            cameras={},
        )

        self.left_arm = ViperX(left_arm_config)
        self.right_arm = ViperX(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._shadow_debug_enabled = bool(config.show_debugging_graphs)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

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
        # Robot is considered connected if arms are connected
        # Cameras are optional (lazy loading)
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
        )

    def connect(self, calibrate: bool = True, connect_cameras: bool = True) -> None:
        """Connect both arms and optionally cameras.
        
        Args:
            calibrate: Whether to run calibration if needed
            connect_cameras: Whether to connect cameras (lazy loading for faster connection)
        """
        self.left_arm.connect(calibrate, connect_cameras=False)  # Don't connect arm cameras
        self.right_arm.connect(calibrate, connect_cameras=False)  # Don't connect arm cameras

        # Lazy camera connection: only connect if requested
        if connect_cameras:
            for cam in self.cameras.values():
                cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    def get_shadow_debug_status(self) -> dict[str, dict[str, float | int | None]]:
        if not self._shadow_debug_enabled:
            return {}

        status = {}
        left_status = self.left_arm.get_shadow_debug_status()
        right_status = self.right_arm.get_shadow_debug_status()

        for joint, values in left_status.items():
            status[f"left_{joint}"] = values
        for joint, values in right_status.items():
            status[f"right_{joint}"] = values

        return status
