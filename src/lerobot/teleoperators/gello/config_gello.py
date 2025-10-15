#!/usr/bin/env python

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

from dataclasses import dataclass, field
from typing import Any

from lerobot.motors import MotorCalibration, MotorNormMode

from ..config import TeleoperatorConfig
from ...motors import Motor


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str  # Port to connect to the arm

    motors: dict[str, Motor] = field(default_factory=lambda: {})

    default_calibration: dict[str, MotorCalibration] | None = None

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    moving_time: float = 0.1


@TeleoperatorConfig.register_subclass("gelloha")
@dataclass
class GellohaConfig(GelloConfig):
    motors: dict[str, Motor] = field(default_factory=lambda:
        {
            "waist": Motor(1, "xl330-m288", MotorNormMode.RADIANS),
            "shoulder": Motor(2, "xl330-m288", MotorNormMode.RADIANS),
            "elbow": Motor(3, "xl330-m288", MotorNormMode.RADIANS),
            "forearm_roll": Motor(4, "xl330-m288", MotorNormMode.RADIANS),
            "wrist_angle": Motor(5, "xl330-m288", MotorNormMode.RADIANS),
            "wrist_rotate": Motor(6, "xl330-m288", MotorNormMode.RADIANS),
            "gripper": Motor(7, "xl330-m077", MotorNormMode.RADIANS),
        }
    )

    default_calibration: dict[str, MotorCalibration] | None = field(default_factory=lambda:
        {
            "waist": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "shoulder": MotorCalibration(id=2, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
            "elbow": MotorCalibration(id=3, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
            "forearm_roll": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "wrist_angle": MotorCalibration(id=5, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
            "wrist_rotate": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
            "gripper": MotorCalibration(id=7, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        }
    )