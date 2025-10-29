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

from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_widowx")
@dataclass
class BiWidowXConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str

    left_arm_id: str | None = None
    right_arm_id: str | None = None
    left_arm_calibration_dir: Path | None = None
    right_arm_calibration_dir: Path | None = None

    # Optional per-arm settings (aligned with WidowXConfig)
    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    left_arm_max_relative_target: float | dict[str, float] | None = 5.0
    right_arm_max_relative_target: float | dict[str, float] | None = 5.0

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    left_arm_moving_time: float = 0.1
    right_arm_moving_time: float = 0.1

    # Use aloha2 gripper servo
    left_arm_use_aloha2_gripper_servo: bool = False
    right_arm_use_aloha2_gripper_servo: bool = False
