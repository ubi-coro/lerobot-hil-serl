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
from typing import Optional

from lerobot.teleoperators import TeleoperatorConfig, TeleopEvents


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""
    device: Optional[str] = None
    path: Optional[str] = None
    action_scale: list[float] = field(default_factory=lambda: [1.0 * 6])

    # gripper
    initial_gripper_pos: float = 0.0
    gripper_close_button_idx: int | None = 1
    gripper_open_button_idx: int | None = 0
    gripper_continuous: bool = False
    gripper_gain: float = 0.05

    # buttons
    button_mapping: dict[int, dict[str, TeleopEvents | bool]] = field(default_factory=dict)



