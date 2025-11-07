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

from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""
    device: Optional[str] = None
    path: Optional[str] = None

    initial_gripper_pos: float = 0.0
    gripper_close_button_idx: bool = 1
    gripper_open_button_idx: bool = 0
    gripper_continuous: bool = False
    gripper_gain: float = 0.05


if __name__ == "__main__":
    from easyhid import Enumeration
    hid = Enumeration()
    for device in hid.find():
        print("Device:", device.product_string)
        print("Path:", device.path)
        print("=" * 5)



