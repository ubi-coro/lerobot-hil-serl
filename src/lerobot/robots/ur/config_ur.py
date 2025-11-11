#!/usr/bin/env python
import logging
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.robots.ur.config_tf_controller import TaskFrameControllerConfig


@RobotConfig.register_subclass("ur")
@dataclass
class URConfig(TaskFrameControllerConfig, RobotConfig):
    """
    frequency: CB2=125, UR3e=500
    lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
    gain: [100, 2000] proportional gain for following target position
    max_pos_speed: m/s
    max_rot_speed: rad/s
    tcp_offset_pose: 6d pose
    payload_mass: float
    payload_cog: 3d position, center of gravity
    soft_real_time: enables round-robin scheduling and real-time priority
        requires running scripts/rtprio_setup.sh before hand.

    """
    model: str = "ur5e"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # gripper
    use_gripper: bool = False  # attempts to initialize gripper from RTDEControlInterface
    gripper_frequency: float = 50.0
    gripper_vel: float = 1.0  # [0-1]
    gripper_force: float = 1.0  # [0-1]
    gripper_soft_real_time: bool = False
    gripper_rt_core: int = 4

    def __post_init__(self):
        RobotConfig.__post_init__(self)
        TaskFrameControllerConfig.__post_init__(self)