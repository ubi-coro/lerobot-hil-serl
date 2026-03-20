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
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Sequence

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.robots.ur.config_tf_controller import TaskFrameControllerConfig


@RobotConfig.register_subclass("ur")
@dataclass
class URConfig(RobotConfig):
    robot_ip: str
    model: str = "ur5e"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # controller parameters
    frequency: float = 500.0
    payload_mass: Optional[float] = None
    payload_cog: Optional[Sequence[float]] = None
    tcp_offset_pose: Optional[list[float]] = None
    soft_real_time: bool = False
    rt_core: int = 3
    launch_timeout: float = 10.0
    get_max_k: int = 128
    receive_keys: Optional[list[str]] = None
    shm_manager: Optional[SharedMemoryManager] = None

    # safety
    max_pose_rpy: list[float] = field(default_factory = lambda: [float("inf")] * 6)
    min_pose_rpy: list[float] = field(default_factory = lambda: [-float("inf")] * 6)
    wrench_limits: list[float] = field(default_factory = lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])
    speed_limits: list[float] = field(default_factory = lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])

    # deadband
    deadband_pos: float = 0.001  # [m/s]
    deadband_rot: float = 0.01  # [rad/s]
    leak_rate_pos: float = 5.0  # [1/s]
    leak_rate_rot: float = 5.0  # [1/s]

    # contact-aware scaling of wrench limits
    enable_contact_aware_force_scaling: list[bool] = field(default_factory = lambda: [False] * 6)
    contact_desired_wrench: list[float] = field(default_factory = lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])  # desired max contact force at equilibrium (N)
    contact_limit_scale_theta: Optional[list[float]] = None  # minimum force limit scaling factor, usually computed automatically
    contact_limit_scale_min: list[float] = field(default_factory = lambda: [0.1] * 6)  # minimum force limit scaling factor

    # flag
    use_degrees: bool = False  # Set to `True` for backward compatibility with previous policies/dataset
    verbose: bool = False
    mock: bool = False
    debug: bool = False
    debug_axis: int = 0
