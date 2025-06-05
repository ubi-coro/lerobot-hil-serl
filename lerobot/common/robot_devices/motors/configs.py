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

import abc
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("ur")
@dataclass
class URArmConfig:
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
    robot_ip: str
    shm_manager: Optional[SharedMemoryManager] = None
    frequency: float = 500.0
    soft_real_time: bool = False
    rt_core: int = 3
    launch_timeout: float = 3.0
    verbose: bool = False
    receive_keys: Optional[list[str]] = None
    mock: bool = False

    # gripper
    gripper_ip: Optional[str] = None
    gripper_port: int = 1000
    gripper_frequency: float = 60.0

    # controller parameters
    lookahead_time: float = 0.1
    gain: int = 300
    get_max_k: int = 128
    payload_mass: Optional[float] = None
    payload_cog: Optional[float] = None
    tcp_offset_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.12, 0.0, 0.0, 0.0])

    # safety
    max_pos_speed: float = 0.25
    max_rot_speed: float = 0.6
    joints_init: Optional[float] = None
    joints_init_speed: Optional[float] = 1.05
    
    # latency
    obs_latency: float = 0.0001
    action_latency: float = 0.1
    gripper_latency: float = 0.1










