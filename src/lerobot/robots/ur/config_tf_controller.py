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
from typing import Optional, Sequence, Any

import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@dataclass
class TaskFrameControllerConfig:
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

    # safety
    max_pose_rpy: list[float] = field(default_factory = lambda: [float("inf")] * 6)
    min_pose_rpy: list[float] = field(default_factory = lambda: [-float("inf")] * 6)
    wrench_limits: list[float] = field(default_factory = lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])
    speed_limits: list[float] = field(default_factory = lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])

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

    def __post_init__(self):
        if self.contact_limit_scale_theta is None:
            if self.verbose:
                logging.info(f"=== Compute parameters for exponential contact force limit scaling: ===")

            self.contact_limit_scale_theta = [0.0] * 6
            for i in range(6):
                if not self.enable_contact_aware_force_scaling[i]:
                    continue

                if self.wrench_limits[i] == float("inf"):
                    self.wrench_limits[i] = 2.0 * self.contact_desired_wrench[i]

                # Compute theta
                theta = self.compute_theta(
                    self.wrench_limits[i],  # assume uniform limits
                    self.contact_desired_wrench[i],
                    self.contact_limit_scale_min[i],
                )

                # Evaluate scale and derivative at f_star
                s_star, ds_df_star = self.exp_scale_and_derivative(
                    self.contact_desired_wrench[i],
                    theta,
                    self.contact_limit_scale_min[i]
                )
                g_prime = self.wrench_limits[i] * ds_df_star

                if self.verbose:
                    logging.info(f" {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-Axis:")
                    logging.info(f"  Computed θ = {theta:.4f}")
                    logging.info(f"  At f* = {self.contact_desired_wrench[i]} N:")
                    logging.info(f"    s(f*) = {s_star:.4f}")
                    logging.info(f"    s'(f*) = {ds_df_star:.4f}")
                    logging.info(f"    g'(f*) = F_max * s'(f*) = {g_prime:.4f}")

                    # Bifurcation check
                    if abs(g_prime) < 1.0:
                        logging.info("  --> Stable fixed point (|g'(f*)| < 1)")
                    else:
                        logging.warning("  --> Unstable: bifurcation/oscillation likely (|g'(f*)| >= 1)")

                if abs(g_prime) >= 1.0:
                    raise ValueError(f"Likely oscillation on {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-axis contact "
                                     f"force limiter, run again with verbose=True and check parameters!")

                self.contact_limit_scale_theta[i] = theta

    @staticmethod
    def compute_theta(F_max: float, f_star: float, s_min: float) -> float:
        """
        Compute the decay constant θ so that the fixed-point condition
            f_star = F_max * [s_min + (1 - s_min) * exp(-f_star/θ)]
        is satisfied exactly.
        """
        s_star = f_star / F_max
        if not (s_min < s_star < 1.0):
            raise ValueError("Require s_min < f_star/F_max < 1.0")
        ratio = (s_star - s_min) / (1.0 - s_min)
        return -f_star / np.log(ratio)

    @staticmethod
    def exp_scale_and_derivative(f: float, theta: float, s_min: float) -> tuple:
        """
        Returns the scale s(f) and its derivative s'(f) for exponential-decay-to-floor.
        s(f) = s_min + (1 - s_min)*exp(-f/theta)
        s'(f) = -(1 - s_min)/theta * exp(-f/theta)
        """
        exp_term = np.exp(-f / theta)
        s = s_min + (1 - s_min) * exp_term
        ds_df = -(1 - s_min) / theta * exp_term
        return s, ds_df
