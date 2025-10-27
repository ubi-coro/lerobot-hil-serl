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
from datetime import time
from functools import cached_property
from pprint import pformat
from typing import Any

from lerobot.robots import RobotConfig

from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so100_follower":
        from .bi_so100_follower import BiSO100Follower

        return BiSO100Follower(config)
    elif config.type == "bi_viperx":
        from .bi_viperx import BiViperX

        return BiViperX(config)
    elif config.type == "reachy2":
        from .reachy2 import Reachy2Robot

        return Reachy2Robot(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    elif config.type == "ur":
        from .ur import TF_UR

        return TF_UR(config)
    else:
        raise ValueError(config.type)


# TODO(pepijn): Move to pipeline step to make sure we don't have to do this in the robot code and send action to robot is clean for use in dataset
def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions


class MultiRobot(Robot):
    """
    Generic multi-robot wrapper that accepts a dictionary of robots.

    Example:
        robots = {
            "left": SO100Follower(left_cfg),
            "right": SO100Follower(right_cfg),
        }
        cameras = make_cameras_from_configs(camera_cfgs)

        multi = MultiRobot(robots=robots, cameras=cameras)
    """

    name = "multi_robot"

    def __init__(self, robots: dict[str, Robot]):
        self.robots = robots

        self.cameras = {}
        for robot in self.robots.values():
            self.cameras.update(robot.cameras)
            robot.cameras = {}

    # --- Features ---
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {}
        for name, robot in self.robots.items():
            for motor in robot.bus.motors:
                ft[f"{name}.{motor}.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (cam_cfg.height, cam_cfg.width, 3)
            for cam, cam_cfg in getattr(self, "config", {}).get("cameras", {}).items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # --- Connection / Calibration ---

    @property
    def is_connected(self) -> bool:
        return all(r.bus.is_connected for r in self.robots.values()) and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        for robot in self.robots.values():
            robot.connect(calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return all(r.is_calibrated for r in self.robots.values())

    def calibrate(self) -> None:
        for robot in self.robots.values():
            robot.calibrate()

    def configure(self) -> None:
        for robot in self.robots.values():
            robot.configure()

    def setup_motors(self) -> None:
        for robot in self.robots.values():
            robot.setup_motors()

    # --- Obs / Action ---

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        for name, robot in self.robots.items():
            robot_obs = robot.get_observation()
            obs_dict.update({f"{name}_{k}": v for k, v in robot_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        results = {}
        for name, robot in self.robots.items():
            sub_action = {
                k.removeprefix(f"{name}_"): v
                for k, v in action.items()
                if k.startswith(f"{name}_")
            }
            res = robot.send_action(sub_action)
            results.update({f"{name}_{k}": v for k, v in res.items()})
        return results

    def disconnect(self) -> None:
        for robot in self.robots.values():
            robot.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()



