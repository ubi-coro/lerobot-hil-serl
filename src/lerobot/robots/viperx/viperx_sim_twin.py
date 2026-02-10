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
from typing import Any

from lerobot.robots import Robot
from lerobot.robots.viperx import ViperX, SimViperX
from lerobot.robots.viperx.config_viperx_sim_twin import ViperXSimTwinConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager

logger = logging.getLogger(__name__)

class ViperXSimTwin(Robot):
    config_class = ViperXSimTwinConfig
    name = "viperx_sim_twin"

    def __init__(
        self,
        config: ViperXSimTwinConfig,
    ):
        super().__init__(config)
        self.config = config
        self.sim_robot = SimViperX(config.sim_config)
        self.real_robot = ViperX(config.real_config)
        self.cameras = self.real_robot.cameras
        self._last_motor_obs = self.real_robot._last_motor_obs
        self.sim = SimManager.get()

        self.joint_names = self.sim_robot.joint_names


    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        result = self.real_robot.send_action(action)
        self.sim_robot.send_action(action)
        return result

    @property
    def observation_features(self) -> dict:
        return self.real_robot.observation_features

    @property
    def action_features(self) -> dict:
        return self.real_robot.action_features

    @property
    def is_connected(self) -> bool:
        return self.real_robot.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.real_robot.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.real_robot.is_calibrated

    def calibrate(self) -> None:
        self.real_robot.calibrate()

    def configure(self) -> None:
        self.real_robot.configure()

    def get_observation(self) -> dict[str, Any]:
        return self.real_robot.get_observation()

    def disconnect(self) -> None:
        self.real_robot.disconnect()

