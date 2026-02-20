from typing import Any

import gymnasium as gym
import numpy as np
import torch
from lerobot.cameras import Camera, make_cameras_from_configs

from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config

from lerobot.robots import Robot, make_robot_from_config

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig


class ManipulationPrimitiveNet(gym.Env):
    def __init__(self, config: ManipulationPrimitiveNetConfig):
        self.config = config

        # initialize hardware environments
        robot_dict, teleop_dict, cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}

        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(robot_dict, teleop_dict, cameras, device=self.config.device)
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        pass

    def connect(self) -> tuple[dict[str, Robot], dict[str, Teleoperator], dict[str, Camera]]:
        assert self.robot is not None, "Robot config must be provided for real robot environment"

        # Handle multi robot configuration
        robot_dict = {}
        for name in self.robot:
            robot_dict[name] = make_robot_from_config(self.robot[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = {}
        for name in self.teleop:
            teleop_dict[name] = make_teleoperator_from_config(self.teleop[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras





