from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.cameras import Camera
    from lerobot.teleoperators import Teleoperator
    from lerobot.robots import Robot
    from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig


class ManipulationPrimitiveNet(gym.Env):
    """Gym wrapper that chains manipulation primitives using typed transitions."""

    def __init__(self, config: "ManipulationPrimitiveNetConfig"):

        self.config = config
        self.transitions = {source: (target, transition) for source, target, transition in self.config.transitions}

        # initialize hardware environments
        robot_dict, teleop_dict, cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}
        self._transitions = {}

        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(robot_dict, teleop_dict, cameras, device=getattr(self.config, "device", "cpu"))
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor
            self._transitions[name] = []

        for source, target, transition in self.config.transitions:
            self._transitions[source].append((target, transition))

        self._active_primitive = self.config.reset_primitive
        self._last_reset_info: dict[str, Any] = {}
        self._episode_step_count = 0
        self._primitive_step_count = 0
        self._needs_reset = False

    def connect(self) -> tuple[dict[str, "Robot"], dict[str, "Teleoperator"], dict[str, "Camera"]]:
        assert self.config.robot is not None, "Robot config must be provided for real robot environment"

        from lerobot.cameras import make_cameras_from_configs
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.robots import make_robot_from_config

        # Handle multi robot configuration
        robot_dict = {}
        for name in self.config.robot:
            robot_dict[name] = make_robot_from_config(self.config.robot[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = {}
        for name in self.config.teleop:
            teleop_dict[name] = make_teleoperator_from_config(self.config.teleop[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.config.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step active primitive once and evaluate at most one outgoing transition."""
        if self._needs_reset:
            raise RuntimeError("step() called after MP-Net episode finished; call reset() before stepping again.")

        obs, reward, terminated, truncated, info = self._step_env_and_check_transitions(action)
        self._needs_reset = terminated or truncated

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        self._episode_step_count = 0
        self._primitive_step_count = 0
        self._needs_reset = False

        obs = None
        info = {"active_primitive": self._active_primitive, "seed": None}
        for name, env in self._envs.items():
            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            options = {} if options is None else dict(options)
            _obs, _info = env.reset(seed=env_seed, options=options)

            if name == self._active_primitive:
                obs = _obs
                info.update(_info)

        # If we start in a reset primitive, route to start primitive inside reset()
        if self._active_primitive not in self.config.terminals:
            self._active_primitive = self.config.reset_primitive

        obs, _info = self._step_reset_path_until_start(obs=obs, info=info)
        info.update(_info)

        self._last_reset_info = info
        return obs, info

    def _step_env_and_check_transitions(self, action: np.ndarray | torch.Tensor):
        active = self._active_primitive
        if active not in self._envs:
            raise KeyError(f"Unknown active primitive '{active}'.")

        obs, reward, prim_terminated, prim_truncated, info = self._envs[active].step(action)
        self._episode_step_count += 1
        self._primitive_step_count += 1

        info = dict(info or {})
        info["step"] = self._primitive_step_count
        info["primitive_terminated"] = bool(prim_terminated)
        info["primitive_truncated"] = bool(prim_truncated)

        terminated = False
        truncated = False

        transition_info = {
            "from": active,
            "to": active,
            "reason": None,
        }

        # check for transitions
        for target, transition in self._transitions[self._active_primitive]:
            transition_result = transition.evaluate(obs=obs, info=info)
            if not transition_result.condition_fulfilled:
                continue

            # condition has fired
            self._primitive_step_count = 0
            self._active_primitive = target
            obs, info = self._envs[self._active_primitive].reset()

            reward += transition_result.additional_reward
            terminated |= transition_result.terminated
            truncated |= transition_result.truncated

            transition_info["to"] = target
            transition_info["reason"] = transition_result.reason
            break

        if self.config.primitives[self._active_primitive].is_terminal:
            terminated = True
            transition_info["reason"] = "terminal_primitive_no_transition"

        info["transition"] = transition_info
        info["active_primitive"] = self._active_primitive
        return obs, reward, terminated, truncated, info

    def _step_reset_path_until_start(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        while self._active_primitive != self.config.start_primitive:
            action = self._sample_action(self._active_primitive)
            obs, reward, terminated, truncated, step_info = self._step_env_and_check_transitions(action)
            info.update(step_info)  # keep at least prior info dict if env returns empty

            print(step_info)
        return obs, info

    def _sample_action(self, current_primitive: str) -> Any:
        ft = self.config.primitives[current_primitive].features[ACTION]
        return np.random.uniform(low=-1, high=1, size=ft.shape)


