from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
if TYPE_CHECKING:
    from lerobot.cameras import Camera
    from lerobot.teleoperators import Teleoperator
    from lerobot.robots import Robot
    from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig


class ManipulationPrimitiveNet(gym.Env):
    def __init__(self, config: "ManipulationPrimitiveNetConfig"):

        self.config = config

        # initialize hardware environments
        robot_dict, teleop_dict, cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}

        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(robot_dict, teleop_dict, cameras, device=getattr(self.config, "device", "cpu"))
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor

        self._active_primitive = self.config.start_primitive
        self._last_reset_info: dict[str, Any] = {}
        self._episode_step_count = 0


    def _resolve_reset_start(self) -> str:
        if self.config.start_primitive in self._envs:
            return self.config.start_primitive

        for reset_primitive in getattr(self.config, "reset_primitives", []):
            if reset_primitive in self._envs:
                return reset_primitive

        raise KeyError(
            f"Unable to resolve reset start primitive: start_primitive='{self.config.start_primitive}' "
            "and no configured reset primitive exists in env map."
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        requested_start = options.get("start_primitive")

        if requested_start is not None:
            if requested_start not in self._envs:
                raise KeyError(f"Unknown reset primitive '{requested_start}'.")
            next_active = requested_start
            reset_reason = "explicit_option"
        else:
            next_active = self._resolve_reset_start()
            reset_reason = "start_primitive" if next_active == self.config.start_primitive else "reset_primitive_fallback"

        self._active_primitive = next_active
        self._episode_step_count = 0

        reset_info: dict[str, Any] = {
            "active_primitive": self._active_primitive,
            "reset_reason": reset_reason,
            "requested_start": requested_start,
        }

        if seed is not None:
            reset_info["seed"] = seed

        obs = None
        for name, env in self._envs.items():
            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            env_options = dict(options)
            env_options["active_primitive"] = name
            if name != self._active_primitive:
                env_options["inactive_reset"] = True

            if hasattr(env, "reset"):
                env_obs, env_info = env.reset(seed=env_seed, options=env_options)
            else:
                env_obs, env_info = {}, {}

            if name == self._active_primitive:
                obs = env_obs
                reset_info["primitive_reset_info"] = dict(env_info or {})

        if obs is None:
            raise RuntimeError(f"Failed to reset active primitive '{self._active_primitive}'.")

        self._last_reset_info = reset_info
        return obs, reset_info

    @staticmethod
    def _evaluate_transition(transition: Any, obs: dict[str, np.ndarray], info: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Normalize transition interfaces to a bool + metadata contract."""
        if hasattr(transition, "evaluate"):
            result = transition.evaluate(obs=obs, info=info)
        elif hasattr(transition, "check"):
            result = transition.check(obs=obs, info=info)
        else:
            raise AttributeError("Transition must define either `evaluate(obs, info)` or `check(obs, info)`")

        if isinstance(result, bool):
            return result, {}
        if isinstance(result, tuple):
            condition = bool(result[0])
            metadata = result[1] if len(result) > 1 else {}
            return condition, metadata or {}
        if isinstance(result, dict):
            return bool(result.get("condition_fulfilled", result.get("triggered", False))), result

        raise TypeError(f"Unsupported transition evaluation output type: {type(result)!r}")

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        active = self._active_primitive
        if active not in self._envs:
            raise KeyError(f"Unknown active primitive '{active}'.")

        obs, reward, terminated, truncated, info = self._envs[active].step(action)
        self._episode_step_count += 1

        transition_info = {
            "from": active,
            "to": active,
            "reason": None,
            "transition_name": None,
            "transition_type": None,
        }

        for source, default_target, transition in self.config.transitions:
            if source != active:
                continue

            fired, transition_metadata = self._evaluate_transition(transition=transition, obs=obs, info=info)
            if not fired:
                continue

            transition_target = transition_metadata.get("next_primitive", default_target)
            reward += float(transition_metadata.get("additional_reward", 0.0))
            terminated = bool(terminated or transition_metadata.get("terminated", False))
            truncated = bool(truncated or transition_metadata.get("truncated", False))

            transition_info = {
                "from": source,
                "to": transition_target,
                "reason": transition_metadata.get("reason", "transition_fired"),
                "transition_name": transition_metadata.get("transition_name", transition.__class__.__name__),
                "transition_type": transition_metadata.get("transition_type", transition.__class__.__name__),
            }

            self._active_primitive = transition_target
            break

        info = dict(info)
        info["transition"] = transition_info
        info["active_primitive"] = self._active_primitive

        return obs, reward, terminated, truncated, info

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




