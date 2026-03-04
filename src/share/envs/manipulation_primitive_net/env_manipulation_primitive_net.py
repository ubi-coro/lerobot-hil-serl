from dataclasses import asdict, is_dataclass
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
    """Gym env that composes manipulation primitives with explicit transitions."""

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
        self._needs_reset = False

    @staticmethod
    def _default_action_for_env(env: Any) -> Any:
        action_space = getattr(env, "action_space", None)
        if action_space is not None and hasattr(action_space, "sample"):
            sampled_action = action_space.sample()
            if isinstance(sampled_action, np.ndarray):
                return np.zeros_like(sampled_action)
            return sampled_action
        return np.zeros((1,), dtype=np.float32)

    def _step_reset_path_until_start(
        self,
        obs: dict[str, np.ndarray],
        info: dict[str, Any],
        *,
        max_steps: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
        reset_steps = 0

        while self._active_primitive != self.config.start_primitive:
            if reset_steps >= max_steps:
                raise RuntimeError(
                    "Exceeded maximum reset transition steps while routing to start primitive. "
                    f"Active='{self._active_primitive}', start='{self.config.start_primitive}'."
                )

            transitioned = False
            for source, default_target, transition in self.config.transitions:
                if source != self._active_primitive:
                    continue

                fired, transition_metadata = self._evaluate_transition(transition=transition, obs=obs, info=info)
                if not fired:
                    continue

                self._active_primitive = transition_metadata.get("next_primitive", default_target)
                transitioned = True
                break

            if not transitioned:
                raise RuntimeError(
                    "Failed to route reset path to start primitive: no transition fired from "
                    f"primitive '{self._active_primitive}'."
                )

            if self._active_primitive != self.config.start_primitive:
                action = self._default_action_for_env(self._envs[self._active_primitive])
                obs, _, _, _, info = self._envs[self._active_primitive].step(action)
                info = dict(info or {})
            reset_steps += 1

        return obs, info, reset_steps


    def _resolve_reset_start(self) -> str:
        configured_reset_primitives = getattr(self.config, "reset_primitives", [])
        for reset_primitive in configured_reset_primitives:
            if reset_primitive in self._envs:
                return reset_primitive

        if self.config.start_primitive in self._envs:
            return self.config.start_primitive

        raise KeyError(
            f"Unable to resolve reset start primitive: start_primitive='{self.config.start_primitive}' "
            "and no configured reset primitive exists in env map."
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset all primitive envs and return an observation from the start primitive domain."""
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
        self._needs_reset = False

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

        reset_steps = 0
        if self._active_primitive != self.config.start_primitive:
            max_steps = max(1, len(self.config.transitions) + len(self._envs))
            primitive_reset_info = dict(reset_info.get("primitive_reset_info") or {})
            obs, primitive_reset_info, reset_steps = self._step_reset_path_until_start(
                obs=obs,
                info=primitive_reset_info,
                max_steps=max_steps,
            )
            reset_info["primitive_reset_info"] = primitive_reset_info
            reset_info["active_primitive"] = self._active_primitive
            reset_info["reset_transition_steps"] = reset_steps

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
        if is_dataclass(result):
            metadata = asdict(result)
            return bool(metadata.get("condition_fulfilled", metadata.get("triggered", False))), metadata

        raise TypeError(f"Unsupported transition evaluation output type: {type(result)!r}")

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step active primitive once and evaluate at most one outgoing transition."""
        if self._needs_reset:
            raise RuntimeError("step() called after MP-Net episode finished; call reset() before stepping again.")

        active = self._active_primitive
        if active not in self._envs:
            raise KeyError(f"Unknown active primitive '{active}'.")

        obs, reward, prim_terminated, prim_truncated, info = self._envs[active].step(action)
        self._episode_step_count += 1

        info = dict(info or {})
        info.setdefault("episode_step_count", self._episode_step_count)
        info["primitive_done"] = bool(prim_terminated or prim_truncated)
        info["primitive_terminated"] = bool(prim_terminated)
        info["primitive_truncated"] = bool(prim_truncated)
        if "primitive_done_reason" not in info:
            if prim_terminated:
                info["primitive_done_reason"] = "primitive_terminated"
            elif prim_truncated:
                info["primitive_done_reason"] = "primitive_truncated"

        terminated = False
        truncated = False

        transition_info = {
            "from": active,
            "to": active,
            "reason": None,
            "transition_name": None,
            "transition_type": None,
        }

        transition_fired = False
        transition_additional_reward = 0.0
        for source, default_target, transition in self.config.transitions:
            if source != active:
                continue

            fired, transition_metadata = self._evaluate_transition(transition=transition, obs=obs, info=info)
            if not fired:
                continue

            transition_fired = True
            transition_target = transition_metadata.get("next_primitive", default_target)
            transition_additional_reward = float(transition_metadata.get("additional_reward", 0.0))
            reward += transition_additional_reward
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

        if not transition_fired and bool(getattr(self.config.primitives.get(active), "is_terminal_primitive", False)):
            terminated = True
            transition_info = {
                "from": active,
                "to": active,
                "reason": "terminal_primitive_no_transition",
                "transition_name": None,
                "transition_type": "terminal_policy",
            }

        info["transition"] = transition_info
        info["active_primitive"] = self._active_primitive
        info["segment_done"] = bool(transition_fired and transition_info["to"] != transition_info["from"])
        if info["segment_done"]:
            info["segment_from"] = transition_info["from"]
            info["segment_to"] = transition_info["to"]
            info["segment_reason"] = transition_info["reason"] or "transition_fired"
            if transition_additional_reward != 0.0:
                info["segment_additional_reward"] = transition_additional_reward

        if terminated or truncated:
            self._needs_reset = True

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

