import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.processor import create_transition, TransitionKey, EnvTransition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from lerobot.cameras import Camera
from lerobot.teleoperators import Teleoperator, TeleopEvents
from lerobot.robots import Robot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig


class ManipulationPrimitiveNet(gym.Env):
    """Gym wrapper that chains manipulation primitives using typed transitions."""

    def __init__(self, config: ManipulationPrimitiveNetConfig):

        self.config = config

        # initialize hardware environments
        robot_dict, teleop_dict, cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}
        self._transitions: dict[str, list[Transition]] = {}

        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(robot_dict, teleop_dict, cameras, device=getattr(self.config, "device", "cpu"))
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor
            self._transitions[name] = []

        for transition in self.config.transitions:
            self._transitions[transition.source].append(transition)

        self._active = self.config.reset_primitive
        self._last_reset_info: dict[str, Any] = {}
        self._episode_step_count = 0
        self._primitive_step_count = 0
        self._needs_full_reset = True

    @property
    def active_id(self) -> str:
        return self._active

    @property
    def active_primitive(self) -> ManipulationPrimitiveConfig:
        return self.config.primitives[self._active]

    @property
    def action_dim(self) -> int:
        return self.active_primitive.features[ACTION].shape[0]

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

    def step(self, action: np.ndarray | torch.Tensor) -> EnvTransition:
        """Step active primitive once and evaluate at most one outgoing transition."""
        if self._needs_full_reset:
            raise RuntimeError("step() called after MP-Net episode finished; call reset() before stepping again.")

        transition = self._step_env_and_check_transitions(action)
        self._needs_full_reset = self.config.primitives[self._active].is_terminal

        return transition

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvTransition:
        super().reset(seed=seed)

        self._primitive_step_count = 0

        obs = {}
        info = {"seed": None}
        if self._needs_full_reset:
            # If we start in a reset primitive, route to start primitive inside reset()
            if self._active not in self.config.terminals:
                self._active = self.config.reset_primitive

            obs, _info = self._step_reset_path_until_start(obs=obs, info=info)
            info.update(_info)
            self._needs_full_reset = False
            self._episode_step_count = 0

        # pass down reset call
        for name, env in self._envs.items():
            self._env_processors[name].reset()
            self._action_processors[name].reset()

            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            options = {} if options is None else dict(options)
            _obs, _info = env.reset(seed=env_seed, options=options)

            # store observation of the active primitive
            if name == self._active:
                obs = _obs
                info.update(_info)

        transition = create_transition(observation=obs, info=info)
        processed_transition = self._env_processors[self._active](transition)
        self._last_reset_info = processed_transition[TransitionKey.INFO]
        return processed_transition

    def _step_env_and_check_transitions(self, action: torch.Tensor) -> EnvTransition:
        self._episode_step_count += 1
        self._primitive_step_count += 1
        active = self._active
        if active not in self._envs:
            raise KeyError(f"Unknown active primitive '{active}'.")

        # 1) Process action
        info = {}
        if self.config.primitives[active].policy is None:
            info[TeleopEvents.IS_INTERVENTION] = True

        action_transition = create_transition(action=action, info=info)
        processed_action_transition = self._action_processors[active](action_transition)

        if processed_action_transition[TransitionKey.INFO].get(TeleopEvents.INTERVENTION_COMPLETED, False):
            return processed_action_transition

        # 2) Step environment
        obs, reward, terminated, truncated, info = self._envs[active].step(processed_action_transition[TransitionKey.ACTION])

        # 3) Read out info and possibly overwrite action
        complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
        info.update(processed_action_transition[TransitionKey.INFO].copy())

        if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
            action_to_record = complementary_data[TELEOP_ACTION_KEY]
        else:
            action_to_record = action

        # 4) Process observation
        transition = create_transition(
            observation=obs,
            action=action_to_record,
            reward=reward + processed_action_transition[TransitionKey.REWARD],
            done=terminated or processed_action_transition[TransitionKey.DONE],
            truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
            info=info,
            complementary_data=complementary_data,
        )
        processed_transition = self._env_processors[active](transition)
        obs = processed_transition[TransitionKey.OBSERVATION]

        # 5) Build info
        info = processed_transition.get(TransitionKey.INFO, {})
        info["primitive_step"] = self._primitive_step_count
        info["episode_step"] = self._episode_step_count
        info["transition_from"] = active
        info["transition_to"] = active
        info["transition_reason"] = None

        # 6) Check for transitions
        for transition in self._transitions[self._active]:
            result = transition.evaluate(obs=obs, info=info)
            if not (result.terminated or result.truncated):
                continue

            # condition has fired
            self._primitive_step_count = 0
            self._active = transition.target

            processed_transition[TransitionKey.REWARD] += result.reward
            processed_transition[TransitionKey.DONE] |= result.terminated
            processed_transition[TransitionKey.TRUNCATED] |= result.truncated
            info["transition_to"] = transition.target
            info["transition_reason"] = result.reason
            break

        processed_transition[TransitionKey.INFO] = info
        return processed_transition

    def _step_reset_path_until_start(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        while self._active != self.config.start_primitive:
            start_loop_t = time.perf_counter()

            action = self._sample_action(self._active)
            transition = self._step_env_and_check_transitions(action)
            obs = transition[TransitionKey.OBSERVATION]
            info.update(transition[TransitionKey.INFO])  # keep at least prior info dict if env returns empty

            dt_load = time.perf_counter() - start_loop_t
            precise_sleep(1 / self.config.fps - dt_load)
        return obs, info

    def _sample_action(self, current_primitive: str) -> Any:
        ft = self.config.primitives[current_primitive].features[ACTION]
        return 2 * torch.rand(size=ft.shape) - 1

