from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any
from venv import create

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.processor import create_transition, TransitionKey, EnvTransition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import Transition
from tests.processor.test_libero_processor import observation

if TYPE_CHECKING:
    from lerobot.cameras import Camera
    from lerobot.teleoperators import Teleoperator, TeleopEvents
    from lerobot.robots import Robot
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
        self._needs_full_reset = True

    @property
    def active_primitive(self) -> str:
        return self._active_primitive

    @property
    def action_dim(self) -> int:
        return self.config.primitives[self.active_primitive].features[ACTION].shape[0]

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
        self._needs_full_reset = self.config.primitives[self._active_primitive].is_terminal

        return transition

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvTransition:
        super().reset(seed=seed)

        self._episode_step_count = 0
        self._primitive_step_count = 0

        obs = {}
        info = {"seed": None}
        if self._needs_full_reset:
            # If we start in a reset primitive, route to start primitive inside reset()
            if self._active_primitive not in self.config.terminals:
                self._active_primitive = self.config.reset_primitive

            obs, _info = self._step_reset_path_until_start(obs=obs, info=info)
            info.update(_info)
            self._needs_full_reset = False

        # pass down reset call
        for name, env in self._envs.items():
            self._env_processors[name].reset()
            self._action_processors[name].reset()

            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            options = {} if options is None else dict(options)
            _obs, _info = env.reset(seed=env_seed, options=options)

            # store observation of the active primitive
            if name == self._active_primitive:
                obs = _obs
                info.update(_info)

        transition = create_transition(observation=obs, info=info)
        processed_transition = self._env_processors[self._active_primitive](transition)
        self._last_reset_info = processed_transition[TransitionKey.INFO]
        return processed_transition

    def _step_env_and_check_transitions(self, action: np.ndarray | torch.Tensor) -> EnvTransition:
        self._episode_step_count += 1
        self._primitive_step_count += 1
        active = self._active_primitive
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

        # Determine which action to store (either action that went in, or teleop action that was written as complementary data)
        if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
            action_to_record = complementary_data[TELEOP_ACTION_KEY]
        else:
            action_to_record = action_transition[TransitionKey.ACTION]

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
        reward = processed_transition[TransitionKey.REWARD]

        # 5) Build info
        info = processed_transition.get(TransitionKey.INFO, {})
        info["step"] = self._primitive_step_count
        info["active_primitive"] = self._active_primitive
        info["transition_from"] = active
        info["transition_to"] = active
        info["transition_reason"] = None

        # 6) Check for transitions
        for target, transition in self._transitions[self._active_primitive]:
            transition_result = transition.evaluate(obs=obs, info=info)
            if not transition_result.condition_fulfilled:
                continue

            # condition has fired
            self._primitive_step_count = 0
            self._active_primitive = target

            reward += transition_result.additional_reward
            processed_transition[TransitionKey.DONE] |= transition_result.terminated
            processed_transition[TransitionKey.TRUNCATED] |= transition_result.truncated
            info["transition_to"] = target
            info["transition_reason"] = transition_result.reason
            break

        return processed_transition

    def _step_reset_path_until_start(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        while self._active_primitive != self.config.start_primitive:
            action = self._sample_action(self._active_primitive)
            transition = self._step_env_and_check_transitions(action)
            obs = transition[TransitionKey.OBSERVATION]
            info.update(transition[TransitionKey.INFO])  # keep at least prior info dict if env returns empty

            print(transition[TransitionKey.INFO])
        return obs, info

    def _sample_action(self, current_primitive: str) -> Any:
        ft = self.config.primitives[current_primitive].features[ACTION]
        return np.random.uniform(low=-1, high=1, size=ft.shape)


