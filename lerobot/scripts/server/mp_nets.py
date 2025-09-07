import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, Callable, Optional, Dict, Tuple, Any

import draccus
import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.envs import EnvConfig
from lerobot.common.envs.ur_env import UREnv
from lerobot.common.envs.wrapper.hilserl import TimeLimitWrapper, ImageCropResizeWrapper, TorchActionWrapper, \
    ConvertToLeRobotObservation, BatchCompatibleWrapper
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseInterventionWrapper
from lerobot.common.envs.wrapper.tff import StaticTaskFrameActionWrapper, StaticTaskFrameResetWrapper
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.ur import UR
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs.types import PolicyFeature, FeatureType


class AMPObsWrapper(gymnasium.Wrapper):
    def __init__(self,
                 env,
                 use_prev_action: bool = False,
                 use_xy_position: bool = False,
                 use_torque: bool = False,
                 device: str = "cuda"):
        super().__init__(env)
        self.device = device
        self.prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        self.use_prev_action = use_prev_action
        self.use_xy_position = use_xy_position
        self.use_torque = use_torque

        state_dim = 10
        if self.use_prev_action:
            state_dim += env.action_space.shape[0]
        if self.use_xy_position:
            state_dim += 2
        if self.use_torque:
            state_dim += 3

        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=np.full(state_dim, -1),
                high=np.full(state_dim, 1),
                shape=(state_dim, ),
                dtype=np.float32),
            **{key: value for key, value in env.observation_space.items() if "image" in key}
        })

    def _obs(self, obs):
        # [v_x-c, f_x-z, (f_a-c,) (*a), (p_x, p_y,) p_z]

        new_state = [
            obs["observation.main_eef_speed"][0],
            obs["observation.main_eef_speed"][1],
            obs["observation.main_eef_speed"][2],
            obs["observation.main_eef_speed"][3],
            obs["observation.main_eef_speed"][4],
            obs["observation.main_eef_speed"][5],
            obs["observation.main_eef_wrench"][0],
            obs["observation.main_eef_wrench"][1],
            obs["observation.main_eef_wrench"][2],
        ]

        if self.use_torque:
            new_state.extend([
                obs["observation.main_eef_wrench"][3],
                obs["observation.main_eef_wrench"][4],
                obs["observation.main_eef_wrench"][5]
            ])

        if self.use_prev_action:
            new_state.extend(self.prev_action)

        if self.use_xy_position:
            new_state.extend([
                obs["observation.main_eef_pos"][0],
                obs["observation.main_eef_pos"][1],
            ])

        new_state.append(obs["observation.main_eef_pos"][2])

        new_obs = {
            "observation.state": torch.tensor(new_state).to(device=self.device),
            **{key: value for key, value in obs.items() if "image" in key}
        }
        obs_info = {"observation.main_eef_pos": obs["observation.main_eef_pos"]}

        return new_obs, obs_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("is_intervention", False):
            self.prev_action = info["action_intervention"]
        else:
            self.prev_action = action

        new_obs, obs_info = self._obs(obs)
        info.update(obs_info)
        return new_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action[:] = 0.0
        new_obs, obs_info = self._obs(obs)
        info.update(obs_info)
        return new_obs, info

@dataclass
class PolicyConfig:
    indices: Dict[str, Sequence[int]] = field(default_factory=lambda: dict())
    action_bounds: Dict[str, Dict[Literal["min", "max"], Sequence[float]]] = field(default_factory=lambda: dict())
    config: Optional[SACConfig] = None
    pretrained_policy_name_or_path: Optional[str] = None


@dataclass
class WrapperConfig:
    # frame stacking
    stack_frames: Optional[int] = 2

    # Time-limit
    control_time_s: Optional[float] = None

    # cropping
    crop_params_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    crop_resize_size: Optional[Tuple[int, int]] = None

    # SpaceMouse wrapper settings
    spacemouse_devices: Optional[Dict[str, Any]] = None
    spacemouse_action_scale: Optional[Dict[str, Sequence[float]]] = None
    spacemouse_intercept_with_button: bool = False


@dataclass
class MPConfig:
    is_terminal: bool = False
    transitions: Dict[str, str] = field(default_factory=lambda: dict())
    tff: Dict[str, TaskFrameCommand] = field(default_factory=lambda: dict())
    policy: PolicyConfig = PolicyConfig()
    wrapper: WrapperConfig = WrapperConfig()

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
        }
    )

    def __post_init__(self):
        # build action space bounds from space mouse scaling factors
        if (
            self.is_adaptive and
            self.wrapper.spacemouse_devices and
            self.wrapper.spacemouse_action_scale
        ):
            for name in self.tff:
                action_scale = self.wrapper.spacemouse_action_scale[name]
                policy_indices = self.policy.indices[name]
                self.policy.action_bounds[name] = {
                    "min": [-abs(s) for i, s in enumerate(action_scale) if policy_indices[i]],
                    "max": [abs(s) for i, s in enumerate(action_scale) if policy_indices[i]]
                }

    @property
    def is_adaptive(self) -> bool:
        return any([any(indices) for indices in self.policy.indices.values()])

    def gym_kwargs(self) -> dict:
        return {}

    def make(self, mp_net: 'MPNetConfig', robot: Optional[UR] = None):
        if robot is None:
            robot = make_robot_from_config(mp_net.robot)

        env = UREnv(
            robot=robot,
            display_cameras=mp_net.display_cameras
        )

        env = StaticTaskFrameActionWrapper(
            env,
            static_tffs=self.tff,
            action_bounds=self.policy.action_bounds,
            action_indices=self.policy.indices,
            device=mp_net.device
        )

        if self.wrapper.control_time_s is not None:
            env = TimeLimitWrapper(env, fps=mp_net.fps, control_time_s=self.wrapper.control_time_s)

        # SpaceMouse Intervention
        if (
            self.is_adaptive and
            self.wrapper.spacemouse_devices and
            self.wrapper.spacemouse_action_scale
        ):
            env = SpaceMouseInterventionWrapper(
                env,
                devices=self.wrapper.spacemouse_devices,
                action_indices=self.policy.indices,
                action_scale=self.wrapper.spacemouse_action_scale,
                intercept_with_button=self.wrapper.spacemouse_intercept_with_button,
                device=mp_net.device
            )

        env = ConvertToLeRobotObservation(env, device=mp_net.device)

        if self.wrapper.crop_params_dict is not None:
            env = ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.crop_resize_size,
            )

        env = AMPObsWrapper(
            env,
            use_xy_position=getattr(mp_net, "use_xy_position", True),
            use_torque=getattr(mp_net, "use_torque", True),
            device=mp_net.device
        )

        env = BatchCompatibleWrapper(env=env)
        env = TorchActionWrapper(env, device=mp_net.device)

        # set tff manually
        controllers = env.unwrapped.robot.controllers
        for name, tff in self.tff.items():
            controllers[name].send_cmd(tff)

        return env


# Temporary solutions to implement resets from terminal (sometimes any) states
# Ideally, this is just part of the MP net.
# As this introduces cycles, episodes cannot be clearly counted. Still trackable
# as the min / max of episodes run by individual primitives
@dataclass
class ResetConfig:
    # Reset wrapper settings
    pos: Optional[Dict[str, Sequence[float]]] = None
    kp: Optional[Dict[str, Sequence[float]]] = None
    kd: Optional[Dict[str, Sequence[float]]] = None
    noise_std: Optional[Dict[str, Sequence[float]]] = None
    noise_dist: Literal['normal', 'uniform'] = 'uniform'
    safe_reset: bool = True
    threshold: float = 0.005
    timeout: float = 5.0


class InteractionCounter:
    def __init__(self, primitives: dict[str, MPConfig]):
        # initialize per-primitive step budgets and counters
        self._budget: dict[str, int] = {}
        self._count: dict[str, int] = {}
        for name, p in primitives.items():
            if p.is_adaptive and p.policy is not None:
                # use the online_steps from the primitive's SACConfig
                self._budget[name] = p.policy.config.online_steps
                self._count[name] = 0
            else:
                # non-adaptive → treat as already "finished"
                self._budget[name] = 0
                self._count[name] = 0

    def __getitem__(self, item):
        return self._count[item]

    def increment(self, name: str, n: int = 1):
        """Call this every time the given primitive takes n interaction steps."""
        if name in self._count:
            self._count[name] += n

    def is_finished(self, name: str) -> bool:
        """True if this primitive is non-adaptive or has reached its online_steps."""
        # budget == 0 means non-adaptive, so budget <= count ⇒ finished
        return self._count.get(name, 0) >= self._budget.get(name, 0)

    @property
    def global_step(self):
        return sum(self._count.values())

    @property
    def all_finished(self) -> bool:
        """True when every primitive is finished."""
        return all(self.is_finished(pid) for pid in self._count)


@dataclass
class MPNetConfig(draccus.ChoiceRegistry):
    start_primitive: str
    primitives: dict[str, MPConfig]
    reset: ResetConfig = ResetConfig()
    root: str = None

    fps: int = 10
    resume: bool = False
    preload_envs: bool = False
    repo_id: Optional[str] = None
    robot: URConfig = URConfig()
    display_cameras: bool = False

    task: str = ""
    num_episodes: int = 10
    episode: int = 0
    device: str = "cuda"
    storage_device: str = "cpu"
    push_to_hub: bool = False
    seed: int = 42

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def condition_registry(self) -> dict[str, Callable | RewardClassifierConfig]:
        return {}

    @property
    def reset_wrapper(self):
        return None, {}

    def __post_init__(self):
        assert self.start_primitive in self.primitives

        # give each primitive unique id to make indexing with primitives cleaner
        for _id, p in self.primitives.items():
            setattr(p, "id", _id)

        # repo_id is set automatically
        self.repo_id = "/".join(Path(self.root).parts[-2:])

    def check_transitions(
        self,
        current_primitive: MPConfig,
        obs: dict,
        done: bool
    ) -> 'MPConfig':
        if done:  # done could be after timeout, task might not be successful
            assert len(current_primitive.transitions) == 1, \
                "Transitions are ambiguous, only one transition per primitive at the moment."
            return self.primitives[list(current_primitive.transitions)[0]]

        for primitive_name, condition_str in current_primitive.transitions.items():
            condition = self.condition_registry[condition_str]

            if condition(obs):
                return self.primitives[primitive_name]

        return current_primitive


    def make_policies(self, resume: bool = False, path: Optional[str] = None):
        from learner_server_mpn import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK

        policies = {}
        for name, p in self.primitives.items():
            if p.is_adaptive:
                if resume:
                    p.policy.config.pretrained_path = os.path.join(
                        path,
                        name,
                        CHECKPOINTS_DIR,
                        LAST_CHECKPOINT_LINK,
                        "pretrained_model"
                    )
                policies[name] = make_policy(cfg=p.policy.config, env_cfg=p)
                policies[name] = policies[name].eval()
        return policies


    def get_step_counter(self) -> InteractionCounter:
        """
        Returns a fresh InteractionCounter that tracks how many interaction
        steps each adaptive primitive has taken, and knows when each
        has 'finished' its online_steps quota.
        """
        return InteractionCounter(self.primitives)

    def get_policy_configs(self):
        return [p.policy.config for p in self.primitives.values() if p.policy.config is not None]


def reset_mp_net(env, cfg: MPNetConfig):
    wrapper_cls, wrapper_kwargs = cfg.reset_wrapper

    if wrapper_cls is None:
        reset_wrapper_cls = StaticTaskFrameResetWrapper
    else:
        reset_wrapper_cls = wrapper_cls

    reset_env = reset_wrapper_cls(
        env=env,
        static_tffs=cfg.primitives[cfg.start_primitive].tff,
        reset_pos=cfg.reset.pos,
        reset_kp=cfg.reset.kp,
        reset_kd=cfg.reset.kd,
        noise_std=cfg.reset.noise_std,
        noise_dist=cfg.reset.noise_dist,
        safe_reset=cfg.reset.safe_reset,
        threshold=cfg.reset.threshold,
        timeout=cfg.reset.timeout,
        **wrapper_kwargs
    )

    obs, info = reset_env.reset()
    return obs, info




