import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, Callable, Optional, Dict, Tuple, Any

import draccus

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs import EnvConfig
from lerobot.common.envs.ur_env import UREnv
from lerobot.common.envs.wrapper.hilserl import TimeLimitWrapper, ImageCropResizeWrapper, TorchActionWrapper, ConvertToLeRobotObservation
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseInterventionWrapper
from lerobot.common.envs.wrapper.tff import StaticTaskFrameActionWrapper, StaticTaskFrameResetWrapper
from lerobot.common.policies.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.common.policies.reward_model.modeling_classifier import Classifier
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs.types import PolicyFeature, FeatureType


@dataclass
class WrapperConfig:
    # Time-limit
    control_time_s: float = 10.0

    # cropping
    crop_params_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    crop_resize_size: Optional[Tuple[int, int]] = None

    # Reset wrapper settings
    reset_pos: Optional[Dict[str, Sequence[float]]] = None
    reset_kp: Optional[Dict[str, Sequence[float]]] = None
    reset_kd: Optional[Dict[str, Sequence[float]]] = None
    noise_std: Optional[Dict[str, Sequence[float]]] = None
    noise_dist: Literal['normal', 'uniform'] = 'uniform'
    safe_reset: bool = True
    threshold: float = 0.005
    timeout: float = 5.0

    # SpaceMouse wrapper settings
    spacemouse_devices: Optional[Dict[str, Any]] = None
    spacemouse_action_scale: Optional[Dict[str, Sequence[float]]] = None
    spacemouse_intercept_with_button: bool = False


@dataclass
class ManipulationPrimitiveConfig(EnvConfig):
    is_terminal: bool = False
    tff:            Dict[str, TaskFrameCommand]                                       = field(default_factory=lambda: dict())
    transitions:    Dict[str, Callable | RewardClassifierConfig]                      = field(default_factory=lambda: dict())
    policy_indices: Dict[str, Sequence[int]]                                          = field(default_factory=lambda: dict())
    policy_bounds:  Optional[Dict[str, Dict[Literal["min", "max"], Sequence[float]]]] = None
    policy:         Optional[SACConfig]                                               = None
    pretrained_policy_name_or_path: Optional[str]                                     = None
    wrapper:        WrapperConfig                                                     = WrapperConfig()

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

    @property
    def is_adaptive(self):
        return any([any(indices) for indices in self.policy_indices.values()])

    @property
    def has_policy(self):
        return (
            self.is_adaptive and
            self.pretrained_policy_name_or_path is not None and
            isinstance(self.policy, SACPolicy)
        )

    def gym_kwargs(self) -> dict:
        return {}

    def make(self):
        env = UREnv(
            robot=make_robot_from_config(self.robot),
            display_cameras=self.display_cameras
        )

        env = StaticTaskFrameActionWrapper(
            env,
            static_tffs=self.tff,
            action_bounds=self.policy_bounds,
            action_indices=self.policy_indices,
            device=self.device
        )

        # Static Reset
        if self.wrapper.reset_pos:
            env = StaticTaskFrameResetWrapper(
                env,
                static_tffs=self.tff or {},
                reset_pos=self.wrapper.reset_pos,
                reset_kp=self.wrapper.reset_kp,
                reset_kd=self.wrapper.reset_kd,
                noise_std=self.wrapper.noise_std,
                noise_dist=self.wrapper.noise_dist,
                safe_reset=self.wrapper.safe_reset,
                threshold=self.wrapper.threshold,
                timeout=self.wrapper.timeout
            )

        env = TimeLimitWrapper(env, fps=self.fps, control_time_s=self.wrapper.control_time_s)

        # SpaceMouse Intervention
        if (
            self.policy_indices and
            self.wrapper.spacemouse_devices and
            self.wrapper.spacemouse_action_scale
        ):
            env = SpaceMouseInterventionWrapper(
                env,
                devices=self.wrapper.spacemouse_devices,
                action_indices=self.policy_indices,
                action_scale=self.wrapper.spacemouse_action_scale,
                intercept_with_button=self.wrapper.spacemouse_intercept_with_button,
                device=self.device
            )

        env = ConvertToLeRobotObservation(env, device=self.device)

        if self.wrapper.crop_params_dict is not None:
            env = ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.crop_resize_size,
            )

        env = TorchActionWrapper(env, device=self.device)

        return env


@dataclass
class ResetConfig:
    # Reset wrapper settings
    reset_pos: Optional[Dict[str, Sequence[float]]] = None
    reset_kp: Optional[Dict[str, Sequence[float]]] = None
    reset_kd: Optional[Dict[str, Sequence[float]]] = None
    noise_std: Optional[Dict[str, Sequence[float]]] = None
    noise_dist: Literal['normal', 'uniform'] = 'uniform'
    safe_reset: bool = True
    threshold: float = 0.005
    timeout: float = 5.0


class InteractionCounter:
    def __init__(self, primitives: dict[str, ManipulationPrimitiveConfig]):
        # initialize per-primitive step budgets and counters
        self._budget: dict[str, int] = {}
        self._count: dict[str, int] = {}
        for name, p in primitives.items():
            if p.is_adaptive and p.policy is not None:
                # use the online_steps from the primitive's SACConfig
                self._budget[name] = p.policy.online_steps
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
    def all_finished(self) -> bool:
        """True when every primitive is finished."""
        return all(self.is_finished(pid) for pid in self._count)


@dataclass
class MPNetConfig(draccus.ChoiceRegistry):
    start_primitive: str
    primitives: dict[str, ManipulationPrimitiveConfig]
    reset: ResetConfig = ResetConfig()

    robot: URConfig = URConfig()
    display_cameras: bool = False
    fps: int = 10
    resume: bool = False
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    task: str = ""
    num_episodes: int = 10
    episode: int = 0
    device: str = "cuda"
    storage_device: str = "cpu"
    push_to_hub: bool = True
    seed: int = 42

    def __post_init__(self):
        for _id, p in self.primitives.values():
            setattr(p, "id", _id)

        # init models

    def get_policies(self):
        return {name: p.policy for name, p in self.primitives.items() if p.is_adaptive and p.policy is not None}

    def get_interaction_counter(self) -> InteractionCounter:
        """
        Returns a fresh InteractionCounter that tracks how many interaction
        steps each adaptive primitive has taken, and knows when each
        has 'finished' its online_steps quota.
        """
        return InteractionCounter(self.primitives)


def init_datasets(cfg: MPNetConfig) -> Tuple[Dict[str, LeRobotDataset], int]:
    datasets = {}
    min_episode = float('inf')
    for name, primitive in cfg.primitives.values():
        # Configure dataset features based on environment spaces
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": primitive.features["observation.state"].shape,
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": primitive.features["action"].shape,
                "names": None,
            },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }

        # Add image features
        for key in primitive.features:
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": primitive.features[key].shape,
                    "names": None,
                }

        # Create dataset
        dataset_root = Path(cfg.dataset_root) / name
        repo_id = cfg.repo_id + f"-{name}"
        if cfg.resume:
            datasets[primitive.id] = LeRobotDataset(cfg.repo_id, root=dataset_root)
            datasets[primitive.id].start_image_writer(
                num_processes=2,
                num_threads=4 * len(cfg.robot.cameras),
            )
        else:
            datasets[primitive.id] = LeRobotDataset.create(
                cfg.fps,
                repo_id,
                root=dataset_root,
                use_videos=True,
                image_writer_threads=4 * len(cfg.robot.cameras),
                image_writer_processes=2,
                features=features,
            )

        # Update min_episode
        if datasets[primitive.id].num_episodes < min_episode:
            min_episode = datasets[primitive.id].num_episodes

    return datasets, min_episode


def init_models(cfg: MPNetConfig) -> MPNetConfig:
    # not supported yet
    return cfg




