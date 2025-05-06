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

import abc
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import draccus
import gymnasium as gym
import numpy as np

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.common.robot_devices.robots.configs import RobotConfig, AlohaRobotConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()

    def make(self, n_envs: int = 1, use_async_envs: bool = False) -> gym.vector.VectorEnv | None:
        """Makes a gym vector environment according to the config.

            Args:
                cfg (EnvConfig): the config of the environment to instantiate.
                n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
                use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
                    False.

            Raises:
                ValueError: if n_envs < 1
                ModuleNotFoundError: If the requested env package is not installed

            Returns:
                gym.vector.VectorEnv: The parallelized gym.env instance.
            """
        if n_envs < 1:
            raise ValueError("`n_envs must be at least 1")

        package_name = f"gym_{self.type}"

        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as e:
            print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{self.type}]'`")
            raise e

        gym_handle = f"{package_name}/{self.task}"

        # batched version of the env that returns an observation of shape (b, c)
        env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
        env = env_cls(
            [lambda: gym.make(gym_handle, disable_env_checker=True, **self.gym_kwargs) for _ in range(n_envs)]
        )

        return env



@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class VideoRecordConfig:
    """Configuration for video recording in ManiSkill environments."""

    enabled: bool = False
    record_dir: str = "videos"
    trajectory_name: str = "trajectory"


@dataclass
class WrapperConfig:
    """Configuration for environment wrappers."""

    joint_masking_action_space: list[bool] | None = None


@dataclass
class EEActionSpaceConfig:
    """Configuration parameters for end-effector action space."""

    x_step_size: float
    y_step_size: float
    z_step_size: float
    bounds: Dict[str, Any]  # Contains 'min' and 'max' keys with position bounds
    control_mode: str = "gamepad"


@dataclass
class EnvWrapperConfig:
    """Configuration for environment wrappers."""

    ee_action_space_params: EEActionSpaceConfig = field(default_factory=EEActionSpaceConfig)
    display_cameras: bool = False
    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    crop_params_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    resize_size: Optional[Tuple[int, int]] = None
    control_time_s: float = 20.0
    fixed_reset_joint_positions: Optional[Any] = None
    reset_time_s: float = 5.0
    use_gripper: bool = False
    gripper_quantization_threshold: float | None = 0.8
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False
    smoothing_range_factor: Optional[float] = None


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: Optional[RobotConfig] = None
    wrapper: Optional[EnvWrapperConfig] = None
    fps: int = 10
    name: str = "real_robot"
    mode: str = None  # Either "record", "replay", None
    resume: bool = False
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    task: str = ""
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    reward_classifier_pretrained_path: Optional[str] = None

    def gym_kwargs(self) -> dict:
        return {}

    def make(self, **kwargs) -> gym.vector.VectorEnv:
        """
        Factory function to create a vectorized robot environment.

        cfg.
            robot: Robot instance to control
            reward_classifier: Classifier model for computing rewards
            cfg: Configuration object containing environment parameters

        Returns:
            A vectorized gym environment with all the necessary wrappers applied.
        """
        import lerobot.common.envs.wrapper.hilserl as wrapper
        from lerobot.common.envs.robot_env import RobotEnv
        from lerobot.common.envs.wrapper.smoothing import SmoothActionWrapper
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        robot = make_robot_from_config(self.robot)
        # Create base environment
        env = RobotEnv(
            robot=robot,
            display_cameras=self.wrapper.display_cameras,
        )

        # Add observation and image processing
        if self.wrapper.add_joint_velocity_to_observation:
            env = wrapper.AddJointVelocityToObservation(env=env, fps=self.fps)
        if self.wrapper.add_current_to_observation:
            env = wrapper.AddCurrentToObservation(env=env)
        if self.wrapper.add_ee_pose_to_observation:
            env = wrapper.EEObservationWrapper(env=env, ee_pose_limits=self.wrapper.ee_action_space_params.bounds)

        env = wrapper.ConvertToLeRobotObservation(env=env, device=self.device)

        if self.wrapper.crop_params_dict is not None:
            env = wrapper.ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.resize_size,
            )

        # Add reward computation and control wrappers
        reward_classifier = self.init_reward_classifier()
        if reward_classifier is not None:
            env = wrapper.RewardWrapper(env=env, reward_classifier=reward_classifier, device=self.device)
        env = wrapper.TimeLimitWrapper(env=env, control_time_s=self.wrapper.control_time_s, fps=self.fps)
        if self.wrapper.use_gripper:
            env = wrapper.GripperActionWrapper(env=env, quantization_threshold=self.wrapper.gripper_quantization_threshold)
            if self.wrapper.gripper_penalty is not None:
                env = wrapper.GripperPenaltyWrapper(
                    env=env,
                    penalty=self.wrapper.gripper_penalty,
                )

        env = wrapper.EEActionWrapper(
            env=env,
            ee_action_space_params=self.wrapper.ee_action_space_params,
            use_gripper=self.wrapper.use_gripper,
        )

        if self.wrapper.smoothing_range_factor is not None:
            env = SmoothActionWrapper(env, smoothing_range_factor=self.wrapper.smoothing_range_factor, device=self.device)

        if self.wrapper.ee_action_space_params.control_mode == "gamepad":
            env = wrapper.GamepadControlWrapper(
                env=env,
                x_step_size=self.wrapper.ee_action_space_params.x_step_size,
                y_step_size=self.wrapper.ee_action_space_params.y_step_size,
                z_step_size=self.wrapper.ee_action_space_params.z_step_size,
                use_gripper=self.wrapper.use_gripper,
            )
        elif self.wrapper.ee_action_space_params.control_mode == "leader":
            env = wrapper.GearedLeaderControlWrapper(
                env=env,
                ee_action_space_params=self.wrapper.ee_action_space_params,
                use_gripper=self.wrapper.use_gripper,
            )
        elif self.wrapper.ee_action_space_params.control_mode == "leader_automatic":
            env = wrapper.GearedLeaderAutomaticControlWrapper(
                env=env,
                ee_action_space_params=self.wrapper.ee_action_space_params,
                use_gripper=self.wrapper.use_gripper,
            )
        else:
            raise ValueError(f"Invalid control mode: {self.wrapper.ee_action_space_params.control_mode}")

        env = wrapper.ResetWrapper(
            env=env,
            reset_pose=self.wrapper.fixed_reset_joint_positions,
            reset_time_s=self.wrapper.reset_time_s,
        )
        env = wrapper.BatchCompatibleWrapper(env=env)
        env = wrapper.TorchActionWrapper(env=env, device=self.device)

        return env


    def init_reward_classifier(self):
        """
        Load a reward classifier policy from a pretrained path if configured.

        Args:
            self: The environment configuration containing classifier paths

        Returns:
            The loaded classifier model or None if not configured
        """
        if self.reward_classifier_pretrained_path is None:
            return None

        from lerobot.common.policies.reward_model.modeling_classifier import Classifier

        # Get device from config or default to CUDA
        device = getattr(self, "device", "cpu")

        # Load the classifier directly using from_pretrained
        classifier = Classifier.from_pretrained(
            pretrained_name_or_path=self.reward_classifier_pretrained_path,
        )

        # Ensure model is on the correct device
        classifier.to(device)
        classifier.eval()  # Set to evaluation mode

        return classifier


@EnvConfig.register_subclass(name="real_push_cube")
@dataclass
class PushCubeRobotEnvConfig(HILSerlRobotEnvConfig):

    repo_id: str = "jannick-st/push-cube-offline-demos-eval"
    dataset_root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st//push-cube-offline-demos-eval"
    reward_classifier_pretrained_path: Optional[str] = "/media/nvme1/jstranghoener/lerobot/models/jannick-st/push-cube/classifier-300425/checkpoints/last/pretrained_model/"

    robot: AlohaRobotConfig = AlohaRobotConfig(
        calibration_dir="/home/jstranghoener/PycharmProjects/lerobot-hil-serl/.cache/calibration/aloha_default"
    )
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            "observation.image.cam_low": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.image.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.image.cam_left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
            "observation.image.cam_low": f"{OBS_IMAGE}.cam_low",
            "observation.image.cam_high": f"{OBS_IMAGE}.cam_high",
            "observation.image.cam_left_wrist": f"{OBS_IMAGE}.cam_left_wrist",
        }
    )
    wrapper: EnvWrapperConfig = EnvWrapperConfig(
        display_cameras=True,
        control_time_s=300.0,
        add_ee_pose_to_observation=True,
        crop_params_dict={
            "observation.images.cam_high": [
                194,
                1,
                284,
                317
            ],
            "observation.images.cam_left_wrist": [
                2,
                17,
                475,
                621
            ],
            "observation.images.cam_low": [
                212,
                296,
                268,
                341
            ]
        },
        resize_size=(128, 128),
        fixed_reset_joint_positions=[ 0.0,  -24.609375,   -24.433594,    52.558594,    52.822266, -0.43945312,  56.953125,    -2.8125,       4.6242776 ],
        smoothing_range_factor=0.3,
        ee_action_space_params=EEActionSpaceConfig(
            x_step_size=0.02,
            y_step_size=0.02,
            z_step_size=0.02,
            bounds={
                "max": [0.32,  0.22, 0.10],
                "min": [ 0.16, -0.09,  0.08]
            },
            control_mode="leader"
        )
    )
    task: str = "Push the cube over the line"
    num_episodes: int = 40  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = False
    fps: int = 10

    def __post_init__(self):
        if self.mode == "record":
            #self.wrapper.ee_action_space_params = None
            self.wrapper.crop_params_dict = None

        #for cam in self.robot.cameras:
        #    self.robot.cameras[cam].fps = self.fps


@EnvConfig.register_subclass("maniskill_push")
@dataclass
class ManiskillEnvConfig(EnvConfig):
    """Configuration for the ManiSkill environment."""

    name: str = "maniskill/pushcube"
    task: str = "PushCube-v1"
    image_size: int = 64
    control_mode: str = "pd_ee_delta_pose"
    state_dim: int = 25
    action_dim: int = 7
    fps: int = 200
    episode_length: int = 50
    obs_type: str = "rgb"
    render_mode: str = "rgb_array"
    render_size: int = 64
    device: str = "cuda"
    robot: str = "so100"  # This is a hack to make the robot config work
    video_record: VideoRecordConfig = field(default_factory=VideoRecordConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    mock_gripper: bool = False
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(25,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.image": OBS_IMAGE,
            "observation.state": OBS_ROBOT,
        }
    )
    reward_classifier_pretrained_path: Optional[str] = None

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
            "control_mode": self.control_mode,
            "sensor_configs": {"width": self.image_size, "height": self.image_size},
            "num_envs": 1,
        }

    def make(self, n_envs: int = 1, **kwargs) -> gym.vector.VectorEnv | None:
        """
        Factory function to create a ManiSkill environment with standard wrappers.

        Args:
            cfg: Configuration for the ManiSkill environment
            n_envs: Number of parallel environments

        Returns:
            A wrapped ManiSkill environment
        """
        import lerobot.common.envs.wrapper.maniskill as maniskill_wrapper
        from lerobot.common.envs.wrapper.hilserl import StabilizingActionMaskingWrapper
        from lerobot.common.envs.wrapper.smoothing import SmoothActionWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from mani_skill.utils.wrappers.record import RecordEpisode

        env = gym.make(
            self.task,
            obs_mode=self.obs_type,
            control_mode=self.control_mode,
            render_mode=self.render_mode,
            sensor_configs={"width": self.image_size, "height": self.image_size},
            num_envs=n_envs,
        )
        env._max_episode_steps = self.episode_length

        # Add video recording if enabled
        if self.video_record.enabled:
            env = RecordEpisode(
                env,
                output_dir=self.video_record.record_dir,
                save_trajectory=True,
                trajectory_name=self.video_record.trajectory_name,
                save_video=True,
                video_fps=30,
            )

        # Add observation and image processing
        env = maniskill_wrapper.ManiSkillObservationWrapper(env, device=self.device)
        env = ManiSkillVectorEnv(env, ignore_terminations=False, auto_reset=False)
        env._max_episode_steps = env.max_episode_steps = self.episode_length
        env.unwrapped.metadata["render_fps"] = self.fps

        # Add compatibility wrappers
        env = maniskill_wrapper.ManiSkillCompat(env)
        env = maniskill_wrapper.ManiSkillActionWrapper(env)
        env = maniskill_wrapper.ManiSkillMultiplyActionWrapper(env, multiply_factor=0.03)
        env = StabilizingActionMaskingWrapper(env, ref_pose=np.array([0.0, 0.0, 0.02, 0.0, 1.0, 0.0, 0.0]), ax=[0, 1])
        # env = ActionLoggingPlotWrapper(env)
        env = SmoothActionWrapper(env, device=self.device)
        env = maniskill_wrapper.KeyboardControlWrapper(env, ax=[0, 1])
        return env
