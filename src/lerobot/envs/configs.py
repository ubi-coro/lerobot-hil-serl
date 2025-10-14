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
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import draccus

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.tf_env import TaskFrameEnv
from lerobot.processor import AddTeleopActionAsComplimentaryDataStep, AddTeleopEventsAsInfoStep, InterventionActionProcessorStep, \
    DataProcessorPipeline, EnvTransition, ImageCropResizeProcessorStep, TimeLimitProcessorStep, GripperPenaltyProcessorStep, \
    RewardClassifierProcessorStep, AddBatchDimensionProcessorStep, DeviceProcessorStep
from lerobot.processor.converters import identity_transition
from lerobot.processor.tff_processor import VanillaTFFProcessorStep, SixDofVelocityInterventionActionProcessorStep
from lerobot.rl.gym_manipulator import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.envs.robot_env import RobotEnv
from lerobot.robots.ur.tff_controller import TaskFrameCommand
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, DEFAULT_ROBOT_NAME


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
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
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
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
    task: str | None = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
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
class ImagePreprocessingConfig:
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class RewardClassifierConfig:
    """Configuration for reward classification."""

    enable: bool = False
    pretrained_path: str | None = None
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class InverseKinematicsConfig:
    """Configuration for inverse kinematics processing."""

    enable: bool = False
    urdf_path: str | None = None
    target_frame_name: str | None = None
    end_effector_bounds: dict[str, list[float]] | None = None
    end_effector_step_sizes: dict[str, float] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_velocity_to_observation: bool = False
    add_ee_wrench_to_observation: bool = False
    stack_frames: int = 0


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    use_gripper: bool = False
    gripper_penalty: float = 0.0
    max_pos: float = 100.0


@dataclass
class ResetConfig:
    """Configuration for environment reset behavior."""

    fixed_reset_joint_positions: Any | None = None
    reset_time_s: float = 5.0
    terminate_on_success: bool = True


@dataclass
class TaskFrameConfig:
    command: TaskFrameCommand = field(default_factory=TaskFrameCommand.make_default_cmd)
    control_mask: list[int] = field(default_factory=lambda: [1] * 6)


@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_time_s: float | None = None
    display_cameras: bool = False

    image_preprocessing: ImagePreprocessingConfig = None
    reward_classifier: RewardClassifierConfig = None

    observation: ObservationConfig | dict[str, ObservationConfig] = ObservationConfig()
    gripper: GripperConfig | dict[str, GripperConfig] = GripperConfig()
    reset: ResetConfig | dict[str, ResetConfig] = ResetConfig()
    inverse_kinematics: InverseKinematicsConfig | dict[str, InverseKinematicsConfig] = InverseKinematicsConfig()
    task_frame: TaskFrameConfig | dict[str, TaskFrameConfig] = TaskFrameConfig()


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    fps: int = 30
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/agentview_image": f"{OBS_IMAGES}.image",
            "pixels/robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HilSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | dict[str, RobotConfig] | None = None
    teleop: TeleoperatorConfig | dict[str, TeleoperatorConfig] | None = None
    processor: HILSerlProcessorConfig = HILSerlProcessorConfig()
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    name: str = "real_robot"
    root: str | None = None

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def _init_devices(self):
        assert self.robot is not None, "Robot config must be provided for real robot environment"
        assert self.teleop is not None, "Teleop config must be provided for real robot environment"

        # Handle multi robot configuration
        robot_dict = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        for name in robot_dict:
            robot_dict[name].cameras = {}
            robot_dict[name] = make_robot_from_config(robot_dict[name])
            robot_dict[name].connect()

        teleop_dict = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name in teleop_dict:
            teleop_dict[name] = make_teleoperator_from_config(teleop_dict[name])
            teleop_dict[name].connect()

        cameras = make_cameras_from_configs(self.cameras)

        # go through each processor and check if we need to turn scalar configs into configs for each robot
        for attr in ["observation", "gripper", "reset", "inverse_kinematics", "task_frame"]:
            if not isinstance(getattr(self.processor, attr), dict):
                setattr(self.processor, attr, {name: getattr(self.processor, attr) for name in robot_dict})

        return robot_dict, teleop_dict, cameras

    def make(self, device) -> tuple[RobotEnv, Any, Any]:
        robot_dict, teleop_dict, cameras = self._init_devices()

        use_gripper = {name: self.processor.gripper[name].use_gripper for name in self.processor}
        reset_pose = {name: self.processor.reset[name].reset_pose for name in self.processor}
        reset_time_s = {name: self.processor.reset[name].reset_time_s for name in self.processor}

        env = RobotEnv(
            robot=robot_dict,
            use_gripper=use_gripper,
            reset_pose=reset_pose,
            reset_time_s=reset_time_s,
            display_cameras=self.processor.display_cameras,
        )

        env_processor, action_processor = make_default_processors(env, teleop_dict, self, device)

        return env, env_processor, action_processor


@dataclass
class TFHilSerlRobotEnvConfig(HilSerlRobotEnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    def make(self, device) -> tuple[TaskFrameEnv, Any, Any]:
        robot_dict, teleop_dict, cameras = self._init_devices()

        task_frame = {name: self.processor.task_frame[name].command for name in self.processor.task_frame}
        control_mask = {name: self.processor.task_frame[name].control_mask for name in self.processor.task_frame}
        use_gripper = {name: self.processor.gripper[name].use_gripper for name in self.processor.gripper}
        reset_pose = {name: self.processor.reset[name].fixed_reset_joint_positions for name in self.processor.reset}
        reset_time_s = {name: self.processor.reset[name].reset_time_s for name in self.processor.reset}

        env = TaskFrameEnv(
            robot_dict=robot_dict,
            task_frame=task_frame,
            control_mask=control_mask,
            use_gripper=use_gripper,
            reset_pose=reset_pose,
            reset_time_s=reset_time_s,
            display_cameras=self.processor.display_cameras,
        )

        return env, *self._processors(env, teleop_dict, device)

    def _processors(self, env: TaskFrameEnv, teleoperators, device) -> tuple[
        DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
    ]:
        terminate_on_success = {name: self.processor.reset[name].terminate_on_success for name in self.processor.reset}
        stack_frames = {name: self.processor.observation[name].stack_frames for name in self.processor.observation}

        env_pipeline_steps = [
            VanillaTFFProcessorStep(
                add_ee_velocity_to_observation={name: self.processor.observation[name].add_ee_velocity_to_observation for name in env.robot_dict},
                add_ee_wrench_to_observation={name: self.processor.observation[name].add_ee_wrench_to_observation for name in env.robot_dict},
                ee_pos_mask=env.control_mask,
            )
        ]

        if self.processor.image_preprocessing:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=self.processor.image_preprocessing.crop_params_dict,
                    resize_size=self.processor.image_preprocessing.resize_size,
                )
            )

        if self.processor.control_time_s:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(
                    max_episode_steps=int(self.processor.control_time_s * self.fps)
                )
            )

        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                use_gripper={name: self.processor.gripper[name].use_gripper for name in env.robot_dict},
                penalty={name: self.processor.gripper[name].gripper_penalty for name in env.robot_dict},
                max_gripper_pos={name: self.processor.gripper[name].max_pos for name in env.robot_dict}
            )
        )

        if self.processor.reward_classifier:
            env_pipeline_steps.append(
                RewardClassifierProcessorStep(
                    pretrained_path=self.processor.reward_classifier.pretrained_path,
                    device=device,
                    success_threshold=self.processor.reward_classifier.success_threshold,
                    success_reward=self.processor.reward_classifier.success_reward,
                    terminate_on_success=any(terminate_on_success.values()),
                )
            )

        env_pipeline_steps.extend([AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=device)])

        action_pipeline_steps = [
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleoperators),
            AddTeleopEventsAsInfoStep(teleoperators=teleoperators),
            SixDofVelocityInterventionActionProcessorStep(
                use_gripper={name: self.processor.gripper[name].use_gripper for name in env.robot_dict},
                control_mask={name: self.processor.task_frame[name].control_mask for name in self.processor.task_frame},
                terminate_on_success=terminate_on_success,
            )
        ]

        return (
            DataProcessorPipeline(env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition),
            DataProcessorPipeline(action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition),
        )



