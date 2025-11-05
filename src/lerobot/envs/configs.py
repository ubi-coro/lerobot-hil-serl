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
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any

import draccus

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.tf_env import TaskFrameEnv
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    ForwardKinematicsJointsToEEObservation,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    VanillaObservationProcessorStep,
)
from lerobot.processor.converters import identity_transition
from lerobot.processor.hil_processor import AddFootswitchEventsAsInfoStep, AddKeyboardEventsAsInfoStep
from lerobot.processor.robot_kinematic_processor import EEReferenceAndDelta, EEBoundsAndSafety, GripperVelocityToJoint, \
    InverseKinematicsRLStep
from lerobot.processor.tff_processor import VanillaTFFProcessorStep, SixDofVelocityInterventionActionProcessorStep
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.envs.robot_env import RobotEnv
from lerobot.robots.ur.tff_controller import TaskFrameCommand
from lerobot.share.utils import is_union_with_dict
from lerobot.teleoperators import make_teleoperator_from_config, TeleopEvents
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


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
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

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
    observation_height: int = 480
    observation_width: int = 640
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
            self.features["top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )

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
    observation_height: int = 384
    observation_width: int = 384
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
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
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

    enable: bool | dict[str, bool] = False
    urdf_path: str | dict[str, str | None] | None = None
    target_frame_name: str | dict[str, str | None] | None = None
    end_effector_bounds: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None = None
    end_effector_step_sizes: dict[str, float] | dict[str, dict[str, float]] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_velocity_to_observation: bool | dict[str, bool] = False
    add_current_to_observation: bool | dict[str, bool] = False
    add_ee_velocity_to_observation: bool | dict[str, bool] = False
    add_ee_wrench_to_observation: bool | dict[str, bool] = False
    stack_frames: int | dict[str, int] = 0


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    use_gripper: bool | dict[str, bool] = False
    gripper_penalty: float | dict[str, float] = 0.0
    max_pos: float | dict[str, float] = 100.0


@dataclass
class ResetConfig:
    """Configuration for environment reset behavior."""

    fixed_reset_joint_positions: Any | dict[str, Any | None] | None = None
    terminate_on_success: bool | dict[str, bool] = True
    reset_time_s: float = 5.0
    teleop_on_reset: bool = False


@dataclass
class TaskFrameConfig:
    command: TaskFrameCommand | dict[str, TaskFrameCommand] = field(default_factory=TaskFrameCommand.make_default_cmd)
    control_mask: list[int] | dict[str, list[int]] = field(default_factory=lambda: [1] * 6)


@dataclass
class EventConfig:
    key_mapping: dict[TeleopEvents, dict] = field(default_factory=lambda: {})
    foot_switch_mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=lambda: {})


@dataclass
class HookConfig:
    time_env_processor: bool = False
    time_action_processor: bool = False
    log_every: int = 10


@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_time_s: float | None = None
    display_cameras: bool = False

    image_preprocessing: ImagePreprocessingConfig | None = None
    reward_classifier: RewardClassifierConfig | None = None
    events: EventConfig = EventConfig()
    hooks: HookConfig = HookConfig()

    observation: ObservationConfig = ObservationConfig()
    gripper: GripperConfig = GripperConfig()
    reset: ResetConfig = ResetConfig()
    inverse_kinematics: InverseKinematicsConfig = InverseKinematicsConfig()
    task_frame: TaskFrameConfig = TaskFrameConfig()


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
    observation_height: int = 360
    observation_width: int = 360
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
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v2"  # add all tasks
    fps: int = 80
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    multitask_eval: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}",
            "pixels/top": f"{OBS_IMAGE}",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

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

    def __post_init__(self):
        self.features =

    @cached_property
    def action_dim(self):
        robot_dict = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        gripper = self.processor.gripper.use_gripper

        _action_dim = 0
        for name in robot_dict:
            gripper = gripper if isinstance(gripper, bool) else gripper[name]
            _action_dim += 6 # we cannot access this here, so we need to make assumptions, ie 6 joints / dofs

            if gripper:
                _action_dim += 1

        return _action_dim

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def make(self, device: str = "cpu") -> tuple[RobotEnv, Any, Any]:
        robot_dict, teleop_dict, cameras = self._init_devices()

        if self.processor.reset.teleop_on_reset:
            self.processor.reset.fixed_reset_joint_positions = {name: None for name in robot_dict}

        env = RobotEnv(
            robot_dict=robot_dict,
            cameras=cameras,
            use_gripper=self.processor.gripper.use_gripper,
            reset_pose=self.processor.reset.fixed_reset_joint_positions,
            reset_time_s=self.processor.reset.reset_time_s,
            display_cameras=self.processor.display_cameras,
        )

        return env, *self._processors(env, teleop_dict, device)

    def _init_devices(self):
        assert self.robot is not None, "Robot config must be provided for real robot environment"
        assert self.teleop is not None, "Teleop config must be provided for real robot environment"

        # Handle multi robot configuration
        robot_dict = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        for name in robot_dict:
            robot_dict[name].cameras = {}
            robot_dict[name] = make_robot_from_config(robot_dict[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name in teleop_dict:
            teleop_dict[name] = make_teleoperator_from_config(teleop_dict[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.cameras)
        for name in cameras:
            cameras[name].connect()

        # go through each processor and check if we need to turn scalar configs into configs for each robot
        for attr in ["observation", "gripper", "reset", "inverse_kinematics", "task_frame"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in robot_dict})
            setattr(self.processor, attr, _attr)

        return robot_dict, teleop_dict, cameras

    def _processors(self, env: RobotEnv, teleoperators, device):
        """
        if self.name == "gym_hil":
            action_pipeline_steps = [
                InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
                Torch2NumpyActionProcessorStep(),
            ]

            env_pipeline_steps = [
                Numpy2TorchActionProcessorStep(),
                VanillaObservationProcessorStep(),
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]

            return DataProcessorPipeline(
                steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
            ), DataProcessorPipeline(
                steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
            )

        assert isinstance(env, RobotEnv)
        """

        # Full processor pipeline for real robot environment
        # Get robot and motor information for kinematics
        joint_names = env._joint_names_dict

        env_pipeline_steps = [
            VanillaObservationProcessorStep(device=device),
            JointVelocityProcessorStep(
                enable=self.processor.observation.add_joint_velocity_to_observation,
                dt=1.0 / self.fps
            ),
            MotorCurrentProcessorStep(
                enable=self.processor.observation.add_current_to_observation,
                robot_dict=env.robot_dict
            )
        ]

        # Set up kinematics solver if inverse kinematics is configured
        kinematics_solver = {}
        for name in env.robot_dict:
            if self.processor.inverse_kinematics.enable[name]:
                kinematics_solver[name] = RobotKinematics(
                    urdf_path=self.processor.inverse_kinematics[name].urdf_path,
                    target_frame_name=self.processor.inverse_kinematics[name].target_frame_name,
                    joint_names=joint_names[name],
                )
        if kinematics_solver:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                )
            )

        if self.processor.image_preprocessing:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=self.processor.image_preprocessing.crop_params_dict,
                    resize_size=self.processor.image_preprocessing.resize_size,
                )
            )

        # Add time limit processor if reset config exists
        if self.processor.control_time_s:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(max_episode_steps=int(self.processor.control_time_s * self.fps))
            )

        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                use_gripper=self.processor.gripper.use_gripper,
                penalty=self.processor.gripper.gripper_penalty,
                max_gripper_pos=self.processor.gripper.max_pos
            )
        )

        if (
            self.processor.reward_classifier is not None
            and self.processor.reward_classifier.pretrained_path is not None
        ):
            env_pipeline_steps.append(
                RewardClassifierProcessorStep(
                    pretrained_path=self.processor.reward_classifier.pretrained_path,
                    device=device,
                    success_threshold=self.processor.reward_classifier.success_threshold,
                    success_reward=self.processor.reward_classifier.success_reward,
                    terminate_on_success=self.processor.reset.terminate_on_success
                )
            )

        action_pipeline_steps = [AddTeleopEventsAsInfoStep(teleoperators=teleoperators)]

        if self.processor.events.key_mapping:
            action_pipeline_steps.append(AddKeyboardEventsAsInfoStep(mapping=self.processor.events.key_mapping))

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping))

        action_pipeline_steps.append(AddTeleopActionAsComplimentaryDataStep(teleoperators=teleoperators))

        action_pipeline_steps.append(
            InterventionActionProcessorStep(
                teleoperators=teleoperators,
                use_gripper=self.processor.gripper.use_gripper,
                terminate_on_success=self.processor.reset.terminate_on_success
            )
        )

        # Replace InverseKinematicsProcessor with new kinematic processors
        if kinematics_solver:
            inverse_kinematics_steps = [
                MapTensorToDeltaActionDictStep(
                    use_gripper=self.processor.gripper.use_gripper if self.processor.gripper is not None else False
                ),
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=kinematics_solver,
                    end_effector_step_sizes=self.processor.inverse_kinematics.end_effector_step_sizes,
                    motor_names=joint_names,
                    use_latched_reference=False,
                    use_ik_solution=True,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds=self.processor.inverse_kinematics.end_effector_bounds,
                ),
                GripperVelocityToJoint(
                    clip_max=self.processor.max_gripper_pos,
                    speed_factor=1.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsRLStep(
                    kinematics=kinematics_solver, motor_names=joint_names, initial_guess_current_joints=False
                ),
            ]
            action_pipeline_steps.extend(inverse_kinematics_steps)
            action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=joint_names))

        # time the env pipeline with hooks
        if self.processor.hooks.time_action_processor or self.processor.hooks.time_env_processor:
            from lerobot.utils.control_utils import make_step_timing_hooks

        if self.processor.hooks.time_env_processor:
            env_before_hooks, env_after_hooks = make_step_timing_hooks(
                pipeline_steps=env_pipeline_steps,
                label="env",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False,
            )
        else:
            env_before_hooks, env_after_hooks = [], []

        if self.processor.hooks.time_action_processor:
            action_before_hooks, action_after_hooks = make_step_timing_hooks(
                pipeline_steps=action_pipeline_steps,
                label="action",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False,
            )
        else:
            action_before_hooks, action_after_hooks = [], []

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=env_before_hooks, after_step_hooks=env_after_hooks
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=action_before_hooks, after_step_hooks=action_after_hooks
        )


@dataclass
class TFHilSerlRobotEnvConfig(HilSerlRobotEnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""
    @cached_property
    def action_dim(self):
        masks = self.processor.task_frame.control_mask
        masks = masks if isinstance(masks, dict) else {DEFAULT_ROBOT_NAME: masks}
        gripper = self.processor.gripper.use_gripper
        gripper = gripper if isinstance(gripper, dict) else {DEFAULT_ROBOT_NAME: gripper}
        return sum([sum(m) for m in masks.values()]) + sum(gripper.values())

    def make(self, device: str = "cpu") -> tuple[TaskFrameEnv, Any, Any]:
        robot_dict, teleop_dict, cameras = self._init_devices()

        env = TaskFrameEnv(
            robot_dict=robot_dict,
            cameras=cameras,
            task_frame=self.processor.task_frame.command,
            control_mask=self.processor.task_frame.control_mask,
            use_gripper=self.processor.gripper.use_gripper,
            reset_pose=self.processor.reset.fixed_reset_joint_positions,
            reset_time_s=self.processor.reset.reset_time_s,
            display_cameras=self.processor.display_cameras,
        )

        return env, *self._processors(env, teleop_dict, device)

    def _processors(self, env: TaskFrameEnv, teleoperators, device) -> tuple[
        DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
    ]:
        env_pipeline_steps = [
            VanillaTFFProcessorStep(
                add_ee_velocity_to_observation=self.processor.observation.add_ee_velocity_to_observation,
                add_ee_wrench_to_observation=self.processor.observation.add_ee_wrench_to_observation,
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
                use_gripper=self.processor.gripper.use_gripper,
                penalty=self.processor.gripper.gripper_penalty,
                max_gripper_pos=self.processor.gripper.max_pos
            )
        )

        if self.processor.reward_classifier:
            env_pipeline_steps.append(
                RewardClassifierProcessorStep(
                    pretrained_path=self.processor.reward_classifier.pretrained_path,
                    device=device,
                    success_threshold=self.processor.reward_classifier.success_threshold,
                    success_reward=self.processor.reward_classifier.success_reward,
                    terminate_on_success=any(self.processor.reset.terminate_on_success.values()),
                )
            )

        env_pipeline_steps.extend([AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=device)])

        action_pipeline_steps = [
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleoperators),
            AddTeleopEventsAsInfoStep(teleoperators=teleoperators),
            AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping),
            SixDofVelocityInterventionActionProcessorStep(
                use_gripper=self.processor.gripper.use_gripper,
                control_mask=self.processor.task_frame.control_mask,
                terminate_on_success=self.processor.reset.terminate_on_success,
            )
        ]

        return (
            DataProcessorPipeline(env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition),
            DataProcessorPipeline(action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition),
        )



