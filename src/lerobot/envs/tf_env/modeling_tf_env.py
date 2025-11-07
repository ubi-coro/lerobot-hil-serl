import abc
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Tuple

import draccus

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature, PipelineFeatureType
from lerobot.datasets.pipeline_features import create_initial_features, strip_prefix, PREFIXES_TO_STRIP
from lerobot.envs import EnvConfig
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
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, DEFAULT_ROBOT_NAME, REWARD, DONE


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
        assert self.robot is not None, "Robot config must be provided for real robot environment"
        assert self.teleop is not None, "Teleop config must be provided for real robot environment"

        # Handle multi robot configuration
        self.robot = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        for name in self.robot:
            self.robot[name].cameras = {}  # remove per-robot cameras, bc the environment manages its own set of cameras

        # Handle multi teleop configuration
        self.teleop = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}

        # go through each processor and check if we need to turn scalar configs into configs for each robot
        for attr in ["observation", "gripper", "reset", "inverse_kinematics", "task_frame"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in self.robot})
            setattr(self.processor, attr, _attr)

        # Set up kinematics solver if inverse kinematics is configured
        self.joint_names = {name: list(self.robot[name].motors.keys()) for name in self.robot}
        self.kinematics_solver = {}
        for name in self.robot:
            if self.processor.inverse_kinematics.enable[name]:
                self.kinematics_solver[name] = RobotKinematics(
                    urdf_path=self.processor.inverse_kinematics[name].urdf_path,
                    target_frame_name=self.processor.inverse_kinematics[name].target_frame_name,
                    joint_names=self.joint_names[name],
                )

        # set reset position to None if manual teleop reset should be performed instead
        if self.processor.reset.teleop_on_reset:
            self.processor.reset.fixed_reset_joint_positions = {name: None for name in self.robot}

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def make(self, device: str = "cpu") -> Tuple[RobotEnv, DataProcessorPipeline, DataProcessorPipeline]:
        robot_dict, teleop_dict, cameras = self.connect()

        env = RobotEnv(
            robot_dict=robot_dict,
            cameras=cameras,
            use_gripper=self.processor.gripper.use_gripper,
            reset_pose=self.processor.reset.fixed_reset_joint_positions,
            reset_time_s=self.processor.reset.reset_time_s,
            display_cameras=self.processor.display_cameras,
        )

        env_processor = self.make_env_processor(env, device)
        action_processor = self.make_action_processor(teleop_dict, device)
        self.make_features(env, env_processor)

        return env, env_processor, action_processor

    def connect(self):
        robot_dict = {}
        for name in robot_dict:
            robot_dict[name] = make_robot_from_config(self.robot[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = {}
        for name in teleop_dict:
            teleop_dict[name] = make_teleoperator_from_config(self.teleop[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras

    def make_action_processor(self, teleoperators, device) -> DataProcessorPipeline:

        # Primary action pipeline: event handling, teleop action read and intervention logic
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

        # Kinematics
        if self.kinematics_solver:
            inverse_kinematics_steps = [
                MapTensorToDeltaActionDictStep(
                    use_gripper=self.processor.gripper.use_gripper if self.processor.gripper is not None else False
                ),
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=self.kinematics_solver,
                    end_effector_step_sizes=self.processor.inverse_kinematics.end_effector_step_sizes,
                    motor_names=self.joint_names,
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
                    kinematics=self.kinematics_solver, motor_names=self.joint_names, initial_guess_current_joints=False
                ),
            ]
            action_pipeline_steps.extend(inverse_kinematics_steps)
            action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=self.joint_names))

        if self.processor.hooks.time_action_processor:
            from lerobot.utils.control_utils import make_step_timing_hooks
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
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=action_before_hooks, after_step_hooks=action_after_hooks
        )

    def make_env_processor(self, env, device) -> DataProcessorPipeline:

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

        if self.kinematics_solver:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=self.kinematics_solver,
                    motor_names=list(self.joint_names.values()),
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

        if self.processor.hooks.time_env_processor:
            from lerobot.utils.control_utils import make_step_timing_hooks
            env_before_hooks, env_after_hooks = make_step_timing_hooks(
                pipeline_steps=env_pipeline_steps,
                label="env",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False,
            )
        else:
            env_before_hooks, env_after_hooks = [], []

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=env_before_hooks, after_step_hooks=env_after_hooks
        )

    def make_features(self, env, env_processor):
        # build initial features from gym spaces
        initial_obs_features = {}
        for name, space in env.observation_space.items():
            initial_obs_features[name] = PolicyFeature(
                type=FeatureType.VISUAL if len(space.shape) == 3 else FeatureType.STATE,
                shape=space.shape,
            )

        # build action dim
        gripper = self.processor.gripper.use_gripper
        action_dim = 0
        for name in self.robot:
            gripper = gripper if isinstance(gripper, bool) else gripper[name]
            action_dim += 6 # we cannot access this here, so we need to make assumptions, ie 6 joints / dofs

            if gripper:
                action_dim += 1

        # process features with respective pipeline
        obs_features = env_processor.transform_features(create_initial_features(observation=initial_obs_features))[PipelineFeatureType.OBSERVATION]

        # from pipeline features to huggingface features
        self.features = {
            ACTION: {"dtype": "float32", "shape": (action_dim,)},
            OBS_STATE: {"dtype": "float32", "shape": obs_features[OBS_STATE].shape},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None}
        }

        # add visual features
        for key, ft in obs_features.items():
            if ft.type == FeatureType.VISUAL:
                key = strip_prefix(key, PREFIXES_TO_STRIP)
                self.features[f"{OBS_IMAGES}.{key}"] = {
                    "dtype": "video",
                    "shape": ft.shape,
                    "names": ["channels", "height", "width"],
                }


