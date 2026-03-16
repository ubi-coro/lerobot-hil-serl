from dataclasses import dataclass, field, fields
from typing import Type

from lerobot.model.kinematics import RobotKinematics

from lerobot.envs import EnvConfig

from lerobot.cameras import CameraConfig, Camera

from lerobot.teleoperators import TeleoperatorConfig, Teleoperator, TeleopEvents

from lerobot.robots import RobotConfig, Robot

from lerobot.envs.factory import RobotEnvInterface
from lerobot.envs.robot_env.configuration_robot_env import RobotEnvConfig, is_union_with_dict
from lerobot.envs.tf_env import TaskFrameEnv
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep
)
from lerobot.processor.converters import identity_transition
from lerobot.processor.hil_processor import (
    AddFootswitchEventsAsInfoStep,
    AddKeyboardEventsAsInfoStep,
    DiscretizeGripperProcessorStep
)
from lerobot.processor.tf_processor import (
    VanillaTFFProcessorStep,
    SixDofVelocityInterventionActionProcessorStep
)
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.pipeline_features import PREFIXES_TO_STRIP, strip_prefix, create_initial_features
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame
from share.envs.manipulation_primitive.processor_steps import (
    InterventionActionProcessorStep,
    JointsToEEObservation,
    MatchTeleopToPolicyActionProcessorStep,
    RelativeFrameActionProcessor,
    RelativeFrameObservationProcessor,
    RobotActionToPolicyActionProcessorStep,
    ToJointActionProcessorStep,
    VanillaMPObservationProcessorStep,
)
from share.envs.utils import check_task_frame_robot, check_delta_teleoperator
from share.utils.kinematics import get_kinematics


@dataclass
class ImagePreprocessingConfig:
    """Camera crop/resize options applied inside the env processor."""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None  # cam_name -> (top, left, height, width)
    resize_size: tuple[int, int] | None = None
    filter_keys: list[str] | None = None
    display_cameras: bool = False


@dataclass
class KinematicsConfig:
    """Configuration for inverse kinematics processing."""

    enable: bool | dict[str, bool] = False
    use_virtual_reference: bool | dict[str, bool] = True
    urdf_path: str | dict[str, str | None] | None = None
    target_frame_name: str | dict[str, str | None] | None = None
    end_effector_bounds: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None = None
    end_effector_step_sizes: dict[str, float] | dict[str, dict[str, float]] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_position_to_observation: bool | dict[str, bool] = True
    add_joint_velocity_to_observation: bool | dict[str, bool] = False
    add_current_to_observation: bool | dict[str, bool] = False
    add_ee_pos_to_observation: bool | dict[str, bool] = False
    add_ee_velocity_to_observation: bool | dict[str, bool] = False
    add_ee_wrench_to_observation: bool | dict[str, bool] = False
    stack_frames: int | dict[str, int] = 0
    relative_ee_pos: bool | dict[str, bool] = True


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    enable: bool | dict[str, bool] = False
    discretize: bool | dict[str, bool] = False
    max_pos: float | dict[str, float] = 1.0
    min_pos: float | dict[str, float] = 0.0
    static_pos: float | dict[str, float] = 0.0  # sent if enable = False
    penalty: float | dict[str, float | None] | None = None


@dataclass
class EventConfig:
    """Mappings from teleop inputs to structured intervention events."""

    key_mapping: dict[TeleopEvents, dict] = field(default_factory=lambda: {})
    foot_switch_mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=lambda: {})


@dataclass
class HookConfig:
    """Optional processor timing hook configuration."""

    time_env_processor: bool = False
    time_action_processor: bool = False
    log_every: int = 10


@dataclass
class ManipulationPrimitiveProcessorConfig:
    """Top-level processor settings shared across robots and per robot."""

    # for all arms
    control_time_s: float = 10.0
    fps: float = 10.0
    image_preprocessing: ImagePreprocessingConfig | None = None
    events: EventConfig = field(default_factory=EventConfig)
    hooks: HookConfig = field(default_factory=HookConfig)

    # per arm
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)


@EnvConfig.register_subclass(name="manipulation_primitive")
@dataclass
class ManipulationPrimitiveConfig(EnvConfig):
    """Configuration for one manipulation primitive in a primitive net."""
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    processor: ManipulationPrimitiveProcessorConfig = field(default_factory=ManipulationPrimitiveProcessorConfig)
    is_terminal: bool = False

    _kinematics_solver: dict = field(default_factory=dict)
    _joint_names: dict = field(default_factory=dict)

    @property
    def gym_kwargs(self) -> dict:
        """Extra kwargs forwarded to gym environment creation."""
        return {}

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu"
    ):
        """Build the env and both processing pipelines."""
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)  # todo: fix initial_features

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = ManipulationPrimitive(task_frame=self.task_frame, robot_dict=robot_dict, cameras=cameras, display_cameras=display_cameras)

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def make_action_processor(self, robot_dict, teleop_dict, device) -> DataProcessorPipeline:
        """Create the action-side processing pipeline."""
        action_pipeline_steps = []

        # events
        if self.processor.events.key_mapping:
            action_pipeline_steps.append(AddKeyboardEventsAsInfoStep(mapping=self.processor.events.key_mapping))

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(
                AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping))

        try:
            action_pipeline_steps.append(AddTeleopEventsAsInfoStep(teleoperators=teleop_dict))
        except TypeError:
            pass

        action_pipeline_steps.extend([
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleop_dict),  # this checks events and should come after Add*EventsAsInfoStep's

            # make teleop action match policy based on task frame (treat delta ee / vel / force the same):
            # teleop Q:
            # policy delta ee / vel / force: FK + differentiate
            # abs ee: FK
            # delta Q: differentiate
            # abs Q. noop
            #
            # teleop delta ee:
            # policy delta ee / vel / force: noop
            # abs ee: integrate
            # delta Q: IK
            # abs Q: integrate + IK
            MatchTeleopToPolicyActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
                kinematics=self._kinematics_solver,
                use_virtual_reference=self.processor.kinematics.use_virtual_reference
            ),

            # scatter policy / teleop action (depending on is-intervention event) into full task frame action target
            # send feedback to teleoperators if they need it
            InterventionActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
            ),

            #DiscretizeGripperProcessorStep(
            #    gripper_idc=self.gripper_idc,
            #    min_pos=self.processor.gripper.min_pos,
            #    max_pos=self.processor.gripper.max_pos
            #),
        ])

        # action in ee frame instead of in world frame
        if self._any_enabled(self.processor.observation.relative_ee_pos):
            action_pipeline_steps.append(
                RelativeFrameActionProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        is_task_frame_robot = check_task_frame_robot(robot_dict)
        if not all(is_task_frame_robot.values()):
            # after this processor, the action must a dictionary of joint names
            # policy_action: delta vel ->

            action_pipeline_steps.append(
                ToJointActionProcessorStep(
                    is_task_frame_robot=is_task_frame_robot,
                    task_frame=self.task_frame,
                    kinematics=self._kinematics_solver,
                    joint_names=self._joint_names,
                    use_virtual_reference=self.processor.kinematics.use_virtual_reference
                )
            )

        # up until here, the action is a dictionary, this turns it into one tensor
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=self._joint_names))

        # timing hooks
        if self.processor.hooks.time_action_processor:
            from share.utils.control_utils import make_step_timing_hooks
            action_before_hooks, action_after_hooks = make_step_timing_hooks(
                pipeline_steps=action_pipeline_steps,
                label="action",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False
            )
        else:
            action_before_hooks, action_after_hooks = [], []

        return DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=action_before_hooks, after_step_hooks=action_after_hooks
        )

    def make_env_processor(self, device: str = "cpu") -> DataProcessorPipeline:
        """Create the observation/reward-side processing pipeline."""
        env_pipeline_steps = []

        # obs is dict with keys {robot_name}.{axis/joint}.{pos/vel/ee_pos/ee_vel/ee_wrench} | {OBS_IMAGES}{camera_name}
        # {axis} is in {x,y,z,wx,wy,wz}
        if self._kinematics_solver:
            # for all robots that have a solver, we want fetch their joints and add {robot_name}.{axis}.ee_pos to the obs
            env_pipeline_steps.append(
                JointsToEEObservation(
                    kinematics=self._kinematics_solver,
                    motor_names=self._joint_names,
                )
            )

        # action relative to starting pose
        if self._any_enabled(self.processor.observation.relative_ee_pos):
            env_pipeline_steps.append(
                RelativeFrameObservationProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        if self.processor.image_preprocessing:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=self.processor.image_preprocessing.crop_params_dict,
                    resize_size=self.processor.image_preprocessing.resize_size,
                    filter_keys=self.processor.image_preprocessing.filter_keys
                )
            )

        #env_pipeline_steps.append(
        #    GripperPenaltyProcessorStep(
        #        max_gripper_pos=self.processor.gripper.max_pos,
        #        penalty=self.processor.gripper.penalty,
        #    )
        #)

        env_pipeline_steps.extend([
            # builds OBS_STATE based on what we want to have in there
            # if obs has no joint vel and we want it, compute numerically
            # same for ee_vel
            VanillaMPObservationProcessorStep(
                device=device,
                gripper_enable=self.processor.gripper.enable,
                add_joint_position_to_observation=self.processor.observation.add_joint_position_to_observation,
                add_joint_velocity_to_observation=self.processor.observation.add_joint_velocity_to_observation,
                add_current_to_observation=self.processor.observation.add_current_to_observation,
                add_ee_pos_to_observation=self.processor.observation.add_ee_pos_to_observation,
                add_ee_velocity_to_observation=self.processor.observation.add_ee_velocity_to_observation,
                add_ee_wrench_to_observation=self.processor.observation.add_ee_wrench_to_observation,
            ),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device)
        ])

        # timing hooks
        if self.processor.hooks.time_env_processor:
            from share.utils.control_utils import make_step_timing_hooks
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

    @staticmethod
    def _any_enabled(value: bool | dict[str, bool]) -> bool:
        if isinstance(value, dict):
            return any(bool(v) for v in value.values())
        return bool(value)

    def validate(self, robot_dict, teleop_dict):
        """Validate modality compatibility and initialize kinematics state."""
        is_task_frame_robot = check_task_frame_robot(robot_dict)
        is_delta_teleoperator = check_delta_teleoperator(teleop_dict)

        # go through each processor and check if we need to turn scalar configs into configs for each robot
        for attr in ["observation", "gripper", "kinematics"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in robot_dict})
            setattr(self.processor, attr, _attr)

        # Set up kinematics solver if inverse kinematics is configured
        for name, robot in robot_dict.items():
            if self.processor.kinematics.enable[name] and hasattr(robot, "bus"):
                self._joint_names[name] = list(robot.bus.motors.keys())
                self._kinematics_solver[name] = get_kinematics(
                    robot_name=robot.name,
                    urdf_path=self.processor.kinematics.urdf_path[name],
                    target_frame_name=self.processor.kinematics.target_frame_name[name],
                    joint_names=self._joint_names[name],
                )

        # checks per robot
        for name, frame in self.task_frame.items():
            if name not in robot_dict:
                raise ValueError(f"Missing robot for task-frame entry '{name}'.")

            if name not in is_delta_teleoperator:
                raise ValueError(
                    f"Missing teleoperator for '{name}' while validating task-frame teleop compatibility."
                )

            # ENV-101: learnable VEL/FORCE axes require delta teleoperator input.
            for axis in frame.learnable_axis_indices:
                if frame.control_mode[axis] in {ControlMode.VEL, ControlMode.FORCE} and not is_delta_teleoperator[name]:
                    raise ValueError(
                        "Adaptive task-frame axes with VEL/FORCE control require a delta teleoperator. "
                        f"Got robot='{name}', axis={axis}, control_mode={frame.control_mode[axis].name}, "
                        "teleoperator_kind='absolute'."
                    )

            # ENV-102: JOINT-space and joint-only robots must only receive POS axis modes.
            if frame.space == ControlSpace.JOINT:
                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "ControlSpace.JOINT only supports POS axis modes. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            if not is_task_frame_robot[name]:
                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "Joint-only robots only support POS axis modes in this pipeline. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            # ENV-102: TASK-space with absolute-joint teleop or joint-only robot requires kinematics.
            requires_kinematics = frame.space == ControlSpace.TASK and (
                not is_delta_teleoperator[name] or not is_task_frame_robot[name]
            )
            if requires_kinematics and not self.processor.kinematics.enable[name]:
                raise ValueError(
                    "Kinematics must be enabled for TASK-space control when teleop/robot modalities require FK/IK. "
                    f"Set processor.kinematics.enable['{name}']=True."
                )

            if requires_kinematics and name not in self._kinematics_solver:
                raise ValueError(
                    "Kinematics are required but no solver was initialized. "
                    f"Check kinematics config and robot interface for '{name}' "
                    "(urdf_path/target_frame_name/bus availability)."
                )

        # if gripper.enable but the robot has no GRIPPER_KEY action feature, disable

    def infer_features(self, robot_dict, cameras):
        """Infer policy-visible feature specs from configured processors."""
        # process features with respective pipeline
        # get initial obs features from robot_dict instead
        initial_features = {}
        for cam_key, cam in cameras:
            initial_features[f"{OBS_IMAGES}{cam_key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=cam.async_read().shape)

        for name in robot_dict:
            for k, v in robot_dict[name].get_observation().items():
                if isinstance(v, float):
                    shape = (1, )
                elif hasattr(v, "shape"):
                    shape = v.shape
                elif hasattr(v, "__iter__"):
                    shape = (len(v), )
                else:
                    raise ValueError(f"Unknown type for observation {name}.{k}: {type(v)}")
                initial_features[f"{name}.{k}"] = PolicyFeature(type=FeatureType.STATE, shape=shape)

        initial_features = create_initial_features(observation=initial_features)
        env_processor = self.make_env_processor()
        pipeline_features = env_processor.transform_features(initial_features)
        obs_features = pipeline_features[PipelineFeatureType.OBSERVATION]

        action_dim = sum(frame.policy_action_dim for frame in self.task_frame.values())

        # expose state, action and visual features
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            OBS_STATE: pipeline_features[PipelineFeatureType.OBSERVATION][OBS_STATE]
        }
        for key, ft in obs_features.items():
            if ft.type == FeatureType.VISUAL:
                key = strip_prefix(key, PREFIXES_TO_STRIP)
                self.features[f"{OBS_IMAGES}.{key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=ft.shape)
