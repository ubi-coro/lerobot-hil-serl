import types
from dataclasses import dataclass, field, fields
from typing import Tuple, Any, Type, get_origin, Union, get_args

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature, PipelineFeatureType
from lerobot.datasets.pipeline_features import strip_prefix, PREFIXES_TO_STRIP
from lerobot.envs import EnvConfig
from lerobot.envs.configs import HILSerlProcessorConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.envs.robot_env.modeling_robot_env import RobotEnv
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
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
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, DEFAULT_ROBOT_NAME


def is_union_with_dict(field_type) -> bool:
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False


@EnvConfig.register_subclass(name="robot_env")
@dataclass
class RobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | dict[str, RobotConfig] | None = None
    teleop: TeleoperatorConfig | dict[str, TeleoperatorConfig] | None = None
    processor: HILSerlProcessorConfig = HILSerlProcessorConfig()
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    stats: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        # Handle multi robot configuration
        self.robot = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        self.teleop = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name in self.robot:
            self.robot[name].cameras = {}

        self._validate_config()
        self._make_features()

    @property
    def gym_kwargs(self) -> dict:
        return {}

    @property
    def env_cls(self) -> Type[RobotEnvInterface]:
        return RobotEnv

    @property
    def env_kwargs(self) -> dict[str, Any]:
        return {}

    @property
    def initial_features(self):
        return self.env_cls.get_features_from_cfg(self)

    @property
    def gripper_idc(self)-> dict[str, int | None]:
        gripper = self.processor.gripper.use_gripper
        gripper = gripper if isinstance(gripper, dict) else {name: gripper for name in self.robot}

        _gripper_idc = {name: None for name in gripper}
        idx = 0
        for name in gripper:
            idx += 6  # we cannot access this here, so we need to make assumptions, ie 6 joints

            if gripper[name]:
                _gripper_idc[name] = idx
                idx += 1

        return _gripper_idc

    def make(self, device: str = "cpu") -> Tuple[RobotEnvInterface, DataProcessorPipeline, DataProcessorPipeline]:
        robot_dict, teleop_dict, cameras = self.connect()

        env = self.env_cls(
            robot_dict=robot_dict,
            cameras=cameras,
            processor=self.processor,
            **self.env_kwargs
        )

        env_processor = self.make_env_processor(device, env=env)
        action_processor = self.make_action_processor(teleop_dict, device)
        return env, env_processor, action_processor

    def connect(self):
        assert self.robot is not None, "Robot config must be provided for real robot environment"

        # Handle multi robot configuration
        robot_dict = {}
        for name in self.robot:
            robot_dict[name] = make_robot_from_config(self.robot[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = {}
        for name in self.teleop:
            teleop_dict[name] = make_teleoperator_from_config(self.teleop[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras

    def make_action_processor(self, teleoperators, device) -> DataProcessorPipeline:
        action_pipeline_steps = []

        try:
            AddTeleopEventsAsInfoStep(teleoperators=teleoperators)
        except TypeError:
            pass

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
                    use_ik_solution=True
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

    def make_env_processor(self, device, env: RobotEnvInterface | None = None) -> DataProcessorPipeline:
        # Full processor pipeline for real robot environment
        # Get robot and motor information for kinematics

        env_pipeline_steps: list = [
            VanillaObservationProcessorStep(device=device),
            JointVelocityProcessorStep(
                enable=self.processor.observation.add_joint_velocity_to_observation,
                dt=1.0 / self.fps
            ),
            MotorCurrentProcessorStep(
                enable=self.processor.observation.add_current_to_observation,
                robot_dict={}, # todo: put this into complementary info
            )
        ]

        # Set up kinematics solver if inverse kinematics is configured
        if self.kinematics_solver:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=self.kinematics_solver,
                    motor_names=self.joint_names,
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
            env_pipeline_steps.append(TimeLimitProcessorStep(max_episode_steps=int(self.processor.control_time_s * self.fps)))

        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                gripper_idc=self.gripper_idc,
                penalty=self.processor.gripper.penalty,
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

    def _validate_config(self):
        # go through each processor and check if we need to turn scalar configs into configs for each robot
        for attr in ["observation", "gripper", "reset", "inverse_kinematics", "task_frame"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in self.robot})
            setattr(self.processor, attr, _attr)

        # Set up kinematics solver if inverse kinematics is configured
        self.kinematics_solver = {}
        self.joint_names = {}
        for name in self.robot:
            if self.processor.inverse_kinematics.enable[name]:
                self.joint_names[name] = list(self.robot[name].bus.motors.keys())
                self.kinematics_solver[name] = RobotKinematics(
                    urdf_path=self.processor.inverse_kinematics[name].urdf_path,
                    target_frame_name=self.processor.inverse_kinematics[name].target_frame_name,
                    joint_names=self.joint_names[name],
                )

        # set reset position to None if manual teleop reset should be performed instead
        if self.processor.reset.teleop_on_reset:
            self.processor.reset.fixed_reset_joint_positions = {name: None for name in self.robot}


    def _make_features(self):
        # process features with respective pipeline
        env_processor = self.make_env_processor(device="cpu", env=None)
        pipeline_features = env_processor.transform_features(self.initial_features)
        obs_features = pipeline_features[PipelineFeatureType.OBSERVATION]

        # expose state, action and visual features
        self.features = {
            ACTION: pipeline_features[PipelineFeatureType.ACTION][ACTION],
            OBS_STATE: pipeline_features[PipelineFeatureType.OBSERVATION][OBS_STATE]
        }
        for key, ft in obs_features.items():
            if ft.type == FeatureType.VISUAL:
                key = strip_prefix(key, PREFIXES_TO_STRIP)
                self.features[f"{OBS_IMAGES}.{key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=ft.shape)
