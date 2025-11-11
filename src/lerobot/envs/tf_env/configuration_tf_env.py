from dataclasses import dataclass
from typing import Type

from lerobot.envs.factory import RobotEnvInterface
from lerobot.envs.robot_env.configuration_robot_env import RobotEnvConfig
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
from lerobot.processor.hil_processor import AddFootswitchEventsAsInfoStep, AddKeyboardEventsAsInfoStep, \
    GripperOffsetProcessorStep
from lerobot.processor.tff_processor import VanillaTFFProcessorStep, SixDofVelocityInterventionActionProcessorStep, \
    ActionScalingProcessorStep
from lerobot.utils.constants import ACTION


@dataclass
class TFRobotEnvConfig(RobotEnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    def __post_init__(self):
        super().__post_init__()

        # validate action scaling
        action_dim = self.features[ACTION].shape[0]

        if self.processor.task_frame.action_scale is None:
            self.processor.task_frame.action_scale = 1.0

        if isinstance(self.processor.task_frame.action_scale, float):
            s = self.processor.task_frame.action_scale
            self.processor.task_frame.action_scale = [s] * action_dim

        assert action_dim == len(self.processor.task_frame.action_scale)

    @property
    def env_cls(self) -> Type[RobotEnvInterface]:
        return TaskFrameEnv

    @property
    def gripper_idc(self) -> dict[str, int | None]:
        masks = self.processor.task_frame.control_mask
        masks = masks if isinstance(masks, dict) else {name: masks for name in self.robot}
        gripper = self.processor.gripper.use_gripper
        gripper = gripper if isinstance(gripper, dict) else {name: masks for name in self.robot}

        _gripper_idc = {name: None for name in masks}
        idx = 0
        for name in masks:
            idx += sum(masks[name])

            if gripper[name]:
                _gripper_idc[name] = idx
                idx += 1

        return _gripper_idc

    def make_action_processor(self, teleoperators, device) -> DataProcessorPipeline:
        action_pipeline_steps = []

        try:
            AddTeleopEventsAsInfoStep(teleoperators=teleoperators)
        except TypeError:
            pass
        if self.processor.events.key_mapping:
            action_pipeline_steps.append(AddKeyboardEventsAsInfoStep(mapping=self.processor.events.key_mapping))

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(
                AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping))

        action_pipeline_steps.extend([
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleoperators),
            AddTeleopEventsAsInfoStep(teleoperators=teleoperators),
            AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping),
            SixDofVelocityInterventionActionProcessorStep(
                use_gripper=self.processor.gripper.use_gripper,
                control_mask=self.processor.task_frame.control_mask,
                terminate_on_success=self.processor.reset.terminate_on_success,
            ),
            ActionScalingProcessorStep(action_scale=self.processor.task_frame.action_scale),
            GripperOffsetProcessorStep(
                gripper_idc=self.gripper_idc,
                offset=self.processor.gripper.offset
            ),
        ])

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

    def make_env_processor(self, device) -> DataProcessorPipeline:
        env_pipeline_steps: list = [
            VanillaTFFProcessorStep(
                device=device,
                ee_pos_mask=self.processor.task_frame.control_mask,
                use_gripper=self.processor.gripper.use_gripper,
                add_ee_velocity_to_observation=self.processor.observation.add_ee_velocity_to_observation,
                add_ee_wrench_to_observation=self.processor.observation.add_ee_wrench_to_observation,
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

        env_pipeline_steps.extend([
            GripperPenaltyProcessorStep(
                gripper_idc=self.gripper_idc,
                max_gripper_pos=self.processor.gripper.max_pos,
                penalty=self.processor.gripper.penalty,
            ),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device)
        ])

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
