from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
)
from lerobot.processor.converters import identity_transition
from lerobot.processor.hil_processor import (
    AddFootswitchEventsAsInfoStep,
    AddKeyboardEventsAsInfoStep,
    DiscretizeGripperProcessorStep,
)
from lerobot.processor.tf_processor import (
    SixDofVelocityInterventionActionProcessorStep,
    VanillaTFFProcessorStep,
)


@dataclass(slots=True)
class PrimitiveProcessorBuilder:
    """Builds the share-rl action/env processor pipelines for task-frame primitives."""

    robot_names: list[str]
    processor: Any
    fps: int = 30

    def _as_per_robot(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else dict.fromkeys(self.robot_names, value)

    @property
    def gripper_idc(self) -> dict[str, int | None]:
        masks = self._as_per_robot(self.processor.task_frame.control_mask)
        gripper = self._as_per_robot(self.processor.gripper.use_gripper)

        indices: dict[str, int | None] = {}
        idx = 0
        for name in self.robot_names:
            indices[name] = None
            idx += sum(bool(v) for v in masks[name])
            if gripper[name]:
                indices[name] = idx
                idx += 1

        return indices

    def make_action_processor(self, teleoperators: dict[str, Any]) -> DataProcessorPipeline:
        action_pipeline_steps: list = []

        if self.processor.events.key_mapping:
            action_pipeline_steps.append(AddKeyboardEventsAsInfoStep(mapping=self.processor.events.key_mapping))

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(
                AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping)
            )

        action_pipeline_steps.extend(
            [
                AddTeleopEventsAsInfoStep(teleoperators=teleoperators),
                AddTeleopActionAsComplimentaryDataStep(teleoperators=teleoperators),
                SixDofVelocityInterventionActionProcessorStep(
                    teleoperators=teleoperators,
                    use_gripper=self._as_per_robot(self.processor.gripper.use_gripper),
                    control_mask=self._as_per_robot(self.processor.task_frame.control_mask),
                    terminate_on_success=self._as_per_robot(self.processor.reset.terminate_on_success),
                ),
                DiscretizeGripperProcessorStep(
                    gripper_idc=self.gripper_idc,
                    min_pos=self._as_per_robot(self.processor.gripper.min_pos),
                    max_pos=self._as_per_robot(self.processor.gripper.max_pos),
                ),
            ]
        )

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
            steps=action_pipeline_steps,
            to_transition=identity_transition,
            to_output=identity_transition,
            before_step_hooks=action_before_hooks,
            after_step_hooks=action_after_hooks,
        )

    def make_env_processor(self, device: str) -> DataProcessorPipeline:
        env_pipeline_steps: list = [
            VanillaTFFProcessorStep(
                device=device,
                ee_pos_mask=self._as_per_robot(self.processor.observation.ee_pos_mask),
                use_gripper=self._as_per_robot(self.processor.gripper.use_gripper),
                add_ee_velocity_to_observation=self._as_per_robot(
                    self.processor.observation.add_ee_velocity_to_observation
                ),
                add_ee_wrench_to_observation=self._as_per_robot(
                    self.processor.observation.add_ee_wrench_to_observation
                ),
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
                TimeLimitProcessorStep(max_episode_steps=int(self.processor.control_time_s * self.fps))
            )

        if self.processor.reward_classifier:
            env_pipeline_steps.append(
                RewardClassifierProcessorStep(
                    pretrained_path=self.processor.reward_classifier.pretrained_path,
                    device=device,
                    success_threshold=self.processor.reward_classifier.success_threshold,
                    success_reward=self.processor.reward_classifier.success_reward,
                    terminate_on_success=any(self._as_per_robot(self.processor.reset.terminate_on_success).values()),
                )
            )

        env_pipeline_steps.extend(
            [
                GripperPenaltyProcessorStep(
                    gripper_idc=self.gripper_idc,
                    max_gripper_pos=self._as_per_robot(self.processor.gripper.max_pos),
                    penalty=self._as_per_robot(self.processor.gripper.penalty),
                ),
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]
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
            steps=env_pipeline_steps,
            to_transition=identity_transition,
            to_output=identity_transition,
            before_step_hooks=env_before_hooks,
            after_step_hooks=env_after_hooks,
        )
