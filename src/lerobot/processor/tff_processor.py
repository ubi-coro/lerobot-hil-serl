from dataclasses import dataclass, field
from typing import Any, Optional

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR, DEFAULT_ROBOT_NAME
from . import EnvTransition, TransitionKey, PolicyAction, VanillaObservationProcessorStep
from .hil_processor import TELEOP_ACTION_KEY, GRIPPER_KEY

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry, ProcessorStep, RobotActionProcessorStep
from ..robots.ur.tff_controller import TaskFrameCommand
from ..teleoperators import TeleopEvents


@dataclass
@ProcessorStepRegistry.register(name="tf_observation_processor")


class VanillaTFFProcessorStep(VanillaObservationProcessorStep):
    """
    Multi-robot TFF processor. Produces a LeRobot-conform dict with per-robot observations.
    """

    add_ee_velocity_to_observation: dict[str, bool] = field(default_factory=dict)
    add_ee_wrench_to_observation: dict[str, bool] = field(default_factory=dict)
    use_gripper: dict[str, bool] = field(default_factory=dict)
    ee_pos_mask: dict[str, list[int]] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `observation` method to the transition's observation."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ObservationProcessorStep requires an observation in the transition.")

        processed_observation = self.observation(observation.copy())
        new_transition[TransitionKey.OBSERVATION] = processed_observation

        # add gripper pos to complementary info
        processed_complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA).copy()
        for name in self.add_ee_velocity_to_observation:
            if f"{name}.{GRIPPER_KEY}" in observation and self.use_gripper.get(name, False):
                processed_complementary_data[f"{name}.{GRIPPER_KEY}"] = observation[f"{name}.{GRIPPER_KEY}"]
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data

        return new_transition

    def _process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Processes multi-robot observations.
        Expected structure:
          {
            "robot1": {"pixels": ..., "x.ee_pos": ..., ...},
            "robot2": {...}
          }
        """
        processed_obs: dict[str, Any] = {}

        # Handle pixels
        if "pixels" in observation:
            pixels = observation["pixels"]
            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{cam}": img for cam, img in pixels.items()}
            else:
                imgs = {f"{OBS_IMAGE}": pixels}

            for image_key, img in imgs.items():
                processed_obs[image_key] = self._process_single_image(img)

        # Handle state
        state_parts = []
        for name in self.add_ee_velocity_to_observation:
            for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
                if f"{name}.{ax}.ee_pos" in observation and self.ee_pos_mask.get(name, [1]*6)[i]:
                    state_parts.append(observation[f"{name}.{ax}.ee_pos"])

                if f"{name}.{ax}.ee_vel" in observation and self.add_ee_velocity_to_observation.get(name, False):
                    state_parts.append(observation[f"{name}.{ax}.ee_vel"])

                if f"{name}.{ax}.ee_wrench" in observation and self.add_ee_wrench_to_observation.get(name, False):
                    state_parts.append(observation[f"{name}.{ax}.ee_wrench"])

            if f"{name}.{GRIPPER_KEY}" in observation and self.use_gripper.get(name, False):
                state_parts.append(observation[f"{name}.{GRIPPER_KEY}"])

        if state_parts:
            processed_obs[f"{OBS_STATE}"] = torch.tensor(state_parts).type(torch.float32)

        return processed_obs


@dataclass
@ProcessorStepRegistry.register("6dof_intervention_action_processor")
class SixDofVelocityInterventionActionProcessorStep(ProcessorStep):
    """
    Multi-robot intervention processor.
    """

    use_gripper: dict[str, bool] = field(default_factory=dict)
    terminate_on_success: dict[str, bool] = field(default_factory=dict)
    control_mask: dict[str, list[bool]] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Teleop actions come as dict[name -> {axis: value, ...}]
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        if is_intervention and teleop_action_dict:

            for name, teleop_action in teleop_action_dict.items():
                action_list = []

                if isinstance(teleop_action, dict):
                    for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
                        if self.control_mask.get(name, [True]*6)[i]:
                            action_list.append(teleop_action.get(f"{ax}.vel", 0.0))
                    if self.use_gripper.get(name, False):
                        action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
                elif isinstance(teleop_action, np.ndarray):
                    action_list.extend(teleop_action.tolist())
                else:
                    action_list.extend(teleop_action)

            new_transition[TransitionKey.ACTION] = torch.tensor(action_list, dtype=torch.float32)

        # Termination / reward
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or any(
            self.terminate_on_success.get(name, True) and success for name in teleop_action_dict
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info / complementary data
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
            "control_mask": self.control_mask
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features





# current
# obs is large dict with 1-dimensional states and images
# action is task frame command dict


# goal
# lerobot conform dict

# idea:

# action comes from

# AddTeleopActionAsComplimentaryDataStep
# AddTeleopEventsAsInfoStep
# InterventionActionProcessorStep

# GELLO: commands in q space -> mapped to vel space (rip from lerobot-hil-serl)
# SpaceMouse / keyboard: commands in vel space





# EEF or EEF speed action space  | maps <action_type> actions to appropriate command (any axes can have any control mode and be fixed as well)





# policy





# gym manipulator obs: "agent_pos": ndarray and "pixels": dict

# env pipeline:
# VanillaObservationProcessorStep        | turns "pixels", "agent_pos" etc into standard lerobot dict (flat)
# JointVelocityProcessorStep             | doubles state with fixed dt
# MotorCurrentProcessorStep              | adds current to state
# ForwardKinematicsJointsToEEObservation | adds ee position
# ImageCropResizeProcessorStep           | crops and resizes image keys
# TimeLimitProcessorStep                 | sets time limit
# GripperPenaltyProcessorStep            | adds gripper penalty
# reward classifier
# batch dim
# device processor


# action pipeline:
# AddTeleopActionAsComplimentaryDataStep
# AddTeleopEventsAsInfoStep
# InterventionActionProcessorStep



