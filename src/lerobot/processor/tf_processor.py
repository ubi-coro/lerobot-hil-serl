from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_ENV_STATE, OBS_STR
from . import EnvTransition, TransitionKey, PolicyAction, VanillaObservationProcessorStep
from .hil_processor import TELEOP_ACTION_KEY, GRIPPER_KEY

from .pipeline import ProcessorStepRegistry, ProcessorStep
from ..teleoperators import TeleopEvents, Teleoperator


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
        processed_complementary_data["raw_observation"] = observation
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
                if f"{name}.{ax}.ee_pos" in observation and self.ee_pos_mask.get(name, [1] * 6)[i]:
                    state_parts.append(observation[f"{name}.{ax}.ee_pos"])

                if f"{name}.{ax}.ee_vel" in observation and self.add_ee_velocity_to_observation.get(name, False):
                    state_parts.append(observation[f"{name}.{ax}.ee_vel"])

                if f"{name}.{ax}.ee_wrench" in observation and self.add_ee_wrench_to_observation.get(name, False):
                    state_parts.append(observation[f"{name}.{ax}.ee_wrench"])

            if f"{name}.{GRIPPER_KEY}.pos" in observation and self.use_gripper.get(name, False):
                state_parts.append(observation[f"{name}.{GRIPPER_KEY}.pos"])

        if state_parts:
            processed_obs[f"{OBS_STATE}"] = torch.tensor(state_parts).type(torch.float32)

        return processed_obs

    def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the Gym standard to the LeRobot standard.

        This method standardizes the feature dictionary by renaming keys according
        to LeRobot's conventions, ensuring that policies can be constructed correctly.
        It handles various raw key formats, including those with an "observation." prefix.

        **Renaming Rules:**
        - `pixels` or `observation.pixels` -> `observation.image`
        - `pixels.{cam}` or `observation.pixels.{cam}` -> `observation.images.{cam}`
        - `environment_state` or `observation.environment_state` -> `observation.environment_state`
        - `agent_pos` or `observation.agent_pos` -> `observation.state`

        Args:
            features: The policy features dictionary with Gym-style keys.

        Returns:
            The policy features dictionary with standardized LeRobot keys.
        """
        # Build a new features mapping keyed by the same FeatureType buckets
        # We assume callers already placed features in the correct FeatureType.
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {ft: {} for ft in features}

        exact_pairs = {
            "pixels": OBS_IMAGE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }

        # Iterate over all incoming feature buckets and normalize/move each entry
        state_dim = 0
        for src_ft, bucket in features.items():
            state_dim = 0
            for key, feat in list(bucket.items()):
                handled = False

                if self.is_state_key(key):
                    handled = True

                    name, axis, state_type = key.split(".")
                    idx_map = {ax: i for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"])}

                    if (state_type == "ee_vel" and self.add_ee_velocity_to_observation[name]) or \
                            (state_type == "ee_wrench" and self.add_ee_velocity_to_observation[name]) or \
                            (state_type == "ee_pos" and self.ee_pos_mask.get(name, [1] * 6)[idx_map[axis]]) or \
                            (state_type == "pos" and axis == "gripper" and self.use_gripper[name]):
                        state_dim += 1

                if handled:
                    continue

                if len(feat.shape) == 3:
                    feat.shape = (feat.shape[2], feat.shape[0], feat.shape[1])

                # Prefix-based rules (e.g. pixels.cam1 -> OBS_IMAGES.cam1)
                for old_prefix, new_prefix in prefix_pairs.items():
                    prefixed_old = f"{OBS_STR}.{old_prefix}"
                    if key.startswith(prefixed_old):
                        suffix = key[len(prefixed_old):]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix):]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # Exact-name rules (pixels, environment_state, agent_pos)
                for old, new in exact_pairs.items():
                    if key == old or key == f"{OBS_STR}.{old}":
                        new_key = new
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # Default: keep key in the same source FeatureType bucket
                new_features[src_ft][key] = feat

            if state_dim > 0:
                new_features[src_ft][OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

        return new_features

    def is_state_key(self, key):
        return "pos" in key or "vel" in key or "wrench" in key


@dataclass
@ProcessorStepRegistry.register("6dof_intervention_action_processor")
class SixDofVelocityInterventionActionProcessorStep(ProcessorStep):
    """
    Multi-robot intervention processor.
    """

    teleoperators: dict[str, Teleoperator]
    use_gripper: dict[str, bool] = field(default_factory=dict)
    terminate_on_success: dict[str, bool] = field(default_factory=dict)
    control_mask: dict[str, list[bool]] = field(default_factory=dict)

    def __post_init__(self):
        self._intervention_occurred = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        action = transition.get(TransitionKey.ACTION)
        assert isinstance(action, PolicyAction), f"Action should be a PolicyAction type got {type(action)}"
        # assert len(action) == sum([len(self.teleoperators[name].action_features) for name in self.teleoperators])

        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Teleop actions come as dict[name -> {axis: value, ...}]
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Terminate on intervention end to correctly store episode bounds
        self._intervention_occurred = self._intervention_occurred | is_intervention
        if self._intervention_occurred and not is_intervention:
            info[TeleopEvents.INTERVENTION_COMPLETED] = True

        if is_intervention and teleop_action_dict:

            action_list = []
            for name, teleop_action in teleop_action_dict.items():

                if isinstance(teleop_action, dict):
                    for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
                        if self.control_mask.get(name, [True] * 6)[i]:
                            action_list.append(teleop_action.get(f"{ax}.vel", 0.0))
                    if self.use_gripper.get(name, False):
                        action_list.append(teleop_action.get(f"{GRIPPER_KEY}.pos", 1.0))
                elif isinstance(teleop_action, np.ndarray):
                    action_list.extend(teleop_action.tolist())
                else:
                    action_list.extend(teleop_action)

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor


        elif not self._intervention_occurred:  # dont write feedback on intervention end
            # send the current action as feedback to the robots
            idx = 0
            for name, teleop in self.teleoperators.items():
                feedback_action = {}
                for i, ft in enumerate(teleop.action_features):
                    if self.control_mask.get(name, [True] * 6)[i]:
                        feedback_action[ft] = action[idx].item()
                        idx += 1

                if self.use_gripper.get(name, False):
                    feedback_action["gripper.pos"] = action[idx].item()
                    idx += 1

                teleop.send_feedback(feedback_action)

        # Termination / reward
        new_transition[TransitionKey.DONE] = (
                bool(terminate_episode) or
                bool(rerecord_episode) or
                (any(self.terminate_on_success.values()) and success)
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info / complementary data
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition[TransitionKey.ACTION].clone()
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

    def reset(self) -> None:
        self._intervention_occurred = False
