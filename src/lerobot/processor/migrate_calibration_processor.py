import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from zipp.compat.py313 import apply

from lerobot.configs.types import PolicyFeature, PipelineFeatureType
from lerobot.processor import ProcessorStepRegistry, ObservationProcessorStep, EnvTransition, TransitionKey, \
    ProcessorStep, PolicyAction
from lerobot.processor.hil_processor import GRIPPER_KEY, TELEOP_ACTION_KEY
from lerobot.teleoperators import Teleoperator, TeleopEvents
from lerobot.utils.constants import OBS_STATE, ACTION

OLD_CALIB = {
    "homing_offset": [2048, 3072, 3072, -1024, -1024, 2048, -2048, 2048, -2048],
    "drive_mode":    [1, 1, 1, 0, 0, 1, 0, 1, 0],
    "motor_names":   ["waist", "shoulder", "shoulder_shadow", "elbow", "elbow_shadow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"],
}
NEW_CALIB = {
    "waist":         {"id":1, "drive_mode":0, "range_min":0,    "range_max":4095},
    "shoulder":      {"id":2, "drive_mode":1, "range_min":0,    "range_max":4095},
    "shoulder_shadow":{"id":3,"drive_mode":0, "range_min":0,    "range_max":4095},
    "elbow":         {"id":4, "drive_mode":1, "range_min":0,    "range_max":4095},
    "elbow_shadow":  {"id":5, "drive_mode":0, "range_min":0,    "range_max":4095},
    "forearm_roll":  {"id":6, "drive_mode":0, "range_min":0,    "range_max":4095},
    "wrist_angle":   {"id":7, "drive_mode":1, "range_min":0,    "range_max":4095},
    "wrist_rotate":  {"id":8, "drive_mode":0, "range_min":0,    "range_max":4095},
    "gripper":       {"id":9, "drive_mode":0, "range_min":1990, "range_max":2745},
}

RESOLUTION = 4096
HALF_TURN_DEGREE = 180


def invert_calibration(old_state: Tensor, num_robots: int = 1) -> Tensor:
    device = old_state.device
    dtype = old_state.dtype

    # from the new system, in radians, without shadow joints
    new_state = []
    new_idx = -1
    for _ in range(num_robots):
        for old_idx, name in enumerate(OLD_CALIB["motor_names"]):
            new_idx += 1
            if "shadow" in name:
                continue

            if name == "gripper":
                new_state.append(float(old_state[new_idx]) / 100.0)
                continue

            # ---- revert old calibration (degrees → raw ticks) ----
            old_drive_mode = OLD_CALIB["drive_mode"][old_idx]
            homing_offset = OLD_CALIB["homing_offset"][old_idx]

            ticks = old_state[new_idx] / HALF_TURN_DEGREE * (RESOLUTION // 2)
            ticks -= homing_offset

            if old_drive_mode == 1:
                ticks *= -1

            # ---- Apply new calibration (raw ticks → radians) ----
            new_cfg = NEW_CALIB[name]
            min_ = new_cfg["range_min"]
            max_ = new_cfg["range_max"]
            mid = (min_ + max_) / 2
            max_res = RESOLUTION - 1
            rad = (ticks - mid) * 2 * math.pi / max_res
            new_state.append(rad)

    return torch.tensor(new_state, dtype=dtype, device=device)


def apply_calibration(new_state: Tensor, num_robots: int = 1) -> Tensor:
    device = new_state.device
    dtype = new_state.dtype

    # for old policies, in degrees, with shadow joints
    old_state = []
    new_idx = -1
    for _ in range(num_robots):

        for old_idx, name in enumerate(OLD_CALIB["motor_names"]):
            if "shadow" not in name:
                new_idx += 1

            if name == "gripper":
                old_state.append(float(new_state[new_idx]) * 100.0)
                continue

            # ---- Undo new calibration (radians → raw ticks) ----
            new_cfg = NEW_CALIB[name]
            min_ = new_cfg["range_min"]
            max_ = new_cfg["range_max"]
            mid = (min_ + max_) / 2
            max_res = RESOLUTION - 1
            ticks = int((new_state[new_idx] * max_res / (2 * math.pi)) + mid)

            # ---- Apply old calibration (raw ticks → degrees) ----
            old_drive_mode = OLD_CALIB["drive_mode"][old_idx]
            homing_offset = OLD_CALIB["homing_offset"][old_idx]

            if old_drive_mode == 1:
                ticks *= -1

            ticks += homing_offset
            deg = ticks / (RESOLUTION // 2) * HALF_TURN_DEGREE
            old_state.append(deg)

    return torch.tensor(old_state, dtype=dtype, device=device)


@dataclass
@ProcessorStepRegistry.register(name="migrate_calibration_obs_processor")
class MigrateCalibrationObsProcessorStep(ObservationProcessorStep):

    num_robots: int = 1

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        out = observation.copy()
        out[OBS_STATE] = apply_calibration(out[OBS_STATE], num_robots=self.num_robots)
        return out

    def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        # add shadow joints to obs
        obs_features = features[PipelineFeatureType.OBSERVATION]
        obs_features[OBS_STATE].shape = (obs_features[OBS_STATE].shape[0] + 2 * self.num_robots, )
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features


@dataclass
@ProcessorStepRegistry.register("migrate_intervention_action_processor")
class MigrateInterventionActionProcessorStep(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    teleoperators: dict[str, Teleoperator]
    use_gripper: dict[str, bool]
    terminate_on_success: dict[str, bool]

    def __post_init__(self):
        self._disable_torque_on_intervention = {name: hasattr(teleop, "disable_torque") for name, teleop in self.teleoperators.items()}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        action = transition.get(TransitionKey.ACTION)  # num_robots * 9
        assert isinstance(action, PolicyAction), f"Action should be a PolicyAction type got {type(action)}"
        #assert len(action) == sum([len(self.teleoperators[name].action_features) for name in self.teleoperators])

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Teleop actions come as dict[name -> {axis: value, ...}]
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action_dict is not None:

            # loop over teleoperators and concat their actions
            action_list = []
            for name, teleop_action in teleop_action_dict.items():
                # torque leaders off on interventions
                if self._disable_torque_on_intervention[name]:
                    self.teleoperators[name].disable_torque()

                # process teleop action
                if isinstance(teleop_action, dict):
                    # convert teleop_action dict to tensor format
                    for ft in self.teleoperators[name].action_features:
                        if ft == f"{GRIPPER_KEY}.pos" and not self.use_gripper[name]:
                            continue
                        action_list.append(teleop_action.get(ft, 0.0))
                elif isinstance(teleop_action, np.ndarray):
                    action_list.extend(teleop_action.tolist())
                else:
                    action_list.extend(teleop_action)

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)

            # action tensor (for the env) is already num_robots * 7, so we dont modify it
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

            # action tensor (tha we save to task) must look like the original action
            teleop_action_tensor_processed = apply_calibration(teleop_action_tensor, num_robots=len(self.teleoperators))
            complementary_data[TELEOP_ACTION_KEY] = teleop_action_tensor_processed
        else:
            processed_action = invert_calibration(action, num_robots=len(self.teleoperators))
            processed_action_cpu = processed_action.detach().to("cpu", non_blocking=False)

            # send feedback to teleops
            idx = 0
            for teleop_name, teleop in self.teleoperators.items():
                feats = teleop.action_features
                ln = len(feats)
                sub = processed_action_cpu[idx: idx + ln].tolist()  # plain Python floats
                payload = dict(zip(feats, sub))
                teleop.send_feedback(payload)
                idx += ln

            new_transition[TransitionKey.ACTION] = processed_action

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            any(self.terminate_on_success.values()) and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        #complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        #complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

