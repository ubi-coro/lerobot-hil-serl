#!/usr/bin/env python
import logging
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable, Literal

import evdev
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch import Tensor
from tqdm import tqdm

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from lerobot.processor.core import EnvTransition, PolicyAction, TransitionKey
from lerobot.processor.pipeline import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)
from lerobot.utils.constants import OBS_IMAGES

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


def strip_img_prefix(key: str) -> str:
    prefixes_to_strip = tuple(
        f"{token}." for token in (OBS_IMAGES, OBS_IMAGES.split(".")[-1])
    )
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound=Teleoperator)


def _check_teleop_with_events(teleop: Teleoperator) -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


class FootSwitchHandler:
    def __init__(self, device_path="/dev/input/event18", event_names: tuple[str] = (TeleopEvents.SUCCESS, ), toggle: bool = False):
        self.device = evdev.InputDevice(device_path)
        self.events = {name: False for name in event_names}
        self.toggle = toggle
        self.event_names = event_names
        self.running = True

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        logging.info(f"Listening for foot switch events from {self.device.name} ({self.device.path})...")
        for event in self.device.read_loop():
            if not self.running:
                break
            if event.type == evdev.ecodes.EV_KEY:
                key_event = evdev.categorize(event)
                if key_event.keystate == 1:  # Key down
                    if self.toggle:
                        if self.events[self.event_names[0]]:
                            logging.info(f"Foot switch pressed again - {self.event_names} toggled OFF")
                            for name in self.event_names:
                                self.events[name] = False
                        else:
                            logging.info(f"Foot switch pressed - {self.event_names} toggled ON")
                            for name in self.event_names:
                                self.events[name] = True
                    else:
                        logging.info(f"Foot switch pressed - {self.event_names} ON")
                        for name in self.event_names:
                            self.events[name] = True
                elif key_event.keystate == 0 and not self.toggle:  # Key release
                    logging.info(f"Foot switch released - {self.event_names} OFF")
                    for name in self.event_names:
                        self.events[name] = False

    def stop(self):
        self.running = False

    def reset(self):
        self.events = {name: False for name in self.event_names}


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleoperators: dict[str, Teleoperator] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `complementary_data` method to the transition's data."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = complementary_data.copy()

        # avoid unnecessary I/O by only reading when the intervention event is set
        if transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False):
            processed_complementary_data[TELEOP_ACTION_KEY] = {}
            for name in self.teleoperators:
                processed_complementary_data[TELEOP_ACTION_KEY][name] = self.teleoperators[name].get_action()

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleoperators: dict[str, Teleoperator] = field(default_factory=dict)

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        for t in self.teleoperators.values():
            _check_teleop_with_events(t)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)
        for t in self.teleoperators.values():
            for event_name, event_value in t.get_teleop_events().items():
                new_info[event_name] = new_info.get(event_name, False) | event_value

        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_footswitch_events_as_info")
@dataclass
class AddFootswitchEventsAsInfoStep(InfoProcessorStep):
    mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=dict)

    def __post_init__(self):
        self._foot_switch_threads = dict()

        for events, params in self.mapping.items():
            self._foot_switch_threads[events] = FootSwitchHandler(
                device_path=f'/dev/input/event{params["device"]}',
                toggle=bool(params["toggle"]),
                event_names=events
            )
            self._foot_switch_threads[events].start()

    def info(self, info: dict) -> dict:
        new_info = dict(info)
        for handler in self._foot_switch_threads.values():
            for event_name, event_value in handler.events.items():
                new_info[event_name] = new_info.get(event_name, False) | event_value
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        for handler in self._foot_switch_threads.values():
            handler.reset()

    def __del__(self):
        for handler in self._foot_switch_threads.values():
            handler.stop()


@ProcessorStepRegistry.register("add_keyboard_events_as_info")
@dataclass
class AddKeyboardEventsAsInfoStep(InfoProcessorStep):
    mapping: dict[TeleopEvents, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._events = {event: False for event in self.mapping}
        self._is_string_key = {event: isinstance(mapping_key, str) for event, mapping_key in self.mapping.items()}

        from pynput import keyboard

        def on_press(key):
            for event, mapping_key in self.mapping.items():
                try:
                    if self._is_string_key[event]:
                        if key.char == mapping_key:
                            self._events[event] = True
                    else:
                        if key == mapping_key:
                            self._events[event] = True
                except Exception:
                    ...

        def on_release(key):
            for event, mapping_key in self.mapping.items():
                try:
                    if self._is_string_key[event]:
                        if key.char == mapping_key:
                            self._events[event] = False
                    else:
                        if key == mapping_key:
                            self._events[event] = False
                except Exception:
                    ...


        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def info(self, info: dict) -> dict:
        new_info = dict(info)
        for event_name, event_value in self._events.items():
            new_info[event_name] = new_info.get(event_name, False) | event_value
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._events = {event: False for event in self.mapping}

    def __del__(self):
        for l in self._listener.values():
            l.stop()



@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            stripped_key = strip_img_prefix(key)

            key_matches = key in self.crop_params_dict
            stripped_key_matches = stripped_key in self.crop_params_dict
            if self.crop_params_dict is not None and (key_matches or stripped_key_matches):
                if key_matches:
                    crop_params = self.crop_params_dict[key]
                else:
                    crop_params = self.crop_params_dict[stripped_key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def __post_init__(self):
        """Initialize the tqdm progress bar."""
        self._pbar = tqdm(
            total=self.max_episode_steps,
            desc="Episode Progress",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )

    def truncated(self, truncated: bool) -> bool:
        """
        Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix_str(f"Remaining: {self.max_episode_steps - self.current_step}")

        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            if self._pbar is not None:
                self._pbar.close()

        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0
        self._pbar.reset()


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ComplementaryDataProcessorStep):
    """
    Applies a penalty for inefficient gripper usage on multiple robots.

    For each robot:
      - Penalizes actions that try to close an already closed gripper
        or open an already open one.
      - Penalties are added to complementary_data with robot-specific keys.

    Attributes:
        penalty: Negative reward value applied per violation.
        max_gripper_pos: Dict of robot_name -> max gripper pos (for normalization).
    """

    gripper_idc: dict[str, int | None] = field(default_factory=dict)
    penalty: dict[str, float | None] = field(default_factory=dict)
    max_gripper_pos: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert self.gripper_idc.keys() == self.penalty.keys() == self.max_gripper_pos.keys()
        self._robot_names = self.gripper_idc.keys()

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Calculates gripper penalties for each robot and adds them to complementary_data.
        """
        action = self.transition.get(TransitionKey.ACTION)

        new_cd = dict(complementary_data)
        robot_penalty = 0.0
        for name in self._robot_names:
            current_gripper_pos = complementary_data.get(f"{name}.{GRIPPER_KEY}.pos", None)

            if self.gripper_idc[name] is None or self.penalty[name] is None or current_gripper_pos is None:
                continue

            current_gripper_pos = complementary_data.get.get(f"{name}.{GRIPPER_KEY}.pos", None)
            gripper_idx = int(self.gripper_idc[name])
            gripper_action = action[gripper_idx].item()
            max_pos = self.max_gripper_pos[name]

            gripper_action_normalized = gripper_action / max_pos
            gripper_state_normalized = current_gripper_pos / max_pos

            penalty_trigger = (
                (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5)
                or (gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5)
            )

            robot_penalty += self.penalty[name] * int(penalty_trigger)

        new_cd[DISCRETE_PENALTY_KEY] = robot_penalty

        return new_cd

    def get_config(self) -> dict[str, Any]:
        return {
            "gripper_idc": self.gripper_idc,
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("discretize_gripper_processor")
class DiscretizeGripperProcessorStep(ActionProcessorStep):
    """
    Discretizes gripper actions using an internal per-robot gripper state.

    For each robot's gripper action:
      - If the incoming action value is greater than +threshold, the internal
        gripper state is set to 1.0 (fully closed).
      - If the incoming action value is less than -threshold, the internal
        gripper state is set to `min_position[robot_name]` (partially open).
      - If the incoming action value lies in [-threshold, threshold], the
        internal state is left unchanged (i.e., “do nothing” region).

    The processed action written back into the tensor is always this internal
    discretized state, allowing a continuous input signal (e.g. from a joystick
    or SpaceMouse) to act like a three-region controller:
      - Close
      - Open to a defined minimum position
      - No-op in a deadzone around zero

    Attributes
    ----------
    gripper_idc:
        Mapping from robot name to the index of the gripper action in the
        action tensor. Use `None` to skip a robot.
    min_position:
        Mapping from robot name to the minimum gripper position that should be
        used when the input is below -threshold (e.g. 0.8).
    threshold:
        Scalar threshold that defines the deadzone around zero. Defaults to 0.5.
    """

    gripper_idc: dict[str, int | None] = field(default_factory=dict)
    min_pos: dict[str, float] = field(default_factory=dict)
    max_pos: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    mode: Literal["state", "pulse"] = "state"

    def __post_init__(self):
        # Ensure we have consistent keys for indices and min positions
        assert self.gripper_idc.keys() == self.min_pos.keys() == self.max_pos.keys()
        self._robot_names = list(self.gripper_idc.keys())

        # Internal gripper state per robot (initialized to min_position)
        self._gripper_state: dict[str, float] = {
            name: self.min_pos[name] for name in self._robot_names
        }

    def action(self, action: Tensor) -> Tensor:
        """
        Discretizes gripper actions based on the current input and internal state.

        Parameters
        ----------
        action:
            1D action tensor for a single step (or a compatible view) that
            includes gripper entries at the indices specified in `gripper_idc`.

        Returns
        -------
        Tensor
            The same tensor, with gripper entries replaced by the internal
            discretized gripper state.
        """
        # Important: clone so we never modify an inference tensor in-place
        out = action.clone()

        for name in self._robot_names:
            gripper_idx = self.gripper_idc[name]
            if gripper_idx is None:
                continue

            gi = int(gripper_idx)

            # Read current (continuous) gripper input
            input_val = float(out[gi].item())

            # Update internal state based on thresholds
            if self.mode == "pulse":
                if input_val > self.threshold:
                    self._gripper_state[name] = self.max_pos[name]
                elif input_val < -self.threshold:
                    self._gripper_state[name] = self.min_pos[name]
            elif self.mode == "state":
                if input_val > self.threshold:
                    self._gripper_state[name] = self.max_pos[name]
                elif input_val < self.threshold:
                    self._gripper_state[name] = self.min_pos[name]

            # Write discretized value back to the *output* tensor
            out[gi] = out.new_tensor(self._gripper_state[name])

        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "gripper_idc": self.gripper_idc,
            "min_pos": self.min_pos,
            "max_pos": self.max_pos,
            "threshold": self.threshold,
        }

    def reset(self) -> None:
        """
        Resets internal gripper states to their minimum positions.
        """
        for name in self._robot_names:
            self._gripper_state[name] = self.min_pos[name]

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
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
        self._disable_torque_on_intervention = {name: hasattr(teleop, "bus") for name, teleop in self.teleoperators.items()}
        self._intervention_occurred = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        action = transition.get(TransitionKey.ACTION)
        assert isinstance(action, PolicyAction), f"Action should be a PolicyAction type got {type(action)}"
        assert len(action) == sum([len(self.teleoperators[name].action_features) for name in self.teleoperators])

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

        # Terminate on intervention end to correctly store episode bounds
        self._intervention_occurred = self._intervention_occurred  | is_intervention
        if self._intervention_occurred and not is_intervention:
            info[TeleopEvents.INTERVENTION_COMPLETED] = True

        # Override action if intervention is active
        if is_intervention and teleop_action_dict is not None:

            # loop over teleoperators and concat their action
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
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        elif not self._intervention_occurred:  # dont write feedback on intervention end
            # send the current action as feedback to the robots
            idx = 0
            for teleop_name, teleop in self.teleoperators.items():
                feedback_action = {}
                for ft in teleop.action_features:
                    feedback_action[ft] = action[idx]
                    idx += 1
                teleop.send_feedback(feedback_action)

        else:
            # todo: this takes forever (2-3ms -> 25-30ms) when recording normally, ie not interactive

            # torque leader on intervention end
            for name, teleop_action in self.teleoperators.items():
                # torque leaders off on interventions
                if self._disable_torque_on_intervention[name]:
                    self.teleoperators[name].enable_torque()


        # Handle episode termination
        new_transition[TransitionKey.DONE] = (
                bool(terminate_episode) or
                bool(rerecord_episode) or
                (any(self.terminate_on_success.values()) and success)
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
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

    def reset(self) -> None:
        self._intervention_occurred = False


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

