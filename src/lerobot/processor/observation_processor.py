#!/usr/bin/env python
from collections import deque
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessorStep(ObservationProcessorStep):
    """
    Processes standard Gymnasium observations into the LeRobot format.

    This step handles both image and state data from a typical observation dictionary,
    preparing it for use in a LeRobot policy.

    **Image Processing:**
    -   Converts channel-last (H, W, C), `uint8` images to channel-first (C, H, W),
        `float32` tensors.
    -   Normalizes pixel values from the [0, 255] range to [0, 1].
    -   Adds a batch dimension if one is not already present.
    -   Recognizes a single image under the key `"pixels"` and maps it to
        `"observation.image"`.
    -   Recognizes a dictionary of images under the key `"pixels"` and maps them
        to `"observation.images.{camera_name}"`.

    **State Processing:**
    -   Maps the `"environment_state"` key to `"observation.environment_state"`.
    -   Maps the `"agent_pos"` key to `"observation.state"`.
    -   Converts NumPy arrays to PyTorch tensors.
    -   Adds a batch dimension if one is not already present.
    """

    def _process_single_image(self, img: np.ndarray) -> Tensor:
        """
        Processes a single NumPy image array into a channel-first, normalized tensor.

        Args:
            img: A NumPy array representing the image, expected to be in channel-last
                 (H, W, C) format with a `uint8` dtype.

        Returns:
            A `float32` PyTorch tensor in channel-first (B, C, H, W) format, with
            pixel values normalized to the [0, 1] range.

        Raises:
            ValueError: If the input image does not appear to be in channel-last
                        format or is not of `uint8` dtype.
        """
        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        # Add batch dimension if needed
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Validate image format
        _, h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")

        if img_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected torch.uint8 images, but got {img_tensor.dtype}")

        # Convert to channel-first format
        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()

        # Convert to float32 and normalize to [0, 1]
        img_tensor = img_tensor.type(torch.float32) / 255.0

        return img_tensor

    def _process_observation(self, observation):
        """
        Processes both image and state observations.
        """

        processed_obs = observation.copy()

        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")

            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._process_single_image(img)

        if "environment_state" in processed_obs:
            env_state_np = processed_obs.pop("environment_state")
            env_state = torch.from_numpy(env_state_np).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos

        return processed_obs

    def observation(self, observation):
        return self._process_observation(observation)

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
            "environment_state": OBS_ENV_STATE,
            "agent_pos": OBS_STATE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }

        # Iterate over all incoming feature buckets and normalize/move each entry
        for src_ft, bucket in features.items():
            for key, feat in list(bucket.items()):
                handled = False

                # Prefix-based rules (e.g. pixels.cam1 -> OBS_IMAGES.cam1)
                for old_prefix, new_prefix in prefix_pairs.items():
                    prefixed_old = f"{OBS_STR}.{old_prefix}"
                    if key.startswith(prefixed_old):
                        suffix = key[len(prefixed_old) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix) :]
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

        return new_features


@dataclass
@ProcessorStepRegistry.register(name="framestack_processor")
class FrameStackProcessorStep(ObservationProcessorStep):
    """
    Frame stacking processor for state observations.

    - Maintains a rolling buffer of the last `n_frames` for each key in `obs_keys`.
    - Concatenates along the last dimension (features).
    - On the first observation after reset, fills the buffer entirely with that frame.
    """

    n_frames: int = 1
    obs_keys: list[str] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.obs_keys is None:
            self.obs_keys = [OBS_STATE]

        assert self.n_frames >= 1
        self.reset()

    def _stack_frames(self, key: str) -> Tensor:
        """Concatenate buffered frames along the last dimension (features)."""
        buf = list(self._buffers[key])
        return torch.cat(buf, dim=-1)  # (B, D * n_frames)

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        out = observation.copy()

        for key in self.obs_keys:
            if key not in observation:
                continue

            tensor = observation[key]

            if len(self._buffers[key]) == 0:
                # First call after reset → fill with same frame
                for _ in range(self.n_frames):
                    self._buffers[key].append(tensor)
            else:
                self._buffers[key].append(tensor)

            out[key] = self._stack_frames(key)

        return out

    def reset(self):
        """Clear all buffers."""
        self._buffers: dict[str, deque] = {
            key: deque(maxlen=self.n_frames) for key in self.obs_keys
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adjusts feature shapes to account for frame stacking.
        If state dim = D and n_frames = 4 → output dim = D*4.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {
            ft: {} for ft in features
        }

        for ft, bucket in features.items():
            for key, feat in bucket.items():
                if key in self.obs_keys:
                    new_feat = feat.copy()
                    shape = list(new_feat.shape)

                    if len(shape) != 1:
                        raise ValueError(
                            f"FrameStack only supports flat state vectors, got shape {shape}"
                        )

                    shape[0] *= self.n_frames
                    new_feat.shape = tuple(shape)
                    new_features[ft][key] = new_feat
                else:
                    new_features[ft][key] = feat

        return new_features



