#!/usr/bin/env python
import time
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
@ProcessorStepRegistry.register(name="observation_processor_new")
class VanillaObservationProcessorStep_new(ObservationProcessorStep):
    """
    Fast, zero-copy observation processor:
    - Images: keep as uint8 HWC on CPU (no normalization, no permute, no batch dim).
    - States: torch.from_numpy + float cast only if needed; no batch dim here.
    - Let later steps handle batching and device transfer.
    """

    def _to_tensor_uint8_hwc(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Accept numpy (preferred) or torch (already a tensor).
        if isinstance(img, torch.Tensor):
            t = img
        else:
            # Zero-copy from numpy (must be contiguous; typical camera frames are)
            t = torch.from_numpy(img)

        # Sanity: expect 3D HWC or 4D BHWC (some sources may already have batch)
        if t.ndim == 3:
            h, w, c = t.shape
        elif t.ndim == 4:
            _, h, w, c = t.shape
        else:
            raise ValueError(f"Expected image with 3 or 4 dims (HWC or BHWC); got shape {tuple(t.shape)}")

        # Keep channel-last; do NOT normalize or permute here.
        # Ensure dtype is uint8 (zero-copy path stays cheap).
        if t.dtype != torch.uint8:
            # If your source isn’t uint8, this will copy — try to fix at the source if possible.
            t = t.to(torch.uint8)

        # Ensure contiguous memory to avoid downstream surprises.
        if not t.is_contiguous():
            t = t.contiguous()

        return t  # uint8 HWC (or BHWC if upstream provided batch), CPU

    def _to_tensor_1d_or_2d_float(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Zero-copy from numpy when possible
        t = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        if t.dtype != torch.float32:
            t = t.float()
        # No batch dim here; AddBatchDimensionProcessorStep will handle it.
        return t

    def _process_observation(self, observation: dict) -> dict:
        processed_obs = observation.copy()

        # Images: pixels or pixels.{cam}
        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")
            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._to_tensor_uint8_hwc(img)

        # State-like arrays
        if "environment_state" in processed_obs:
            env_state = processed_obs.pop("environment_state")
            processed_obs[OBS_ENV_STATE] = self._to_tensor_1d_or_2d_float(env_state)

        if "agent_pos" in processed_obs:
            agent_pos = processed_obs.pop("agent_pos")
            processed_obs[OBS_STATE] = self._to_tensor_1d_or_2d_float(agent_pos)

        return processed_obs

    def observation(self, observation: dict) -> dict:
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

        return new_features


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
    device: str
    _img_float: torch.Tensor | None = None

    def _process_single_image(self, img: np.ndarray) -> Tensor:
        """
        Processes a single NumPy image array into a channel-first, normalized tensor.

        Args:
            img: A NumPy array representing the image, expected to be in channel-last
                 (H, W, C) format with a `uint8` dtype.

        Returns:
            A `float32` PyTorch tensor in channel-first (C, H, W) format, with
            pixel values normalized to the [0, 1] range.
        """

        img_tensor = torch.from_numpy(img)  # zero-copy view over numpy

        h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")
        img_tensor = einops.rearrange(img_tensor, "h w c -> c h w")

        if self._img_float is None or self._img_float.shape != img_tensor.shape:
            self._img_float = torch.empty_like(img_tensor, dtype=torch.float32)

        self._img_float.copy_(img_tensor)
        self._img_float.mul_(1.0 / 255.0)

        return self._img_float

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
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
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

                if len(feat.shape) == 3:
                    feat.shape = (feat.shape[2], feat.shape[0], feat.shape[1])

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



