"""Processor helpers for CFGRL critic batches and chunked transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_PREFIX, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME, REWARD
from .configuration_cfgrl_critic import CFGRLCriticConfig


def make_cfgrl_critic_pre_post_processors(
    config: CFGRLCriticConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build share-local pre/post processors for the CFGRL critic."""

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        CFGRLCriticImagePreprocessStep(
            image_keys=sorted(config.image_features.keys()),
            image_size=config.backbone.vision_backbone.image_size,
            image_mean=config.backbone.vision_backbone.image_mean,
            image_std=config.backbone.vision_backbone.image_std,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping, stats=dataset_stats),
        CFGRLChunkedTransitionProcessorStep(chunk_size=config.chunk_size),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="cfgrl_critic_image_preprocess")
class CFGRLCriticImagePreprocessStep(ObservationProcessorStep):
    """Normalize critic image observations without XVLA-specific preprocessing."""

    image_keys: list[str]
    image_size: tuple[int, int]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[1] not in {1, 3} and image.shape[-1] in {1, 3}:
            image = image.permute(0, 3, 1, 2).contiguous()
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        if tuple(image.shape[-2:]) != tuple(self.image_size):
            image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        mean = torch.tensor(self.image_mean, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
        return (image - mean) / std

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_obs = dict(observation)
        keys = self.image_keys or [key for key in observation if key.startswith(OBS_IMAGES)]
        for key in keys:
            if key in observation:
                new_obs[key] = self._prepare(observation[key])
        return new_obs

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": list(self.image_keys),
            "image_size": list(self.image_size),
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="cfgrl_chunked_transition")
class CFGRLChunkedTransitionProcessorStep(ObservationProcessorStep):
    """Reshape flat transitions into chunk-native critic training batches."""

    chunk_size: int = 8

    def _split_state_tensors(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        state: dict[str, Any] = {}
        next_state: dict[str, Any] = {}
        has_next = False
        obs_keys = [key for key in batch if key.startswith(OBS_PREFIX)]

        for key in obs_keys:
            value = batch.pop(key)
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == 2:
                state[key] = value[:, 0]
                next_state[key] = value[:, 1]
                has_next = True
            else:
                state[key] = value
        return state, next_state, has_next

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "state" in observation and "next_state" in observation:
            return observation

        batch = dict(observation)
        state, next_state, has_next = self._split_state_tensors(batch)
        if state:
            batch["state"] = state
        if has_next:
            batch["next_state"] = next_state

        if ACTION in batch and torch.is_tensor(batch[ACTION]):
            action = batch[ACTION]
            if action.ndim == 3 and action.shape[1] == 2 * self.chunk_size:
                batch[ACTION] = action[:, : self.chunk_size]
                batch[f"next_{ACTION}"] = action[:, self.chunk_size :]
        if REWARD in batch and torch.is_tensor(batch[REWARD]) and batch[REWARD].ndim == 2 and batch[REWARD].shape[1] == 2 * self.chunk_size:
            batch[REWARD] = batch[REWARD][:, : self.chunk_size]
        if DONE in batch and torch.is_tensor(batch[DONE]) and batch[DONE].ndim == 2 and batch[DONE].shape[1] == 2 * self.chunk_size:
            batch[DONE] = batch[DONE][:, : self.chunk_size]
        return batch

    def get_config(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
