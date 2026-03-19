"""Processor helpers for CFGRL policy training and rollout batches."""

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
from lerobot.utils.constants import OBS_IMAGES, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from .configuration_cfgrl_policy import CFGRLPolicyConfig


def make_cfgrl_policy_pre_post_processors(
    config: CFGRLPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build share-local pre/post processors for the CFGRL policy."""

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        CFGRLImagePreprocessStep(
            image_keys=sorted(config.image_features.keys()),
            image_size=config.backbone.image_size,
            image_mean=config.backbone.image_mean,
            image_std=config.backbone.image_std,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping, stats=dataset_stats),
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
@ProcessorStepRegistry.register(name="cfgrl_image_preprocess")
class CFGRLImagePreprocessStep(ObservationProcessorStep):
    """Normalize policy image observations without relying on XVLA processors."""

    image_keys: list[str]
    image_size: tuple[int, int]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image tensors to normalized ``BCHW`` float batches."""

        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Unsupported image shape {tuple(image.shape)}")
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
        """Apply image preprocessing to all configured visual observation keys."""

        new_obs = dict(observation)
        keys = self.image_keys or [key for key in observation if key.startswith(OBS_IMAGES)]
        for key in keys:
            if key in observation:
                new_obs[key] = self._prepare(observation[key])
        return new_obs

    def get_config(self) -> dict[str, Any]:
        """Return a serializable config so saved processors can be reloaded."""

        return {
            "image_keys": list(self.image_keys),
            "image_size": list(self.image_size),
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Preserve feature contracts because preprocessing changes values, not schema."""

        return features
