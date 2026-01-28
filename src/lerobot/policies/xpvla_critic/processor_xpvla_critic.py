from dataclasses import dataclass
from typing import Any, Optional

import torch

from lerobot.configs.types import PolicyFeature, PipelineFeatureType
from lerobot.policies.xpvla_critic.configuration_xpvla_critic import XPVLACriticConfig
from lerobot.policies.xvla.processor_xvla import XVLAImageToFloatProcessorStep, XVLAImageNetNormalizeProcessorStep, XVLAAddDomainIdProcessorStep
from lerobot.processor import PolicyProcessorPipeline, PolicyAction, RenameObservationsProcessorStep, AddBatchDimensionProcessorStep, \
    DeviceProcessorStep, NormalizerProcessorStep, UnnormalizerProcessorStep, ObservationProcessorStep, ProcessorStepRegistry, TokenizerProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME, ACTION, OBS_PREFIX,
)

try:
    from transformers import AutoTokenizer

    _transformers_available = True
except Exception:
    AutoTokenizer = None
    _transformers_available = False


def make_xpvla_critic_pre_post_processors(
        config: XPVLACriticConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.backbone.tokenizer_name,
            max_length=config.backbone.tokenizer_max_length,
            padding=config.backbone.pad_language_to,
            padding_side=config.backbone.tokenizer_padding_side
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        CriticChunkedSARSAProcessorStep(
            chunk_size=config.chunk_size,
            vlm_cache_key=config.backbone.vlm_cache_key
        )
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
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="critic_chunked_sarsa_processor")
class CriticChunkedSARSAProcessorStep(ObservationProcessorStep):
    """
    Builds a nested SARSA-style batch for the XP-VLA critic.

    Expected input (training-style after delta-index collation):
      OBS_STATE:            [B, 2, ...]           (t and t+H)
      OBS_IMAGES_*:         [B, 2, C, H, W]       (t and t+H)
      vlm_cache_key:        dict[str, Tensor] with tensors shaped [B,2,...] (optional)
      ACTION:               [B, 2H, A]
      REWARD:               [B, H] or [B, 2H] or [B]
      DONE:                 [B] or [B, H] or [B, 2H]

    Output:
      "state":              dict with obs at t
      "next_state":         dict with obs at t+H (if available)
      ACTION:               [B, H, A]
      "next_ACTION":        [B, H, A] (if ACTION had 2H steps)
      REWARD:               [B, H] or [B]
      DONE:                 [B]  (done at chunk boundary t+H)
    """

    chunk_size: int = 50

    # Outer key for cached XVLA.forward_vlm outputs stored inside observation/state dicts.
    # Recommended: "vlm_cache". Set None to disable handling.
    vlm_cache_key: Optional[str] = "vlm_cache"

    # -------------------------
    # helpers
    # -------------------------

    @staticmethod
    def _split_two_timestep_tensor(x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim >= 2 and x.shape[1] == 2:
            return x[:, 0], x[:, 1]
        if x.ndim >= 2 and x.shape[1] == 1:
            return x[:, 0], None
        return x, None

    def _extract_state_like(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        state: dict[str, Any] = {}
        next_state: dict[str, Any] = {}
        has_next = False

        obs_keys: list[str] = []
        for key in batch:
            if key.startswith(OBS_PREFIX):
                obs_keys.append(key)

        for k in obs_keys:
            v = batch.pop(k, None)
            if v is None:
                continue

            # cached VLM dict: split each tensor inside
            if isinstance(v, dict):
                s_dict: dict[str, Any] = {}
                ns_dict: dict[str, Any] = {}
                dict_has_next = False
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        s_v, ns_v = self._split_two_timestep_tensor(vv)
                        s_dict[kk] = s_v
                        if ns_v is not None:
                            ns_dict[kk] = ns_v
                            dict_has_next = True
                    else:
                        s_dict[kk] = vv
                state[k] = s_dict
                if dict_has_next:
                    next_state[k] = ns_dict
                    has_next = True
                continue

            # normal tensor: split [B,2,...]
            if torch.is_tensor(v):
                s_v, ns_v = self._split_two_timestep_tensor(v)
                state[k] = s_v
                if ns_v is not None:
                    next_state[k] = ns_v
                    has_next = True
                continue

            # metadata
            state[k] = v

        return state, next_state, has_next

    # -------------------------
    # main processor hook
    # -------------------------

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.state_key in observation:
            return observation

        batch = dict(observation)
        H = int(self.chunk_size)

        # 1) build nested state/next_state
        state, next_state, has_next = self._extract_state_like(batch)
        if state:
            batch["state"] = state
        if has_next and next_state:
            batch["next_state"] = next_state

        # 2) split ACTION into [B,H,A] and next_ACTION [B,H,A]
        if ACTION in batch and torch.is_tensor(batch[ACTION]):
            a: torch.Tensor = batch[ACTION]
            if a.ndim < 2:
                raise ValueError(f"Expected ACTION with time axis, got {tuple(a.shape)}")

            T = int(a.shape[1])
            if T == 2 * H:
                batch[ACTION] = a[:, :H]
                batch[f"next_{ACTION}"] = a[:, H:]
            elif T == H:
                # already chunked
                pass
            else:
                raise ValueError(f"Expected ACTION T=H or T=2H with H={H}, got T={T}, shape={tuple(a.shape)}")

        return batch

    # -------------------------
    # required interface
    # -------------------------

    def get_config(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "state_key": self.state_key,
            "next_state_key": self.next_state_key,
            "next_action_key": self.next_action_key,
            "vlm_cache_key": self.vlm_cache_key,
        }

    def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This processor restructures the batch (wraps obs into nested dicts), but it does not
        introduce *new* leaf tensors; it just changes where they live.

        In practice you should place this step *after* any processors that rely on flat
        observation keys (normalization/image transforms). Therefore we keep the feature
        declarations unchanged.

        If your pipeline performs schema validation *after* this step and expects explicit
        declarations for 'state'/'next_state', you can uncomment the optional block below.
        """
        # Optional schema declaration (only enable if your infra requires it):
        # if self.state_key not in features[PipelineFeatureType.OBSERVATION]:
        #     features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
        #         type=FeatureType.STATE, shape=()
        #     )
        # if self.next_state_key not in features[PipelineFeatureType.OBSERVATION]:
        #     features[PipelineFeatureType.OBSERVATION][self.next_state_key] = PolicyFeature(
        #         type=FeatureType.STATE, shape=()
        #     )

        return features
