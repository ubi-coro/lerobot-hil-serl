from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from share.policies.cfgrl_common.backbones import MockVisionBackboneConfig
from share.policies.cfgrl_critic.configuration_cfgrl_critic import (
    C51HeadConfig,
    CFGRLCriticConfig,
    CriticBackboneConfig,
    CriticHeadConfig,
    ScalarFlowHeadConfig,
)
from share.policies.cfgrl_critic.modeling_cfgrl_critic import CFGRLCritic
from share.policies.cfgrl_policy.configuration_cfgrl_policy import CFGRLPolicyConfig
from share.policies.cfgrl_policy.modeling_cfgrl_policy import CFGRLPolicy


DEFAULT_IMAGE_KEY = f"{OBS_IMAGES}.main"


def make_input_features(
    *,
    image_size: tuple[int, int] = (32, 32),
    state_dim: int = 6,
    metadata_dim: int = 0,
    include_prev_action: bool = False,
    chunk_size: int = 8,
    action_dim: int = 5,
) -> dict[str, PolicyFeature]:
    height, width = image_size
    features = {
        DEFAULT_IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    if metadata_dim > 0:
        features["task_metadata"] = PolicyFeature(type=FeatureType.STATE, shape=(metadata_dim,))
    if include_prev_action:
        features["prev_action_chunk"] = PolicyFeature(type=FeatureType.ACTION, shape=(chunk_size, action_dim))
    return features


def make_output_features(*, action_dim: int = 5) -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}


def make_policy_config(
    *,
    chunk_size: int = 8,
    n_action_steps: int | None = None,
    action_dim: int = 5,
    state_dim: int = 6,
    image_size: tuple[int, int] = (32, 32),
    metadata_dim: int = 0,
    include_prev_action: bool = False,
    hidden_dim: int = 48,
    num_denoising_steps: int = 4,
    condition_dropout_p: float = 0.0,
    default_rollout_condition: int | None = None,
    default_guidance_scale: float | None = None,
) -> CFGRLPolicyConfig:
    return CFGRLPolicyConfig(
        input_features=make_input_features(
            image_size=image_size,
            state_dim=state_dim,
            metadata_dim=metadata_dim,
            include_prev_action=include_prev_action,
            chunk_size=chunk_size,
            action_dim=action_dim,
        ),
        output_features=make_output_features(action_dim=action_dim),
        backbone=MockVisionBackboneConfig(image_size=image_size, out_dim=32, hidden_dim=16),
        chunk_size=chunk_size,
        n_action_steps=n_action_steps or chunk_size,
        hidden_dim=hidden_dim,
        time_embed_dim=32,
        num_transformer_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        num_denoising_steps=num_denoising_steps,
        condition_dropout_p=condition_dropout_p,
        metadata_keys=["task_metadata"] if metadata_dim > 0 else [],
        previous_action_key="prev_action_chunk" if include_prev_action else None,
        default_rollout_condition=default_rollout_condition,
        default_guidance_scale=default_guidance_scale,
        device="cpu",
    )


def make_policy(**kwargs) -> CFGRLPolicy:
    return CFGRLPolicy(make_policy_config(**kwargs))


def make_critic_config(
    *,
    head: CriticHeadConfig | None = None,
    chunk_size: int = 8,
    action_dim: int = 5,
    state_dim: int = 6,
    image_size: tuple[int, int] = (32, 32),
    metadata_dim: int = 0,
    hidden_dim: int = 48,
    gamma: float = 0.99,
    tau: float = 0.1,
    num_action_samples: int = 2,
) -> CFGRLCriticConfig:
    return CFGRLCriticConfig(
        input_features=make_input_features(
            image_size=image_size,
            state_dim=state_dim,
            metadata_dim=metadata_dim,
            include_prev_action=False,
            chunk_size=chunk_size,
            action_dim=action_dim,
        ),
        output_features=make_output_features(action_dim=action_dim),
        chunk_size=chunk_size,
        gamma=gamma,
        tau=tau,
        num_action_samples=num_action_samples,
        backbone=CriticBackboneConfig(
            vision_backbone=MockVisionBackboneConfig(image_size=image_size, out_dim=32, hidden_dim=16),
            hidden_dim=hidden_dim,
            num_heads=4,
            num_fusion_layers=1,
            dropout=0.0,
        ),
        head=head or ScalarFlowHeadConfig(q_num_samples=2, hidden_dim=32, num_layers=2, num_flow_steps=3),
        metadata_keys=["task_metadata"] if metadata_dim > 0 else [],
        device="cpu",
    )


def make_critic(**kwargs) -> CFGRLCritic:
    return CFGRLCritic(make_critic_config(**kwargs))


def make_minmax_stats(*, state_dim: int = 6, action_dim: int = 5) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "min": -torch.ones(state_dim),
            "max": torch.ones(state_dim),
        },
        ACTION: {
            "min": -torch.ones(action_dim),
            "max": torch.ones(action_dim),
        },
    }


def make_c51_head() -> C51HeadConfig:
    return C51HeadConfig(n_atoms=17, hidden_dim=32, num_layers=2, dropout=0.0)
