"""CPU-only sanity runner for the share-local CFGRL policy and critic."""

from __future__ import annotations

import argparse

import torch

from share.policies.cfgrl_common.synthetic import (
    make_synthetic_cfgrl_critic_batch,
    make_synthetic_cfgrl_policy_batch,
)
from share.policies.cfgrl_policy.configuration_cfgrl_policy import CFGRLPolicyConfig
from share.policies.cfgrl_policy.modeling_cfgrl_policy import CFGRLPolicy
from share.policies.cfgrl_critic.configuration_cfgrl_critic import (
    CFGRLCriticConfig,
    CriticBackboneConfig,
    ScalarFlowHeadConfig,
)
from share.policies.cfgrl_critic.modeling_cfgrl_critic import CFGRLCritic
from share.policies.cfgrl_common.backbones import MockVisionBackboneConfig
from share.policies.cfgrl_common.action_providers import CachedActionProvider
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the tiny CFGRL sanity loop."""

    parser = argparse.ArgumentParser(description="Run a tiny CPU sanity loop for the CFGRL stack.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_features(action_dim: int, state_dim: int, image_size: tuple[int, int]) -> tuple[dict, dict]:
    """Construct minimal policy feature dictionaries for the sanity run."""

    h, w = image_size
    input_features = {
        f"{OBS_IMAGES}.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return input_features, output_features


def main() -> None:
    """Run a tiny CPU training loop and print one rollout shape."""

    args = parse_args()
    device = torch.device(args.device)
    image_size = (32, 32)
    state_dim = 6
    action_dim = 5
    chunk_size = 8
    input_features, output_features = build_features(action_dim, state_dim, image_size)

    policy_cfg = CFGRLPolicyConfig(
        device=str(device),
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features=input_features,
        output_features=output_features,
        backbone=MockVisionBackboneConfig(),
    )
    critic_cfg = CFGRLCriticConfig(
        device=str(device),
        chunk_size=chunk_size,
        input_features=input_features,
        output_features=output_features,
        backbone=CriticBackboneConfig(vision_backbone=MockVisionBackboneConfig()),
        head=ScalarFlowHeadConfig(),
    )

    policy = CFGRLPolicy(policy_cfg).to(device)
    critic = CFGRLCritic(critic_cfg).to(device)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    provider = CachedActionProvider()

    for _ in range(args.steps):
        policy_batch = make_synthetic_cfgrl_policy_batch(
            batch_size=4,
            chunk_size=chunk_size,
            action_dim=action_dim,
            image_size=image_size,
            state_dim=state_dim,
            device=device,
        )
        critic_batch = make_synthetic_cfgrl_critic_batch(
            batch_size=4,
            chunk_size=chunk_size,
            action_dim=action_dim,
            image_size=image_size,
            state_dim=state_dim,
            device=device,
        )

        policy_optim.zero_grad()
        loss_bc, _ = policy.compute_loss(policy_batch, mode="bc")
        loss_bc.backward()
        policy_optim.step()

        policy_optim.zero_grad()
        loss_cfgrl, _ = policy.compute_loss(policy_batch, mode="cfgrl")
        loss_cfgrl.backward()
        policy_optim.step()

        critic_optim.zero_grad()
        critic_loss, _ = critic.compute_loss(critic_batch, action_provider=provider)
        critic_loss.backward()
        critic_optim.step()
        critic.soft_update_target()

    rollout = policy.predict_action_chunk(make_synthetic_cfgrl_policy_batch(
        batch_size=2,
        chunk_size=chunk_size,
        action_dim=action_dim,
        image_size=image_size,
        state_dim=state_dim,
        device=device,
    ))
    print({"policy_chunk_shape": tuple(rollout.shape)})


if __name__ == "__main__":
    main()
