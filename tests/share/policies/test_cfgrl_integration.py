from __future__ import annotations

import pytest
import torch

from lerobot.utils.random_utils import set_seed
from share.policies.cfgrl_common.action_providers import PolicyActionProvider
from share.policies.cfgrl_common.synthetic import (
    make_synthetic_cfgrl_critic_batch,
    make_synthetic_cfgrl_policy_batch,
)
from tests.share.policies.helpers import make_critic, make_policy


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(19)


def test_cfgrl_policy_and_critic_cpu_training_loop():
    policy = make_policy(chunk_size=6, n_action_steps=4, hidden_dim=40, num_denoising_steps=3)
    critic = make_critic(chunk_size=6, hidden_dim=40, num_action_samples=2)

    policy_batch = make_synthetic_cfgrl_policy_batch(
        batch_size=4,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )
    critic_batch = make_synthetic_cfgrl_critic_batch(
        batch_size=4,
        chunk_size=critic.config.chunk_size,
        action_dim=critic.action_dim,
        image_size=critic.config.backbone.vision_backbone.image_size,
        state_dim=critic.state_dim,
    )

    policy_opt = torch.optim.Adam(policy.get_optim_params()["params"], lr=1e-3)
    critic_opt = torch.optim.Adam(critic.get_optim_params()["params"], lr=1e-3)

    initial_policy_params = [param.detach().clone() for param in policy.parameters() if param.requires_grad]
    initial_critic_params = [param.detach().clone() for param in critic.parameters() if param.requires_grad]

    policy.train()
    for _ in range(2):
        policy_opt.zero_grad()
        loss, _ = policy.compute_loss(policy_batch, mode="bc")
        loss.backward()
        policy_opt.step()

    weighted_batch = dict(policy_batch)
    weighted_batch["cfgrl_weight"] = torch.linspace(0.25, 1.0, steps=policy_batch["cfgrl_weight"].shape[0])
    for _ in range(2):
        policy_opt.zero_grad()
        loss, _ = policy.compute_loss(weighted_batch, mode="cfgrl")
        loss.backward()
        policy_opt.step()

    policy.eval()
    critic_batch["policy_action_samples"] = policy.sample_actions(
        critic_batch["state"],
        condition=1,
        guidance_scale=2.0,
        num_samples=critic.config.num_action_samples,
    )
    critic_batch["next_policy_action_samples"] = policy.sample_actions(
        critic_batch["next_state"],
        condition=1,
        guidance_scale=2.0,
        num_samples=critic.config.num_action_samples,
    )

    critic.train()
    for _ in range(2):
        critic_opt.zero_grad()
        loss, _ = critic.compute_loss(critic_batch)
        loss.backward()
        critic_opt.step()
        critic.soft_update_target()

    live_provider = PolicyActionProvider(policy=policy, condition=1, guidance_scale=2.0)
    advantage = critic.compute_advantage(critic_batch, live_provider)
    sampled_chunk = policy.predict_action_chunk(policy_batch)
    first_action = policy.select_action(policy_batch)
    second_action = policy.select_action(policy_batch)

    updated_policy_params = [param.detach() for param in policy.parameters() if param.requires_grad]
    updated_critic_params = [param.detach() for param in critic.parameters() if param.requires_grad]

    assert any(not torch.allclose(before, after) for before, after in zip(initial_policy_params, updated_policy_params, strict=True))
    assert any(not torch.allclose(before, after) for before, after in zip(initial_critic_params, updated_critic_params, strict=True))
    assert advantage.shape == (4,)
    assert sampled_chunk.shape == (4, policy.config.chunk_size, policy.action_dim)
    assert first_action.shape == (4, policy.action_dim)
    assert second_action.shape == (4, policy.action_dim)
    assert torch.isfinite(advantage).all()
