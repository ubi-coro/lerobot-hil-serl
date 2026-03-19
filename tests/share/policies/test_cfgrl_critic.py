from __future__ import annotations

import pytest
import torch

from lerobot.utils.constants import DONE, REWARD
from lerobot.utils.random_utils import set_seed
from share.policies.cfgrl_common.action_providers import CachedActionProvider, PolicyActionProvider
from share.policies.cfgrl_common.synthetic import make_synthetic_cfgrl_critic_batch
from share.policies.cfgrl_critic.configuration_cfgrl_critic import C51HeadConfig, ScalarFlowHeadConfig, ValueFlowsHeadConfig
from tests.share.policies.helpers import make_c51_head, make_critic


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(11)


def test_critic_encodes_observation_and_action_chunk():
    critic = make_critic()
    batch = make_synthetic_cfgrl_critic_batch(
        batch_size=3,
        chunk_size=critic.config.chunk_size,
        action_dim=critic.action_dim,
        image_size=critic.config.backbone.vision_backbone.image_size,
        state_dim=critic.state_dim,
    )

    feat = critic.encode_state_action(batch)
    q = critic.q(batch)
    context_tokens = critic.backbone.encode_context_tokens(batch["state"])

    assert feat.shape == (3, critic.backbone.out_dim)
    assert q.shape == (3,)
    assert context_tokens.shape[0] == 3
    assert context_tokens.shape[1] > 4


def test_critic_chunk_reward_accumulation_and_done_boundary():
    critic = make_critic(chunk_size=4, gamma=0.5)
    reward = torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0]])
    done = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])

    accumulated = critic._accumulate_chunk_reward(reward)
    chunk_done = critic._chunk_done(done)

    expected_reward = torch.tensor([1.0 + 0.5 * 2.0 + 0.25 * 3.0 + 0.125 * 4.0])
    torch.testing.assert_close(accumulated, expected_reward)
    torch.testing.assert_close(chunk_done, torch.tensor([1.0]))


@pytest.mark.parametrize(
    "head_config",
    [
        ScalarFlowHeadConfig(q_num_samples=2, hidden_dim=32, num_layers=2, num_flow_steps=3),
        C51HeadConfig(n_atoms=17, hidden_dim=32, num_layers=2),
        ValueFlowsHeadConfig(q_num_samples=2, hidden_dim=32, num_layers=2, num_flow_steps=3),
    ],
)
def test_critic_head_variants_smoke(head_config):
    critic = make_critic(head=head_config)
    batch = make_synthetic_cfgrl_critic_batch(
        batch_size=2,
        chunk_size=critic.config.chunk_size,
        action_dim=critic.action_dim,
        image_size=critic.config.backbone.vision_backbone.image_size,
        state_dim=critic.state_dim,
    )

    loss, logs = critic.compute_loss(batch)
    q = critic.q(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert q.shape == (2,)
    assert "critic/loss" in logs

    if isinstance(head_config, C51HeadConfig):
        next_out = critic._next_state_out(batch, CachedActionProvider())
        reward_chunk = critic._accumulate_chunk_reward(batch[REWARD])
        done = critic._chunk_done(batch[DONE])
        target = critic.target_head.build_target(
            reward_chunk=reward_chunk,
            done=done,
            gamma_H=float(critic.gamma ** critic.chunk_size),
            next_out=next_out,
        )
        assert target.dist is not None
        assert target.dist.shape == (2, head_config.n_atoms)
        torch.testing.assert_close(target.dist.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=0.0)


def test_critic_soft_update_target_moves_parameters():
    critic = make_critic(tau=0.25)
    source_param = next(critic.backbone.parameters())
    target_param = next(critic.target_backbone.parameters())
    old_target = target_param.detach().clone()

    with torch.no_grad():
        source_param.add_(1.0)

    critic.soft_update_target()

    expected = old_target * 0.75 + source_param.detach() * 0.25
    torch.testing.assert_close(target_param, expected)


def test_critic_projector_only_tuning_mode_is_honest():
    critic = make_critic()
    projector_only = make_critic()
    projector_only.config.backbone.vision_backbone.tune_mode = "projector_only"
    projector_only.backbone._configure_vision_tuning()

    frozen_backbone_params = [param.requires_grad for param in projector_only.backbone.vision_backbone.parameters()]
    projector_params = [param.requires_grad for param in projector_only.backbone.camera_token_proj.parameters()]

    assert all(not requires_grad for requires_grad in frozen_backbone_params)
    assert all(projector_params)


def test_critic_policy_action_provider_path():
    class MockSamplingPolicy:
        def __init__(self, chunk_size: int, action_dim: int):
            self.chunk_size = chunk_size
            self.action_dim = action_dim

        def sample_actions(self, obs, condition=None, guidance_scale=None, num_samples: int = 1, noise=None):
            batch_size = obs["observation.state"].shape[0]
            base = torch.linspace(-0.25, 0.25, self.chunk_size * self.action_dim).view(
                1, 1, self.chunk_size, self.action_dim
            )
            return base.expand(batch_size, num_samples, -1, -1).clone()

    critic = make_critic(head=make_c51_head())
    batch = make_synthetic_cfgrl_critic_batch(
        batch_size=3,
        chunk_size=critic.config.chunk_size,
        action_dim=critic.action_dim,
        image_size=critic.config.backbone.vision_backbone.image_size,
        state_dim=critic.state_dim,
    )
    provider = PolicyActionProvider(policy=MockSamplingPolicy(critic.config.chunk_size, critic.action_dim), condition=1, guidance_scale=2.0)

    value = critic.estimate_value(batch, provider, next_state=False)
    next_value = critic.estimate_value(batch, provider, next_state=True)
    advantage = critic.compute_advantage(batch, provider)

    assert value.shape == (3,)
    assert next_value.shape == (3,)
    assert advantage.shape == (3,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(next_value).all()
    assert torch.isfinite(advantage).all()
