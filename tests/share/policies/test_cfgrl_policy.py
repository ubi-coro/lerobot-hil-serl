from __future__ import annotations

import pytest
import torch

from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from share.policies.cfgrl_common.checkpointing import load_policy_bundle, save_policy_bundle
from share.policies.cfgrl_common.synthetic import make_synthetic_cfgrl_policy_batch
from share.policies.cfgrl_policy import CFGRLPolicy, make_cfgrl_policy_pre_post_processors
from tests.share.policies.helpers import DEFAULT_IMAGE_KEY, make_policy, make_policy_config


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(7)


def test_policy_observation_encoding_with_optional_inputs():
    policy = make_policy(metadata_dim=3, include_prev_action=True, hidden_dim=40)
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=3,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
        metadata_dim=3,
        include_prev_action=True,
    )

    encoding = policy.encode_observations(batch)

    assert encoding.context.shape == (3, policy.config.hidden_dim)
    assert encoding.tokens.shape[0] == 3
    assert encoding.tokens.shape[1] > 4
    assert list(encoding.camera_features) == [DEFAULT_IMAGE_KEY]
    assert encoding.camera_features[DEFAULT_IMAGE_KEY].shape == (3, policy.config.hidden_dim)


def test_policy_condition_mapping_and_dropout():
    policy = make_policy(condition_dropout_p=0.0)

    mapped = policy._raw_condition_to_embedding_index(torch.tensor([0, 1, 4, -2]))
    torch.testing.assert_close(mapped, torch.tensor([1, 2, 2, 1]))

    dropout_policy = make_policy(condition_dropout_p=1.0)
    dropout_policy.train()
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=4,
        chunk_size=dropout_policy.config.chunk_size,
        action_dim=dropout_policy.action_dim,
        image_size=dropout_policy.config.backbone.image_size,
        state_dim=dropout_policy.state_dim,
    )
    cond_idx = dropout_policy._sample_training_condition_indices(
        batch,
        mode="cfgrl",
        batch_size=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(cond_idx, torch.zeros(4, dtype=torch.long))

    bc_cond_idx = policy._sample_training_condition_indices(
        batch,
        mode="bc",
        batch_size=4,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(bc_cond_idx, torch.zeros(4, dtype=torch.long))


def test_policy_bc_and_weighted_cfgrl_losses():
    policy = make_policy(condition_dropout_p=0.0)
    policy.eval()
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=4,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )

    bc_loss, bc_logs = policy.compute_loss(batch, mode="bc")
    assert bc_loss.shape == ()
    assert torch.isfinite(bc_loss)
    assert "policy/loss" in bc_logs

    zero_weight_batch = dict(batch)
    zero_weight_batch["cfgrl_weight"] = torch.zeros(4)
    cfgrl_loss, cfgrl_logs = policy.compute_loss(zero_weight_batch, mode="cfgrl")
    assert cfgrl_loss.shape == ()
    torch.testing.assert_close(cfgrl_loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(cfgrl_logs["policy/weight_mean"], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_policy_flow_training_pair_matches_convention():
    policy = make_policy()
    actions = torch.tensor([[[1.0, -1.0]], [[0.5, 0.25]]])
    noise = torch.tensor([[[3.0, 1.0]], [[-0.5, 1.25]]])
    time = torch.tensor([0.0, 1.0])

    x_t, target = policy._build_flow_training_pair(actions, noise, time)

    torch.testing.assert_close(x_t[0], actions[0])
    torch.testing.assert_close(x_t[1], noise[1])
    torch.testing.assert_close(target, noise - actions)


def test_policy_cfg_sampling_identities():
    policy = make_policy(num_denoising_steps=3)
    policy.eval()
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=2,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )
    noise = torch.randn(2, policy.config.chunk_size, policy.action_dim)

    unconditional = policy.sample_actions(batch, condition=None, noise=noise, num_steps=3)
    cfg_zero = policy.sample_actions(batch, condition=1, guidance_scale=0.0, noise=noise, num_steps=3)
    conditioned = policy.sample_actions(batch, condition=1, guidance_scale=None, noise=noise, num_steps=3)
    cfg_one = policy.sample_actions(batch, condition=1, guidance_scale=1.0, noise=noise, num_steps=3)

    torch.testing.assert_close(unconditional, cfg_zero)
    torch.testing.assert_close(conditioned, cfg_one)


def test_policy_multi_sample_shapes_and_action_queue():
    policy = make_policy(chunk_size=6, n_action_steps=4)
    policy.eval()
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=2,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )

    multi = policy.sample_actions(batch, condition=1, guidance_scale=2.0, num_samples=3)
    chunk = policy.predict_action_chunk(batch)
    first_action = policy.select_action(batch)
    second_action = policy.select_action(batch)

    assert multi.shape == (2, 3, policy.config.chunk_size, policy.action_dim)
    assert chunk.shape == (2, policy.config.chunk_size, policy.action_dim)
    assert first_action.shape == (2, policy.action_dim)
    assert second_action.shape == (2, policy.action_dim)
    assert len(policy._queues[ACTION]) == policy.config.n_action_steps - 2


def test_policy_default_rollout_is_unconditional():
    policy = make_policy(num_denoising_steps=3)
    policy.eval()
    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=2,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )
    noise = torch.randn(2, policy.config.chunk_size, policy.action_dim)

    predicted = policy.predict_action_chunk(batch, noise=noise)
    unconditional = policy.sample_action_chunk(batch, condition=None, guidance_scale=None, noise=noise, num_steps=3)

    torch.testing.assert_close(predicted, unconditional)


def test_policy_projector_only_tuning_mode_is_honest():
    policy = make_policy()
    policy.config.backbone.tune_mode = "projector_only"
    policy._configure_vision_tuning()

    backbone_requires_grad = [param.requires_grad for param in policy.backbone.parameters()]
    projector_requires_grad = [param.requires_grad for param in policy.camera_token_proj.parameters()]

    assert all(not requires_grad for requires_grad in backbone_requires_grad)
    assert all(projector_requires_grad)


def test_policy_checkpoint_bundle_roundtrip(tmp_path):
    config = make_policy_config(default_rollout_condition=1, default_guidance_scale=1.5)
    policy = CFGRLPolicy(config)
    policy.eval()
    preprocessor, postprocessor = make_cfgrl_policy_pre_post_processors(config, dataset_stats=None)
    save_dir = save_policy_bundle(policy, tmp_path / "cfgrl_policy_bundle", preprocessor=preprocessor, postprocessor=postprocessor)
    loaded_policy, loaded_preprocessor, loaded_postprocessor = load_policy_bundle(
        CFGRLPolicy,
        save_dir,
        config=config,
    )
    loaded_policy.eval()

    batch = make_synthetic_cfgrl_policy_batch(
        batch_size=2,
        chunk_size=policy.config.chunk_size,
        action_dim=policy.action_dim,
        image_size=policy.config.backbone.image_size,
        state_dim=policy.state_dim,
    )
    noise = torch.randn(2, policy.config.chunk_size, policy.action_dim)

    original = policy.predict_action_chunk(batch, noise=noise)
    restored = loaded_policy.predict_action_chunk(batch, noise=noise)

    assert loaded_preprocessor is not None
    assert loaded_postprocessor is not None
    assert loaded_policy.config.default_rollout_condition == 1
    assert loaded_policy.config.default_guidance_scale == 1.5
    torch.testing.assert_close(original, restored, atol=1e-6, rtol=0.0)
