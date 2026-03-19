"""Synthetic CPU-safe data generators for CFGRL tests and sanity runs."""

from __future__ import annotations

import torch

from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD


def make_synthetic_observation_batch(
    *,
    batch_size: int = 4,
    image_size: tuple[int, int] = (32, 32),
    state_dim: int = 6,
    metadata_dim: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a minimal visuomotor observation batch."""

    height, width = image_size
    obs = {
        f"{OBS_IMAGES}.main": torch.rand(batch_size, 3, height, width, device=device),
        OBS_STATE: torch.randn(batch_size, state_dim, device=device),
    }
    if metadata_dim > 0:
        obs["task_metadata"] = torch.randn(batch_size, metadata_dim, device=device)
    return obs


def make_synthetic_cfgrl_policy_batch(
    *,
    batch_size: int = 4,
    chunk_size: int = 8,
    action_dim: int = 5,
    image_size: tuple[int, int] = (32, 32),
    state_dim: int = 6,
    metadata_dim: int = 0,
    include_prev_action: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a policy-training batch with CFGRL labels and weights."""

    batch = make_synthetic_observation_batch(
        batch_size=batch_size,
        image_size=image_size,
        state_dim=state_dim,
        metadata_dim=metadata_dim,
        device=device,
    )
    batch[ACTION] = torch.randn(batch_size, chunk_size, action_dim, device=device)
    batch["cfgrl_condition"] = torch.randint(0, 2, (batch_size,), device=device)
    batch["cfgrl_weight"] = torch.rand(batch_size, device=device) + 0.5
    if include_prev_action:
        batch["prev_action_chunk"] = torch.randn(batch_size, chunk_size, action_dim, device=device)
    return batch


def make_synthetic_cfgrl_critic_batch(
    *,
    batch_size: int = 4,
    chunk_size: int = 8,
    action_dim: int = 5,
    image_size: tuple[int, int] = (32, 32),
    state_dim: int = 6,
    metadata_dim: int = 0,
    num_policy_samples: int = 3,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Create a critic-training batch with chunk rewards and cached action samples."""

    state = make_synthetic_observation_batch(
        batch_size=batch_size,
        image_size=image_size,
        state_dim=state_dim,
        metadata_dim=metadata_dim,
        device=device,
    )
    next_state = make_synthetic_observation_batch(
        batch_size=batch_size,
        image_size=image_size,
        state_dim=state_dim,
        metadata_dim=metadata_dim,
        device=device,
    )
    batch: dict[str, object] = {
        "state": state,
        "next_state": next_state,
        ACTION: torch.randn(batch_size, chunk_size, action_dim, device=device),
        f"next_{ACTION}": torch.randn(batch_size, chunk_size, action_dim, device=device),
        REWARD: torch.randn(batch_size, chunk_size, device=device),
        DONE: torch.zeros(batch_size, device=device),
        "policy_action_samples": torch.randn(batch_size, num_policy_samples, chunk_size, action_dim, device=device),
        "next_policy_action_samples": torch.randn(
            batch_size, num_policy_samples, chunk_size, action_dim, device=device
        ),
    }
    return batch
