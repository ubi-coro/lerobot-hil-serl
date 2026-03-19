"""Action-provider abstractions used by the decoupled CFGRL critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION


@runtime_checkable
class SupportsActionSampling(Protocol):
    """Minimal policy sampling protocol required by ``PolicyActionProvider``."""

    def sample_actions(self, obs, condition=None, guidance_scale=None, num_samples: int = 1, noise=None):
        ...


class ActionProvider(Protocol):
    """Interface for supplying current and next action chunks to the critic."""

    def sample_current_actions(self, batch: dict, num_samples: int) -> Tensor:
        ...

    def sample_next_actions(self, batch: dict, num_samples: int) -> Tensor:
        ...


def _ensure_chunk_samples(actions: Tensor, num_samples: int) -> Tensor:
    """Normalize actions to ``[B, K, H, A]`` for critic-side value estimation."""

    if actions.ndim == 3:
        return actions.unsqueeze(1).expand(-1, num_samples, -1, -1)
    if actions.ndim != 4:
        raise ValueError(f"Expected action samples with ndim 3 or 4, got shape {tuple(actions.shape)}")
    if actions.shape[1] == num_samples:
        return actions
    if actions.shape[1] > num_samples:
        return actions[:, :num_samples]
    if actions.shape[1] == 1:
        return actions.expand(-1, num_samples, -1, -1)
    repeats = (num_samples + actions.shape[1] - 1) // actions.shape[1]
    tiled = actions.repeat(1, repeats, 1, 1)
    return tiled[:, :num_samples]


@dataclass
class DatasetActionProvider:
    """Use dataset actions for current and next chunk evaluation."""

    next_action_key: str = f"next_{ACTION}"

    def sample_current_actions(self, batch: dict, num_samples: int) -> Tensor:
        return _ensure_chunk_samples(batch[ACTION], num_samples)

    def sample_next_actions(self, batch: dict, num_samples: int) -> Tensor:
        key = self.next_action_key if self.next_action_key in batch else ACTION
        return _ensure_chunk_samples(batch[key], num_samples)


@dataclass
class CachedActionProvider:
    """Use cached policy-sampled action chunks already stored in the batch."""

    current_key: str = "policy_action_samples"
    next_key: str = "next_policy_action_samples"

    def sample_current_actions(self, batch: dict, num_samples: int) -> Tensor:
        return _ensure_chunk_samples(batch[self.current_key], num_samples)

    def sample_next_actions(self, batch: dict, num_samples: int) -> Tensor:
        return _ensure_chunk_samples(batch[self.next_key], num_samples)


@dataclass
class PolicyActionProvider:
    """Sample current and next chunks directly from a reference policy snapshot."""

    policy: SupportsActionSampling
    condition: int | None = None
    guidance_scale: float | None = None

    def sample_current_actions(self, batch: dict, num_samples: int) -> Tensor:
        return _ensure_chunk_samples(
            self.policy.sample_actions(
                batch["state"],
                condition=self.condition,
                guidance_scale=self.guidance_scale,
                num_samples=num_samples,
            ),
            num_samples,
        )

    def sample_next_actions(self, batch: dict, num_samples: int) -> Tensor:
        return _ensure_chunk_samples(
            self.policy.sample_actions(
                batch["next_state"],
                condition=self.condition,
                guidance_scale=self.guidance_scale,
                num_samples=num_samples,
            ),
            num_samples,
        )
