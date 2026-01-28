from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from lerobot.policies.xpvla_critic.configuration_xpvla_critic import CriticHeadConfig


@dataclass
class TDTarget:
    # For scalar: y: [B]
    # For C51:  target_probs: [B, N_atoms]
    # For IQN:  target_quantiles: [B, N_tau] (or [B, N_tau_target])
    scalar: Optional[Tensor] = None
    dist: Optional[Tensor] = None


class CriticHead(ABC, nn.Module):
    """
    Head consumes backbone features and produces Q outputs,
    defines expectation (for V and advantage), and loss vs TD target.
    """

    @abstractmethod
    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        """Return a dict of outputs for this head (e.g., {'q1':..., 'q2':...} or logits)."""

    @abstractmethod
    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        """Return E[Q] as [B] (typically min over twins, or mean if desired)."""

    @abstractmethod
    def build_target(
        self,
        *,
        reward_chunk: Tensor,      # [B]
        done: Tensor,             # [B]
        gamma_H: float,
        next_out: Dict[str, Tensor],
    ) -> TDTarget:
        """Build TD target given reward, done, and target-network outputs on next state."""

    @abstractmethod
    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        """Compute training loss for online outputs vs TD target."""

    @abstractmethod
    def soft_update_from(self, src: "CriticHead", tau: float) -> None:
        """Update this (target) head params from src (online) head."""


def make_critic_head(feat_dim, config: CriticHeadConfig) -> CriticHead:
    if config.type == "scalar":
        from lerobot.policies.xpvla_critic.heads.scalar import ScalarTwinQHead

        return ScalarTwinQHead(feat_dim=feat_dim, config=config)
    elif config.type == "c51":
        from lerobot.policies.xpvla_critic.heads.c51 import C51TwinQHead

        return C51TwinQHead(feat_dim=feat_dim, config=config)
    elif config.type == "iqn":
        from lerobot.policies.xpvla_critic.heads.iqn import IQNTwinQHead

        return IQNTwinQHead(feat_dim=feat_dim, config=config)
    else:
        raise ValueError(f"Unknown head_type={config.type}")
