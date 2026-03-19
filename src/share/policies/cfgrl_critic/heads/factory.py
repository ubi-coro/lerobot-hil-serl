"""Factory interfaces shared by the CFGRL critic head implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor, nn

from share.policies.cfgrl_critic.configuration_cfgrl_critic import CriticHeadConfig


@dataclass
class TDTarget:
    """Generic container for scalar or distributional TD targets."""

    scalar: Optional[Tensor] = None
    dist: Optional[Tensor] = None


class CriticHead(ABC, nn.Module):
    """Abstract interface implemented by all critic head variants."""

    @abstractmethod
    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def build_target(
        self,
        *,
        reward_chunk: Tensor,
        done: Tensor,
        gamma_H: float,
        next_out: Dict[str, Tensor],
    ) -> TDTarget:
        raise NotImplementedError

    @abstractmethod
    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def soft_update_from(self, src: "CriticHead", tau: float) -> None:
        raise NotImplementedError


def make_critic_head(feat_dim: int, config: CriticHeadConfig) -> CriticHead:
    """Instantiate the requested critic head implementation."""

    if config.type == "scalar_flow":
        from .scalar_flow import ScalarFlowTwinQHead

        return ScalarFlowTwinQHead(feat_dim=feat_dim, config=config)
    if config.type == "c51":
        from .c51 import C51TwinQHead

        return C51TwinQHead(feat_dim=feat_dim, config=config)
    if config.type == "iqn":
        from .iqn import IQNTwinQHead

        return IQNTwinQHead(feat_dim=feat_dim, config=config)
    if config.type == "value_flows":
        from .value_flows import ValueFlowsTwinQHead

        return ValueFlowsTwinQHead(feat_dim=feat_dim, config=config)
    raise ValueError(f"Unknown head_type={config.type}")
