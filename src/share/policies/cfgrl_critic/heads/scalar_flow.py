"""Scalar flow-based critic head for chunk return estimation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from share.policies.cfgrl_common.modules import MLP
from share.policies.cfgrl_critic.configuration_cfgrl_critic import ScalarFlowHeadConfig
from .factory import CriticHead, TDTarget


class _ReturnVectorField(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.net = MLP(
            feat_dim + 2,
            hidden_dim,
            1,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, noisy_return: Tensor, time: Tensor, feat: Tensor) -> Tensor:
        return self.net(torch.cat([feat, noisy_return, time], dim=-1))


class ScalarFlowTwinQHead(CriticHead):
    """Twin scalar-flow critic head with flow-based TD supervision."""

    def __init__(self, feat_dim: int, config: ScalarFlowHeadConfig):
        super().__init__()
        self.config = config
        self.flow1 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)
        self.flow2 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "q1": self._estimate_q(feat, which=1),
            "q2": self._estimate_q(feat, which=2),
        }

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        return torch.minimum(out["q1"], out["q2"])

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        v_next = torch.minimum(next_out["q1"], next_out["q2"])
        y = reward_chunk + (1.0 - done) * gamma_H * v_next
        return TDTarget(scalar=y)

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        if target.scalar is None:
            raise ValueError("ScalarFlow loss expects a scalar target")
        return F.mse_loss(out["q1"], target.scalar) + F.mse_loss(out["q2"], target.scalar)

    def loss_from_target(self, feat: Tensor, target_scalar: Tensor) -> Tensor:
        batch_size = feat.shape[0]
        device = feat.device
        noise = torch.randn(batch_size, 1, device=device, dtype=feat.dtype)
        time = torch.rand(batch_size, 1, device=device, dtype=feat.dtype)
        target = target_scalar.view(batch_size, 1)
        noisy_return = time * target + (1.0 - time) * noise
        target_field = target - noise
        vf1 = self.flow1(noisy_return, time, feat)
        vf2 = self.flow2(noisy_return, time, feat)
        return F.mse_loss(vf1, target_field) + F.mse_loss(vf2, target_field)

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        return {
            "q1": out["q1"].view(B, K).mean(dim=1),
            "q2": out["q2"].view(B, K).mean(dim=1),
        }

    def soft_update_from(self, src: "ScalarFlowTwinQHead", tau: float) -> None:
        for param, src_param in zip(self.parameters(), src.parameters(), strict=True):
            param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)

    def _flow(self, which: int) -> _ReturnVectorField:
        return self.flow1 if which == 1 else self.flow2

    def _compute_flow_returns(
        self,
        *,
        noise: Tensor,
        feat: Tensor,
        which: int,
    ) -> Tensor:
        value = noise
        step = 1.0 / float(self.config.num_flow_steps)
        net = self._flow(which)
        for idx in range(self.config.num_flow_steps):
            time = torch.full_like(noise, idx * step)
            value = value + step * net(value, time, feat)
            if self.config.clip_flow_returns:
                value = value.clamp(self.config.v_min, self.config.v_max)
        return value

    def _estimate_q(self, feat: Tensor, *, which: int) -> Tensor:
        batch_size = feat.shape[0]
        num_samples = int(self.config.q_num_samples)
        noise = torch.randn(batch_size * num_samples, 1, device=feat.device, dtype=feat.dtype)
        feat_rep = feat.repeat_interleave(num_samples, dim=0)
        ret = self._compute_flow_returns(noise=noise, feat=feat_rep, which=which)
        ret = ret.view(batch_size, num_samples).mean(dim=1)
        if self.config.clip_flow_returns:
            ret = ret.clamp(self.config.v_min, self.config.v_max)
        return ret
