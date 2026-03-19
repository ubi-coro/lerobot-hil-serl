"""Implicit Quantile Network head preserved as an optional critic variant."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from share.policies.cfgrl_common.modules import MLP
from share.policies.cfgrl_critic.configuration_cfgrl_critic import IQNHeadConfig
from .factory import CriticHead, TDTarget


def quantile_huber_loss(pred: Tensor, target: Tensor, taus: Tensor, kappa: float = 1.0) -> Tensor:
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = torch.abs(taus.unsqueeze(2) - (td.detach() < 0).float())
    return (weight * huber / kappa).mean()


class TauEmbedding(nn.Module):
    """Cosine tau embedding used by the IQN critic head."""

    def __init__(self, n_cos: int, out_dim: int) -> None:
        super().__init__()
        self.n_cos = n_cos
        self.linear = nn.Linear(n_cos, out_dim)

    def forward(self, taus: Tensor) -> Tensor:
        idx = torch.arange(1, self.n_cos + 1, device=taus.device, dtype=taus.dtype).view(1, 1, -1)
        emb = torch.cos(torch.pi * idx * taus.unsqueeze(-1))
        return F.relu(self.linear(emb))


class IQNTwinQHead(CriticHead):
    """Twin IQN head with quantile regression loss and expectation helpers."""

    def __init__(self, feat_dim: int, config: IQNHeadConfig):
        super().__init__()
        self.config = config
        self.tau_embed = TauEmbedding(config.n_cos, config.tau_embed_dim)
        self.tau_proj = nn.Linear(config.tau_embed_dim, feat_dim)
        self.q1 = MLP(feat_dim, config.hidden_dim, 1, num_layers=config.num_layers, dropout=config.dropout)
        self.q2 = MLP(feat_dim, config.hidden_dim, 1, num_layers=config.num_layers, dropout=config.dropout)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        batch_size = feat.shape[0]
        taus = torch.rand((batch_size, self.config.n_tau), device=feat.device, dtype=feat.dtype)
        tau_features = self.tau_proj(self.tau_embed(taus))
        feat = feat.unsqueeze(1) * tau_features
        return {
            "q1": self.q1(feat).squeeze(-1),
            "q2": self.q2(feat).squeeze(-1),
            "taus": taus,
        }

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        return torch.minimum(out["q1"].mean(dim=1), out["q2"].mean(dim=1))

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        z1 = next_out["q1"]
        z2 = next_out["q2"]
        z = torch.where(z1.mean(dim=1, keepdim=True) <= z2.mean(dim=1, keepdim=True), z1, z2)
        return TDTarget(dist=reward_chunk.unsqueeze(1) + (1.0 - done).unsqueeze(1) * gamma_H * z)

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        if target.dist is None:
            raise ValueError("IQN loss expects a quantile distribution target")
        return quantile_huber_loss(out["q1"], target.dist, out["taus"], kappa=self.config.kappa) + quantile_huber_loss(
            out["q2"], target.dist, out["taus"], kappa=self.config.kappa
        )

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        q1 = out["q1"].view(B, K, -1).mean(dim=1)
        q2 = out["q2"].view(B, K, -1).mean(dim=1)
        return {"q1": q1, "q2": q2}

    def soft_update_from(self, src: "IQNTwinQHead", tau: float) -> None:
        for param, src_param in zip(self.parameters(), src.parameters(), strict=True):
            param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)
