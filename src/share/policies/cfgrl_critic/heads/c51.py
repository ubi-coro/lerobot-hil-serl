"""Discrete distributional C51 critic head used by the CFGRL critic family."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from share.policies.cfgrl_common.modules import MLP
from share.policies.cfgrl_critic.configuration_cfgrl_critic import C51HeadConfig
from .factory import CriticHead, TDTarget


def c51_projection(
    *,
    reward: Tensor,
    done: Tensor,
    gamma: float,
    atoms: Tensor,
    next_probs: Tensor,
    vmin: float,
    vmax: float,
) -> Tensor:
    """Project Bellman targets onto the fixed C51 support."""

    batch_size, n_atoms = next_probs.shape
    dz = (vmax - vmin) / (n_atoms - 1)
    tz = reward.view(batch_size, 1) + (1.0 - done).view(batch_size, 1) * gamma * atoms.view(1, n_atoms)
    tz = tz.clamp(vmin, vmax)

    b = (tz - vmin) / dz
    lower = b.floor().long()
    upper = b.ceil().long()
    target = torch.zeros((batch_size, n_atoms), device=next_probs.device, dtype=next_probs.dtype)

    offset = torch.arange(batch_size, device=next_probs.device).view(batch_size, 1) * n_atoms
    lower_idx = (lower + offset).reshape(-1)
    upper_idx = (upper + offset).reshape(-1)

    probs = next_probs.reshape(-1)
    b_flat = b.reshape(-1)
    lower_flat = lower.reshape(-1).float()
    upper_flat = upper.reshape(-1).float()
    same_bin = (lower_flat == upper_flat).to(dtype=next_probs.dtype)

    lower_mass = probs * (upper_flat - b_flat + same_bin)
    upper_mass = probs * (b_flat - lower_flat) * (1.0 - same_bin)

    target_flat = target.reshape(-1)
    target_flat.index_add_(0, lower_idx, lower_mass)
    target_flat.index_add_(0, upper_idx, upper_mass)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return target


class C51TwinQHead(CriticHead):
    """Twin C51 head with expectation and target-building helpers."""

    def __init__(self, feat_dim: int, config: C51HeadConfig):
        super().__init__()
        self.config = config
        self.logits1 = MLP(feat_dim, config.hidden_dim, config.n_atoms, num_layers=config.num_layers, dropout=config.dropout)
        self.logits2 = MLP(feat_dim, config.hidden_dim, config.n_atoms, num_layers=config.num_layers, dropout=config.dropout)
        atoms = torch.linspace(config.v_min, config.v_max, config.n_atoms)
        self.register_buffer("atoms", atoms, persistent=True)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {"logits1": self.logits1(feat), "logits2": self.logits2(feat)}

    def _probs(self, out: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if "probs1" in out and "probs2" in out:
            return out["probs1"], out["probs2"]
        return torch.softmax(out["logits1"], dim=-1), torch.softmax(out["logits2"], dim=-1)

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        probs1, probs2 = self._probs(out)
        q1 = (probs1 * self.atoms.view(1, -1)).sum(dim=-1)
        q2 = (probs2 * self.atoms.view(1, -1)).sum(dim=-1)
        return torch.minimum(q1, q2)

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        probs1, probs2 = next_out["probs1"], next_out["probs2"]
        exp1 = (probs1 * self.atoms.view(1, -1)).sum(dim=-1)
        exp2 = (probs2 * self.atoms.view(1, -1)).sum(dim=-1)
        next_probs = torch.where((exp1 <= exp2).unsqueeze(-1), probs1, probs2)
        return TDTarget(
            dist=c51_projection(
                reward=reward_chunk,
                done=done,
                gamma=gamma_H,
                atoms=self.atoms,
                next_probs=next_probs,
                vmin=self.config.v_min,
                vmax=self.config.v_max,
            )
        )

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        if target.dist is None:
            raise ValueError("C51 loss expects a distribution target")
        logp1 = F.log_softmax(out["logits1"], dim=-1)
        logp2 = F.log_softmax(out["logits2"], dim=-1)
        return (-(target.dist * logp1).sum(dim=-1).mean()) + (-(target.dist * logp2).sum(dim=-1).mean())

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        probs1 = torch.softmax(out["logits1"], dim=-1).view(B, K, -1).mean(dim=1)
        probs2 = torch.softmax(out["logits2"], dim=-1).view(B, K, -1).mean(dim=1)
        eps = 1e-8
        return {"probs1": probs1.clamp(min=eps), "probs2": probs2.clamp(min=eps)}

    def soft_update_from(self, src: "C51TwinQHead", tau: float) -> None:
        for param, src_param in zip(self.parameters(), src.parameters(), strict=True):
            param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)
