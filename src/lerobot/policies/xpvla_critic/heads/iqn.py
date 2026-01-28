from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.xpvla_critic.configuration_xpvla_critic import IQNHeadConfig
from lerobot.policies.xpvla_critic.heads.factory import CriticHead, TDTarget
from lerobot.policies.xpvla_critic.nets import MLP

Tensor = torch.Tensor


def quantile_huber_loss(pred: Tensor, target: Tensor, taus: Tensor, kappa: float = 1.0) -> Tensor:
    """
    pred:   [B, N]   quantile values for taus
    target: [B, M]   target quantile values for taus'
    taus:   [B, N]   quantiles used for pred
    """
    # pairwise TD error: [B, N, M]
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_td = td.abs()

    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    # quantile weight
    tau = taus.unsqueeze(2)  # [B,N,1]
    weight = torch.abs(tau - (td.detach() < 0).float())
    return (weight * huber / kappa).mean()


class TauEmbedding(nn.Module):
    def __init__(self, n_cos: int, out_dim: int):
        super().__init__()
        self.n_cos = n_cos
        self.lin = nn.Linear(n_cos, out_dim)

    def forward(self, taus: Tensor) -> Tensor:
        # taus: [B,N]
        i = torch.arange(1, self.n_cos + 1, device=taus.device, dtype=taus.dtype).view(1, 1, -1)
        x = torch.cos(i * 3.141592653589793 * taus.unsqueeze(-1))  # [B,N,n_cos]
        return F.relu(self.lin(x))  # [B,N,out_dim]


class IQNTwinQHead(CriticHead):
    def __init__(self, feat_dim: int, config: IQNHeadConfig):
        super().__init__()
        self.config = config
        self.tau_embed = TauEmbedding(config.n_cos, config.tau_embed_dim)
        self.tau_proj = torch.nn.Linear(config.tau_embed_dim, feat_dim)

        # Map (feat ⊙ tau_emb) -> scalar
        self.q1 = MLP(feat_dim, config.hidden_dim, 1, num_layers=config.num_layers, dropout=config.dropout)
        self.q2 = MLP(feat_dim, config.hidden_dim, 1, num_layers=config.num_layers, dropout=config.dropout)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        B = feat.shape[0]
        taus = self._sample_taus(B, self.config.n_tau, feat.device)  # [B,N]
        tau_e = self.tau_embed(taus)          # [B, N, tau_embed_dim]
        tau_phi = self.tau_proj(tau_e)        # [B, N, feat_dim]
        feat_exp = feat.unsqueeze(1)          # [B, 1, feat_dim]
        f = feat_exp * tau_phi                # [B, N, feat_dim]
        q1 = self.q1(f).squeeze(-1)           # [B, N]
        q2 = self.q2(f).squeeze(-1)           # [B, N]
        return {"q1": q1, "q2": q2, "taus": taus}

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        # mean over taus
        q1 = out["q1"].mean(dim=1)
        q2 = out["q2"].mean(dim=1)
        return torch.minimum(q1, q2)

    @torch.no_grad()
    def reduce_over_action_samples(self, out: dict[str, Tensor], *, B: int, K: int) -> dict[str, Tensor]:
        """
        Reduce IQN target-head outputs computed for B*K policy action samples down to B.

        Input:
          out["q1"], out["q2"]: [B*K, T]  (T = n_tau used in that forward)

        Output:
          {"q1": [B, T], "q2": [B, T]}

        Semantics:
          Elementwise mean over K action samples, for each tau index.
          (This is consistent if the head uses the same taus across the B*K batch;
           if taus are independently sampled per row, this still works but has higher variance.)
        """
        q1 = out["q1"]
        q2 = out["q2"]

        if q1.ndim != 2:
            raise ValueError(f"IQN reduce expects q1 [B*K,T], got {tuple(q1.shape)}")
        T = q1.shape[1]
        if q1.shape[0] != B * K:
            raise ValueError(f"Expected q1 batch dim {B*K}, got {q1.shape[0]}")

        q1 = q1.view(B, K, T).mean(dim=1)
        q2 = q2.view(B, K, T).mean(dim=1)
        return {"q1": q1, "q2": q2}

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        # next_out provides quantile values on some tau set (not stored); simplest:
        # treat next_out["q*"] as samples from Z(s',a') and build target quantiles by Bellman backup.
        z1 = next_out["q1"]  # [B,N]
        z2 = next_out["q2"]
        z = torch.where(z1.mean(dim=1, keepdim=True) <= z2.mean(dim=1, keepdim=True), z1, z2)

        target = reward_chunk.unsqueeze(1) + (1.0 - done).unsqueeze(1) * gamma_H * z  # [B,N]
        return TDTarget(dist=target)

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        assert target.dist is not None
        y = target.dist  # [B,M] where M = n_tau_target-ish (here equals N from reduce)
        taus = out["taus"]  # [B,N]
        loss1 = quantile_huber_loss(out["q1"], y, taus, kappa=self.config.kappa)
        loss2 = quantile_huber_loss(out["q2"], y, taus, kappa=self.config.kappa)
        return loss1 + loss2

    def soft_update_from(self, src: "IQNTwinQHead", tau: float) -> None:
        for p, sp in zip(self.parameters(), src.parameters()):
            p.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    @staticmethod
    def _sample_taus(B: int, N: int, device: torch.device) -> Tensor:
        return torch.rand((B, N), device=device)
