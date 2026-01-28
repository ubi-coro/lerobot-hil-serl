from typing import Dict

import torch
import torch.nn.functional as F

from lerobot.policies.xpvla_critic.configuration_xpvla_critic import C51HeadConfig
from lerobot.policies.xpvla_critic.heads.factory import CriticHead, TDTarget
from lerobot.policies.xpvla_critic.nets import MLP

Tensor = torch.Tensor


def c51_projection(
    *,
    reward: Tensor,           # [B]
    done: Tensor,             # [B]
    gamma: float,
    atoms: Tensor,            # [N]
    next_probs: Tensor,       # [B, N]
    vmin: float,
    vmax: float,
) -> Tensor:
    """
    Standard C51 projection onto fixed support atoms.
    Returns target probs [B,N].
    """
    B, N = next_probs.shape
    device = next_probs.device
    dz = (vmax - vmin) / (N - 1)

    # Tz = r + (1-d) * gamma * z
    Tz = reward.view(B, 1) + (1.0 - done).view(B, 1) * gamma * atoms.view(1, N)
    Tz = Tz.clamp(vmin, vmax)

    b = (Tz - vmin) / dz
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros((B, N), device=device, dtype=next_probs.dtype)

    # Distribute probability mass
    offset = torch.arange(B, device=device).view(B, 1) * N
    l_idx = (l + offset).view(-1)
    u_idx = (u + offset).view(-1)

    p = next_probs.view(-1)
    b_flat = b.view(-1)
    l_flat = l.view(-1).float()
    u_flat = u.view(-1).float()

    m.view(-1).index_add_(0, l_idx, p * (u_flat - b_flat))
    m.view(-1).index_add_(0, u_idx, p * (b_flat - l_flat))

    return m


class C51TwinQHead(CriticHead):
    def __init__(self, feat_dim: int, config: C51HeadConfig):
        super().__init__()
        self.config = config
        self.logits1 = MLP(feat_dim, config.hidden_dim, config.n_atoms, num_layers=config.num_layers, dropout=config.dropout)
        self.logits2 = MLP(feat_dim, config.hidden_dim, config.n_atoms, num_layers=config.num_layers, dropout=config.dropout)

        atoms = torch.linspace(config.vmin, config.vmax, config.n_atoms)
        self.register_buffer("atoms", atoms, persistent=True)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "logits1": self.logits1(feat),
            "logits2": self.logits2(feat),
        }

    def _probs(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        p1 = self._probs(out["logits1"])
        p2 = self._probs(out["logits2"])
        q1 = (p1 * self.atoms.view(1, -1)).sum(dim=-1)
        q2 = (p2 * self.atoms.view(1, -1)).sum(dim=-1)
        return torch.minimum(q1, q2)

    @torch.no_grad()
    def reduce_over_action_samples(self, out: dict[str, torch.Tensor], dim: int = 1) -> dict[str, torch.Tensor]:
        """
        out["logits1"], out["logits2"]: [B, K, n_atoms]
        Returns probs averaged over K: [B, n_atoms]
        """
        logits1 = out["logits1"]
        logits2 = out["logits2"]

        # average distribution over action samples in probability space
        probs1 = torch.softmax(logits1, dim=-1).mean(dim=dim)
        probs2 = torch.softmax(logits2, dim=-1).mean(dim=dim)

        # small clamp for numerical safety (optional)
        eps = 1e-8
        probs1 = probs1.clamp(min=eps)
        probs2 = probs2.clamp(min=eps)

        return {"probs1": probs1, "probs2": probs2}

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        # next_out comes from reduce_over_action_samples, so we expect probs
        p1 = next_out["probs1"]
        p2 = next_out["probs2"]
        p_next = torch.where(
            (self._exp_from_probs(p1) <= self._exp_from_probs(p2)).unsqueeze(-1),
            p1,
            p2,
        )  # pick “min” distribution by expected value (pragmatic)

        target_probs = c51_projection(
            reward=reward_chunk,
            done=done,
            gamma=gamma_H,
            atoms=self.atoms,
            next_probs=p_next,
            vmin=self.config.vmin,
            vmax=self.config.vmax,
        )
        return TDTarget(dist=target_probs)

    def _exp_from_probs(self, p: Tensor) -> Tensor:
        return (p * self.atoms.view(1, -1)).sum(dim=-1)

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        assert target.dist is not None
        m = target.dist  # [B,N]
        # Cross-entropy between target probs and predicted probs
        logp1 = F.log_softmax(out["logits1"], dim=-1)
        logp2 = F.log_softmax(out["logits2"], dim=-1)
        loss1 = -(m * logp1).sum(dim=-1).mean()
        loss2 = -(m * logp2).sum(dim=-1).mean()
        return loss1 + loss2

    def soft_update_from(self, src: "C51TwinQHead", tau: float) -> None:
        for p, sp in zip(self.parameters(), src.parameters()):
            p.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
