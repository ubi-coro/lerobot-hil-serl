from typing import Dict

import torch
import torch.nn.functional as F

from lerobot.policies.xpvla_critic.configuration_xpvla_critic import ScalarHeadConfig
from lerobot.policies.xpvla_critic.heads.factory import CriticHead, TDTarget
from lerobot.policies.xpvla_critic.nets import MLP

Tensor = torch.Tensor


class ScalarTwinQHead(CriticHead):
    def __init__(self, feat_dim: int, config: ScalarHeadConfig):
        super().__init__()
        self.q1 = MLP(feat_dim, config.hidden_dim, 1)
        self.q2 = MLP(feat_dim, config.hidden_dim, 1)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "q1": self.q1(feat).squeeze(-1),
            "q2": self.q2(feat).squeeze(-1),
        }

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        return torch.minimum(out["q1"], out["q2"])

    def build_target(self, *, reward_chunk: Tensor, done: Tensor, gamma_H: float, next_out: Dict[str, Tensor]) -> TDTarget:
        v_next = torch.minimum(next_out["q1"], next_out["q2"])
        y = reward_chunk + (1.0 - done) * gamma_H * v_next
        return TDTarget(scalar=y)

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        assert target.scalar is not None
        y = target.scalar
        return F.mse_loss(out["q1"], y) + F.mse_loss(out["q2"], y)

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        q1 = out["q1"].view(B, K).mean(dim=1)
        q2 = out["q2"].view(B, K).mean(dim=1)
        return {"q1": q1, "q2": q2}

    def soft_update_from(self, src: "ScalarTwinQHead", tau: float) -> None:
        for p, sp in zip(self.parameters(), src.parameters()):
            p.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
