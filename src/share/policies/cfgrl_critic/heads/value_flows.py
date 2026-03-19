"""Continuous return-flow critic head for richer CFGRL value learning."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from share.policies.cfgrl_critic.configuration_cfgrl_critic import ValueFlowsHeadConfig
from .factory import CriticHead, TDTarget


class _ReturnVectorField(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = feat_dim + 2
        for idx in range(num_layers - 1):
            layers += [nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim if num_layers > 1 else in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_return: Tensor, time: Tensor, feat: Tensor) -> Tensor:
        return self.net(torch.cat([feat, noisy_return, time], dim=-1))


class ValueFlowsTwinQHead(CriticHead):
    """Twin value-flow head implementing the continuous return-flow objective."""

    def __init__(self, feat_dim: int, config: ValueFlowsHeadConfig):
        super().__init__()
        self.cfg = config
        self.flow1 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)
        self.flow2 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "q1": self._estimate_q(feat, which=1),
            "q2": self._estimate_q(feat, which=2),
        }

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        return torch.minimum(out["q1"], out["q2"])

    def build_target(self, *args, **kwargs) -> TDTarget:
        raise NotImplementedError("ValueFlows uses loss_from_batch().")

    def loss(self, out: Dict[str, Tensor], target: TDTarget) -> Tensor:
        raise NotImplementedError("ValueFlows uses loss_from_batch().")

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        return {"q1": out["q1"].view(B, K).mean(dim=1), "q2": out["q2"].view(B, K).mean(dim=1)}

    def soft_update_from(self, src: "ValueFlowsTwinQHead", tau: float) -> None:
        for param, src_param in zip(self.parameters(), src.parameters(), strict=True):
            param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)

    def loss_from_batch(
        self,
        *,
        batch: dict[str, Any],
        critic: Any,
        reward_chunk: Tensor,
        done: Tensor,
        gamma_H: float,
        action_provider: Any,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.cfg
        batch_size = reward_chunk.shape[0]
        device = reward_chunk.device

        feat_sa = critic.encode_state_action(batch, state_key="state", action_key="action", use_target=False)
        next_actions = action_provider.sample_next_actions(batch, critic.config.num_action_samples)
        num_samples = next_actions.shape[1]
        feat_next_flat = self._encode_next_features_flat(critic, batch, next_actions, use_target=True)

        masks = (1.0 - done).view(batch_size, 1)
        rewards = reward_chunk.view(batch_size, 1)

        with torch.no_grad():
            ret_noise = torch.randn(batch_size, 1, device=device)
        ret_stds1 = self._ret_std_from_jac(ret_noise, feat_sa, which=1, use_target=True, critic=critic).squeeze(-1)
        ret_stds2 = self._ret_std_from_jac(ret_noise, feat_sa, which=2, use_target=True, critic=critic).squeeze(-1)
        ret_stds = torch.minimum(ret_stds1, ret_stds2) if cfg.q_agg == "min" else 0.5 * (ret_stds1 + ret_stds2)
        weights = torch.sigmoid(-cfg.confidence_weight_temp / (ret_stds + 1e-8)) + 0.5
        weights = weights.detach()

        noise = torch.randn(batch_size, 1, device=device)
        time = torch.rand(batch_size, 1, device=device)

        next_returns = self._mix_over_actions_flow_returns(
            critic=critic,
            noises=noise,
            feat_next_flat=feat_next_flat,
            B=batch_size,
            K=num_samples,
            end_times=None,
            use_target=True,
        )
        next_returns = (
            torch.minimum(next_returns["r1"], next_returns["r2"])
            if cfg.ret_agg == "min"
            else 0.5 * (next_returns["r1"] + next_returns["r2"])
        )
        returns = rewards + gamma_H * masks * next_returns

        noisy_returns = time * returns + (1.0 - time) * noise
        target_field = returns - noise
        vf1 = self._flow(which=1, use_target=False, critic=critic)(noisy_returns, time, feat_sa)
        vf2 = self._flow(which=2, use_target=False, critic=critic)(noisy_returns, time, feat_sa)
        bcfm_loss = ((vf1 - target_field) ** 2 + (vf2 - target_field) ** 2).mean(dim=-1)

        noisy_next = self._mix_over_actions_flow_returns(
            critic=critic,
            noises=noise,
            feat_next_flat=feat_next_flat,
            B=batch_size,
            K=num_samples,
            end_times=time,
            use_target=True,
        )
        noisy_next_returns = (
            torch.minimum(noisy_next["r1"], noisy_next["r2"])
            if cfg.ret_agg == "min"
            else 0.5 * (noisy_next["r1"] + noisy_next["r2"])
        )
        noisy_returns_dcfm = rewards + gamma_H * masks * noisy_next_returns
        vf1_d = self._flow(which=1, use_target=False, critic=critic)(noisy_returns_dcfm, time, feat_sa)
        vf2_d = self._flow(which=2, use_target=False, critic=critic)(noisy_returns_dcfm, time, feat_sa)

        target_vf_next = self._mix_over_actions_vector_field(
            critic=critic,
            noisy_next_returns=noisy_next_returns,
            times=time,
            feat_next_flat=feat_next_flat,
            B=batch_size,
            K=num_samples,
            use_target=True,
        )
        target_vf = (
            torch.minimum(target_vf_next["vf1"], target_vf_next["vf2"])
            if cfg.ret_agg == "min"
            else 0.5 * (target_vf_next["vf1"] + target_vf_next["vf2"])
        )
        dcfm_loss = ((vf1_d - target_vf) ** 2 + (vf2_d - target_vf) ** 2).mean(dim=-1)

        per_sample = cfg.bcfm_lambda * bcfm_loss + cfg.dcfm_lambda * dcfm_loss
        loss = (weights * per_sample).mean()

        with torch.no_grad():
            q_noise = torch.randn(batch_size, 1, device=device)
        q1 = (
            q_noise + self._flow(which=1, use_target=False, critic=critic)(q_noise, torch.zeros_like(q_noise), feat_sa)
        ).squeeze(-1)
        q2 = (
            q_noise + self._flow(which=2, use_target=False, critic=critic)(q_noise, torch.zeros_like(q_noise), feat_sa)
        ).squeeze(-1)
        if cfg.clip_flow_returns:
            q1 = q1.clamp(cfg.v_min, cfg.v_max)
            q2 = q2.clamp(cfg.v_min, cfg.v_max)
        q = torch.minimum(q1, q2) if cfg.q_agg == "min" else 0.5 * (q1 + q2)

        logs = {
            "critic/loss": loss.detach(),
            "critic/bcfm_loss": bcfm_loss.mean().detach(),
            "critic/dcfm_loss": dcfm_loss.mean().detach(),
            "critic/q_mean": q.mean().detach(),
            "critic/q_std_mean": ret_stds.mean().detach(),
            "critic/weight_mean": weights.mean().detach(),
        }
        return loss, logs

    def _flow(self, *, which: int, use_target: bool, critic: Any) -> _ReturnVectorField:
        head = critic.target_head if use_target else critic.head
        return head.flow1 if which == 1 else head.flow2

    def _estimate_q(self, feat: Tensor, *, which: int) -> Tensor:
        batch_size = feat.shape[0]
        num_samples = int(self.cfg.q_num_samples)
        noise = torch.randn(batch_size * num_samples, 1, device=feat.device)
        feat_rep = feat.repeat_interleave(num_samples, dim=0)
        ret = self._compute_flow_returns(
            noises=noise,
            feat=feat_rep,
            which=which,
            init_times=None,
            end_times=None,
            return_jac_eps_prod=False,
            use_target=False,
            critic=None,
        )
        ret = ret.view(batch_size, num_samples).mean(dim=1)
        if self.cfg.clip_flow_returns:
            ret = ret.clamp(self.cfg.v_min, self.cfg.v_max)
        return ret

    def _compute_flow_returns(
        self,
        *,
        noises: Tensor,
        feat: Tensor,
        which: int,
        init_times: Optional[Tensor],
        end_times: Optional[Tensor],
        return_jac_eps_prod: bool,
        use_target: bool,
        critic: Optional[Any],
    ) -> Tensor | Tuple[Tensor, Tensor]:
        cfg = self.cfg
        device = noises.device
        num = noises.shape[0]
        if init_times is None:
            init_times = torch.zeros(num, 1, device=device, dtype=noises.dtype)
        if end_times is None:
            end_times = torch.ones(num, 1, device=device, dtype=noises.dtype)

        step = (end_times - init_times) / float(cfg.num_flow_steps)
        net = self._flow(which=which, use_target=use_target, critic=critic) if critic is not None else (self.flow1 if which == 1 else self.flow2)
        noisy = noises
        if return_jac_eps_prod:
            noisy = noisy.clone().requires_grad_(True)
            jac = torch.ones_like(noisy)

        for idx in range(cfg.num_flow_steps):
            time = init_times + step * float(idx)
            vf = net(noisy, time, feat)
            noisy_new = noisy + step * vf
            if return_jac_eps_prod:
                grad = torch.autograd.grad(vf.sum(), noisy, create_graph=True)[0]
                jac = jac + step * grad * jac
                noisy = noisy_new.detach().requires_grad_(True)
            else:
                noisy = noisy_new
            if cfg.clip_flow_returns:
                noisy = noisy.clamp(cfg.v_min, cfg.v_max)

        if return_jac_eps_prod:
            return noisy, jac
        return noisy

    def _ret_std_from_jac(
        self,
        noises: Tensor,
        feat_sa: Tensor,
        *,
        which: int,
        use_target: bool,
        critic: Any,
    ) -> Tensor:
        with torch.enable_grad():
            _, jac = self._compute_flow_returns(
                noises=noises,
                feat=feat_sa,
                which=which,
                init_times=None,
                end_times=None,
                return_jac_eps_prod=True,
                use_target=use_target,
                critic=critic,
            )
            return jac.abs().detach()

    def _encode_next_features_flat(
        self,
        critic: Any,
        batch: dict[str, Any],
        next_actions: Tensor,
        *,
        use_target: bool,
    ) -> Tensor:
        batch_size, num_samples, _, _ = next_actions.shape
        rep_state = critic._repeat_tree(batch["next_state"], num_samples)
        flat_actions = next_actions.reshape(batch_size * num_samples, *next_actions.shape[2:])
        backbone = critic.target_backbone if use_target else critic.backbone
        return backbone(state=rep_state, action=flat_actions)

    def _mix_over_actions_flow_returns(
        self,
        *,
        critic: Any,
        noises: Tensor,
        feat_next_flat: Tensor,
        B: int,
        K: int,
        end_times: Optional[Tensor],
        use_target: bool,
    ) -> Dict[str, Tensor]:
        noises_rep = noises.repeat_interleave(K, dim=0)
        end_rep = end_times.repeat_interleave(K, dim=0) if end_times is not None else None
        r1 = self._compute_flow_returns(
            noises=noises_rep,
            feat=feat_next_flat,
            which=1,
            init_times=None,
            end_times=end_rep,
            return_jac_eps_prod=False,
            use_target=use_target,
            critic=critic,
        ).view(B, K, 1).mean(dim=1)
        r2 = self._compute_flow_returns(
            noises=noises_rep,
            feat=feat_next_flat,
            which=2,
            init_times=None,
            end_times=end_rep,
            return_jac_eps_prod=False,
            use_target=use_target,
            critic=critic,
        ).view(B, K, 1).mean(dim=1)
        return {"r1": r1, "r2": r2}

    def _mix_over_actions_vector_field(
        self,
        *,
        critic: Any,
        noisy_next_returns: Tensor,
        times: Tensor,
        feat_next_flat: Tensor,
        B: int,
        K: int,
        use_target: bool,
    ) -> Dict[str, Tensor]:
        ret_rep = noisy_next_returns.repeat_interleave(K, dim=0)
        time_rep = times.repeat_interleave(K, dim=0)
        vf1 = self._flow(which=1, use_target=use_target, critic=critic)(ret_rep, time_rep, feat_next_flat).view(B, K, 1).mean(dim=1)
        vf2 = self._flow(which=2, use_target=use_target, critic=critic)(ret_rep, time_rep, feat_next_flat).view(B, K, 1).mean(dim=1)
        return {"vf1": vf1, "vf2": vf2}
