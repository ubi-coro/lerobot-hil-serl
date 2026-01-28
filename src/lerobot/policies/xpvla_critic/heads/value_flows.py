from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.xpvla_critic.heads.factory import CriticHead

Tensor = torch.Tensor


class _ReturnVectorField(nn.Module):
    """
    v_theta(r_t, t, feat_sa) -> dr/dt (vector field) for scalar return r.
    """
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = feat_dim + 2  # feat + (noisy_return, time)
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim if num_layers > 1 else in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_return: Tensor, time: Tensor, feat: Tensor) -> Tensor:
        x = torch.cat([feat, noisy_return, time], dim=-1)
        return self.net(x)  # [B,1]


class ValueFlowsTwinQHead(CriticHead):
    """
    Value Flows distributional critic as a flow-matching return model.

    - Maintains two flow vector fields (critic_flow1/2) like the JAX reference.
    - Computes a scalar Q estimate by sampling eps ~ N(0,1) and integrating returns.
    - Implements critic loss via BCFM + DCFM and confidence weighting, using target flows.
    """
    def __init__(self, feat_dim: int, config: "ValueFlowsHeadConfig"):
        super().__init__()
        self.cfg = config

        self.flow1 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)
        self.flow2 = _ReturnVectorField(feat_dim, config.hidden_dim, config.num_layers, config.dropout)

        # For convenience in your existing code-paths, we expose “q1/q2” scalars as forward output.
        # Those are Monte-Carlo expectations (small sample count), used for advantage labeling etc.
        # Training loss does NOT use build_target/loss; it uses loss_from_batch().
        #
        # Target flows live in the *target head* via deepcopy in XPVLACritic.

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        # Produce q1/q2 by sampling and integrating the flow to t=1.
        q1 = self._estimate_q(feat, which=1)
        q2 = self._estimate_q(feat, which=2)
        return {"q1": q1, "q2": q2}

    def expectation(self, out: Dict[str, Tensor]) -> Tensor:
        # Like other twin critics: min aggregation
        return torch.minimum(out["q1"], out["q2"])

    def build_target(self, *args, **kwargs) -> "TDTarget":
        raise NotImplementedError("ValueFlows uses loss_from_batch() (BCFM+DCFM), not build_target().")

    def loss(self, out: Dict[str, Tensor], target: "TDTarget") -> Tensor:
        raise NotImplementedError("ValueFlows uses loss_from_batch() (BCFM+DCFM), not loss(out,target).")

    @torch.no_grad()
    def reduce_over_action_samples(self, out: Dict[str, Tensor], *, B: int, K: int) -> Dict[str, Tensor]:
        # Here out is {"q1":[B*K], "q2":[B*K]} from forward(feat_next_flat).
        return {
            "q1": out["q1"].view(B, K).mean(dim=1),
            "q2": out["q2"].view(B, K).mean(dim=1),
        }

    def soft_update_from(self, src: "ValueFlowsTwinQHead", tau: float) -> None:
        for p, sp in zip(self.parameters(), src.parameters()):
            p.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    # -------------------------
    # Value Flows critic loss (BCFM + DCFM)
    # -------------------------

    def loss_from_batch(
        self,
        *,
        batch: dict[str, Any],
        critic: Any,
        reward_chunk: Tensor,   # [B]
        done: Tensor,           # [B] float {0,1}
        gamma_H: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Faithful to the JAX reference, with adaptation:
          - next_actions are provided by cached next_policy_actions [B,K,H,A]
          - we treat the policy-induced mixture over actions by averaging over K samples.

        Required:
          - critic.backbone / critic.target_backbone
          - critic._get_cached_policy_actions(...) and cached next_state present
        """
        cfg = self.cfg
        B = reward_chunk.shape[0]
        dev = reward_chunk.device

        # Current (s,a_ds) feature for dataset action
        feat_sa = critic.encode_sa(batch, state_key="state", action_key="action", use_target=False)  # [B,D]

        # Next-state sampled actions a_j ~ pi_k(s')
        next_actions = critic._get_cached_policy_actions(batch, next_state=True)  # [B,K,H,A]
        K = next_actions.shape[1]

        # Build next features for each sampled action: [B*K, D] using TARGET backbone (like ref uses target critics)
        feat_next_flat = self._encode_next_features_flat(critic, batch, next_actions, use_target=True)  # [B*K,D]

        # Convenience: masks
        masks = (1.0 - done).view(B, 1)  # [B,1]
        rewards = reward_chunk.view(B, 1)  # [B,1]

        # 1) Confidence weights via Jacobian product on TARGET flows at (s,a_ds)
        with torch.no_grad():
            ret_noises = torch.randn(B, 1, device=dev)
        ret_stds1 = self._ret_std_from_jac(
            ret_noises, feat_sa, which=1, use_target=True, critic=critic
        ).squeeze(-1)
        ret_stds2 = self._ret_std_from_jac(
            ret_noises, feat_sa, which=2, use_target=True, critic=critic
        ).squeeze(-1)

        if cfg.q_agg == "min":
            ret_stds = torch.minimum(ret_stds1, ret_stds2)
        else:
            ret_stds = 0.5 * (ret_stds1 + ret_stds2)

        weights = torch.sigmoid(-cfg.confidence_weight_temp / (ret_stds + 1e-8)) + 0.5
        weights = weights.detach()  # stop-gradient like ref

        # 2) BCFM loss
        noises = torch.randn(B, 1, device=dev)
        times = torch.rand(B, 1, device=dev)

        # next_returns under TARGET flows, mixed over K actions
        next_returns = self._mix_over_actions_flow_returns(
            critic=critic,
            noises=noises,
            feat_next_flat=feat_next_flat,
            B=B,
            K=K,
            end_times=None,
            use_target=True,
        )  # [B,1]

        if cfg.ret_agg == "min":
            next_returns = torch.minimum(next_returns["r1"], next_returns["r2"])
        else:
            next_returns = 0.5 * (next_returns["r1"] + next_returns["r2"])

        returns = rewards + gamma_H * masks * next_returns  # [B,1]

        noisy_returns = times * returns + (1.0 - times) * noises
        target_vector_field = returns - noises

        vf1 = self._flow(which=1, use_target=False, critic=critic)(noisy_returns, times, feat_sa)
        vf2 = self._flow(which=2, use_target=False, critic=critic)(noisy_returns, times, feat_sa)
        bcfm_loss = ((vf1 - target_vector_field) ** 2 + (vf2 - target_vector_field) ** 2).mean(dim=-1)  # [B]

        # 3) DCFM loss
        noisy_next = self._mix_over_actions_flow_returns(
            critic=critic,
            noises=noises,
            feat_next_flat=feat_next_flat,
            B=B,
            K=K,
            end_times=times,   # integrate only up to t
            use_target=True,
        )  # dict of [B,1]

        if cfg.ret_agg == "min":
            noisy_next_returns = torch.minimum(noisy_next["r1"], noisy_next["r2"])
        else:
            noisy_next_returns = 0.5 * (noisy_next["r1"] + noisy_next["r2"])

        noisy_returns_dcfm = rewards + gamma_H * masks * noisy_next_returns  # [B,1]

        vf1_d = self._flow(which=1, use_target=False, critic=critic)(noisy_returns_dcfm, times, feat_sa)
        vf2_d = self._flow(which=2, use_target=False, critic=critic)(noisy_returns_dcfm, times, feat_sa)

        # target vector field from TARGET flows at (s', a'~pi), evaluated at (noisy_next_returns, t)
        tvf_next = self._mix_over_actions_vector_field(
            critic=critic,
            noisy_next_returns=noisy_next_returns,
            times=times,
            feat_next_flat=feat_next_flat,
            B=B,
            K=K,
            use_target=True,
        )  # dict of [B,1]

        if cfg.ret_agg == "min":
            target_vf = torch.minimum(tvf_next["vf1"], tvf_next["vf2"])
        else:
            target_vf = 0.5 * (tvf_next["vf1"] + tvf_next["vf2"])

        dcfm_loss = ((vf1_d - target_vf) ** 2 + (vf2_d - target_vf) ** 2).mean(dim=-1)  # [B]

        # 4) Combine + weight
        per_sample = cfg.bcfm_lambda * bcfm_loss + cfg.dcfm_lambda * dcfm_loss
        loss = (weights * per_sample).mean()

        # 5) Logging q estimates (as in ref)
        with torch.no_grad():
            q_noises = torch.randn(B, 1, device=dev)
        q1 = (q_noises + self._flow(which=1, use_target=False, critic=critic)(q_noises, torch.zeros_like(q_noises), feat_sa)).squeeze(-1)
        q2 = (q_noises + self._flow(which=2, use_target=False, critic=critic)(q_noises, torch.zeros_like(q_noises), feat_sa)).squeeze(-1)

        if cfg.clip_flow_returns:
            q1 = q1.clamp(cfg.v_min, cfg.v_max)
            q2 = q2.clamp(cfg.v_min, cfg.v_max)

        if cfg.q_agg == "min":
            q = torch.minimum(q1, q2)
        else:
            q = 0.5 * (q1 + q2)

        logs = {
            "critic/loss": loss.detach(),
            "critic/bcfm_loss": bcfm_loss.mean().detach(),
            "critic/dcfm_loss": dcfm_loss.mean().detach(),
            "critic/q_mean": q.mean().detach(),
            "critic/q_std_mean": ret_stds.mean().detach(),
            "critic/weight_mean": weights.mean().detach(),
        }
        return loss, logs

    # -------------------------
    # Helpers
    # -------------------------

    def _flow(self, *, which: int, use_target: bool, critic: Any) -> _ReturnVectorField:
        head = critic.target_head if use_target else critic.head
        assert isinstance(head, ValueFlowsTwinQHead)
        return head.flow1 if which == 1 else head.flow2

    def _estimate_q(self, feat: Tensor, *, which: int) -> Tensor:
        """
        Monte-Carlo estimate E[Return] by integrating flow from eps ~ N(0,1) to t=1.
        Output: [B]
        """
        B = feat.shape[0]
        n = int(self.cfg.q_num_samples)
        dev = feat.device

        eps = torch.randn(B * n, 1, device=dev)
        feat_rep = feat.repeat_interleave(n, dim=0)
        ret = self._compute_flow_returns(
            noises=eps,
            feat=feat_rep,
            which=which,
            init_times=None,
            end_times=None,
            return_jac_eps_prod=False,
            use_target=False,
            critic=None,  # not needed for non-target
        )  # [B*n,1]
        ret = ret.view(B, n).mean(dim=1)  # [B]
        if self.cfg.clip_flow_returns:
            ret = ret.clamp(self.cfg.v_min, self.cfg.v_max)
        return ret

    def _compute_flow_returns(
        self,
        *,
        noises: Tensor,          # [N,1]
        feat: Tensor,            # [N,D]
        which: int,
        init_times: Optional[Tensor],
        end_times: Optional[Tensor],
        return_jac_eps_prod: bool,
        use_target: bool,
        critic: Optional[Any],
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Euler integration, faithful to the reference.
        If return_jac_eps_prod=True, also propagate jac_eps_prod like their jvp computation.
        """
        cfg = self.cfg
        N = noises.shape[0]
        dev = noises.device

        if init_times is None:
            init_times = torch.zeros(N, 1, device=dev, dtype=noises.dtype)
        if end_times is None:
            end_times = torch.ones(N, 1, device=dev, dtype=noises.dtype)

        step = (end_times - init_times) / float(cfg.num_flow_steps)

        net = self._flow(which=which, use_target=use_target, critic=critic) if critic is not None else (self.flow1 if which == 1 else self.flow2)

        noisy = noises
        if return_jac_eps_prod:
            # We need gradients through noisy
            noisy = noisy.clone().requires_grad_(True)
            jac = torch.ones_like(noisy)

        for i in range(cfg.num_flow_steps):
            t = init_times + step * float(i)
            vf = net(noisy, t, feat)  # [N,1]
            noisy_new = noisy + step * vf

            if return_jac_eps_prod:
                # jac := jac + step * (d vf / d noisy) * jac   (scalar case)
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
            ret, jac = self._compute_flow_returns(
                noises=noises,
                feat=feat_sa,
                which=which,
                init_times=None,
                end_times=None,
                return_jac_eps_prod=True,
                use_target=use_target,
                critic=critic,
            )
            # reference uses sqrt(jac^2) == abs(jac)
            return jac.abs().detach()

    def _encode_next_features_flat(
        self,
        critic: Any,
        batch: dict[str, Any],
        next_actions: Tensor,     # [B,K,H,A]
        *,
        use_target: bool,
    ) -> Tensor:
        """
        Compute backbone features for (next_state, next_actions_j) for all j, flattened to [B*K, D].
        Mirrors your v() pattern but returns features, not Q values.
        """
        B, K, H, A = next_actions.shape
        state = batch["next_state"]
        tokens = batch.get("language_tokens", batch.get("obs_language_tokens", None))
        vlm = critic._get_cached_vlm_features(batch, next_state=True)

        rep_state = {k: (v.repeat_interleave(K, dim=0) if torch.is_tensor(v) else v) for k, v in state.items()}
        rep_tokens = tokens.repeat_interleave(K, dim=0) if torch.is_tensor(tokens) else tokens

        # NOTE: cached VLM enc is a dict; repeat_interleave on it is not defined.
        # In your spec, cached VLM output is exactly forward_vlm’s dict; you should cache per-sample,
        # and in batching it should already be aligned to B*K if you want to use it here.
        # If you want to keep caching as [B,...] dict, the simplest is: do NOT repeat it here,
        # and let backbone recompute for next_state in ValueFlows training; otherwise store
        # per-action-sample cache. Most setups cache per-state only, so recompute here is fine.
        #
        # If your backbone already supports state[self.config.vlm_cache_key] at [B,...],
        # you can omit passing it and rely on recompute for this path.
        _ = vlm  # intentionally unused in this helper to avoid dict repetition semantics

        flat_actions = next_actions.reshape(B * K, H, A)

        bb = critic.target_backbone if use_target else critic.backbone
        feat_next = bb(state=rep_state, action=flat_actions, language_tokens=rep_tokens)  # [B*K, D]
        return feat_next

    def _mix_over_actions_flow_returns(
        self,
        *,
        critic: Any,
        noises: Tensor,           # [B,1]
        feat_next_flat: Tensor,   # [B*K,D]
        B: int,
        K: int,
        end_times: Optional[Tensor],
        use_target: bool,
    ) -> Dict[str, Tensor]:
        """
        Compute return samples under flow1/flow2 for each action-sample, then average over K.
        """
        noises_rep = noises.repeat_interleave(K, dim=0)  # [B*K,1]
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
        ).view(B, K, 1).mean(dim=1)  # [B,1]

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
        noisy_next_returns: Tensor,  # [B,1]
        times: Tensor,              # [B,1]
        feat_next_flat: Tensor,     # [B*K,D]
        B: int,
        K: int,
        use_target: bool,
    ) -> Dict[str, Tensor]:
        """
        Evaluate target vector fields at (noisy_next_returns, times, next_feat) and average over K actions.
        """
        r_rep = noisy_next_returns.repeat_interleave(K, dim=0)  # [B*K,1]
        t_rep = times.repeat_interleave(K, dim=0)

        vf1 = self._flow(which=1, use_target=use_target, critic=critic)(r_rep, t_rep, feat_next_flat).view(B, K, 1).mean(dim=1)
        vf2 = self._flow(which=2, use_target=use_target, critic=critic)(r_rep, t_rep, feat_next_flat).view(B, K, 1).mean(dim=1)
        return {"vf1": vf1, "vf2": vf2}
