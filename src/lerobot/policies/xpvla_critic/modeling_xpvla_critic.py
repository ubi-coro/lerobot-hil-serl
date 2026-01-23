import copy
from typing import Optional, Any

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xpvla_critic.configuration_xpvla_critic import XPVLACriticConfig, XPVLACriticBackboneConfig
from lerobot.policies.xpvla_critic.heads.factory import make_critic_head
from lerobot.policies.xpvla_critic.heads.value_flows import ValueFlowsTwinQHead
from lerobot.policies.xpvla_critic.nets import CrossAttentionBlock, FiLM
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.soft_transformer import timestep_embedding, DomainAwareLinear
from lerobot.utils.constants import ACTION, REWARD, DONE, OBS_LANGUAGE_TOKENS, OBS_STATE, OBS_IMAGES


@PreTrainedPolicy.register_subclass("xpvla_critic")
class XPVLACritic(PreTrainedPolicy):

    config_class = XPVLACriticConfig
    name = "xpvla_critic"

    def __init__(self, config: XPVLACriticConfig):
        super().__init__(config)

        self.gamma = float(config.gamma)
        self.H = int(config.chunk_size)
        self.tau = float(config.tau)

        self.backbone = CriticBackbone(config.backbone)
        self.head = make_critic_head(config.head)

        self.target_backbone = copy.deepcopy(self.backbone).eval()
        self.target_head = copy.deepcopy(self.head).eval()

        for p in self.target_backbone.parameters():
            p.requires_grad_(False)
        for p in self.target_head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def soft_update_target(self) -> None:
        tau = self.tau
        for p, tp in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        self.target_head.soft_update_from(self.head, tau=tau)

    def _accumulate_chunk_reward(self, reward: Tensor) -> Tensor:
        if reward.ndim == 1:
            return reward
        if reward.ndim != 2 or reward.shape[1] != self.H:
            raise ValueError(f"Expected reward [B,{self.H}] or [B], got {tuple(reward.shape)}")
        gammas = (self.gamma ** torch.arange(self.H, device=reward.device, dtype=reward.dtype)).view(1, self.H)
        return (reward * gammas).sum(dim=1)

    def _get_cached_vlm_features(self, batch: dict[str, Any], *, next_state: bool) -> Optional[Tensor]:
        key = self.config.next_vlm_features_key if next_state else self.config.vlm_features_key
        return batch.get(key, None)

    def encode_sa(
        self,
        batch: dict[str, Any],
        *,
        state_key: str = "state",
        action_key: str = ACTION,
        use_target: bool = False,
    ) -> Tensor:
        bb = self.target_backbone if use_target else self.backbone
        state = batch[state_key]
        action = batch[action_key]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)

        # caching hook
        vlm_features = self._get_cached_vlm_features(batch, next_state=(state_key == "next_state"))

        feat = bb(state=state, action=action, language_tokens=tokens, vlm_features=vlm_features)
        return feat

    def q_out(self, batch: dict[str, Any], *, use_target: bool = False) -> dict[str, Tensor]:
        feat = self.encode_sa(batch, use_target=use_target)
        hd = self.target_head if use_target else self.head
        return hd(feat)

    def q(self, batch: dict[str, Any], *, use_target: bool = False) -> Tensor:
        out = self.q_out(batch, use_target=use_target)
        hd = self.target_head if use_target else self.head
        return hd.expectation(out)

    def _get_policy_actions(self, batch: dict[str, Any], *, next_state: bool) -> Tensor:
        key = self.config.next_policy_actions_key if next_state else self.config.policy_actions_key
        a = batch[key]
        if a.ndim != 4:
            raise ValueError(f"Expected {key} [B,K,H,A], got {tuple(a.shape)}")
        if a.shape[2] != self.H:
            raise ValueError(f"Expected H={self.H}, got {a.shape[2]}")
        return a

    def encode_sa(
        self,
        batch: dict[str, Any],
        *,
        state_key: str = "state",
        action_key: str = ACTION,
        use_target: bool = False,
    ) -> Tensor:
        bb = self.target_backbone if use_target else self.backbone
        state = batch[state_key]                 # state already contains cached forward_vlm dict
        action = batch[action_key]               # [B,H,A]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)
        return bb(state=state, action=action, language_tokens=tokens)

    def v(self, batch: dict[str, Any], *, next_state: bool = False, use_target: bool = False) -> Tensor:
        actions = self._get_policy_actions(batch, next_state=next_state)  # [B,K,H,A]
        B, K, H, A = actions.shape

        skey = "next_state" if next_state else "state"
        state = batch[skey]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)

        rep_state = self._repeat_tree(state, K)  # repeats proprio, images, AND cached forward_vlm dict tensors
        rep_tokens = tokens.repeat_interleave(K, dim=0) if torch.is_tensor(tokens) else tokens
        flat_actions = actions.reshape(B * K, H, A)

        bb = self.target_backbone if use_target else self.backbone
        hd = self.target_head if use_target else self.head

        feat = bb(state=rep_state, action=flat_actions, language_tokens=rep_tokens)
        out = hd(feat)
        q = hd.expectation(out)  # [B*K] for scalar and for "expected value" of distributional heads
        return q.view(B, K).mean(dim=1)

    @torch.no_grad()
    def _next_state_policy_out(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        actions = self._get_policy_actions(batch, next_state=True)  # [B,K,H,A]
        B, K, H, A = actions.shape

        state = batch["next_state"]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)

        rep_state = self._repeat_tree(state, K)
        rep_tokens = tokens.repeat_interleave(K, dim=0) if torch.is_tensor(tokens) else tokens
        flat_actions = actions.reshape(B * K, H, A)

        feat = self.target_backbone(state=rep_state, action=flat_actions, language_tokens=rep_tokens)
        out = self.target_head(feat)  # [B*K,...]
        return self.target_head.reduce_over_action_samples(out, B=B, K=K)

    def td_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Delegates distribution-specific target + loss to the head.
        """
        reward = batch[REWARD]
        done = batch[DONE].float()
        R = self._accumulate_chunk_reward(reward)
        gamma_H = float(self.gamma ** self.H)

        if isinstance(self.head, ValueFlowsTwinQHead):
            loss, logs = self.head.loss_from_batch(
                batch=batch,
                critic=self,
                reward_chunk=R,
                done=done,
                gamma_H=gamma_H,
            )
            return loss, logs

        with torch.no_grad():
            # Build target-network outputs on next state using *dataset next_action* OR policy samples?
            # For policy evaluation TD: we need V(s') under pi_k. We'll use v() by policy samples and
            # then let head build targets appropriately.
            #
            # But distributional heads need a next-state *distribution* under pi_k.
            # We compute next_out by evaluating target critic on each sampled next action and averaging.
            next_out = self._next_state_policy_out(batch)

            target = self.target_head.build_target(
                reward_chunk=R,
                done=done,
                gamma_H=gamma_H,
                next_out=next_out,
            )

        out = self.q_out(batch, use_target=False)
        loss = self.head.loss(out, target)

        logs = {
            "critic/loss": loss.detach(),
            "critic/R_mean": R.mean().detach(),
        }
        # Optional: log scalar expectations for monitoring
        with torch.no_grad():
            logs["critic/q_mean"] = self.head.expectation(out).mean().detach()
        return loss, logs

    @torch.no_grad()
    def _next_state_policy_out(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """
        Compute target-network Q outputs on next_state for policy-sampled actions a_j ~ pi_k(s').
        For distributional heads, we need a representation of the distribution under pi_k.
        We provide "averaged head outputs" in a head-specific compatible form.

        Convention:
          - Scalar: return {'q1': [B], 'q2': [B]} as mean over samples.
          - C51: return {'logits1': [B,N], 'logits2': [B,N]} as log-mean-exp over samples (or mean probs).
          - IQN: return {'q1': [B,T], 'q2': [B,T]} mean over samples for each tau (head handles taus).
        """
        actions = self._get_policy_actions(batch, next_state=True)  # [B,K,H,A]
        B, K, H, A = actions.shape

        state = batch["next_state"]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)
        vlm = self._get_cached_vlm_features(batch, next_state=True)

        rep_state: dict[str, Any] = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                rep_state[k] = v.repeat_interleave(K, dim=0)
            else:
                rep_state[k] = v

        rep_tokens = tokens.repeat_interleave(K, dim=0) if torch.is_tensor(tokens) else tokens
        rep_vlm = vlm.repeat_interleave(K, dim=0) if torch.is_tensor(vlm) else vlm
        flat_actions = actions.reshape(B * K, H, A)

        feat = self.target_backbone(state=rep_state, action=flat_actions, language_tokens=rep_tokens, vlm_features=rep_vlm)
        out = self.target_head(feat)  # head-specific dict on [B*K,...]

        # Let head reduce across K in a consistent way:
        return self.target_head.reduce_over_action_samples(out, B=B, K=K)

    @torch.no_grad()
    def advantage(self, batch: dict[str, Any], *, margin: float = 0.0) -> tuple[Tensor, Tensor]:
        q_sa = self.q(batch, use_target=False)
        v_s = self.v(batch, next_state=False, use_target=False)
        adv = q_sa - v_s
        label = adv > margin
        return adv, label

    @staticmethod
    def _repeat_tree(x: Any, K: int) -> Any:
        if torch.is_tensor(x):
            return x.repeat_interleave(K, dim=0)
        if isinstance(x, dict):
            return {k: XPVLACritic._repeat_tree(v, K) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [XPVLACritic._repeat_tree(v, K) for v in x]
            return type(x)(t)
        return x


class CriticBackbone(nn.Module):
    """
    Critic backbone operating on:
      - state: (text tokens + images) via XVLA.forward_vlm + (vlm_proj, aux_visual_proj)
      - proprio + action chunk: via XVLA.action_encoder with t≈0, domain_id fixed to 0
    And three fusion choices:
      (1) film: pool(action_tokens), pool(state_tokens) -> FiLM + MLP
      (2) cross_attn: action_tokens attend to state_tokens -> pool(action_tokens)
      (3) concat_transformer: concat tokens -> optional reused XVLA blocks -> pool(action segment)
    """

    def __init__(self, config: XPVLACriticBackboneConfig) -> None:
        super().__init__()
        self.config = config

        # load xvla checkpoint
        if config.xvla.pretrained_path is None:
            self.xvla = XVLAPolicy(config.xvla)
        else:
            self.xvla = XVLAPolicy.from_pretrained(config.xvla.pretrained_path, config=config.xvla)

        # clone modules
        self.vlm_proj = copy.deepcopy(self.xvla.model.transformer.vlm_proj)
        self.aux_visual_proj = copy.deepcopy(self.xvla.model.transformer.aux_visual_proj)
        self.action_encoder = copy.deepcopy(self.xvla.model.transformer.action_encoder)
        self.pos_emb = copy.deepcopy(self.xvla.model.transformer.pos_emb)

        self.blocks = nn.ModuleList()
        if self.config.num_reused_xvla_blocks > 0:
            _blocks = self.xvla.model.transformer.blocks
            n_blocks = min([self.config.num_reused_xvla_blocks, len(_blocks)])
            self.blocks = nn.ModuleList([copy.deepcopy(b) for b in _blocks[:n_blocks]])

        # build (state, action) fusion modules
        hidden = config.hidden_dim
        if config.fusion == "film":
            self.film = FiLM(hidden, config.dropout)
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.cross_blocks = None

        elif config.fusion == "cross_attn":
            self.film = None
            self.cross_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        hidden=hidden,
                        n_heads=config.n_heads,
                        mlp_ratio=config.mlp_ratio,
                        dropout=config.dropout,
                    )
                    for _ in range(config.num_cross_attn_blocks)
                ]
            )
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        elif config.fusion == "concat_transformer":
            self.film = None
            self.cross_blocks = None
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

        else:
            raise ValueError(f"Unknown fusion='{config.fusion}'")

        self._apply_freezing()

    def _apply_freezing(self) -> None:
        def _freeze(m: nn.Module) -> None:
            for p in m.parameters():
                p.requires_grad = False

        if self.config.freeze_vlm:
            _freeze(self.xvla.model.vlm)

        if self.config.freeze_vlm_proj:
            _freeze(self.vlm_proj)

        if self.config.freeze_aux_visual_proj:
            _freeze(self.aux_visual_proj)

        if self.config.freeze_action_encoder:
            _freeze(self.action_encoder)

        if self.config.freeze_transformer_blocks:
            _freeze(self.blocks)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def out_dim(self) -> int:
        return self.config.hidden_dim

    def _domain_id(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.full(
            (batch_size,),
            int(self.config.fixed_domain_id),
            dtype=torch.long,
            device=device,
        )

    def encode_action_tokens(
        self,
        action_chunk: Tensor,     # [B, H, A]
        proprio: Tensor,          # [B, P]
    ) -> Tensor:
        """
        Reuse XVLA action_encoder exactly, with t≈0 and domain_id=0.
        Output: [B, H, hidden]
        """
        B, H, _ = action_chunk.shape
        dev = action_chunk.device
        did = self._domain_id(B, dev)

        # Build time tokens like XVLA does
        t = torch.full((B,), float(self.config.action_encoder_timestep), device=dev, dtype=proprio.dtype)
        time_emb = timestep_embedding(t, self.xvla.model.transformer.dim_time)  # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, H, -1)

        proprio_tokens = proprio.unsqueeze(1).expand(B, H, proprio.shape[-1])
        action_tokens = torch.cat([action_chunk, proprio_tokens, time_tokens], dim=-1)

        # DomainAwareLinear expects domain_id, so pass fixed 0
        return self.action_encoder(action_tokens, did)  # type: ignore[misc]

    def forward(
        self,
        *,
        state: dict[str, Any],
        action: Tensor,  # [B,H,A]
        language_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Intended batch structure:

          state: dict containing at least
            - OBS_STATE: [B,P] proprio
            - one or more image tensors under keys starting with OBS_IMAGES,
              e.g. f"{OBS_IMAGES}_cam1", f"{OBS_IMAGES}_cam2", ...
            - optionally: config.vlm_features_key storing raw forward_vlm(...) output dict

          action: [B,H,A] action chunk
          language_tokens: [B, L] token ids (unconditioned), usually batch[OBS_LANGUAGE_TOKENS]

        Returns:
          fused feature vector [B, hidden] for Q/V heads.
        """
        proprio: Tensor = state[OBS_STATE]
        action_chunk: Tensor = action

        B = proprio.shape[0]
        dev = proprio.device
        did = self._domain_id(B, dev)

        # -------------------------
        # 1) Get raw VLM outputs (cached or freshly computed)
        # -------------------------
        # Cached object is EXACTLY what forward_vlm returns (a dict), per your spec.
        enc = state.get(self.config.vlm_features_key, None)
        if enc is None:
            # Build pixel_values from all camera keys in state
            pixel_values = {k: v for k, v in state.items() if isinstance(k, str) and k.startswith(OBS_IMAGES)}
            # image_mask can be provided by your processors; otherwise synthesize a trivial mask
            image_mask = state.get("image_mask", None)
            if image_mask is None:
                # [B, num_cams] all present
                image_mask = torch.ones((B, len(pixel_values)), device=dev, dtype=torch.bool)

            enc = self.xvla.model.forward_vlm(  # type: ignore[attr-defined]
                input_ids=language_tokens,
                pixel_values=pixel_values,
                image_mask=image_mask,
            )

        # enc now is raw forward_vlm output dict
        vlm_features: Tensor = enc["vlm_features"]              # [B, T_vlm, D]
        aux_visual: Optional[Tensor] = enc.get("aux_visual_inputs", None)  # [B, T_aux, D] or None

        # -------------------------
        # 2) Project raw VLM tokens to critic hidden space (like XVLA does)
        # -------------------------
        # vlm_proj and aux_visual_proj were cloned from XVLA transformer
        if isinstance(self.vlm_proj, DomainAwareLinear):
            vlm_h = self.vlm_proj(vlm_features, did)  # type: ignore[misc]
        else:
            vlm_h = self.vlm_proj(vlm_features)

        tokens = [vlm_h]

        if self.config.use_aux_visual_inputs and aux_visual is not None:
            if isinstance(self.aux_visual_proj, DomainAwareLinear):
                aux_h = self.aux_visual_proj(aux_visual, did)  # type: ignore[misc]
            else:
                aux_h = self.aux_visual_proj(aux_visual)
            tokens.append(aux_h)

        state_tokens = torch.cat(tokens, dim=1)  # [B, T_state, hidden]

        # -------------------------
        # 3) Encode action tokens
        # -------------------------
        action_tokens = self.encode_action_tokens(
            action_chunk=action_chunk,
            proprio=proprio,
        )  # [B, H, hidden]

        # -------------------------
        # 4) Fuse (state_tokens, action_tokens) and pool -> [B, hidden]
        # -------------------------
        if self.config.fusion == "film":
            a = self._pool(action_tokens)
            s = self._pool(state_tokens)
            fused = self.film(a, s)  # type: ignore[union-attr]
            return self.post(fused)

        if self.config.fusion == "cross_attn":
            x = action_tokens
            for blk in self.cross_blocks:  # type: ignore[union-attr]
                x = blk(x, state_tokens)
            fused = self._pool(x)
            return self.post(fused)

        # concat_transformer
        x = torch.cat([action_tokens, state_tokens], dim=1)
        if self.config.use_pos_emb and self.pos_emb is not None:
            seq_len = x.shape[1]
            if seq_len > self.pos_emb.shape[1]:
                raise ValueError(f"Critic seq_len={seq_len} exceeds pos_emb limit={self.pos_emb.shape[1]}")
            x = x + self.pos_emb[:, :seq_len, :]

        if len(self.blocks) > 0:
            for blk in self.blocks:
                x = blk(x)

        H = action_tokens.shape[1]
        fused = self._pool(x[:, :H, :])  # pool only action segment
        return self.post(fused)

    def _pool(self, tokens: Tensor) -> Tensor:
        if self.config.pool == "mean":
            return tokens.mean(dim=1)
        elif self.config.pool == "first":
            return tokens[:, 0, ...]
        elif self.config.pool == "last":
            return tokens[:, -1, ...]
        raise ValueError(f"Unknown pool='{self.config.pool}'")

    @staticmethod
    def _is_domain_aware(module: nn.Module) -> bool:
        # DomainAwareLinear in soft_transformer has signature forward(x, domain_id).
        # We detect it conservatively by checking for nn.Embedding attributes used there.
        return hasattr(module, "fc") and hasattr(module, "bias") and isinstance(getattr(module, "fc"), nn.Embedding)


