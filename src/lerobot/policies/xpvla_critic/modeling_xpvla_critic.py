import copy
from typing import Optional, Any

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xpvla_critic.configuration_xpvla_critic import XPVLACriticConfig
from lerobot.policies.xpvla_critic.heads.factory import make_critic_head
from lerobot.policies.xpvla_critic.heads.value_flows import ValueFlowsTwinQHead
from lerobot.policies.xpvla_critic.nets import CrossAttentionBlock, FiLM
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.soft_transformer import timestep_embedding, DomainAwareLinear
from lerobot.utils.constants import ACTION, REWARD, DONE, OBS_LANGUAGE_TOKENS, OBS_STATE, OBS_IMAGES


class XPVLACritic(PreTrainedPolicy):

    config_class = XPVLACriticConfig
    name = "xpvla_critic"

    def __init__(self, config: XPVLACriticConfig, **kwargs):
        super().__init__(config)

        self.gamma = float(config.gamma)
        self.H = int(config.chunk_size)
        self.tau = float(config.tau)

        self.backbone = CriticBackbone(config)
        self.head = make_critic_head(feat_dim=self.backbone.out_dim, config=config.head)

        self.target_backbone = copy.deepcopy(self.backbone).eval()
        self.target_head = copy.deepcopy(self.head).eval()

        for p in self.target_backbone.parameters():
            p.requires_grad_(False)
        for p in self.target_head.parameters():
            p.requires_grad_(False)

    def forward(self, batch):
        return self.td_loss(batch)

    def q(self, batch: dict[str, Any], *, use_target: bool = False) -> Tensor:
        """Compute the expected scalar Q(s, a_chunk) from the head output.

        For distributional heads, returns the expectation; for scalar heads, returns the scalar Q.
        If use_target=True, uses the target backbone/head.
        """
        out = self.q_out(batch, use_target=use_target)
        hd = self.target_head if use_target else self.head
        return hd.expectation(out)

    def q_out(self, batch: dict[str, Any], *, use_target: bool = False) -> dict[str, Tensor]:
        """Compute raw, possibly distributional head outputs for Q(s, a_chunk).

        Encodes (state, action_chunk, optional language) and runs the selected head.
        If use_target=True, uses the target backbone/head.
        """
        feat = self.encode_sa(batch, use_target=use_target)
        hd = self.target_head if use_target else self.head
        return hd(feat)

    def v(self, batch: dict[str, Any], *, next_state: bool = False, use_target: bool = False) -> Tensor:
        """Estimate V(s) (or V(s')) by averaging Q over cached policy action samples.

        Uses cached policy action chunks [B,K,H,A], evaluates Q for each sample, and returns
        mean_k Q(s, a_k) as [B]. Set next_state=True to evaluate V(s').
        """
        actions = self._get_cached_policy_actions(batch, next_state=next_state)  # [B,K,H,A]
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
    def advantage(self, batch: dict[str, Any]) -> Tensor:
        """Compute advantage labels for CFGRL-style extraction.

        Returns (advantage, label) where advantage = Q(s,a_chunk) - V(s) and label is a boolean
        thresholded by margin.
        """
        q_sa = self.q(batch, use_target=False)
        v_s = self.v(batch, next_state=False, use_target=False)
        adv = q_sa - v_s
        return adv

    def encode_sa(
        self,
        batch: dict[str, Any],
        *,
        state_key: str = "state",
        action_key: str = ACTION,
        use_target: bool = False,
    ) -> Tensor:
        """Encode a (state, action_chunk) pair into a fixed-size feature vector.

        Expects batch[state_key] to be a nested dict (potentially containing cached VLM outputs)
        and batch[action_key] to be an action chunk [B,H,A]. Optionally consumes language tokens.
        """
        bb = self.target_backbone if use_target else self.backbone
        state = batch[state_key]                 # state already contains cached forward_vlm dict
        action = batch[action_key]               # [B,H,A]
        tokens = batch.get(OBS_LANGUAGE_TOKENS, None)
        return bb(state=state, action=action, language_tokens=tokens)

    @torch.no_grad()
    def _next_state_policy_out(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Compute target-head outputs on next states under cached policy action samples.

        Evaluates the target critic on K sampled next-state action chunks and reduces them
        (using the head’s reducer) into an aggregated representation suitable for TD targets.
        """
        actions = self._get_cached_policy_actions(batch, next_state=True)  # [B,K,H,A]
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
        """Compute a TD-style critic loss for chunked returns.

        Accumulates the chunk reward R, forms the discount gamma^H, and delegates target-building
        and loss computation to the head. For ValueFlows, calls the specialized loss routine.
        Returns (loss, logging_dict).
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
    def soft_update_target(self) -> None:
        """Polyak-update the target critic parameters.

        Performs an in-place exponential moving average update of the target backbone and delegates
        the head update to target_head.soft_update_from(...).
        """
        tau = self.tau
        for p, tp in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        self.target_head.soft_update_from(self.head, tau=tau)

    def get_optim_params(self) -> dict:
        return {
            "params": [p for p in self.parameters() if p.requires_grad]
        }

    def _get_cached_policy_actions(self, batch: dict[str, Any], *, next_state: bool = False) -> Tensor:
        """Fetch cached policy action chunks from the nested state dict.

        Reads config.backbone.policy_actions_key from state or next_state and validates
        the shape is [B,K,H,A], where H is the configured chunk horizon.
        """
        s = batch["next_state"] if next_state else batch["state"]
        a_pi = s[self.config.backbone.policy_actions_key]
        if a_pi.ndim != 4 or a_pi.shape[2] != self.H:
            raise ValueError(f"Expected {self.config.backbone.policy_actions_key} to be [B,K,H={self.H},A], got {tuple(a_pi.shape)}")
        return a_pi

    def _accumulate_chunk_reward(self, reward: Tensor) -> Tensor:
        """Compute the discounted return over a chunk."""
        if reward.ndim == 1:
            return reward
        if reward.ndim != 2 or reward.shape[1] != self.H:
            raise ValueError(f"Expected reward [B,{self.H}] or [B], got {tuple(reward.shape)}")
        gammas = (self.gamma ** torch.arange(self.H, device=reward.device, dtype=reward.dtype)).view(1, self.H)
        return (reward * gammas).sum(dim=1)

    @staticmethod
    def _repeat_tree(x: Any, K: int) -> Any:
        """Repeat a nested pytree along the batch dimension."""
        if torch.is_tensor(x):
            return x.repeat_interleave(K, dim=0)
        if isinstance(x, dict):
            return {k: XPVLACritic._repeat_tree(v, K) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [XPVLACritic._repeat_tree(v, K) for v in x]
            return type(x)(t)
        return x

    # --- API ---
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return self._get_cached_policy_actions(batch)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]

    def reset(self):
        pass


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

    def __init__(self, config: XPVLACriticConfig) -> None:
        super().__init__()
        self.config = config

        # load xvla checkpoint
        self.xvla = XVLAPolicy.from_pretrained(config.backbone.pretrained_path, config=config.backbone)

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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def out_dim(self) -> int:
        return self.config.hidden_dim

    def forward(
        self,
        *,
        state: dict[str, Any],
        action: Tensor,  # [B,H,A]
        language_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode (state, action_chunk) into a fused feature vector.

        Inputs:
        - state: nested dict containing at least proprio (OBS_STATE) and image tensors (OBS_IMAGES*),
          and optionally a cached vlm_cache_key dict equal to XVLA.forward_vlm output.
        - action: action chunk [B,H,A]
        - language_tokens: optional token ids [B,L]

        Behavior:

        Uses cached VLM outputs if present; otherwise runs XVLA.forward_vlm.
        Projects VLM (and optional auxiliary visual) tokens into hidden space.
        Encodes action tokens via XVLA action encoder at near-zero timestep.
        Fuses state/action tokens using the configured fusion strategy and pools to [B,hidden].
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
        enc = state.get(self.config.vlm_cache_key, None)
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

    def encode_action_tokens(
        self,
        action_chunk: Tensor,     # [B, H, A]
        proprio: Tensor,          # [B, P]
    ) -> Tensor:
        """Encode an action chunk into token embeddings using XVLA’s action encoder.

        Builds per-step tokens by concatenating (action, proprio, time-embedding) and runs the
        cloned action encoder with a fixed domain id. Returns [B,H,hidden].
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

    def _apply_freezing(self) -> None:
        """Sets requires_grad=False for selected components."""
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

    def _domain_id(self, batch_size: int, device: torch.device) -> Tensor:
        """Create a fixed domain-id tensor for domain-aware XVLA layers."""
        return torch.full(
            (batch_size,),
            int(self.config.fixed_domain_id),
            dtype=torch.long,
            device=device,
        )

    def _pool(self, tokens: Tensor) -> Tensor:
        """Pool a token sequence to a single vector.

        Supported modes: mean over sequence, first token, or last token.
        """
        if self.config.pool == "mean":
            return tokens.mean(dim=1)
        elif self.config.pool == "first":
            return tokens[:, 0, ...]
        elif self.config.pool == "last":
            return tokens[:, -1, ...]
        raise ValueError(f"Unknown pool='{self.config.pool}'")
