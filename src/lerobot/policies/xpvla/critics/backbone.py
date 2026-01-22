# lerobot/policies/xpvla/critics/backbone.py

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from lerobot.policies.xvla.modeling_xvla import XVLAModel
from lerobot.policies.xvla.soft_transformer import TransformerBlock, timestep_embedding

from ..configuration_xpvla import CriticBackboneConfig


@dataclass
class CriticBackboneBundle:
    """
    A lightweight “parts kit” cloned from XVLA to reuse pretrained representations.
    These modules are owned by the critic (deep-copied), so critic training will not
    mutate the policy’s XVLA.
    """
    # A full XVLAModel clone is the simplest way to reuse forward_vlm unchanged.
    # (You indicated you may prefer this.)
    xvla_model: XVLAModel

    # Reused submodules from the XVLA transformer (SoftPromptedTransformer)
    vlm_proj: nn.Module
    aux_visual_proj: nn.Module
    action_encoder: nn.Module
    pos_emb: Optional[nn.Parameter]
    reused_blocks: nn.ModuleList  # may be empty


class _CrossAttentionBlock(nn.Module):
    """
    Pre-LN cross-attention + FFN block, batch_first.
    Queries come from action tokens; keys/values from state tokens.
    """

    def __init__(self, hidden: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden * mlp_ratio), hidden),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        q_tokens = q_tokens + self.drop(out)
        q_tokens = q_tokens + self.ff(self.norm_ff(q_tokens))
        return q_tokens


class _FiLM(nn.Module):
    """
    FiLM modulation of a feature vector x by conditioning vector c:
        y = LN(x) * (1 + gamma(c)) + beta(c)
    """
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.to_gamma = nn.Sequential(nn.Linear(hidden, hidden), nn.Dropout(dropout))
        self.to_beta = nn.Sequential(nn.Linear(hidden, hidden), nn.Dropout(dropout))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x_n = self.norm(x)
        gamma = self.to_gamma(c)
        beta = self.to_beta(c)
        return x_n * (1.0 + gamma) + beta


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

    def __init__(self, cfg: CriticBackboneConfig, bundle: CriticBackboneBundle) -> None:
        super().__init__()
        self.cfg = cfg

        self.xvla_model = bundle.xvla_model
        self.vlm_proj = bundle.vlm_proj
        self.aux_visual_proj = bundle.aux_visual_proj
        self.action_encoder = bundle.action_encoder
        self.reused_blocks = bundle.reused_blocks

        # We store a copy of pos_emb parameter if provided.
        # This is only used in concat_transformer mode.
        self.pos_emb = bundle.pos_emb

        hidden = cfg.hidden_size

        if cfg.fusion == "film":
            self.film = _FiLM(hidden, cfg.dropout)
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.cross_blocks = None

        elif cfg.fusion == "cross_attn":
            self.film = None
            self.cross_blocks = nn.ModuleList(
                [
                    _CrossAttentionBlock(
                        hidden=hidden,
                        n_heads=cfg.n_heads,
                        mlp_ratio=cfg.mlp_ratio,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.n_cross_attn_blocks)
                ]
            )
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

        elif cfg.fusion == "concat_transformer":
            self.film = None
            self.cross_blocks = None
            self.post = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

        else:
            raise ValueError(f"Unknown fusion='{cfg.fusion}'")

        self._apply_freezing()

    def _apply_freezing(self) -> None:
        def _freeze(m: nn.Module) -> None:
            for p in m.parameters():
                p.requires_grad = False

        if self.cfg.freeze_vlm:
            _freeze(self.xvla_model.vlm)

        if self.cfg.freeze_vlm_proj:
            _freeze(self.vlm_proj)

        if self.cfg.freeze_aux_visual_proj:
            _freeze(self.aux_visual_proj)

        if self.cfg.freeze_action_encoder:
            _freeze(self.action_encoder)

        if self.cfg.freeze_reused_transformer_blocks:
            _freeze(self.reused_blocks)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _domain_id(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.full(
            (batch_size,),
            int(self.cfg.fixed_domain_id),
            dtype=torch.long,
            device=device,
        )

    def encode_state_tokens(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
    ) -> Tensor:
        """
        Returns state tokens in critic hidden space: [B, T_state, H]
        """
        enc = self.xvla_model.forward_vlm(input_ids=input_ids, pixel_values=image_input, image_mask=image_mask)
        vlm_features = enc["vlm_features"]  # [B, T_vlm, D]
        aux_visual = enc["aux_visual_inputs"]  # [B, T_aux, D]

        B = input_ids.shape[0]
        did = self._domain_id(B, vlm_features.device)

        # Project to hidden space
        if self._is_domain_aware(self.vlm_proj):
            vlm_h = self.vlm_proj(vlm_features, did)  # type: ignore[misc]
        else:
            vlm_h = self.vlm_proj(vlm_features)

        tokens = [vlm_h]

        if self.cfg.use_aux_visual_inputs:
            if self._is_domain_aware(self.aux_visual_proj):
                aux_h = self.aux_visual_proj(aux_visual, did)  # type: ignore[misc]
            else:
                aux_h = self.aux_visual_proj(aux_visual)
            tokens.append(aux_h)

        return torch.cat(tokens, dim=1)

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
        t = torch.full((B,), float(self.cfg.action_encoder_t), device=dev, dtype=proprio.dtype)
        time_emb = timestep_embedding(t, self.xvla_model.transformer.dim_time)  # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, H, -1)

        proprio_tokens = proprio.unsqueeze(1).expand(B, H, proprio.shape[-1])
        action_tokens = torch.cat([action_chunk, proprio_tokens, time_tokens], dim=-1)

        # DomainAwareLinear expects domain_id, so pass fixed 0
        return self.action_encoder(action_tokens, did)  # type: ignore[misc]

    def forward(
        self,
        *,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        proprio: Tensor,
        action_chunk: Tensor,
    ) -> Tensor:
        """
        Returns a fused feature vector [B, hidden] suitable for Q/V heads.
        """
        state_tokens = self.encode_state_tokens(input_ids, image_input, image_mask)
        action_tokens = self.encode_action_tokens(action_chunk, proprio)

        if self.cfg.fusion == "film":
            a = self._pool(action_tokens)
            s = self._pool(state_tokens)
            fused = self.film(a, s)  # type: ignore[union-attr]
            return self.post(fused)

        if self.cfg.fusion == "cross_attn":
            x = action_tokens
            for blk in self.cross_blocks:  # type: ignore[union-attr]
                x = blk(x, state_tokens)
            fused = self._pool(x)
            return self.post(fused)

        # concat_transformer
        x = torch.cat([action_tokens, state_tokens], dim=1)
        if self.cfg.use_pos_emb and self.pos_emb is not None:
            seq_len = x.shape[1]
            if seq_len > self.pos_emb.shape[1]:
                raise ValueError(f"Critic seq_len={seq_len} exceeds pos_emb limit={self.pos_emb.shape[1]}")
            x = x + self.pos_emb[:, :seq_len, :]

        if len(self.reused_blocks) > 0:
            for blk in self.reused_blocks:
                x = blk(x)

        # Pool only the action segment (first H tokens)
        H = action_tokens.shape[1]
        fused = self._pool(x[:, :H, :])
        return self.post(fused)

    def _pool(self, tokens: Tensor) -> Tensor:
        if self.cfg.pool == "mean":
            return tokens.mean(dim=1)
        raise ValueError(f"Unknown pool='{self.cfg.pool}'")

    @staticmethod
    def _is_domain_aware(module: nn.Module) -> bool:
        # DomainAwareLinear in soft_transformer has signature forward(x, domain_id).
        # We detect it conservatively by checking for nn.Embedding attributes used there.
        return hasattr(module, "fc") and hasattr(module, "bias") and isinstance(getattr(module, "fc"), nn.Embedding)
