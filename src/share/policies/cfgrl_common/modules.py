"""Reusable neural network blocks for the share-local CFGRL stack."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(time: Tensor, dim: int) -> Tensor:
    """Return a sinusoidal embedding for normalized time values in ``[0, 1]``."""
    if dim <= 0:
        raise ValueError("dim must be > 0")
    half = dim // 2
    scale = math.log(10_000) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=time.device, dtype=time.dtype) * -scale)
    args = time.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLP(nn.Module):
    """Small GELU MLP used across policy, critic, and backbone adapters."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if activation is None:
            activation = nn.GELU()

        layers: list[nn.Module] = []
        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)

        for idx, (src, dst) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(src, dst))
            is_last = idx == len(dims) - 2
            if not is_last:
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """A compact residual cross-attention block used by the critic backbone."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: Tensor, context_tokens: Tensor) -> Tensor:
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(context_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        query_tokens = query_tokens + self.dropout(out)
        query_tokens = query_tokens + self.ff(self.norm_ff(query_tokens))
        return query_tokens


class ActionChunkTokenEncoder(nn.Module):
    """Encode chunked actions into hidden tokens with optional time/context input."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        chunk_size: int,
        *,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.time_embed_dim = time_embed_dim
        self.input_proj = nn.Linear(action_dim + time_embed_dim, hidden_dim)
        self.positional = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, action_chunk: Tensor, *, time: Tensor, context: Tensor | None = None) -> Tensor:
        batch_size, chunk_size, _ = action_chunk.shape
        if chunk_size != self.chunk_size:
            raise ValueError(f"Expected chunk size {self.chunk_size}, got {chunk_size}")

        time_emb = sinusoidal_time_embedding(time, self.time_embed_dim)
        time_tokens = time_emb.unsqueeze(1).expand(batch_size, chunk_size, -1)
        tokens = self.input_proj(torch.cat([action_chunk, time_tokens], dim=-1))
        tokens = tokens + self.positional[:, :chunk_size]
        if context is not None:
            tokens = tokens + self.context_proj(context).unsqueeze(1)
        return self.norm(self.dropout(tokens))


class ObservationFusion(nn.Module):
    """Project concatenated observation features into a shared conditioning vector."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply DiT-style adaptive layer-norm modulation."""

    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT-style transformer block with adaLN-zero conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        cond_dim: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply self-attention and MLP updates conditioned on ``cond``."""

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(6, dim=-1)
        attn_in = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class DiTFinalLayer(nn.Module):
    """Final adaLN-modulated projection from hidden action tokens to actions."""

    def __init__(self, hidden_dim: int, out_dim: int, *, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Project conditioned hidden tokens to the policy output space."""

        shift, scale = self.ada_ln(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))
