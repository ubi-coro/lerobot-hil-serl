from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FiLM(nn.Module):
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


class CrossAttentionBlock(nn.Module):
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
