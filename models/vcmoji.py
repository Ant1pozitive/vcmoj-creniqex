import torch
import torch.nn as nn
import torch.nn.functional as F
from .njx_attention import NJXAttention
from .variational import VariationalLatent

class VC_MOJI_Block(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, window_size=16, exchange_frac=0.125, latent_dim=64, iterative_steps=3, dropout=0.1):
        super().__init__()
        self.attn = NJXAttention(dim, n_heads=n_heads, window_size=window_size, exchange_frac=exchange_frac, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.variational = VariationalLatent(dim, latent_dim, expand_dim=dim)
        self.iterative_steps = iterative_steps
        # orthogonality regularizer will be computed externally (loss)

    def forward(self, x):
        # iterative refinement loop (simple)
        for _ in range(self.iterative_steps):
            residual = x
            x = self.norm1(x)
            x = residual + self.attn(x)
            residual2 = x
            x = self.norm2(x)
            x = residual2 + self.mlp(x)
        # apply variational latent bottleneck
        z_broadcast, mu, logvar = self.variational(x)
        x = x + z_broadcast  # conditional fusion (additive)
        return x, mu, logvar

class VC_MOJI_Transformer(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=8, mlp_ratio=4.0, window_size=16, latent_dim=64, iterative_steps=3, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.layers = nn.ModuleList([
            VC_MOJI_Block(dim, n_heads=n_heads, mlp_ratio=mlp_ratio, window_size=window_size, exchange_frac=0.125, latent_dim=latent_dim, iterative_steps=iterative_steps)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T) token ids
        returns logits (B, T, V), list of mus/logvars for KL accumulation
        """
        B, T = x.shape
        device = x.device
        h = self.token_emb(x) + self.pos_emb[:, :T, :].to(device)
        mus = []
        logvars = []
        for layer in self.layers:
            h, mu, logvar = layer(h)
            mus.append(mu)
            logvars.append(logvar)
        h = self.norm(h)
        logits = self.head(h)
        return logits, mus, logvars
