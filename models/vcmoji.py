import torch
import torch.nn as nn
import torch.nn.functional as F
from .njx_attention import NJXAttention
from .variational import VariationalLatent
from .quantizer import CreniqQuantizer

class ChannelJitterExchange(nn.Module):
    """
    Selects k channels (min(k,D)), adds zero-mean noise, and randomly rearranges the channels.
    It is controlled through the learnable gate.
    """

    def __init__(self, dim, k_channels=36, jitter_std=0.02, zero_mean=True, learnable_gate=True):
        super().__init__()
        self.dim = dim
        self.k = min(k_channels, dim)
        self.jitter_std = jitter_std
        self.zero_mean = zero_mean
        self.gate = nn.Parameter(torch.tensor(0.0)) if learnable_gate else None

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        device = x.device
        k = self.k
        if k == 0:
            return x, 0.0
        # choose k indices (stochastic per forward)
        idx = torch.randperm(D, device=device)[:k]
        # create noise (B, T, k)
        noise = torch.randn(B, T, k, device=device) * self.jitter_std
        if self.zero_mean:
            noise = noise - noise.mean(dim=(0,1), keepdim=True)
        gate = torch.sigmoid(self.gate) if self.gate is not None else 1.0
        # apply additive noise
        x_new = x.clone()
        x_new[:,:, idx] = x_new[:,:, idx] + gate * noise
        # perform random permutation among selected channels (exchange)
        perm = torch.randperm(k, device=device)
        x_selected = x_new[:,:, idx][:,:, perm]
        x_new[:,:, idx] = x_selected
        return x_new, gate.item() if isinstance(gate, float) else gate

class VC_MOJI_Block(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, window_size=16, exchange_frac=0.125,
                 latent_dim=64, iterative_steps=3, dropout=0.1, num_codebooks=7, codebook_size=256,
                 commitment_weight=1.0, contrastive_weight=1.0, cj_k_channels=36, cj_jitter_std=0.02):
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
        self.variational = VariationalLatent(dim, latent_dim)
        # quantizer works on latent z (latent_dim)
        self.quantizer = CreniqQuantizer(latent_dim, num_codebooks=num_codebooks, codebook_size=codebook_size,
                                         commitment_weight=commitment_weight, contrastive_weight=contrastive_weight)
        self.expand = nn.Linear(latent_dim, dim)  # expand quantized latent to dim
        self.iterative_steps = iterative_steps
        self.channel_jitter = ChannelJitterExchange(dim, k_channels=cj_k_channels, jitter_std=cj_jitter_std)

    def forward(self, x):
        """
        Returns:
          x_out (B,T,D), mu, logvar, commit_loss, contrastive_loss, cj_gate
        """
        for _ in range(self.iterative_steps):
            residual = x
            x = self.norm1(x)
            x = residual + self.attn(x)
            residual2 = x
            x = self.norm2(x)
            x = residual2 + self.mlp(x)

        # channel jitter exchange before variational fusion
        x, cj_gate = self.channel_jitter(x)

        # variational latent (pooled)
        z, mu, logvar = self.variational(x)  # z: (B, latent_dim)
        # quantize z
        quant_z, commit_loss, contrastive_loss, indices = self.quantizer(z)  # quant_z: (B, latent_dim)
        # expand and broadcast
        z_exp = self.expand(quant_z)  # (B, dim)
        z_broadcast = z_exp.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, dim)
        x = x + z_broadcast
        return x, mu, logvar, commit_loss, contrastive_loss, cj_gate

class VC_MOJI_Transformer(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=8, mlp_ratio=4.0, window_size=16,
                 latent_dim=64, iterative_steps=3, max_seq_len=512, **kwargs):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.layers = nn.ModuleList([
            VC_MOJI_Block(dim, n_heads=n_heads, mlp_ratio=mlp_ratio, window_size=window_size,
                          latent_dim=latent_dim, iterative_steps=iterative_steps, **kwargs)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        returns:
          logits (B,T,V), list of mus, logvars, total_commit_loss, total_contrastive_loss, cj_stats
        """
        B, T = x.shape
        device = x.device
        h = self.token_emb(x) + self.pos_emb[:, :T, :].to(device)
        mus = []
        logvars = []
        total_commit = 0.0
        total_contrast = 0.0
        cj_gates = []
        for layer in self.layers:
            h, mu, logvar, commit_loss, contrastive_loss, cj_gate = layer(h)
            mus.append(mu)
            logvars.append(logvar)
            total_commit = total_commit + (commit_loss if isinstance(commit_loss, torch.Tensor) else torch.tensor(commit_loss, device=device))
            total_contrast = total_contrast + (contrastive_loss if isinstance(contrastive_loss, torch.Tensor) else torch.tensor(contrastive_loss, device=device))
            cj_gates.append(cj_gate)
        h = self.norm(h)
        logits = self.head(h)
        return logits, mus, logvars, total_commit, total_contrast, cj_gates
