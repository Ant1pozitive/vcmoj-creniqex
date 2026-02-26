import torch
import torch.nn as nn
import torch.nn.functional as F
from .njx_attention import NJXAttention
from .variational import VariationalLatent
from .quantizer import CreniqQuantizer
from .eck_lock import ECKLock

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
        B, T, D = x.shape
        device = x.device
        k = self.k
        if k == 0:
            return x, 0.0
        idx = torch.randperm(D, device=device)[:k]
        noise = torch.randn(B, T, k, device=device) * self.jitter_std
        if self.zero_mean:
            noise = noise - noise.mean(dim=(0,1), keepdim=True)
        gate = torch.sigmoid(self.gate) if self.gate is not None else 1.0
        x_new = x.clone()
        x_new[:,:, idx] = x_new[:,:, idx] + gate * noise
        perm = torch.randperm(k, device=device)
        x_selected = x_new[:,:, idx][:,:, perm]
        x_new[:,:, idx] = x_selected
        return x_new, gate.item() if isinstance(gate, float) else gate

class VC_MOJI_Block(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, window_size=16, exchange_frac=0.125,
                 latent_dim=64, iterative_steps=3, dropout=0.1, num_codebooks=7, codebook_size=256,
                 commitment_weight=1.0, contrastive_weight=1.0, cj_k_channels=36, cj_jitter_std=0.02,
                 eck_key_dim=48, eck_topk=None):
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
        self.quantizer = CreniqQuantizer(latent_dim, num_codebooks=num_codebooks, codebook_size=codebook_size,
                                         commitment_weight=commitment_weight, contrastive_weight=contrastive_weight)
        self.expand = nn.Linear(latent_dim, dim)
        self.iterative_steps = iterative_steps
        self.channel_jitter = ChannelJitterExchange(dim, k_channels=cj_k_channels, jitter_std=cj_jitter_std)

        # ECKLock module integrated at start of block (before attention)
        self.eck_lock = ECKLock(dim, key_dim=eck_key_dim, hard_topk=eck_topk)

    def forward(self, x):
        """
        returns:
          x_out (B,T,D), mu, logvar, commit_loss, contrastive_loss, cj_gate, eck_stats
        """
        # ECK lock first (channel gating conditioned on context)
        x, eck_stats = self.eck_lock(x)

        for _ in range(self.iterative_steps):
            residual = x
            x = self.norm1(x)
            x = residual + self.attn(x)
            residual2 = x
            x = self.norm2(x)
            x = residual2 + self.mlp(x)

        x, cj_gate = self.channel_jitter(x)

        z, mu, logvar = self.variational(x)
        quant_z, commit_loss, contrastive_loss, indices = self.quantizer(z)
        z_exp = self.expand(quant_z)
        z_broadcast = z_exp.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + z_broadcast
        return x, mu, logvar, commit_loss, contrastive_loss, cj_gate, eck_stats

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

    def forward(self, x, return_emb_for_jacobian=False):
        """
        returns:
          logits (B,T,V), list of mus, logvars, total_commit, total_contrast, cj_gates, eck_stats_list, emb (optional)
        If return_emb_for_jacobian=True, returns embeddings used as input (requires_grad set), enabling jacobian reg.
        """
        B, T = x.shape
        device = x.device
        emb = self.token_emb(x) + self.pos_emb[:, :T, :].to(device)  # (B,T,D)
        if return_emb_for_jacobian:
            emb = emb.clone().requires_grad_(True)
        h = emb
        mus = []
        logvars = []
        total_commit = torch.tensor(0.0, device=device)
        total_contrast = torch.tensor(0.0, device=device)
        cj_gates = []
        eck_stats_list = []
        for layer in self.layers:
            h, mu, logvar, commit_loss, contrastive_loss, cj_gate, eck_stats = layer(h)
            mus.append(mu)
            logvars.append(logvar)
            total_commit = total_commit + commit_loss
            total_contrast = total_contrast + contrastive_loss
            cj_gates.append(cj_gate)
            eck_stats_list.append(eck_stats)
        h = self.norm(h)
        logits = self.head(h)
        if return_emb_for_jacobian:
            return logits, mus, logvars, total_commit, total_contrast, cj_gates, eck_stats_list, emb
        return logits, mus, logvars, total_commit, total_contrast, cj_gates, eck_stats_list
