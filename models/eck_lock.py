import torch
import torch.nn as nn
import torch.nn.functional as F

class ECKLock(nn.Module):
    """
    Embedding-Conditioned Keyed Lock (ECKLock).
    - pooled context -> projected key -> per-channel gate
    - key_dim defaults to 48 (r48)
    - optional hard_topk: zeroes all but top-k gates (deterministic during forward)
    """

    def __init__(self, dim, key_dim=48, hard_topk=None, gate_init=0.0, eps=1e-6):
        """
        dim: channel dimension (D)
        key_dim: internal projection dim (r48)
        hard_topk: int or None — if set, enforce only top-k gates (hard lock)
        """
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim
        self.hard_topk = hard_topk
        self.eps = eps

        self.ctx_proj = nn.Sequential(
            nn.Linear(dim, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, dim)
        )
        # bias/gain for gate; initialize near gate_init (so gate ~ sigmoid(gate_init) initially)
        self.g_bias = nn.Parameter(torch.full((dim,), gate_init))
        # optional per-channel threshold (learnable)
        self.threshold = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (x_gated, gate_stats)
        gate_stats: dict {mean_gate, topk_fraction}
        """
        B, T, D = x.shape
        device = x.device
        # pool context
        ctx = x.mean(dim=1)  # (B, D)
        gate_pre = self.ctx_proj(ctx) + self.g_bias  # (B, D)
        gate = torch.sigmoid(gate_pre)  # (B, D) in (0,1)
        # optional hard top-k: per sample topk
        topk_frac = 0.0
        if self.hard_topk is not None and self.hard_topk > 0:
            k = min(self.hard_topk, D)
            # compute topk mask per sample
            topk_vals, topk_idx = torch.topk(gate, k, dim=-1)
            mask = torch.zeros_like(gate, device=device)
            mask.scatter_(1, topk_idx, 1.0)
            gate = gate * mask
            topk_frac = float(k) / float(D)

        # apply gate multiplicatively
        gate_exp = gate.unsqueeze(1)  # (B,1,D)
        x_out = x * gate_exp

        stats = {'mean_gate': gate.mean().item(), 'topk_fraction': topk_frac}
        return x_out, stats
