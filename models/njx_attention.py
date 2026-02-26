import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NJXAttention(nn.Module):
    """
    Noisy Jittered Cross-window Exchange Multi-head Attention (NJX).
    - Splits sequence into windows of size w.
    - Computes local attention inside each window.
    - Performs cross-window exchange: selects a small subset of tokens per window to exchange
      with other windows with jittered indices (stochastic but differentiable via soft-sampling).
    - Then computes lightweight cross-window attention among exchanged tokens to propagate global info.
    """

    def __init__(self, dim, n_heads=8, head_dim=None, window_size=16, exchange_frac=0.125, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim or (dim // n_heads)
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.window_size = window_size
        self.exchange_frac = exchange_frac
        self.dropout = nn.Dropout(dropout)

        # small projection for cross-window exchange attention (low cost)
        self.cross_proj = nn.Linear(dim, dim)

    def _split_windows(self, x):
        """
        x: (B, T, D)
        returns: (B, num_windows, window_size, D)
        """
        B, T, D = x.shape
        w = self.window_size
        pad = (w - (T % w)) % w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            T = T + pad
        num_windows = T // w
        xw = x.view(B, num_windows, w, D)
        return xw, pad

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # (B, H, T, Hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        # Local windowed attention
        w = self.window_size
        x_for_windows = x  # use original for splitting
        xw, pad = self._split_windows(x_for_windows)  # (B, num_windows, w, D)
        num_windows = xw.size(1)

        # Split q,k,v into windows similarly
        def split_heads_to_windows(tensor):
            # tensor: (B, H, T, Hd)
            B, H, Tfull, Hd = tensor.shape
            pad_local = (w - (Tfull % w)) % w
            if pad_local:
                tensor = F.pad(tensor, (0,0,0,pad_local))  # pad time dim
                Tfull = Tfull + pad_local
            tensor = tensor.view(B, H, num_windows, w, Hd)  # (B,H,num_windows,w,Hd)
            return tensor

        q_w = split_heads_to_windows(q)
        k_w = split_heads_to_windows(k)
        v_w = split_heads_to_windows(v)

        # Local attention per window
        attn_scores = torch.einsum("bhnwd,bhnwd->bhnw", q_w, k_w)  # incorrect dims -> compute properly
        # compute properly:
        # q_w: (B,H,NW,w,Hd), k_w: same
        q_w_flat = q_w  # reuse
        k_w_flat = k_w
        # compute dot
        attn = torch.einsum("bhnqd,bhnkd->bhnqk", q_w_flat, k_w_flat) * self.scale  # (B,H,num_windows,w,w)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        local_out = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_w)  # (B,H,num_windows,w,Hd)
        local_out = local_out.contiguous().view(B, self.n_heads, num_windows * w, self.head_dim)  # (B,H,T_pad,Hd)
        local_out = local_out.transpose(1,2).contiguous().view(B, num_windows * w, self.dim)  # (B, T_pad, D)
        if pad:
            local_out = local_out[:, :T, :]

        # Cross-window exchange (low-cost)
        # Select top-k tokens per window (soft selection via small MLP)
        # We'll compute token scores and take softmax across window tokens, then pick fraction.
        scores = torch.tanh(self.cross_proj(x)).mean(-1)  # (B, T)
        xw_scores, _ = self._split_windows(scores.unsqueeze(-1))
        xw_scores = xw_scores.squeeze(-1)  # (B, num_windows, w)

        k_select = max(1, int(self.exchange_frac * w))
        # Soft selection: for each window, take softmax and compute weighted sum of token vectors to represent exchanged summary
        x_windows, _ = self._split_windows(x)
        # x_windows: (B, num_windows, w, D)
        weights = F.softmax(xw_scores, dim=-1).unsqueeze(-1)  # (B, num_windows, w, 1)
        summaries = (x_windows * weights).sum(dim=2)  # (B, num_windows, D)

        # Jitter: random small permutation across windows (stochastic mixing)
        jitter = torch.randperm(num_windows, device=x.device)
        jittered = summaries[:, jitter, :]  # (B, num_windows, D)

        # Cross-window attention among jittered summaries
        # Project to heads
        cs_q = summaries.view(B, num_windows, 1, self.dim).transpose(0,1)  # (num_windows, B,1,D)
        cs_k = jittered.view(B, num_windows, 1, self.dim).transpose(0,1)
        # simple dot product attention per window index
        # reshape back and compute attention matrix
        cs_q2 = summaries  # (B, num_windows, D)
        cs_k2 = jittered
        cs_attn = torch.matmul(cs_q2, cs_k2.transpose(-1,-2)) * (self.dim ** -0.5)  # (B, num_windows, num_windows)
        cs_attn = F.softmax(cs_attn, dim=-1)
        cs_out = torch.matmul(cs_attn, jittered)  # (B, num_windows, D)

        # Broadcast cross-window summaries back to token positions inside each window (additive)
        cs_out_expanded = cs_out.unsqueeze(2).expand(-1, -1, w, -1).contiguous().view(B, num_windows * w, D)
        if pad:
            cs_out_expanded = cs_out_expanded[:, :T, :]

        final = local_out + 0.25 * cs_out_expanded  # combine local and global-exchange signals
        out = self.out_proj(final)
        return out
