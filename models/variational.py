import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalLatent(nn.Module):
    """
    Simple variational bottleneck: maps input features to (mu, logvar), samples with reparam trick,
    and returns expanded latent to be fused back into sequence (additive conditioning).
    """

    def __init__(self, input_dim, latent_dim, expand_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.expand_dim = expand_dim or input_dim

        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)
        self.expand = nn.Linear(latent_dim, self.expand_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        We'll aggregate x along time (mean) to produce a global context for the variational bottleneck,
        then produce latent z and broadcast back to T positions.
        """
        B, T, D = x.shape
        pooled = x.mean(dim=1)  # (B, D)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, latent_dim)
        z_exp = self.expand(z)  # (B, expand_dim)
        z_broadcast = z_exp.unsqueeze(1).expand(-1, T, -1)  # (B, T, expand_dim)
        # return z (for KL), broadcasted latent addition
        return z_broadcast, mu, logvar
