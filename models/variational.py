# models/variational.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalLatent(nn.Module):
    """
    Returns the raw (not expanded) latent z(B, latent_dim),
    as well as mu/logvar for ELBO. Expansion is performed above (to insert a quantizer between mu and expand).
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        Return z_sample (B, latent_dim), mu (B,latent_dim), logvar (B,latent_dim)
        """
        B, T, D = x.shape
        pooled = x.mean(dim=1)  # (B, D)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, latent_dim)
        return z, mu, logvar
