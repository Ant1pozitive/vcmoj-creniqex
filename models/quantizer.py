import torch
import torch.nn as nn
import torch.nn.functional as F

class CreniqQuantizer(nn.Module):
    """
    Contrastive Residual Noise-Injected Quantizer (VQ-like)
    - num_codebooks: split the latent into num_codebooks sub-vectors (along the feature axis)
    - codebook_size: the number of vectors in each sub-codebook
    - commitment loss + contrastive InfoNCE loss
    """

    def __init__(self, latent_dim, num_codebooks=7, codebook_size=256, commitment_weight=1.0, contrastive_weight=1.0, tau=0.07):
        super().__init__()
        assert latent_dim % num_codebooks == 0, "latent_dim must be divisible by num_codebooks"
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.sub_dim = latent_dim // num_codebooks

        # embeddings: (num_codebooks, codebook_size, sub_dim)
        emb = torch.randn(num_codebooks, codebook_size, self.sub_dim)
        self.register_parameter("embeddings", nn.Parameter(emb))
        self.commitment_weight = commitment_weight
        self.contrastive_weight = contrastive_weight
        self.tau = tau

    def forward(self, z):
        """
        z: (B, latent_dim)
        returns:
          quant_z: (B, latent_dim)  (straight-through)
          commit_loss: scalar tensor
          contrastive_loss: scalar tensor
          indices: (B, num_codebooks) ints
        """
        B, L = z.shape
        nc = self.num_codebooks
        sd = self.sub_dim
        # split z -> (B, nc, sd)
        z_splits = z.view(B, nc, sd)  # (B, nc, sd)
        # compute distances per codebook
        # embeddings: (nc, codebook_size, sd)
        emb = self.embeddings  # (nc, K, sd)

        # compute squared distances efficiently
        # For each codebook i, compute distance between z_splits[:,i,:] and emb[i,:,:]
        # We'll vectorize: expand dims
        zs = z_splits.unsqueeze(2)  # (B, nc, 1, sd)
        em = emb.unsqueeze(0)       # (1, nc, K, sd)
        dists = ((zs - em) ** 2).sum(dim=-1)  # (B, nc, K)
        indices = dists.argmin(dim=-1)  # (B, nc)

        # gather embeddings
        # indices -> expand to gather
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, sd)  # (B, nc, sd)
        quant_splits = torch.gather(emb.unsqueeze(0).expand(B, -1, -1, -1), 2, indices_exp.unsqueeze(2)).squeeze(2)
        # simpler: use loop for clarity (cost ok for prototype)
        # quant_splits = torch.stack([emb[i, indices[:, i], :] for i in range(nc)], dim=1)  # alternative (keeping above)

        quant_splits = quant_splits  # (B, nc, sd)
        quant_z = quant_splits.view(B, L)

        # commitment loss (MSE between z and quantized — stopgrad on embeddings)
        commit_loss = F.mse_loss(quant_z.detach(), z)  # encourage encoder to commit to embeddings
        # embed loss (push embeddings to encoder outputs) - we omit embedding update via EMA for simplicity;
        # instead, add a small loss to move embeddings towards z (using stopgrad on z)
        embed_loss = F.mse_loss(quant_z, z.detach())
        total_commit = commit_loss + embed_loss

        # straight-through estimator
        quant_z_st = z + (quant_z - z).detach()

        # contrastive InfoNCE: compare normalized quant_z (positive) vs other batch items (negatives)
        z_norm = F.normalize(z, dim=-1)  # (B, L)
        q_norm = F.normalize(quant_z.detach(), dim=-1)  # (B, L) treat quant as key/anchor
        logits = torch.matmul(q_norm, z_norm.t()) / self.tau  # (B, B)
        labels = torch.arange(B, device=z.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        # scale losses
        commit_loss_scaled = self.commitment_weight * total_commit
        contrastive_loss_scaled = self.contrastive_weight * contrastive_loss

        return quant_z_st, commit_loss_scaled, contrastive_loss_scaled, indices
