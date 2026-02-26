import torch
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    # standard gaussian prior
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # sum over latent dim -> (B,)

class VC_MOJI_Loss:
    """
    Combines reconstruction (cross-entropy), KL from multiple latents (sum), and orthogonality penalty.
    """

    def __init__(self, recon_weight=1.0, kl_weight=1e-3, ortho_weight=1e-4):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.ortho_weight = ortho_weight

    def __call__(self, logits, targets, mus, logvars, model=None):
        """
        logits: (B,T,V)
        targets: (B,T)
        mus/logvars: list of tensors per layer: each (B, latent_dim)
        """
        B = targets.shape[0]
        recon = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

        # KL: average over batch
        kl = 0.0
        for mu, logvar in zip(mus, logvars):
            kl += kl_divergence(mu, logvar).mean()
        loss = self.recon_weight * recon + self.kl_weight * kl

        ortho_pen = 0.0
        if model is not None:
            # apply orthogonality penalty to qkv_proj weights in NJXAttention modules
            for name, module in model.named_modules():
                if module.__class__.__name__ == "NJXAttention":
                    # qkv_proj: [D, 3D]
                    W = module.qkv_proj.weight  # (3D, D) or (D,3D) depending - in our code it's Linear(dim,3*dim) so W.shape=(3*dim, dim)
                    # we'll encourage rows (projection filters) to be orthogonal
                    W2 = W @ W.t()
                    eye = torch.eye(W2.size(0), device=W2.device)
                    ortho_pen = ((W2 - eye) ** 2).sum()
                    ortho_pen += ortho_pen
            loss = loss + self.ortho_weight * ortho_pen

        return loss, {'recon': recon.item(), 'kl': kl.item() if isinstance(kl, torch.Tensor) else float(kl), 'ortho': ortho_pen.item() if isinstance(ortho_pen, torch.Tensor) else float(ortho_pen)}
