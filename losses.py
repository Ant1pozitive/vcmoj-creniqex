import torch
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

class VC_MOJI_Loss:
    """
    CE + KL + ortho + commitment + contrastive
    """

    def __init__(self, recon_weight=1.0, kl_weight=1e-3, ortho_weight=1e-4,
                 commitment_weight=1.0, contrastive_weight=1.0):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.ortho_weight = ortho_weight
        self.commitment_weight = commitment_weight
        self.contrastive_weight = contrastive_weight

    def __call__(self, logits, targets, mus, logvars, model=None, commit_loss_tensor=None, contrastive_loss_tensor=None):
        B = targets.shape[0]
        recon = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

        kl = 0.0
        for mu, logvar in zip(mus, logvars):
            kl += kl_divergence(mu, logvar).mean()
        loss = self.recon_weight * recon + self.kl_weight * kl

        # orthogonality penalty (same approach)
        ortho_pen = torch.tensor(0.0, device=logits.device)
        if model is not None:
            for name, module in model.named_modules():
                if module.__class__.__name__ == "NJXAttention":
                    W = module.qkv_proj.weight  # shape (3*D, D)
                    W2 = W @ W.t()
                    eye = torch.eye(W2.size(0), device=W2.device)
                    ortho_pen = ortho_pen + ((W2 - eye) ** 2).sum()

            loss = loss + self.ortho_weight * ortho_pen

        # add commitment and contrastive (these are already scaled by internal weights in quantizer)
        commit_scalar = commit_loss_tensor if commit_loss_tensor is not None else torch.tensor(0.0, device=logits.device)
        contrast_scalar = contrastive_loss_tensor if contrastive_loss_tensor is not None else torch.tensor(0.0, device=logits.device)

        loss = loss + commit_scalar + contrast_scalar

        metrics = {
            'recon': recon.item(),
            'kl': kl.item() if isinstance(kl, torch.Tensor) else float(kl),
            'ortho': ortho_pen.item() if isinstance(ortho_pen, torch.Tensor) else float(ortho_pen),
            'commit': commit_scalar.item() if isinstance(commit_scalar, torch.Tensor) else float(commit_scalar),
            'contrast': contrast_scalar.item() if isinstance(contrast_scalar, torch.Tensor) else float(contrast_scalar)
        }
        return loss, metrics
