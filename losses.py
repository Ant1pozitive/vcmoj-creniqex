import torch
import torch.nn.functional as F
from torch import autograd

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

def label_smoothed_nll_loss(logits, targets, eps=0.1):
    """
    Standard label smoothing for cross-entropy.
    """
    V = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1.0 - eps) * nll + eps * smooth_loss
    return loss.mean()

def kernel_alignment_penalty(logits, targets, kernel_scale=0.5):
    """
    Simple kernel alignment: compares normalized logits to one-hot target vectors
    (encourages margin/alignment). This is a light-weight KCE.
    """
    B, T, V = logits.shape
    logits_flat = logits.view(B*T, V)
    targets_flat = targets.view(B*T)
    logits_norm = F.normalize(logits_flat, dim=-1)
    target_onehot = F.one_hot(targets_flat, num_classes=V).float()
    target_norm = F.normalize(target_onehot, dim=-1)
    # cosine similarity matrix diagonal -> want them to be high; penalty = 1 - cos
    sim = (logits_norm * target_norm).sum(dim=-1)
    penalty = (1.0 - sim).mean()
    return penalty * kernel_scale

def estimate_jacobian_penalty(logits, emb, num_iters=3):
    """
    Estimate Frobenius norm of Jacobian of logits w.r.t. input embedding 'emb'
    using Hutchinson / random vector trick:
      E[ || J^T v ||^2 ] = ||J||_F^2 for v ~ N(0,I)
    We'll sample v with same shape as logits and compute grad of (logits * v).sum() wrt emb.
    """
    penalty = torch.tensor(0.0, device=logits.device)
    for _ in range(num_iters):
        v = torch.randn_like(logits)
        scalar = (logits * v).sum()
        grads = autograd.grad(scalar, emb, create_graph=True)[0]  # (B,T,D)
        penalty = penalty + (grads.pow(2).sum())
    penalty = penalty / float(num_iters)
    # normalize by batch-size for stability
    penalty = penalty / (emb.size(0) * emb.size(1))
    return penalty

class VC_MOJI_Loss:
    """
    CE + KL + ortho + commitment + contrastive + RCE (label smooth + kernel) + jacobian reg
    """

    def __init__(self, recon_weight=1.0, kl_weight=1e-3, ortho_weight=1e-4,
                 commitment_weight=1.0, contrastive_weight=1.0,
                 rce_eps=0.1, rce_kernel_scale=0.5, jreg_weight=0.0, jreg_iters=3):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.ortho_weight = ortho_weight
        self.commitment_weight = commitment_weight
        self.contrastive_weight = contrastive_weight
        self.rce_eps = rce_eps
        self.rce_kernel_scale = rce_kernel_scale
        self.jreg_weight = jreg_weight
        self.jreg_iters = jreg_iters

    def __call__(self, logits, targets, mus, logvars, model=None,
                 commit_loss_tensor=None, contrastive_loss_tensor=None, emb_for_jacobian=None):
        B = targets.shape[0]

        # RCE: label-smoothed CE + kernel alignment
        recon = label_smoothed_nll_loss(logits, targets, eps=self.rce_eps)
        kernel_pen = kernel_alignment_penalty(logits, targets, kernel_scale=self.rce_kernel_scale)
        rce = recon + kernel_pen

        kl = 0.0
        for mu, logvar in zip(mus, logvars):
            kl += kl_divergence(mu, logvar).mean()
        loss = self.recon_weight * rce + self.kl_weight * kl

        # orthogonality penalty
        ortho_pen = torch.tensor(0.0, device=logits.device)
        if model is not None:
            for name, module in model.named_modules():
                if module.__class__.__name__ == "NJXAttention":
                    W = module.qkv_proj.weight
                    W2 = W @ W.t()
                    eye = torch.eye(W2.size(0), device=W2.device)
                    ortho_pen = ortho_pen + ((W2 - eye) ** 2).sum()
            loss = loss + self.ortho_weight * ortho_pen

        # commitment / contrastive
        commit_scalar = commit_loss_tensor if commit_loss_tensor is not None else torch.tensor(0.0, device=logits.device)
        contrast_scalar = contrastive_loss_tensor if contrastive_loss_tensor is not None else torch.tensor(0.0, device=logits.device)
        loss = loss + commit_scalar + contrast_scalar

        # Jacobian reg (if embedding provided)
        jpen = torch.tensor(0.0, device=logits.device)
        if emb_for_jacobian is not None and self.jreg_weight > 0.0:
            jpen = estimate_jacobian_penalty(logits, emb_for_jacobian, num_iters=self.jreg_iters)
            loss = loss + self.jreg_weight * jpen

        metrics = {
            'rce_recon': recon.item(),
            'rce_kernel': kernel_pen.item(),
            'kl': kl.item() if isinstance(kl, torch.Tensor) else float(kl),
            'ortho': ortho_pen.item() if isinstance(ortho_pen, torch.Tensor) else float(ortho_pen),
            'commit': commit_scalar.item() if isinstance(commit_scalar, torch.Tensor) else float(commit_scalar),
            'contrast': contrast_scalar.item() if isinstance(contrast_scalar, torch.Tensor) else float(contrast_scalar),
            'jpen': jpen.item() if isinstance(jpen, torch.Tensor) else float(jpen)
        }
        return loss, metrics
