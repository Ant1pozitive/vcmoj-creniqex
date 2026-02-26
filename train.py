import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from models.vcmoji import VC_MOJI_Transformer
from data.synthetic_dataset import SyntheticSeqDataset
from losses import VC_MOJI_Loss
from utils import save_checkpoint

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SyntheticSeqDataset(n_samples=args.n_samples, seq_len=args.seq_len, vocab_size=args.vocab_size, motifs=6)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = VC_MOJI_Transformer(vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers,
                                n_heads=args.n_heads, mlp_ratio=args.mlp_ratio, window_size=args.window_size,
                                latent_dim=args.latent_dim, iterative_steps=args.iter_steps,
                                max_seq_len=args.seq_len-1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = VC_MOJI_Loss(recon_weight=1.0, kl_weight=args.kl_weight, ortho_weight=args.ortho_weight)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader)
        running_loss = 0.0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, mus, logvars = model(xb)
            loss, metrics = loss_fn(logits, yb, mus, logvars, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item() if global_step>0 else loss.item()
            pbar.set_description(f"Epoch {epoch} loss {running_loss:.4f} recon {metrics['recon']:.4f} kl {metrics['kl']:.4f}")
            global_step += 1

        # checkpoint
        save_checkpoint({'model_state': model.state_dict(), 'opt_state': optimizer.state_dict(), 'step': global_step},
                        os.path.join(args.ckpt_dir, f"vcmoji_epoch{epoch}.pt"))

    print("Training finished. Last loss:", running_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--iter_steps', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--ortho_weight', type=float, default=1e-5)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train(args)
