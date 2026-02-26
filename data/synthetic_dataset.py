import torch
from torch.utils.data import Dataset

class SyntheticSeqDataset(Dataset):
    """
    Creates simple synthetic sequences for next-token prediction:
    - vocabulary of size V
    - sequences are random but with embedded deterministic motifs to allow learning
    """

    def __init__(self, n_samples=20000, seq_len=128, vocab_size=256, motifs=5, seed=0):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = torch.Generator().manual_seed(seed)
        # Pre-generate samples with occasional motifs
        self.data = torch.randint(1, vocab_size, (n_samples, seq_len), generator=rng)
        # insert deterministic motifs: repeated patterns
        for i in range(n_samples):
            for m in range(motifs):
                start = (i * (m+1) + m*7) % (seq_len - 8)
                motif = torch.tensor([(m+1)*3 + j for j in range(4)]) % vocab_size
                self.data[i, start:start+4] = motif

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1].long(), seq[1:].long()  # (input, target)
