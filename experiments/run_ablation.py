"""
run_ablation.py

Launches an ablation grid over VC‑MOJI components and writes results to CSV.

Assumes train.py exposes a callable function:

    from train import train_experiment

Where:
    metrics = train_experiment(config: dict) -> dict

The returned dict MUST contain at least:
    {
        "train_loss": float,
        "val_loss": float,
        "val_metric": float  # e.g. accuracy or perplexity
    }

If your train.py does not yet expose this function,
refactor it accordingly (CLI can wrap this function).
"""

import os
import csv
import itertools
import argparse
import json
from datetime import datetime

import torch

# ===== EXPECTED TRAIN ENTRYPOINT =====
try:
    from train import train_experiment
except ImportError as e:
    raise ImportError(
        "train.py must define train_experiment(config: dict) -> dict"
    ) from e


# ==========================================================
# Ablation grid definition
# ==========================================================

def build_grid():
    """
    Returns a list of configuration dictionaries.
    Modify this to expand/reduce the search space.
    """

    base_config = {
        "epochs": 3,
        "batch_size": 64,
        "n_samples": 6000,
        "dim": 256,
        "n_layers": 3,
        "n_heads": 8,
        "window_size": 16,
        "latent_dim": 64,
        "kl_weight": 1e-3,
        "ortho_weight": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    grid_space = {
        # Core architectural toggles
        "use_njx": [False, True],
        "use_variational": [False, True],
        "use_quantizer": [False, True],
        "use_channel_jitter": [False, True],
        "use_ecklock": [False, True],
        "use_jacobian_reg": [False, True],

        # Quantizer params
        "num_codebooks": [7],
        "codebook_size": [256],

        # Channel jitter params
        "cj_k_channels": [0, 36],
        "cj_jitter_std": [0.02],

        # ECK params
        "eck_key_dim": [48],
        "eck_topk": [0, 16],

        # RCE params
        "rce_eps": [0.0, 0.1],
        "rce_kernel_scale": [0.5],

        # Jacobian
        "jreg_weight": [0.0, 1e-4],
        "jreg_iters": [3],
    }

    keys = list(grid_space.keys())
    values = list(grid_space.values())

    configs = []
    for combination in itertools.product(*values):
        cfg = base_config.copy()
        for k, v in zip(keys, combination):
            cfg[k] = v
        configs.append(cfg)

    return configs


# ==========================================================
# Runner
# ==========================================================

def run_ablation(output_csv: str, max_runs: int = None):
    configs = build_grid()

    if max_runs is not None:
        configs = configs[:max_runs]

    print(f"Total runs: {len(configs)}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = None

    with open(output_csv, mode="w", newline="") as f:
        writer = None

        for run_id, config in enumerate(configs):
            print("=" * 80)
            print(f"Run {run_id + 1}/{len(configs)}")
            print(json.dumps(config, indent=2))

            try:
                metrics = train_experiment(config)
            except Exception as e:
                print(f"Run failed with error: {e}")
                metrics = {
                    "train_loss": float("nan"),
                    "val_loss": float("nan"),
                    "val_metric": float("nan"),
                }

            row = {**config, **metrics}

            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(row)
            f.flush()

    print(f"Ablation finished. Results written to: {output_csv}")


# ==========================================================
# CLI
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=f"experiments/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Limit number of runs (for debugging)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args.output, args.max_runs)
