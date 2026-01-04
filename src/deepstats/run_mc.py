#!/usr/bin/env python
"""Monte Carlo validation for influence function inference.

Compares naive, influence, and bootstrap methods across 8 models.

Usage:
    python run_mc.py --M 50 --N 1000 --models linear poisson --methods naive influence
    python run_mc.py --M 10 --N 500 --models linear --methods naive influence bootstrap
"""

import argparse
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from dgp import get_dgp, DGPS
from families import get_family, FAMILIES
from inference import naive, influence, bootstrap, METHODS
from metrics import compute_metrics, print_table


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    """Simulation configuration."""
    M: int = 50                   # Monte Carlo simulations
    N: int = 1000                 # Sample size
    n_folds: int = 50             # Cross-fitting folds (98% train)
    epochs: int = 100             # Training epochs
    lr: float = 0.01              # Learning rate
    batch_size: int = 64          # Batch size
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1          # Dropout
    weight_decay: float = 1e-4    # L2 regularization
    seed: int = 42


# =============================================================================
# Run One Simulation
# =============================================================================

def run_one_sim(
    sim_id: int,
    model_name: str,
    method_names: List[str],
    config: Config,
) -> List[dict]:
    """Run one MC simulation for a model."""
    dgp = get_dgp(model_name, seed=config.seed + sim_id * 1000)
    family = get_family(model_name)
    data = dgp.generate(config.N)
    mu_true = data.mu_true

    method_funcs = {"naive": naive, "influence": influence, "bootstrap": bootstrap}
    results = []

    for method_name in method_names:
        try:
            mu_hat, se = method_funcs[method_name](
                data.X, data.T, data.Y, family, config
            )
            bias = mu_hat - mu_true
            ci_lo = mu_hat - 1.96 * se
            ci_hi = mu_hat + 1.96 * se
            covered = ci_lo <= mu_true <= ci_hi

            results.append({
                "sim_id": sim_id,
                "model": model_name,
                "method": method_name,
                "mu_hat": mu_hat,
                "se": se,
                "mu_true": mu_true,
                "bias": bias,
                "covered": covered,
            })
        except Exception as e:
            print(f"Error in {model_name}/{method_name} sim {sim_id}: {e}")
            results.append({
                "sim_id": sim_id,
                "model": model_name,
                "method": method_name,
                "mu_hat": np.nan,
                "se": np.nan,
                "mu_true": mu_true,
                "bias": np.nan,
                "covered": False,
            })

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo validation for influence function inference"
    )
    parser.add_argument("--M", type=int, default=50, help="Number of MC sims")
    parser.add_argument("--N", type=int, default=1000, help="Sample size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-folds", type=int, default=50, help="Cross-fitting folds")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "poisson", "logit"],
        choices=list(DGPS.keys()),
        help="Models to run",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["naive", "influence"],
        choices=list(METHODS.keys()),
        help="Inference methods",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="mc_results.csv", help="Output CSV")

    args = parser.parse_args()

    config = Config(
        M=args.M,
        N=args.N,
        epochs=args.epochs,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    print("=" * 60)
    print("Influence Function Validation")
    print("=" * 60)
    print(f"M={config.M}, N={config.N}, epochs={config.epochs}, folds={config.n_folds}")
    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print("=" * 60)

    all_results = []

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        dgp = get_dgp(model_name, seed=config.seed)
        mu_true = dgp.compute_true_mu()
        print(f"True μ* = {mu_true:.6f}")

        for sim_id in tqdm(range(config.M), desc=model_name):
            results = run_one_sim(sim_id, model_name, args.methods, config)
            all_results.extend(results)

    # Save raw results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nRaw results saved to: {args.output}")

    # Compute and print metrics
    metrics = compute_metrics(df)
    metrics.to_csv(args.output.replace(".csv", ".metrics.csv"), index=False)
    print(f"Metrics saved to: {args.output.replace('.csv', '.metrics.csv')}")

    print("\n")
    print_table(metrics)


if __name__ == "__main__":
    main()
