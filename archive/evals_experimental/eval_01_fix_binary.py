"""
================================================================================
EVAL 01 FIX: Experimental Fixes for Failing Binary Families
================================================================================

Problem: Logit, Probit, NegBin fail stricter eval criteria (RMSE < 0.15, Corr > 0.8)

This script tests various fixes:
1. Lower LR (0.005 vs 0.01)
2. Tighter gradient clipping (1.0 vs 10.0)
3. More epochs (400 vs 200)
4. Larger n (4000 vs 2000)
5. Smaller final layer init (gain=0.1)
6. Batch normalization
7. Deeper architecture [64, 64, 32]

Run: python3 -m evals.eval_01_fix_binary 2>&1 | tee evals/reports/eval_01_fix_binary.txt
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, norm
from scipy.special import expit
from typing import List, Optional, Callable
from dataclasses import dataclass, field

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference.families import get_family


# =============================================================================
# Thresholds (same as main eval)
# =============================================================================

RMSE_THRESHOLD = 0.15
CORR_THRESHOLD = 0.8
SEEDS = [42, 123, 456, 789, 999]  # 5 seeds for faster experiments


# =============================================================================
# DGP for failing families
# =============================================================================

FAILING_DGPS = {
    "logit": {
        "alpha": lambda x: 0.5 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.binomial(1, expit(alpha + beta * T)).astype(float),
    },
    "probit": {
        "alpha": lambda x: 0.5 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.binomial(1, norm.cdf(alpha + beta * T)).astype(float),
    },
    "negbin": {
        "alpha": lambda x: 0.5 + 0.2 * x,
        "beta": lambda x: 0.3 + 0.1 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: _generate_negbin(alpha, beta, T, rng, r=2.0),
        "family_kwargs": {"overdispersion": 0.5},
    },
}


def _generate_negbin(alpha, beta, T, rng, r=2.0):
    mu = np.exp(np.clip(alpha + beta * T, -10, 5))
    p = r / (r + mu)
    return rng.negative_binomial(r, p).astype(float)


def generate_dgp(family_name: str, n: int, seed: int):
    config = FAILING_DGPS[family_name]
    rng = np.random.default_rng(seed)

    X = rng.uniform(-2, 2, n)
    T = rng.normal(0, 1, n)

    alpha_true = config["alpha"](X)
    beta_true = config["beta"](X)
    Y = config["generate_y"](alpha_true, beta_true, T, rng)

    theta_true = np.column_stack([alpha_true, beta_true])
    return Y, T, X.reshape(-1, 1), theta_true


# =============================================================================
# Modified StructuralNet with BatchNorm option
# =============================================================================

class StructuralNetBN(nn.Module):
    """StructuralNet with optional BatchNorm and configurable init."""

    def __init__(
        self,
        input_dim: int,
        theta_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        use_batchnorm: bool = False,
        final_init_gain: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.final_layer = nn.Linear(prev_dim, theta_dim)
        self.network = nn.Sequential(*layers)

        # Initialize
        self._init_weights(final_init_gain)

    def _init_weights(self, final_gain: float):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small init for final layer (helps binary families)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=final_gain)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        return self.final_layer(self.network(x))


# =============================================================================
# Modified training with configurable options
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    lr: float = 0.01
    epochs: int = 200
    patience: int = 50
    grad_clip: float = 10.0
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    use_batchnorm: bool = False
    final_init_gain: float = 1.0
    n: int = 2000

    def __str__(self):
        parts = []
        if self.lr != 0.01:
            parts.append(f"lr={self.lr}")
        if self.epochs != 200:
            parts.append(f"ep={self.epochs}")
        if self.grad_clip != 10.0:
            parts.append(f"clip={self.grad_clip}")
        if self.hidden_dims != [64, 32]:
            parts.append(f"arch={self.hidden_dims}")
        if self.use_batchnorm:
            parts.append("bn")
        if self.final_init_gain != 1.0:
            parts.append(f"init={self.final_init_gain}")
        if self.n != 2000:
            parts.append(f"n={self.n}")
        return "+".join(parts) if parts else "baseline"


def train_with_config(
    Y, T, X, family, config: TrainingConfig, verbose: bool = False
):
    """Train with specified configuration."""
    n = X.shape[0]
    d_x = X.shape[1]
    theta_dim = family.theta_dim

    # Create network
    net = StructuralNetBN(
        input_dim=d_x,
        theta_dim=theta_dim,
        hidden_dims=config.hidden_dims,
        use_batchnorm=config.use_batchnorm,
        final_init_gain=config.final_init_gain,
    )

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Train/val split
    n_val = max(1, int(n * 0.1))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, T_train, Y_train = X_t[train_idx], T_t[train_idx], Y_t[train_idx]
    X_val, T_val, Y_val = X_t[val_idx], T_t[val_idx], Y_t[val_idx]

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        net.train()
        optimizer.zero_grad()

        theta = net(X_train)
        losses = family.loss(Y_train, T_train, theta)
        loss = losses.mean()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.grad_clip)

        optimizer.step()

        # Validation
        net.eval()
        with torch.no_grad():
            theta_val = net(X_val)
            val_losses = family.loss(Y_val, T_val, theta_val)
            val_loss = val_losses.mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        theta_hat = net(X_t)

    return theta_hat.numpy()


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(theta_hat, theta_true):
    metrics = {}

    for i, name in enumerate(["alpha", "beta"]):
        hat_i = theta_hat[:, i]
        true_i = theta_true[:, i]

        rmse = np.sqrt(np.mean((hat_i - true_i) ** 2))
        corr = pearsonr(hat_i, true_i)[0]

        metrics[f"rmse_{name}"] = rmse
        metrics[f"corr_{name}"] = corr

    # Check pass
    metrics["passed"] = (
        metrics["rmse_alpha"] < RMSE_THRESHOLD and
        metrics["rmse_beta"] < RMSE_THRESHOLD and
        metrics["corr_alpha"] > CORR_THRESHOLD and
        metrics["corr_beta"] > CORR_THRESHOLD
    )

    return metrics


def run_experiment(family_name: str, config: TrainingConfig, seeds: List[int]):
    """Run experiment for one family with one config across all seeds."""
    family_kwargs = FAILING_DGPS[family_name].get("family_kwargs", {})
    family = get_family(family_name, **family_kwargs)

    all_metrics = []
    for seed in seeds:
        Y, T, X, theta_true = generate_dgp(family_name, config.n, seed)
        theta_hat = train_with_config(Y, T, X, family, config)
        metrics = compute_metrics(theta_hat, theta_true)
        metrics["seed"] = seed
        all_metrics.append(metrics)

    # Aggregate
    n_pass = sum(1 for m in all_metrics if m["passed"])
    rmse_beta_mean = np.mean([m["rmse_beta"] for m in all_metrics])
    rmse_beta_max = max(m["rmse_beta"] for m in all_metrics)
    corr_beta_min = min(m["corr_beta"] for m in all_metrics)

    return {
        "n_pass": n_pass,
        "n_seeds": len(seeds),
        "rmse_beta_mean": rmse_beta_mean,
        "rmse_beta_max": rmse_beta_max,
        "corr_beta_min": corr_beta_min,
        "per_seed": all_metrics,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("EVAL 01 FIX: Experimental Fixes for Failing Binary Families")
    print("=" * 80)
    print(f"\nThresholds: RMSE < {RMSE_THRESHOLD}, Corr > {CORR_THRESHOLD}")
    print(f"Seeds: {SEEDS}")

    # Define configurations to test
    configs = {
        "baseline": TrainingConfig(),
        "lr=0.005": TrainingConfig(lr=0.005),
        "clip=1.0": TrainingConfig(grad_clip=1.0),
        "epochs=400": TrainingConfig(epochs=400, patience=100),
        "n=4000": TrainingConfig(n=4000),
        "init=0.1": TrainingConfig(final_init_gain=0.1),
        "batchnorm": TrainingConfig(use_batchnorm=True),
        "deeper": TrainingConfig(hidden_dims=[64, 64, 32]),
        "all_fixes": TrainingConfig(
            lr=0.005,
            grad_clip=1.0,
            epochs=400,
            patience=100,
            n=4000,
            final_init_gain=0.1,
            use_batchnorm=True,
            hidden_dims=[64, 64, 32],
        ),
    }

    families = ["logit", "probit", "negbin"]

    print("\n" + "-" * 80)
    print("RUNNING EXPERIMENTS")
    print("-" * 80)

    results = {}
    for family_name in families:
        results[family_name] = {}
        print(f"\n{family_name.upper()}")
        print("-" * 40)

        for config_name, config in configs.items():
            result = run_experiment(family_name, config, SEEDS)
            results[family_name][config_name] = result

            status = "PASS" if result["n_pass"] == len(SEEDS) else f"{result['n_pass']}/{len(SEEDS)}"
            print(f"  {config_name:<15} {status:<8} RMSE(β)={result['rmse_beta_mean']:.3f} (max={result['rmse_beta_max']:.3f}) Corr(β)min={result['corr_beta_min']:.3f}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<15}", end="")
    for family in families:
        print(f" {family:<12}", end="")
    print()
    print("-" * 55)

    for config_name in configs.keys():
        print(f"{config_name:<15}", end="")
        for family in families:
            r = results[family][config_name]
            status = f"{r['n_pass']}/{r['n_seeds']}"
            print(f" {status:<12}", end="")
        print()

    print("-" * 55)

    # Find best config per family
    print("\nBEST CONFIG PER FAMILY:")
    for family in families:
        best = max(results[family].items(), key=lambda x: x[1]["n_pass"])
        print(f"  {family}: {best[0]} ({best[1]['n_pass']}/{best[1]['n_seeds']} pass)")

    print("=" * 80)


if __name__ == "__main__":
    main()
