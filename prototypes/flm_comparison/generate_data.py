"""
Generate test data for R vs Python comparison.

Creates identical data that both implementations will use.
Uses LINEAR family for simplest comparison (constant Hessian weight).
"""

import numpy as np
import json
from pathlib import Path

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Configuration
N = 2000  # Sample size
d = 10    # Covariate dimension
M = 20    # Number of simulations for coverage

def generate_linear_data(n: int, d: int = 10, seed: int = None) -> dict:
    """Generate linear DGP with heterogeneous treatment effects."""
    if seed is not None:
        np.random.seed(seed)

    # Covariates: Uniform(-1, 1)
    X = np.random.uniform(-1, 1, (n, d))

    # True structural functions
    alpha_star = (
        np.sin(np.pi * X[:, 0]) +
        X[:, 1] ** 2 -
        np.cos(np.pi * X[:, 2])
    )

    beta_star = (
        0.5 +  # baseline
        0.3 * X[:, 0] +
        0.2 * np.tanh(X[:, 1]) -
        0.1 * X[:, 2]
    )

    # Treatment (confounded)
    T = 0.3 * beta_star + 0.2 * X[:, 3:6].sum(axis=1) + np.random.randn(n)

    # Outcome: Y = alpha + beta*T + epsilon
    Y = alpha_star + beta_star * T + np.random.randn(n)

    # True target
    mu_true = beta_star.mean()

    return {
        'X': X,
        'T': T,
        'Y': Y,
        'alpha_star': alpha_star,
        'beta_star': beta_star,
        'mu_true': mu_true,
    }


def main():
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Test Data for R vs Python Comparison")
    print("=" * 60)
    print(f"N = {N}, d = {d}, M = {M} simulations")
    print(f"Seed = {SEED}")
    print()

    # Generate M datasets for coverage estimation
    all_mu_true = []

    for sim_id in range(M):
        data = generate_linear_data(N, d, seed=SEED + sim_id)
        all_mu_true.append(data['mu_true'])

        # Save each simulation's data
        np.savetxt(output_dir / f'X_{sim_id}.csv', data['X'], delimiter=',')
        np.savetxt(output_dir / f'T_{sim_id}.csv', data['T'], delimiter=',')
        np.savetxt(output_dir / f'Y_{sim_id}.csv', data['Y'], delimiter=',')
        np.savetxt(output_dir / f'alpha_star_{sim_id}.csv', data['alpha_star'], delimiter=',')
        np.savetxt(output_dir / f'beta_star_{sim_id}.csv', data['beta_star'], delimiter=',')

    # Save metadata
    metadata = {
        'N': N,
        'd': d,
        'M': M,
        'seed': SEED,
        'mu_true': all_mu_true,
        'mu_true_mean': float(np.mean(all_mu_true)),
        'family': 'linear',
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {M} datasets")
    print(f"Mean mu_true = {np.mean(all_mu_true):.6f}")
    print(f"Saved to: {output_dir}")
    print()
    print("Files created:")
    print(f"  - X_{{0..{M-1}}}.csv")
    print(f"  - T_{{0..{M-1}}}.csv")
    print(f"  - Y_{{0..{M-1}}}.csv")
    print(f"  - alpha_star_{{0..{M-1}}}.csv")
    print(f"  - beta_star_{{0..{M-1}}}.csv")
    print(f"  - metadata.json")


if __name__ == '__main__':
    main()
