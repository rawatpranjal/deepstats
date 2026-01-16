#!/usr/bin/env python3
"""Generate plot images for Quick Start documentation."""

import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_inference import structural_dml

def main():
    # Same setup as Quick Start
    np.random.seed(42)
    torch.manual_seed(42)
    n = 2000
    X = np.random.randn(n, 5)
    T = np.random.randn(n)

    # Heterogeneous treatment effect: β(X) = 0.5 + 0.3*X₁
    alpha = 0.2 * X[:, 0]
    beta = 0.5 + 0.3 * X[:, 1]
    prob = 1 / (1 + np.exp(-(alpha + beta * T)))
    Y = np.random.binomial(1, prob).astype(float)

    print("Running structural_dml (this may take a moment)...")

    # Run with store_data=True for prediction methods
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='logit',
        hidden_dims=[64, 32],
        epochs=100,
        n_folds=50,
        store_data=True
    )

    print(result.summary())

    # Ensure _static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), '_static')
    os.makedirs(static_dir, exist_ok=True)

    # Generate and save plots
    dist_path = os.path.join(static_dir, 'quickstart_distributions.png')
    hetero_path = os.path.join(static_dir, 'quickstart_heterogeneity.png')

    print(f"\nGenerating distribution plot: {dist_path}")
    result.plot_distributions(save_path=dist_path)

    print(f"Generating heterogeneity plot: {hetero_path}")
    result.plot_heterogeneity(feature_idx=1, save_path=hetero_path)

    print("\nDone! Generated plots:")
    print(f"  - {dist_path}")
    print(f"  - {hetero_path}")

if __name__ == '__main__':
    main()
