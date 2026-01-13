"""
Eval 01: Parameter Recovery (θ)

Goal: Verify StructuralNet recovers the parameter manifold θ*(x).

Oracle:
    α*(x) = 0.5 * sin(x)
    β*(x) = 1.0 + 0.5 * x

Criteria:
    - RMSE(α̂, α*) < 0.2
    - RMSE(β̂, β*) < 0.1
    - Corr(α̂, α*) > 0.8
    - Corr(β̂, β*) > 0.9
"""

import sys
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import generate_canonical_dgp, CanonicalDGP


def train_theta_net(Y, T, X, epochs=200, hidden_dims=[64, 32], lr=0.01, verbose=False):
    """
    Train StructuralNet to recover θ(x) = [α(x), β(x)].

    Returns:
        theta_hat: (n, 2) estimated parameters
    """
    from deep_inference.models import StructuralNet, train_structural_net

    n = X.shape[0]
    d_x = X.shape[1]

    # Create network
    net = StructuralNet(input_dim=d_x, theta_dim=2, hidden_dims=hidden_dims)

    # Define logistic loss
    def loss_fn(y, t, theta):
        """Batched logistic loss."""
        alpha = theta[:, 0]
        beta = theta[:, 1]
        logits = alpha + beta * t
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Train
    history = train_structural_net(
        model=net,
        X=X,
        T=T,
        Y=Y,
        loss_fn=loss_fn,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )

    # Get predictions
    with torch.no_grad():
        theta_hat = net(X)

    return theta_hat.numpy(), history


def compute_recovery_metrics(theta_hat, theta_true):
    """
    Compute parameter recovery metrics.

    Returns dict with:
        - rmse_alpha, rmse_beta
        - corr_alpha, corr_beta
        - mean_alpha, mean_beta (hat)
        - true_mean_alpha, true_mean_beta
    """
    alpha_hat = theta_hat[:, 0]
    beta_hat = theta_hat[:, 1]
    alpha_true = theta_true[:, 0]
    beta_true = theta_true[:, 1]

    metrics = {
        # RMSE
        "rmse_alpha": np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)),
        "rmse_beta": np.sqrt(np.mean((beta_hat - beta_true) ** 2)),
        # Correlation
        "corr_alpha": pearsonr(alpha_hat, alpha_true)[0],
        "corr_beta": pearsonr(beta_hat, beta_true)[0],
        # Means
        "mean_alpha_hat": np.mean(alpha_hat),
        "mean_beta_hat": np.mean(beta_hat),
        "mean_alpha_true": np.mean(alpha_true),
        "mean_beta_true": np.mean(beta_true),
        # Std
        "std_alpha_hat": np.std(alpha_hat),
        "std_beta_hat": np.std(beta_hat),
        "std_alpha_true": np.std(alpha_true),
        "std_beta_true": np.std(beta_true),
    }

    return metrics


def run_eval_01(n=5000, epochs=200, seed=42, verbose=True):
    """
    Run parameter recovery evaluation.

    Args:
        n: Sample size (large for clean signal)
        epochs: Training epochs
        seed: Random seed
        verbose: Print progress
    """
    print("=" * 60)
    print("EVAL 01: PARAMETER RECOVERY")
    print("=" * 60)

    dgp = CanonicalDGP()
    print(f"\nDGP:")
    print(f"  α*(x) = {dgp.A0} + {dgp.A1}·sin(x)")
    print(f"  β*(x) = {dgp.B0} + {dgp.B1}·x")
    print(f"  X ~ Uniform({dgp.X_low}, {dgp.X_high})")

    # Generate data
    print(f"\nGenerating data (n={n}, seed={seed})...")
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)

    print(f"  Y: mean={Y.mean():.4f}, P(Y=1)={Y.mean():.3f}")
    print(f"  T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"  X: range=[{X.min():.2f}, {X.max():.2f}]")

    # Train network
    print(f"\nTraining StructuralNet (epochs={epochs})...")
    theta_hat, history = train_theta_net(Y, T, X, epochs=epochs, verbose=False)

    # Compute metrics
    metrics = compute_recovery_metrics(theta_hat, theta_true.numpy())

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Recovery Metrics ---")
    print(f"{'Param':<10} {'RMSE':<10} {'Corr':<10} {'Mean(hat)':<12} {'Mean(true)':<12}")
    print("-" * 54)
    print(f"{'α':<10} {metrics['rmse_alpha']:<10.4f} {metrics['corr_alpha']:<10.4f} "
          f"{metrics['mean_alpha_hat']:<12.4f} {metrics['mean_alpha_true']:<12.4f}")
    print(f"{'β':<10} {metrics['rmse_beta']:<10.4f} {metrics['corr_beta']:<10.4f} "
          f"{metrics['mean_beta_hat']:<12.4f} {metrics['mean_beta_true']:<12.4f}")

    print("\n--- Std Deviation ---")
    print(f"{'Param':<10} {'Std(hat)':<12} {'Std(true)':<12}")
    print("-" * 34)
    print(f"{'α':<10} {metrics['std_alpha_hat']:<12.4f} {metrics['std_alpha_true']:<12.4f}")
    print(f"{'β':<10} {metrics['std_beta_hat']:<12.4f} {metrics['std_beta_true']:<12.4f}")

    # Training quality
    print("\n--- Training Quality ---")
    if history.val_losses:
        print(f"  Final val loss: {history.val_losses[-1]:.4f}")
        print(f"  Best val loss: {min(history.val_losses):.4f}")
        print(f"  Best epoch: {np.argmin(history.val_losses) + 1}")
    print(f"  Final train loss: {history.train_losses[-1]:.4f}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "RMSE(α) < 0.2": metrics["rmse_alpha"] < 0.2,
        "RMSE(β) < 0.1": metrics["rmse_beta"] < 0.1,
        "Corr(α) > 0.8": metrics["corr_alpha"] > 0.8,
        "Corr(β) > 0.9": metrics["corr_beta"] > 0.9,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 01: PASS")
    else:
        print("EVAL 01: FAIL")
    print("=" * 60)

    return {
        "metrics": metrics,
        "theta_hat": theta_hat,
        "theta_true": theta_true.numpy(),
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_01(n=5000, epochs=200, seed=42)
