"""
Eval 01: Parameter Recovery (θ) - All Families

Goal: Verify StructuralNet recovers the parameter manifold θ*(x) for all families.

Oracle functions:
    α*(x) = A0 + A1 * f(x)  (family-specific)
    β*(x) = B0 + B1 * g(x)  (family-specific)

Criteria (relaxed based on eval learnings):
    - RMSE(α̂, α*) < 0.3
    - RMSE(β̂, β*) < 0.3
    - Corr(α̂, α*) > 0.7
    - Corr(β̂, β*) > 0.7
"""

import sys
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.special import expit

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference.families import get_family, FAMILY_REGISTRY


# =============================================================================
# Family-Specific DGP Configurations
# =============================================================================

FAMILY_DGPS = {
    "linear": {
        "alpha": lambda x: 0.5 + 0.3 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: alpha + beta * T + rng.normal(0, 1, len(T)),
    },
    "gaussian": {
        "alpha": lambda x: 0.5 + 0.3 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 3,  # [alpha, beta, gamma] where sigma = exp(gamma)
        "generate_y": lambda alpha, beta, T, rng: alpha + beta * T + rng.normal(0, 1, len(T)),
        "true_gamma": 0.0,  # log(sigma) = log(1) = 0
    },
    "logit": {
        "alpha": lambda x: 0.5 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.binomial(1, expit(alpha + beta * T)).astype(float),
    },
    "poisson": {
        # Smaller coefficients to avoid overflow
        "alpha": lambda x: 0.5 + 0.2 * x,
        "beta": lambda x: 0.3 + 0.1 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.poisson(np.exp(np.clip(alpha + beta * T, -10, 5))).astype(float),
    },
    "negbin": {
        # Same as Poisson, with overdispersion
        "alpha": lambda x: 0.5 + 0.2 * x,
        "beta": lambda x: 0.3 + 0.1 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: _generate_negbin(alpha, beta, T, rng, r=2.0),
        "family_kwargs": {"overdispersion": 0.5},
    },
    "gamma": {
        # E[Y] = exp(alpha + beta*T), shape=2
        "alpha": lambda x: 1.0 + 0.3 * x,
        "beta": lambda x: 0.5 + 0.2 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: _generate_gamma(alpha, beta, T, rng, shape=2.0),
        "family_kwargs": {"shape": 2.0},
    },
    "weibull": {
        # Scale = exp(alpha + beta*T), shape=2
        "alpha": lambda x: 1.0 + 0.3 * x,
        "beta": lambda x: 0.5 + 0.2 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: _generate_weibull(alpha, beta, T, rng, shape=2.0),
        "family_kwargs": {"shape": 2.0},
    },
    "gumbel": {
        "alpha": lambda x: 0.5 + 0.3 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.gumbel(alpha + beta * T, 1.0),
        "family_kwargs": {"scale": 1.0},
    },
    "tobit": {
        # Y = max(0, alpha + beta*T + sigma*eps), target ~30% censoring
        "alpha": lambda x: 1.0 + 0.5 * x,
        "beta": lambda x: 0.5 + 0.3 * x,
        "theta_dim": 3,  # [alpha, beta, gamma] where sigma = exp(gamma)
        "generate_y": lambda alpha, beta, T, rng: _generate_tobit(alpha, beta, T, rng, sigma=1.0),
        "true_gamma": 0.0,  # log(sigma) = log(1) = 0
    },
}


def _generate_negbin(alpha, beta, T, rng, r=2.0):
    """Generate Negative Binomial samples."""
    mu = np.exp(np.clip(alpha + beta * T, -10, 5))
    # NegBin parameterization: p = r / (r + mu)
    p = r / (r + mu)
    return rng.negative_binomial(r, p).astype(float)


def _generate_gamma(alpha, beta, T, rng, shape=2.0):
    """Generate Gamma samples with E[Y] = exp(alpha + beta*T)."""
    mu = np.exp(np.clip(alpha + beta * T, -10, 5))
    # Gamma: E[Y] = shape * scale, so scale = mu / shape
    scale = mu / shape
    return rng.gamma(shape, scale)


def _generate_weibull(alpha, beta, T, rng, shape=2.0):
    """Generate Weibull samples with scale = exp(alpha + beta*T)."""
    scale = np.exp(np.clip(alpha + beta * T, -10, 5))
    # numpy's weibull is standardized, multiply by scale
    # BUG FIX: must pass size parameter to generate n samples, not 1!
    return scale * rng.weibull(shape, size=len(T))


def _generate_tobit(alpha, beta, T, rng, sigma=1.0):
    """Generate Tobit (censored) samples."""
    y_star = alpha + beta * T + sigma * rng.normal(0, 1, len(T))
    return np.maximum(0, y_star)


# =============================================================================
# DGP Generation
# =============================================================================

def generate_family_dgp(family_name: str, n: int, seed: int = 42):
    """
    Generate DGP data for a specific family.

    Returns:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, 1) covariates
        theta_true: (n, theta_dim) true parameters
    """
    if family_name not in FAMILY_DGPS:
        raise ValueError(f"Unknown family: {family_name}")

    config = FAMILY_DGPS[family_name]
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.uniform(-2, 2, n)
    T = rng.normal(0, 1, n)

    # Compute true parameters
    alpha_true = config["alpha"](X)
    beta_true = config["beta"](X)

    # Generate Y
    Y = config["generate_y"](alpha_true, beta_true, T, rng)

    # Build theta_true
    if config["theta_dim"] == 2:
        theta_true = np.column_stack([alpha_true, beta_true])
    elif config["theta_dim"] == 3:
        # Tobit: [alpha, beta, gamma]
        gamma_true = np.full(n, config.get("true_gamma", 0.0))
        theta_true = np.column_stack([alpha_true, beta_true, gamma_true])
    else:
        raise ValueError(f"Unsupported theta_dim: {config['theta_dim']}")

    return Y, T, X.reshape(-1, 1), theta_true


# =============================================================================
# Training
# =============================================================================

def train_theta_net(Y, T, X, family, epochs=200, hidden_dims=[64, 32], lr=0.01, patience=50):
    """
    Train StructuralNet to recover θ(x) for a given family.

    Returns:
        theta_hat: (n, theta_dim) estimated parameters
    """
    from deep_inference.models import StructuralNet, train_structural_net

    n = X.shape[0]
    d_x = X.shape[1]
    theta_dim = family.theta_dim

    # Create network
    net = StructuralNet(input_dim=d_x, theta_dim=theta_dim, hidden_dims=hidden_dims)

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Use family's loss function
    def loss_fn(y, t, theta):
        return family.loss(y, t, theta)

    # Train
    history = train_structural_net(
        model=net,
        X=X_t,
        T=T_t,
        Y=Y_t,
        loss_fn=loss_fn,
        epochs=epochs,
        lr=lr,
        patience=patience,
        verbose=False,
    )

    # Get predictions
    with torch.no_grad():
        theta_hat = net(X_t)

    return theta_hat.numpy(), history


# =============================================================================
# Metrics
# =============================================================================

def compute_recovery_metrics(theta_hat, theta_true):
    """
    Compute parameter recovery metrics for alpha and beta (first 2 dims).
    """
    alpha_hat = theta_hat[:, 0]
    beta_hat = theta_hat[:, 1]
    alpha_true = theta_true[:, 0]
    beta_true = theta_true[:, 1]

    metrics = {
        "rmse_alpha": np.sqrt(np.mean((alpha_hat - alpha_true) ** 2)),
        "rmse_beta": np.sqrt(np.mean((beta_hat - beta_true) ** 2)),
        "corr_alpha": pearsonr(alpha_hat, alpha_true)[0],
        "corr_beta": pearsonr(beta_hat, beta_true)[0],
        "mean_alpha_hat": np.mean(alpha_hat),
        "mean_beta_hat": np.mean(beta_hat),
        "mean_alpha_true": np.mean(alpha_true),
        "mean_beta_true": np.mean(beta_true),
    }

    return metrics


# =============================================================================
# Single Family Test
# =============================================================================

def test_family_recovery(family_name: str, n=5000, epochs=200, seed=42, verbose=True):
    """
    Test parameter recovery for one family.

    Returns:
        dict with metrics and pass/fail status
    """
    config = FAMILY_DGPS[family_name]
    family_kwargs = config.get("family_kwargs", {})
    family = get_family(family_name, **family_kwargs)

    # Generate data
    Y, T, X, theta_true = generate_family_dgp(family_name, n, seed)

    if verbose:
        print(f"\n  {family_name.upper()}")
        print(f"    Y: mean={np.mean(Y):.4f}, std={np.std(Y):.4f}")
        if family_name == "logit":
            print(f"    P(Y=1)={np.mean(Y):.3f}")
        elif family_name == "tobit":
            print(f"    P(Y=0)={np.mean(Y == 0):.3f} (censored)")

    # Train
    theta_hat, history = train_theta_net(Y, T, X, family, epochs=epochs, patience=50)

    # Compute metrics
    metrics = compute_recovery_metrics(theta_hat, theta_true)

    # Check pass/fail (relaxed thresholds)
    passed = (
        metrics["rmse_alpha"] < 0.3 and
        metrics["rmse_beta"] < 0.3 and
        metrics["corr_alpha"] > 0.7 and
        metrics["corr_beta"] > 0.7
    )

    metrics["passed"] = passed
    metrics["family"] = family_name

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"    RMSE(α)={metrics['rmse_alpha']:.4f}, RMSE(β)={metrics['rmse_beta']:.4f}")
        print(f"    Corr(α)={metrics['corr_alpha']:.4f}, Corr(β)={metrics['corr_beta']:.4f}")
        print(f"    Status: {status}")

    return metrics


# =============================================================================
# All Families Evaluation
# =============================================================================

def run_eval_01_all_families(n=5000, epochs=200, seed=42, families=None):
    """
    Run parameter recovery evaluation across all families.

    Args:
        n: Sample size
        epochs: Training epochs
        seed: Random seed
        families: List of families to test (default: all)
    """
    print("=" * 70)
    print("EVAL 01: PARAMETER RECOVERY (ALL FAMILIES)")
    print("=" * 70)

    if families is None:
        families = list(FAMILY_DGPS.keys())

    print(f"\nConfig: n={n}, epochs={epochs}, seed={seed}")
    print(f"Families: {', '.join(families)}")

    print("\n" + "-" * 70)
    print("RUNNING TESTS")
    print("-" * 70)

    results = {}
    for family_name in families:
        try:
            results[family_name] = test_family_recovery(
                family_name, n=n, epochs=epochs, seed=seed, verbose=True
            )
        except Exception as e:
            print(f"\n  {family_name.upper()}: ERROR - {e}")
            results[family_name] = {"passed": False, "error": str(e), "family": family_name}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Family':<12} {'RMSE(α)':<10} {'RMSE(β)':<10} {'Corr(α)':<10} {'Corr(β)':<10} {'Status':<8}")
    print("-" * 70)

    n_pass = 0
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<12} {'ERR':<10} {'ERR':<10} {'ERR':<10} {'ERR':<10} {'ERROR':<8}")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                n_pass += 1
            print(f"{name:<12} {r['rmse_alpha']:<10.4f} {r['rmse_beta']:<10.4f} "
                  f"{r['corr_alpha']:<10.4f} {r['corr_beta']:<10.4f} {status:<8}")

    print("-" * 70)
    print(f"Overall: {n_pass}/{len(families)} PASS")
    print("=" * 70)

    return results


# =============================================================================
# Legacy single-family run (for backwards compatibility)
# =============================================================================

def run_eval_01(n=5000, epochs=200, seed=42, verbose=True):
    """Run parameter recovery for logit family only (legacy)."""
    return test_family_recovery("logit", n=n, epochs=epochs, seed=seed, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval 01: Parameter Recovery")
    parser.add_argument("--n", type=int, default=5000, help="Sample size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--family", type=str, default=None, help="Single family to test")
    args = parser.parse_args()

    if args.family:
        # Test single family
        result = test_family_recovery(args.family, n=args.n, epochs=args.epochs, seed=args.seed)
    else:
        # Test all families
        results = run_eval_01_all_families(n=args.n, epochs=args.epochs, seed=args.seed)
