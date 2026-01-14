"""
================================================================================
EVAL 01: PARAMETER RECOVERY (θ)
================================================================================

WHAT THIS TESTS:
    Can StructuralNet learn θ(x) = [α(x), β(x)] from data?

    We generate data from a known DGP with true parameters θ*(x), train a
    neural network to predict θ̂(x), and check if θ̂ ≈ θ*.

WHY IT MATTERS:
    This is the FIRST STAGE of influence function inference. If the network
    can't learn θ(x), nothing downstream will work. This test catches:
    - Broken loss functions
    - Training instability
    - Family implementation bugs

WHY IT'S NOT SUFFICIENT:
    Thanks to Neyman orthogonality, you can have:
    - Imperfect θ̂ → correct influence function → GOOD inference
    - Perfect θ̂ → wrong Λ estimation → BAD inference

    The coverage test (eval_06) is the real validation. This is just a
    prerequisite sanity check.

BINARY FAMILIES (logit, probit):
    Binary outcomes carry ~1 bit of information per observation, making
    first-stage recovery fundamentally harder than continuous outcomes.

    We auto-scale sample size: n=4000 for logit/probit, n=2000 for others.
    This is not weakening the test—it's providing the sample size the
    statistical problem requires.

MULTI-SEED VALIDATION:
    Neural net training is stochastic. One seed proves nothing.
    We run 10 seeds and report mean ± std + worst-case metrics.

    Pass criteria:
    - PASS: ALL seeds pass individual thresholds
    - UNSTABLE: 60%+ seeds pass
    - FAIL: <60% seeds pass

--------------------------------------------------------------------------------

Oracle functions:
    α*(x) = A0 + A1 * f(x)  (family-specific)
    β*(x) = B0 + B1 * g(x)  (family-specific)

Thresholds (theory-aligned, Assumption 5: o(n^{-1/4})):
    - RMSE(α̂, α*) < 0.15
    - RMSE(β̂, β*) < 0.15
    - Corr(α̂, α*) > 0.8
    - Corr(β̂, β*) > 0.8

Sample sizes:
    - Binary families (logit, probit): n=8000 (2x default due to ~1 bit/obs)
    - All other families: n=5000
"""

import sys
import numpy as np
import torch
from scipy.stats import pearsonr, norm
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
    "probit": {
        # P(Y=1) = Phi(alpha + beta*T), same as logit but normal CDF
        "alpha": lambda x: 0.5 * np.sin(x),
        "beta": lambda x: 1.0 + 0.5 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.binomial(1, norm.cdf(alpha + beta * T)).astype(float),
    },
    "beta": {
        # Y ~ Beta(mu*phi, (1-mu)*phi), mu = sigmoid(alpha + beta*T)
        "alpha": lambda x: 0.5 + 0.3 * x,
        "beta": lambda x: 0.3 + 0.2 * x,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: _generate_beta(alpha, beta, T, rng, phi=5.0),
        "family_kwargs": {"precision": 5.0},
    },
    "zip": {
        # Zero-Inflated Poisson: mixture of zeros and Poisson
        "alpha": lambda x: 0.5 + 0.2 * x,
        "beta": lambda x: 0.3 + 0.1 * x,
        "theta_dim": 4,  # [alpha, beta, gamma, delta]
        "generate_y": lambda alpha, beta, T, rng: _generate_zip(alpha, beta, T, rng, gamma=0.0, delta=0.0),
        "true_gamma": 0.0,  # Fixed zero-inflation intercept
        "true_delta": 0.0,  # Fixed zero-inflation slope
    },
}


# =============================================================================
# Challenging DGPs (high-dim X, interactions, nonlinear)
# =============================================================================

CHALLENGING_DGPS = {
    "logit_highdim": {
        # d_X = 10, interactions, higher frequency sinusoids
        "d_x": 10,
        "alpha": lambda X: 0.5 + 0.3 * np.sin(2*np.pi*X[:, 0]) + 0.2 * X[:, 1] * X[:, 2],
        "beta": lambda X: 1.0 + 0.5 * X[:, 0] + 0.3 * np.sin(3*np.pi*X[:, 1]) + 0.2 * X[:, 2]**2,
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: rng.binomial(1, expit(alpha + beta * T)).astype(float),
    },
    "linear_highdim": {
        # d_X = 10, interactions, polynomial terms
        "d_x": 10,
        "alpha": lambda X: 0.5 + 0.3 * X[:, 0] * X[:, 1] + 0.2 * np.sin(2*np.pi*X[:, 2]),
        "beta": lambda X: 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]**2 - 0.2 * X[:, 2] * X[:, 3],
        "theta_dim": 2,
        "generate_y": lambda alpha, beta, T, rng: alpha + beta * T + rng.normal(0, 1, len(T)),
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


def _generate_beta(alpha, beta, T, rng, phi=5.0):
    """Generate Beta samples with mu = sigmoid(alpha + beta*T)."""
    eta = alpha + beta * T
    mu = expit(eta)
    # Clamp mu away from boundaries
    mu = np.clip(mu, 0.01, 0.99)
    a = mu * phi
    b = (1 - mu) * phi
    return rng.beta(a, b)


def _generate_zip(alpha, beta, T, rng, gamma=0.0, delta=0.0):
    """Generate Zero-Inflated Poisson samples.

    Args:
        alpha, beta: Poisson rate parameters, log(lambda) = alpha + beta*T
        gamma, delta: Zero-inflation parameters, pi = sigmoid(gamma + delta*T)
    """
    n = len(T)
    # Poisson rate
    lam = np.exp(np.clip(alpha + beta * T, -10, 5))
    # Zero-inflation probability (fixed for simplicity in DGP)
    pi = expit(gamma + delta * T)

    # Generate mixture
    # 1. Draw from structural zero vs Poisson
    is_structural_zero = rng.binomial(1, pi).astype(bool)
    # 2. Draw Poisson for non-structural-zero observations
    Y = np.zeros(n)
    Y[~is_structural_zero] = rng.poisson(lam[~is_structural_zero])
    return Y.astype(float)


# =============================================================================
# DGP Generation
# =============================================================================

def generate_family_dgp(family_name: str, n: int, seed: int = 42):
    """
    Generate DGP data for a specific family.

    Returns:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        theta_true: (n, theta_dim) true parameters
    """
    # Check both standard and challenging DGPs
    if family_name in FAMILY_DGPS:
        config = FAMILY_DGPS[family_name]
        d_x = 1
    elif family_name in CHALLENGING_DGPS:
        config = CHALLENGING_DGPS[family_name]
        d_x = config.get("d_x", 1)
    else:
        raise ValueError(f"Unknown family: {family_name}. Available: {list(FAMILY_DGPS.keys()) + list(CHALLENGING_DGPS.keys())}")

    rng = np.random.default_rng(seed)

    # Generate covariates
    if d_x == 1:
        X = rng.uniform(-2, 2, n)
        alpha_true = config["alpha"](X)
        beta_true = config["beta"](X)
        X = X.reshape(-1, 1)
    else:
        X = rng.uniform(-2, 2, (n, d_x))
        alpha_true = config["alpha"](X)
        beta_true = config["beta"](X)

    T = rng.normal(0, 1, n)

    # Generate Y
    Y = config["generate_y"](alpha_true, beta_true, T, rng)

    # Build theta_true
    if config["theta_dim"] == 2:
        theta_true = np.column_stack([alpha_true, beta_true])
    elif config["theta_dim"] == 3:
        # Tobit/Gaussian: [alpha, beta, gamma]
        gamma_true = np.full(n, config.get("true_gamma", 0.0))
        theta_true = np.column_stack([alpha_true, beta_true, gamma_true])
    elif config["theta_dim"] == 4:
        # ZIP: [alpha, beta, gamma, delta]
        gamma_true = np.full(n, config.get("true_gamma", 0.0))
        delta_true = np.full(n, config.get("true_delta", 0.0))
        theta_true = np.column_stack([alpha_true, beta_true, gamma_true, delta_true])
    else:
        raise ValueError(f"Unsupported theta_dim: {config['theta_dim']}")

    return Y, T, X, theta_true


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
    Compute metrics dynamically for all parameter dimensions.

    Mapping:
        idx 0 -> alpha (varies with X)
        idx 1 -> beta (varies with X)
        idx 2 -> gamma (aux 1, often constant)
        idx 3 -> delta (aux 2, often constant)

    For varying params: check RMSE + Correlation
    For constant params: check RMSE only (correlation undefined)
    """
    metrics = {}
    param_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    n_params = theta_true.shape[1]

    for i in range(n_params):
        name = param_names[i] if i < len(param_names) else f"p{i}"

        hat_i = theta_hat[:, i]
        true_i = theta_true[:, i]

        # 1. Bias/Accuracy (RMSE) - Always applicable
        rmse = np.sqrt(np.mean((hat_i - true_i) ** 2))
        metrics[f"rmse_{name}"] = rmse
        metrics[f"mean_{name}_hat"] = np.mean(hat_i)
        metrics[f"mean_{name}_true"] = np.mean(true_i)

        # 2. Shape (Correlation) - Only applicable if truth varies
        true_std = np.std(true_i)
        if true_std > 1e-6:
            corr = pearsonr(hat_i, true_i)[0]
            metrics[f"corr_{name}"] = corr
            metrics[f"is_constant_{name}"] = False
        else:
            # If truth is constant, correlation is meaningless
            metrics[f"corr_{name}"] = np.nan
            metrics[f"is_constant_{name}"] = True

    metrics["n_params"] = n_params
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
    # Handle both standard and challenging DGPs
    if family_name in FAMILY_DGPS:
        config = FAMILY_DGPS[family_name]
        base_family = family_name
    elif family_name in CHALLENGING_DGPS:
        config = CHALLENGING_DGPS[family_name]
        # Extract base family from name (e.g., "logit_highdim" -> "logit")
        base_family = family_name.split("_")[0]
    else:
        raise ValueError(f"Unknown family: {family_name}")

    family_kwargs = config.get("family_kwargs", {})
    family = get_family(base_family, **family_kwargs)

    # Generate data
    Y, T, X, theta_true = generate_family_dgp(family_name, n, seed)

    if verbose:
        print(f"\n  {family_name.upper()}")
        print(f"    Y: mean={np.mean(Y):.4f}, std={np.std(Y):.4f}")
        if family_name in ("logit", "probit"):
            print(f"    P(Y=1)={np.mean(Y):.3f}")
        elif family_name == "tobit":
            print(f"    P(Y=0)={np.mean(Y == 0):.3f} (censored)")
        elif family_name == "zip":
            print(f"    P(Y=0)={np.mean(Y == 0):.3f} (zeros)")

    # Train
    theta_hat, history = train_theta_net(Y, T, X, family, epochs=epochs, patience=50)

    # Compute metrics
    metrics = compute_recovery_metrics(theta_hat, theta_true)

    # 1. Primary Parameters (Alpha/Beta) - theory-aligned thresholds
    alpha_pass = (metrics["rmse_alpha"] < RMSE_THRESHOLD) and (metrics["corr_alpha"] > CORR_THRESHOLD)
    beta_pass = (metrics["rmse_beta"] < RMSE_THRESHOLD) and (metrics["corr_beta"] > CORR_THRESHOLD)

    # 2. Auxiliary Parameters (Gamma/Delta) - RMSE only (constant targets)
    aux_pass = True
    if metrics["n_params"] > 2:
        if metrics["rmse_gamma"] > AUX_RMSE_THRESHOLD:
            aux_pass = False

    if metrics["n_params"] > 3:
        if metrics["rmse_delta"] > AUX_RMSE_THRESHOLD:
            aux_pass = False

    passed = alpha_pass and beta_pass and aux_pass

    metrics["passed"] = passed
    metrics["family"] = family_name

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"    RMSE(α)={metrics['rmse_alpha']:.4f}, RMSE(β)={metrics['rmse_beta']:.4f}")
        print(f"    Corr(α)={metrics['corr_alpha']:.4f}, Corr(β)={metrics['corr_beta']:.4f}")
        # Show auxiliary parameters if present
        if metrics["n_params"] > 2:
            gamma_type = "constant" if metrics["is_constant_gamma"] else "varies"
            print(f"    RMSE(γ)={metrics['rmse_gamma']:.4f} ({gamma_type}, true={metrics['mean_gamma_true']:.2f})")
        if metrics["n_params"] > 3:
            delta_type = "constant" if metrics["is_constant_delta"] else "varies"
            print(f"    RMSE(δ)={metrics['rmse_delta']:.4f} ({delta_type}, true={metrics['mean_delta_true']:.2f})")
        print(f"    Status: {status}")

    return metrics


# =============================================================================
# Multi-Seed Validation
# =============================================================================

# 10 seeds for robust validation
DEFAULT_SEEDS = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]

# Theory-aligned thresholds (Assumption 5: o(n^{-1/4}) ≈ 0.15 for n=2000)
RMSE_THRESHOLD = 0.15  # was 0.3
CORR_THRESHOLD = 0.8   # was 0.7
AUX_RMSE_THRESHOLD = 0.5  # for auxiliary params (gamma, delta)

# Binary families require more data (~1 bit/observation vs continuous)
BINARY_FAMILIES = {"logit", "probit"}
N_BINARY = 8000  # sample size for binary families (2x default due to weak signal)
N_DEFAULT = 5000  # sample size for continuous families


def get_sample_size(family_name: str) -> int:
    """Get appropriate sample size for family. Binary families need more data."""
    base_family = family_name.split("_")[0]  # handle "logit_highdim" etc.
    return N_BINARY if base_family in BINARY_FAMILIES else N_DEFAULT


def test_family_recovery_multiseed(family_name: str, n=5000, epochs=200,
                                   seeds=None, verbose=True):
    """
    Run parameter recovery across multiple seeds, report ALL metrics with mean ± std.

    Args:
        family_name: Name of the family to test
        n: Sample size
        epochs: Training epochs
        seeds: List of random seeds (default: [42, 123, 456, 789, 999])
        verbose: Print detailed output

    Returns:
        dict with per-seed results and aggregated statistics
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    all_metrics = []
    for seed in seeds:
        metrics = test_family_recovery(family_name, n=n, epochs=epochs, seed=seed, verbose=False)
        metrics["seed"] = seed
        all_metrics.append(metrics)

    # Aggregate all numeric metrics
    aggregated = {"family": family_name, "seeds": seeds, "per_seed": all_metrics}

    # Find numeric keys (excluding non-numeric and NaN)
    numeric_keys = []
    for key, val in all_metrics[0].items():
        if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
            numeric_keys.append(key)

    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    # Worst-case metrics (critical for exposing instability)
    aggregated["rmse_alpha_max"] = max(m["rmse_alpha"] for m in all_metrics)
    aggregated["rmse_beta_max"] = max(m["rmse_beta"] for m in all_metrics)
    aggregated["corr_alpha_min"] = min(m["corr_alpha"] for m in all_metrics if not np.isnan(m["corr_alpha"]))
    aggregated["corr_beta_min"] = min(m["corr_beta"] for m in all_metrics if not np.isnan(m["corr_beta"]))

    # Count per-seed passes
    aggregated["n_pass"] = sum(1 for m in all_metrics if m.get("passed", False))
    aggregated["n_seeds"] = len(seeds)

    # Pass logic: PASS (all), UNSTABLE (60%+), FAIL (<60%)
    pass_rate = aggregated["n_pass"] / aggregated["n_seeds"]
    if pass_rate == 1.0:
        aggregated["status"] = "PASS"
        aggregated["passed"] = True
    elif pass_rate >= 0.6:
        aggregated["status"] = "UNSTABLE"
        aggregated["passed"] = False  # UNSTABLE is not a pass
    else:
        aggregated["status"] = "FAIL"
        aggregated["passed"] = False

    if verbose:
        _print_multiseed_results(aggregated)

    return aggregated


def _print_multiseed_results(agg):
    """Print multi-seed results in tabular format."""
    family = agg["family"]
    seeds = agg["seeds"]
    per_seed = agg["per_seed"]

    print(f"\n  {family.upper()} ({len(seeds)} seeds)")
    print("    Per-seed results:")

    # Header
    print(f"    {'Seed':<6} {'RMSE(α)':<9} {'RMSE(β)':<9} {'Corr(α)':<9} {'Corr(β)':<9} {'Status':<6}")

    for m in per_seed:
        status = "PASS" if m.get("passed", False) else "FAIL"
        print(f"    {m['seed']:<6} {m['rmse_alpha']:<9.4f} {m['rmse_beta']:<9.4f} "
              f"{m['corr_alpha']:<9.4f} {m['corr_beta']:<9.4f} {status:<6}")

    # Aggregate + Worst-case
    print("\n    Aggregate (mean ± std):")
    print(f"      RMSE(α): {agg['rmse_alpha_mean']:.4f} ± {agg['rmse_alpha_std']:.4f}")
    print(f"      RMSE(β): {agg['rmse_beta_mean']:.4f} ± {agg['rmse_beta_std']:.4f}")
    print(f"      Corr(α): {agg['corr_alpha_mean']:.4f} ± {agg['corr_alpha_std']:.4f}")
    print(f"      Corr(β): {agg['corr_beta_mean']:.4f} ± {agg['corr_beta_std']:.4f}")

    print("\n    Worst-case:")
    print(f"      Max RMSE(α): {agg['rmse_alpha_max']:.4f}")
    print(f"      Max RMSE(β): {agg['rmse_beta_max']:.4f}")
    print(f"      Min Corr(α): {agg['corr_alpha_min']:.4f}")
    print(f"      Min Corr(β): {agg['corr_beta_min']:.4f}")

    status = agg.get("status", "FAIL")
    print(f"\n    Status: {status} ({agg['n_pass']}/{agg['n_seeds']} seeds pass)")


# =============================================================================
# All Families Evaluation
# =============================================================================

def run_eval_01_all_families(n=None, epochs=200, seeds=None, families=None, single_seed=False):
    """
    Run parameter recovery evaluation across all families.

    Args:
        n: Sample size (None = auto-scale: 4000 for binary, 2000 for others)
        epochs: Training epochs
        seeds: List of random seeds (default: 10 seeds)
        families: List of families to test (default: all)
        single_seed: If True, run single seed only (legacy mode)
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print("=" * 70)
    print("EVAL 01: PARAMETER RECOVERY (ALL FAMILIES)")
    print("=" * 70)

    if families is None:
        families = list(FAMILY_DGPS.keys())

    if n is None:
        n_str = f"auto (binary={N_BINARY}, other={N_DEFAULT})"
    else:
        n_str = str(n)

    if single_seed:
        print(f"\nConfig: n={n_str}, epochs={epochs}, seed={seeds[0]} (SINGLE SEED MODE)")
    else:
        print(f"\nConfig: n={n_str}, epochs={epochs}, seeds={len(seeds)} seeds")
    print(f"Families: {', '.join(families)}")

    print("\n" + "-" * 70)
    print("RUNNING TESTS")
    print("-" * 70)

    results = {}
    for family_name in families:
        # Auto-scale n for binary families if not specified
        n_family = n if n is not None else get_sample_size(family_name)

        try:
            if single_seed:
                results[family_name] = test_family_recovery(
                    family_name, n=n_family, epochs=epochs, seed=seeds[0], verbose=True
                )
            else:
                results[family_name] = test_family_recovery_multiseed(
                    family_name, n=n_family, epochs=epochs, seeds=seeds, verbose=True
                )
        except Exception as e:
            import traceback
            print(f"\n  {family_name.upper()}: ERROR - {e}")
            traceback.print_exc()
            results[family_name] = {"passed": False, "error": str(e), "family": family_name}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if single_seed:
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
    else:
        # Multi-seed summary with worst-case
        print(f"\n{'Family':<12} {'RMSE(β) mean±std':<16} {'RMSE(β) max':<12} {'Corr(β) min':<12} {'Seeds':<10} {'Status':<10}")
        print("-" * 80)

        n_pass = 0
        n_unstable = 0
        for name, r in results.items():
            if "error" in r:
                print(f"{name:<12} {'ERR':<16} {'ERR':<12} {'ERR':<12} {'ERR':<10} {'ERROR':<10}")
            else:
                status = r.get("status", "FAIL")
                if status == "PASS":
                    n_pass += 1
                elif status == "UNSTABLE":
                    n_unstable += 1
                rmse_b = f"{r['rmse_beta_mean']:.3f}±{r['rmse_beta_std']:.3f}"
                rmse_b_max = f"{r['rmse_beta_max']:.3f}"
                corr_b_min = f"{r['corr_beta_min']:.3f}"
                seeds_pass = f"{r['n_pass']}/{r['n_seeds']}"
                print(f"{name:<12} {rmse_b:<16} {rmse_b_max:<12} {corr_b_min:<12} {seeds_pass:<10} {status:<10}")

    print("-" * (70 if single_seed else 80))
    if single_seed:
        print(f"Overall: {n_pass}/{len(families)} PASS")
    else:
        print(f"Overall: {n_pass} PASS, {n_unstable} UNSTABLE, {len(families) - n_pass - n_unstable} FAIL")
    print("=" * 70)

    return results


# =============================================================================
# Legacy single-family run (for backwards compatibility)
# =============================================================================

def run_eval_01(n=5000, epochs=200, seed=42, verbose=True):
    """Run parameter recovery for logit family only (legacy single-seed mode)."""
    return test_family_recovery("logit", n=n, epochs=epochs, seed=seed, verbose=verbose)


def run_eval_01_multiseed(n=5000, epochs=200, seeds=None, verbose=True):
    """Run parameter recovery for logit family with multi-seed validation."""
    return test_family_recovery_multiseed("logit", n=n, epochs=epochs, seeds=seeds, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval 01: Parameter Recovery")
    parser.add_argument("--n", type=int, default=None,
                        help="Sample size (default: auto-scale, 4000 for binary, 2000 for others)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seeds for multi-seed validation")
    parser.add_argument("--single-seed", action="store_true",
                        help="Run single seed only (legacy mode, uses first seed)")
    parser.add_argument("--family", type=str, default=None, help="Single family to test")
    parser.add_argument("--challenging", action="store_true",
                        help="Include challenging DGPs (high-dim, interactions)")
    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.family:
        # Test single family
        n_family = args.n if args.n is not None else get_sample_size(args.family)
        if args.single_seed:
            result = test_family_recovery(args.family, n=n_family, epochs=args.epochs, seed=seeds[0])
        else:
            result = test_family_recovery_multiseed(args.family, n=n_family, epochs=args.epochs, seeds=seeds)
    else:
        # Build family list
        families = list(FAMILY_DGPS.keys())
        if args.challenging:
            families += list(CHALLENGING_DGPS.keys())

        # Test all families
        results = run_eval_01_all_families(
            n=args.n, epochs=args.epochs, seeds=seeds, families=families, single_seed=args.single_seed
        )
