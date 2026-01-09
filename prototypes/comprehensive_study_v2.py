"""
Comprehensive Validation Study v2.

Goal: Can we recover complex heterogeneous patterns with enough data and expressive networks?

Configuration (FROZEN):
- N = 50,000
- K = 50
- M = 100

Features:
- Complex nonlinear α*, β* with 6 signal + 14 noise features
- Deep narrow networks [128, 64, 32, 16]
- Slow learning rate (0.001) with early stopping
- Train/val gap tracking for overfitting detection
- Full metrics: RMSE, Corr, Coverage, SE Ratio, RMSE(μ̂)

Usage:
    python3 prototypes/comprehensive_study_v2.py --family poisson
    python3 prototypes/comprehensive_study_v2.py --family linear
    python3 prototypes/comprehensive_study_v2.py --family logit
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src2.core.algorithm import structural_dml_core, DMLResult
from src2.families import get_family
from tqdm import tqdm


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

N_FIXED = 50000
K_FIXED = 50
M_FIXED = 100
D_FIXED = 20  # 6 signal + 14 noise


# =============================================================================
# COMPLEX NONLINEAR DGP
# =============================================================================

def compute_alpha_star(X: np.ndarray) -> np.ndarray:
    """
    Complex nonlinear intercept.

    α*(X) = sin(2π·X_0) + X_1³ - 2·cos(π·X_2) + exp(X_3/3)·I(X_3>0) + 0.5·X_4·X_5
    """
    return (
        np.sin(2 * np.pi * X[:, 0]) +
        X[:, 1] ** 3 -
        2 * np.cos(np.pi * X[:, 2]) +
        np.exp(X[:, 3] / 3) * (X[:, 3] > 0) +
        0.5 * X[:, 4] * X[:, 5]
    )


def compute_beta_star(X: np.ndarray) -> np.ndarray:
    """
    Complex nonlinear treatment effect.

    β*(X) = cos(2π·X_0)·sin(π·X_1) + 0.8·tanh(3·X_2) - 0.5·X_3² + 0.3·X_4·I(X_5>0)
    """
    return (
        np.cos(2 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]) +
        0.8 * np.tanh(3 * X[:, 2]) -
        0.5 * X[:, 3] ** 2 +
        0.3 * X[:, 4] * (X[:, 5] > 0)
    )


def compute_mu_star_mc(n_mc: int = 100000) -> float:
    """Compute μ* = E[β(X)] via Monte Carlo."""
    np.random.seed(42)
    X_mc = np.random.uniform(-1, 1, (n_mc, D_FIXED))
    beta_mc = compute_beta_star(X_mc)
    return beta_mc.mean()


# Pre-compute μ* for consistency
MU_STAR = compute_mu_star_mc()


def generate_dgp(
    n: int,
    family: str,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Generate complex nonlinear DGP data.

    Args:
        n: Sample size
        family: 'poisson', 'linear', or 'logit'
        seed: Random seed

    Returns:
        X, T, Y, mu_true, alpha_star, beta_star
    """
    if seed is not None:
        np.random.seed(seed)

    # Covariates: 6 signal + 14 noise (unnormalized)
    X = np.random.uniform(-1, 1, (n, D_FIXED))

    # Treatment (unnormalized)
    T = np.random.uniform(-1, 1, n)

    # True heterogeneous parameters
    alpha_star = compute_alpha_star(X)
    beta_star = compute_beta_star(X)

    # Generate Y based on family
    if family == 'poisson':
        # Scale to keep λ reasonable
        scale = 0.3
        eta = np.clip(scale * (alpha_star + beta_star * T), -5, 5)
        lam = np.exp(eta)
        Y = np.random.poisson(lam).astype(float)
        # Adjust for scale
        alpha_scaled = alpha_star * scale
        beta_scaled = beta_star * scale
        mu_true = MU_STAR * scale

    elif family == 'linear':
        # Linear model with noise
        eps = np.random.randn(n) * 0.5
        Y = alpha_star + beta_star * T + eps
        alpha_scaled = alpha_star
        beta_scaled = beta_star
        mu_true = MU_STAR

    elif family == 'logit':
        # Logit model
        scale = 0.5
        eta = scale * (alpha_star + beta_star * T)
        prob = 1 / (1 + np.exp(-eta))
        Y = (np.random.rand(n) < prob).astype(float)
        alpha_scaled = alpha_star * scale
        beta_scaled = beta_star * scale
        mu_true = MU_STAR * scale

    else:
        raise ValueError(f"Unknown family: {family}")

    return X, T, Y, mu_true, alpha_scaled, beta_scaled


# =============================================================================
# METRICS EXTRACTION
# =============================================================================

def extract_training_metrics(result: DMLResult) -> Dict[str, float]:
    """Extract overfitting metrics from training histories."""
    histories = result.diagnostics.get('histories', [])
    if not histories:
        return {
            'mean_train_loss': 0.0,
            'mean_val_loss': 0.0,
            'train_val_gap': 0.0,
            'mean_best_epoch': 0.0,
        }

    train_losses = []
    val_losses = []
    best_epochs = []

    for h in histories:
        if h.train_losses:
            train_losses.append(h.train_losses[-1])
        if h.val_losses:
            val_losses.append(h.val_losses[-1])
        best_epochs.append(h.best_epoch)

    mean_train = np.mean(train_losses) if train_losses else 0.0
    mean_val = np.mean(val_losses) if val_losses else 0.0

    return {
        'mean_train_loss': mean_train,
        'mean_val_loss': mean_val,
        'train_val_gap': mean_val - mean_train,
        'mean_best_epoch': np.mean(best_epochs),
    }


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_sim(
    family: str,
    seed: int,
    hidden_dims: list = [128, 64, 32, 16],
    lr: float = 0.001,
    patience: int = 20,
    max_epochs: int = 500,
) -> Dict[str, Any]:
    """Run a single simulation with full metrics."""

    fam = get_family(family)

    # Generate data
    X, T, Y, mu_true, alpha_star, beta_star = generate_dgp(
        n=N_FIXED, family=family, seed=seed
    )

    # Determine three-way splitting
    three_way = fam.hessian_depends_on_theta() if hasattr(fam, 'hessian_depends_on_theta') else False

    # Run DML
    result = structural_dml_core(
        Y=Y, T=T, X=X,
        loss_fn=fam.loss,
        target_fn=fam.default_target,
        theta_dim=fam.theta_dim,
        n_folds=K_FIXED,
        hidden_dims=hidden_dims,
        epochs=max_epochs,
        lr=lr,
        three_way=three_way,
        gradient_fn=fam.gradient if hasattr(fam, 'gradient') else None,
        hessian_fn=fam.hessian if hasattr(fam, 'hessian') else None,
        lambda_method='aggregate',
        verbose=False,
    )

    # Extract estimated parameters
    alpha_hat = result.theta_hat[:, 0]
    beta_hat = result.theta_hat[:, 1]

    # Phase 1: Parameter Recovery
    rmse_alpha = np.sqrt(np.mean((alpha_hat - alpha_star) ** 2))
    rmse_beta = np.sqrt(np.mean((beta_hat - beta_star) ** 2))
    corr_alpha = np.corrcoef(alpha_hat, alpha_star)[0, 1]
    corr_beta = np.corrcoef(beta_hat, beta_star)[0, 1]
    r2_alpha = compute_r2(alpha_star, alpha_hat)
    r2_beta = compute_r2(beta_star, beta_hat)

    # Phase 2: Inference
    covered = result.ci_lower <= mu_true <= result.ci_upper

    # Phase 3: Diagnostics
    training_metrics = extract_training_metrics(result)

    return {
        # Metadata
        'sim_id': seed,
        'family': family,
        'N': N_FIXED,
        'K': K_FIXED,

        # Phase 1: Parameter Recovery
        'rmse_alpha': rmse_alpha,
        'rmse_beta': rmse_beta,
        'corr_alpha': corr_alpha,
        'corr_beta': corr_beta,
        'r2_alpha': r2_alpha,
        'r2_beta': r2_beta,

        # Phase 2: Inference
        'mu_true': mu_true,
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'covered': covered,

        # Phase 3: Diagnostics
        'pct_regularized': result.diagnostics.get('pct_regularized', 0),
        'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
        'train_loss': training_metrics['mean_train_loss'],
        'val_loss': training_metrics['mean_val_loss'],
        'train_val_gap': training_metrics['train_val_gap'],
        'best_epoch': training_metrics['mean_best_epoch'],
    }


# =============================================================================
# FULL STUDY
# =============================================================================

def run_study(
    family: str,
    M: int = M_FIXED,
    verbose: bool = True
) -> pd.DataFrame:
    """Run comprehensive study for a single family."""

    results = []
    start_time = datetime.now()

    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE STUDY v2: {family.upper()}")
        print(f"{'='*70}")
        print(f"N={N_FIXED:,}, M={M}, K={K_FIXED}, d={D_FIXED}")
        print(f"μ* = {MU_STAR:.6f} (raw), scaled per family")
        print(f"NN: [128, 64, 32, 16], lr=0.001, patience=20, max_epochs=500")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    pbar = tqdm(range(M), desc=f"{family}", disable=not verbose)
    for sim in pbar:
        try:
            r = run_single_sim(family=family, seed=sim)
            results.append(r)

            # Update progress bar
            cov = np.mean([x['covered'] for x in results]) * 100
            corr_b = np.mean([x['corr_beta'] for x in results])
            pbar.set_postfix({
                'Cov': f'{cov:.0f}%',
                'Corr(β)': f'{corr_b:.2f}',
            })

        except Exception as e:
            print(f"  Sim {sim} FAILED: {e}", flush=True)
            continue

    df = pd.DataFrame(results)

    # Print summary
    if verbose and len(df) > 0:
        print_summary(df)

    return df


def print_summary(df: pd.DataFrame):
    """Print comprehensive summary."""
    print(f"\n{'='*70}")
    print("PHASE 1: PARAMETER RECOVERY")
    print(f"{'='*70}")
    print(f"  RMSE(α):  {df['rmse_alpha'].mean():.4f}")
    print(f"  RMSE(β):  {df['rmse_beta'].mean():.4f}")
    print(f"  Corr(α):  {df['corr_alpha'].mean():.3f}")
    print(f"  Corr(β):  {df['corr_beta'].mean():.3f}")
    print(f"  R²(α):    {df['r2_alpha'].mean():.3f}")
    print(f"  R²(β):    {df['r2_beta'].mean():.3f}")

    print(f"\n{'='*70}")
    print("PHASE 2: INFERENCE")
    print(f"{'='*70}")
    coverage = df['covered'].mean() * 100
    mu_hats = df['mu_hat'].values
    mu_true = df['mu_true'].iloc[0]
    ses = df['se'].values
    emp_sd = mu_hats.std()
    se_ratio = ses.mean() / emp_sd if emp_sd > 0 else float('inf')
    rmse_mu = np.sqrt(np.mean((mu_hats - mu_true) ** 2))
    bias = mu_hats.mean() - mu_true

    print(f"  Coverage:     {coverage:.1f}%")
    print(f"  SE Ratio:     {se_ratio:.2f}")
    print(f"  RMSE(μ̂):      {rmse_mu:.6f}")
    print(f"  Bias:         {bias:.6f}")
    print(f"  Empirical SD: {emp_sd:.6f}")
    print(f"  Mean SE:      {ses.mean():.6f}")

    print(f"\n{'='*70}")
    print("PHASE 3: DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"  Reg Rate:       {df['pct_regularized'].mean():.1f}%")
    print(f"  Min Eigenvalue: {df['min_eigenvalue'].mean():.4f}")
    print(f"  Train Loss:     {df['train_loss'].mean():.4f}")
    print(f"  Val Loss:       {df['val_loss'].mean():.4f}")
    print(f"  Train/Val Gap:  {df['train_val_gap'].mean():.4f}")
    print(f"  Best Epoch:     {df['best_epoch'].mean():.0f}")

    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")

    cov_pass = 93 <= coverage <= 97
    se_pass = 0.9 <= se_ratio <= 1.1
    corr_a_pass = df['corr_alpha'].mean() > 0.80
    corr_b_pass = df['corr_beta'].mean() > 0.60
    gap_pass = abs(df['train_val_gap'].mean()) < 0.1

    print(f"  Coverage 93-97%:   {coverage:.1f}% {'PASS' if cov_pass else 'FAIL'}")
    print(f"  SE Ratio 0.9-1.1:  {se_ratio:.2f} {'PASS' if se_pass else 'FAIL'}")
    print(f"  Corr(α) > 0.80:    {df['corr_alpha'].mean():.3f} {'PASS' if corr_a_pass else 'FAIL'}")
    print(f"  Corr(β) > 0.60:    {df['corr_beta'].mean():.3f} {'PASS' if corr_b_pass else 'FAIL'}")
    print(f"  |Gap| < 0.1:       {abs(df['train_val_gap'].mean()):.3f} {'PASS' if gap_pass else 'FAIL'}")

    overall = cov_pass and se_pass and corr_b_pass
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Validation Study v2')
    parser.add_argument('--family', type=str, required=True,
                        choices=['poisson', 'linear', 'logit'],
                        help='Model family')
    parser.add_argument('--M', type=int, default=M_FIXED,
                        help=f'Number of simulations (default: {M_FIXED})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')

    args = parser.parse_args()

    # Run study
    df = run_study(family=args.family, M=args.M, verbose=True)

    # Save results
    if args.output is None:
        args.output = f'/Users/pranjal/deepest/results/comprehensive_v2_{args.family}_N{N_FIXED}_M{args.M}.csv'

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
