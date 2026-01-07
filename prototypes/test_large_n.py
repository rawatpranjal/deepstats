"""Test large N for parameter recovery."""
import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from src2.core.algorithm import structural_dml_core
from src2.families import get_family


def generate_poisson_dgp(n: int, seed: int = None):
    """Generate Poisson DGP with tracked true parameters."""
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n, 10)
    T = np.random.uniform(-1, 1, n)
    
    # True heterogeneous parameters
    alpha_star = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    beta_star = 0.3 * X[:, 2] - 0.2 * X[:, 3]
    
    # Scale parameters
    scale = 0.3
    alpha_s = alpha_star * scale
    beta_s = beta_star * scale
    
    mu_true = beta_star.mean() * scale
    
    # Generate Y
    eta = np.clip(alpha_s + beta_s * T, -5, 5)
    lam = np.exp(eta)
    Y = np.random.poisson(lam).astype(float)
    
    return X, T, Y, mu_true, alpha_s, beta_s


def run_test(N: int, M: int, K: int = 50):
    """Run M simulations at sample size N."""
    fam = get_family('poisson')
    
    results = []
    for sim in range(M):
        X, T, Y, mu_true, alpha_true, beta_true = generate_poisson_dgp(N, seed=sim)
        
        result = structural_dml_core(
            Y=Y, T=T, X=X,
            loss_fn=fam.loss,
            target_fn=fam.default_target,
            theta_dim=fam.theta_dim,
            n_folds=K,
            hidden_dims=[64, 32],
            epochs=50,
            lr=0.01,
            three_way=True,
            gradient_fn=fam.gradient,
            hessian_fn=fam.hessian,
            lambda_method='aggregate',
            verbose=False,
        )
        
        # Get estimated parameters
        alpha_hat = result.theta_hat[:, 0]
        beta_hat = result.theta_hat[:, 1]
        
        # Compute correlations
        corr_alpha = np.corrcoef(alpha_true, alpha_hat)[0, 1]
        corr_beta = np.corrcoef(beta_true, beta_hat)[0, 1]
        
        covered = result.ci_lower <= mu_true <= result.ci_upper
        
        results.append({
            'covered': covered,
            'corr_alpha': corr_alpha,
            'corr_beta': corr_beta,
            'se': result.se,
            'mu_hat': result.mu_hat,
            'pct_regularized': result.diagnostics.get('pct_regularized', 0),
            'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
        })
        
        if (sim + 1) % 10 == 0:
            print(f"  Sim {sim + 1}/{M}...")
    
    # Aggregate
    coverages = np.array([r['covered'] for r in results])
    corr_alphas = np.array([r['corr_alpha'] for r in results])
    corr_betas = np.array([r['corr_beta'] for r in results])
    mu_hats = np.array([r['mu_hat'] for r in results])
    ses = np.array([r['se'] for r in results])
    pct_regs = np.array([r['pct_regularized'] for r in results])
    min_eigs = np.array([r['min_eigenvalue'] for r in results])
    
    emp_sd = mu_hats.std()
    se_ratio = ses.mean() / emp_sd if emp_sd > 0 else float('inf')
    
    return {
        'N': N,
        'M': M,
        'coverage': coverages.mean() * 100,
        'se_ratio': se_ratio,
        'corr_alpha': corr_alphas.mean(),
        'corr_beta': corr_betas.mean(),
        'pct_regularized': pct_regs.mean(),
        'min_eigenvalue': min_eigs.mean(),
        'emp_sd': emp_sd,
        'mean_se': ses.mean(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DATA-RICH VALIDATION: N=20000, M=50")
    print("=" * 70)
    
    N = 20000
    M = 50
    K = 50
    
    print(f"\nConfiguration: N={N}, M={M}, K={K}")
    print("-" * 70)
    
    result = run_test(N, M, K)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nInference Metrics:")
    print(f"  Coverage: {result['coverage']:.1f}%")
    print(f"  SE Ratio: {result['se_ratio']:.2f}")
    print(f"  Emp SD: {result['emp_sd']:.5f}")
    print(f"  Mean SE: {result['mean_se']:.5f}")
    print(f"  Reg Rate: {result['pct_regularized']:.1f}%")
    print(f"  Min Eigenvalue: {result['min_eigenvalue']:.4f}")
    
    print(f"\nParameter Recovery:")
    print(f"  Corr(α): {result['corr_alpha']:.3f}")
    print(f"  Corr(β): {result['corr_beta']:.3f}")
    
    # Check targets
    print(f"\n{'='*70}")
    print("VALIDATION CHECK")
    print("=" * 70)
    
    cov_pass = 93 <= result['coverage'] <= 97
    se_pass = 0.9 <= result['se_ratio'] <= 1.2
    alpha_pass = result['corr_alpha'] > 0.7
    beta_pass = result['corr_beta'] > 0.5
    
    print(f"  Coverage 93-97%: {result['coverage']:.1f}% {'PASS' if cov_pass else 'FAIL'}")
    print(f"  SE Ratio 0.9-1.2: {result['se_ratio']:.2f} {'PASS' if se_pass else 'FAIL'}")
    print(f"  Corr(α) > 0.7: {result['corr_alpha']:.3f} {'PASS' if alpha_pass else 'FAIL'}")
    print(f"  Corr(β) > 0.5: {result['corr_beta']:.3f} {'PASS' if beta_pass else 'FAIL'}")
    
    overall = cov_pass and se_pass and alpha_pass and beta_pass
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
