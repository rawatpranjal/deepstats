"""
Lambda Estimation Investigation for Logit with Binary T

Investigates whether the 100% regularization rate is:
1. A fundamental mathematical issue (rank-1 Hessians)
2. An estimation bug (MLP overfitting)

Tests:
1. Diagnostic: What does current MLP actually predict?
2. Control: Continuous T (should have full-rank Hessians)
3. Fix A: AggregateLambdaEstimator
4. Fix B: Ridge regression
5. Fix C: PropensityWeightedLambdaEstimator
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import src2 components
from src2.core.lambda_estimator import LambdaEstimator, AggregateLambdaEstimator
from src2.core.autodiff import compute_hessian
from src2.families import LogitFamily
from src2.models import StructuralNet, train_structural_net
from src2.utils import batch_inverse
from src2 import structural_dml


# =============================================================================
# Data Generation
# =============================================================================

def generate_logit_dgp(n: int, binary_t: bool = True, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate Logit DGP data.

    Returns: X, T, Y, mu_true
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, 10)

    # Treatment
    if binary_t:
        T = np.random.binomial(1, 0.5, n).astype(float)
    else:
        T = np.random.uniform(-1, 1, n)

    # Heterogeneous parameters (simple for testing)
    alpha_star = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    beta_star = 0.3 * X[:, 2] - 0.2 * X[:, 3]

    # True mu
    mu_true = beta_star.mean()

    # Generate Y
    logits = alpha_star + beta_star * T
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p).astype(float)

    return X, T, Y, mu_true


# =============================================================================
# Propensity-Weighted Lambda Estimator
# =============================================================================

class PropensityWeightedLambdaEstimator:
    """
    Computes Λ(x) = E[ℓ_θθ | X=x] using propensity weighting.

    For binary T:
    Λ(x) = (1 - e(x)) · E[ℓ_θθ | T=0] + e(x) · E[ℓ_θθ | T=1]

    where e(x) = P(T=1 | X=x).
    """

    def __init__(self, theta_dim: int = 2):
        self.theta_dim = theta_dim
        self.propensity_model = None
        self.Lambda0_mean = None
        self.Lambda1_mean = None

    def fit(self, X: Tensor, T: Tensor, hessians: Tensor) -> 'PropensityWeightedLambdaEstimator':
        """
        Fit propensity model and compute arm-specific Lambda means.
        """
        X_np = X.numpy() if isinstance(X, Tensor) else X
        T_np = T.numpy() if isinstance(T, Tensor) else T

        # Fit propensity model
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X_np, T_np)

        # Separate by treatment arm
        T0_mask = T_np == 0
        T1_mask = T_np == 1

        # Compute arm-specific means
        if isinstance(hessians, Tensor):
            self.Lambda0_mean = hessians[T0_mask].mean(dim=0)
            self.Lambda1_mean = hessians[T1_mask].mean(dim=0)
        else:
            self.Lambda0_mean = torch.tensor(hessians[T0_mask].mean(axis=0))
            self.Lambda1_mean = torch.tensor(hessians[T1_mask].mean(axis=0))

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict Λ(x) using propensity-weighted combination.
        """
        X_np = X.numpy() if isinstance(X, Tensor) else X

        # Get propensity scores
        e_x = self.propensity_model.predict_proba(X_np)[:, 1]

        # Weighted combination
        n = len(X_np)
        Lambda = torch.zeros(n, self.theta_dim, self.theta_dim)

        for i in range(n):
            Lambda[i] = (1 - e_x[i]) * self.Lambda0_mean + e_x[i] * self.Lambda1_mean

        return Lambda


# =============================================================================
# Modified DML with Lambda Method Selection
# =============================================================================

def run_logit_simulation(
    N: int = 2000,
    n_folds: int = 20,
    lambda_method: str = 'mlp',
    binary_t: bool = True,
    seed: int = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run single Logit simulation with specified Lambda estimation method.

    lambda_method: 'mlp', 'ridge', 'aggregate', 'propensity'
    """
    # Generate data
    X, T, Y, mu_true = generate_logit_dgp(N, binary_t=binary_t, seed=seed)

    # Run structural DML with appropriate Lambda method
    # We need to modify the algorithm to use our Lambda method
    # For now, let's directly test the Lambda estimation

    # Setup
    X_t = torch.tensor(X, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    logit_family = LogitFamily()

    # Split data for one "fold" test
    n = len(Y)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    X_train, T_train, Y_train = X_t[train_idx], T_t[train_idx], Y_t[train_idx]
    X_eval, T_eval, Y_eval = X_t[eval_idx], T_t[eval_idx], Y_t[eval_idx]

    # Further split train into theta and lambda (60/40)
    n_theta = int(0.6 * len(train_idx))
    theta_idx = train_idx[:n_theta]
    lambda_idx = train_idx[n_theta:]

    X_theta = X_t[theta_idx]
    T_theta = T_t[theta_idx]
    Y_theta = Y_t[theta_idx]

    X_lambda = X_t[lambda_idx]
    T_lambda = T_t[lambda_idx]
    Y_lambda = Y_t[lambda_idx]

    # Train theta model
    model = StructuralNet(input_dim=10, theta_dim=2, hidden_dims=[64, 32])
    train_structural_net(
        model=model,
        X=X_theta,
        T=T_theta,
        Y=Y_theta,
        loss_fn=logit_family.loss,
        epochs=50,
        lr=0.01,
        verbose=False
    )

    # Get theta predictions
    model.eval()
    with torch.no_grad():
        theta_lambda = model(X_lambda)
        theta_eval = model(X_eval)

    # Compute Hessians on lambda data
    hessians_lambda = logit_family.hessian(Y_lambda, T_lambda, theta_lambda)

    # Fit Lambda estimator based on method
    if lambda_method == 'mlp':
        lambda_est = LambdaEstimator(method='mlp', theta_dim=2)
        lambda_est.fit(X_lambda, hessians_lambda)
    elif lambda_method == 'ridge':
        lambda_est = LambdaEstimator(method='ridge', theta_dim=2)
        lambda_est.fit(X_lambda, hessians_lambda)
    elif lambda_method == 'aggregate':
        lambda_est = AggregateLambdaEstimator(theta_dim=2)
        lambda_est.fit(X_lambda, hessians_lambda)
    elif lambda_method == 'propensity':
        lambda_est = PropensityWeightedLambdaEstimator(theta_dim=2)
        lambda_est.fit(X_lambda, T_lambda, hessians_lambda)
    else:
        raise ValueError(f"Unknown lambda_method: {lambda_method}")

    # Predict Lambda on eval data
    Lambda_eval = lambda_est.predict(X_eval)

    # Analyze eigenvalues
    n_eval = len(eval_idx)
    min_eigenvalues = []
    n_regularized = 0

    for i in range(n_eval):
        try:
            eigvals = torch.linalg.eigvalsh(Lambda_eval[i])
            min_eig = eigvals.min().item()
            min_eigenvalues.append(min_eig)
            if min_eig < 1e-6:
                n_regularized += 1
        except:
            min_eigenvalues.append(0)
            n_regularized += 1

    regularization_rate = n_regularized / n_eval * 100

    # Compute influence function (simplified)
    Lambda_inv_eval = batch_inverse(Lambda_eval, ridge=1e-4)

    # Get gradients
    l_theta_eval = logit_family.gradient(Y_eval, T_eval, theta_eval)

    # Compute psi values
    psi_values = []
    for i in range(n_eval):
        beta_i = theta_eval[i, 1].item()
        h_grad = torch.tensor([0.0, 1.0])  # Gradient of beta w.r.t. theta
        correction = (h_grad @ Lambda_inv_eval[i] @ l_theta_eval[i]).item()
        psi_i = beta_i - correction
        psi_values.append(psi_i)

    psi_values = np.array(psi_values)
    mu_hat = psi_values.mean()
    se = psi_values.std() / np.sqrt(n_eval)

    # Coverage
    ci_lower = mu_hat - 1.96 * se
    ci_upper = mu_hat + 1.96 * se
    covered = ci_lower <= mu_true <= ci_upper

    results = {
        'mu_true': mu_true,
        'mu_hat': mu_hat,
        'se': se,
        'covered': covered,
        'regularization_rate': regularization_rate,
        'min_eigenvalue': min(min_eigenvalues) if min_eigenvalues else 0,
        'mean_min_eigenvalue': np.mean(min_eigenvalues) if min_eigenvalues else 0,
        'n_regularized': n_regularized,
        'n_eval': n_eval,
    }

    if verbose:
        print(f"Lambda method: {lambda_method}, Binary T: {binary_t}")
        print(f"  Regularization rate: {regularization_rate:.1f}%")
        print(f"  Min eigenvalue: {results['min_eigenvalue']:.6f}")
        print(f"  μ_hat: {mu_hat:.4f}, SE: {se:.4f}")
        print(f"  Covered: {covered}")

    return results


def run_mc_test(
    M: int = 30,
    N: int = 2000,
    lambda_method: str = 'mlp',
    binary_t: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Monte Carlo test with specified Lambda method.
    """
    results = []

    for sim in range(M):
        if verbose and (sim + 1) % 10 == 0:
            print(f"  Simulation {sim + 1}/{M}...")

        try:
            res = run_logit_simulation(
                N=N,
                lambda_method=lambda_method,
                binary_t=binary_t,
                seed=sim,
                verbose=False
            )
            results.append(res)
        except Exception as e:
            print(f"  Simulation {sim} failed: {e}")
            continue

    if not results:
        return {'error': 'All simulations failed'}

    # Aggregate results
    mu_hats = np.array([r['mu_hat'] for r in results])
    ses = np.array([r['se'] for r in results])
    coverages = np.array([r['covered'] for r in results])
    reg_rates = np.array([r['regularization_rate'] for r in results])
    min_eigs = np.array([r['min_eigenvalue'] for r in results])

    # SE ratio
    empirical_sd = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else float('inf')

    return {
        'lambda_method': lambda_method,
        'binary_t': binary_t,
        'M': M,
        'coverage': coverages.mean() * 100,
        'se_ratio': se_ratio,
        'mean_regularization_rate': reg_rates.mean(),
        'mean_min_eigenvalue': min_eigs.mean(),
        'mean_mu_hat': mu_hats.mean(),
        'empirical_sd': empirical_sd,
        'mean_se': mean_se,
    }


# =============================================================================
# Diagnostic: Check what MLP predicts
# =============================================================================

def run_diagnostic():
    """
    Detailed diagnostic of Lambda estimation.
    """
    print("=" * 70)
    print("DIAGNOSTIC: Lambda Estimation Analysis")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    X, T, Y, mu_true = generate_logit_dgp(2000, binary_t=True)

    X_t = torch.tensor(X, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    logit_family = LogitFamily()

    # Train a simple theta model
    model = StructuralNet(input_dim=10, theta_dim=2, hidden_dims=[64, 32])
    train_structural_net(model, X_t, T_t, Y_t, logit_family.loss, epochs=50, verbose=False)

    model.eval()
    with torch.no_grad():
        theta = model(X_t)

    # Compute Hessians
    hessians = logit_family.hessian(Y_t, T_t, theta)

    # Analyze individual Hessians by T value
    print("\n1. INDIVIDUAL HESSIAN ANALYSIS")
    print("-" * 50)

    T_np = T.astype(int)

    # T=0 observations
    t0_idx = np.where(T_np == 0)[0][:5]
    print("\nT=0 observations (first 5):")
    for i in t0_idx:
        H = hessians[i]
        eigvals = torch.linalg.eigvalsh(H)
        print(f"  Obs {i}: H = {H.numpy().round(4)}")
        print(f"          eigenvalues = {eigvals.numpy().round(6)}")

    # T=1 observations
    t1_idx = np.where(T_np == 1)[0][:5]
    print("\nT=1 observations (first 5):")
    for i in t1_idx:
        H = hessians[i]
        eigvals = torch.linalg.eigvalsh(H)
        print(f"  Obs {i}: H = {H.numpy().round(4)}")
        print(f"          eigenvalues = {eigvals.numpy().round(6)}")

    # Average Hessians by T
    print("\n2. AGGREGATE HESSIANS BY TREATMENT ARM")
    print("-" * 50)

    H_t0_mean = hessians[T_np == 0].mean(dim=0)
    H_t1_mean = hessians[T_np == 1].mean(dim=0)
    H_all_mean = hessians.mean(dim=0)

    print(f"\nMean Hessian for T=0:")
    print(f"  {H_t0_mean.numpy().round(4)}")
    print(f"  eigenvalues: {torch.linalg.eigvalsh(H_t0_mean).numpy().round(6)}")

    print(f"\nMean Hessian for T=1:")
    print(f"  {H_t1_mean.numpy().round(4)}")
    print(f"  eigenvalues: {torch.linalg.eigvalsh(H_t1_mean).numpy().round(6)}")

    print(f"\nMean Hessian (all observations):")
    print(f"  {H_all_mean.numpy().round(4)}")
    print(f"  eigenvalues: {torch.linalg.eigvalsh(H_all_mean).numpy().round(6)}")

    # Propensity-weighted average (assuming e=0.5)
    H_propensity = 0.5 * H_t0_mean + 0.5 * H_t1_mean
    print(f"\nPropensity-weighted average (e=0.5):")
    print(f"  {H_propensity.numpy().round(4)}")
    print(f"  eigenvalues: {torch.linalg.eigvalsh(H_propensity).numpy().round(6)}")

    # Test different Lambda methods
    print("\n3. LAMBDA ESTIMATOR COMPARISON")
    print("-" * 50)

    # Split data
    n = len(X)
    perm = np.random.permutation(n)
    train_idx = perm[:1600]
    test_idx = perm[1600:]

    X_train = X_t[train_idx]
    T_train = T_t[train_idx]
    hessians_train = hessians[train_idx]

    X_test = X_t[test_idx]

    methods = {
        'MLP': LambdaEstimator(method='mlp', theta_dim=2),
        'Ridge': LambdaEstimator(method='ridge', theta_dim=2),
        'Aggregate': AggregateLambdaEstimator(theta_dim=2),
        'Propensity': PropensityWeightedLambdaEstimator(theta_dim=2),
    }

    for name, estimator in methods.items():
        print(f"\n{name} Lambda Estimator:")

        if name == 'Propensity':
            estimator.fit(X_train, T_train, hessians_train)
        else:
            estimator.fit(X_train, hessians_train)

        Lambda_pred = estimator.predict(X_test)

        # Analyze predictions
        min_eigs = []
        for i in range(len(test_idx)):
            eigvals = torch.linalg.eigvalsh(Lambda_pred[i])
            min_eigs.append(eigvals.min().item())

        n_singular = sum(1 for e in min_eigs if e < 1e-6)

        print(f"  Predictions on test set ({len(test_idx)} obs):")
        print(f"    Min eigenvalue (min): {min(min_eigs):.6f}")
        print(f"    Min eigenvalue (mean): {np.mean(min_eigs):.6f}")
        print(f"    Singular (<1e-6): {n_singular} ({100*n_singular/len(test_idx):.1f}%)")

        # Show first prediction
        print(f"  First prediction:")
        print(f"    {Lambda_pred[0].numpy().round(4)}")
        print(f"    eigenvalues: {torch.linalg.eigvalsh(Lambda_pred[0]).numpy().round(6)}")


# =============================================================================
# Main Test Suite
# =============================================================================

def main():
    """Run full investigation."""

    print("=" * 70)
    print("LAMBDA ESTIMATION INVESTIGATION FOR LOGIT")
    print("=" * 70)

    # Run diagnostic first
    run_diagnostic()

    # Run MC tests
    print("\n" + "=" * 70)
    print("MONTE CARLO TESTS")
    print("=" * 70)

    M = 30  # Simulations per test
    N = 2000  # Sample size

    test_configs = [
        # (lambda_method, binary_t)
        ('mlp', True),
        ('mlp', False),
        ('ridge', True),
        ('ridge', False),
        ('aggregate', True),
        ('aggregate', False),
        ('propensity', True),
        ('propensity', False),
    ]

    results_table = []

    for lambda_method, binary_t in test_configs:
        t_type = "Binary" if binary_t else "Continuous"
        print(f"\nTesting: {lambda_method.upper()} with {t_type} T...")

        result = run_mc_test(
            M=M,
            N=N,
            lambda_method=lambda_method,
            binary_t=binary_t,
            verbose=True
        )

        results_table.append(result)

        print(f"  Coverage: {result['coverage']:.1f}%")
        print(f"  SE Ratio: {result['se_ratio']:.2f}")
        print(f"  Regularization Rate: {result['mean_regularization_rate']:.1f}%")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<15} {'T Type':<12} {'Coverage':<10} {'SE Ratio':<10} {'Reg Rate':<10} {'Min Eig':<12}")
    print("-" * 70)

    for r in results_table:
        t_type = "Binary" if r['binary_t'] else "Continuous"
        print(f"{r['lambda_method']:<15} {t_type:<12} {r['coverage']:.1f}%      {r['se_ratio']:.2f}       {r['mean_regularization_rate']:.1f}%       {r['mean_min_eigenvalue']:.6f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare binary vs continuous T for MLP
    mlp_binary = [r for r in results_table if r['lambda_method'] == 'mlp' and r['binary_t']][0]
    mlp_continuous = [r for r in results_table if r['lambda_method'] == 'mlp' and not r['binary_t']][0]

    print("\n1. BINARY vs CONTINUOUS T (MLP method):")
    print(f"   Binary T:     Reg rate = {mlp_binary['mean_regularization_rate']:.1f}%")
    print(f"   Continuous T: Reg rate = {mlp_continuous['mean_regularization_rate']:.1f}%")

    if mlp_binary['mean_regularization_rate'] > 50 and mlp_continuous['mean_regularization_rate'] < 20:
        print("   → Binary T causes singular Hessians, continuous T does not")

    # Best method for binary T
    binary_results = [r for r in results_table if r['binary_t']]
    best_coverage = max(binary_results, key=lambda x: x['coverage'] if 93 <= x['coverage'] <= 97 else -abs(x['coverage'] - 95))
    best_se_ratio = min(binary_results, key=lambda x: abs(x['se_ratio'] - 1.0))
    lowest_reg = min(binary_results, key=lambda x: x['mean_regularization_rate'])

    print("\n2. BEST METHODS FOR BINARY T:")
    print(f"   Best coverage: {best_coverage['lambda_method']} ({best_coverage['coverage']:.1f}%)")
    print(f"   Best SE ratio: {best_se_ratio['lambda_method']} ({best_se_ratio['se_ratio']:.2f})")
    print(f"   Lowest reg rate: {lowest_reg['lambda_method']} ({lowest_reg['mean_regularization_rate']:.1f}%)")

    # Recommendation
    print("\n3. RECOMMENDATION:")

    # Find method with best overall performance for binary T
    for r in binary_results:
        if r['mean_regularization_rate'] < 20 and 90 <= r['coverage'] <= 100 and 0.8 <= r['se_ratio'] <= 1.2:
            print(f"   → Use {r['lambda_method'].upper()} method for Logit with binary T")
            print(f"     Coverage: {r['coverage']:.1f}%, SE Ratio: {r['se_ratio']:.2f}, Reg Rate: {r['mean_regularization_rate']:.1f}%")
            break
    else:
        print("   → No clear winner. Aggregate or Propensity may need further tuning.")


if __name__ == "__main__":
    main()
